#include "self_compiler/frontend/onnx_importer.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace self_compiler::frontend {

namespace {

// ================================================================
// 第一部分：Protobuf 最小二进制解码器
// ================================================================
//
// Protobuf 是 ONNX 使用的二进制序列化格式。
// 每个字段编码为 (tag, value)，tag = (字段编号 << 3) | 线格式类型。
// 线格式类型决定 value 的编码方式：
//   0 = varint（变长整数）
//   1 = 64 位定长
//   2 = 长度前缀（字符串、子消息、packed 数组）
//   5 = 32 位定长
//
// 这里只实现解码 ONNX 所需的最小子集，不依赖任何外部库。

enum class WireType : int {
    kVarint = 0,
    kFixed64 = 1,
    kLengthDelimited = 2,
    kFixed32 = 5,
};

// 在字节流上逐步前进的读取器
struct ProtoReader {
    const uint8_t* data;
    size_t size;
    size_t pos = 0;

    bool HasMore() const { return pos < size; }

    uint8_t ReadByte() {
        if (pos >= size) {
            throw std::runtime_error("protobuf: 读取越界");
        }
        return data[pos++];
    }

    // 读取 varint 编码的无符号整数
    // 每个字节的低 7 位是数据，最高位表示"后面还有字节"
    uint64_t ReadVarint() {
        uint64_t result = 0;
        int shift = 0;
        while (true) {
            uint8_t byte = ReadByte();
            result |= static_cast<uint64_t>(byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) {
                break;
            }
            shift += 7;
            if (shift >= 64) {
                throw std::runtime_error("protobuf: varint 超过 64 位");
            }
        }
        return result;
    }

    // 读取字段标签，返回 (字段编号, 线格式类型)
    std::pair<int, WireType> ReadTag() {
        uint64_t tag = ReadVarint();
        int field_number = static_cast<int>(tag >> 3);
        auto wire_type = static_cast<WireType>(tag & 0x07);
        return {field_number, wire_type};
    }

    // 读取指定长度的原始字节，返回字符串
    std::string ReadBytes(size_t len) {
        if (pos + len > size) {
            throw std::runtime_error("protobuf: 字节读取越界");
        }
        std::string result(reinterpret_cast<const char*>(data + pos), len);
        pos += len;
        return result;
    }

    // 读取长度前缀的字符串
    std::string ReadString() {
        auto len = static_cast<size_t>(ReadVarint());
        return ReadBytes(len);
    }

    // 读取长度前缀的子消息，返回一个新的读取器
    ProtoReader ReadSubMessage() {
        auto len = static_cast<size_t>(ReadVarint());
        if (pos + len > size) {
            throw std::runtime_error("protobuf: 子消息越界");
        }
        ProtoReader sub{data + pos, len, 0};
        pos += len;
        return sub;
    }

    // 跳过一个不关心的字段
    void SkipField(WireType type) {
        switch (type) {
            case WireType::kVarint:
                ReadVarint();
                break;
            case WireType::kFixed64:
                if (pos + 8 > size) {
                    throw std::runtime_error("protobuf: fixed64 越界");
                }
                pos += 8;
                break;
            case WireType::kLengthDelimited: {
                auto len = static_cast<size_t>(ReadVarint());
                if (pos + len > size) {
                    throw std::runtime_error("protobuf: 长度前缀数据越界");
                }
                pos += len;
                break;
            }
            case WireType::kFixed32:
                if (pos + 4 > size) {
                    throw std::runtime_error("protobuf: fixed32 越界");
                }
                pos += 4;
                break;
            default:
                throw std::runtime_error("protobuf: 未知的线格式类型");
        }
    }
};

// ================================================================
// 第二部分：ONNX 数据结构
// ================================================================
//
// 这些结构体对应 onnx.proto 中的关键消息类型。
// 只保留构建 Graph IR 需要的字段，忽略文档字符串等辅助信息。

// 一个维度：要么是具体数值，要么是符号名（如 "batch_size"）
struct OnnxDimension {
    int64_t value = -1;       // 具体数值，-1 表示动态
    std::string param;        // 符号名
};

// Tensor 的类型信息
struct OnnxTensorType {
    int32_t elem_type = 0;    // ONNX 数据类型枚举值
    std::vector<OnnxDimension> shape;
};

// 图的输入/输出/中间值的描述
struct OnnxValueInfo {
    std::string name;
    OnnxTensorType tensor_type;
};

// 一个计算节点
struct OnnxNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, std::string> attributes;
};

// 整个图
struct OnnxGraph {
    std::string name;
    std::vector<OnnxNode> nodes;
    std::vector<OnnxValueInfo> inputs;       // 图的输入（含权重）
    std::vector<OnnxValueInfo> outputs;      // 图的输出
    std::vector<OnnxValueInfo> value_infos;  // 中间 tensor 的类型信息
    std::vector<OnnxValueInfo> initializer_infos;  // 权重 tensor 信息
};

// ================================================================
// 第三部分：ONNX Protobuf 字段编号与解析
// ================================================================
//
// 以下常量来自 onnx.proto 中各消息类型的字段编号。
// 每个 ParseXxx 函数对应解析一种 protobuf 消息。

namespace field {
    // ModelProto
    constexpr int kModelGraph = 7;

    // GraphProto
    constexpr int kGraphNode = 1;
    constexpr int kGraphName = 2;
    constexpr int kGraphInitializer = 5;
    constexpr int kGraphInput = 11;
    constexpr int kGraphOutput = 12;
    constexpr int kGraphValueInfo = 13;

    // NodeProto
    constexpr int kNodeInput = 1;
    constexpr int kNodeOutput = 2;
    constexpr int kNodeName = 3;
    constexpr int kNodeOpType = 4;
    constexpr int kNodeAttribute = 5;

    // AttributeProto
    constexpr int kAttributeName = 1;
    constexpr int kAttributeInt = 3;
    constexpr int kAttributeInts = 8;

    // ValueInfoProto
    constexpr int kValueInfoName = 1;
    constexpr int kValueInfoType = 2;

    // TypeProto
    constexpr int kTypeTensorType = 1;

    // TypeProto.Tensor
    constexpr int kTensorElemType = 1;
    constexpr int kTensorShape = 2;

    // TensorShapeProto
    constexpr int kShapeDim = 1;

    // TensorShapeProto.Dimension
    constexpr int kDimValue = 1;
    constexpr int kDimParam = 2;

    // TensorProto（只取名字、维度和数据类型，跳过实际权重数据）
    constexpr int kTensorProtoDims = 1;
    constexpr int kTensorProtoDataType = 2;
    constexpr int kTensorProtoName = 8;
}  // namespace field

bool IsTargetNodeAttribute(const std::string& name) {
    return name == "kernel_shape" ||
        name == "strides" ||
        name == "pads" ||
        name == "axis" ||
        name == "perm" ||
        name == "group";
}

std::string JoinIntValues(const std::vector<int64_t>& values) {
    std::string joined;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            joined += ",";
        }
        joined += std::to_string(values[i]);
    }
    return joined;
}

bool TryGetUniformValue(const std::vector<int64_t>& values, int64_t& out) {
    if (values.empty()) {
        return false;
    }

    out = values.front();
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] != out) {
            return false;
        }
    }
    return true;
}

void StoreDerivedAliases(
    const std::string& attr_name,
    const std::vector<int64_t>& values,
    std::map<std::string, std::string>& attrs) {
    int64_t uniform_value = 0;
    if (!TryGetUniformValue(values, uniform_value)) {
        return;
    }

    if (attr_name == "kernel_shape") {
        attrs["kernel_size"] = std::to_string(uniform_value);
    } else if (attr_name == "strides") {
        attrs["stride"] = std::to_string(uniform_value);
    } else if (attr_name == "pads") {
        attrs["padding"] = std::to_string(uniform_value);
    }
}

void ParseAttribute(ProtoReader reader, std::map<std::string, std::string>& attrs) {
    std::string attr_name;
    bool has_scalar_int = false;
    int64_t scalar_int = 0;
    std::vector<int64_t> int_values;

    while (reader.HasMore()) {
        auto [f, w] = reader.ReadTag();
        if (f == field::kAttributeName && w == WireType::kLengthDelimited) {
            attr_name = reader.ReadString();
        } else if (f == field::kAttributeInt && w == WireType::kVarint) {
            has_scalar_int = true;
            scalar_int = static_cast<int64_t>(reader.ReadVarint());
        } else if (f == field::kAttributeInts && w == WireType::kVarint) {
            int_values.push_back(static_cast<int64_t>(reader.ReadVarint()));
        } else if (f == field::kAttributeInts && w == WireType::kLengthDelimited) {
            ProtoReader packed = reader.ReadSubMessage();
            while (packed.HasMore()) {
                int_values.push_back(static_cast<int64_t>(packed.ReadVarint()));
            }
        } else {
            reader.SkipField(w);
        }
    }

    if (!IsTargetNodeAttribute(attr_name)) {
        return;
    }

    if (!int_values.empty()) {
        attrs[attr_name] = JoinIntValues(int_values);
        StoreDerivedAliases(attr_name, int_values, attrs);
        return;
    }

    if (has_scalar_int) {
        attrs[attr_name] = std::to_string(scalar_int);
    }
}

OnnxDimension ParseDimension(ProtoReader reader) {
    OnnxDimension dim;
    while (reader.HasMore()) {
        auto [f, w] = reader.ReadTag();
        if (f == field::kDimValue && w == WireType::kVarint) {
            dim.value = static_cast<int64_t>(reader.ReadVarint());
        } else if (f == field::kDimParam && w == WireType::kLengthDelimited) {
            dim.param = reader.ReadString();
        } else {
            reader.SkipField(w);
        }
    }
    return dim;
}

std::vector<OnnxDimension> ParseTensorShape(ProtoReader reader) {
    std::vector<OnnxDimension> dims;
    while (reader.HasMore()) {
        auto [f, w] = reader.ReadTag();
        if (f == field::kShapeDim && w == WireType::kLengthDelimited) {
            dims.push_back(ParseDimension(reader.ReadSubMessage()));
        } else {
            reader.SkipField(w);
        }
    }
    return dims;
}

OnnxTensorType ParseTensorType(ProtoReader reader) {
    OnnxTensorType tt;
    while (reader.HasMore()) {
        auto [f, w] = reader.ReadTag();
        if (f == field::kTensorElemType && w == WireType::kVarint) {
            tt.elem_type = static_cast<int32_t>(reader.ReadVarint());
        } else if (f == field::kTensorShape && w == WireType::kLengthDelimited) {
            tt.shape = ParseTensorShape(reader.ReadSubMessage());
        } else {
            reader.SkipField(w);
        }
    }
    return tt;
}

OnnxValueInfo ParseValueInfo(ProtoReader reader) {
    OnnxValueInfo info;
    while (reader.HasMore()) {
        auto [f, w] = reader.ReadTag();
        if (f == field::kValueInfoName && w == WireType::kLengthDelimited) {
            info.name = reader.ReadString();
        } else if (f == field::kValueInfoType && w == WireType::kLengthDelimited) {
            // TypeProto 只取 tensor_type 分支
            ProtoReader type_reader = reader.ReadSubMessage();
            while (type_reader.HasMore()) {
                auto [tf, tw] = type_reader.ReadTag();
                if (tf == field::kTypeTensorType && tw == WireType::kLengthDelimited) {
                    info.tensor_type = ParseTensorType(type_reader.ReadSubMessage());
                } else {
                    type_reader.SkipField(tw);
                }
            }
        } else {
            reader.SkipField(w);
        }
    }
    return info;
}

OnnxNode ParseNode(ProtoReader reader) {
    OnnxNode node;
    while (reader.HasMore()) {
        auto [f, w] = reader.ReadTag();
        if (f == field::kNodeInput && w == WireType::kLengthDelimited) {
            node.inputs.push_back(reader.ReadString());
        } else if (f == field::kNodeOutput && w == WireType::kLengthDelimited) {
            node.outputs.push_back(reader.ReadString());
        } else if (f == field::kNodeName && w == WireType::kLengthDelimited) {
            node.name = reader.ReadString();
        } else if (f == field::kNodeOpType && w == WireType::kLengthDelimited) {
            node.op_type = reader.ReadString();
        } else if (f == field::kNodeAttribute && w == WireType::kLengthDelimited) {
            ParseAttribute(reader.ReadSubMessage(), node.attributes);
        } else {
            reader.SkipField(w);
        }
    }
    return node;
}

// 从 TensorProto 中只提取名字、维度和数据类型，跳过实际权重数据
OnnxValueInfo ParseInitializerInfo(ProtoReader reader) {
    OnnxValueInfo info;
    std::vector<int64_t> dims;
    int32_t data_type = 0;

    while (reader.HasMore()) {
        auto [f, w] = reader.ReadTag();
        if (f == field::kTensorProtoName && w == WireType::kLengthDelimited) {
            info.name = reader.ReadString();
        } else if (f == field::kTensorProtoDims && w == WireType::kVarint) {
            // 非 packed 编码：每个维度单独一个 varint
            dims.push_back(static_cast<int64_t>(reader.ReadVarint()));
        } else if (f == field::kTensorProtoDims && w == WireType::kLengthDelimited) {
            // packed 编码：所有维度打包在一个长度前缀块里
            ProtoReader packed = reader.ReadSubMessage();
            while (packed.HasMore()) {
                dims.push_back(static_cast<int64_t>(packed.ReadVarint()));
            }
        } else if (f == field::kTensorProtoDataType && w == WireType::kVarint) {
            data_type = static_cast<int32_t>(reader.ReadVarint());
        } else {
            reader.SkipField(w);
        }
    }

    info.tensor_type.elem_type = data_type;
    for (int64_t d : dims) {
        OnnxDimension dim;
        dim.value = d;
        info.tensor_type.shape.push_back(dim);
    }
    return info;
}

OnnxGraph ParseGraph(ProtoReader reader) {
    OnnxGraph graph;
    while (reader.HasMore()) {
        auto [f, w] = reader.ReadTag();
        if (f == field::kGraphNode && w == WireType::kLengthDelimited) {
            graph.nodes.push_back(ParseNode(reader.ReadSubMessage()));
        } else if (f == field::kGraphName && w == WireType::kLengthDelimited) {
            graph.name = reader.ReadString();
        } else if (f == field::kGraphInitializer && w == WireType::kLengthDelimited) {
            graph.initializer_infos.push_back(ParseInitializerInfo(reader.ReadSubMessage()));
        } else if (f == field::kGraphInput && w == WireType::kLengthDelimited) {
            graph.inputs.push_back(ParseValueInfo(reader.ReadSubMessage()));
        } else if (f == field::kGraphOutput && w == WireType::kLengthDelimited) {
            graph.outputs.push_back(ParseValueInfo(reader.ReadSubMessage()));
        } else if (f == field::kGraphValueInfo && w == WireType::kLengthDelimited) {
            graph.value_infos.push_back(ParseValueInfo(reader.ReadSubMessage()));
        } else {
            reader.SkipField(w);
        }
    }
    return graph;
}

OnnxGraph ParseModel(const uint8_t* data, size_t size) {
    ProtoReader reader{data, size, 0};
    OnnxGraph graph;
    while (reader.HasMore()) {
        auto [f, w] = reader.ReadTag();
        if (f == field::kModelGraph && w == WireType::kLengthDelimited) {
            graph = ParseGraph(reader.ReadSubMessage());
        } else {
            reader.SkipField(w);
        }
    }
    return graph;
}

// ================================================================
// 第四部分：ONNX → Graph IR 转换
// ================================================================

// ONNX 数据类型枚举值 → IR DataType
// 参考 onnx.proto 中 TensorProto.DataType 的定义
ir::DataType MapDataType(int32_t onnx_type) {
    switch (onnx_type) {
        case 1:  return ir::DataType::kFloat32;    // FLOAT
        case 6:  return ir::DataType::kInt32;       // INT32
        case 7:  return ir::DataType::kInt32;       // INT64 → 近似 int32
        case 10: return ir::DataType::kFloat32;     // FLOAT16 → 近似 float32
        case 16: return ir::DataType::kBFloat16;    // BFLOAT16
        default: return ir::DataType::kFloat32;     // 其余类型默认 float32
    }
}

}  // namespace

// ================================================================
// 公共接口
// ================================================================

self_compiler::Status OnnxImporter::Import(
    const std::string& input_path, ir::Graph& graph) const {

    // ---- 1. 读取整个文件到内存 ----
    std::ifstream file(input_path, std::ios::binary);
    if (!file.is_open()) {
        return Status::Error("无法打开 ONNX 文件: " + input_path);
    }
    std::vector<uint8_t> buffer(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    if (buffer.empty()) {
        return Status::Error("ONNX 文件为空: " + input_path);
    }

    // ---- 2. 解析 protobuf ----
    OnnxGraph onnx_graph;
    try {
        onnx_graph = ParseModel(buffer.data(), buffer.size());
    } catch (const std::exception& e) {
        return Status::Error(
            "ONNX protobuf 解析失败: " + std::string(e.what()));
    }
    if (onnx_graph.nodes.empty()) {
        return Status::Error("ONNX 模型中没有算子节点。");
    }

    // ---- 3. 收集所有 tensor 的类型信息 ----
    // 类型信息分散在三个来源：graph inputs、graph outputs、value_info
    // 以及 initializer（权重）的 TensorProto 自带维度和 dtype
    std::map<std::string, OnnxTensorType> type_map;

    for (const auto& vi : onnx_graph.inputs) {
        type_map[vi.name] = vi.tensor_type;
    }
    for (const auto& vi : onnx_graph.outputs) {
        type_map[vi.name] = vi.tensor_type;
    }
    for (const auto& vi : onnx_graph.value_infos) {
        type_map[vi.name] = vi.tensor_type;
    }
    // initializer 的类型信息作为补充（graph inputs 中可能已包含）
    for (const auto& vi : onnx_graph.initializer_infos) {
        if (type_map.find(vi.name) == type_map.end()) {
            type_map[vi.name] = vi.tensor_type;
        }
    }

    // 区分真正的模型输入和权重（initializer）
    std::set<std::string> initializer_names;
    for (const auto& vi : onnx_graph.initializer_infos) {
        initializer_names.insert(vi.name);
    }

    // ---- 4. 收集所有出现过的 tensor 名 ----
    std::set<std::string> all_tensor_names;
    for (const auto& vi : onnx_graph.inputs) {
        all_tensor_names.insert(vi.name);
    }
    for (const auto& vi : onnx_graph.outputs) {
        all_tensor_names.insert(vi.name);
    }
    for (const auto& node : onnx_graph.nodes) {
        for (const auto& name : node.inputs) {
            if (!name.empty()) all_tensor_names.insert(name);
        }
        for (const auto& name : node.outputs) {
            if (!name.empty()) all_tensor_names.insert(name);
        }
    }

    // ---- 5. 为每个 tensor 创建 IR Tensor ----
    // 动态维度（如 "batch_size"）用 1 作为默认值
    constexpr int64_t kDefaultDynamicDim = 1;

    graph = ir::Graph();
    std::map<std::string, int> name_to_id;

    for (const auto& tensor_name : all_tensor_names) {
        ir::Shape shape;
        ir::DataType dtype = ir::DataType::kFloat32;

        auto it = type_map.find(tensor_name);
        if (it != type_map.end()) {
            dtype = MapDataType(it->second.elem_type);
            for (const auto& dim : it->second.shape) {
                shape.dims.push_back(
                    dim.value > 0 ? dim.value : kDefaultDynamicDim);
            }
        }

        // 没有类型信息的中间 tensor，给一个 1 维占位 shape
        if (shape.dims.empty()) {
            shape.dims = {kDefaultDynamicDim};
        }

        int id = graph.AddTensor(tensor_name, shape, dtype).id;
        name_to_id[tensor_name] = id;
    }

    // ---- 6. 创建 Input op（只为非权重的图输入创建）----
    for (const auto& vi : onnx_graph.inputs) {
        if (initializer_names.count(vi.name)) {
            continue;  // 权重不需要 Input op
        }
        int tid = name_to_id.at(vi.name);
        graph.AddOperation("input." + vi.name, ir::OpKind::kInput, {}, {tid});
    }

    // ---- 7. 为每个 ONNX 节点创建 IR Operation ----
    // 所有 ONNX 算子统一映射为 kUnknown，原始 op_type 保存在 attributes 中。
    // 不做虚假的语义映射——ONNX 的 MatMul 和我们 IR 的 kLinear 含义不同。
    for (size_t i = 0; i < onnx_graph.nodes.size(); ++i) {
        const auto& node = onnx_graph.nodes[i];

        std::vector<int> inputs;
        for (const auto& name : node.inputs) {
            if (name.empty()) continue;  // ONNX 用空字符串表示缺失的可选输入
            auto it = name_to_id.find(name);
            if (it != name_to_id.end()) {
                inputs.push_back(it->second);
            }
        }

        std::vector<int> outputs;
        for (const auto& name : node.outputs) {
            if (name.empty()) continue;
            auto it = name_to_id.find(name);
            if (it != name_to_id.end()) {
                outputs.push_back(it->second);
            }
        }

        // 如果节点没有名字，用 "op_type_序号" 作为名字
        std::string op_name = node.name;
        if (op_name.empty()) {
            op_name = node.op_type + "_" + std::to_string(i);
        }

        auto& op = graph.AddOperation(
            op_name, ir::OpKind::kUnknown, inputs, outputs);
        op.attributes["onnx_op_type"] = node.op_type;
        for (const auto& attr : node.attributes) {
            op.attributes[attr.first] = attr.second;
        }
    }

    // ---- 8. 创建 Output op ----
    for (const auto& vi : onnx_graph.outputs) {
        int tid = name_to_id.at(vi.name);
        graph.AddOperation(
            "output." + vi.name, ir::OpKind::kOutput, {tid}, {});
    }

    return Status::Ok();
}

}  // namespace self_compiler::frontend
