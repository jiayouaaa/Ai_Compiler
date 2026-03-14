#include "self_compiler/passes/canonicalize_pass.h"

#include <cctype>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "self_compiler/ir/op_registry.h"

namespace {

int Attr_Stoi(const std::string& text);

self_compiler::Status CheckTensorId(const self_compiler::ir::Graph& graph) {
    const auto& tensors = graph.tensors();
    const auto& operations = graph.operations();

    for (const auto& op : operations) {
        for (int input_tensor_id : op.inputs) {
            if (input_tensor_id < 0 || input_tensor_id >= static_cast<int>(tensors.size())) {
                return self_compiler::Status::Error(
                    op.name + ": Invalid tensor ID in operation inputs: " + std::to_string(input_tensor_id));
            }
        }

        for (int output_tensor_id : op.outputs) {
            if (output_tensor_id < 0 || output_tensor_id >= static_cast<int>(tensors.size())) {
                return self_compiler::Status::Error(
                    op.name + ": Invalid tensor ID in operation outputs: " + std::to_string(output_tensor_id));
            }
        }
    }

    return self_compiler::Status::Ok();
}

self_compiler::Status CheckGraphObjectIds(const self_compiler::ir::Graph& graph) {
    const auto& tensors = graph.tensors();
    const auto& operations = graph.operations();

    for (std::size_t index = 0; index < tensors.size(); ++index) {
        if (tensors[index].id != static_cast<int>(index)) {
            return self_compiler::Status::Error(
                "Tensor index " + std::to_string(index) +
                " does not match tensor.id " + std::to_string(tensors[index].id));
        }
    }

    for (std::size_t index = 0; index < operations.size(); ++index) {
        if (operations[index].id != static_cast<int>(index)) {
            return self_compiler::Status::Error(
                "Operation index " + std::to_string(index) +
                " does not match op.id " + std::to_string(operations[index].id));
        }
    }

    return self_compiler::Status::Ok();
}

self_compiler::Status CheckOpArity(const self_compiler::ir::Operation& op) {
    const auto actual_input_count = static_cast<int>(op.inputs.size());
    const auto actual_output_count = static_cast<int>(op.outputs.size());

    const auto* info = self_compiler::ir::OpRegistry::Instance().Find(op.kind);

    // 未注册的算子（kUnknown 或新增但尚未注册的）：不做 arity 限制
    if (!info) {
        return self_compiler::Status::Ok();
    }

    // num_activation_inputs == -1 表示不限制 activation 输入数（如 Concatenation）
    // 此时只检查输出数
    if (info->num_activation_inputs < 0) {
        if (info->num_outputs >= 0 && actual_output_count != info->num_outputs) {
            return self_compiler::Status::Error(
                op.name + ": " + info->name + " should have " +
                std::to_string(info->num_outputs) + " outputs, but has " +
                std::to_string(actual_output_count));
        }
        return self_compiler::Status::Ok();
    }

    // 合法的 inputs 范围 = activation 数 + 必须 weight 数 ~ activation 数 + 全部 weight 数
    const int min_inputs = info->MinInputs();
    const int max_inputs = info->MaxInputs();

    if (actual_input_count < min_inputs) {
        return self_compiler::Status::Error(
            op.name + ": " + info->name + " requires at least " +
            std::to_string(min_inputs) + " inputs, but has " +
            std::to_string(actual_input_count));
    }
    if (actual_input_count > max_inputs) {
        return self_compiler::Status::Error(
            op.name + ": " + info->name + " allows at most " +
            std::to_string(max_inputs) + " inputs, but has " +
            std::to_string(actual_input_count));
    }
    // num_outputs == -1 表示不限制输出数量（如 Split）
    if (info->num_outputs >= 0 && actual_output_count != info->num_outputs) {
        return self_compiler::Status::Error(
            op.name + ": " + info->name + " should have " +
            std::to_string(info->num_outputs) + " outputs, but has " +
            std::to_string(actual_output_count));
    }

    return self_compiler::Status::Ok();
}

bool HasTensorRank(const self_compiler::ir::Tensor& tensor, std::size_t expected_rank) {
    return tensor.shape.dims.size() == expected_rank;
}

self_compiler::Status CheckSameDType(
    const self_compiler::ir::Operation& op,
    const self_compiler::ir::Tensor& lhs,
    const self_compiler::ir::Tensor& rhs,
    const std::string& lhs_label,
    const std::string& rhs_label) {
    if (lhs.dtype != rhs.dtype) {
        return self_compiler::Status::Error(
            op.name + ": " + lhs_label + " dtype " +
            self_compiler::ir::ToString(lhs.dtype) + " does not match " +
            rhs_label + " dtype " + self_compiler::ir::ToString(rhs.dtype));
    }
    return self_compiler::Status::Ok();
}

self_compiler::Status CheckSameShape(
    const self_compiler::ir::Operation& op,
    const self_compiler::ir::Tensor& lhs,
    const self_compiler::ir::Tensor& rhs,
    const std::string& lhs_label,
    const std::string& rhs_label) {
    if (lhs.shape.dims != rhs.shape.dims) {
        return self_compiler::Status::Error(
            op.name + ": " + lhs_label + " shape " + lhs.shape.ToString() +
            " does not match " + rhs_label + " shape " + rhs.shape.ToString());
    }
    return self_compiler::Status::Ok();
}

self_compiler::Status CheckOpTensorSemantics(
    const self_compiler::ir::Operation& op,
    const self_compiler::ir::Graph& graph) {
    const auto& tensors = graph.tensors();

    auto get_input = [&](std::size_t index) -> const self_compiler::ir::Tensor& {
        return tensors[static_cast<std::size_t>(op.inputs[index])];
    };
    auto get_output = [&](std::size_t index) -> const self_compiler::ir::Tensor& {
        return tensors[static_cast<std::size_t>(op.outputs[index])];
    };

    if (op.kind == self_compiler::ir::OpKind::kInput) {
        const auto& output = get_output(0);
        if (output.shape.dims.empty()) {
            return self_compiler::Status::Error(
                op.name + ": Input output tensor should not have empty shape");
        }
        return self_compiler::Status::Ok();
    }

    if (op.kind == self_compiler::ir::OpKind::kEmbedding) {
        const auto& input = get_input(0);
        const auto& output = get_output(0);

        if (!HasTensorRank(input, 2)) {
            return self_compiler::Status::Error(
                op.name + ": Embedding input tensor should have rank 2, but has rank " +
                std::to_string(input.shape.dims.size()));
        }
        if (!HasTensorRank(output, 3)) {
            return self_compiler::Status::Error(
                op.name + ": Embedding output tensor should have rank 3, but has rank " +
                std::to_string(output.shape.dims.size()));
        }
        if (input.dtype != self_compiler::ir::DataType::kInt32) {
            return self_compiler::Status::Error(
                op.name + ": Embedding input tensor should have dtype i32, but has " +
                self_compiler::ir::ToString(input.dtype));
        }
        if (output.dtype == self_compiler::ir::DataType::kInt32) {
            return self_compiler::Status::Error(
                op.name + ": Embedding output tensor should be a floating-point activation, but got i32");
        }
        if (input.shape.dims[0] != output.shape.dims[0] ||
            input.shape.dims[1] != output.shape.dims[1]) {
            return self_compiler::Status::Error(
                op.name + ": Embedding should preserve batch/sequence dims, but input shape " +
                input.shape.ToString() + " maps to output shape " + output.shape.ToString());
        }
        return self_compiler::Status::Ok();
    }

    if (op.kind == self_compiler::ir::OpKind::kTransformerBlock) {
        const auto& input = get_input(0);
        const auto& output = get_output(0);
        if (!HasTensorRank(input, 3) || !HasTensorRank(output, 3)) {
            return self_compiler::Status::Error(
                op.name + ": TransformerBlock input/output tensors should both have rank 3");
        }

        auto shape_status = CheckSameShape(op, input, output, "input", "output");
        if (!shape_status.ok) {
            return shape_status;
        }
        auto dtype_status = CheckSameDType(op, input, output, "input", "output");
        if (!dtype_status.ok) {
            return dtype_status;
        }

        const int hidden_size = Attr_Stoi(op.attributes.find("hidden_size")->second);
        if (input.shape.dims[2] != hidden_size) {
            return self_compiler::Status::Error(
                op.name + ": hidden_size attribute " + std::to_string(hidden_size) +
                " does not match tensor hidden dim " + std::to_string(input.shape.dims[2]));
        }
        return self_compiler::Status::Ok();
    }

    if (op.kind == self_compiler::ir::OpKind::kLmHead) {
        const auto& input = get_input(0);
        const auto& output = get_output(0);
        if (!HasTensorRank(input, 3) || !HasTensorRank(output, 3)) {
            return self_compiler::Status::Error(
                op.name + ": LmHead input/output tensors should both have rank 3");
        }
        if (input.shape.dims[0] != output.shape.dims[0] ||
            input.shape.dims[1] != output.shape.dims[1]) {
            return self_compiler::Status::Error(
                op.name + ": LmHead should preserve batch/sequence dims, but input shape " +
                input.shape.ToString() + " maps to output shape " + output.shape.ToString());
        }
        auto dtype_status = CheckSameDType(op, input, output, "input", "output");
        if (!dtype_status.ok) {
            return dtype_status;
        }
        return self_compiler::Status::Ok();
    }

    if (op.kind == self_compiler::ir::OpKind::kOutput) {
        const auto& input = get_input(0);
        if (input.shape.dims.empty()) {
            return self_compiler::Status::Error(
                op.name + ": Output input tensor should not have empty shape");
        }
        return self_compiler::Status::Ok();
    }

    if (op.kind == self_compiler::ir::OpKind::kResidualAdd) {
        const auto& lhs = get_input(0);
        const auto& rhs = get_input(1);
        const auto& output = get_output(0);

        auto lhs_rhs_shape = CheckSameShape(op, lhs, rhs, "input0", "input1");
        if (!lhs_rhs_shape.ok) {
            return lhs_rhs_shape;
        }
        auto lhs_rhs_dtype = CheckSameDType(op, lhs, rhs, "input0", "input1");
        if (!lhs_rhs_dtype.ok) {
            return lhs_rhs_dtype;
        }
        auto out_shape = CheckSameShape(op, lhs, output, "input0", "output");
        if (!out_shape.ok) {
            return out_shape;
        }
        auto out_dtype = CheckSameDType(op, lhs, output, "input0", "output");
        if (!out_dtype.ok) {
            return out_dtype;
        }
        return self_compiler::Status::Ok();
    }

    if (op.kind == self_compiler::ir::OpKind::kRmsNorm ||
        op.kind == self_compiler::ir::OpKind::kRope ||
        op.kind == self_compiler::ir::OpKind::kAttention ||
        op.kind == self_compiler::ir::OpKind::kSoftmax) {
        const auto& input = get_input(0);
        const auto& output = get_output(0);
        auto shape_status = CheckSameShape(op, input, output, "input", "output");
        if (!shape_status.ok) {
            return shape_status;
        }
        return CheckSameDType(op, input, output, "input", "output");
    }

    if (op.kind == self_compiler::ir::OpKind::kLinear ||
        op.kind == self_compiler::ir::OpKind::kQkvProject ||
        op.kind == self_compiler::ir::OpKind::kSwiGLU) {
        const auto& input = get_input(0);
        const auto& output = get_output(0);
        if (input.shape.dims.size() != output.shape.dims.size()) {
            return self_compiler::Status::Error(
                op.name + ": input rank " + std::to_string(input.shape.dims.size()) +
                " does not match output rank " + std::to_string(output.shape.dims.size()));
        }
        return CheckSameDType(op, input, output, "input", "output");
    }

    return self_compiler::Status::Ok();
}

self_compiler::Status CheckRequiredAttributes(const self_compiler::ir::Operation& op) {
    const auto* info = self_compiler::ir::OpRegistry::Instance().Find(op.kind);

    // 未注册的算子或没有必需属性的算子：跳过
    if (!info || info->required_attributes.empty()) {
        return self_compiler::Status::Ok();
    }

    for (const auto& attr : info->required_attributes) {
        if (op.attributes.find(attr) == op.attributes.end()) {
            return self_compiler::Status::Error(
                op.name + ": Missing required attribute '" + attr +
                "' for " + info->name + " operation");
        }
    }

    return self_compiler::Status::Ok();
}

int Attr_Stoi(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("Empty string cannot be converted to integer");
    }

    int sign = 1;
    std::size_t index = 0;
    if (text[0] == '-') {
        sign = -1;
        index = 1;
    }

    if (index == text.size()) {
        throw std::invalid_argument("Sign only string cannot be converted to integer");
    }

    int result = 0;
    for (; index < text.size(); ++index) {
        const unsigned char ch = static_cast<unsigned char>(text[index]);
        if (std::isdigit(ch) == 0) {
            throw std::invalid_argument("Invalid integer string: " + text);
        }

        if (result > (std::numeric_limits<int>::max() - (text[index] - '0')) / 10) {
            throw std::out_of_range("Integer out of range: " + text);
        }
        result = result * 10 + (text[index] - '0');
    }

    return sign * result;
}

self_compiler::Status CheckAttributeValue(const self_compiler::ir::Operation& op) {
    const auto* info = self_compiler::ir::OpRegistry::Instance().Find(op.kind);

    if (!info || info->required_attributes.empty()) {
        return self_compiler::Status::Ok();
    }

    for (const auto& attr : info->required_attributes) {
        const auto it = op.attributes.find(attr);
        if (it == op.attributes.end()) {
            return self_compiler::Status::Error(
                op.name + ": Missing attribute value for '" + attr + "'");
        }

        const auto& attr_str_value = it->second;

        try {
            const int attr_value = Attr_Stoi(attr_str_value);
            if (attr_value <= 0) {
                return self_compiler::Status::Error(
                    op.name + ": Attribute '" + attr +
                    "' must be a positive integer, but got '" + attr_str_value + "'");
            }
        } catch (const std::exception&) {
            return self_compiler::Status::Error(
                op.name + ": Attribute '" + attr +
                "' must be a valid integer, but got '" + attr_str_value + "'");
        }
    }

    return self_compiler::Status::Ok();
}

std::string CanonicalizeIdentifierBody(const std::string& source) {
    std::string result;
    result.reserve(source.size());
    bool last_was_underscore = false;

    for (unsigned char ch : source) {
        if (std::isalnum(ch) != 0) {
            result.push_back(static_cast<char>(std::tolower(ch)));
            last_was_underscore = false;
        } else if (!last_was_underscore) {
            result.push_back('_');
            last_was_underscore = true;
        }
    }

    while (!result.empty() && result.front() == '_') {
        result.erase(result.begin());
    }
    while (!result.empty() && result.back() == '_') {
        result.pop_back();
    }

    return result;
}

std::string CanonicalizeName(const std::string& raw_name, const std::string& fallback_prefix, int id) {
    std::string source = raw_name;
    if (source.empty()) {
        source = fallback_prefix + "_" + std::to_string(id);
    }

    std::string result = CanonicalizeIdentifierBody(source);
    if (result.empty()) {
        return fallback_prefix + "_" + std::to_string(id);
    }
    return result;
}

std::string CanonicalizeAttributeKey(const std::string& raw_key) {
    return CanonicalizeIdentifierBody(raw_key);
}

self_compiler::Status CheckAttributeRelations(const self_compiler::ir::Operation& op) {
    if (op.kind != self_compiler::ir::OpKind::kTransformerBlock) {
        return self_compiler::Status::Ok();
    }

    const int hidden_size = Attr_Stoi(op.attributes.find("hidden_size")->second);
    const int num_attention_heads = Attr_Stoi(op.attributes.find("num_attention_heads")->second);
    const int num_key_value_heads = Attr_Stoi(op.attributes.find("num_key_value_heads")->second);

    if (hidden_size % num_attention_heads != 0) {
        return self_compiler::Status::Error(
            op.name + ": hidden_size " + std::to_string(hidden_size) +
            " must be divisible by num_attention_heads " +
            std::to_string(num_attention_heads));
    }

    if (num_attention_heads % num_key_value_heads != 0) {
        return self_compiler::Status::Error(
            op.name + ": num_attention_heads " +
            std::to_string(num_attention_heads) +
            " must be divisible by num_key_value_heads " +
            std::to_string(num_key_value_heads));
    }

    return self_compiler::Status::Ok();
}

std::string Trim(const std::string& s) {
    if (s.empty()) {
        return "";
    }

    int left = 0;
    int right = static_cast<int>(s.size()) - 1;
    while (left <= right && std::isspace(static_cast<unsigned char>(s[left])) != 0) {
        ++left;
    }
    while (left <= right && std::isspace(static_cast<unsigned char>(s[right])) != 0) {
        --right;
    }

    return (left <= right) ? s.substr(left, right - left + 1) : "";
}

std::string ToLowerCopy(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

bool IsIntegerString(const std::string& s) {
    if (s.empty()) {
        return false;
    }

    int i = 0;
    if (s[0] == '+' || s[0] == '-') {
        if (s.size() == 1) {
            return false;
        }
        i = 1;
    }

    for (; i < static_cast<int>(s.size()); ++i) {
        if (std::isdigit(static_cast<unsigned char>(s[i])) == 0) {
            return false;
        }
    }

    return true;
}

std::string NormalizeIntegerString(const std::string& s) {
    int i = 0;
    bool negative = false;

    if (s[0] == '+' || s[0] == '-') {
        negative = (s[0] == '-');
        i = 1;
    }

    while (i < static_cast<int>(s.size()) && s[i] == '0') {
        ++i;
    }

    std::string core = (i == static_cast<int>(s.size())) ? "0" : s.substr(i);
    if (core == "0") {
        return "0";
    }
    return negative ? "-" + core : core;
}

std::string CanonicalizeAttributeValue(std::string s) {
    s = Trim(s);

    const std::string lower = ToLowerCopy(s);
    if (lower == "true") {
        return "True";
    }
    if (lower == "false") {
        return "False";
    }

    if (IsIntegerString(s)) {
        return NormalizeIntegerString(s);
    }

    return s;
}

void NormalizeAttributes(self_compiler::ir::Operation& op) {
    decltype(op.attributes) normalized;

    for (const auto& item : op.attributes) {
        const auto key = CanonicalizeAttributeKey(item.first);
        const auto value = CanonicalizeAttributeValue(item.second);
        normalized[key] = value;
    }

    op.attributes = std::move(normalized);
}

void NormalizeNames(self_compiler::ir::Graph& graph) {
    for (auto& tensor : graph.tensors()) {
        tensor.name = CanonicalizeName(tensor.name, "tensor", tensor.id);
    }

    for (auto& op : graph.operations()) {
        op.name = CanonicalizeName(op.name, "op", op.id);
    }
}

bool ContainId(const std::vector<int>& ids, int target) {
    auto it = std::find(ids.begin(), ids.end(), target);
    return it != ids.end();
}

self_compiler::Status ValidateProducerConsumerConsistency(
    const self_compiler::ir::Graph& graph) {
    const auto& tensors = graph.tensors();
    const auto& operations = graph.operations();
    const int tensor_count = static_cast<int>(tensors.size());
    const int operation_count = static_cast<int>(operations.size());

    for (const auto& tensor : tensors) {
        if (tensor.producer_op != -1) {
            if (tensor.producer_op < 0 || tensor.producer_op >= operation_count) {
                return self_compiler::Status::Error(
                    "Tensor T" + std::to_string(tensor.id) +
                    " has an invalid producer_op O" + std::to_string(tensor.producer_op));
            }

            if (!ContainId(operations[tensor.producer_op].outputs, tensor.id)) {
                return self_compiler::Status::Error(
                    "Tensor T" + std::to_string(tensor.id) +
                    " producer_op O" + std::to_string(tensor.producer_op) +
                    " does not list it as output");
            }
        }

        for (int consumer_op_id : tensor.consumer_ops) {
            if (consumer_op_id < 0 || consumer_op_id >= operation_count) {
                return self_compiler::Status::Error(
                    "Tensor T" + std::to_string(tensor.id) +
                    " has an invalid consumer_op O" + std::to_string(consumer_op_id));
            }

            if (!ContainId(operations[consumer_op_id].inputs, tensor.id)) {
                return self_compiler::Status::Error(
                    "Tensor T" + std::to_string(tensor.id) +
                    " consumer_op O" + std::to_string(consumer_op_id) +
                    " does not list it as input");
            }
        }
    }

    for (const auto& op : operations) {
        for (int input_tensor_id : op.inputs) {
            if (input_tensor_id < 0 || input_tensor_id >= tensor_count) {
                return self_compiler::Status::Error(
                    "Operation O" + std::to_string(op.id) +
                    " has an invalid input tensor ID: " + std::to_string(input_tensor_id));
            }

            if (!ContainId(tensors[input_tensor_id].consumer_ops, op.id)) {
                return self_compiler::Status::Error(
                    "Operation O" + std::to_string(op.id) +
                    " lists T" + std::to_string(input_tensor_id) +
                    " as input, but it does not list O" + std::to_string(op.id) +
                    " as consumer");
            }
        }

        for (int output_tensor_id : op.outputs) {
            if (output_tensor_id < 0 || output_tensor_id >= tensor_count) {
                return self_compiler::Status::Error(
                    "Operation O" + std::to_string(op.id) +
                    " has an invalid output tensor ID: " + std::to_string(output_tensor_id));
            }

            if (tensors[output_tensor_id].producer_op != op.id) {
                return self_compiler::Status::Error(
                    "Operation O" + std::to_string(op.id) +
                    " lists T" + std::to_string(output_tensor_id) +
                    " as output, but its producer_op is O" +
                    std::to_string(tensors[output_tensor_id].producer_op));
            }
        }
    }

    return self_compiler::Status::Ok();
}

self_compiler::Status CheckTensorMetadata(const self_compiler::ir::Graph& graph) {
    const auto& tensors = graph.tensors();

    for (const auto& tensor : tensors) {
        if (tensor.dtype == self_compiler::ir::DataType::kUnknown) {
            return self_compiler::Status::Error(
                "Tensor T" + std::to_string(tensor.id) +
                " (" + tensor.name + ") has unknown dtype");
        }

        if (tensor.shape.dims.empty()) {
            return self_compiler::Status::Error(
                "Tensor T" + std::to_string(tensor.id) +
                " (" + tensor.name + ") has empty shape");
        }

        for (std::size_t dim_index = 0; dim_index < tensor.shape.dims.size(); ++dim_index) {
            const auto dim = tensor.shape.dims[dim_index];
            if (dim <= 0) {
                return self_compiler::Status::Error(
                    "Tensor T" + std::to_string(tensor.id) +
                    " (" + tensor.name + ") has invalid dimension at axis " +
                    std::to_string(dim_index) + ": " + std::to_string(dim));
            }
        }
    }

    return self_compiler::Status::Ok();
}

}  // namespace

namespace self_compiler::passes {

std::string CanonicalizePass::name() const {
    return "CanonicalizePass";
}

self_compiler::Status CanonicalizePass::Run(ir::Graph& graph) {
    auto tensor_status = CheckTensorId(graph);
    if (!tensor_status.ok) {
        return self_compiler::Status::Error("Tensor ID check failed: " + tensor_status.message);
    }

    auto graph_object_id_status = CheckGraphObjectIds(graph);
    if (!graph_object_id_status.ok) {
        return self_compiler::Status::Error("Graph object id check failed: " + graph_object_id_status.message);
    }

    auto tensor_metadata_status = CheckTensorMetadata(graph);
    if (!tensor_metadata_status.ok) {
        return self_compiler::Status::Error(
            "Tensor metadata check failed: " + tensor_metadata_status.message);
    }

    for (const auto& op : graph.operations()) {
        auto arity_status = CheckOpArity(op);
        if (!arity_status.ok) {
            return self_compiler::Status::Error("Operation arity check failed: " + arity_status.message);
        }

        auto required_attributes_status = CheckRequiredAttributes(op);
        if (!required_attributes_status.ok) {
            return self_compiler::Status::Error("Required attributes check failed: " + required_attributes_status.message);
        }

        auto attribute_value_status = CheckAttributeValue(op);
        if (!attribute_value_status.ok) {
            return self_compiler::Status::Error("Attribute value check failed: " + attribute_value_status.message);
        }

        auto attribute_relations_status = CheckAttributeRelations(op);
        if (!attribute_relations_status.ok) {
            return self_compiler::Status::Error("Attribute relations check failed: " + attribute_relations_status.message);
        }

        auto op_tensor_semantics_status = CheckOpTensorSemantics(op, graph);
        if (!op_tensor_semantics_status.ok) {
            return self_compiler::Status::Error(
                "Operation tensor semantics check failed: " + op_tensor_semantics_status.message);
        }
    }

    auto producer_consumer_status = ValidateProducerConsumerConsistency(graph);
    if (!producer_consumer_status.ok) {
        return self_compiler::Status::Error(
            "Producer/consumer consistency check failed: " + producer_consumer_status.message);
    }

    for (auto& op : graph.operations()) {
        NormalizeAttributes(op);
    }

    NormalizeNames(graph);

    for (auto& op : graph.operations()) {
        op.attributes["canonicalized"] = "true";
    }

    return self_compiler::Status::Ok();
}

}  // namespace self_compiler::passes

