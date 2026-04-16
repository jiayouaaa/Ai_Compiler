#include "self_compiler/ir/op_registry.h"

namespace self_compiler::ir {

void OpRegistry::Register(OpKind kind, OpInfo info) {
    registry_[kind] = std::move(info);
}

const OpInfo* OpRegistry::Find(OpKind kind) const {
    auto it = registry_.find(kind);
    if (it == registry_.end()) {
        return nullptr;
    }
    return &it->second;
}

// 全局单例，首次调用时注册所有算子
//
// 每个算子的字段含义：
//   name                    — 显示名
//   num_activation_inputs   — activation（数据流）输入数
//   num_outputs             — 输出数（-1 = 不限制）
//   weight_inputs           — weight（参数流）输入描述列表
//   required_attributes     — 必须存在的 attribute 名
//   npu_supported           — Ethos-U55 是否支持
//
// op.inputs 约定：前 num_activation_inputs 个是 activation，之后是 weight
const OpRegistry& OpRegistry::Instance() {
    static OpRegistry registry = []() {
        OpRegistry r;

        // ================================================================
        // 图边界算子（无 weight）
        // ================================================================
        r.Register(OpKind::kInput, {
            "Input", 0, 1, {}, {}, false});
        r.Register(OpKind::kOutput, {
            "Output", 1, 0, {}, {}, false});

        // ================================================================
        // Transformer 高层算子（lowering 前的高层表示，weight 在 lowering 时展开）
        // ================================================================
        r.Register(OpKind::kTransformerBlock, {
            "TransformerBlock", 1, 1, {},
            {"hidden_size", "intermediate_size",
             "num_attention_heads", "num_key_value_heads"},
            false});
        r.Register(OpKind::kEmbedding, {
            "Embedding", 1, 1,
            {{"weight", true}},
            {}, false});
        r.Register(OpKind::kLmHead, {
            "LmHead", 1, 1,
            {{"weight", true}},
            {}, false});

        // ================================================================
        // Transformer 子算子（lowering 产物）
        // ================================================================
        r.Register(OpKind::kRmsNorm, {
            "RmsNorm", 1, 1,
            {{"weight", true}},
            {"hidden_size"}, true});
        r.Register(OpKind::kQkvProject, {
            "QkvProject", 1, 1,
            {{"wq", true}, {"wk", true}, {"wv", true}},
            {"hidden_size", "num_attention_heads", "num_key_value_heads"},
            false});
        r.Register(OpKind::kRope, {
            "Rope", 1, 1,
            {},
            {}, false});
        r.Register(OpKind::kAttention, {
            "Attention", 1, 1,
            {{"wo", true}},
            {"num_attention_heads", "num_key_value_heads"},
            false});
        r.Register(OpKind::kSplitQkv, {
            "SplitQkv", 1, 3,      // 1 个 activation 输入（QKV 拼接），3 个输出（Q/K/V）
            {},
            {"num_attention_heads", "num_key_value_heads"},
            false});
        r.Register(OpKind::kBatchMatMul, {
            "BatchMatMul", 2, 1,    // 2 个 activation 输入，1 个输出
            {},
            {}, true});
        r.Register(OpKind::kCausalMask, {
            "CausalMask", 1, 1,     // 加因果掩码
            {},
            {}, false});
        r.Register(OpKind::kSoftmax, {
            "Softmax", 1, 1, {}, {}, true});
        r.Register(OpKind::kResidualAdd, {
            "ResidualAdd", 2, 1, {}, {}, true});
        r.Register(OpKind::kSwiGLU, {
            "SwiGLU", 1, 1,
            {{"w_gate", true}, {"w_up", true}, {"w_down", true}},
            {"hidden_size", "intermediate_size"},
            false});
        r.Register(OpKind::kLinear, {
            "Linear", 1, 1,
            {{"weight", true}, {"bias", false}},
            {}, true});

        // ================================================================
        // 通用算子（ONNX 导入 / Ethos-U55 支持）
        // ================================================================
        r.Register(OpKind::kConv2D, {
            "Conv2D", 1, 1,
            {{"weight", true}, {"bias", false}},
            {"kernel_size", "stride", "padding"},
            true});
        r.Register(OpKind::kDepthwiseConv2D, {
            "DepthwiseConv2D", 1, 1,
            {{"weight", true}, {"bias", false}},
            {"kernel_size", "stride", "padding"},
            true});
        r.Register(OpKind::kFullyConnected, {
            "FullyConnected", 1, 1,
            {{"weight", true}, {"bias", false}},
            {}, true});
        r.Register(OpKind::kMaxPool, {
            "MaxPool", 1, 1, {},
            {"kernel_size", "stride"}, true});
        r.Register(OpKind::kAvgPool, {
            "AvgPool", 1, 1, {},
            {"kernel_size", "stride"}, true});
        r.Register(OpKind::kRelu, {
            "Relu", 1, 1, {}, {}, true});
        r.Register(OpKind::kRelu6, {
            "Relu6", 1, 1, {}, {}, true});
        r.Register(OpKind::kSigmoid, {
            "Sigmoid", 1, 1, {}, {}, true});
        r.Register(OpKind::kTanh, {
            "Tanh", 1, 1, {}, {}, true});
        r.Register(OpKind::kAdd, {
            "Add", 2, 1, {}, {}, true});
        r.Register(OpKind::kMul, {
            "Mul", 2, 1, {}, {}, true});
        r.Register(OpKind::kReshape, {
            "Reshape", 1, 1,
            {{"shape", false}},
            {}, true});
        r.Register(OpKind::kTranspose, {
            "Transpose", 1, 1, {}, {}, true});
        r.Register(OpKind::kPad, {
            "Pad", 1, 1,
            {{"paddings", false}},
            {}, true});
        r.Register(OpKind::kConcatenation, {
            "Concatenation", -1, 1, {}, {}, true});
        r.Register(OpKind::kSplit, {
            "Split", 1, -1, {}, {}, true});

        // ================================================================
        // KV Cache 算子
        // ================================================================
        r.Register(OpKind::kKVCacheWrite, {
            "KVCacheWrite", 2, 1,   // inputs: [new_kv, cache], output: [updated_cache]
            {},
            {}, false});
        r.Register(OpKind::kKVCacheRead, {
            "KVCacheRead", 1, 1,    // input: [cache], output: [slice]
            {},
            {}, false});

        // kUnknown 不注册——查表返回 nullptr 时表示未识别算子

        return r;
    }();
    return registry;
}

}  // namespace self_compiler::ir
