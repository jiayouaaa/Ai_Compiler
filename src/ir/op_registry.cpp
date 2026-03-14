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
const OpRegistry& OpRegistry::Instance() {
    static OpRegistry registry = []() {
        OpRegistry r;

        // ================================================================
        // 图边界算子
        // ================================================================
        r.Register(OpKind::kInput, {
            "Input", 0, 0, 1, {}, false});
        r.Register(OpKind::kOutput, {
            "Output", 1, 1, 0, {}, false});

        // ================================================================
        // Transformer 高层算子
        // ================================================================
        r.Register(OpKind::kTransformerBlock, {
            "TransformerBlock", 1, 1, 1,
            {"hidden_size", "intermediate_size",
             "num_attention_heads", "num_key_value_heads"},
            false});
        r.Register(OpKind::kEmbedding, {
            "Embedding", 1, 1, 1, {}, false});
        r.Register(OpKind::kLmHead, {
            "LmHead", 1, 1, 1, {}, false});

        // ================================================================
        // Transformer 子算子（lowering 产物）
        // ================================================================
        r.Register(OpKind::kRmsNorm, {
            "RmsNorm", 1, 1, 1, {"hidden_size"}, true});
        r.Register(OpKind::kQkvProject, {
            "QkvProject", 1, 1, 1,
            {"hidden_size", "num_attention_heads", "num_key_value_heads"},
            false});
        r.Register(OpKind::kRope, {
            "Rope", 1, 1, 1, {}, false});
        r.Register(OpKind::kAttention, {
            "Attention", 1, 1, 1,
            {"num_attention_heads", "num_key_value_heads"},
            false});
        r.Register(OpKind::kSoftmax, {
            "Softmax", 1, 1, 1, {}, true});
        r.Register(OpKind::kResidualAdd, {
            "ResidualAdd", 2, 2, 1, {}, true});
        r.Register(OpKind::kSwiGLU, {
            "SwiGLU", 1, 1, 1,
            {"hidden_size", "intermediate_size"},
            false});
        r.Register(OpKind::kLinear, {
            "Linear", 1, 1, 1, {}, true});

        // ================================================================
        // 通用算子（ONNX 导入 / Ethos-U55 支持）
        // ================================================================
        r.Register(OpKind::kConv2D, {
            "Conv2D", 2, 3, 1,
            {"kernel_size", "stride", "padding"},
            true});
        r.Register(OpKind::kDepthwiseConv2D, {
            "DepthwiseConv2D", 2, 3, 1,
            {"kernel_size", "stride", "padding"},
            true});
        r.Register(OpKind::kFullyConnected, {
            "FullyConnected", 2, 3, 1, {}, true});
        r.Register(OpKind::kMaxPool, {
            "MaxPool", 1, 1, 1, {"kernel_size", "stride"}, true});
        r.Register(OpKind::kAvgPool, {
            "AvgPool", 1, 1, 1, {"kernel_size", "stride"}, true});
        r.Register(OpKind::kRelu, {
            "Relu", 1, 1, 1, {}, true});
        r.Register(OpKind::kRelu6, {
            "Relu6", 1, 1, 1, {}, true});
        r.Register(OpKind::kSigmoid, {
            "Sigmoid", 1, 1, 1, {}, true});
        r.Register(OpKind::kTanh, {
            "Tanh", 1, 1, 1, {}, true});
        r.Register(OpKind::kAdd, {
            "Add", 2, 2, 1, {}, true});
        r.Register(OpKind::kMul, {
            "Mul", 2, 2, 1, {}, true});
        r.Register(OpKind::kReshape, {
            "Reshape", 1, 2, 1, {}, true});
        r.Register(OpKind::kTranspose, {
            "Transpose", 1, 1, 1, {}, true});
        r.Register(OpKind::kPad, {
            "Pad", 1, 2, 1, {}, true});
        r.Register(OpKind::kConcatenation, {
            "Concatenation", 1, 100, 1, {}, true});
        r.Register(OpKind::kSplit, {
            "Split", 1, 1, -1, {}, true});

        // kUnknown 不注册——查表返回 nullptr 时表示未识别算子

        return r;
    }();
    return registry;
}

}  // namespace self_compiler::ir
