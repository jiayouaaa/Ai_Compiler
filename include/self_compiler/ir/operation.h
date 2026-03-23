#pragma once

#include <map>
#include <string>
#include <vector>

namespace self_compiler::ir {

enum class OpKind {
    // ---- 图边界 ----
    kInput,
    kOutput,

    // ---- Transformer 高层算子 ----
    kTransformerBlock,
    kEmbedding,
    kLmHead,

    // ---- Transformer 子算子（lowering 产物）----
    kRmsNorm,
    kQkvProject,
    kRope,
    kAttention,
    kSplitQkv,
    kBatchMatMul,
    kCausalMask,
    kSoftmax,
    kResidualAdd,
    kSwiGLU,
    kLinear,

    // ---- 通用算子（ONNX 导入 / Ethos-U55 支持）----
    kConv2D,
    kDepthwiseConv2D,
    kFullyConnected,
    kMaxPool,
    kAvgPool,
    kRelu,
    kRelu6,
    kSigmoid,
    kTanh,
    kAdd,
    kMul,
    kReshape,
    kTranspose,
    kPad,
    kConcatenation,
    kSplit,

    // ---- 未识别 ----
    kUnknown,
};

inline std::string ToString(OpKind kind) {
    switch (kind) {
        case OpKind::kInput:             return "Input";
        case OpKind::kOutput:            return "Output";
        case OpKind::kTransformerBlock:  return "TransformerBlock";
        case OpKind::kEmbedding:         return "Embedding";
        case OpKind::kLmHead:            return "LmHead";
        case OpKind::kRmsNorm:           return "RmsNorm";
        case OpKind::kQkvProject:        return "QkvProject";
        case OpKind::kRope:              return "Rope";
        case OpKind::kAttention:         return "Attention";
        case OpKind::kSplitQkv:          return "SplitQkv";
        case OpKind::kBatchMatMul:       return "BatchMatMul";
        case OpKind::kCausalMask:        return "CausalMask";
        case OpKind::kSoftmax:           return "Softmax";
        case OpKind::kResidualAdd:       return "ResidualAdd";
        case OpKind::kSwiGLU:            return "SwiGLU";
        case OpKind::kLinear:            return "Linear";
        case OpKind::kConv2D:            return "Conv2D";
        case OpKind::kDepthwiseConv2D:   return "DepthwiseConv2D";
        case OpKind::kFullyConnected:    return "FullyConnected";
        case OpKind::kMaxPool:           return "MaxPool";
        case OpKind::kAvgPool:           return "AvgPool";
        case OpKind::kRelu:              return "Relu";
        case OpKind::kRelu6:             return "Relu6";
        case OpKind::kSigmoid:           return "Sigmoid";
        case OpKind::kTanh:              return "Tanh";
        case OpKind::kAdd:               return "Add";
        case OpKind::kMul:               return "Mul";
        case OpKind::kReshape:           return "Reshape";
        case OpKind::kTranspose:         return "Transpose";
        case OpKind::kPad:               return "Pad";
        case OpKind::kConcatenation:     return "Concatenation";
        case OpKind::kSplit:             return "Split";
        default:                         return "Unknown";
    }
}

struct Operation {
    int id = -1;
    std::string name;
    OpKind kind = OpKind::kUnknown;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::map<std::string, std::string> attributes;
};

}  // namespace self_compiler::ir
