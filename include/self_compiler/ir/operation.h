#pragma once

#include <map>
#include <string>
#include <vector>

namespace self_compiler::ir {

enum class OpKind {
    kInput,
    kEmbedding,
    kTransformerBlock,
    kRmsNorm,
    kLinear,
    kQkvProject,
    kRope,
    kAttention,
    kSoftmax,
    kResidualAdd,
    kSwiGLU,
    kLmHead,
    kOutput,
    kUnknown,
};

inline std::string ToString(OpKind kind) {
    switch (kind) {
        case OpKind::kInput:
            return "Input";
        case OpKind::kEmbedding:
            return "Embedding";
        case OpKind::kTransformerBlock:
            return "TransformerBlock";
        case OpKind::kRmsNorm:
            return "RmsNorm";
        case OpKind::kLinear:
            return "Linear";
        case OpKind::kQkvProject:
            return "QkvProject";
        case OpKind::kRope:
            return "Rope";
        case OpKind::kAttention:
            return "Attention";
        case OpKind::kSoftmax:
            return "Softmax";
        case OpKind::kResidualAdd:
            return "ResidualAdd";
        case OpKind::kSwiGLU:
            return "SwiGLU";
        case OpKind::kLmHead:
            return "LmHead";
        case OpKind::kOutput:
            return "Output";
        default:
            return "Unknown";
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
