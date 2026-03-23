#include "self_compiler/passes/recognize_onnx_ops_pass.h"

#include <cctype>
#include <string>

namespace self_compiler::passes {

namespace {

std::string ToLowerCopy(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool IsRecognizableUnknownOp(const ir::Operation& op) {
    return op.kind == ir::OpKind::kUnknown &&
        op.attributes.find("onnx_op_type") != op.attributes.end();
}

bool HasExactInputCount(const ir::Operation& op, int expected) {
    return static_cast<int>(op.inputs.size()) == expected;
}

ir::OpKind RecognizeKind(const ir::Operation& op) {
    const auto it = op.attributes.find("onnx_op_type");
    if (it == op.attributes.end()) {
        return ir::OpKind::kUnknown;
    }

    const std::string onnx_op_type = ToLowerCopy(it->second);

    if (onnx_op_type == "relu" && HasExactInputCount(op, 1)) {
        return ir::OpKind::kRelu;
    }
    if (onnx_op_type == "sigmoid" && HasExactInputCount(op, 1)) {
        return ir::OpKind::kSigmoid;
    }
    if (onnx_op_type == "tanh" && HasExactInputCount(op, 1)) {
        return ir::OpKind::kTanh;
    }
    if (onnx_op_type == "add" && HasExactInputCount(op, 2)) {
        return ir::OpKind::kAdd;
    }
    if (onnx_op_type == "mul" && HasExactInputCount(op, 2)) {
        return ir::OpKind::kMul;
    }
    if (onnx_op_type == "softmax" && HasExactInputCount(op, 1)) {
        return ir::OpKind::kSoftmax;
    }
    if (onnx_op_type == "transpose" && HasExactInputCount(op, 1)) {
        return ir::OpKind::kTranspose;
    }
    if (onnx_op_type == "reshape" &&
        (HasExactInputCount(op, 1) || HasExactInputCount(op, 2))) {
        return ir::OpKind::kReshape;
    }
    if (onnx_op_type == "split" && HasExactInputCount(op, 1)) {
        return ir::OpKind::kSplit;
    }
    if (onnx_op_type == "concat" && !op.inputs.empty()) {
        return ir::OpKind::kConcatenation;
    }

    return ir::OpKind::kUnknown;
}

}  // namespace

std::string RecognizeOnnxOpsPass::name() const {
    return "RecognizeOnnxOpsPass";
}

self_compiler::Status RecognizeOnnxOpsPass::Run(ir::Graph& graph) {
    for (auto& op : graph.operations()) {
        if (!IsRecognizableUnknownOp(op)) {
            continue;
        }

        const ir::OpKind recognized_kind = RecognizeKind(op);
        if (recognized_kind != ir::OpKind::kUnknown) {
            op.kind = recognized_kind;
            op.attributes["recognized"] = "true";
        }
    }

    return self_compiler::Status::Ok();
}

}  // namespace self_compiler::passes
