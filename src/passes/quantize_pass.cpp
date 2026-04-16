#include "self_compiler/passes/quantize_pass.h"

namespace self_compiler::passes {

std::string QuantizePass::name() const {
    return "QuantizePass";
}

self_compiler::Status QuantizePass::Run(ir::Graph& graph) {
    for (auto& tensor : graph.tensors()) {
        // 只量化浮点 tensor
        if (tensor.dtype != ir::DataType::kBFloat16 &&
            tensor.dtype != ir::DataType::kFloat32) {
            continue;
        }

        // 对称量化：假定 max_abs = 1.0（教学简化版）
        // 真实编译器会用 calibration 数据集统计真实 min/max
        constexpr float kAssumedMaxAbs = 1.0f;
        constexpr float kInt8Max = 127.0f;

        tensor.scale = kAssumedMaxAbs / kInt8Max;
        tensor.zero_point = 0;  // 对称量化
        tensor.dtype = ir::DataType::kInt8;
    }

    return Status::Ok();
}

}  // namespace self_compiler::passes
