#pragma once

#include "self_compiler/passes/pass.h"

namespace self_compiler::passes {

// QuantizePass：将浮点 tensor 转为 int8 对称量化。
//
// 对每个 bf16/f32 的 tensor：
//   1. 根据 shape 和 dtype 估算数值范围（假定 max_abs = 1.0）
//   2. 计算 scale = max_abs / 127（对称量化，zero_point = 0）
//   3. 将 dtype 改为 int8，填入 scale 和 zero_point
//
// 限制：
//   - 当前为"假量化"（fake quantization），不做真实 calibration
//   - 使用 per-tensor 对称量化
//   - 不插入反量化 op，假设全图统一量化
class QuantizePass final : public Pass {
public:
    std::string name() const override;
    self_compiler::Status Run(ir::Graph& graph) override;
};

}  // namespace self_compiler::passes
