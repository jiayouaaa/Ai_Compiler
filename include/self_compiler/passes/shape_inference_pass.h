#pragma once

#include "self_compiler/passes/pass.h"

namespace self_compiler::passes {

// ShapeInferencePass：为每个 op 按输入 shape 推导输出 shape。
//
// 工作方式（双模式）：
//   - 推导模式：如果 op 的输出 tensor shape 为空，则按推导规则填入
//   - 校验模式：如果 op 的输出 tensor shape 已存在，则验证是否与推导结果一致
// 不一致即报错，定位到具体 op 与 tensor。
//
// 设计意图：
//   - 当前流水线中 LowerTransformerToRuntimePass 仍内联填 shape，
//     本 pass 以校验器身份运行，起到回归保护作用
//   - Phase 3 移除内联 shape 计算后，本 pass 自动切换为推导器
//   - 规则覆盖 Llama 推理链路涉及的所有 op；未覆盖的 op（Conv2D/Pool 等）
//     在输出 shape 已填写时直接放行，未填写时跳过
class ShapeInferencePass final : public Pass {
public:
    std::string name() const override;
    self_compiler::Status Run(ir::Graph& graph) override;
};

}  // namespace self_compiler::passes
