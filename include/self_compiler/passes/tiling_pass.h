#pragma once

#include "self_compiler/passes/pass.h"

namespace self_compiler::passes {

// TilingPass：对 MatMul/Linear 类 op 按 SRAM 容量做分块。
//
// 对每个 kBatchMatMul 和 kLinear op：
//   1. 计算如果不 tiling，输入+输出 tensor 需要的总字节数
//   2. 如果超过 SRAM 容量，按 M 维和 N 维切块
//   3. 把一个大 op 替换为多个 TiledOp（同 OpKind，带 tile offset 属性）
//   4. 每个 TiledOp 的输出 tensor 是 tile 大小
//
// 当前限制：
//   - 不沿 K 维切（避免需要累加逻辑）
//   - tile_size 用贪心法选择，不做 autotuning
class TilingPass final : public Pass {
public:
    std::string name() const override;
    self_compiler::Status Run(ir::Graph& graph) override;
};

}  // namespace self_compiler::passes
