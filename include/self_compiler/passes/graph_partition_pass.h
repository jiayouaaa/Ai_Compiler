#pragma once

#include "self_compiler/passes/pass.h"

namespace self_compiler::passes {

// 图分割 Pass：根据 OpRegistry 的 npu_supported 字段，
// 给每个 op 标记执行目标（NPU 或 CPU）。
// 标记结果写入 op.attributes["exec_target"]，值为 "NPU" 或 "CPU"。
class GraphPartitionPass final : public Pass {
public:
    std::string name() const override;
    self_compiler::Status Run(ir::Graph& graph) override;
};

}  // namespace self_compiler::passes
