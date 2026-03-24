#pragma once

#include "self_compiler/backend/ethosu_command.h"
#include "self_compiler/ir/graph.h"
#include "self_compiler/memory/memory_planner.h"

namespace self_compiler::backend {

// Ethos-U55 NPU 后端
// 将 lowered 后的 Graph + MemoryPlan 翻译成 Ethos-U55 命令流
class EthosUBackend {
public:
    EthosUCommandStream Emit(
        const ir::Graph& graph,
        const memory::MemoryPlan& plan) const;
};

}  // namespace self_compiler::backend
