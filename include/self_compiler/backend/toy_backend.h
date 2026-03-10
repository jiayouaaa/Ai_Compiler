#pragma once

#include "self_compiler/backend/command_stream.h"
#include "self_compiler/ir/graph.h"
#include "self_compiler/memory/memory_planner.h"

namespace self_compiler::backend {

class ToyAcceleratorBackend {
public:
    CommandStream Emit(const ir::Graph& graph, const memory::MemoryPlan& plan) const;
};

}  // namespace self_compiler::backend
