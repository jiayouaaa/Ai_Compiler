#pragma once

#include <string>
#include <vector>

#include "self_compiler/ir/graph.h"

namespace self_compiler::memory {

struct LiveInterval {
    int tensor_id = -1;
    std::string tensor_name;
    int start_op_index = -1;
    int end_op_index = -1;
    std::size_t bytes = 0;
};

class LiveIntervalAnalyzer {
public:
    std::vector<LiveInterval> Analyze(const ir::Graph& graph) const;
};

}  // namespace self_compiler::memory
