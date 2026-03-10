#pragma once

#include <cstddef>
#include <vector>

#include "self_compiler/memory/live_interval.h"

namespace self_compiler::memory {

struct BufferAllocation {
    int tensor_id = -1;
    std::string tensor_name;
    std::size_t offset = 0;
    std::size_t size_in_bytes = 0;
};

struct MemoryPlan {
    std::vector<BufferAllocation> allocations;
    std::size_t total_bytes = 0;
    std::size_t peak_bytes = 0;
};

class MemoryPlanner {
public:
    MemoryPlan BuildPlan(const std::vector<LiveInterval>& intervals) const;
};

}  // namespace self_compiler::memory
