#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "self_compiler/ir/graph.h"
#include "self_compiler/memory/live_interval.h"

namespace self_compiler::memory {

enum class MemoryRegion {
    kSram,
    kDram,
    kFlash,
};

inline const char* ToString(MemoryRegion region) {
    switch (region) {
        case MemoryRegion::kSram:
            return "SRAM";
        case MemoryRegion::kDram:
            return "DRAM";
        case MemoryRegion::kFlash:
            return "FLASH";
        default:
            return "UNKNOWN";
    }
}

struct BufferAllocation {
    int tensor_id = -1;
    std::string tensor_name;
    MemoryRegion region = MemoryRegion::kSram;
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
    MemoryPlan BuildPlan(const ir::Graph& graph, const std::vector<LiveInterval>& intervals) const;
};

}  // namespace self_compiler::memory
