#include "self_compiler/memory/memory_planner.h"

#include <algorithm>
#include <array>
#include <vector>

namespace self_compiler::memory {

namespace {

constexpr std::size_t kSramThresholdBytes = 256 * 1024;
constexpr std::size_t kSramCapacityBytes = 256 * 1024;  // Ethos-U55 SRAM 硬件上限

// 判断两个 live interval 的生命周期是否重叠
// 重叠的 tensor 不能共享内存，不重叠的可以
bool Overlaps(const LiveInterval& a, const LiveInterval& b) {
    // a 在 b 之前结束，或 b 在 a 之前结束 → 不重叠
    // 注意：end == start 算重叠（同一个 op 里一个消费完、另一个才产生，但保守起见算重叠）
    return a.start_op_index <= b.end_op_index &&
           b.start_op_index <= a.end_op_index;
}

MemoryRegion ClassifyRegion(const ir::Graph& graph, const LiveInterval& interval) {
    const ir::Tensor* tensor = graph.FindTensor(interval.tensor_id);
    if (tensor != nullptr && tensor->producer_op == -1) {
        return MemoryRegion::kFlash;
    }

    if (interval.bytes > kSramThresholdBytes) {
        return MemoryRegion::kDram;
    }

    return MemoryRegion::kSram;
}

std::size_t RegionIndex(MemoryRegion region) {
    switch (region) {
        case MemoryRegion::kSram:
            return 0;
        case MemoryRegion::kDram:
            return 1;
        case MemoryRegion::kFlash:
            return 2;
        default:
            return 0;
    }
}

struct AllocatedBlock {
    std::size_t offset = 0;
    std::size_t size = 0;
    const LiveInterval* interval = nullptr;
};

struct RegionState {
    std::vector<AllocatedBlock> blocks;
    std::size_t high_watermark = 0;
};

}  // namespace

MemoryPlan MemoryPlanner::BuildPlan(
    const ir::Graph& graph,
    const std::vector<LiveInterval>& intervals) const {
    // 按 bytes 从大到小排序（大 tensor 优先分配，减少碎片）
    std::vector<std::size_t> order(intervals.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return intervals[a].bytes > intervals[b].bytes;
    });

    // 每个 tensor 的分配结果，按原始 tensor_id 索引
    std::vector<BufferAllocation> allocs(intervals.size());
    std::array<RegionState, 3> region_states;
    for (auto& state : region_states) {
        state.blocks.reserve(intervals.size());
    }

    for (std::size_t idx : order) {
        const auto& interval = intervals[idx];
        const MemoryRegion region = ClassifyRegion(graph, interval);
        RegionState& region_state = region_states[RegionIndex(region)];

        // 尝试在已有的块中找一个可以复用的位置
        // 条件：大小 >= 当前 tensor，且生命周期不重叠
        std::size_t best_offset = 0;
        bool found_reuse = false;
        std::size_t best_waste = SIZE_MAX;  // 选浪费最小的（best-fit）

        const bool allow_reuse = region != MemoryRegion::kFlash;
        for (const auto& block : region_state.blocks) {
            if (allow_reuse && block.size >= interval.bytes && !Overlaps(*block.interval, interval)) {
                std::size_t waste = block.size - interval.bytes;
                if (waste < best_waste) {
                    best_waste = waste;
                    best_offset = block.offset;
                    found_reuse = true;
                }
            }
        }

        if (found_reuse) {
            // 复用已有的块
            allocs[idx].offset = best_offset;
            allocs[idx].size_in_bytes = interval.bytes;
        } else {
            // 没有可复用的块，在末尾新分配
            allocs[idx].offset = region_state.high_watermark;
            allocs[idx].size_in_bytes = interval.bytes;
            region_state.high_watermark += interval.bytes;
        }

        allocs[idx].tensor_id = interval.tensor_id;
        allocs[idx].tensor_name = interval.tensor_name;
        allocs[idx].region = region;

        // 记录这个分配块
        region_state.blocks.push_back({allocs[idx].offset, interval.bytes, &interval});
    }

    // SRAM 溢出检测：如果 SRAM 用量超过硬件限制，把最大的 SRAM tensor 降级到 DRAM
    // 反复降级直到 SRAM 用量在限制内
    int spill_count = 0;
    while (region_states[RegionIndex(MemoryRegion::kSram)].high_watermark > kSramCapacityBytes) {
        // 在当前 SRAM 分配中找最大的 tensor 降级
        std::size_t largest_idx = SIZE_MAX;
        std::size_t largest_bytes = 0;
        for (std::size_t i = 0; i < allocs.size(); ++i) {
            if (allocs[i].region == MemoryRegion::kSram && allocs[i].size_in_bytes > largest_bytes) {
                largest_bytes = allocs[i].size_in_bytes;
                largest_idx = i;
            }
        }
        if (largest_idx == SIZE_MAX) break;  // 没有 SRAM tensor 可降级

        // 降级到 DRAM
        allocs[largest_idx].region = MemoryRegion::kDram;
        ++spill_count;

        // 重新计算所有区域（简单做法：清空并重建）
        for (auto& state : region_states) {
            state.blocks.clear();
            state.high_watermark = 0;
        }
        for (std::size_t idx : order) {
            const auto& interval = intervals[idx];
            const MemoryRegion region = allocs[idx].region;
            RegionState& rs = region_states[RegionIndex(region)];

            std::size_t best_offset = 0;
            bool found_reuse = false;
            std::size_t best_waste = SIZE_MAX;

            const bool allow_reuse = region != MemoryRegion::kFlash;
            for (const auto& block : rs.blocks) {
                if (allow_reuse && block.size >= interval.bytes &&
                    !Overlaps(*block.interval, interval)) {
                    std::size_t waste = block.size - interval.bytes;
                    if (waste < best_waste) {
                        best_waste = waste;
                        best_offset = block.offset;
                        found_reuse = true;
                    }
                }
            }

            if (found_reuse) {
                allocs[idx].offset = best_offset;
            } else {
                allocs[idx].offset = rs.high_watermark;
                rs.high_watermark += interval.bytes;
            }

            rs.blocks.push_back({allocs[idx].offset, interval.bytes, &interval});
        }
    }

    // 组装结果
    MemoryPlan plan;
    plan.allocations = std::move(allocs);
    plan.sram_bytes = region_states[RegionIndex(MemoryRegion::kSram)].high_watermark;
    plan.dram_bytes = region_states[RegionIndex(MemoryRegion::kDram)].high_watermark;
    plan.flash_bytes = region_states[RegionIndex(MemoryRegion::kFlash)].high_watermark;
    plan.total_bytes = plan.sram_bytes + plan.dram_bytes + plan.flash_bytes;
    plan.peak_bytes = plan.total_bytes;
    plan.sram_spill_count = spill_count;

    return plan;
}

}  // namespace self_compiler::memory
