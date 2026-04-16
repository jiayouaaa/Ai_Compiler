#include "self_compiler/memory/memory_planner.h"

#include <algorithm>
#include <array>
#include <vector>

namespace self_compiler::memory {

namespace {

constexpr std::size_t kSramCapacityBytes = 256 * 1024;  // Ethos-U55 SRAM 硬件上限
constexpr std::size_t kAlignment = 16;                   // DMA 对齐要求

// 向上对齐到 kAlignment 的倍数
std::size_t AlignUp(std::size_t value) {
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}

// ============================================================
// FreeList：管理空闲区间，支持分配、回收、相邻块合并
// ============================================================

struct FreeBlock {
    std::size_t offset;
    std::size_t size;
};

class FreeList {
public:
    // 初始化：整个区域从 0 开始，无限大（用 SIZE_MAX/2 代表）
    FreeList() {
        blocks_.push_back({0, SIZE_MAX / 2});
    }

    // 分配 `size` 字节（已对齐），返回 offset。
    // 使用 best-fit：在所有够大的空闲块中选最小的，减少碎片。
    // 返回 SIZE_MAX 表示分配失败。
    std::size_t Allocate(std::size_t size) {
        std::size_t best_idx = SIZE_MAX;
        std::size_t best_waste = SIZE_MAX;

        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            if (blocks_[i].size >= size) {
                std::size_t waste = blocks_[i].size - size;
                if (waste < best_waste) {
                    best_waste = waste;
                    best_idx = i;
                }
            }
        }

        if (best_idx == SIZE_MAX) return SIZE_MAX;

        std::size_t offset = blocks_[best_idx].offset;

        // 缩小或移除该空闲块
        if (blocks_[best_idx].size == size) {
            blocks_.erase(blocks_.begin() + static_cast<std::ptrdiff_t>(best_idx));
        } else {
            blocks_[best_idx].offset += size;
            blocks_[best_idx].size -= size;
        }

        return offset;
    }

    // 回收 [offset, offset+size) 区间，并与相邻空闲块合并
    void Free(std::size_t offset, std::size_t size) {
        // 按 offset 找到插入位置（blocks_ 保持按 offset 升序）
        std::size_t insert_pos = 0;
        while (insert_pos < blocks_.size() && blocks_[insert_pos].offset < offset) {
            ++insert_pos;
        }

        blocks_.insert(
            blocks_.begin() + static_cast<std::ptrdiff_t>(insert_pos),
            FreeBlock{offset, size});

        // 尝试与右邻合并
        if (insert_pos + 1 < blocks_.size()) {
            auto& cur = blocks_[insert_pos];
            auto& right = blocks_[insert_pos + 1];
            if (cur.offset + cur.size == right.offset) {
                cur.size += right.size;
                blocks_.erase(blocks_.begin() + static_cast<std::ptrdiff_t>(insert_pos + 1));
            }
        }

        // 尝试与左邻合并
        if (insert_pos > 0) {
            auto& left = blocks_[insert_pos - 1];
            auto& cur = blocks_[insert_pos];
            if (left.offset + left.size == cur.offset) {
                left.size += cur.size;
                blocks_.erase(blocks_.begin() + static_cast<std::ptrdiff_t>(insert_pos));
            }
        }
    }

private:
    std::vector<FreeBlock> blocks_;  // 按 offset 升序
};

// ============================================================
// 事件驱动 Linear-Scan
// ============================================================

enum class EventType {
    kDeath,  // 放在 kBirth 前面，同一 op 内先回收再分配
    kBirth,
};

struct Event {
    int op_index;
    EventType type;
    std::size_t tensor_idx;  // intervals 数组的下标
};

MemoryRegion ClassifyRegion(const ir::Graph& graph, const LiveInterval& interval) {
    const ir::Tensor* tensor = graph.FindTensor(interval.tensor_id);
    // 无 producer 的 tensor（前端给定的 weight / 常量）→ FLASH
    if (tensor != nullptr && tensor->producer_op == -1) {
        return MemoryRegion::kFlash;
    }
    // persistent tensor（KV Cache 等）→ DRAM（常驻，不适合放 SRAM）
    if (tensor != nullptr && tensor->persistent) {
        return MemoryRegion::kDram;
    }
    return MemoryRegion::kSram;  // 先全部尝试 SRAM，溢出时再降级
}

std::size_t RegionIndex(MemoryRegion region) {
    switch (region) {
        case MemoryRegion::kSram:  return 0;
        case MemoryRegion::kDram:  return 1;
        case MemoryRegion::kFlash: return 2;
        default:                   return 0;
    }
}

// 在指定 region 内用 linear-scan 分配
// allocs 输出每个 tensor 的 offset
// 返回该 region 的 high watermark
std::size_t LinearScanAllocate(
    const std::vector<LiveInterval>& intervals,
    const std::vector<std::size_t>& tensor_indices,  // 属于该 region 的 tensor 下标
    const std::vector<MemoryRegion>& region_map,
    MemoryRegion target_region,
    std::vector<BufferAllocation>& allocs) {

    if (tensor_indices.empty()) return 0;

    // 构建事件列表
    std::vector<Event> events;
    events.reserve(tensor_indices.size() * 2);
    for (std::size_t idx : tensor_indices) {
        events.push_back({intervals[idx].start_op_index, EventType::kBirth, idx});
        events.push_back({intervals[idx].end_op_index, EventType::kDeath, idx});
    }

    // 排序：按 op_index 升序；同 op 内 Birth 先于 Death
    // （同一 op 中 tensor 出生和死亡时，必须先分配再回收，
    //  否则 Death 会对未分配的 offset=0 调用 Free，产生虚假空闲块）
    // 不同 tensor 的 Death 仍然在 Birth 之前处理——通过对 tensor_idx 区分
    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
        if (a.op_index != b.op_index) return a.op_index < b.op_index;
        // 同一 op 内：同一 tensor 的 Birth 必须在 Death 前
        // 不同 tensor 的 Death 先于 Birth（回收旧 tensor 给新 tensor 复用）
        if (a.tensor_idx == b.tensor_idx) return a.type > b.type;  // kBirth(1) > kDeath(0)
        return a.type < b.type;  // kDeath(0) < kBirth(1)
    });

    FreeList free_list;
    std::size_t high_watermark = 0;

    for (const auto& event : events) {
        std::size_t idx = event.tensor_idx;
        if (event.type == EventType::kBirth) {
            std::size_t aligned_size = AlignUp(intervals[idx].bytes);
            if (aligned_size == 0) aligned_size = kAlignment;
            std::size_t offset = free_list.Allocate(aligned_size);

            allocs[idx].offset = offset;
            allocs[idx].size_in_bytes = intervals[idx].bytes;

            std::size_t end = offset + aligned_size;
            if (end > high_watermark) {
                high_watermark = end;
            }
        } else {
            // Death：回收该 tensor 占用的对齐后空间
            std::size_t aligned_size = AlignUp(intervals[idx].bytes);
            if (aligned_size == 0) aligned_size = kAlignment;
            free_list.Free(allocs[idx].offset, aligned_size);
        }
    }

    return high_watermark;
}

}  // namespace

MemoryPlan MemoryPlanner::BuildPlan(
    const ir::Graph& graph,
    const std::vector<LiveInterval>& intervals) const {

    std::size_t n = intervals.size();
    std::vector<BufferAllocation> allocs(n);
    std::vector<MemoryRegion> region_map(n);

    // 初始化每个 tensor 的基本信息和 region 分类
    std::array<std::vector<std::size_t>, 3> region_indices;  // SRAM / DRAM / FLASH
    for (std::size_t i = 0; i < n; ++i) {
        allocs[i].tensor_id = intervals[i].tensor_id;
        allocs[i].tensor_name = intervals[i].tensor_name;
        region_map[i] = ClassifyRegion(graph, intervals[i]);
        allocs[i].region = region_map[i];
        region_indices[RegionIndex(region_map[i])].push_back(i);
    }

    // FLASH 区域：顺序分配，无需回收（常驻）
    {
        std::size_t offset = 0;
        for (std::size_t idx : region_indices[RegionIndex(MemoryRegion::kFlash)]) {
            std::size_t aligned = AlignUp(intervals[idx].bytes);
            allocs[idx].offset = offset;
            allocs[idx].size_in_bytes = intervals[idx].bytes;
            offset += aligned;
        }
    }

    // SRAM 区域：linear-scan 分配
    std::size_t sram_watermark = LinearScanAllocate(
        intervals, region_indices[RegionIndex(MemoryRegion::kSram)],
        region_map, MemoryRegion::kSram, allocs);

    // SRAM 溢出检测：如果超出容量，把最大的 tensor 降级到 DRAM，重试
    int spill_count = 0;
    while (sram_watermark > kSramCapacityBytes) {
        // 在 SRAM 候选中找最大的 tensor
        auto& sram_indices = region_indices[RegionIndex(MemoryRegion::kSram)];
        std::size_t largest_pos = 0;
        std::size_t largest_bytes = 0;
        for (std::size_t i = 0; i < sram_indices.size(); ++i) {
            if (intervals[sram_indices[i]].bytes > largest_bytes) {
                largest_bytes = intervals[sram_indices[i]].bytes;
                largest_pos = i;
            }
        }

        if (largest_bytes == 0) break;  // 无法再降级

        // 移动到 DRAM
        std::size_t moved_idx = sram_indices[largest_pos];
        region_map[moved_idx] = MemoryRegion::kDram;
        allocs[moved_idx].region = MemoryRegion::kDram;
        sram_indices.erase(sram_indices.begin() + static_cast<std::ptrdiff_t>(largest_pos));
        region_indices[RegionIndex(MemoryRegion::kDram)].push_back(moved_idx);
        ++spill_count;

        // 重新分配 SRAM
        sram_watermark = LinearScanAllocate(
            intervals, sram_indices, region_map, MemoryRegion::kSram, allocs);
    }

    // DRAM 区域：linear-scan 分配
    std::size_t dram_watermark = LinearScanAllocate(
        intervals, region_indices[RegionIndex(MemoryRegion::kDram)],
        region_map, MemoryRegion::kDram, allocs);

    // FLASH watermark
    std::size_t flash_watermark = 0;
    for (std::size_t idx : region_indices[RegionIndex(MemoryRegion::kFlash)]) {
        std::size_t end = allocs[idx].offset + AlignUp(intervals[idx].bytes);
        if (end > flash_watermark) flash_watermark = end;
    }

    MemoryPlan plan;
    plan.allocations = std::move(allocs);
    plan.sram_bytes = sram_watermark;
    plan.dram_bytes = dram_watermark;
    plan.flash_bytes = flash_watermark;
    plan.total_bytes = plan.sram_bytes + plan.dram_bytes + plan.flash_bytes;
    plan.peak_bytes = plan.total_bytes;
    plan.sram_spill_count = spill_count;

    return plan;
}

}  // namespace self_compiler::memory
