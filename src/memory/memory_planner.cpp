#include "self_compiler/memory/memory_planner.h"

namespace self_compiler::memory {

MemoryPlan MemoryPlanner::BuildPlan(const std::vector<LiveInterval>& intervals) const {
    MemoryPlan plan;
    std::size_t next_offset = 0;

    for (const auto& interval : intervals) {
        BufferAllocation allocation;
        allocation.tensor_id = interval.tensor_id;
        allocation.tensor_name = interval.tensor_name;
        allocation.offset = next_offset;
        allocation.size_in_bytes = interval.bytes;
        next_offset += interval.bytes;

        plan.allocations.push_back(allocation);
        plan.total_bytes += interval.bytes;
    }

    plan.peak_bytes = plan.total_bytes;

    // 伪代码：后续请用真实静态内存规划逻辑替换这里
    // 按开始时间对 live interval 排序
    // 维护按结束时间排序的活动区间集合
    // 对不重叠的生命周期尝试复用已经释放的 buffer
    // 可进一步扩展为 SRAM / DRAM / Flash 多内存区域规划
    // 可进一步按延迟与峰值内存对分配方案评分

    return plan;
}

}  // namespace self_compiler::memory
