// MemoryPlanner Linear-Scan 单元测试

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "self_compiler/ir/graph.h"
#include "self_compiler/memory/live_interval.h"
#include "self_compiler/memory/memory_planner.h"

using namespace self_compiler;
using namespace self_compiler::ir;
using namespace self_compiler::memory;

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_BEGIN(name) \
    do { \
        tests_run++; \
        const char* test_name = (name); \
        try {

#define TEST_END() \
            tests_passed++; \
            std::cout << "  PASS: " << test_name << "\n"; \
        } catch (const std::exception& e) { \
            std::cout << "  FAIL: " << test_name << " - " << e.what() << "\n"; \
        } catch (...) { \
            std::cout << "  FAIL: " << test_name << " - unknown exception\n"; \
        } \
    } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { if (!(cond)) throw std::runtime_error(msg); } while(0)

// 构建一个最小 Graph 用于 region 分类
// 有 producer 的 tensor → SRAM，无 producer 的 → FLASH
Graph MakeGraphWithProducers(int num_tensors, const std::vector<int>& no_producer_ids) {
    Graph g;
    for (int i = 0; i < num_tensors; ++i) {
        g.AddTensor("t" + std::to_string(i), {{1, 1, 64}}, DataType::kBFloat16);
    }
    // 给有 producer 的 tensor 标记 producer_op
    for (auto& t : g.tensors()) {
        bool is_no_producer = false;
        for (int id : no_producer_ids) {
            if (t.id == id) { is_no_producer = true; break; }
        }
        if (!is_no_producer) {
            t.producer_op = 0;  // 假设 op 0 产生它
        }
    }
    return g;
}

// 测试 1: 两个不重叠的 tensor 应该能复用同一块内存
void TestNonOverlappingReuse() {
    TEST_BEGIN("non-overlapping tensors reuse memory");

    Graph g = MakeGraphWithProducers(2, {});
    // T0: 活在 op [0,1], 128 bytes
    // T1: 活在 op [2,3], 128 bytes
    // 它们不重叠，应该复用同一块 offset
    std::vector<LiveInterval> intervals = {
        {0, "t0", 0, 1, 128},
        {1, "t1", 2, 3, 128},
    };

    MemoryPlanner planner;
    auto plan = planner.BuildPlan(g, intervals);

    // 两个 tensor 应在 SRAM，且 peak 只有一个 tensor 的大小（对齐后）
    ASSERT_TRUE(plan.sram_bytes <= 128 + 16, "should reuse: sram_bytes=" + std::to_string(plan.sram_bytes));
    ASSERT_TRUE(plan.allocations[0].region == MemoryRegion::kSram, "T0 should be SRAM");
    ASSERT_TRUE(plan.allocations[1].region == MemoryRegion::kSram, "T1 should be SRAM");

    TEST_END();
}

// 测试 2: 两个重叠的 tensor 不能复用
void TestOverlappingNoReuse() {
    TEST_BEGIN("overlapping tensors cannot reuse");

    Graph g = MakeGraphWithProducers(2, {});
    // T0: 活在 op [0,3], 128 bytes
    // T1: 活在 op [1,2], 256 bytes
    // 它们重叠，需要分别分配
    std::vector<LiveInterval> intervals = {
        {0, "t0", 0, 3, 128},
        {1, "t1", 1, 2, 256},
    };

    MemoryPlanner planner;
    auto plan = planner.BuildPlan(g, intervals);

    // 两者 offset 不能重叠
    auto& a0 = plan.allocations[0];
    auto& a1 = plan.allocations[1];
    bool no_overlap = (a0.offset + 128 <= a1.offset) || (a1.offset + 256 <= a0.offset);
    ASSERT_TRUE(no_overlap, "overlapping tensors must have non-overlapping offsets");

    TEST_END();
}

// 测试 3: 16 字节对齐
void TestAlignment() {
    TEST_BEGIN("offsets are 16-byte aligned");

    Graph g = MakeGraphWithProducers(3, {});
    std::vector<LiveInterval> intervals = {
        {0, "t0", 0, 1, 100},  // 不是 16 的倍数
        {1, "t1", 0, 1, 50},
        {2, "t2", 0, 1, 33},
    };

    MemoryPlanner planner;
    auto plan = planner.BuildPlan(g, intervals);

    for (const auto& alloc : plan.allocations) {
        if (alloc.region == MemoryRegion::kSram || alloc.region == MemoryRegion::kDram) {
            ASSERT_TRUE(alloc.offset % 16 == 0,
                "offset " + std::to_string(alloc.offset) + " not 16-aligned for " + alloc.tensor_name);
        }
    }

    TEST_END();
}

// 测试 4: FLASH region — 无 producer 的 tensor 放 FLASH
void TestFlashRegion() {
    TEST_BEGIN("no-producer tensors go to FLASH");

    Graph g = MakeGraphWithProducers(3, {1});  // T1 无 producer
    std::vector<LiveInterval> intervals = {
        {0, "t0", 0, 2, 128},
        {1, "t1", 0, 5, 256},   // 无 producer → FLASH
        {2, "t2", 1, 3, 128},
    };

    MemoryPlanner planner;
    auto plan = planner.BuildPlan(g, intervals);

    ASSERT_TRUE(plan.allocations[1].region == MemoryRegion::kFlash, "T1 should be FLASH");
    ASSERT_TRUE(plan.allocations[0].region == MemoryRegion::kSram, "T0 should be SRAM");

    TEST_END();
}

// 测试 5: SRAM 溢出降级到 DRAM
void TestSramSpill() {
    TEST_BEGIN("SRAM overflow spills to DRAM");

    Graph g = MakeGraphWithProducers(2, {});
    // 两个大 tensor 同时活跃，总共超过 256KB
    std::vector<LiveInterval> intervals = {
        {0, "big0", 0, 5, 200 * 1024},  // 200KB
        {1, "big1", 0, 5, 200 * 1024},  // 200KB，同时活跃 → 需要 400KB > 256KB
    };

    MemoryPlanner planner;
    auto plan = planner.BuildPlan(g, intervals);

    ASSERT_TRUE(plan.sram_spill_count > 0, "should have spill count > 0");
    // 至少一个应该在 DRAM
    bool has_dram = false;
    for (const auto& alloc : plan.allocations) {
        if (alloc.region == MemoryRegion::kDram) has_dram = true;
    }
    ASSERT_TRUE(has_dram, "at least one tensor should be in DRAM");

    TEST_END();
}

// 测试 6: linear-scan 复用效果优于不复用
void TestLinearScanEfficiency() {
    TEST_BEGIN("linear-scan reuse reduces peak");

    Graph g = MakeGraphWithProducers(4, {});
    // 4 个 tensor，两两不重叠，应该只需要 2 个 tensor 的空间
    // T0: [0,1] 1KB, T1: [0,1] 1KB, T2: [2,3] 1KB, T3: [2,3] 1KB
    std::vector<LiveInterval> intervals = {
        {0, "t0", 0, 1, 1024},
        {1, "t1", 0, 1, 1024},
        {2, "t2", 2, 3, 1024},
        {3, "t3", 2, 3, 1024},
    };

    MemoryPlanner planner;
    auto plan = planner.BuildPlan(g, intervals);

    // peak 应该约 2KB（2 个同时活跃的 tensor），而非 4KB
    ASSERT_TRUE(plan.sram_bytes <= 3 * 1024,
        "peak should be ~2KB not 4KB, got " + std::to_string(plan.sram_bytes));

    TEST_END();
}

int main() {
    std::cout << "=== MemoryPlanner Linear-Scan 单元测试 ===\n";

    TestNonOverlappingReuse();
    TestOverlappingNoReuse();
    TestAlignment();
    TestFlashRegion();
    TestSramSpill();
    TestLinearScanEfficiency();

    std::cout << "\n结果: " << tests_passed << "/" << tests_run << " 通过\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
