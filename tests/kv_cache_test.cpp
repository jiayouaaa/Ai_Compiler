// KV Cache 建模单元测试

#include <iostream>
#include <string>
#include <vector>

#include "self_compiler/ir/graph.h"
#include "self_compiler/ir/op_registry.h"
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

// 测试 1: KVCacheWrite 和 KVCacheRead op 已注册
void TestKVCacheOpsRegistered() {
    TEST_BEGIN("KVCacheWrite/Read ops registered");

    const auto* write_info = OpRegistry::Instance().Find(OpKind::kKVCacheWrite);
    const auto* read_info = OpRegistry::Instance().Find(OpKind::kKVCacheRead);

    ASSERT_TRUE(write_info != nullptr, "KVCacheWrite should be registered");
    ASSERT_TRUE(read_info != nullptr, "KVCacheRead should be registered");
    ASSERT_TRUE(write_info->num_activation_inputs == 2, "KVCacheWrite should have 2 inputs");
    ASSERT_TRUE(write_info->num_outputs == 1, "KVCacheWrite should have 1 output");
    ASSERT_TRUE(read_info->num_activation_inputs == 1, "KVCacheRead should have 1 input");
    ASSERT_TRUE(read_info->num_outputs == 1, "KVCacheRead should have 1 output");

    TEST_END();
}

// 测试 2: persistent tensor 被分配到 DRAM
void TestPersistentTensorInDram() {
    TEST_BEGIN("persistent tensor allocated in DRAM");

    Graph g;
    // K cache: persistent, 预留 max_seq_len=2048
    auto& k_cache = g.AddTensor("k_cache", {{1, 32, 2048, 64}}, DataType::kBFloat16);
    k_cache.persistent = true;
    k_cache.max_seq_len = 2048;
    k_cache.producer_op = 0;  // 有 producer，不是 FLASH

    // 普通 tensor
    auto& temp = g.AddTensor("temp", {{1, 32, 128, 64}}, DataType::kBFloat16);
    temp.producer_op = 0;

    std::vector<LiveInterval> intervals = {
        {0, "k_cache", 0, 10, 1 * 32 * 2048 * 64 * 2},  // ~8MB
        {1, "temp", 0, 2, 1 * 32 * 128 * 64 * 2},       // ~512KB
    };

    MemoryPlanner planner;
    auto plan = planner.BuildPlan(g, intervals);

    ASSERT_TRUE(plan.allocations[0].region == MemoryRegion::kDram,
        "k_cache should be in DRAM (persistent)");

    TEST_END();
}

// 测试 3: KV Cache 图结构——可以构建 KVCacheWrite + KVCacheRead 子图
void TestKVCacheSubgraph() {
    TEST_BEGIN("KV cache subgraph construction");

    Graph g;

    // K_new: 当前步产出的新 K [B, nH, 1, d]
    int k_new = g.AddTensor("k_new", {{1, 32, 1, 64}}, DataType::kBFloat16).id;

    // K_cache: persistent, [B, nH, max_seq, d]
    auto& k_cache_tensor = g.AddTensor("k_cache", {{1, 32, 2048, 64}}, DataType::kBFloat16);
    k_cache_tensor.persistent = true;
    k_cache_tensor.max_seq_len = 2048;
    int k_cache = k_cache_tensor.id;

    // KVCacheWrite: 把 k_new 写入 k_cache 的 pos 位置
    int k_updated = g.AddTensor("k_updated", {{1, 32, 2048, 64}}, DataType::kBFloat16).id;
    auto& write_op = g.AddOperation("kv_write", OpKind::kKVCacheWrite,
        {k_new, k_cache}, {k_updated});
    write_op.attributes["pos"] = "0";

    // KVCacheRead: 从 k_cache 读取 [0:pos+1]
    int k_slice = g.AddTensor("k_slice", {{1, 32, 1, 64}}, DataType::kBFloat16).id;
    auto& read_op = g.AddOperation("kv_read", OpKind::kKVCacheRead,
        {k_updated}, {k_slice});
    read_op.attributes["pos"] = "0";

    // 验证图结构
    ASSERT_TRUE(g.tensors().size() == 4, "should have 4 tensors");
    ASSERT_TRUE(g.operations().size() == 2, "should have 2 ops");
    ASSERT_TRUE(g.tensors()[k_cache].persistent, "k_cache should be persistent");

    // 验证连接关系
    ASSERT_TRUE(g.tensors()[k_updated].producer_op == 0, "k_updated produced by write_op");
    ASSERT_TRUE(g.tensors()[k_slice].producer_op == 1, "k_slice produced by read_op");

    TEST_END();
}

// 测试 4: persistent tensor 的 max_seq_len 字段
void TestMaxSeqLenField() {
    TEST_BEGIN("persistent tensor max_seq_len field");

    Graph g;
    auto& cache = g.AddTensor("cache", {{1, 32, 4096, 64}}, DataType::kBFloat16);
    cache.persistent = true;
    cache.max_seq_len = 4096;

    ASSERT_TRUE(cache.persistent == true, "should be persistent");
    ASSERT_TRUE(cache.max_seq_len == 4096, "max_seq_len should be 4096");

    // 非 persistent tensor 的默认值
    auto& normal = g.AddTensor("normal", {{1, 32, 128, 64}}, DataType::kBFloat16);
    ASSERT_TRUE(normal.persistent == false, "normal tensor should not be persistent");
    ASSERT_TRUE(normal.max_seq_len == 0, "normal tensor max_seq_len should be 0");

    TEST_END();
}

int main() {
    std::cout << "=== KV Cache 建模单元测试 ===\n";

    TestKVCacheOpsRegistered();
    TestPersistentTensorInDram();
    TestKVCacheSubgraph();
    TestMaxSeqLenField();

    std::cout << "\n结果: " << tests_passed << "/" << tests_run << " 通过\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
