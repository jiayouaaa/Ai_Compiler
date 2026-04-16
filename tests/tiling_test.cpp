// TilingPass 单元测试

#include <iostream>
#include <string>
#include <vector>

#include "self_compiler/ir/graph.h"
#include "self_compiler/passes/tiling_pass.h"

using namespace self_compiler;
using namespace self_compiler::ir;
using namespace self_compiler::passes;

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

// 测试 1: 小 MatMul 不需要 tiling（放得进 SRAM）
void TestSmallMatMulNoTiling() {
    TEST_BEGIN("small BatchMatMul needs no tiling");

    Graph g;
    // [2, 32, 16, 16] × [2, 32, 16, 16] → [2, 32, 16, 16]
    // 3 tensors × 2×32×16×16 × 2 bytes = 3 × 32KB = 96KB < 256KB
    int a = g.AddTensor("a", {{2, 32, 16, 16}}, DataType::kBFloat16).id;
    int b = g.AddTensor("b", {{2, 32, 16, 16}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{2, 32, 16, 16}}, DataType::kBFloat16).id;

    g.AddOperation("mm", OpKind::kBatchMatMul, {a, b}, {out});

    TilingPass pass;
    ASSERT_TRUE(pass.Run(g).ok, "pass should succeed");
    ASSERT_TRUE(g.operations().size() == 1, "no tiling should happen, got " +
        std::to_string(g.operations().size()) + " ops");

    TEST_END();
}

// 测试 2: 大 MatMul 需要 tiling
void TestLargeMatMulTiled() {
    TEST_BEGIN("large BatchMatMul gets tiled");

    Graph g;
    // [1, 1, 2048, 64] × [1, 1, 2048, 64] → [1, 1, 2048, 2048]
    // output alone = 2048*2048*2 = 8MB >> 256KB → must tile
    int a = g.AddTensor("q", {{1, 1, 2048, 64}}, DataType::kBFloat16).id;
    int b = g.AddTensor("k", {{1, 1, 2048, 64}}, DataType::kBFloat16).id;
    int out = g.AddTensor("scores", {{1, 1, 2048, 2048}}, DataType::kBFloat16).id;

    g.AddOperation("score_mm", OpKind::kBatchMatMul, {a, b}, {out});

    TilingPass pass;
    ASSERT_TRUE(pass.Run(g).ok, "pass should succeed");
    ASSERT_TRUE(g.operations().size() > 1,
        "should be tiled into multiple ops, got " + std::to_string(g.operations().size()));

    // 验证所有 tile op 都有 tile_m_offset 属性
    for (const auto& op : g.operations()) {
        auto it = op.attributes.find("tile_m_offset");
        ASSERT_TRUE(it != op.attributes.end(),
            "tile op " + op.name + " missing tile_m_offset attr");
    }

    // 验证 tile offset 覆盖完整的 M 维
    int total_m = 0;
    for (const auto& op : g.operations()) {
        total_m += std::stoi(op.attributes.at("tile_m_size"));
    }
    ASSERT_TRUE(total_m == 2048,
        "tile sizes should sum to 2048, got " + std::to_string(total_m));

    TEST_END();
}

// 测试 3: Linear 大矩阵 tiling
void TestLargeLinearTiled() {
    TEST_BEGIN("large Linear gets tiled");

    Graph g;
    // input [1, 4096, 2048] × weight [2048, 2048] → [1, 4096, 2048]
    // input alone = 4096*2048*2 = 16MB >> 256KB
    int in = g.AddTensor("in", {{1, 4096, 2048}}, DataType::kBFloat16).id;
    int w = g.AddTensor("w", {{2048, 2048}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{1, 4096, 2048}}, DataType::kBFloat16).id;

    g.AddOperation("proj", OpKind::kLinear, {in, w}, {out});

    TilingPass pass;
    ASSERT_TRUE(pass.Run(g).ok, "pass should succeed");
    ASSERT_TRUE(g.operations().size() > 1,
        "should be tiled, got " + std::to_string(g.operations().size()));

    // 验证 tile sizes sum to S=4096
    int total_s = 0;
    for (const auto& op : g.operations()) {
        total_s += std::stoi(op.attributes.at("tile_m_size"));
    }
    ASSERT_TRUE(total_s == 4096,
        "tile sizes should sum to 4096, got " + std::to_string(total_s));

    TEST_END();
}

// 测试 4: 非 MatMul/Linear op 不受影响
void TestNonMatMulUnaffected() {
    TEST_BEGIN("non-MatMul ops unaffected");

    Graph g;
    int in = g.AddTensor("in", {{1, 4096, 2048}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{1, 4096, 2048}}, DataType::kBFloat16).id;

    g.AddOperation("softmax", OpKind::kSoftmax, {in}, {out});

    TilingPass pass;
    ASSERT_TRUE(pass.Run(g).ok, "pass should succeed");
    ASSERT_TRUE(g.operations().size() == 1, "softmax should not be tiled");

    TEST_END();
}

int main() {
    std::cout << "=== TilingPass 单元测试 ===\n";

    TestSmallMatMulNoTiling();
    TestLargeMatMulTiled();
    TestLargeLinearTiled();
    TestNonMatMulUnaffected();

    std::cout << "\n结果: " << tests_passed << "/" << tests_run << " 通过\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
