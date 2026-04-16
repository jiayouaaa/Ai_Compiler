// QuantizePass 单元测试

#include <iostream>
#include <string>

#include "self_compiler/ir/graph.h"
#include "self_compiler/passes/quantize_pass.h"

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

// 测试 1: bf16 tensor 被量化为 int8
void TestBf16ToInt8() {
    TEST_BEGIN("bf16 tensor quantized to int8");

    Graph g;
    g.AddTensor("hidden", {{2, 128, 2048}}, DataType::kBFloat16);

    QuantizePass pass;
    ASSERT_TRUE(pass.Run(g).ok, "pass should succeed");

    auto& t = g.tensors()[0];
    ASSERT_TRUE(t.dtype == DataType::kInt8, "dtype should be int8");
    ASSERT_TRUE(t.scale > 0.0f, "scale should be positive");
    ASSERT_TRUE(t.zero_point == 0, "zero_point should be 0 (symmetric)");

    TEST_END();
}

// 测试 2: f32 tensor 也被量化
void TestF32ToInt8() {
    TEST_BEGIN("f32 tensor quantized to int8");

    Graph g;
    g.AddTensor("weight", {{2048, 2048}}, DataType::kFloat32);

    QuantizePass pass;
    ASSERT_TRUE(pass.Run(g).ok, "pass should succeed");

    ASSERT_TRUE(g.tensors()[0].dtype == DataType::kInt8, "dtype should be int8");

    TEST_END();
}

// 测试 3: i32 tensor 不被量化
void TestI32NotQuantized() {
    TEST_BEGIN("i32 tensor not quantized");

    Graph g;
    g.AddTensor("ids", {{2, 128}}, DataType::kInt32);

    QuantizePass pass;
    ASSERT_TRUE(pass.Run(g).ok, "pass should succeed");

    ASSERT_TRUE(g.tensors()[0].dtype == DataType::kInt32, "i32 should stay i32");
    ASSERT_TRUE(g.tensors()[0].scale == 0.0f, "scale should remain 0");

    TEST_END();
}

// 测试 4: 混合 dtype 图——只有浮点被量化
void TestMixedDtypeGraph() {
    TEST_BEGIN("mixed dtype graph");

    Graph g;
    g.AddTensor("ids", {{2, 128}}, DataType::kInt32);
    g.AddTensor("hidden", {{2, 128, 2048}}, DataType::kBFloat16);
    g.AddTensor("weight", {{2048, 2048}}, DataType::kFloat32);

    QuantizePass pass;
    ASSERT_TRUE(pass.Run(g).ok, "pass should succeed");

    ASSERT_TRUE(g.tensors()[0].dtype == DataType::kInt32, "ids should stay i32");
    ASSERT_TRUE(g.tensors()[1].dtype == DataType::kInt8, "hidden should be int8");
    ASSERT_TRUE(g.tensors()[2].dtype == DataType::kInt8, "weight should be int8");

    TEST_END();
}

int main() {
    std::cout << "=== QuantizePass 单元测试 ===\n";

    TestBf16ToInt8();
    TestF32ToInt8();
    TestI32NotQuantized();
    TestMixedDtypeGraph();

    std::cout << "\n结果: " << tests_passed << "/" << tests_run << " 通过\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
