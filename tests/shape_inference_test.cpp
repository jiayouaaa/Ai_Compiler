// ShapeInferencePass 单元测试
//
// 每个测试构造一个最小 Graph，运行 ShapeInferencePass，
// 验证输出 tensor 的 shape 是否符合预期。
// 不依赖 GoogleTest，使用 assert + 返回码。

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "self_compiler/ir/graph.h"
#include "self_compiler/ir/operation.h"
#include "self_compiler/ir/tensor.h"
#include "self_compiler/ir/value_shape.h"
#include "self_compiler/passes/shape_inference_pass.h"

using namespace self_compiler;
using namespace self_compiler::ir;
using namespace self_compiler::passes;

// ============================================================
// 测试辅助宏
// ============================================================

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

#define EXPECT_SHAPE(tensor, ...) \
    do { \
        std::vector<std::int64_t> expected = {__VA_ARGS__}; \
        if ((tensor).shape.dims != expected) { \
            throw std::runtime_error( \
                std::string("shape mismatch: got ") + (tensor).shape.ToString() + \
                " expected " + Shape{expected}.ToString()); \
        } \
    } while(0)

#define EXPECT_OK(status) \
    do { \
        if (!(status).ok) { \
            throw std::runtime_error("expected Ok but got error: " + (status).message); \
        } \
    } while(0)

#define EXPECT_ERROR(status) \
    do { \
        if ((status).ok) { \
            throw std::runtime_error("expected error but got Ok"); \
        } \
    } while(0)

// ============================================================
// 测试用例
// ============================================================

// 测试 1: RmsNorm — 一元 elementwise，输出 = 输入 shape
void TestRmsNormShapePreserved() {
    TEST_BEGIN("RmsNorm shape preserved");

    Graph g;
    int in = g.AddTensor("in", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int w  = g.AddTensor("w",  {{2048}},          DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{2, 128, 2048}}, DataType::kBFloat16).id;

    auto& op = g.AddOperation("rmsnorm", OpKind::kRmsNorm, {in, w}, {out});
    op.attributes["hidden_size"] = "2048";

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 128, 2048);

    TEST_END();
}

// 测试 2: RmsNorm — 校验模式下 shape 不一致应报错
void TestRmsNormShapeMismatch() {
    TEST_BEGIN("RmsNorm shape mismatch detected");

    Graph g;
    int in = g.AddTensor("in", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int w  = g.AddTensor("w",  {{2048}},          DataType::kBFloat16).id;
    // 故意设置错误的输出 shape
    int out = g.AddTensor("out", {{2, 128, 1024}}, DataType::kBFloat16).id;

    auto& op = g.AddOperation("rmsnorm", OpKind::kRmsNorm, {in, w}, {out});
    op.attributes["hidden_size"] = "2048";

    ShapeInferencePass pass;
    EXPECT_ERROR(pass.Run(g));

    TEST_END();
}

// 测试 3: RmsNorm — 推导模式（输出 shape 为空，应自动填入）
void TestRmsNormInferEmpty() {
    TEST_BEGIN("RmsNorm infer empty output shape");

    Graph g;
    int in = g.AddTensor("in", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int w  = g.AddTensor("w",  {{2048}},          DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{}}, DataType::kBFloat16).id;

    auto& op = g.AddOperation("rmsnorm", OpKind::kRmsNorm, {in, w}, {out});
    op.attributes["hidden_size"] = "2048";

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 128, 2048);

    TEST_END();
}

// 测试 4: Embedding — [B,S] + weight[V,H] → [B,S,H]
void TestEmbeddingShape() {
    TEST_BEGIN("Embedding infer [B,S,H]");

    Graph g;
    int ids = g.AddTensor("ids", {{2, 128}}, DataType::kInt32).id;
    int wt  = g.AddTensor("emb_w", {{128256, 2048}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{}}, DataType::kBFloat16).id;

    g.AddOperation("emb", OpKind::kEmbedding, {ids, wt}, {out});

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 128, 2048);

    TEST_END();
}

// 测试 5: LmHead — [B,S,H] + weight[H,V] → [B,S,V]
void TestLmHeadShape() {
    TEST_BEGIN("LmHead infer [B,S,V]");

    Graph g;
    int in = g.AddTensor("in", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int wt = g.AddTensor("lm_w", {{2048, 128256}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{}}, DataType::kBFloat16).id;

    g.AddOperation("lm_head", OpKind::kLmHead, {in, wt}, {out});

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 128, 128256);

    TEST_END();
}

// 测试 6: QkvProject — [B,S,H] → [B,S, nH*d + 2*nKV*d]
void TestQkvProjectShape() {
    TEST_BEGIN("QkvProject infer qkv_dim");

    Graph g;
    // Llama-3.2-1B: H=2048, nH=32, nKV=8, d=64
    // qkv_dim = 32*64 + 2*8*64 = 2048 + 1024 = 3072
    int in = g.AddTensor("in", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int wq = g.AddTensor("wq", {{2048, 2048}}, DataType::kBFloat16).id;
    int wk = g.AddTensor("wk", {{2048, 512}}, DataType::kBFloat16).id;
    int wv = g.AddTensor("wv", {{2048, 512}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{}}, DataType::kBFloat16).id;

    auto& op = g.AddOperation("qkv", OpKind::kQkvProject, {in, wq, wk, wv}, {out});
    op.attributes["hidden_size"] = "2048";
    op.attributes["num_attention_heads"] = "32";
    op.attributes["num_key_value_heads"] = "8";

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 128, 3072);

    TEST_END();
}

// 测试 7: SplitQkv — [B,S,qkv_dim] → 3× [B,nH,S,d]
void TestSplitQkvShape() {
    TEST_BEGIN("SplitQkv infer 3 outputs");

    Graph g;
    // qkv_dim=3072, nH=32, nKV=8, d=3072/(32+2*8)=64
    int in = g.AddTensor("in", {{2, 128, 3072}}, DataType::kBFloat16).id;
    int q = g.AddTensor("q", {{}}, DataType::kBFloat16).id;
    int k = g.AddTensor("k", {{}}, DataType::kBFloat16).id;
    int v = g.AddTensor("v", {{}}, DataType::kBFloat16).id;

    auto& op = g.AddOperation("split", OpKind::kSplitQkv, {in}, {q, k, v});
    op.attributes["num_attention_heads"] = "32";
    op.attributes["num_key_value_heads"] = "8";

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[q], 2, 32, 128, 64);
    EXPECT_SHAPE(g.tensors()[k], 2, 32, 128, 64);
    EXPECT_SHAPE(g.tensors()[v], 2, 32, 128, 64);

    TEST_END();
}

// 测试 8: BatchMatMul — 标准模式 [B,nH,S,S] × [B,nH,S,d] → [B,nH,S,d]
void TestBatchMatMulStandard() {
    TEST_BEGIN("BatchMatMul standard mode");

    Graph g;
    int a = g.AddTensor("weights", {{2, 32, 128, 128}}, DataType::kBFloat16).id;
    int b = g.AddTensor("v", {{2, 32, 128, 64}}, DataType::kBFloat16).id;
    int out = g.AddTensor("ctx", {{}}, DataType::kBFloat16).id;

    g.AddOperation("ctx_mm", OpKind::kBatchMatMul, {a, b}, {out});

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 32, 128, 64);

    TEST_END();
}

// 测试 9: BatchMatMul — 转置右操作数 Q[B,nH,S,d] × K[B,nH,S,d] → [B,nH,S,S]
void TestBatchMatMulTransposeRhs() {
    TEST_BEGIN("BatchMatMul transpose-rhs mode (Q*K^T)");

    Graph g;
    int q = g.AddTensor("q", {{2, 32, 128, 64}}, DataType::kBFloat16).id;
    int k = g.AddTensor("k", {{2, 32, 128, 64}}, DataType::kBFloat16).id;
    int out = g.AddTensor("scores", {{}}, DataType::kBFloat16).id;

    g.AddOperation("score_mm", OpKind::kBatchMatMul, {q, k}, {out});

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 32, 128, 128);

    TEST_END();
}

// 测试 10: Mul 广播 — [B,nH,S,S] × [1] → [B,nH,S,S]
void TestMulScalarBroadcast() {
    TEST_BEGIN("Mul scalar broadcast");

    Graph g;
    int a = g.AddTensor("scores", {{2, 32, 128, 128}}, DataType::kBFloat16).id;
    int s = g.AddTensor("scale", {{1}}, DataType::kBFloat16).id;
    int out = g.AddTensor("scaled", {{}}, DataType::kBFloat16).id;

    g.AddOperation("scale_mul", OpKind::kMul, {a, s}, {out});

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 32, 128, 128);

    TEST_END();
}

// 测试 11: ResidualAdd — 两个 [B,S,H] → [B,S,H]
void TestResidualAddShape() {
    TEST_BEGIN("ResidualAdd same shape");

    Graph g;
    int a = g.AddTensor("a", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int b = g.AddTensor("b", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{}}, DataType::kBFloat16).id;

    g.AddOperation("add", OpKind::kResidualAdd, {a, b}, {out});

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 128, 2048);

    TEST_END();
}

// 测试 12: ResidualAdd — shape 不一致应报错
void TestResidualAddMismatch() {
    TEST_BEGIN("ResidualAdd shape mismatch");

    Graph g;
    int a = g.AddTensor("a", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int b = g.AddTensor("b", {{2, 128, 1024}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{}}, DataType::kBFloat16).id;

    g.AddOperation("add", OpKind::kResidualAdd, {a, b}, {out});

    ShapeInferencePass pass;
    EXPECT_ERROR(pass.Run(g));

    TEST_END();
}

// 测试 13: Reshape — 校验元素总数一致
void TestReshapeElementCount() {
    TEST_BEGIN("Reshape element count matches");

    Graph g;
    // [2,32,128,64] → [2,128,2048], 元素 = 2*32*128*64 = 524288 = 2*128*2048
    int in = g.AddTensor("in", {{2, 32, 128, 64}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{2, 128, 2048}}, DataType::kBFloat16).id;

    g.AddOperation("reshape", OpKind::kReshape, {in}, {out});

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));

    TEST_END();
}

// 测试 14: Reshape — 元素总数不一致应报错
void TestReshapeElementCountMismatch() {
    TEST_BEGIN("Reshape element count mismatch");

    Graph g;
    int in = g.AddTensor("in", {{2, 32, 128, 64}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{2, 128, 1024}}, DataType::kBFloat16).id;

    g.AddOperation("reshape", OpKind::kReshape, {in}, {out});

    ShapeInferencePass pass;
    EXPECT_ERROR(pass.Run(g));

    TEST_END();
}

// 测试 15: Linear — [B,S,H_in] + weight[H_in,H_out] → [B,S,H_out]
void TestLinearShape() {
    TEST_BEGIN("Linear shape inference");

    Graph g;
    int in = g.AddTensor("in", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int wt = g.AddTensor("wo", {{2048, 2048}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{}}, DataType::kBFloat16).id;

    g.AddOperation("proj", OpKind::kLinear, {in, wt}, {out});

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 128, 2048);

    TEST_END();
}

// 测试 16: SwiGLU — [B,S,H] + gates → [B,S,H]
void TestSwiGLUShape() {
    TEST_BEGIN("SwiGLU shape inference");

    Graph g;
    // H=2048, I=8192
    int in = g.AddTensor("in", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int wg = g.AddTensor("w_gate", {{2048, 8192}}, DataType::kBFloat16).id;
    int wu = g.AddTensor("w_up",   {{2048, 8192}}, DataType::kBFloat16).id;
    int wd = g.AddTensor("w_down", {{8192, 2048}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{}}, DataType::kBFloat16).id;

    auto& op = g.AddOperation("swiglu", OpKind::kSwiGLU, {in, wg, wu, wd}, {out});
    op.attributes["hidden_size"] = "2048";
    op.attributes["intermediate_size"] = "8192";

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 128, 2048);

    TEST_END();
}

// 测试 17: TransformerBlock — 输出 = 输入 shape
void TestTransformerBlockShape() {
    TEST_BEGIN("TransformerBlock shape preserved");

    Graph g;
    int in = g.AddTensor("in", {{2, 128, 2048}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{2, 128, 2048}}, DataType::kBFloat16).id;

    auto& op = g.AddOperation("block", OpKind::kTransformerBlock, {in}, {out});
    op.attributes["hidden_size"] = "2048";
    op.attributes["intermediate_size"] = "8192";
    op.attributes["num_attention_heads"] = "32";
    op.attributes["num_key_value_heads"] = "8";

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 128, 2048);

    TEST_END();
}

// 测试 18: Softmax — 输出 = 输入 shape
void TestSoftmaxShape() {
    TEST_BEGIN("Softmax shape preserved");

    Graph g;
    int in = g.AddTensor("in", {{2, 32, 128, 128}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{}}, DataType::kBFloat16).id;

    g.AddOperation("sm", OpKind::kSoftmax, {in}, {out});

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 32, 128, 128);

    TEST_END();
}

// 测试 19: CausalMask — 输出 = 输入 shape
void TestCausalMaskShape() {
    TEST_BEGIN("CausalMask shape preserved");

    Graph g;
    int in = g.AddTensor("in", {{2, 32, 128, 128}}, DataType::kBFloat16).id;
    int out = g.AddTensor("out", {{}}, DataType::kBFloat16).id;

    g.AddOperation("mask", OpKind::kCausalMask, {in}, {out});

    ShapeInferencePass pass;
    EXPECT_OK(pass.Run(g));
    EXPECT_SHAPE(g.tensors()[out], 2, 32, 128, 128);

    TEST_END();
}

// ============================================================
// main
// ============================================================

int main() {
    std::cout << "=== ShapeInferencePass 单元测试 ===\n";

    TestRmsNormShapePreserved();
    TestRmsNormShapeMismatch();
    TestRmsNormInferEmpty();
    TestEmbeddingShape();
    TestLmHeadShape();
    TestQkvProjectShape();
    TestSplitQkvShape();
    TestBatchMatMulStandard();
    TestBatchMatMulTransposeRhs();
    TestMulScalarBroadcast();
    TestResidualAddShape();
    TestResidualAddMismatch();
    TestReshapeElementCount();
    TestReshapeElementCountMismatch();
    TestLinearShape();
    TestSwiGLUShape();
    TestTransformerBlockShape();
    TestSoftmaxShape();
    TestCausalMaskShape();

    std::cout << "\n结果: " << tests_passed << "/" << tests_run << " 通过\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
