#include "self_compiler/passes/shape_inference_pass.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "self_compiler/ir/op_registry.h"

namespace self_compiler::passes {

namespace {

// ============================================================
// 工具函数
// ============================================================

// 从 attributes 安全地取 int 值。key 缺失时返回 0 并填写 error。
int AttrInt(const ir::Operation& op, const std::string& key, std::string& error) {
    auto it = op.attributes.find(key);
    if (it == op.attributes.end()) {
        error = "ShapeInference: " + op.name + " (" + ir::ToString(op.kind) +
                "): 缺少 attribute: " + key;
        return 0;
    }
    return std::stoi(it->second);
}

// 计算 shape 的元素总数
std::int64_t ElementCount(const ir::Shape& shape) {
    if (shape.dims.empty()) return 0;
    std::int64_t n = 1;
    for (auto d : shape.dims) n *= d;
    return n;
}

// 判断 shape 是否为空（未填写）
bool IsEmpty(const ir::Shape& shape) {
    return shape.dims.empty();
}

// 判断两个 shape 是否相等
bool ShapeEqual(const ir::Shape& a, const ir::Shape& b) {
    return a.dims == b.dims;
}

// 错误信息前缀
std::string Prefix(const ir::Operation& op) {
    return "ShapeInference: " + op.name + " (" + ir::ToString(op.kind) + "): ";
}

// ============================================================
// 推导结果
// ============================================================

// 单个 op 的推导结果：
//   handled=true  → output_shapes 里有推导出的每个输出 shape
//   handled=false → 该 op 无推导规则，跳过
//   error 非空   → 推导过程发现矛盾
struct InferResult {
    bool handled = false;
    std::vector<ir::Shape> output_shapes;
    std::string error;

    static InferResult Skip() { return {false, {}, ""}; }
    static InferResult Ok(std::vector<ir::Shape> shapes) {
        return {true, std::move(shapes), ""};
    }
    static InferResult Err(const std::string& msg) {
        return {true, {}, msg};  // handled=true：已处理但发现错误
    }
};

// ============================================================
// 各类别推导规则
// ============================================================

// --- 图边界：kInput / kOutput ---
// 不推导。Input 的输出 shape 由前端给定；Output 没有输出 tensor。
// 返回 Skip。

// --- Elementwise 一元（输出 = 唯一 activation 输入） ---
// 适用于：RmsNorm, Rope, CausalMask, Softmax, Relu, Relu6, Sigmoid, Tanh
InferResult InferUnaryElementwise(
    const ir::Operation& op,
    const ir::Tensor& activation) {
    return InferResult::Ok({activation.shape});
}

// --- Elementwise 二元（两输入 shape 相同，输出 = 同 shape） ---
// 适用于：ResidualAdd
InferResult InferBinaryElementwise(
    const ir::Operation& op,
    const ir::Tensor& lhs,
    const ir::Tensor& rhs) {
    if (!ShapeEqual(lhs.shape, rhs.shape)) {
        return InferResult::Err(
            Prefix(op) + "两个输入 shape 不一致: " +
            lhs.shape.ToString() + " vs " + rhs.shape.ToString());
    }
    return InferResult::Ok({lhs.shape});
}

// --- 带广播的二元运算（支持标量 [1] 广播） ---
// 适用于：Add, Mul
InferResult InferBroadcastBinary(
    const ir::Operation& op,
    const ir::Tensor& lhs,
    const ir::Tensor& rhs) {
    // 完全相同
    if (ShapeEqual(lhs.shape, rhs.shape)) {
        return InferResult::Ok({lhs.shape});
    }
    // 一方是标量 [1]，输出取另一方
    if (lhs.shape.dims.size() == 1 && lhs.shape.dims[0] == 1) {
        return InferResult::Ok({rhs.shape});
    }
    if (rhs.shape.dims.size() == 1 && rhs.shape.dims[0] == 1) {
        return InferResult::Ok({lhs.shape});
    }
    return InferResult::Err(
        Prefix(op) + "不支持的广播组合: " +
        lhs.shape.ToString() + " 和 " + rhs.shape.ToString());
}

// --- TransformerBlock（输出 shape = 输入 shape） ---
InferResult InferTransformerBlock(
    const ir::Operation& op,
    const ir::Tensor& input) {
    return InferResult::Ok({input.shape});
}

// --- Embedding ---
// inputs[0] = token_ids [B, S]  (i32)
// inputs[1] = weight    [V, H]
// output    = [B, S, H]
InferResult InferEmbedding(
    const ir::Operation& op,
    const ir::Tensor& token_ids,
    const ir::Tensor& weight) {
    if (token_ids.shape.dims.size() != 2) {
        return InferResult::Err(
            Prefix(op) + "token_ids 应为 rank 2，实际 rank " +
            std::to_string(token_ids.shape.dims.size()));
    }
    if (weight.shape.dims.size() != 2) {
        return InferResult::Err(
            Prefix(op) + "weight 应为 rank 2，实际 rank " +
            std::to_string(weight.shape.dims.size()));
    }
    std::int64_t B = token_ids.shape.dims[0];
    std::int64_t S = token_ids.shape.dims[1];
    std::int64_t H = weight.shape.dims[1];
    return InferResult::Ok({{{B, S, H}}});
}

// --- LmHead ---
// inputs[0] = hidden [B, S, H]
// inputs[1] = weight [H, V]
// output    = [B, S, V]
InferResult InferLmHead(
    const ir::Operation& op,
    const ir::Tensor& hidden,
    const ir::Tensor& weight) {
    if (hidden.shape.dims.size() != 3) {
        return InferResult::Err(
            Prefix(op) + "hidden 应为 rank 3，实际 rank " +
            std::to_string(hidden.shape.dims.size()));
    }
    if (weight.shape.dims.size() != 2) {
        return InferResult::Err(
            Prefix(op) + "weight 应为 rank 2，实际 rank " +
            std::to_string(weight.shape.dims.size()));
    }
    std::int64_t B = hidden.shape.dims[0];
    std::int64_t S = hidden.shape.dims[1];
    std::int64_t V = weight.shape.dims[1];
    return InferResult::Ok({{{B, S, V}}});
}

// --- QkvProject ---
// activation input [B, S, H]
// output [B, S, nH*d + 2*nKV*d]
// 其中 d = H / num_attention_heads
InferResult InferQkvProject(
    const ir::Operation& op,
    const ir::Tensor& input) {
    if (input.shape.dims.size() != 3) {
        return InferResult::Err(
            Prefix(op) + "输入应为 rank 3，实际 rank " +
            std::to_string(input.shape.dims.size()));
    }
    std::string err;
    int hidden_size = AttrInt(op, "hidden_size", err);
    if (!err.empty()) return InferResult::Err(err);
    int num_heads = AttrInt(op, "num_attention_heads", err);
    if (!err.empty()) return InferResult::Err(err);
    int num_kv_heads = AttrInt(op, "num_key_value_heads", err);
    if (!err.empty()) return InferResult::Err(err);
    if (num_heads <= 0) return InferResult::Err(Prefix(op) + "num_attention_heads 必须 > 0");
    int head_dim = hidden_size / num_heads;

    std::int64_t B = input.shape.dims[0];
    std::int64_t S = input.shape.dims[1];
    std::int64_t qkv_dim = static_cast<std::int64_t>(
        num_heads * head_dim + 2 * num_kv_heads * head_dim);
    return InferResult::Ok({{{B, S, qkv_dim}}});
}

// --- SplitQkv ---
// input [B, S, qkv_dim]，其中 qkv_dim = nH*d + 2*nKV*d
// 3 个输出全部是 [B, nH, S, d]（K/V 在 GQA 扩展后与 Q 头数一致）
InferResult InferSplitQkv(
    const ir::Operation& op,
    const ir::Tensor& input) {
    if (input.shape.dims.size() != 3) {
        return InferResult::Err(
            Prefix(op) + "输入应为 rank 3，实际 rank " +
            std::to_string(input.shape.dims.size()));
    }
    std::string err;
    int num_heads = AttrInt(op, "num_attention_heads", err);
    if (!err.empty()) return InferResult::Err(err);
    int num_kv_heads = AttrInt(op, "num_key_value_heads", err);
    if (!err.empty()) return InferResult::Err(err);

    std::int64_t B = input.shape.dims[0];
    std::int64_t S = input.shape.dims[1];
    std::int64_t qkv_dim = input.shape.dims[2];

    // d = qkv_dim / (nH + 2*nKV)
    int divisor = num_heads + 2 * num_kv_heads;
    if (divisor <= 0) {
        return InferResult::Err(
            Prefix(op) + "(num_heads + 2*num_kv_heads) 必须 > 0, 实际 = " +
            std::to_string(divisor));
    }
    if (qkv_dim % divisor != 0) {
        return InferResult::Err(
            Prefix(op) + "qkv_dim " + std::to_string(qkv_dim) +
            " 无法被 (num_heads + 2*num_kv_heads) = " +
            std::to_string(divisor) + " 整除");
    }
    std::int64_t d = qkv_dim / divisor;
    std::int64_t nH = static_cast<std::int64_t>(num_heads);

    // 3 个输出: Q/K/V 全部 [B, nH, S, d]（GQA 扩展后）
    ir::Shape head_shape = {{B, nH, S, d}};
    return InferResult::Ok({head_shape, head_shape, head_shape});
}

// --- BatchMatMul ---
// 两种模式（自动判断）：
//   标准：A[..., M, K] × B[..., K, N] → [..., M, N]
//   转置右操作数：A[..., M, K] × B[..., N, K] → [..., M, N]
//
// 在 Llama attention 中：
//   score_matmul: Q[B,nH,S,d] × K[B,nH,S,d] → [B,nH,S,S]  （转置右操作数）
//   context_matmul: weights[B,nH,S,S] × V[B,nH,S,d] → [B,nH,S,d]  （标准）
InferResult InferBatchMatMul(
    const ir::Operation& op,
    const ir::Tensor& lhs,
    const ir::Tensor& rhs) {
    if (lhs.shape.dims.size() < 2 || rhs.shape.dims.size() < 2) {
        return InferResult::Err(
            Prefix(op) + "BatchMatMul 输入 rank 必须 >= 2, 实际: " +
            lhs.shape.ToString() + " 和 " + rhs.shape.ToString());
    }
    if (lhs.shape.dims.size() != rhs.shape.dims.size()) {
        return InferResult::Err(
            Prefix(op) + "两个输入 rank 不一致: " +
            lhs.shape.ToString() + " 和 " + rhs.shape.ToString());
    }

    std::size_t rank = lhs.shape.dims.size();

    // 检查 batch 维度是否一致
    for (std::size_t i = 0; i + 2 < rank; ++i) {
        if (lhs.shape.dims[i] != rhs.shape.dims[i]) {
            return InferResult::Err(
                Prefix(op) + "batch 维度不一致 axis " + std::to_string(i) +
                ": " + std::to_string(lhs.shape.dims[i]) +
                " vs " + std::to_string(rhs.shape.dims[i]));
        }
    }

    std::int64_t M = lhs.shape.dims[rank - 2];
    std::int64_t K_lhs = lhs.shape.dims[rank - 1];
    std::int64_t K_rhs = rhs.shape.dims[rank - 2];
    std::int64_t N_rhs = rhs.shape.dims[rank - 1];

    ir::Shape out_shape;
    // 复制 batch 维度
    for (std::size_t i = 0; i + 2 < rank; ++i) {
        out_shape.dims.push_back(lhs.shape.dims[i]);
    }

    if (K_lhs == K_rhs) {
        // 标准模式：A[...,M,K] × B[...,K,N] → [...,M,N]
        out_shape.dims.push_back(M);
        out_shape.dims.push_back(N_rhs);
    } else if (K_lhs == N_rhs) {
        // 转置右操作数模式：A[...,M,K] × B[...,N,K] → [...,M,N]
        // 即 A × Bᵀ
        out_shape.dims.push_back(M);
        out_shape.dims.push_back(K_rhs);
    } else {
        return InferResult::Err(
            Prefix(op) + "矩阵乘法维度不匹配: " +
            lhs.shape.ToString() + " × " + rhs.shape.ToString() +
            " (既不满足标准 K=" + std::to_string(K_lhs) +
            " vs " + std::to_string(K_rhs) +
            "，也不满足转置右操作数 K=" + std::to_string(K_lhs) +
            " vs N=" + std::to_string(N_rhs) + ")");
    }

    return InferResult::Ok({out_shape});
}

// --- Reshape ---
// 如果输出 shape 已存在，校验元素总数一致；如果为空，无法推导（跳过）。
// Reshape 的目标 shape 在当前 IR 里由 lowering 直接写入输出 tensor，
// 而非通过 attribute 指定。
InferResult InferReshape(
    const ir::Operation& op,
    const ir::Tensor& input,
    const ir::Tensor& output) {
    if (IsEmpty(output.shape)) {
        // 无法推导——目标 shape 信息不在 attributes 里
        return InferResult::Skip();
    }
    // 校验元素总数一致
    std::int64_t in_count = ElementCount(input.shape);
    std::int64_t out_count = ElementCount(output.shape);
    if (in_count != out_count) {
        return InferResult::Err(
            Prefix(op) + "元素总数不一致: 输入 " +
            input.shape.ToString() + " (" + std::to_string(in_count) +
            " 个) vs 输出 " + output.shape.ToString() +
            " (" + std::to_string(out_count) + " 个)");
    }
    // 已通过校验，返回现有 shape（不修改）
    return InferResult::Ok({output.shape});
}

// --- Linear ---
// activation inputs[0] [B, S, H_in]
// weight     inputs[1] [H_in, H_out]
// output     [B, S, H_out]
InferResult InferLinear(
    const ir::Operation& op,
    const ir::Tensor& input,
    const ir::Tensor& weight) {
    if (input.shape.dims.size() < 2 || weight.shape.dims.size() != 2) {
        return InferResult::Err(
            Prefix(op) + "Linear 输入 rank >= 2, weight rank == 2, 实际: " +
            input.shape.ToString() + " 和 " + weight.shape.ToString());
    }
    ir::Shape out;
    // 复制 batch 维度（除最后一维）
    for (std::size_t i = 0; i + 1 < input.shape.dims.size(); ++i) {
        out.dims.push_back(input.shape.dims[i]);
    }
    // 最后一维 = weight 的输出维度
    out.dims.push_back(weight.shape.dims[1]);
    return InferResult::Ok({out});
}

// --- SwiGLU ---
// activation inputs[0] [B, S, H]
// w_gate     inputs[1] [H, I]
// w_up       inputs[2] [H, I]
// w_down     inputs[3] [I, H]
// output     [B, S, H]  （= 输入 batch 维 + w_down 的最后一维）
InferResult InferSwiGLU(
    const ir::Operation& op,
    const std::vector<const ir::Tensor*>& inputs) {
    const auto& activation = *inputs[0];
    if (activation.shape.dims.size() < 2) {
        return InferResult::Err(
            Prefix(op) + "activation rank 应 >= 2, 实际 rank " +
            std::to_string(activation.shape.dims.size()));
    }
    // w_down 是第 4 个 input（index 3）
    if (inputs.size() < 4) {
        return InferResult::Err(
            Prefix(op) + "SwiGLU 需要至少 4 个输入 (activation + 3 weights), 实际 " +
            std::to_string(inputs.size()));
    }
    const auto& w_down = *inputs[3];
    if (w_down.shape.dims.size() != 2) {
        return InferResult::Err(
            Prefix(op) + "w_down 应为 rank 2, 实际 rank " +
            std::to_string(w_down.shape.dims.size()));
    }

    ir::Shape out;
    for (std::size_t i = 0; i + 1 < activation.shape.dims.size(); ++i) {
        out.dims.push_back(activation.shape.dims[i]);
    }
    out.dims.push_back(w_down.shape.dims[1]);
    return InferResult::Ok({out});
}

// ============================================================
// 总分发：根据 op.kind 调用对应推导函数
// ============================================================

InferResult InferOp(
    const ir::Operation& op,
    const std::vector<const ir::Tensor*>& inputs,
    const std::vector<const ir::Tensor*>& outputs) {

    using K = ir::OpKind;

    switch (op.kind) {
        // 图边界：跳过
        case K::kInput:
        case K::kOutput:
            return InferResult::Skip();

        // 高层算子
        case K::kTransformerBlock:
            return InferTransformerBlock(op, *inputs[0]);
        case K::kEmbedding:
            return InferEmbedding(op, *inputs[0], *inputs[1]);
        case K::kLmHead:
            return InferLmHead(op, *inputs[0], *inputs[1]);

        // Elementwise 一元
        case K::kRmsNorm:
        case K::kRope:
        case K::kCausalMask:
        case K::kSoftmax:
        case K::kRelu:
        case K::kRelu6:
        case K::kSigmoid:
        case K::kTanh:
            return InferUnaryElementwise(op, *inputs[0]);

        // Elementwise 二元（严格同 shape）
        case K::kResidualAdd:
            return InferBinaryElementwise(op, *inputs[0], *inputs[1]);

        // 广播二元
        case K::kAdd:
        case K::kMul:
            return InferBroadcastBinary(op, *inputs[0], *inputs[1]);

        // Transformer 子算子
        case K::kQkvProject:
            return InferQkvProject(op, *inputs[0]);
        case K::kSplitQkv:
            return InferSplitQkv(op, *inputs[0]);
        case K::kBatchMatMul:
            return InferBatchMatMul(op, *inputs[0], *inputs[1]);

        // Reshape：特殊处理——需要查看现有输出 shape
        case K::kReshape:
            return InferReshape(op, *inputs[0], *outputs[0]);

        // Linear
        case K::kLinear:
            return InferLinear(op, *inputs[0], *inputs[1]);

        // SwiGLU：需要 4 个输入
        case K::kSwiGLU:
            return InferSwiGLU(op, inputs);

        // FullyConnected：与 Linear 类似
        case K::kFullyConnected:
            if (inputs.size() >= 2) {
                return InferLinear(op, *inputs[0], *inputs[1]);
            }
            return InferResult::Skip();

        // KV Cache 算子：输出 shape 由外部指定，这里只做校验
        case K::kKVCacheWrite:
        case K::kKVCacheRead:
            return InferResult::Skip();

        // 未覆盖的 op：Conv2D / Pool / Pad / Concat / Split / Transpose 等
        // 当前 Llama 链路不涉及这些，直接跳过
        default:
            return InferResult::Skip();
    }
}

}  // namespace

// ============================================================
// Pass 接口实现
// ============================================================

std::string ShapeInferencePass::name() const {
    return "ShapeInferencePass";
}

self_compiler::Status ShapeInferencePass::Run(ir::Graph& graph) {
    for (const auto& op : graph.operations()) {
        // 收集输入 tensor 指针
        std::vector<const ir::Tensor*> inputs;
        inputs.reserve(op.inputs.size());
        for (int tid : op.inputs) {
            const ir::Tensor* t = graph.FindTensor(tid);
            if (!t) {
                return Status::Error(
                    Prefix(op) + "找不到输入 tensor T" + std::to_string(tid));
            }
            inputs.push_back(t);
        }

        // 收集输出 tensor 指针
        std::vector<const ir::Tensor*> outputs;
        outputs.reserve(op.outputs.size());
        for (int tid : op.outputs) {
            const ir::Tensor* t = graph.FindTensor(tid);
            if (!t) {
                return Status::Error(
                    Prefix(op) + "找不到输出 tensor T" + std::to_string(tid));
            }
            outputs.push_back(t);
        }

        // 推导
        auto result = InferOp(op, inputs, outputs);

        // 推导过程中发现错误
        if (!result.error.empty()) {
            return Status::Error(result.error);
        }

        // 该 op 无推导规则，跳过
        if (!result.handled) {
            continue;
        }

        // 输出数量校验
        if (result.output_shapes.size() != op.outputs.size()) {
            return Status::Error(
                Prefix(op) + "推导出 " +
                std::to_string(result.output_shapes.size()) +
                " 个输出 shape，但 op 有 " +
                std::to_string(op.outputs.size()) + " 个输出");
        }

        // 对每个输出：推导模式（填入）或校验模式（比对）
        for (std::size_t i = 0; i < op.outputs.size(); ++i) {
            ir::Tensor* out_tensor = graph.FindTensor(op.outputs[i]);
            const ir::Shape& inferred = result.output_shapes[i];

            if (IsEmpty(out_tensor->shape)) {
                // 推导模式：填入推导出的 shape
                out_tensor->shape = inferred;
            } else {
                // 校验模式：比对
                if (!ShapeEqual(out_tensor->shape, inferred)) {
                    return Status::Error(
                        Prefix(op) + "输出 T" +
                        std::to_string(out_tensor->id) + " (" +
                        out_tensor->name + ") 现有 shape " +
                        out_tensor->shape.ToString() +
                        " 与推导结果 " + inferred.ToString() + " 不一致");
                }
            }
        }
    }

    return Status::Ok();
}

}  // namespace self_compiler::passes
