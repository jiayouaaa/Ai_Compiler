#include "self_compiler/passes/tiling_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace self_compiler::passes {

namespace {

constexpr std::size_t kSramCapacity = 256 * 1024;  // 256KB

// 根据 dtype 估算单个元素字节数
std::size_t ElementSize(ir::DataType dtype) {
    switch (dtype) {
        case ir::DataType::kFloat32:
        case ir::DataType::kInt32:
            return 4;
        case ir::DataType::kInt8:
            return 1;
        default:
            return 2;  // bf16
    }
}

// 判断一个 op 是否需要 tiling
bool NeedsTiling(const ir::Operation& op) {
    return op.kind == ir::OpKind::kBatchMatMul ||
           op.kind == ir::OpKind::kLinear;
}

// 对 kLinear 做 tiling：
// input [B,S,H_in] × weight [H_in,H_out] → output [B,S,H_out]
// 沿 S 维切块（因为 S 是独立的序列维度）
//
// 对 kBatchMatMul 做 tiling：
// A [...,M,K] × B [...,K,N] → C [...,M,N]
// 沿 M 维切块
struct TileDecision {
    bool should_tile = false;
    std::int64_t tile_m = 0;  // M 维 tile 大小（0 = 不 tile）
    std::int64_t original_m = 0;
};

TileDecision DecideTiling(
    const ir::Operation& op,
    const ir::Graph& graph) {

    TileDecision decision;

    if (op.outputs.empty() || op.inputs.empty()) return decision;

    const ir::Tensor* out = graph.FindTensor(op.outputs[0]);
    const ir::Tensor* in0 = graph.FindTensor(op.inputs[0]);
    if (!out || !in0) return decision;

    std::size_t elem_size = ElementSize(out->dtype);

    if (op.kind == ir::OpKind::kLinear && in0->shape.dims.size() == 3 && op.inputs.size() >= 2) {
        // input [B,S,H_in], weight [H_in,H_out], output [B,S,H_out]
        const ir::Tensor* weight = graph.FindTensor(op.inputs[1]);
        if (!weight) return decision;

        std::int64_t B = in0->shape.dims[0];
        std::int64_t S = in0->shape.dims[1];
        std::int64_t H_in = in0->shape.dims[2];
        std::int64_t H_out = weight->shape.dims.size() >= 2 ? weight->shape.dims[1] : 0;

        // 维度校验：所有维度必须为正数
        if (B <= 0 || S <= 0 || H_in <= 0 || H_out <= 0) return decision;

        // 一个 tile 需要的字节 = input_tile + weight + output_tile
        // input_tile: [B, tile_s, H_in], weight: [H_in, H_out], output_tile: [B, tile_s, H_out]
        // weight 是共享的，不随 tile 变化
        std::size_t weight_bytes = static_cast<std::size_t>(H_in * H_out) * elem_size;
        auto tile_bytes = [&](std::int64_t tile_s) -> std::size_t {
            return static_cast<std::size_t>(B * tile_s * H_in) * elem_size
                 + weight_bytes
                 + static_cast<std::size_t>(B * tile_s * H_out) * elem_size;
        };

        // 如果全量放得下，不需要 tile
        if (tile_bytes(S) <= kSramCapacity) return decision;

        // 从 S 开始二分搜索最大的 tile_s 使其放入 SRAM
        std::int64_t tile_s = S;
        while (tile_s > 1 && tile_bytes(tile_s) > kSramCapacity) {
            tile_s /= 2;
        }
        if (tile_s < 1) tile_s = 1;

        decision.should_tile = true;
        decision.tile_m = tile_s;
        decision.original_m = S;

    } else if (op.kind == ir::OpKind::kBatchMatMul &&
               in0->shape.dims.size() >= 2 && op.inputs.size() >= 2) {
        const ir::Tensor* in1 = graph.FindTensor(op.inputs[1]);
        if (!in1) return decision;

        std::size_t rank = in0->shape.dims.size();
        std::int64_t M = in0->shape.dims[rank - 2];
        std::int64_t K = in0->shape.dims[rank - 1];

        // 估算 batch 维度大小
        std::int64_t batch = 1;
        for (std::size_t i = 0; i + 2 < rank; ++i) {
            batch *= in0->shape.dims[i];
        }

        // 维度校验
        if (M <= 0 || K <= 0 || batch <= 0) return decision;

        // 判断 standard vs transpose-rhs
        std::int64_t N;
        if (K == in1->shape.dims[rank - 2]) {
            N = in1->shape.dims[rank - 1];  // standard
        } else {
            N = in1->shape.dims[rank - 2];  // transpose-rhs
        }

        // tile 沿 M 维: input_tile [batch, tile_m, K] + rhs [batch, K, N] + out_tile [batch, tile_m, N]
        auto tile_bytes = [&](std::int64_t tile_m) -> std::size_t {
            return static_cast<std::size_t>(batch * tile_m * K) * elem_size
                 + static_cast<std::size_t>(batch * K * N) * elem_size
                 + static_cast<std::size_t>(batch * tile_m * N) * elem_size;
        };

        if (tile_bytes(M) <= kSramCapacity) return decision;

        std::int64_t tile_m = M;
        while (tile_m > 1 && tile_bytes(tile_m) > kSramCapacity) {
            tile_m /= 2;
        }
        if (tile_m < 1) tile_m = 1;

        decision.should_tile = true;
        decision.tile_m = tile_m;
        decision.original_m = M;
    }

    return decision;
}

// 重建所有 tensor 的 producer/consumer 引用
void RebuildRefs(ir::Graph& graph) {
    for (auto& tensor : graph.tensors()) {
        tensor.producer_op = -1;
        tensor.consumer_ops.clear();
    }
    for (const auto& op : graph.operations()) {
        for (int tid : op.outputs) {
            ir::Tensor* t = graph.FindTensor(tid);
            if (t) t->producer_op = op.id;
        }
        for (int tid : op.inputs) {
            ir::Tensor* t = graph.FindTensor(tid);
            if (t) t->consumer_ops.push_back(op.id);
        }
    }
}

}  // namespace

std::string TilingPass::name() const {
    return "TilingPass";
}

self_compiler::Status TilingPass::Run(ir::Graph& graph) {
    std::vector<ir::Operation> new_ops;
    new_ops.reserve(graph.operations().size());

    for (const auto& op : graph.operations()) {
        if (!NeedsTiling(op)) {
            new_ops.push_back(op);
            continue;
        }

        auto decision = DecideTiling(op, graph);
        if (!decision.should_tile) {
            new_ops.push_back(op);
            continue;
        }

        // 把一个大 op 拆成多个 tile op
        std::int64_t M = decision.original_m;
        std::int64_t tile_m = decision.tile_m;

        // 在循环前拷贝原始输出 tensor 的信息，避免循环内 FindTensor
        // 返回的裸指针在 AddTensor 触发 vector 扩容后失效
        const ir::Tensor* orig_out_ptr = graph.FindTensor(op.outputs[0]);
        const ir::Shape orig_out_shape = orig_out_ptr->shape;
        const std::string orig_out_name = orig_out_ptr->name;
        const ir::DataType orig_out_dtype = orig_out_ptr->dtype;

        for (std::int64_t m_off = 0; m_off < M; m_off += tile_m) {
            std::int64_t actual_tile = std::min(tile_m, M - m_off);

            // 创建 tile 输出 tensor（使用循环前拷贝的值，不再持有裸指针）
            ir::Shape tile_shape = orig_out_shape;

            // 找到 M 维的位置并修改
            if (op.kind == ir::OpKind::kLinear) {
                // [B, S, H_out] → tile: [B, tile_s, H_out]
                tile_shape.dims[1] = actual_tile;
            } else {
                // [..., M, N] → tile: [..., tile_m, N]
                tile_shape.dims[tile_shape.dims.size() - 2] = actual_tile;
            }

            int tile_out_id = graph.AddTensor(
                orig_out_name + ".tile_m" + std::to_string(m_off),
                tile_shape, orig_out_dtype).id;

            ir::Operation tile_op = op;  // 复制原始 op 的所有属性
            tile_op.name = op.name + ".tile_m" + std::to_string(m_off);
            tile_op.outputs = {tile_out_id};
            tile_op.attributes["tile_m_offset"] = std::to_string(m_off);
            tile_op.attributes["tile_m_size"] = std::to_string(actual_tile);
            tile_op.attributes["original_m"] = std::to_string(M);

            new_ops.push_back(tile_op);
        }
    }

    // 重新编号
    for (std::size_t i = 0; i < new_ops.size(); ++i) {
        new_ops[i].id = static_cast<int>(i);
    }
    graph.operations() = new_ops;
    RebuildRefs(graph);

    return Status::Ok();
}

}  // namespace self_compiler::passes
