#include "self_compiler/passes/lower_transformer_to_runtime_pass.h"

#include "self_compiler/ir/op_registry.h"

namespace self_compiler::passes {

namespace {

// 按需提取子 op 所需的 attributes
std::map<std::string, std::string> PickAttrs(
    const std::map<std::string, std::string>& src,
    const std::vector<std::string>& keys) {
    std::map<std::string, std::string> dst;
    for (const auto& key : keys) {
        auto it = src.find(key);
        if (it != src.end()) {
            dst[key] = it->second;
        }
    }
    return dst;
}

// 重建整个 graph 的 tensor↔op 引用关系
void RebuildTensorRefs(ir::Graph& graph) {
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

// 根据注册表的 weight 描述，为一个 op 创建 weight tensor 并加入 inputs
// weight_shapes 按顺序对应注册表中 weight_inputs 的每个 WeightDesc
void AttachWeights(
    ir::Graph& graph,
    ir::Operation& op,
    const std::string& prefix,
    const std::vector<ir::Shape>& weight_shapes,
    ir::DataType dtype) {

    const auto* info = ir::OpRegistry::Instance().Find(op.kind);
    if (!info) return;

    for (size_t i = 0; i < info->weight_inputs.size() && i < weight_shapes.size(); ++i) {
        const auto& wd = info->weight_inputs[i];
        int wt = graph.AddTensor(
            prefix + "." + wd.name_suffix,
            weight_shapes[i], dtype).id;
        op.inputs.push_back(wt);
    }
}

}  // namespace

std::string LowerTransformerToRuntimePass::name() const {
    return "LowerTransformerToRuntimePass";
}

self_compiler::Status LowerTransformerToRuntimePass::Run(ir::Graph& graph) {
    std::vector<ir::Operation> lowered;
    lowered.reserve(graph.operations().size() + 16);

    for (const auto& op : graph.operations()) {
        if (op.kind != ir::OpKind::kTransformerBlock) {
            lowered.push_back(op);
            continue;
        }

        // CanonicalizePass 已保证：恰好 1 输入 1 输出、属性存在且为正整数、整除关系
        const int block_input = op.inputs[0];
        const int block_output = op.outputs[0];
        const std::string& prefix = op.name;

        // 从 attributes 读取维度参数（canonicalize 已验证存在性和合法性）
        const int hidden_size = std::stoi(op.attributes.at("hidden_size"));
        const int intermediate_size = std::stoi(op.attributes.at("intermediate_size"));
        const int num_heads = std::stoi(op.attributes.at("num_attention_heads"));
        const int num_kv_heads = std::stoi(op.attributes.at("num_key_value_heads"));

        // 从 block 输入 tensor 取 batch、seq_len、dtype
        const ir::Tensor* input_tensor = graph.FindTensor(block_input);
        const ir::DataType dtype = input_tensor->dtype;
        const std::int64_t B = input_tensor->shape.dims[0];
        const std::int64_t S = input_tensor->shape.dims[1];

        // 各阶段的真实 shape
        const ir::Shape hidden_shape = {{B, S, static_cast<std::int64_t>(hidden_size)}};
        const int head_dim = hidden_size / num_heads;
        const std::int64_t qkv_dim = static_cast<std::int64_t>(
            num_heads * head_dim + 2 * num_kv_heads * head_dim);
        const ir::Shape qkv_shape = {{B, S, qkv_dim}};

        // 维度常量，用于计算 weight 的 shape
        const std::int64_t H = static_cast<std::int64_t>(hidden_size);
        const std::int64_t H_kv = static_cast<std::int64_t>(num_kv_heads * head_dim);
        const std::int64_t I = static_cast<std::int64_t>(intermediate_size);

        // --- 新建 7 个中间 tensor，按真实 shape ---
        const int rmsnorm0_out = graph.AddTensor(
            prefix + ".rmsnorm0_out", hidden_shape, dtype).id;
        const int qkv_out = graph.AddTensor(
            prefix + ".qkv_out", qkv_shape, dtype).id;
        const int rope_out = graph.AddTensor(
            prefix + ".rope_out", qkv_shape, dtype).id;
        const int attention_out = graph.AddTensor(
            prefix + ".attention_out", hidden_shape, dtype).id;
        const int residual0_out = graph.AddTensor(
            prefix + ".residual0_out", hidden_shape, dtype).id;
        const int rmsnorm1_out = graph.AddTensor(
            prefix + ".rmsnorm1_out", hidden_shape, dtype).id;
        const int swiglu_out = graph.AddTensor(
            prefix + ".swiglu_out", hidden_shape, dtype).id;

        // --- 构建 8 个子 op，按需分发 attributes，附加 weight tensor ---

        ir::Operation norm0;
        norm0.name = prefix + ".rmsnorm0";
        norm0.kind = ir::OpKind::kRmsNorm;
        norm0.inputs = {block_input};
        norm0.outputs = {rmsnorm0_out};
        norm0.attributes = PickAttrs(op.attributes, {"hidden_size"});
        AttachWeights(graph, norm0, prefix + ".rmsnorm0",
            {{{H}}},  // weight: [H]
            dtype);

        ir::Operation qkv;
        qkv.name = prefix + ".qkv_project";
        qkv.kind = ir::OpKind::kQkvProject;
        qkv.inputs = {rmsnorm0_out};
        qkv.outputs = {qkv_out};
        qkv.attributes = PickAttrs(op.attributes,
            {"hidden_size", "num_attention_heads", "num_key_value_heads"});
        AttachWeights(graph, qkv, prefix + ".qkv_project",
            {{{H, H}}, {{H, H_kv}}, {{H, H_kv}}},  // wq:[H,H] wk:[H,H_kv] wv:[H,H_kv]
            dtype);

        ir::Operation rope;
        rope.name = prefix + ".rope";
        rope.kind = ir::OpKind::kRope;
        rope.inputs = {qkv_out};
        rope.outputs = {rope_out};
        rope.attributes = PickAttrs(op.attributes,
            {"num_attention_heads", "max_position_embeddings"});
        // rope 无 weight（注册表里 weight_inputs 为空，AttachWeights 不做任何事）
        AttachWeights(graph, rope, prefix + ".rope", {}, dtype);

        // --- attention 细拆：7 个子 op 替代原来的 1 个 attention 黑盒 ---
        // 数据流：rope_out → split_qkv → score_matmul → score_scale →
        //         causal_mask → softmax → context_matmul → output_proj → attention_out

        const std::int64_t nH = static_cast<std::int64_t>(num_heads);
        const std::int64_t d = static_cast<std::int64_t>(head_dim);

        // attention 内部的中间 tensor shape
        const ir::Shape q_shape = {{B, nH, S, d}};        // [B, num_heads, S, head_dim]
        const ir::Shape kv_shape = {{B, nH, S, d}};       // [B, num_heads, S, head_dim]（GQA 扩展后）
        const ir::Shape score_shape = {{B, nH, S, S}};    // [B, num_heads, S, S]
        const ir::Shape context_shape = {{B, nH, S, d}};  // [B, num_heads, S, head_dim]

        // 中间 tensor
        const int q_out = graph.AddTensor(
            prefix + ".attn.q", q_shape, dtype).id;
        const int k_out = graph.AddTensor(
            prefix + ".attn.k", kv_shape, dtype).id;
        const int v_out = graph.AddTensor(
            prefix + ".attn.v", kv_shape, dtype).id;
        const int raw_scores = graph.AddTensor(
            prefix + ".attn.raw_scores", score_shape, dtype).id;
        const int scaled_scores = graph.AddTensor(
            prefix + ".attn.scaled_scores", score_shape, dtype).id;
        const int masked_scores = graph.AddTensor(
            prefix + ".attn.masked_scores", score_shape, dtype).id;
        const int attn_weights = graph.AddTensor(
            prefix + ".attn.weights", score_shape, dtype).id;
        const int context = graph.AddTensor(
            prefix + ".attn.context", context_shape, dtype).id;
        const int concat_out = graph.AddTensor(
            prefix + ".attn.concat", hidden_shape, dtype).id;

        // ① split_qkv: 从 rope_out [B,S,Q+K+V] 拆出 Q/K/V 并变形为多头格式
        //    内含：按维度拆分 + reshape 成 [B,nH,S,d] + GQA 头扩展
        ir::Operation split_qkv;
        split_qkv.name = prefix + ".attn.split_qkv";
        split_qkv.kind = ir::OpKind::kSplitQkv;
        split_qkv.inputs = {rope_out};
        split_qkv.outputs = {q_out, k_out, v_out};
        split_qkv.attributes = PickAttrs(op.attributes,
            {"num_attention_heads", "num_key_value_heads"});

        // ② score_matmul: Q × Kᵀ → raw_scores [B,nH,S,S]
        ir::Operation score_mm;
        score_mm.name = prefix + ".attn.score_matmul";
        score_mm.kind = ir::OpKind::kBatchMatMul;
        score_mm.inputs = {q_out, k_out};
        score_mm.outputs = {raw_scores};

        // ③ score_scale: raw_scores × scale_factor → scaled_scores
        //    scale_factor 是一个标量 tensor，值 = 1/sqrt(head_dim)
        const int scale_tensor = graph.AddTensor(
            prefix + ".attn.scale_factor", {{1}}, dtype).id;
        ir::Operation scale;
        scale.name = prefix + ".attn.score_scale";
        scale.kind = ir::OpKind::kMul;
        scale.inputs = {raw_scores, scale_tensor};
        scale.outputs = {scaled_scores};

        // ④ causal_mask: 把未来位置设为 -∞
        ir::Operation mask;
        mask.name = prefix + ".attn.causal_mask";
        mask.kind = ir::OpKind::kCausalMask;
        mask.inputs = {scaled_scores};
        mask.outputs = {masked_scores};

        // ⑤ softmax: 归一化为概率
        ir::Operation sm;
        sm.name = prefix + ".attn.softmax";
        sm.kind = ir::OpKind::kSoftmax;
        sm.inputs = {masked_scores};
        sm.outputs = {attn_weights};

        // ⑥ context_matmul: attn_weights × V → context [B,nH,S,d]
        ir::Operation ctx_mm;
        ctx_mm.name = prefix + ".attn.context_matmul";
        ctx_mm.kind = ir::OpKind::kBatchMatMul;
        ctx_mm.inputs = {attn_weights, v_out};
        ctx_mm.outputs = {context};

        // ⑦ reshape: [B,nH,S,d] → [B,S,H]（多头拼接回原始维度）
        ir::Operation reshape;
        reshape.name = prefix + ".attn.reshape";
        reshape.kind = ir::OpKind::kReshape;
        reshape.inputs = {context};
        reshape.outputs = {concat_out};

        // ⑧ output_proj: [B,S,H] × Wo → [B,S,H]
        ir::Operation out_proj;
        out_proj.name = prefix + ".attn.output_proj";
        out_proj.kind = ir::OpKind::kLinear;
        out_proj.inputs = {concat_out};
        out_proj.outputs = {attention_out};
        AttachWeights(graph, out_proj, prefix + ".attn.output_proj",
            {{{H, H}}},  // wo: [H, H]
            dtype);

        ir::Operation add0;
        add0.name = prefix + ".residual0";
        add0.kind = ir::OpKind::kResidualAdd;
        add0.inputs = {block_input, attention_out};
        add0.outputs = {residual0_out};
        // residual_add 无 weight

        ir::Operation norm1;
        norm1.name = prefix + ".rmsnorm1";
        norm1.kind = ir::OpKind::kRmsNorm;
        norm1.inputs = {residual0_out};
        norm1.outputs = {rmsnorm1_out};
        norm1.attributes = PickAttrs(op.attributes, {"hidden_size"});
        AttachWeights(graph, norm1, prefix + ".rmsnorm1",
            {{{H}}},  // weight: [H]
            dtype);

        ir::Operation mlp;
        mlp.name = prefix + ".swiglu";
        mlp.kind = ir::OpKind::kSwiGLU;
        mlp.inputs = {rmsnorm1_out};
        mlp.outputs = {swiglu_out};
        mlp.attributes = PickAttrs(op.attributes,
            {"hidden_size", "intermediate_size"});
        AttachWeights(graph, mlp, prefix + ".swiglu",
            {{{H, I}}, {{H, I}}, {{I, H}}},  // w_gate:[H,I] w_up:[H,I] w_down:[I,H]
            dtype);

        ir::Operation add1;
        add1.name = prefix + ".residual1";
        add1.kind = ir::OpKind::kResidualAdd;
        add1.inputs = {residual0_out, swiglu_out};
        add1.outputs = {block_output};
        // residual_add 无 weight

        lowered.push_back(norm0);
        lowered.push_back(qkv);
        lowered.push_back(rope);
        lowered.push_back(split_qkv);
        lowered.push_back(score_mm);
        lowered.push_back(scale);
        lowered.push_back(mask);
        lowered.push_back(sm);
        lowered.push_back(ctx_mm);
        lowered.push_back(reshape);
        lowered.push_back(out_proj);
        lowered.push_back(add0);
        lowered.push_back(norm1);
        lowered.push_back(mlp);
        lowered.push_back(add1);
    }

    // 重新编号所有 op 的 id
    for (std::size_t index = 0; index < lowered.size(); ++index) {
        lowered[index].id = static_cast<int>(index);
    }
    graph.operations() = lowered;

    // 修复问题 4：重建 tensor 的 producer_op / consumer_ops
    RebuildTensorRefs(graph);

    return self_compiler::Status::Ok();
}

}  // namespace self_compiler::passes
