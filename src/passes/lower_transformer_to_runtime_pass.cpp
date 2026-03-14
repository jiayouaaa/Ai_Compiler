#include "self_compiler/passes/lower_transformer_to_runtime_pass.h"

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

        // --- 构建 8 个子 op，按需分发 attributes ---

        ir::Operation norm0;
        norm0.name = prefix + ".rmsnorm0";
        norm0.kind = ir::OpKind::kRmsNorm;
        norm0.inputs = {block_input};
        norm0.outputs = {rmsnorm0_out};
        norm0.attributes = PickAttrs(op.attributes, {"hidden_size"});

        ir::Operation qkv;
        qkv.name = prefix + ".qkv_project";
        qkv.kind = ir::OpKind::kQkvProject;
        qkv.inputs = {rmsnorm0_out};
        qkv.outputs = {qkv_out};
        qkv.attributes = PickAttrs(op.attributes,
            {"hidden_size", "num_attention_heads", "num_key_value_heads"});

        ir::Operation rope;
        rope.name = prefix + ".rope";
        rope.kind = ir::OpKind::kRope;
        rope.inputs = {qkv_out};
        rope.outputs = {rope_out};
        rope.attributes = PickAttrs(op.attributes,
            {"num_attention_heads", "max_position_embeddings"});

        ir::Operation attn;
        attn.name = prefix + ".attention";
        attn.kind = ir::OpKind::kAttention;
        attn.inputs = {rope_out};
        attn.outputs = {attention_out};
        attn.attributes = PickAttrs(op.attributes,
            {"num_attention_heads", "num_key_value_heads"});

        ir::Operation add0;
        add0.name = prefix + ".residual0";
        add0.kind = ir::OpKind::kResidualAdd;
        add0.inputs = {block_input, attention_out};
        add0.outputs = {residual0_out};

        ir::Operation norm1;
        norm1.name = prefix + ".rmsnorm1";
        norm1.kind = ir::OpKind::kRmsNorm;
        norm1.inputs = {residual0_out};
        norm1.outputs = {rmsnorm1_out};
        norm1.attributes = PickAttrs(op.attributes, {"hidden_size"});

        ir::Operation mlp;
        mlp.name = prefix + ".swiglu";
        mlp.kind = ir::OpKind::kSwiGLU;
        mlp.inputs = {rmsnorm1_out};
        mlp.outputs = {swiglu_out};
        mlp.attributes = PickAttrs(op.attributes,
            {"hidden_size", "intermediate_size"});

        ir::Operation add1;
        add1.name = prefix + ".residual1";
        add1.kind = ir::OpKind::kResidualAdd;
        add1.inputs = {residual0_out, swiglu_out};
        add1.outputs = {block_output};

        lowered.push_back(norm0);
        lowered.push_back(qkv);
        lowered.push_back(rope);
        lowered.push_back(attn);
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
