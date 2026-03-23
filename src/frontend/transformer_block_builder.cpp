#include "self_compiler/frontend/transformer_block_builder.h"

namespace self_compiler::frontend {

ir::Graph TransformerBlockBuilder::Build(const TransformerBlockSpec& spec) const {
    ir::Graph graph;

    const int input_ids = graph.AddTensor("input_ids", {{spec.batch, spec.sequence_length}}, ir::DataType::kInt32).id;
    const int hidden_states = graph.AddTensor("hidden_states", {{spec.batch, spec.sequence_length, spec.hidden_size}}, ir::DataType::kBFloat16).id;
    const int block_output = graph.AddTensor("block_output", {{spec.batch, spec.sequence_length, spec.hidden_size}}, ir::DataType::kBFloat16).id;
    const int logits = graph.AddTensor("logits", {{spec.batch, spec.sequence_length, spec.vocab_size}}, ir::DataType::kBFloat16).id;

    // Embedding weight: 词表 [vocab_size, hidden_size]
    const int embedding_weight = graph.AddTensor("embedding.weight",
        {{static_cast<std::int64_t>(spec.vocab_size), static_cast<std::int64_t>(spec.hidden_size)}},
        ir::DataType::kBFloat16).id;

    // LmHead weight: 输出投影 [hidden_size, vocab_size]
    const int lm_head_weight = graph.AddTensor("lm_head.weight",
        {{static_cast<std::int64_t>(spec.hidden_size), static_cast<std::int64_t>(spec.vocab_size)}},
        ir::DataType::kBFloat16).id;

    graph.AddOperation("input", ir::OpKind::kInput, {}, {input_ids});
    graph.AddOperation("embedding", ir::OpKind::kEmbedding, {input_ids, embedding_weight}, {hidden_states});
    auto& block = graph.AddOperation("transformer_block_0", ir::OpKind::kTransformerBlock, {hidden_states}, {block_output});
    block.attributes["hidden_size"] = std::to_string(spec.hidden_size);
    block.attributes["intermediate_size"] = std::to_string(spec.intermediate_size);
    block.attributes["num_attention_heads"] = std::to_string(spec.num_attention_heads);
    block.attributes["num_key_value_heads"] = std::to_string(spec.num_key_value_heads);
    graph.AddOperation("lm_head", ir::OpKind::kLmHead, {block_output, lm_head_weight}, {logits});
    graph.AddOperation("output", ir::OpKind::kOutput, {logits}, {});

    return graph;
}

}  // namespace self_compiler::frontend
