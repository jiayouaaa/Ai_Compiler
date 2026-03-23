#include "self_compiler/frontend/llama_config_importer.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <string>

namespace self_compiler::frontend {

namespace {

struct LlamaConfigSpec {
    int batch = 1;
    int sequence_length = 16;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_attention_heads = 0;
    int num_hidden_layers = 0;
    int num_key_value_heads = 0;
    int vocab_size = 0;
    int max_position_embeddings = 0;
    self_compiler::ir::DataType activation_dtype = self_compiler::ir::DataType::kBFloat16;
    std::string model_type;
};

bool TryExtractInt(const std::string& text, const std::string& key, int& value) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return false;
    }
    value = std::stoi(match[1].str());
    return true;
}

bool TryExtractString(const std::string& text, const std::string& key, std::string& value) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return false;
    }
    value = match[1].str();
    return true;
}

self_compiler::ir::DataType ParseTorchDType(const std::string& text) {
    if (text == "bfloat16") {
        return self_compiler::ir::DataType::kBFloat16;
    }
    if (text == "float32" || text == "float") {
        return self_compiler::ir::DataType::kFloat32;
    }
    return self_compiler::ir::DataType::kUnknown;
}

self_compiler::Status ValidateConfig(const LlamaConfigSpec& spec) {
    if (spec.model_type != "llama") {
        return self_compiler::Status::Error(
            "LlamaConfigImporter 只支持 model_type = \"llama\" 的配置。");
    }
    if (spec.hidden_size <= 0) {
        return self_compiler::Status::Error("Llama config 字段 hidden_size 必须大于 0。");
    }
    if (spec.intermediate_size <= 0) {
        return self_compiler::Status::Error("Llama config 字段 intermediate_size 必须大于 0。");
    }
    if (spec.num_attention_heads <= 0) {
        return self_compiler::Status::Error("Llama config 字段 num_attention_heads 必须大于 0。");
    }
    if (spec.num_hidden_layers <= 0) {
        return self_compiler::Status::Error("Llama config 字段 num_hidden_layers 必须大于 0。");
    }
    if (spec.num_key_value_heads <= 0) {
        return self_compiler::Status::Error("Llama config 字段 num_key_value_heads 必须大于 0。");
    }
    if (spec.vocab_size <= 0) {
        return self_compiler::Status::Error("Llama config 字段 vocab_size 必须大于 0。");
    }
    if (spec.max_position_embeddings <= 0) {
        return self_compiler::Status::Error("Llama config 字段 max_position_embeddings 必须大于 0。");
    }
    if (spec.sequence_length <= 0) {
        return self_compiler::Status::Error("Llama config 的预览 sequence_length 必须大于 0。");
    }
    if (spec.hidden_size % spec.num_attention_heads != 0) {
        return self_compiler::Status::Error("hidden_size 必须能被 num_attention_heads 整除。");
    }
    if (spec.num_attention_heads % spec.num_key_value_heads != 0) {
        return self_compiler::Status::Error("num_attention_heads 必须能被 num_key_value_heads 整除。");
    }
    if (spec.activation_dtype == self_compiler::ir::DataType::kUnknown) {
        return self_compiler::Status::Error("当前仅支持 torch_dtype 为 bfloat16 或 float32。");
    }
    return self_compiler::Status::Ok();
}

ir::Graph BuildLlamaGraph(const LlamaConfigSpec& spec) {
    ir::Graph graph;

    const int input_ids = graph.AddTensor(
        "input_ids",
        {{spec.batch, spec.sequence_length}},
        ir::DataType::kInt32).id;
    const int embedding_output = graph.AddTensor(
        "embedding_output",
        {{spec.batch, spec.sequence_length, spec.hidden_size}},
        spec.activation_dtype).id;

    graph.AddOperation("input", ir::OpKind::kInput, {}, {input_ids});

    // Embedding weight: [vocab_size, hidden_size]
    const int embedding_weight = graph.AddTensor(
        "embedding.weight",
        {{static_cast<std::int64_t>(spec.vocab_size),
          static_cast<std::int64_t>(spec.hidden_size)}},
        spec.activation_dtype).id;
    graph.AddOperation("embedding", ir::OpKind::kEmbedding,
        {input_ids, embedding_weight}, {embedding_output});

    int current_tensor = embedding_output;
    for (int layer_index = 0; layer_index < spec.num_hidden_layers; ++layer_index) {
        const int block_output = graph.AddTensor(
            "transformer_block_" + std::to_string(layer_index) + "_output",
            {{spec.batch, spec.sequence_length, spec.hidden_size}},
            spec.activation_dtype).id;

        auto& block = graph.AddOperation(
            "transformer_block_" + std::to_string(layer_index),
            ir::OpKind::kTransformerBlock,
            {current_tensor},
            {block_output});
        block.attributes["hidden_size"] = std::to_string(spec.hidden_size);
        block.attributes["intermediate_size"] = std::to_string(spec.intermediate_size);
        block.attributes["num_attention_heads"] = std::to_string(spec.num_attention_heads);
        block.attributes["num_key_value_heads"] = std::to_string(spec.num_key_value_heads);
        block.attributes["layer_index"] = std::to_string(layer_index);
        block.attributes["max_position_embeddings"] = std::to_string(spec.max_position_embeddings);
        current_tensor = block_output;
    }

    const int logits = graph.AddTensor(
        "logits",
        {{spec.batch, spec.sequence_length, spec.vocab_size}},
        spec.activation_dtype).id;

    // LmHead weight: [hidden_size, vocab_size]
    const int lm_head_weight = graph.AddTensor(
        "lm_head.weight",
        {{static_cast<std::int64_t>(spec.hidden_size),
          static_cast<std::int64_t>(spec.vocab_size)}},
        spec.activation_dtype).id;
    graph.AddOperation("lm_head", ir::OpKind::kLmHead,
        {current_tensor, lm_head_weight}, {logits});
    graph.AddOperation("output", ir::OpKind::kOutput, {logits}, {});

    return graph;
}

}  // namespace

self_compiler::Status LlamaConfigImporter::Import(const std::string& input_path, ir::Graph& graph) const {
    std::ifstream input(input_path);
    if (!input.is_open()) {
        return self_compiler::Status::Error("无法打开 Llama config 文件: " + input_path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string content = buffer.str();
    if (content.empty()) {
        return self_compiler::Status::Error("Llama config 文件为空: " + input_path);
    }

    LlamaConfigSpec spec;
    std::string torch_dtype = "bfloat16";

    if (!TryExtractString(content, "model_type", spec.model_type)) {
        return self_compiler::Status::Error("Llama config 缺少必需字段 model_type。");
    }
    if (!TryExtractInt(content, "hidden_size", spec.hidden_size)) {
        return self_compiler::Status::Error("Llama config 缺少必需字段 hidden_size。");
    }
    if (!TryExtractInt(content, "intermediate_size", spec.intermediate_size)) {
        return self_compiler::Status::Error("Llama config 缺少必需字段 intermediate_size。");
    }
    if (!TryExtractInt(content, "num_attention_heads", spec.num_attention_heads)) {
        return self_compiler::Status::Error("Llama config 缺少必需字段 num_attention_heads。");
    }
    if (!TryExtractInt(content, "num_hidden_layers", spec.num_hidden_layers)) {
        return self_compiler::Status::Error("Llama config 缺少必需字段 num_hidden_layers。");
    }
    if (!TryExtractInt(content, "num_key_value_heads", spec.num_key_value_heads)) {
        return self_compiler::Status::Error("Llama config 缺少必需字段 num_key_value_heads。");
    }
    if (!TryExtractInt(content, "vocab_size", spec.vocab_size)) {
        return self_compiler::Status::Error("Llama config 缺少必需字段 vocab_size。");
    }
    if (!TryExtractInt(content, "max_position_embeddings", spec.max_position_embeddings)) {
        return self_compiler::Status::Error("Llama config 缺少必需字段 max_position_embeddings。");
    }

    if (TryExtractInt(content, "sequence_length", spec.sequence_length)) {
    } else if (spec.max_position_embeddings < spec.sequence_length) {
        spec.sequence_length = spec.max_position_embeddings;
    }

    if (TryExtractInt(content, "batch", spec.batch)) {
    }
    if (TryExtractString(content, "torch_dtype", torch_dtype)) {
    }
    spec.activation_dtype = ParseTorchDType(torch_dtype);

    auto validate_status = ValidateConfig(spec);
    if (!validate_status.ok) {
        return validate_status;
    }

    graph = BuildLlamaGraph(spec);
    return self_compiler::Status::Ok();
}

}  // namespace self_compiler::frontend
