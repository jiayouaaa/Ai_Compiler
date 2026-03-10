#include "self_compiler/frontend/json_importer.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "self_compiler/frontend/transformer_block_builder.h"

namespace self_compiler::frontend {

namespace {

bool TryExtractInt(const std::string& text, const std::string& key, int& value) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return false;
    }
    value = std::stoi(match[1].str());
    return true;
}

self_compiler::Status ValidateSpec(const TransformerBlockSpec& spec) {
    if (spec.batch <= 0) {
        return self_compiler::Status::Error("JSON 字段 batch 必须大于 0。");
    }
    if (spec.sequence_length <= 0) {
        return self_compiler::Status::Error("JSON 字段 sequence_length 必须大于 0。");
    }
    if (spec.hidden_size <= 0) {
        return self_compiler::Status::Error("JSON 字段 hidden_size 必须大于 0。");
    }
    if (spec.intermediate_size <= 0) {
        return self_compiler::Status::Error("JSON 字段 intermediate_size 必须大于 0。");
    }
    if (spec.num_attention_heads <= 0) {
        return self_compiler::Status::Error("JSON 字段 num_attention_heads 必须大于 0。");
    }
    if (spec.num_key_value_heads <= 0) {
        return self_compiler::Status::Error("JSON 字段 num_key_value_heads 必须大于 0。");
    }
    if (spec.vocab_size <= 0) {
        return self_compiler::Status::Error("JSON 字段 vocab_size 必须大于 0。");
    }
    if (spec.hidden_size % spec.num_attention_heads != 0) {
        return self_compiler::Status::Error("hidden_size 必须能被 num_attention_heads 整除。");
    }
    if (spec.num_attention_heads % spec.num_key_value_heads != 0) {
        return self_compiler::Status::Error("num_attention_heads 必须能被 num_key_value_heads 整除。");
    }
    return self_compiler::Status::Ok();
}

}  // namespace

self_compiler::Status JsonImporter::Import(const std::string& input_path, ir::Graph& graph) const {
    std::ifstream input(input_path);
    if (!input.is_open()) {
        return self_compiler::Status::Error("无法打开 JSON 输入文件: " + input_path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string content = buffer.str();
    if (content.empty()) {
        return self_compiler::Status::Error("JSON 输入文件为空: " + input_path);
    }

    TransformerBlockSpec spec;
    std::vector<std::string> overridden_fields;
    int value = 0;

    if (TryExtractInt(content, "batch", value)) {
        spec.batch = value;
        overridden_fields.push_back("batch");
    }
    if (TryExtractInt(content, "sequence_length", value)) {
        spec.sequence_length = value;
        overridden_fields.push_back("sequence_length");
    }
    if (TryExtractInt(content, "hidden_size", value)) {
        spec.hidden_size = value;
        overridden_fields.push_back("hidden_size");
    }
    if (TryExtractInt(content, "intermediate_size", value)) {
        spec.intermediate_size = value;
        overridden_fields.push_back("intermediate_size");
    }
    if (TryExtractInt(content, "num_attention_heads", value)) {
        spec.num_attention_heads = value;
        overridden_fields.push_back("num_attention_heads");
    }
    if (TryExtractInt(content, "num_key_value_heads", value)) {
        spec.num_key_value_heads = value;
        overridden_fields.push_back("num_key_value_heads");
    }
    if (TryExtractInt(content, "vocab_size", value)) {
        spec.vocab_size = value;
        overridden_fields.push_back("vocab_size");
    }

    if (overridden_fields.empty()) {
        return self_compiler::Status::Error(
            "未在 JSON 中解析到任何 TransformerBlockSpec 字段。"
            "请至少提供 batch、sequence_length、hidden_size 等整数键值。");
    }

    auto validate_status = ValidateSpec(spec);
    if (!validate_status.ok) {
        return validate_status;
    }

    TransformerBlockBuilder builder;
    graph = builder.Build(spec);
    return self_compiler::Status::Ok();
}

}  // namespace self_compiler::frontend
