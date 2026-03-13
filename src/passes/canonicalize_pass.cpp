#include "self_compiler/passes/canonicalize_pass.h"

#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

self_compiler::Status CheckTensorId(const self_compiler::ir::Graph& graph) {
    const auto& tensors = graph.tensors();
    const auto& operations = graph.operations();

    for (const auto& op : operations) {
        for (int input_tensor_id : op.inputs) {
            if (input_tensor_id < 0 || input_tensor_id >= static_cast<int>(tensors.size())) {
                return self_compiler::Status::Error(
                    op.name + ": Invalid tensor ID in operation inputs: " + std::to_string(input_tensor_id));
            }
        }

        for (int output_tensor_id : op.outputs) {
            if (output_tensor_id < 0 || output_tensor_id >= static_cast<int>(tensors.size())) {
                return self_compiler::Status::Error(
                    op.name + ": Invalid tensor ID in operation outputs: " + std::to_string(output_tensor_id));
            }
        }
    }

    return self_compiler::Status::Ok();
}

self_compiler::Status CheckOpArity(const self_compiler::ir::Operation& op) {
    const auto actual_input_count = op.inputs.size();
    const auto actual_output_count = op.outputs.size();

    if (op.kind == self_compiler::ir::OpKind::kInput) {
        if (actual_input_count != 0) {
            return self_compiler::Status::Error(
                op.name + ": Input operation should not have any inputs, but has " +
                std::to_string(actual_input_count));
        }
        if (actual_output_count != 1) {
            return self_compiler::Status::Error(
                op.name + ": Input operation should have exactly 1 output, but has " +
                std::to_string(actual_output_count));
        }
    } else if (op.kind == self_compiler::ir::OpKind::kOutput) {
        if (actual_input_count != 1) {
            return self_compiler::Status::Error(
                op.name + ": Output operation should have exactly 1 input, but has " +
                std::to_string(actual_input_count));
        }
        if (actual_output_count != 0) {
            return self_compiler::Status::Error(
                op.name + ": Output operation should not have any outputs, but has " +
                std::to_string(actual_output_count));
        }
    } else if (op.kind == self_compiler::ir::OpKind::kResidualAdd) {
        if (actual_input_count != 2) {
            return self_compiler::Status::Error(
                op.name + ": ResidualAdd operation should have exactly 2 inputs, but has " +
                std::to_string(actual_input_count));
        }
        if (actual_output_count != 1) {
            return self_compiler::Status::Error(
                op.name + ": ResidualAdd operation should have exactly 1 output, but has " +
                std::to_string(actual_output_count));
        }
    } else if (op.kind == self_compiler::ir::OpKind::kTransformerBlock) {
        if (actual_input_count != 1) {
            return self_compiler::Status::Error(
                op.name + ": TransformerBlock operation should have exactly 1 input, but has " +
                std::to_string(actual_input_count));
        }
        if (actual_output_count != 1) {
            return self_compiler::Status::Error(
                op.name + ": TransformerBlock operation should have exactly 1 output, but has " +
                std::to_string(actual_output_count));
        }
    } else {
        if (actual_input_count != 1) {
            return self_compiler::Status::Error(
                op.name + ": operation should have exactly 1 input, but has " +
                std::to_string(actual_input_count));
        }
        if (actual_output_count != 1) {
            return self_compiler::Status::Error(
                op.name + ": operation should have exactly 1 output, but has " +
                std::to_string(actual_output_count));
        }
    }

    return self_compiler::Status::Ok();
}

self_compiler::Status CheckRequiredAttributes(const self_compiler::ir::Operation& op) {
    if (op.kind != self_compiler::ir::OpKind::kTransformerBlock) {
        return self_compiler::Status::Ok();
    }

    const std::vector<std::string> required_attributes = {
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
    };

    for (const auto& attr : required_attributes) {
        if (op.attributes.find(attr) == op.attributes.end()) {
            return self_compiler::Status::Error(
                op.name + ": Missing required attribute '" + attr +
                "' for " + self_compiler::ir::ToString(op.kind) + " operation");
        }
    }

    return self_compiler::Status::Ok();
}

int Attr_Stoi(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("Empty string cannot be converted to integer");
    }

    int sign = 1;
    std::size_t index = 0;
    if (text[0] == '-') {
        sign = -1;
        index = 1;
    }

    if (index == text.size()) {
        throw std::invalid_argument("Sign only string cannot be converted to integer");
    }

    int result = 0;
    for (; index < text.size(); ++index) {
        const unsigned char ch = static_cast<unsigned char>(text[index]);
        if (std::isdigit(ch) == 0) {
            throw std::invalid_argument("Invalid integer string: " + text);
        }

        if (result > (std::numeric_limits<int>::max() - (text[index] - '0')) / 10) {
            throw std::out_of_range("Integer out of range: " + text);
        }
        result = result * 10 + (text[index] - '0');
    }

    return sign * result;
}

self_compiler::Status CheckAttributeValue(const self_compiler::ir::Operation& op) {
    if (op.kind != self_compiler::ir::OpKind::kTransformerBlock) {
        return self_compiler::Status::Ok();
    }

    const std::vector<std::string> required_attributes = {
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
    };

    for (const auto& attr : required_attributes) {
        const auto it = op.attributes.find(attr);
        if (it == op.attributes.end()) {
            return self_compiler::Status::Error(
                op.name + ": Missing attribute value for '" + attr + "'");
        }

        const auto& attr_str_value = it->second;

        try {
            const int attr_value = Attr_Stoi(attr_str_value);
            if (attr_value <= 0) {
                return self_compiler::Status::Error(
                    op.name + ": Attribute '" + attr +
                    "' must be a positive integer, but got '" + attr_str_value + "'");
            }
        } catch (const std::exception&) {
            return self_compiler::Status::Error(
                op.name + ": Attribute '" + attr +
                "' must be a valid integer, but got '" + attr_str_value + "'");
        }
    }

    return self_compiler::Status::Ok();
}

std::string CanonicalizeIdentifierBody(const std::string& source) {
    std::string result;
    result.reserve(source.size());
    bool last_was_underscore = false;

    for (unsigned char ch : source) {
        if (std::isalnum(ch) != 0) {
            result.push_back(static_cast<char>(std::tolower(ch)));
            last_was_underscore = false;
        } else if (!last_was_underscore) {
            result.push_back('_');
            last_was_underscore = true;
        }
    }

    while (!result.empty() && result.front() == '_') {
        result.erase(result.begin());
    }
    while (!result.empty() && result.back() == '_') {
        result.pop_back();
    }

    return result;
}

std::string CanonicalizeName(const std::string& raw_name, const std::string& fallback_prefix, int id) {
    std::string source = raw_name;
    if (source.empty()) {
        source = fallback_prefix + "_" + std::to_string(id);
    }

    std::string result = CanonicalizeIdentifierBody(source);
    if (result.empty()) {
        return fallback_prefix + "_" + std::to_string(id);
    }
    return result;
}

std::string CanonicalizeAttributeKey(const std::string& raw_key) {
    return CanonicalizeIdentifierBody(raw_key);
}

self_compiler::Status CheckAttributeRelations(const self_compiler::ir::Operation& op) {
    if (op.kind != self_compiler::ir::OpKind::kTransformerBlock) {
        return self_compiler::Status::Ok();
    }

    const int hidden_size = Attr_Stoi(op.attributes.find("hidden_size")->second);
    const int num_attention_heads = Attr_Stoi(op.attributes.find("num_attention_heads")->second);
    const int num_key_value_heads = Attr_Stoi(op.attributes.find("num_key_value_heads")->second);

    if (hidden_size % num_attention_heads != 0) {
        return self_compiler::Status::Error(
            op.name + ": hidden_size " + std::to_string(hidden_size) +
            " must be divisible by num_attention_heads " +
            std::to_string(num_attention_heads));
    }

    if (num_attention_heads % num_key_value_heads != 0) {
        return self_compiler::Status::Error(
            op.name + ": num_attention_heads " +
            std::to_string(num_attention_heads) +
            " must be divisible by num_key_value_heads " +
            std::to_string(num_key_value_heads));
    }

    return self_compiler::Status::Ok();
}

std::string Trim(const std::string& s) {
    if (s.empty()) {
        return "";
    }

    int left = 0;
    int right = static_cast<int>(s.size()) - 1;
    while (left <= right && std::isspace(static_cast<unsigned char>(s[left])) != 0) {
        ++left;
    }
    while (left <= right && std::isspace(static_cast<unsigned char>(s[right])) != 0) {
        --right;
    }

    return (left <= right) ? s.substr(left, right - left + 1) : "";
}

std::string ToLowerCopy(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

bool IsIntegerString(const std::string& s) {
    if (s.empty()) {
        return false;
    }

    int i = 0;
    if (s[0] == '+' || s[0] == '-') {
        if (s.size() == 1) {
            return false;
        }
        i = 1;
    }

    for (; i < static_cast<int>(s.size()); ++i) {
        if (std::isdigit(static_cast<unsigned char>(s[i])) == 0) {
            return false;
        }
    }

    return true;
}

std::string NormalizeIntegerString(const std::string& s) {
    int i = 0;
    bool negative = false;

    if (s[0] == '+' || s[0] == '-') {
        negative = (s[0] == '-');
        i = 1;
    }

    while (i < static_cast<int>(s.size()) && s[i] == '0') {
        ++i;
    }

    std::string core = (i == static_cast<int>(s.size())) ? "0" : s.substr(i);
    if (core == "0") {
        return "0";
    }
    return negative ? "-" + core : core;
}

std::string CanonicalizeAttributeValue(std::string s) {
    s = Trim(s);

    const std::string lower = ToLowerCopy(s);
    if (lower == "true") {
        return "True";
    }
    if (lower == "false") {
        return "False";
    }

    if (IsIntegerString(s)) {
        return NormalizeIntegerString(s);
    }

    return s;
}

void NormalizeAttributes(self_compiler::ir::Operation& op) {
    decltype(op.attributes) normalized;

    for (const auto& item : op.attributes) {
        const auto key = CanonicalizeAttributeKey(item.first);
        const auto value = CanonicalizeAttributeValue(item.second);
        normalized[key] = value;
    }

    op.attributes = std::move(normalized);
}

void NormalizeNames(self_compiler::ir::Graph& graph) {
    for (auto& tensor : graph.tensors()) {
        tensor.name = CanonicalizeName(tensor.name, "tensor", tensor.id);
    }

    for (auto& op : graph.operations()) {
        op.name = CanonicalizeName(op.name, "op", op.id);
    }
}

}  // namespace

namespace self_compiler::passes {

std::string CanonicalizePass::name() const {
    return "CanonicalizePass";
}

self_compiler::Status CanonicalizePass::Run(ir::Graph& graph) {
    auto tensor_status = CheckTensorId(graph);
    if (!tensor_status.ok) {
        return self_compiler::Status::Error("Tensor ID check failed: " + tensor_status.message);
    }

    for (const auto& op : graph.operations()) {
        auto arity_status = CheckOpArity(op);
        if (!arity_status.ok) {
            return self_compiler::Status::Error("Operation arity check failed: " + arity_status.message);
        }

        auto required_attributes_status = CheckRequiredAttributes(op);
        if (!required_attributes_status.ok) {
            return self_compiler::Status::Error("Required attributes check failed: " + required_attributes_status.message);
        }

        auto attribute_value_status = CheckAttributeValue(op);
        if (!attribute_value_status.ok) {
            return self_compiler::Status::Error("Attribute value check failed: " + attribute_value_status.message);
        }

        auto attribute_relations_status = CheckAttributeRelations(op);
        if (!attribute_relations_status.ok) {
            return self_compiler::Status::Error("Attribute relations check failed: " + attribute_relations_status.message);
        }
    }

    for (auto& op : graph.operations()) {
        NormalizeAttributes(op);
    }

    NormalizeNames(graph);

    for (auto& op : graph.operations()) {
        op.attributes["canonicalized"] = "true";
    }

    // 伪代码：后续请用真实规范化逻辑替换这里
    // 遍历图中的每个算子：
    //   消除无效的 reshape / transpose
    //   删除无用常量
    //   统一属性布局
    //   规范张量命名和边顺序

    return self_compiler::Status::Ok();
}

}  // namespace self_compiler::passes
