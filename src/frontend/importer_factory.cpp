#include "self_compiler/frontend/importer_factory.h"

#include <algorithm>

#include "self_compiler/frontend/json_importer.h"
#include "self_compiler/frontend/onnx_importer.h"

namespace self_compiler::frontend {

namespace {

std::string ToLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

bool EndsWith(const std::string& text, const std::string& suffix) {
    if (text.size() < suffix.size()) {
        return false;
    }
    return text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

InputFormat ImporterFactory::ParseFormat(const std::string& text) {
    const std::string lower = ToLower(text);
    if (lower == "spec") return InputFormat::kSpec;
    if (lower == "json") return InputFormat::kJson;
    if (lower == "onnx") return InputFormat::kOnnx;
    if (lower == "tflite") return InputFormat::kTFLite;
    if (lower == "stablehlo") return InputFormat::kStableHlo;
    if (lower == "mlir") return InputFormat::kMlir;
    return InputFormat::kUnknown;
}

InputFormat ImporterFactory::InferFormatFromPath(const std::string& path) {
    const std::string lower = ToLower(path);
    if (EndsWith(lower, ".json")) return InputFormat::kJson;
    if (EndsWith(lower, ".onnx")) return InputFormat::kOnnx;
    if (EndsWith(lower, ".tflite")) return InputFormat::kTFLite;
    if (EndsWith(lower, ".mlir")) return InputFormat::kMlir;
    if (EndsWith(lower, ".stablehlo")) return InputFormat::kStableHlo;
    return InputFormat::kUnknown;
}

std::unique_ptr<Importer> ImporterFactory::Create(InputFormat format) {
    switch (format) {
        case InputFormat::kJson:
            return std::make_unique<JsonImporter>();
        case InputFormat::kOnnx:
            return std::make_unique<OnnxImporter>();
        default:
            return nullptr;
    }
}

}  // namespace self_compiler::frontend
