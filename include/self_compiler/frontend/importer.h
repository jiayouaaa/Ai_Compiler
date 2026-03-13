#pragma once

#include <string>

#include "self_compiler/common/status.h"
#include "self_compiler/ir/graph.h"

namespace self_compiler::frontend {

enum class InputFormat {
    kSpec,
    kLlamaConfig,
    kJson,
    kOnnx,
    kTFLite,
    kStableHlo,
    kMlir,
    kUnknown,
};

inline std::string ToString(InputFormat format) {
    switch (format) {
        case InputFormat::kSpec:
            return "spec";
        case InputFormat::kLlamaConfig:
            return "llama_config";
        case InputFormat::kJson:
            return "json";
        case InputFormat::kOnnx:
            return "onnx";
        case InputFormat::kTFLite:
            return "tflite";
        case InputFormat::kStableHlo:
            return "stablehlo";
        case InputFormat::kMlir:
            return "mlir";
        default:
            return "unknown";
    }
}

class Importer {
public:
    virtual ~Importer() = default;
    virtual InputFormat format() const = 0;
    virtual self_compiler::Status Import(const std::string& input_path, ir::Graph& graph) const = 0;
};

}  // namespace self_compiler::frontend
