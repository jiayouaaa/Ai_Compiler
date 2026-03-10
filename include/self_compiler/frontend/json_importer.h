#pragma once

#include "self_compiler/frontend/importer.h"

namespace self_compiler::frontend {

class JsonImporter final : public Importer {
public:
    InputFormat format() const override { return InputFormat::kJson; }
    self_compiler::Status Import(const std::string& input_path, ir::Graph& graph) const override;
};

}  // namespace self_compiler::frontend
