#pragma once

#include <memory>
#include <string>

#include "self_compiler/frontend/importer.h"

namespace self_compiler::frontend {

class ImporterFactory {
public:
    static InputFormat ParseFormat(const std::string& text);
    static InputFormat InferFormatFromPath(const std::string& path);
    static std::unique_ptr<Importer> Create(InputFormat format);
};

}  // namespace self_compiler::frontend
