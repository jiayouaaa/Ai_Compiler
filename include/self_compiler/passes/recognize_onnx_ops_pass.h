#pragma once

#include "self_compiler/passes/pass.h"

namespace self_compiler::passes {

class RecognizeOnnxOpsPass final : public Pass {
public:
    std::string name() const override;
    self_compiler::Status Run(ir::Graph& graph) override;
};

}  // namespace self_compiler::passes
