#pragma once

#include <memory>
#include <vector>

#include "self_compiler/common/status.h"
#include "self_compiler/passes/pass.h"

namespace self_compiler::passes {

class PassManager {
public:
    void AddPass(std::unique_ptr<Pass> pass);
    self_compiler::Status Run(ir::Graph& graph) const;

private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

}  // namespace self_compiler::passes
