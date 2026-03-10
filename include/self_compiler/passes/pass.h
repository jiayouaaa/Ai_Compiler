#pragma once

#include <string>

#include "self_compiler/common/status.h"
#include "self_compiler/ir/graph.h"

namespace self_compiler::passes {

class Pass {
public:
    virtual ~Pass() = default;
    virtual std::string name() const = 0;
    virtual self_compiler::Status Run(ir::Graph& graph) = 0;
};

}  // namespace self_compiler::passes
