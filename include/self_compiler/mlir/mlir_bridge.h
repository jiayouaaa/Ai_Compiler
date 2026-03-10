#pragma once

#include <ostream>

#include "self_compiler/common/status.h"
#include "self_compiler/ir/graph.h"

namespace self_compiler::mlir {

class MlirBridge {
public:
    self_compiler::Status ExportDialectSkeleton(const ir::Graph& graph, std::ostream& out) const;
};

}  // namespace self_compiler::mlir
