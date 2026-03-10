#pragma once

#include <ostream>
#include <string>

#include "self_compiler/common/status.h"

namespace self_compiler::app {

struct RunOptions {
    bool dump_graph = true;
    bool dump_command_stream = true;
    bool dump_mlir_stub = true;
    std::string input_path;
    std::string input_format;
};

class CompilerApp {
public:
    self_compiler::Status Run(const RunOptions& options, std::ostream& out) const;
};

}  // namespace self_compiler::app
