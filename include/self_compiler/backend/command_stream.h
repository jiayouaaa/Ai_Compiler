#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace self_compiler::backend {

struct Command {
    std::string opcode;
    std::vector<std::string> operands;
};

class CommandStream {
public:
    void Add(Command command);
    void Dump(std::ostream& out) const;

private:
    std::vector<Command> commands_;
};

}  // namespace self_compiler::backend
