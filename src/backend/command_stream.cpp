#include "self_compiler/backend/command_stream.h"

namespace self_compiler::backend {

void CommandStream::Add(Command command) {
    commands_.push_back(std::move(command));
}

void CommandStream::Dump(std::ostream& out) const {
    for (const auto& command : commands_) {
        out << command.opcode;
        for (const auto& operand : command.operands) {
            out << " " << operand;
        }
        out << "\n";
    }
}

}  // namespace self_compiler::backend
