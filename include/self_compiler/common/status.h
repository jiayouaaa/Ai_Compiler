#pragma once

#include <string>

namespace self_compiler {

struct Status {
    bool ok = true;
    std::string message;

    static Status Ok() {
        return {true, ""};
    }

    static Status Error(std::string text) {
        return {false, std::move(text)};
    }
};

}  // namespace self_compiler
