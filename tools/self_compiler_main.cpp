#include <iostream>
#include <string>

#include "self_compiler/app/compiler_app.h"

int main(int argc, char** argv) {
    self_compiler::app::RunOptions options;
    std::string mode = "demo";

    for (int index = 1; index < argc; ++index) {
        std::string arg = argv[index];
        if (arg == "--mode" && index + 1 < argc) {
            mode = argv[++index];
        } else if (arg == "--input" && index + 1 < argc) {
            options.input_path = argv[++index];
        } else if (arg == "--format" && index + 1 < argc) {
            options.input_format = argv[++index];
        }
    }

    if (mode != "demo") {
        std::cerr << "暂不支持的模式: " << mode << "\n";
        return 1;
    }

    self_compiler::app::CompilerApp app;
    auto status = app.Run(options, std::cout);
    if (!status.ok) {
        std::cerr << status.message << "\n";
        return 1;
    }
    return 0;
}
