#include "self_compiler/passes/pass_manager.h"

namespace self_compiler::passes {

void PassManager::AddPass(std::unique_ptr<Pass> pass) {
    passes_.push_back(std::move(pass));
}

self_compiler::Status PassManager::Run(ir::Graph& graph) const {
    for (const auto& pass : passes_) {
        auto status = pass->Run(graph);
        if (!status.ok) {
            return self_compiler::Status::Error(pass->name() + " failed: " + status.message);
        }
    }
    return self_compiler::Status::Ok();
}

}  // namespace self_compiler::passes
