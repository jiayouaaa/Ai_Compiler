#include "self_compiler/passes/graph_partition_pass.h"

#include "self_compiler/ir/op_registry.h"

namespace self_compiler::passes {

std::string GraphPartitionPass::name() const {
    return "GraphPartitionPass";
}

self_compiler::Status GraphPartitionPass::Run(ir::Graph& graph) {
    const auto& registry = ir::OpRegistry::Instance();

    for (auto& op : graph.operations()) {
        // Input / Output 是图边界，不在任何硬件上执行
        if (op.kind == ir::OpKind::kInput || op.kind == ir::OpKind::kOutput) {
            op.attributes["exec_target"] = "HOST";
            continue;
        }

        const auto* info = registry.Find(op.kind);

        if (info && info->npu_supported) {
            op.attributes["exec_target"] = "NPU";
        } else {
            op.attributes["exec_target"] = "CPU";
        }
    }

    return self_compiler::Status::Ok();
}

}  // namespace self_compiler::passes
