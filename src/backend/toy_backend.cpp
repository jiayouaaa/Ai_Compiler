#include "self_compiler/backend/toy_backend.h"

namespace self_compiler::backend {

CommandStream ToyAcceleratorBackend::Emit(const ir::Graph& graph, const memory::MemoryPlan& plan) const {
    CommandStream stream;

    for (const auto& alloc : plan.allocations) {
        stream.Add({"ALLOC", {
            alloc.tensor_name,
            memory::ToString(alloc.region),
            std::to_string(alloc.offset),
            std::to_string(alloc.size_in_bytes)}});
    }

    for (const auto& op : graph.operations()) {
        stream.Add({"EXEC", {op.name, ir::ToString(op.kind)}});
    }

    // 伪代码：后续请用真实 backend 映射逻辑替换这里
    // 将每个 lowering 后的算子映射成 toy accelerator 指令序列
    // 当张量跨内存区域时插入 DMA / load / store 命令
    // 按依赖顺序序列化命令流

    return stream;
}

}  // namespace self_compiler::backend
