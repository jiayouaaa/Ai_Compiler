#include "self_compiler/memory/live_interval.h"

namespace self_compiler::memory {

namespace {

std::size_t EstimateTensorBytes(const ir::Tensor& tensor) {
    std::size_t element_size = 2;  // bf16 默认
    if (tensor.dtype == ir::DataType::kFloat32 || tensor.dtype == ir::DataType::kInt32) {
        element_size = 4;
    } else if (tensor.dtype == ir::DataType::kInt8) {
        element_size = 1;
    }

    std::size_t elements = 1;
    for (auto dim : tensor.shape.dims) {
        elements *= static_cast<std::size_t>(dim > 0 ? dim : 1);
    }
    return elements * element_size;
}

}  // namespace

std::vector<LiveInterval> LiveIntervalAnalyzer::Analyze(const ir::Graph& graph) const {
    std::vector<LiveInterval> intervals;
    intervals.reserve(graph.tensors().size());

    for (const auto& tensor : graph.tensors()) {
        LiveInterval interval;
        interval.tensor_id = tensor.id;
        interval.tensor_name = tensor.name;
        interval.start_op_index = tensor.producer_op < 0 ? 0 : tensor.producer_op;
        interval.end_op_index = interval.start_op_index;
        for (int consumer : tensor.consumer_ops) {
            if (consumer > interval.end_op_index) {
                interval.end_op_index = consumer;
            }
        }
        interval.bytes = EstimateTensorBytes(tensor);
        intervals.push_back(interval);
    }

    return intervals;
}

}  // namespace self_compiler::memory
