#pragma once

#include <string>
#include <vector>

#include "self_compiler/ir/value_shape.h"

namespace self_compiler::ir {

struct Tensor {
    int id = -1;
    std::string name;
    Shape shape;
    DataType dtype = DataType::kUnknown;
    int producer_op = -1;
    std::vector<int> consumer_ops;

    // 量化参数（scale==0 表示未量化）
    float scale = 0.0f;
    std::int32_t zero_point = 0;

    // 持久化标记（KV Cache 等跨推理步骤存活的 tensor）
    bool persistent = false;
    std::int64_t max_seq_len = 0;
};

}  // namespace self_compiler::ir
