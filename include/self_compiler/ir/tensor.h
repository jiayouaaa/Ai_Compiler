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
};

}  // namespace self_compiler::ir
