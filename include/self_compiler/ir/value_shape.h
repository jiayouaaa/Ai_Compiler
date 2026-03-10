#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace self_compiler::ir {

struct Shape {
    std::vector<std::int64_t> dims;

    std::string ToString() const {
        std::ostringstream out;
        out << "[";
        for (std::size_t index = 0; index < dims.size(); ++index) {
            if (index != 0) {
                out << ", ";
            }
            out << dims[index];
        }
        out << "]";
        return out.str();
    }
};

enum class DataType {
    kBFloat16,
    kFloat32,
    kInt32,
    kUnknown,
};

inline std::string ToString(DataType type) {
    switch (type) {
        case DataType::kBFloat16:
            return "bf16";
        case DataType::kFloat32:
            return "f32";
        case DataType::kInt32:
            return "i32";
        default:
            return "unknown";
    }
}

}  // namespace self_compiler::ir
