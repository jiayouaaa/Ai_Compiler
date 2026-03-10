#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "self_compiler/ir/operation.h"
#include "self_compiler/ir/tensor.h"

namespace self_compiler::ir {

class Graph {
public:
    Tensor& AddTensor(const std::string& name, const Shape& shape, DataType dtype);
    Operation& AddOperation(const std::string& name, OpKind kind, const std::vector<int>& inputs, const std::vector<int>& outputs);

    std::vector<Tensor>& tensors() { return tensors_; }
    const std::vector<Tensor>& tensors() const { return tensors_; }

    std::vector<Operation>& operations() { return operations_; }
    const std::vector<Operation>& operations() const { return operations_; }

    Tensor* FindTensor(int id);
    const Tensor* FindTensor(int id) const;

    void Dump(std::ostream& out) const;

private:
    std::vector<Tensor> tensors_;
    std::vector<Operation> operations_;
};

inline Tensor& Graph::AddTensor(const std::string& name, const Shape& shape, DataType dtype) {
    Tensor tensor;
    tensor.id = static_cast<int>(tensors_.size());
    tensor.name = name;
    tensor.shape = shape;
    tensor.dtype = dtype;
    tensors_.push_back(tensor);
    return tensors_.back();
}

inline Operation& Graph::AddOperation(const std::string& name, OpKind kind, const std::vector<int>& inputs, const std::vector<int>& outputs) {
    Operation op;
    op.id = static_cast<int>(operations_.size());
    op.name = name;
    op.kind = kind;
    op.inputs = inputs;
    op.outputs = outputs;
    operations_.push_back(op);

    for (int tensor_id : outputs) {
        tensors_[tensor_id].producer_op = op.id;
    }
    for (int tensor_id : inputs) {
        tensors_[tensor_id].consumer_ops.push_back(op.id);
    }
    return operations_.back();
}

inline Tensor* Graph::FindTensor(int id) {
    if (id < 0 || id >= static_cast<int>(tensors_.size())) {
        return nullptr;
    }
    return &tensors_[id];
}

inline const Tensor* Graph::FindTensor(int id) const {
    if (id < 0 || id >= static_cast<int>(tensors_.size())) {
        return nullptr;
    }
    return &tensors_[id];
}

inline void Graph::Dump(std::ostream& out) const {
    out << "图结构导出\n";
    out << "张量列表：\n";
    for (const auto& tensor : tensors_) {
        out << "  [T" << tensor.id << "] " << tensor.name << " shape=" << tensor.shape.ToString()
            << " dtype=" << ToString(tensor.dtype) << "\n";
    }
    out << "算子列表：\n";
    for (const auto& op : operations_) {
        out << "  [O" << op.id << "] " << op.name << " kind=" << ToString(op.kind) << " inputs=";
        for (int value : op.inputs) {
            out << " T" << value;
        }
        out << " outputs=";
        for (int value : op.outputs) {
            out << " T" << value;
        }
        out << "\n";
    }
}

}  // namespace self_compiler::ir
