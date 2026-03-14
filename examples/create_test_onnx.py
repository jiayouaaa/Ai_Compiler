#!/usr/bin/env python3
"""
生成一个简单的 ONNX 测试模型，用于验证 OnnxImporter。

模型结构（3 层前馈网络）：
  input [1, 10]
    -> MatMul(input, w1) -> hidden1 [1, 20]
    -> Relu(hidden1)     -> hidden2 [1, 20]
    -> MatMul(hidden2, w2) -> hidden3 [1, 5]
    -> Add(hidden3, b2)    -> output [1, 5]

用法：
  pip install onnx numpy
  python create_test_onnx.py
"""

import sys

try:
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("需要安装依赖：pip install onnx numpy")
    sys.exit(1)

# ---- 模型输入 ----
X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])

# ---- 权重（initializer）----
w1 = numpy_helper.from_array(
    np.random.randn(10, 20).astype(np.float32), name="w1")
w2 = numpy_helper.from_array(
    np.random.randn(20, 5).astype(np.float32), name="w2")
b2 = numpy_helper.from_array(
    np.zeros(5).astype(np.float32), name="b2")

# ---- 计算节点 ----
matmul1 = helper.make_node("MatMul", ["input", "w1"], ["hidden1"], name="matmul1")
relu1 = helper.make_node("Relu", ["hidden1"], ["hidden2"], name="relu1")
matmul2 = helper.make_node("MatMul", ["hidden2", "w2"], ["hidden3"], name="matmul2")
add1 = helper.make_node("Add", ["hidden3", "b2"], ["output"], name="add1")

# ---- 模型输出 ----
Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])

# ---- 中间 tensor 的类型信息 ----
hidden1_info = helper.make_tensor_value_info("hidden1", TensorProto.FLOAT, [1, 20])
hidden2_info = helper.make_tensor_value_info("hidden2", TensorProto.FLOAT, [1, 20])
hidden3_info = helper.make_tensor_value_info("hidden3", TensorProto.FLOAT, [1, 5])

# ---- 组装图和模型 ----
graph = helper.make_graph(
    [matmul1, relu1, matmul2, add1],
    "test_feedforward",
    [X], [Y],
    initializer=[w1, w2, b2],
    value_info=[hidden1_info, hidden2_info, hidden3_info],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7

output_path = "test_model.onnx"
onnx.save(model, output_path)
print(f"已生成 {output_path}（{len(open(output_path, 'rb').read())} 字节）")
