#include "self_compiler/frontend/onnx_importer.h"

#include <fstream>

namespace self_compiler::frontend {

self_compiler::Status OnnxImporter::Import(const std::string& input_path, ir::Graph& graph) const {
    std::ifstream input(input_path, std::ios::binary);
    if (!input.is_open()) {
        return self_compiler::Status::Error("无法打开 ONNX 输入文件: " + input_path);
    }

    graph = ir::Graph();
    return self_compiler::Status::Error(
        "当前版本已经预留 ONNX importer 接口，但还没有接入真正的 protobuf/ONNX 解析逻辑。"
        "你可以先使用 JSON importer 跑通完整编译链路，后续再把 ONNX 节点映射到统一 Graph IR。");
}

}  // namespace self_compiler::frontend
