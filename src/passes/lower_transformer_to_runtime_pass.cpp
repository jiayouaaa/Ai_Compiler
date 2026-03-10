#include "self_compiler/passes/lower_transformer_to_runtime_pass.h"

namespace self_compiler::passes {

std::string LowerTransformerToRuntimePass::name() const {
    return "LowerTransformerToRuntimePass";
}

self_compiler::Status LowerTransformerToRuntimePass::Run(ir::Graph& graph) {
    std::vector<ir::Operation> lowered;
    lowered.reserve(graph.operations().size() + 16);

    for (const auto& op : graph.operations()) {
        if (op.kind != ir::OpKind::kTransformerBlock) {
            lowered.push_back(op);
            continue;
        }

        ir::Operation norm0 = op;
        norm0.name = op.name + ".rmsnorm0";
        norm0.kind = ir::OpKind::kRmsNorm;

        ir::Operation qkv = op;
        qkv.name = op.name + ".qkv";
        qkv.kind = ir::OpKind::kQkvProject;

        ir::Operation rope = op;
        rope.name = op.name + ".rope";
        rope.kind = ir::OpKind::kRope;

        ir::Operation attention = op;
        attention.name = op.name + ".attention";
        attention.kind = ir::OpKind::kAttention;

        ir::Operation add0 = op;
        add0.name = op.name + ".residual0";
        add0.kind = ir::OpKind::kResidualAdd;

        ir::Operation norm1 = op;
        norm1.name = op.name + ".rmsnorm1";
        norm1.kind = ir::OpKind::kRmsNorm;

        ir::Operation mlp = op;
        mlp.name = op.name + ".swiglu";
        mlp.kind = ir::OpKind::kSwiGLU;

        ir::Operation add1 = op;
        add1.name = op.name + ".residual1";
        add1.kind = ir::OpKind::kResidualAdd;

        lowered.push_back(norm0);
        lowered.push_back(qkv);
        lowered.push_back(rope);
        lowered.push_back(attention);
        lowered.push_back(add0);
        lowered.push_back(norm1);
        lowered.push_back(mlp);
        lowered.push_back(add1);

        // 伪代码：后续请用真实 lowering 逻辑替换这里
        // 如果当前高层算子是 TransformerBlock：
        //   将其展开为：
        //     rmsnorm
        //     q/k/v 投影
        //     rope 位置编码
        //     attention 打分 / softmax / 上下文聚合
        //     输出投影
        //     残差相加
        //     rmsnorm
        //     gate/up/down 线性层
        //     swiglu 非线性门控
        //     残差相加
        //   用新的中间张量重建数据依赖关系
    }

    for (std::size_t index = 0; index < lowered.size(); ++index) {
        lowered[index].id = static_cast<int>(index);
    }
    graph.operations() = lowered;
    return self_compiler::Status::Ok();
}

}  // namespace self_compiler::passes
