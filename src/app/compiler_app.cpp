#include "self_compiler/app/compiler_app.h"

#include <memory>

#include "self_compiler/backend/ethosu_backend.h"
#include "self_compiler/backend/toy_backend.h"
#include "self_compiler/frontend/importer_factory.h"
#include "self_compiler/frontend/transformer_block_builder.h"
#include "self_compiler/memory/live_interval.h"
#include "self_compiler/memory/memory_planner.h"
#include "self_compiler/mlir/mlir_bridge.h"
#include "self_compiler/passes/canonicalize_pass.h"
#include "self_compiler/passes/graph_partition_pass.h"
#include "self_compiler/passes/lower_transformer_to_runtime_pass.h"
#include "self_compiler/passes/pass_manager.h"
#include "self_compiler/passes/recognize_onnx_ops_pass.h"

namespace self_compiler::app {

self_compiler::Status CompilerApp::Run(const RunOptions& options, std::ostream& out) const {
    ir::Graph graph;

    if (options.input_path.empty()) {
        frontend::TransformerBlockSpec spec;
        frontend::TransformerBlockBuilder builder;
        graph = builder.Build(spec);
    } else {
        frontend::InputFormat format = frontend::InputFormat::kUnknown;
        if (!options.input_format.empty() && options.input_format != "auto") {
            format = frontend::ImporterFactory::ParseFormat(options.input_format);
        }
        if (format == frontend::InputFormat::kUnknown) {
            format = frontend::ImporterFactory::InferFormatFromPath(options.input_path);
        }
        if (format == frontend::InputFormat::kUnknown) {
            return self_compiler::Status::Error("无法识别输入格式。请显式传入 --format，或使用带扩展名的输入文件。")
                ;
        }
        if (format == frontend::InputFormat::kSpec) {
            frontend::TransformerBlockSpec spec;
            frontend::TransformerBlockBuilder builder;
            graph = builder.Build(spec);
        } else {
            auto importer = frontend::ImporterFactory::Create(format);
            if (!importer) {
                return self_compiler::Status::Error("当前版本尚未实现该输入格式的 importer: " + frontend::ToString(format));
            }
            auto import_status = importer->Import(options.input_path, graph);
            if (!import_status.ok) {
                return import_status;
            }
        }
    }

    if (options.dump_graph) {
        out << "==== 高层图 ====\n";
        graph.Dump(out);
    }

    passes::PassManager pass_manager;
    pass_manager.AddPass(std::make_unique<passes::CanonicalizePass>());
    pass_manager.AddPass(std::make_unique<passes::RecognizeOnnxOpsPass>());
    pass_manager.AddPass(std::make_unique<passes::LowerTransformerToRuntimePass>());
    pass_manager.AddPass(std::make_unique<passes::CanonicalizePass>());  // lowering 后再验证一次
    pass_manager.AddPass(std::make_unique<passes::GraphPartitionPass>());  // 标记每个 op 的执行目标
    auto pass_status = pass_manager.Run(graph);
    if (!pass_status.ok) {
        return pass_status;
    }

    out << "\n==== Lowering 之后的图 ====\n";
    graph.Dump(out);

    memory::LiveIntervalAnalyzer analyzer;
    auto intervals = analyzer.Analyze(graph);
    memory::MemoryPlanner planner;
    auto plan = planner.BuildPlan(graph, intervals);

    out << "\n==== 静态内存规划 ====\n";
    out << "total_bytes=" << plan.total_bytes << " peak_bytes=" << plan.peak_bytes << "\n";
    out << "  SRAM:  " << plan.sram_bytes << " bytes\n";
    out << "  DRAM:  " << plan.dram_bytes << " bytes\n";
    out << "  FLASH: " << plan.flash_bytes << " bytes\n";
    if (plan.sram_spill_count > 0) {
        out << "  [警告] " << plan.sram_spill_count
            << " 个 tensor 因 SRAM 容量不足被降级到 DRAM\n";
    }
    for (const auto& alloc : plan.allocations) {
        out << "  " << alloc.tensor_name << " " << memory::ToString(alloc.region)
            << " offset=" << alloc.offset
            << " size=" << alloc.size_in_bytes << "\n";
    }

    backend::ToyAcceleratorBackend toy_backend;
    auto command_stream = toy_backend.Emit(graph, plan);
    if (options.dump_command_stream) {
        out << "\n==== Toy Backend 命令流 ====\n";
        command_stream.Dump(out);
    }

    // Ethos-U55 NPU 命令流
    backend::EthosUBackend ethosu_backend;
    auto ethosu_stream = ethosu_backend.Emit(graph, plan);
    out << "\n==== Ethos-U55 命令流 ====\n";
    ethosu_stream.Dump(out);

    if (options.dump_mlir_stub) {
        mlir::MlirBridge bridge;
        out << "\n==== MLIR Bridge 占位输出 ====\n";
        bridge.ExportDialectSkeleton(graph, out);
    }

    return self_compiler::Status::Ok();
}

}  // namespace self_compiler::app
