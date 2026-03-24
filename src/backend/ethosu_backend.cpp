#include "self_compiler/backend/ethosu_backend.h"

#include <string>

#include "self_compiler/ir/op_registry.h"
#include "self_compiler/ir/operation.h"
#include "self_compiler/memory/memory_planner.h"

namespace self_compiler::backend {

namespace {

// 根据 OpKind 选择 NPU 执行命令
// Conv/FullyConnected/Linear → NPU_OP_CONV（全连接等价于 1×1 卷积）
// Depthwise → NPU_OP_DEPTHWISE
// MaxPool/AvgPool → NPU_OP_POOL
// 其余 NPU 支持的算子 → NPU_OP_ELEMENTWISE
EthosUOpCode SelectOpCode(ir::OpKind kind) {
    switch (kind) {
        case ir::OpKind::kConv2D:
        case ir::OpKind::kFullyConnected:
        case ir::OpKind::kLinear:
            return EthosUOpCode::NPU_OP_CONV;
        case ir::OpKind::kDepthwiseConv2D:
            return EthosUOpCode::NPU_OP_DEPTHWISE;
        case ir::OpKind::kMaxPool:
        case ir::OpKind::kAvgPool:
            return EthosUOpCode::NPU_OP_POOL;
        default:
            return EthosUOpCode::NPU_OP_ELEMENTWISE;
    }
}

// 查找某个 tensor 在 MemoryPlan 中的分配信息
const memory::BufferAllocation* FindAlloc(
    const memory::MemoryPlan& plan, int tensor_id) {
    for (const auto& alloc : plan.allocations) {
        if (alloc.tensor_id == tensor_id) {
            return &alloc;
        }
    }
    return nullptr;
}

// 根据内存区域返回基地址前缀（模拟硬件地址映射）
// 真实硬件上这些是由链接脚本和 DTS 决定的，这里用固定值模拟
uint32_t RegionBaseAddress(memory::MemoryRegion region) {
    switch (region) {
        case memory::MemoryRegion::kSram:
            return 0x20000000;   // 典型的 Cortex-M SRAM 起始地址
        case memory::MemoryRegion::kDram:
            return 0x60000000;   // 外部 DRAM 起始地址
        case memory::MemoryRegion::kFlash:
            return 0x08000000;   // Flash 起始地址
        default:
            return 0x00000000;
    }
}

// 计算 tensor 的硬件地址 = 区域基地址 + offset
uint32_t TensorAddress(const memory::BufferAllocation& alloc) {
    return RegionBaseAddress(alloc.region)
         + static_cast<uint32_t>(alloc.offset);
}

// 生成 NPU 配置命令（NPU_SET_XXX）
EthosUCommand MakeSetCmd(
    const std::string& register_name,
    uint32_t value,
    const std::string& comment) {
    EthosUCommand cmd;
    cmd.kind = EthosUCommand::Kind::kSetRegister;
    cmd.register_name = register_name;
    cmd.value = value;
    cmd.comment = comment;
    return cmd;
}

// 生成 DMA 命令
EthosUCommand MakeDmaCmd(EthosUOpCode opcode, const std::string& comment) {
    EthosUCommand cmd;
    cmd.kind = EthosUCommand::Kind::kDma;
    cmd.opcode = opcode;
    cmd.comment = comment;
    return cmd;
}

// 生成 NPU 执行命令
EthosUCommand MakeOpCmd(EthosUOpCode opcode, const std::string& comment) {
    EthosUCommand cmd;
    cmd.kind = EthosUCommand::Kind::kNpuOp;
    cmd.opcode = opcode;
    cmd.comment = comment;
    return cmd;
}

// 生成 CPU 回退标记
EthosUCommand MakeCpuFallback(const std::string& comment) {
    EthosUCommand cmd;
    cmd.kind = EthosUCommand::Kind::kCpuFallback;
    cmd.comment = comment;
    return cmd;
}

}  // namespace

EthosUCommandStream EthosUBackend::Emit(
    const ir::Graph& graph,
    const memory::MemoryPlan& plan) const {

    EthosUCommandStream stream;
    const auto& registry = ir::OpRegistry::Instance();

    for (const auto& op : graph.operations()) {
        // 跳过图边界
        if (op.kind == ir::OpKind::kInput ||
            op.kind == ir::OpKind::kOutput) {
            continue;
        }

        // 检查执行目标（由 GraphPartitionPass 标记）
        auto target_it = op.attributes.find("exec_target");
        bool is_npu = target_it != op.attributes.end() &&
                      target_it->second == "NPU";

        if (!is_npu) {
            // CPU fallback：生成标记，由 runtime 处理
            stream.commands.push_back(
                MakeCpuFallback(op.name + " (" + ir::ToString(op.kind) + ")"));
            continue;
        }

        // ---- 以下为 NPU 支持的算子 ----

        const auto* info = registry.Find(op.kind);

        // 如果算子有 weight 输入，先 DMA 搬运 weight 到 SRAM
        if (info && !info->weight_inputs.empty()) {
            // weight 输入在 op.inputs 的后半段（activation 之后）
            int weight_start = info->num_activation_inputs;
            for (int i = weight_start; i < static_cast<int>(op.inputs.size()); ++i) {
                const auto* walloc = FindAlloc(plan, op.inputs[i]);
                if (walloc && walloc->region == memory::MemoryRegion::kFlash) {
                    uint32_t src_addr = TensorAddress(*walloc);
                    stream.commands.push_back(
                        MakeSetCmd("DMA0_SRC", src_addr,
                            walloc->tensor_name + " FLASH→SRAM"));
                    stream.commands.push_back(
                        MakeSetCmd("DMA0_LEN",
                            static_cast<uint32_t>(walloc->size_in_bytes), ""));
                    stream.commands.push_back(
                        MakeDmaCmd(EthosUOpCode::NPU_OP_DMA_START,
                            "搬运 " + walloc->tensor_name));
                    stream.commands.push_back(
                        MakeDmaCmd(EthosUOpCode::NPU_OP_DMA_WAIT, ""));
                }
            }
        }

        // 配置 IFM（第一个 activation 输入）
        if (!op.inputs.empty()) {
            const auto* ifm_alloc = FindAlloc(plan, op.inputs[0]);
            if (ifm_alloc) {
                stream.commands.push_back(
                    MakeSetCmd("IFM_BASE0", TensorAddress(*ifm_alloc),
                        ifm_alloc->tensor_name));
                stream.commands.push_back(
                    MakeSetCmd("IFM_SIZE",
                        static_cast<uint32_t>(ifm_alloc->size_in_bytes), ""));
            }
        }

        // 配置 IFM2（第二个 activation 输入，如 ResidualAdd 的两路输入）
        if (info && info->num_activation_inputs >= 2 && op.inputs.size() >= 2) {
            const auto* ifm2_alloc = FindAlloc(plan, op.inputs[1]);
            if (ifm2_alloc) {
                stream.commands.push_back(
                    MakeSetCmd("IFM2_BASE0", TensorAddress(*ifm2_alloc),
                        ifm2_alloc->tensor_name));
            }
        }

        // 配置 OFM（输出）
        if (!op.outputs.empty()) {
            const auto* ofm_alloc = FindAlloc(plan, op.outputs[0]);
            if (ofm_alloc) {
                stream.commands.push_back(
                    MakeSetCmd("OFM_BASE0", TensorAddress(*ofm_alloc),
                        ofm_alloc->tensor_name));
                stream.commands.push_back(
                    MakeSetCmd("OFM_SIZE",
                        static_cast<uint32_t>(ofm_alloc->size_in_bytes), ""));
            }
        }

        // 配置 weight 地址（如果有）
        if (info && !info->weight_inputs.empty()) {
            int weight_start = info->num_activation_inputs;
            if (weight_start < static_cast<int>(op.inputs.size())) {
                const auto* walloc = FindAlloc(plan, op.inputs[weight_start]);
                if (walloc) {
                    stream.commands.push_back(
                        MakeSetCmd("WEIGHT_BASE", TensorAddress(*walloc),
                            walloc->tensor_name));
                    stream.commands.push_back(
                        MakeSetCmd("WEIGHT_LENGTH",
                            static_cast<uint32_t>(walloc->size_in_bytes), ""));
                }
            }
        }

        // 设置块依赖（简化：所有 op 使用 BLOCKDEP=0，即等上一个完成）
        stream.commands.push_back(MakeSetCmd("BLOCKDEP", 0, ""));

        // 发出执行命令
        EthosUOpCode opcode = SelectOpCode(op.kind);
        stream.commands.push_back(
            MakeOpCmd(opcode, op.name));
    }

    // 命令流末尾：停止 NPU
    stream.commands.push_back(
        MakeOpCmd(EthosUOpCode::NPU_OP_STOP, "命令流结束"));

    return stream;
}

}  // namespace self_compiler::backend
