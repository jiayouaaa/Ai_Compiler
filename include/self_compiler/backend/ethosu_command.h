#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace self_compiler::backend {

// Ethos-U55 NPU 命令类型
// 编号与 Arm Ethos-U55 TRM 及 ethos_u55_regs.py 一致
enum class EthosUOpCode : uint16_t {
    // ---- 执行命令（cmd0，非阻塞）----
    NPU_OP_STOP         = 0x000,  // 停止 NPU，触发中断通知 CPU
    NPU_OP_IRQ          = 0x001,  // 触发中断（不停止）
    NPU_OP_CONV         = 0x002,  // 卷积 / 全连接
    NPU_OP_DEPTHWISE    = 0x003,  // 深度可分离卷积
    NPU_OP_POOL         = 0x005,  // 池化
    NPU_OP_ELEMENTWISE  = 0x006,  // 逐元素运算（Add/Mul/Relu/Softmax 等）
    NPU_OP_DMA_START    = 0x010,  // 启动 DMA 搬运
    NPU_OP_DMA_WAIT     = 0x011,  // 等待 DMA 完成
    NPU_OP_KERNEL_WAIT  = 0x012,  // 等待上一个 kernel 完成
};

// 一条 Ethos-U55 命令
// 对应 TRM 中的 cmd0（无 payload）或 cmd1（有 payload）格式
struct EthosUCommand {
    enum class Kind {
        kNpuOp,         // NPU 执行命令（NPU_OP_CONV / NPU_OP_POOL 等）
        kSetRegister,   // NPU 配置命令（NPU_SET_IFM_BASE / NPU_SET_WEIGHT_BASE 等）
        kDma,           // DMA 搬运命令（DMA_START + DMA_WAIT）
        kCpuFallback,   // CPU 回退标记（非 NPU 命令，由 runtime 处理）
    };

    Kind kind = Kind::kNpuOp;
    EthosUOpCode opcode = EthosUOpCode::NPU_OP_STOP;
    std::string register_name;      // 配置命令的寄存器名（如 "IFM_BASE0"）
    uint32_t value = 0;             // 配置值或地址
    std::string comment;            // 人类可读的注释
};

// Ethos-U55 命令流
struct EthosUCommandStream {
    std::vector<EthosUCommand> commands;

    void Dump(std::ostream& out) const {
        for (const auto& cmd : commands) {
            switch (cmd.kind) {
                case EthosUCommand::Kind::kNpuOp:
                    out << OpCodeToString(cmd.opcode);
                    break;
                case EthosUCommand::Kind::kSetRegister:
                    out << "NPU_SET_" << cmd.register_name
                        << "  0x" << std::hex << cmd.value << std::dec;
                    break;
                case EthosUCommand::Kind::kDma:
                    out << OpCodeToString(cmd.opcode);
                    break;
                case EthosUCommand::Kind::kCpuFallback:
                    out << "CPU_FALLBACK";
                    break;
            }
            if (!cmd.comment.empty()) {
                out << "  ; " << cmd.comment;
            }
            out << "\n";
        }
    }

    static const char* OpCodeToString(EthosUOpCode code) {
        switch (code) {
            case EthosUOpCode::NPU_OP_STOP:        return "NPU_OP_STOP";
            case EthosUOpCode::NPU_OP_IRQ:         return "NPU_OP_IRQ";
            case EthosUOpCode::NPU_OP_CONV:        return "NPU_OP_CONV";
            case EthosUOpCode::NPU_OP_DEPTHWISE:   return "NPU_OP_DEPTHWISE";
            case EthosUOpCode::NPU_OP_POOL:        return "NPU_OP_POOL";
            case EthosUOpCode::NPU_OP_ELEMENTWISE:  return "NPU_OP_ELEMENTWISE";
            case EthosUOpCode::NPU_OP_DMA_START:   return "NPU_OP_DMA_START";
            case EthosUOpCode::NPU_OP_DMA_WAIT:    return "NPU_OP_DMA_WAIT";
            case EthosUOpCode::NPU_OP_KERNEL_WAIT: return "NPU_OP_KERNEL_WAIT";
            default:                                return "UNKNOWN";
        }
    }
};

}  // namespace self_compiler::backend
