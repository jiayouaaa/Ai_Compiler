# self_compiler

一个 C++17 编写的教学型 AI 编译器，以 Llama-3.2-1B 为目标模型，实现从模型描述到硬件命令流的完整编译流水线。

## 项目定位

本项目是一个 AI 编译器学习项目，融合三条主线：

1. **Transformer Block Mini Compiler** -- 把 Transformer 模型描述编译为可执行的命令序列
2. **Static Memory Planner** -- 用 Linear-Scan 算法为 tensor 做静态内存分配
3. **MLIR dialect/lowering** -- 预留 MLIR 集成接口（当前为 stub）

参考工程为 Arm Ethos-U NPU 的 Vela/Regor 编译器。

## 编译流水线

```
输入（JSON / Llama config / ONNX）
  |
  v
+-- 前端 (frontend/) --+
|  ImporterFactory      |  按文件后缀自动选择 importer
|  TransformerBlock     |  内建 Llama-3.2-1B demo
|  Builder              |
+-----------------------+
  |
  v  统一 Graph IR (Tensor[] + Operation[])
  |
+-- Pass 流水线 (passes/) -----+
|  1. CanonicalizePass          |  名称规范化 + 语义校验
|  2. RecognizeOnnxOpsPass      |  ONNX op 映射
|  3. ShapeInferencePass        |  shape 推导/校验（17 种 op）
|  4. LowerTransformerToRuntime |  TransformerBlock -> 15 个子 op
|  5. ShapeInferencePass        |  lowered 子 op shape 校验
|  6. CanonicalizePass          |  二次验证
|  7. GraphPartitionPass        |  标记 NPU/CPU 执行目标
+-------------------------------+
  |
  v
+-- 内存规划 (memory/) ---------+
|  LiveIntervalAnalyzer         |  tensor 活跃区间分析
|  MemoryPlanner (linear-scan)  |  FreeList + 16B 对齐
|  SRAM(256KB) / DRAM / FLASH   |  三 region 分配 + 溢出降级
+-------------------------------+
  |
  v
+-- 后端 (backend/) -----------+
|  ToyAcceleratorBackend       |  伪命令流（教学用）
|  EthosUBackend               |  Ethos-U55 NPU 寄存器命令
+------------------------------+
  |
  v
+-- MLIR (mlir/) ---------------+
|  MlirBridge (stub)            |  占位，待接入真实 dialect
+-------------------------------+
```

## Lowering 后的 Transformer Block 数据流

以 Llama-3.2-1B 默认参数（B=1, S=16, H=2048, nH=32, nKV=8, d=64）为例：

```
input [1, 16, 2048]
  -> RmsNorm -> [1, 16, 2048]
  -> QkvProject -> [1, 16, 3072]       (Q+K+V 拼接)
  -> Rope -> [1, 16, 3072]
  -> SplitQkv -> Q/K/V [1, 32, 16, 64]
  -> BatchMatMul(Q * K^T) -> [1, 32, 16, 16]   (scores)
  -> Mul(scale) -> [1, 32, 16, 16]
  -> CausalMask -> [1, 32, 16, 16]
  -> Softmax -> [1, 32, 16, 16]
  -> BatchMatMul(* V) -> [1, 32, 16, 64]       (context)
  -> Reshape -> [1, 16, 2048]
  -> Linear(Wo) -> [1, 16, 2048]
  -> ResidualAdd(+input) -> [1, 16, 2048]
  -> RmsNorm -> [1, 16, 2048]
  -> SwiGLU(gate/up/down) -> [1, 16, 2048]
  -> ResidualAdd -> output [1, 16, 2048]
```

## 目录结构

```
self_compiler/
  include/self_compiler/
    ir/                 # 核心 IR：Shape, Tensor, Operation, Graph, OpRegistry
    passes/             # Pass 接口 + 7 个 pass 实现
    memory/             # LiveInterval + MemoryPlanner
    backend/            # Toy / Ethos-U55 后端
    frontend/           # Importer 接口 + Factory
    app/                # CompilerApp 流水线整合
    common/             # Status 等公共类型
    mlir/               # MLIR bridge 接口
  src/                  # 对应的 .cpp 实现
  tools/                # CLI 入口 (self_compiler_main.cpp)
  tests/                # 单元测试 (33 个用例)
  examples/             # 示例输入文件
  scripts/              # Windows UTF-8 辅助脚本
```

## 构建

依赖：CMake 3.20+、C++17 编译器（MinGW-w64 或 MSVC）。

```bash
cd self_compiler

# 配置（MinGW）
cmake --preset core-debug

# 构建
cmake --build --preset build-core-debug

# 运行全部测试（6 个测试套件，33 个用例）
ctest --test-dir build/core-debug-mingw --output-on-failure
```

### CLI 用法

```bash
# 内建 demo（Llama-3.2-1B 默认参数）
./build/core-debug-mingw/self_compiler_cli.exe --mode demo

# 自定义 JSON 输入
./build/core-debug-mingw/self_compiler_cli.exe --mode demo --input examples/transformer_block.json

# Llama config.json 输入
./build/core-debug-mingw/self_compiler_cli.exe --mode demo --input ../Llama-3.2-1B/config.json --format llama_config
```

### Windows 中文环境

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_utf8.ps1 -Mode ctest
```

## 测试套件

| 测试 | 用例数 | 覆盖内容 |
|------|--------|---------|
| self_compiler_smoke | 1 | CLI 端到端冒烟测试 |
| shape_inference_test | 19 | 17 种 op 的 shape 推导/校验（推导模式 + 校验模式 + 错误检测） |
| memory_planner_test | 6 | linear-scan 复用、对齐、FLASH 分类、SRAM 溢出降级 |
| quantize_test | 4 | bf16/f32 -> int8 对称量化，i32 不量化 |
| tiling_test | 4 | 大/小 MatMul 分块决策，Linear 分块，非 MatMul op 不受影响 |
| kv_cache_test | 4 | KVCacheWrite/Read 注册、persistent tensor DRAM 分配、子图构建 |

## 模块详解

### IR 层 (`ir/`)

所有模块的公共语言。核心类型：

- **Shape** -- 维度向量 `vector<int64_t>`
- **Tensor** -- id、name、shape、dtype、producer/consumer 关系、量化参数（scale/zero_point）、持久化标记（persistent/max_seq_len）
- **Operation** -- id、name、OpKind（50+ 种枚举）、inputs/outputs（tensor id）、attributes（string map）
- **Graph** -- `vector<Tensor>` + `vector<Operation>`
- **OpRegistry** -- 全局注册表，每个 OpKind 的 arity、weight 描述、必须属性、NPU 支持

inputs 排列约定：**activation 在前，weight 在后**。

### Pass 流水线 (`passes/`)

| Pass | 作用 |
|------|------|
| CanonicalizePass | tensor id/dtype/shape 校验，名称规范化，属性格式化，producer/consumer 一致性 |
| RecognizeOnnxOpsPass | ONNX 通用 op -> 项目 OpKind 映射 |
| ShapeInferencePass | 双模式：空 shape 自动填入（推导），非空 shape 比对（校验） |
| LowerTransformerToRuntimePass | 1 个 TransformerBlock -> 15 个子 op（含 7 步 Attention 细拆） |
| GraphPartitionPass | 查询 OpRegistry.npu_supported，标记 exec_target=NPU/CPU |
| QuantizePass | bf16/f32 -> int8 对称量化（独立可用，未默认接入流水线） |
| TilingPass | 对 BatchMatMul/Linear 按 SRAM 容量做 M 维分块（独立可用） |

### 内存规划 (`memory/`)

- **LiveIntervalAnalyzer** -- 按 op 拓扑序分析 tensor 的 [birth, death] 区间
- **MemoryPlanner** -- Linear-Scan 算法 + FreeList（best-fit 分配 + 相邻块合并）
  - 16 字节 DMA 对齐
  - SRAM (256KB) / DRAM / FLASH 三 region
  - SRAM 超容时自动将最大 tensor 降级到 DRAM
  - persistent tensor（KV Cache）直接分配到 DRAM

### 后端 (`backend/`)

- **ToyAcceleratorBackend** -- 每个 op 翻译为 LOAD/COMPUTE/STORE 伪命令
- **EthosUBackend** -- Arm Ethos-U55 NPU 寄存器级命令（NPU_SET_xxx / NPU_OP_xxx），地址从 MemoryPlan 的 region 基址计算

### KV Cache 基础设施

为自回归推理预备的 IR 扩展（lowering 集成待完成）：

- `kKVCacheWrite` -- 把当前步的 K/V 写入 cache 的指定位置
- `kKVCacheRead` -- 从 cache 读取历史 K/V
- `Tensor::persistent` -- 标记跨推理步骤存活的 tensor
- `Tensor::max_seq_len` -- 预留的最大序列长度

## 已完成 vs 待完成

| 状态 | 内容 |
|------|------|
| done | 多前端导入（JSON / Llama config / ONNX stub） |
| done | 统一 Graph IR + OpRegistry（50+ 算子） |
| done | CanonicalizePass（完整语义校验） |
| done | ShapeInferencePass（17 种 op，推导+校验双模式） |
| done | LowerTransformerToRuntimePass（15 个子 op） |
| done | GraphPartitionPass（NPU/CPU 分区） |
| done | Linear-Scan MemoryPlanner（FreeList + 16B 对齐 + 三 region + 溢出降级） |
| done | Toy / Ethos-U55 双后端命令流 |
| done | QuantizePass 骨架（per-tensor 对称量化） |
| done | TilingPass 骨架（M 维分块） |
| done | KV Cache IR 扩展（OpKind + persistent tensor + DRAM 路由） |
| todo | KV Cache lowering 集成（SplitQkv -> KVCacheWrite -> BatchMatMul） |
| todo | MLIR dialect 实现（当前为 stub） |
| todo | TilingPass 接入主流水线 + 数据流拼接 |
| todo | QuantizePass 接入主流水线 + calibration |
| todo | Backend 抽象接口（替代硬编码双后端） |
| todo | Attention Fusion（MatMul+Scale+Mask+Softmax+MatMul -> FlashAttention） |
| todo | AutoTuning / Cost Model |

## 许可

学习项目，仅供个人学习使用。
