# Self Compiler 项目框架文档

## 1. 项目定位

本项目的最终目标是：**做一个可以实际部署到具体硬件的 AI 编译器。**

不是演示项目，不是 PPT 原型。最终要能接收一个真实的模型文件（ONNX / Llama config），经过完整的编译流水线，生成可以在目标硬件上执行的指令。

项目把三个方向融合为一条流水线：

1. `多格式模型前端` — 导入 ONNX / JSON / Llama config，输出统一 Graph IR
2. `图优化与 Lowering` — 算子融合、Transformer Block 展开、shape 推断
3. `资源约束与代码生成` — 静态内存规划、buffer 复用、硬件指令生成

前端输入能力：

- `JsonImporter`：已实现
- `OnnxImporter`：已实现（手写 protobuf 解码器，可导入真实 .onnx 文件）
- `LlamaConfigImporter`：已实现
- 预留：`TFLite`、`StableHLO`、`MLIR`

项目核心问题：

> 输入一个真实的模型文件，如何经过前端导入、统一 IR、图优化、lowering、静态内存规划和目标硬件代码生成，最终变成可以在硬件上执行的指令流？

---

## 2. 三个方向如何融合成一个项目

### 2.1 Transformer Block Mini Compiler

负责项目的“模型前端”和“高层语义入口”：

- 输入一个简化版 Transformer Block 配置
- 或从外部模型格式导入 Transformer block 结构
- 建立高层图表示（`Graph / Tensor / Operation`）
- 将高层 `TransformerBlock` 作为 IR 的初始语义单元

### 2.2 Static Memory Planner

负责项目的“中后端资源约束处理”：

- 分析 tensor 的 producer / consumer
- 计算 live interval
- 进行静态 buffer 规划与复用
- 输出 SRAM offset / size / peak memory 等结果

### 2.3 MLIR dialect + lowering

负责项目的“编译器正规化方向”：

- 先在本项目内实现**自定义轻量 IR**
- 同时预留 MLIR bridge
- 后续可以把高层 graph 导出为自定义 dialect，或者 lower 到更低层表示

所以，三者不是并列的三个孤立项目，而是一条完整流水线的三个层面：

```text
多前端导入器
    -> 统一高层图 IR
    -> 规范化 / Lowering
    -> 静态内存规划
    -> Toy 后端命令流
    -> （可选）MLIR dialect / lowering 扩展
```

---

## 3. 项目目录结构

```text
self_compiler/
├─ CMakeLists.txt
├─ CMakePresets.json
├─ docs/
│  └─ PROJECT_FRAMEWORK.md
├─ examples/
│  └─ transformer_block.json
├─ scripts/
│  └─ configure_dev_env.ps1
├─ include/self_compiler/
│  ├─ app/
│  ├─ backend/
│  ├─ common/
│  ├─ frontend/
│  ├─ ir/
│  ├─ memory/
│  ├─ mlir/
│  └─ passes/
├─ src/
│  ├─ app/
│  ├─ backend/
│  ├─ frontend/
│  ├─ memory/
│  ├─ mlir/
│  └─ passes/
└─ tools/
   └─ self_compiler_main.cpp
```

这套结构遵循一个原则：

> 每个目录都对应编译器的一层职责，不创建没有语义价值的文件。

---

## 4. 模块职责

### 4.1 `frontend/`

职责：

- 定义 `Importer` 抽象
- 支持不同输入源导入
- 把”模型结构”翻译成”编译器可以处理的 `Graph/Op/Tensor`”

当前规划的 importer：

- `TransformerBlockSpec` 直接建图
- `JsonImporter`：已实现
- `OnnxImporter`：已实现（手写 protobuf 解码器，零外部依赖）
- `LlamaConfigImporter`：已实现
- 预留 `TFLiteImporter`
- 预留 `StableHloImporter`
- 预留 `MlirImporter`

### 4.2 `ir/`

职责：

- 定义核心 IR 抽象：`Shape`、`Tensor`、`Operation`、`Graph`
- 作为前端、pass、memory planner、backend 的公共语言
- 保证不论前端输入格式是什么，进入中端后都统一处理

### 4.3 `passes/`

职责：

- 图规范化（canonicalize）
- 高层 `TransformerBlock` 向更低层 runtime ops 的 lowering
- 为后续接入 MLIR dialect conversion 保留结构对应关系

### 4.4 `memory/`

职责：

- 分析 tensor 的 live interval
- 做静态内存规划
- 支撑 future work：greedy / best-fit / hill-climbing allocator

### 4.5 `backend/`

职责：

- 将 lower 后的 IR 映射到 toy accelerator command stream
- 输出命令序列、buffer plan、执行顺序摘要

### 4.6 `mlir/`

职责：

- 当前阶段提供 MLIR bridge 占位接口
- 后续可扩展为：
  - 自定义 dialect
  - lower 到 `func / scf / arith / memref`
  - 再 lower 到 LLVM 或 toy backend

---

## 5. 编译流水线（当前 → 最终目标）

### 当前已实现的流水线

```text
外部输入（spec / json / onnx / llama_config）
    ↓
多前端导入器（JsonImporter / OnnxImporter / LlamaConfigImporter）
    ↓
统一高层图 IR（Graph / Tensor / Operation）
    ↓
CanonicalizePass（合法性检查 + 属性规范化）
    ↓
LowerTransformerToRuntimePass（TransformerBlock → 8 个子 op）
    ↓
Live Interval Analysis（线性分析，无 reuse）
    ↓
Static Memory Planning（顺序分配，无 buffer 复用）
    ↓
Toy Backend 伪命令流（ALLOC + EXEC）
```

### 最终目标流水线（可实际部署）

```text
外部输入（ONNX / Llama config / TFLite / ...）
    ↓
多前端导入器
    ↓
统一高层图 IR
    ↓
CanonicalizePass（合法性 + 规范化）
    ↓
算子识别 Pass（kUnknown → 具体 OpKind）
    ↓
算子融合 Pass（Conv+BN+Relu → FusedOp）
    ↓
LowerTransformerToRuntimePass（TransformerBlock → 细粒度 op）
    ↓
Shape 推断 Pass（自动推导中间 tensor 的 shape）
    ↓
图分割 Pass（哪些 op 在加速器，哪些 fallback CPU）
    ↓
                    ┌─── 路径 A：自研后端 ───┐
                    │                        │
                    ↓                        │
            权重量化 + 布局转换               │
                    ↓                        │
            Live Interval Analysis           │
                    ↓                        │
            Buffer Reuse 内存规划            │
                    ↓                        │
            DMA 调度 + 指令生成              │
                    ↓                        │
            目标硬件二进制                    │
                    │                        │
                    ├─── 路径 B：MLIR ───────┤
                    │                        │
                    ↓                        │
            导出到 MLIR 自定义 dialect        │
                    ↓                        │
            MLIR lowering pipeline           │
            (func → scf → arith → memref)   │
                    ↓                        │
            LLVM IR                          │
                    ↓                        │
            LLVM 后端代码生成                │
                    ↓                        │
            目标硬件二进制                    │
                    └────────────────────────┘
```

---

## 5.1 MLIR / LLVM 在流水线中的位置

```text
┌─────────────────────────────────────────────────────┐
│                    我们的编译器                        │
│                                                      │
│  前端 → 高层图 IR → 图优化 → Lowering → 内存规划     │
│                                                      │
│         ↓ 到这里为止是我们自己写的                     │
└─────────┬───────────────────────────────────────────┘
          │
          │ 导出到 MLIR（可选路径）
          ↓
┌─────────────────────────────────────────────────────┐
│                    MLIR 层                            │
│                                                      │
│  我们的自定义 Dialect（self_compiler dialect）        │
│    ↓ dialect conversion                              │
│  标准 Dialect（func / arith / memref / scf）         │
│    ↓ lowering                                        │
│  LLVM Dialect                                        │
│                                                      │
│  MLIR 提供的能力：                                    │
│  - 统一的 Pass 基础设施（不用自己写 PassManager）     │
│  - 成熟的 Dialect 转换机制                            │
│  - 内置的 canonicalization / CSE / DCE               │
│  - 多层 IR 共存（高层 + 低层可以在同一个 module 里）  │
└─────────┬───────────────────────────────────────────┘
          │
          │ LLVM Dialect → LLVM IR
          ↓
┌─────────────────────────────────────────────────────┐
│                    LLVM 层                            │
│                                                      │
│  LLVM IR → 机器无关优化 → 指令选择 → 寄存器分配      │
│    → 指令调度 → 目标代码生成                          │
│                                                      │
│  LLVM 提供的能力：                                    │
│  - 成熟的 CPU 后端（x86 / ARM / RISC-V）            │
│  - 指令级优化（循环展开、向量化、常量折叠）           │
│  - 二进制生成（.o / .so / 可执行文件）                │
│                                                      │
│  LLVM 不提供的：                                      │
│  - 图级优化（算子融合、内存规划 → 我们自己做）        │
│  - NPU/专用加速器后端（→ 我们自己做）                │
└─────────────────────────────────────────────────────┘
```

### 两条路径的选择

| | 路径 A：自研后端 | 路径 B：经 MLIR/LLVM |
|--|----------------|---------------------|
| 适用场景 | NPU、DSP 等专用加速器 | CPU、GPU 等 LLVM 已支持的目标 |
| 优势 | 完全控制指令生成、内存布局、DMA 调度 | 复用 LLVM 成熟的优化和代码生成 |
| 劣势 | 所有后端工作都自己做 | 受限于 LLVM 支持的目标架构 |
| 工业实例 | Arm Ethos-U Vela、华为 CANN | TVM → LLVM、IREE → LLVM |

**两条路径不互斥。** 真实的 AI 编译器通常同时支持：图中大部分 op 走自研加速器后端，少量不支持的 op fallback 到 CPU（经 LLVM 生成 CPU 代码）。

---

## 6. 已完成 vs 待实现

### 已完成

- 多前端 importer 架构 + 三个真实 importer（Json / ONNX / LlamaConfig）
- 高层图 IR（Graph / Tensor / Operation）
- PassManager + CanonicalizePass（9+ 条规则）
- LowerTransformerToRuntimePass（TransformerBlock → 8 个子 op，真实 shape + 数据依赖）
- Live Interval 分析 + 静态内存顺序分配
- Toy Backend 伪命令流
- CLI 主程序 + CTest 冒烟测试
- 基于文件后缀的输入格式自动推断

### 待实现（按优先级排序）

#### 近期（补全 lowering + 图优化）

- 算子属性解析（ONNX 节点的 kernel_size、axis 等）
- 算子识别 Pass（kUnknown → 具体 OpKind）
- 权重 tensor 建模（让子 op 的 inputs 包含 weight tensor）
- Attention 细拆（matmul → scale → mask → softmax → matmul → output_proj）
- Shape 推断 Pass（自动推导缺失的中间 tensor shape）
- Lowering 后验证 Pass

#### 中期（资源优化 + 真实后端）

- Buffer reuse 内存规划（greedy / best-fit allocator）
- 算子融合 Pass（识别常见融合模式）
- 图分割 Pass（加速器 / CPU fallback 分界）
- 权重量化框架（float32 → int8）
- 真实硬件后端（替代 Toy Backend，生成可执行指令）

#### 远期（MLIR 集成 + 产品化）

- MLIR 自定义 dialect 定义与注册
- MLIR lowering pipeline（dialect → func/arith/memref → LLVM）
- LLVM 后端代码生成
- DMA 调度与计算-搬运重叠
- Runtime 集成（内存管理 / 同步 / 错误处理）

---

## 7. 开发路线

### Phase 1：编译器骨架 ✅ 已完成

- Graph / Tensor / Operation 跑通
- CLI 能从 spec / json 两类入口进入编译流水线
- PassManager 能跑 pass
- Backend 能输出伪命令流

### Phase 2：真实 importer ✅ 已完成

- JsonImporter、LlamaConfigImporter、OnnxImporter 均已实现
- OnnxImporter 使用手写 protobuf 解码器，可导入真实 .onnx 文件
- 所有 importer 输出统一落到 Graph IR

### Phase 3：Transformer Block lowering ✅ 第一版完成

- TransformerBlock 展开为 8 个子 op
- 真实 shape（qkv_out 为 [B,S,Q+K+V]）
- 残差跳跃连接、tensor 引用重建
- 待补：权重 tensor、attention 细拆、lowering 后验证

### Phase 4：图优化 Pass 补全 ← 当前阶段

- 算子识别（kUnknown → 具体 OpKind）
- Shape 推断
- 算子融合
- 图分割

### Phase 5：内存优化

- Live interval 精确分析
- Buffer reuse（greedy / best-fit）
- 多 memory region 支持

### Phase 6：真实硬件后端 — Arm Ethos-U55 NPU

- 目标硬件：Arm Ethos-U55
- 验证环境：Corstone-300 FVP（免费模拟器）
- 参考编译器：`D:\Ai_kernel\Ai_Compiler`（Vela/Regor）
- 权重量化（float32 → int8）
- 算子支持检查（Ethos-U55 支持的算子集 vs 需要 CPU fallback 的算子）
- 命令流生成（替代 ToyBackend，生成 Ethos-U55 command stream）
- DMA 调度 + SRAM 分块搬运

### Phase 7：MLIR 集成（可选保留，不在主线）

- 全自研路线已确定，MLIR 不作为主要依赖
- 保留 MLIR bridge 接口，后续可作为对照实验或学术扩展
- 如需启用：定义自定义 dialect → lowering 到标准 dialect → 导出 LLVM IR

---

## 8. 环境配置

### 8.1 当前机器已检测到的工具

当前工作目录环境检测结果：

- `cmake`：可用（4.0.0）
- `g++`：可用（8.1.0）
- `clang++`：当前不可用
- `llvm-config`：当前不可用
- `mingw32-make`：可用

这意味着：

- **核心 C++ 骨架可以直接开发和编译**
- **MLIR 真正接入目前还缺 LLVM/MLIR 环境**

### 8.2 本阶段推荐配置

#### 最低开发配置（现在就能开工）

- CMake >= 3.20
- `mingw32-make` 或 `Ninja`
- g++ / clang++ 任一可用 C++17 编译器

#### MLIR 扩展配置（后续接入）

建议按 MLIR 官方流程准备：

1. 安装 `git`、`ninja`、C++ toolchain
2. 获取 `llvm-project`
3. 以 `-DLLVM_ENABLE_PROJECTS=mlir` 构建 LLVM/MLIR
4. 在本项目配置时设置 `MLIR_DIR` / `LLVM_DIR`

### 8.3 推荐环境变量

建议你后续在本机设置：

```powershell
$env:LLVM_DIR="<your llvm install>/lib/cmake/llvm"
$env:MLIR_DIR="<your llvm install>/lib/cmake/mlir"
```

### 8.4 项目构建步骤

#### 核心骨架构建（无 MLIR）

```powershell
cd self_compiler
cmake --preset core-debug
cmake --build --preset build-core-debug
ctest --test-dir build/core-debug-mingw --output-on-failure
```

#### 运行默认示例（直接内建 spec）

```powershell
cd self_compiler
./build/core-debug-mingw/self_compiler_cli.exe --mode demo
```

#### 运行 JSON 示例输入

```powershell
cd self_compiler
./build/core-debug-mingw/self_compiler_cli.exe --mode demo --input examples/transformer_block.json
```

#### 启用 MLIR 构建（等你装好 LLVM/MLIR 后）

```powershell
cd self_compiler
$env:LLVM_DIR="<path>"
$env:MLIR_DIR="<path>"
cmake --preset mlir-debug
cmake --build --preset build-mlir-debug
```

### 8.5 快速检查脚本

项目已提供：

`scripts/configure_dev_env.ps1`

`scripts/run_utf8.ps1`

作用：

- 检查 `cmake` / `g++` / `mingw32-make` / `ninja` / `llvm-config`
- 提示当前是否具备 MLIR 开发条件
- 创建核心构建目录
- 在 Windows 下切换控制台到 UTF-8，并以 UTF-8 日志方式运行 `ctest` / `self_compiler_cli`

#### Windows 下避免中文乱码的推荐运行方式

```powershell
cd self_compiler
powershell -ExecutionPolicy Bypass -File .\scripts\run_utf8.ps1 -Mode ctest
```

```powershell
cd self_compiler
powershell -ExecutionPolicy Bypass -File .\scripts\run_utf8.ps1 -Mode cli
```

```powershell
cd self_compiler
powershell -ExecutionPolicy Bypass -File .\scripts\run_utf8.ps1 -Mode cli -InputPath ..\Llama-3.2-1B\config.json -Format llama_config
```

说明：

- 脚本会先执行 `chcp 65001`
- 同时设置 PowerShell 控制台输入/输出编码为 UTF-8
- 运行结果还会额外保存到 `build/core-debug-mingw/utf8-logs/`，便于查看 UTF-8 版本日志

---

## 9. 你在这个项目里会真正学到什么

项目做完后，你对 AI 编译器的掌握会从“知道概念名词”提升到“能自己搭一个 compiler skeleton 并持续扩展”的层级。

至少会掌握：

- 如何设计最小可用 IR
- 如何设计多 importer 架构
- 如何写 pass manager 和 pass pipeline
- 如何把高层 op lower 到低层 op
- 如何做 tensor 生命周期分析
- 如何做静态内存规划
- 如何把 IR 映射到 toy backend
- 如何为 MLIR dialect/lowering 预留工程接口

---

## 10. 你现在最适合怎么开始

你接下来直接按下面顺序写代码：

1. 跑通 `self_compiler_cli`
2. 用 `examples/transformer_block.json` 跑一遍真实 JSON 导入路径
3. 看懂 `Graph / Tensor / Operation`
4. 看懂 `Importer` 抽象和 `JsonImporter / OnnxImporter`
5. 自己把 `OnnxImporter` 接到真正的 protobuf 解析
6. 自己把 `MemoryPlanner` 从顺序分配升级成 live-range reuse
7. 最后接 MLIR bridge

这条路线最稳。
