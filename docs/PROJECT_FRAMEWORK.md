# Self Compiler 项目框架文档

## 1. 项目定位

本项目把以下三个方向融合为**一个完整 AI 编译器项目**：

1. `Transformer Block Mini Compiler`
2. `Static Memory Planner`
3. `MLIR dialect + lowering`

同时，项目的前端输入能力升级为：

- 自定义模型 `JSON`
- `ONNX`
- 后续预留：`TFLite`、`StableHLO`、`MLIR`

最终目标不是做一个“算子演示玩具”，而是做一个**有清晰编译流水线、可扩展、可演示、可继续深挖**的 AI 编译器原型。

项目核心问题：

> 输入一个简化版 Transformer Block 或外部模型表示，如何经过前端导入、统一 IR、图优化、lowering、静态内存规划和 toy backend codegen，最终变成可执行的伪命令流？

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
- 把“模型结构”翻译成“编译器可以处理的 `Graph/Op/Tensor`”

当前规划的 importer：

- `TransformerBlockSpec` 直接建图
- `JsonImporter`
- `OnnxImporter`
- 预留 `TFLiteImporter`
- 预留 `StableHloImporter`
- 预留 `MlirImporter`

当前实现状态：

- `JsonImporter`：**已实现最小可用版本**，支持从平铺 JSON 读取 `batch / sequence_length / hidden_size / intermediate_size / num_attention_heads / num_key_value_heads / vocab_size`
- `OnnxImporter`：**已建立接口，但当前明确返回“未接入真实 ONNX 解析”**，避免伪造编译结果误导开发
- `TFLite / StableHLO / MLIR`：仅保留扩展点

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

## 5. 编译流水线

```text
外部输入（spec / json / onnx / ...）
    ↓
导入器
    ↓
统一高层图 IR
    ↓
Canonicalize Pass
    ↓
Lower TransformerBlock -> Runtime Ops
    ↓
Live Interval Analysis
    ↓
Static Memory Planning
    ↓
Toy 后端命令流
    ↓
报告导出 / 图导出 / （未来）MLIR 导出
```

---

## 6. 当前阶段功能边界

### 本阶段一定实现

- 多前端 importer 架构
- `TransformerBlockSpec` 直接建图
- `JsonImporter` 最小可用实现
- `OnnxImporter` 接口与错误边界
- 高层图 IR
- `PassManager`
- 至少 1 个 canonicalize pass
- 至少 1 个 lowering pass
- live interval 分析框架
- 静态内存规划框架
- toy backend 命令流框架
- CLI 主程序
- 基于文件后缀的输入格式自动推断

### 本阶段保留伪代码 / 待你实现的部分

- 高质量 rewrite 规则集合
- 完整 attention lowering 细节
- 完整 buffer reuse 最优算法
- 真正 ONNX protobuf 导入
- 真正 TFLite / StableHLO / MLIR importer
- 真正 MLIR dialect 定义与注册
- 真正 LLVM/目标硬件 codegen

也就是说：

> **这不是演示 PPT 项目，而是“骨架真实、关键逻辑留给你亲手实现”的编译器训练项目。**

---

## 7. 开发顺序建议

### Phase 1：先把编译器骨架跑通

- `Graph / Tensor / Operation` 跑通
- CLI 能从 `spec / json` 两类真实入口进入编译流水线
- `PassManager` 能跑 pass
- backend 能输出伪命令流

### Phase 2：补真实 importer

- 先继续增强 `JsonImporter`
- 再补 `OnnxImporter`
- importer 输出必须统一落到 `Graph IR`

### Phase 3：补 Transformer Block lowering

- 把高层 `TransformerBlock` 展开成：
  - `RMSNorm`
  - `QKV projections`
  - `RoPE`
  - `Attention`
  - `ResidualAdd`
  - `SwiGLU`
  - `LMHead`（可选不在 block 内）

### Phase 4：补 Memory Planner

- 先实现最简单的线性分配
- 再实现 live interval
- 再做 buffer reuse
- 再考虑多 memory region

### Phase 5：补 MLIR bridge

- 先把 IR dump 成类似 MLIR 的 textual form
- 再真正接 MLIR dialect / conversion

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

作用：

- 检查 `cmake` / `g++` / `mingw32-make` / `ninja` / `llvm-config`
- 提示当前是否具备 MLIR 开发条件
- 创建核心构建目录

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
