# Corstone-300 FVP 环境搭建与 Hello World 实践笔记

本文档记录从零开始在 WSL2 Ubuntu 上搭建 Ethos-U55 NPU 仿真环境、
编译并在 FVP 上运行 Cortex-M55 裸机 Hello World 的完整过程，
以及过程中的疑问与答疑。

日期：2026-04-18
目标硬件：Corstone-300 (Cortex-M55 + Ethos-U55)
仿真器：Arm Fast Model FVP Corstone-300 11.27.42
目的：为 self_compiler 后续在 FVP 上运行编译产物打基础

---

## 目录

- [第一部分：环境搭建流程](#第一部分环境搭建流程)
  - [1. WSL2 验证](#1-wsl2-验证)
  - [2. 基础开发工具](#2-基础开发工具)
  - [3. Arm GNU Toolchain 14.2 升级](#3-arm-gnu-toolchain-142-升级)
  - [4. 克隆运行时仓库](#4-克隆运行时仓库)
  - [5. 下载安装 FVP](#5-下载安装-fvp)
  - [6. 处理 libpython3.9 依赖](#6-处理-libpython39-依赖)
  - [7. 配置 .bashrc 环境变量](#7-配置-bashrc-环境变量)
- [第二部分：Hello World 文件详解](#第二部分hello-world-文件详解)
  - [startup.c 逐行讲解](#startupc-逐行讲解)
  - [hello.c 逐行讲解](#helloc-逐行讲解)
  - [corstone300.ld 链接脚本语法](#corstone300ld-链接脚本语法)
- [第三部分：概念澄清问答](#第三部分概念澄清问答)
  - [Q1. 为什么 startup.c 一定是第一个运行？](#q1-为什么-startupc-一定是第一个运行)
  - [Q2. 为什么 `> ITCM` 就是"第一个放进 ITCM 的段"？](#q2-为什么--itcm-就是第一个放进-itcm-的段)
  - [Q3. `AT > ITCM` 里 ITCM 就是 ROM 吗？](#q3-at--itcm-里-itcm-就是-rom-吗)
  - [Q4. startup 真的做了 ROM→RAM 拷贝吗？](#q4-startup-真的做了-romram-拷贝吗)
  - [Q5. `__StackTop` 为什么就是栈顶？越界谁管？](#q5-__stacktop-为什么就是栈顶越界谁管)
  - [Q6. 链接脚本里的地址是虚拟地址吗？](#q6-链接脚本里的地址是虚拟地址吗)
- [第四部分：踩坑记录](#第四部分踩坑记录)
- [第五部分：参考资料](#第五部分参考资料)

---

## 第一部分：环境搭建流程

### 1. WSL2 验证

```bash
wsl --version        # 确认 WSL 已装
wsl --list --verbose # 确认有 Ubuntu 发行版
```

本机状态：`WSL 2.5.9.0` + `Ubuntu-20.04 (WSL2)`，开箱即用。

进入 WSL：`wsl -d Ubuntu-20.04`

### 2. 基础开发工具

```bash
sudo apt install -y build-essential git cmake ninja-build \
                    gcc-arm-none-eabi \
                    flatbuffers-compiler libflatbuffers-dev \
                    curl wget unzip xxd
```

得到：
- gcc / g++ 9.4.0
- cmake 3.16.3
- ninja 1.10.0
- **arm-none-eabi-gcc 9.2.1** ← 这个版本太老，不支持 Cortex-M55
- flatc 1.11.0

### 3. Arm GNU Toolchain 14.2 升级

**问题**：Ubuntu 20.04 源里的 arm-none-eabi-gcc 9.2.1 不支持 Cortex-M55（M55 从 GCC 10.1 才加入）。

**验证**：
```bash
echo 'int main(void){return 0;}' > /tmp/t.c
arm-none-eabi-gcc -mcpu=cortex-m55 -mthumb -c /tmp/t.c -o /tmp/t.o
# 报错：unrecognized -mcpu target: cortex-m55
```

**解决**：从 Arm 官网下载最新工具链（14.2.rel1），解压到 `~/arm-gnu-14/`，加到 PATH。

```bash
cd ~
wget https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz
tar -xJf arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi.tar.xz
mv arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi arm-gnu-14
```

验证 M55 支持：
```bash
$HOME/arm-gnu-14/bin/arm-none-eabi-gcc -mcpu=cortex-m55 -mthumb -c /tmp/t.c -o /tmp/t.o
# OK
```

### 4. 克隆运行时仓库

需要三个仓库：
- `tflite-micro` — 微控制器推理框架
- `ethos-u-core-driver` — NPU 驱动层
- `ethos-u-core-platform` — Corstone-300 板级支持包

```bash
mkdir -p ~/ethos-u-stack && cd ~/ethos-u-stack
git clone --depth 1 https://github.com/tensorflow/tflite-micro.git
git clone --depth 1 https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-driver.git
git clone --depth 1 https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform.git
```

**踩坑**：
- `git.mlplatform.org` 当前不可用（404）
- `review.mlplatform.org` 对匿名克隆返回 403
- `github.com/ARM-software/*` 的 ethos-u 仓库是空壳（clone 成功但无 commit）
- **只有 `gitlab.arm.com/artificial-intelligence/ethos-u/*` 可用**

### 5. 下载安装 FVP

> **注意**：FVP 下载需要在 Arm Developer 网站注册账号、接受 EULA，**必须手动操作浏览器**。

1. 访问 https://developer.arm.com/downloads/-/arm-ecosystem-fvps
2. 选 "IoT FVPs" → "Corstone-300 Ecosystem FVP" → 选 Linux（x86_64）
3. 下载文件：`FVP_Corstone_SSE-300_11.27_42_Linux64.tgz`（约 113 MiB）
4. 放到 Windows 可访问路径（比如 `D:\Ai_infra\llama3\FVP\`）

在 WSL 里解压安装：

```bash
mkdir -p ~/fvp && cd ~/fvp
tar -xzf /mnt/d/Ai_infra/llama3/FVP/FVP_Corstone_SSE-300_11.27_42_Linux64.tgz
./FVP_Corstone_SSE-300.sh \
    --i-agree-to-the-contained-eula \
    --no-interactive \
    --destination ~/fvp/corstone-300 \
    -q
```

安装后关键路径：
- FVP 二进制：`~/fvp/corstone-300/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55`
- Python 库：`~/fvp/corstone-300/python/lib/libpython3.9.so.1.0`

### 6. 处理 libpython3.9 依赖

**问题**：FVP 二进制动态链接 `libpython3.9.so.1.0`，但 Ubuntu 20.04 默认只有 Python 3.8。

**解决**：FVP 自带一份 libpython3.9，设置 `LD_LIBRARY_PATH` 指向它即可：
```bash
export LD_LIBRARY_PATH="$HOME/fvp/corstone-300/python/lib:$LD_LIBRARY_PATH"
```

验证：
```bash
FVP_Corstone_SSE-300_Ethos-U55 --version
# Fast Models [11.27.42 (Dec  9 2024)]
```

### 7. 配置 .bashrc 环境变量

把上面的 PATH 和 LD_LIBRARY_PATH 写入 `~/.bashrc`，每次打开终端自动加载：

```bash
# === Ethos-U / FVP toolchain ===
export PATH="$HOME/arm-gnu-14/bin:$PATH"
export PATH="$HOME/fvp/corstone-300/models/Linux64_GCC-9.3:$PATH"
export LD_LIBRARY_PATH="$HOME/fvp/corstone-300/python/lib:$LD_LIBRARY_PATH"
```

**踩坑**：从 Windows 侧通过 `wsl.exe bash -c "echo ... >> ~/.bashrc"` 写 `.bashrc` 时，
`$PATH` 会被提前展开成 Windows 的巨型 PATH 字符串（包含 MSVC / VS Code / CUDA 等），
导致 `.bashrc` 被污染。**最终用 Python 脚本从 WSL 内直接写**，
见 `scripts/wsl_clean_bashrc.py`。

---

## 第二部分：Hello World 文件详解

本目录下五个文件组成一个最小可运行的 M55 裸机程序。

### 文件清单

| 文件 | 作用 |
|------|------|
| `startup.c` | 向量表 + Reset_Handler |
| `hello.c` | main + Arm Semihosting 打印 |
| `corstone300.ld` | Corstone-300 内存布局 |
| `build.sh` | 编译命令（自包含环境） |
| `run.sh` | FVP 启动命令（自包含环境） |

### startup.c 逐行讲解

```c
extern int main(void);
```
声明 main 存在（在 hello.c 中定义）。`extern` 表示"符号在别处定义，链接时解析"。

```c
extern unsigned int __StackTop;
```
**关键**：`__StackTop` 不是 C 变量，而是 **linker script 定义的地址符号**。
取 `&__StackTop` 得到 DTCM 末端地址。类似还有 `_etext`、`__bss_start__` 等。

```c
__attribute__((noreturn))
void Reset_Handler(void) {
    main();
    for (;;) { }
}
```
- `noreturn`：告诉编译器此函数永不返回，省掉返回代码
- 函数体：调 main，main 若返回则死循环

```c
__attribute__((section(".vectors"), used))
void (* const g_pfnVectors[])(void) = {
    (void (*)(void)) &__StackTop,   /* [0] 初始 SP */
    Reset_Handler,                   /* [1] Reset */
};
```
- `void (*)(void)` = "返回 void、不带参数的函数指针"类型
- `__attribute__((section(".vectors")))` = 把数组放到名为 `.vectors` 的自定义段
- `__attribute__((used))` = 即使无人引用也不准删（防 `--gc-sections` 优化掉）
- 数组第 0 项：把 `__StackTop` 地址强转成函数指针（其实不是函数，占 4 字节位置而已）
- 数组第 1 项：Reset_Handler 函数地址

**内存布局**：
```
0x10000000: [__StackTop 的 32-bit 值]     ← 硬件读这个当 SP
0x10000004: [Reset_Handler 函数地址]      ← 硬件读这个当 PC
```

### hello.c 逐行讲解

使用 **Arm Semihosting 协议**直接打印到 FVP 终端，不依赖任何 C 库。

**Semihosting 协议**：程序执行 `bkpt 0xAB` 时，模拟器拦截，根据 r0/r1 寄存器代为执行主机端操作。

| op | 名字 | 含义 |
|----|------|------|
| 0x04 | SYS_WRITE0 | r1 指向 NUL 结尾字符串，打印到终端 |
| 0x18 | SYS_EXIT | 终止仿真 |

```c
static inline void sh_write0(const char* s) {
    register unsigned op __asm__("r0") = 0x04;
    register const char* str __asm__("r1") = s;
    __asm__ volatile(
        "bkpt 0xAB\n"
        : "+r"(op)
        : "r"(str)
        : "memory"
    );
}
```

**GCC 硬寄存器绑定**：
- `register unsigned op __asm__("r0") = 0x04;` — 把变量 `op` 强绑到 r0 寄存器，初值 0x04
- 必须这么做，因为 semihosting 协议**要求参数精确在 r0/r1**，不能让编译器自选

**扩展内联汇编语法**：
```
__asm__ volatile (
    "汇编指令"       <- 模板
    : 输出操作数     <- 汇编会改写的 C 变量
    : 输入操作数     <- 汇编会读的 C 变量
    : clobber 列表   <- 汇编破坏的其他资源
);
```

具体到这段：
- `"bkpt 0xAB\n"` — 一条软中断指令，立即数 0xAB 是 semihosting 约定
- `"+r"(op)` — 输出：`+` 表示读写，`r` 表示任意通用寄存器（实际固定在 r0）
- `"r"(str)` — 输入：只读，任意通用寄存器（实际固定在 r1）
- `"memory"` — clobber：可能修改内存，阻止编译器跨此汇编优化
- `volatile` — 这段汇编有副作用，不准删

```c
int main(void) {
    sh_write0("Hello World from Cortex-M55 on Corstone-300 FVP!\n");
    sh_write0("This message came through Arm Semihosting.\n");
    sh_exit();
    return 0;  // 到不了
}
```

### corstone300.ld 链接脚本语法

链接脚本是独立的小语言，核心回答三个问题：

1. **内存里有哪些区域？** → `MEMORY` 块
2. **程序从哪里开始？** → `ENTRY` 指令
3. **代码/数据怎么摆？** → `SECTIONS` 块

#### MEMORY 块

```ld
MEMORY
{
    ITCM (rx)  : ORIGIN = 0x10000000, LENGTH = 512K
    DTCM (rwx) : ORIGIN = 0x30000000, LENGTH = 512K
}
```

语法：`名字 (属性) : ORIGIN = 起始, LENGTH = 长度`

属性字母：
- `r` — readable
- `w` — writable
- `x` — executable

#### ENTRY 指令

```ld
ENTRY(Reset_Handler)
```

告诉链接器：ELF 文件头的 `e_entry` 字段填 `Reset_Handler` 地址。
对 Cortex-M 硬件复位无关紧要（硬件只看向量表），
但对调试器、FVP 的 `-a` 加载器是有用的元信息。

#### SECTIONS 块

```ld
SECTIONS
{
    .vectors : { KEEP(*(.vectors)) } > ITCM

    .text : {
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > ITCM

    .data : { *(.data*) } > DTCM AT > ITCM

    .bss : { *(.bss*) *(COMMON) } > DTCM

    __StackTop = ORIGIN(DTCM) + LENGTH(DTCM);
}
```

**关键语法**：

| 符号 | 含义 |
|------|------|
| `*(.vectors)` | 所有输入 `.o` 文件里的 `.vectors` 段，`*` 是 object 通配 |
| `KEEP(...)` | 保护壳，阻止 `--gc-sections` 删除未被引用的段 |
| `> ITCM` | VMA（运行时地址）放 ITCM |
| `AT > ITCM` | LMA（加载地址）放 ITCM |
| `. = ALIGN(4)` | 把位置计数器对齐到 4 的倍数 |
| `ORIGIN(DTCM)` | 内置函数，返回区域起始地址 |
| `LENGTH(DTCM)` | 内置函数，返回区域长度 |

**`.data : { ... } > DTCM AT > ITCM` 双地址含义**：
- **VMA = DTCM**：运行时 `.data` 在 DTCM（可读写）
- **LMA = ITCM**：烧录/加载时 `.data` 的初值保存在 ITCM
- startup 代码负责从 ITCM（LMA）拷贝到 DTCM（VMA）
- **本 hello world 省略了这个拷贝**（详见 Q4）

---

## 第三部分：概念澄清问答

### Q1. 为什么 startup.c 一定是第一个运行？

**答：没有任何"运行时"去选择 startup.c。是 CPU 硬件直接从固定地址取指令开始执行，
我们通过 linker script 让那个地址恰好落在 startup.c 的代码里。**

Cortex-M55 上电瞬间的硬件行为：
```
1. 读内存 [VTOR]     → 作为 SP 初值
2. 读内存 [VTOR + 4] → 作为 PC 初值
3. 从 PC 开始取指令执行
```

VTOR 复位默认值通常是 0x00000000（或 0x10000000，取决于 secure 配置）。
linker script 把 `.vectors` 段放在 ITCM 最前面（0x10000000），
所以硬件读到的就是我们的向量表：第 0 项是 `__StackTop`，第 1 项是 `Reset_Handler`。

```
硬件复位
  ├─ SP ← 0x30080000
  ├─ PC ← &Reset_Handler
  ▼
开始执行 Reset_Handler 的机器码
  ▼
Reset_Handler 调 main()
  ▼
main() 调 sh_write0() → 打印
```

不是"startup.c 第一个运行"，而是 **"0x10000000 那个地址存了什么，什么就第一个运行"**。

### Q2. 为什么 `> ITCM` 就是"第一个放进 ITCM 的段"？

**答：链接器按 SECTIONS 块里从上到下的顺序处理每个输出段；
每个 MEMORY 区域维护一个"当前游标"，起初是 `ORIGIN`，放一个段就往后推进。**

```
初始：ITCM 游标 = 0x10000000, DTCM 游标 = 0x30000000

处理 .vectors (8 字节)：放 ITCM 游标位置 → ITCM 游标 = 0x10000008
处理 .text (0x80 字节)：放 ITCM 游标位置 → ITCM 游标 = 0x10000088
处理 .data：VMA 放 DTCM，LMA 放 ITCM
处理 .bss：VMA 放 DTCM
```

**顺序 = 地址。** 如果把 `.text` 写在 `.vectors` 前面，0x10000000 就是 `.text`，
硬件读到第一条指令的机器码当 SP 用，程序直接崩。

### Q3. `AT > ITCM` 里 ITCM 就是 ROM 吗？

**答：linker script 里根本没有"ROM"的概念。"ITCM 就是 ROM" 只是**工程惯例**。**

linker 眼里只有 `MEMORY` 区域（带属性的地址区间），不知道哪块是 ROM 哪块是 RAM。

**惯例由来**：在真实 MCU 上，程序员用烧录器把 ELF 的 "LMA 内容" 写进 Flash（不掉电），
运行时 RAM 是随机值，所以全局变量的初值必须来自 Flash。
Flash 扮演"存初值"的角色 = ROM。

在 Corstone-300 上，没有 Flash，**ITCM 本质是 RAM**，但承担了"放代码和初值"的角色。
所以约定俗成把 ITCM 当 ROM 用。

```
换成别的芯片，Flash 在 0x08000000：
MEMORY {
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = ...
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = ...
}
.data : { ... } > RAM AT > FLASH   ← 此时 Flash 扮演 ROM
```

### Q4. startup 真的做了 ROM→RAM 拷贝吗？

**答：没有。本 hello world 的 startup.c 偷懒了。**

```c
void Reset_Handler(void) {
    main();           // 直接调 main
    for (;;) { }      // main 返回后死循环
}
```

**没有做**：
- 把 `.data` 段从 LMA 拷到 VMA
- 把 `.bss` 段清零

**为什么能跑**：hello world 没有已初始化/未初始化全局变量，`.data` 和 `.bss` 都是空的。

**正经的 Reset_Handler** 应该是：

```c
extern unsigned int __data_start__, __data_end__, __data_load__;
extern unsigned int __bss_start__, __bss_end__;

void Reset_Handler(void) {
    // 拷贝 .data：LMA → VMA
    unsigned int *src = &__data_load__;
    unsigned int *dst = &__data_start__;
    while (dst < &__data_end__) *dst++ = *src++;

    // 清零 .bss
    unsigned int *bss = &__bss_start__;
    while (bss < &__bss_end__) *bss++ = 0;

    main();
    for (;;);
}
```

linker script 也要相应添加符号定义：
```ld
.data :
{
    __data_start__ = .;
    *(.data*)
    __data_end__ = .;
} > DTCM AT > ITCM
__data_load__ = LOADADDR(.data);
```

### Q5. `__StackTop` 为什么就是栈顶？越界谁管？

**答：linker 只管静态布局，栈溢出是运行时问题，linker 完全不管。**

`__StackTop = ORIGIN(DTCM) + LENGTH(DTCM);` 在链接时计算出一个固定地址
（0x30080000），写入 ELF 符号表。

startup.c 把这个地址当 SP 初值填入向量表第 0 项：
```c
(void (*)(void)) &__StackTop,
```

CPU 复位时读到这个值，设 SP = 0x30080000。从此刻起 SP 由硬件管理，和 linker 再无关系。

**Cortex-M 栈向下长**：
```
0x30080000  ┌─────────┐ ← __StackTop (SP 初值)
            │ (空闲)  │
            │   ↓     │  <- SP 随函数调用下降
            │  栈帧   │
0x30001000  ├─────────┤ ← __bss_end__
            │ .bss    │
0x30000200  ├─────────┤
            │ .data   │
0x30000000  └─────────┘
```

**硬件不检查越界**：SP 可一路减到 `.bss` 区域，悄悄覆盖全局变量。
这是嵌入式开发经典坑。

真实项目防御手段：
1. **MPU（Memory Protection Unit）** — 把栈和 .bss 之间某段标记为不可访问
2. **显式栈段 + 大小** — 链接时检查是否超过 DTCM 容量
3. **栈金丝雀（stack canary）** — 栈底埋魔法数，定期检查

Hello world 只用 < 100 字节栈，DTCM 512KB 绰绰有余，所以"能跑不代表健壮"。

### Q6. 链接脚本里的地址是虚拟地址吗？

**答：看平台。Cortex-M55 裸机是物理地址，Linux 用户态程序是虚拟地址。**

"虚拟地址"的前提是 CPU 有 MMU（Memory Management Unit）。

| 场景 | CPU | MMU | linker script 地址 |
|------|-----|-----|------|
| Linux 用户态 | Cortex-A / x86 | 有 | 进程虚拟地址 |
| Linux 内核 | Cortex-A / x86 | 有（内核自己配） | 内核虚拟地址 |
| **Cortex-M55 裸机** | Cortex-M55 | **没有** | **物理地址** |

Cortex-M 全系无 MMU，CPU 发出的地址直接送到总线。
linker script 里 `0x10000000` 就是 CPU 未来往总线上发的那个数字，一点都不"虚"。

**"VMA (Virtual Memory Address)"的命名是历史遗留**，在裸机场景里 VMA 和 LMA 都是物理地址，
只是代表不同时间的物理地址：
- **VMA** = 运行时这段数据所在的物理地址
- **LMA** = 加载/上电时这段数据存放的物理地址

两者不同的典型场景：`.data` 段 VMA 在 RAM，LMA 在 ROM/Flash，startup 拷贝。

**注意**：Cortex-M 有 MPU（Memory Protection Unit），但 MPU 只做**保护**（谁能读/写/执行哪段），
不做**翻译**。所以 MPU 和地址虚拟化无关。

---

## 第四部分：踩坑记录

### 坑 1：arm-none-eabi-gcc 9.2 不支持 Cortex-M55
**现象**：`unrecognized -mcpu target: cortex-m55`
**原因**：M55 在 GCC 10.1 才加入，Ubuntu 20.04 源里是 9.2.1
**解决**：装 Arm GNU Toolchain 14.2

### 坑 2：`git.mlplatform.org` / `review.mlplatform.org` 都不可用
**现象**：克隆 ethos-u-core-driver 报 404 / 403
**原因**：官方仓库迁移，老链接失效
**解决**：用 `gitlab.arm.com/artificial-intelligence/ethos-u/*`

### 坑 3：FVP 报 `libpython3.9.so.1.0: cannot open shared object file`
**现象**：跑 `FVP_Corstone_SSE-300_Ethos-U55 --version` 报错
**原因**：Ubuntu 20.04 默认 Python 是 3.8
**解决**：FVP 自带 libpython3.9，设 `LD_LIBRARY_PATH="$HOME/fvp/corstone-300/python/lib:..."`

### 坑 4：wsl.exe 传命令时 `$PATH` / `$HOME` 被 Windows 侧提前展开
**现象**：`echo 'export PATH=$HOME/... >> ~/.bashrc'` 写入的是展开后的 Windows PATH
**原因**：Git Bash / PowerShell 侧的 shell 解析了 `$PATH`
**解决**：用 Python 脚本从 WSL 内直接写 `.bashrc`（`scripts/wsl_clean_bashrc.py`）

### 坑 5：`./run.sh` 报 `FVP_Corstone_SSE-300_Ethos-U55: command not found`
**现象**：终端里能 `which FVP_...`，但 `./run.sh` 找不到
**原因**：
- `./run.sh` 启动子 shell，继承父 shell 的 env
- 用户终端在添加 `.bashrc` 前就开了，父 shell 的 PATH 不含 FVP
- 子 shell 继承的 PATH 也没有 FVP
**解决**：让脚本**自包含环境**——脚本开头自己 `export PATH`，不依赖 `.bashrc`

### 坑 6：WSL 中找不到 `ai_infra`
**现象**：`cd /mnt/d/ai_infra` 报目录不存在
**原因**：
- Windows 不区分大小写，Linux 严格区分
- 实际目录名是 `Ai_infra`（A、I 都大写）
**解决**：用 Tab 补全，或 `cd /mnt/d/Ai_infra/`

---

## 第五部分：参考资料

### 官方文档
- [Corstone SSE-300 Technical Reference Manual](https://developer.arm.com/documentation/101773/latest/)
- [Arm Semihosting for AArch32 and AArch64 (SMFC)](https://developer.arm.com/documentation/100863/latest/)
- [Cortex-M55 Technical Reference Manual](https://developer.arm.com/documentation/101051/latest/)
- [GCC Extended Inline Assembly](https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html)
- [LD 链接器脚本手册](https://sourceware.org/binutils/docs/ld/Scripts.html)

### 工具下载
- [Arm GNU Toolchain](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
- [Arm Ecosystem FVPs](https://developer.arm.com/downloads/-/arm-ecosystem-fvps)
- [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro)
- [Ethos-U Core Driver](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-driver)
- [Ethos-U Core Platform](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform)

### 下一步计划
- M2：选定最小目标模型（单 Conv2D 或 MNIST 3 层）
- M3：self_compiler 端到端走完该模型
- M4：为 EthosUBackend 加二进制命令流编码器
- M5：TFLite flatbuffer 打包 + 嵌入固件 + FVP 验证

参见项目根目录 `README.md`。
