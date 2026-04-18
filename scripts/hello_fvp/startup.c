/*
 * 最小启动代码 for Cortex-M55 on Corstone-300 FVP
 *
 * Cortex-M 复位序列（硬件自动）：
 *   1. 从地址 0x00000000 读取初始 SP（栈顶地址）
 *   2. 从地址 0x00000004 读取初始 PC（Reset_Handler 地址）
 *   3. 跳转到 PC 开始执行
 *
 * 所以我们要做的：
 *   - 定义一个向量表（数组），第 0 项是 SP，第 1 项是 Reset_Handler
 *   - 把这个数组通过 linker script 放到 0x10000000（ITCM 起始）
 *   - Reset_Handler 调 main() 后死循环
 *
 * 注：__StackTop 由 linker script 定义 = DTCM 末端地址
 */

extern int main(void);
extern unsigned int __StackTop;

__attribute__((noreturn))
void Reset_Handler(void) {
    main();
    for (;;) { /* main 返回后永远停在这里 */ }
}

/* 向量表：只填必需的两项，其他表项为 0（如果触发会 HardFault） */
__attribute__((section(".vectors"), used))
void (* const g_pfnVectors[])(void) = {
    (void (*)(void)) &__StackTop,   /* [0] 初始 SP */
    Reset_Handler,                   /* [1] Reset */
    /* 其他异常/中断略，当前 hello world 不会触发 */
};
