/*
 * Corstone-300 最小 linker script（只用 ITCM + DTCM）
 *
 * 内存映射摘录（来自 Corstone SSE-300 TRM）：
 *   ITCM (secure alias) : 0x10000000, 512 KB  — 存代码
 *   DTCM (secure alias) : 0x30000000, 512 KB  — 存数据/栈
 *   BRAM (secure alias) : 0x11000000,   1 MB
 *   SRAM (secure alias) : 0x31000000,   2 MB
 *
 * hello world 小到只用 ITCM + DTCM 就够了。
 */

MEMORY
{
    ITCM (rx)  : ORIGIN = 0x10000000, LENGTH = 512K
    DTCM (rwx) : ORIGIN = 0x30000000, LENGTH = 512K
}

ENTRY(Reset_Handler)

SECTIONS
{
    /* 向量表必须在 ITCM 最前面，复位时 CPU 读 0x10000000 / 0x10000004 */
    .vectors :
    {
        KEEP(*(.vectors))
    } > ITCM

    /* 代码和只读数据 */
    .text :
    {
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > ITCM

    /* 已初始化数据：存储在 ITCM（ROM），运行时拷到 DTCM（RAM）
       当前 hello world 没有已初始化全局变量，保留段定义以备扩展 */
    .data :
    {
        *(.data*)
        . = ALIGN(4);
    } > DTCM AT > ITCM

    /* 未初始化数据（清零） */
    .bss :
    {
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
    } > DTCM

    /* 栈顶 = DTCM 末端（向下增长） */
    __StackTop = ORIGIN(DTCM) + LENGTH(DTCM);
}
