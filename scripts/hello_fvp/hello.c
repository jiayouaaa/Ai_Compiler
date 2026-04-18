/*
 * hello.c - 通过 Arm Semihosting 打印 Hello World
 *
 * 原理：
 *   Semihosting 是 Arm 定义的"让运行中的程序跟调试器/模拟器对话"的协议。
 *   程序执行 BKPT #0xAB 指令（Thumb）时，模拟器会拦截，
 *   根据 r0 的 operation code 执行主机端操作（print/read/exit 等），
 *   结果写回 r0。
 *
 *   operation 0x04 = SYS_WRITE0: 把 r1 指向的以 NUL 结尾的字符串打印到主机终端
 *   operation 0x18 = SYS_EXIT:   程序结束
 */

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

static inline void sh_exit(void) {
    register unsigned op __asm__("r0") = 0x18;
    register unsigned code __asm__("r1") = 0x20026;  /* ADP_Stopped_ApplicationExit */
    __asm__ volatile(
        "bkpt 0xAB\n"
        :
        : "r"(op), "r"(code)
        : "memory"
    );
}

int main(void) {
    sh_write0("Hello World from Cortex-M55 on Corstone-300 FVP!\n");
    sh_write0("This message came through Arm Semihosting.\n");
    sh_exit();
    return 0;  /* 不会到达 */
}
