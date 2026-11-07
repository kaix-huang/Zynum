/* SPDX-License-Identifier: MIT
 * Zig 0.17.0 lib/compiler_rt/stack_probe.zig, x86_64 zig_probe_stack.
 *
 * The MIT License (Expat)
 *
 * Copyright (c) Zig contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

    .text
    .weak __zig_probe_stack
    .hidden __zig_probe_stack
    .type __zig_probe_stack, @function
__zig_probe_stack:
    push   %rcx
    mov    %rax, %rcx
    cmp    $0x1000, %rcx
    jb     2f
1:
    sub    $0x1000, %rsp
    orl    $0, 16(%rsp)
    sub    $0x1000, %rcx
    cmp    $0x1000, %rcx
    ja     1b
2:
    sub    %rcx, %rsp
    orl    $0, 16(%rsp)
    add    %rax, %rsp
    pop    %rcx
    ret
    .size __zig_probe_stack, .-__zig_probe_stack

    .section .note.GNU-stack,"",@progbits
