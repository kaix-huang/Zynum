// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Apple AMX instruction emitters and operand encoders.

pub inline fn nopOpImm5(comptime op: usize, comptime imm5: usize) void {
    asm volatile ("nop\nnop\nnop\n.word (0x201000 + (%[op] << 5) + %[imm5])"
        :
        : [op] "i" (op),
          [imm5] "i" (imm5),
        : .{ .memory = true });
}

pub inline fn opGpr(comptime op: usize, gpr: usize) void {
    asm volatile (".word (0x201000 + (%[op] << 5) + 0%[gpr] - ((0%[gpr] >> 4) * 6))"
        :
        : [op] "i" (op),
          [gpr] "r" (gpr),
        : .{ .memory = true });
}

pub inline fn opGprNoMem(comptime op: usize, gpr: usize) void {
    asm volatile (".word (0x201000 + (%[op] << 5) + 0%[gpr] - ((0%[gpr] >> 4) * 6))"
        :
        : [op] "i" (op),
          [gpr] "r" (gpr),
        : .{});
}

pub inline fn ldx(gpr: usize) void {
    opGpr(0, gpr);
}

pub inline fn ldy(gpr: usize) void {
    opGpr(1, gpr);
}

pub inline fn stz(gpr: usize) void {
    opGpr(5, gpr);
}

pub inline fn fma64(gpr: usize) void {
    opGprNoMem(10, gpr);
}

pub inline fn fma32(gpr: usize) void {
    opGprNoMem(12, gpr);
}

pub inline fn matfp(gpr: usize) void {
    opGprNoMem(21, gpr);
}

pub inline fn set() void {
    nopOpImm5(17, 0);
}

pub inline fn clr() void {
    nopOpImm5(17, 1);
}

pub inline fn ptrRowFlags(ptr: anytype, row: usize, flags: usize) usize {
    return @intFromPtr(ptr) + ((row + flags * 64) << 56);
}

pub inline fn fma32Operand(skip_z: bool) usize {
    return if (skip_z) (1 << 27) else 0;
}

pub inline fn fma64Operand(skip_z: bool) usize {
    return if (skip_z) (1 << 27) else 0;
}

pub inline fn fma32RowOperand(row: usize, skip_z: bool) usize {
    return (row << 20) | fma32Operand(skip_z);
}

pub inline fn fma64RowOperand(row: usize, skip_z: bool) usize {
    return (row << 20) | fma64Operand(skip_z);
}

pub inline fn fma64XyOperand(xrow: usize, yrow: usize, zrow: usize, skip_z: bool) usize {
    return (xrow << 16) | (yrow << 6) | (zrow << 20) | fma64Operand(skip_z);
}

pub inline fn matfp32RowOperand(row: usize) usize {
    return (4 << 42) | (row << 20);
}

pub inline fn fma32XyRowOperand(xrow: usize, yrow: usize, zrow: usize, skip_z: bool) usize {
    return (xrow << 16) | (yrow << 6) | (zrow << 20) | fma32Operand(skip_z);
}
