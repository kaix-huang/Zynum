// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

pub const read_svl_x14 =
    \\
    \\rdsvl x14, #1
    \\
;

pub const ptrue_p0_s = ptrue("p0", "s");
pub const ptrue_pn8_s = ptrue("pn8", "s");
pub const ptrue_p0_d = ptrue("p0", "d");
pub const ptrue_p1_d = ptrue("p1", "d");
pub const ptrue_pn8_d = ptrue("pn8", "d");

fn scalarAddrOffset(comptime base: []const u8, comptime offset: comptime_int) []const u8 {
    if (offset == 0) return std.fmt.comptimePrint("[{s}]", .{base});
    return std.fmt.comptimePrint("[{s}, #{d}]", .{ base, offset });
}

pub fn sveDgemvTransReduceStore8F64Asm(comptime acc_first: comptime_int) []const u8 {
    comptime var text: []const u8 = "";
    inline for (0..8) |i| {
        const addr = scalarAddrOffset("x5", i * @sizeOf(f64));
        text = text ++ std.fmt.comptimePrint(
            \\
            \\faddv d1, p1, z{d}.d
            \\ldr d2, {s}
            \\fmadd d2, d0, d1, d2
            \\str d2, {s}
        , .{ acc_first + i, addr, addr });
    }
    return text;
}

pub fn ptrue(comptime pred: []const u8, comptime lane: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ptrue {s}.{s}
        \\
    , .{ pred, lane });
}

fn laneLoadSuffix(comptime lane: []const u8) []const u8 {
    if (std.mem.eql(u8, lane, "b")) return "b";
    if (std.mem.eql(u8, lane, "s")) return "w";
    if (std.mem.eql(u8, lane, "d")) return "d";
    @compileError("unsupported AArch64 lane suffix");
}

fn laneCountMnemonic(comptime lane: []const u8) []const u8 {
    if (std.mem.eql(u8, lane, "s")) return "cntw";
    if (std.mem.eql(u8, lane, "d")) return "cntd";
    @compileError("unsupported SVE element count lane");
}

fn laneIncMnemonic(comptime lane: []const u8) []const u8 {
    if (std.mem.eql(u8, lane, "s")) return "incw";
    if (std.mem.eql(u8, lane, "d")) return "incd";
    @compileError("unsupported SVE increment lane");
}

fn laneIndexShift(comptime lane: []const u8) comptime_int {
    if (std.mem.eql(u8, lane, "s")) return 2;
    if (std.mem.eql(u8, lane, "d")) return 3;
    @compileError("unsupported indexed SVE lane");
}

fn scalarReg(comptime lane: []const u8, comptime index: comptime_int) []const u8 {
    if (std.mem.eql(u8, lane, "s")) return std.fmt.comptimePrint("s{d}", .{index});
    if (std.mem.eql(u8, lane, "d")) return std.fmt.comptimePrint("d{d}", .{index});
    @compileError("unsupported floating-point scalar lane");
}

fn gprZero(comptime lane: []const u8) []const u8 {
    if (std.mem.eql(u8, lane, "s")) return "wzr";
    if (std.mem.eql(u8, lane, "d")) return "xzr";
    @compileError("unsupported zero register lane");
}

fn returnGpr(comptime lane: []const u8) []const u8 {
    if (std.mem.eql(u8, lane, "s")) return "w0";
    if (std.mem.eql(u8, lane, "d")) return "x0";
    @compileError("unsupported return register lane");
}

fn log2Pow2(comptime value: comptime_int) comptime_int {
    comptime var shift: comptime_int = 0;
    comptime var current: comptime_int = value;
    while (current > 1) : ({
        current /= 2;
        shift += 1;
    }) {}
    if ((@as(comptime_int, 1) << shift) != value) @compileError("value must be a power of two");
    return shift;
}

fn vlAddr(comptime base: []const u8, comptime offset: comptime_int) []const u8 {
    if (offset == 0) return std.fmt.comptimePrint("[{s}]", .{base});
    return std.fmt.comptimePrint("[{s}, #{d}, MUL VL]", .{ base, offset });
}

fn ld1(
    comptime lane: []const u8,
    comptime z_reg: comptime_int,
    comptime pred: []const u8,
    comptime base: []const u8,
    comptime vl_offset: comptime_int,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ld1{s} {{ z{d}.{s} }}, {s}/z, {s}
    , .{ laneLoadSuffix(lane), z_reg, lane, pred, vlAddr(base, vl_offset) });
}

fn st1(
    comptime lane: []const u8,
    comptime z_reg: comptime_int,
    comptime pred: []const u8,
    comptime base: []const u8,
    comptime vl_offset: comptime_int,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\st1{s} {{ z{d}.{s} }}, {s}, {s}
    , .{ laneLoadSuffix(lane), z_reg, lane, pred, vlAddr(base, vl_offset) });
}

fn ld1Indexed(
    comptime lane: []const u8,
    comptime z_reg: comptime_int,
    comptime pred: []const u8,
    comptime base: []const u8,
    comptime index: []const u8,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ld1{s} {{ z{d}.{s} }}, {s}/z, [{s}, {s}, lsl #{d}]
    , .{ laneLoadSuffix(lane), z_reg, lane, pred, base, index, laneIndexShift(lane) });
}

fn st1Indexed(
    comptime lane: []const u8,
    comptime z_reg: comptime_int,
    comptime pred: []const u8,
    comptime base: []const u8,
    comptime index: []const u8,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\st1{s} {{ z{d}.{s} }}, {s}, [{s}, {s}, lsl #{d}]
    , .{ laneLoadSuffix(lane), z_reg, lane, pred, base, index, laneIndexShift(lane) });
}

pub const SveDgemvTransF64Mode = enum {
    predicated,
    full_n8,
    full_n8_acc2,
};

fn sveDgemvTransColumnPointers8Asm() []const u8 {
    comptime var text: []const u8 =
        \\
        \\mov x8, x2
    ;
    inline for (9..16) |reg| {
        text = text ++ std.fmt.comptimePrint(
            \\
            \\add x{d}, x{d}, x3
        , .{ reg, reg - 1 });
    }
    return text;
}

fn sveDupZeroF64RangeAsm(comptime first: comptime_int, comptime count: comptime_int) []const u8 {
    comptime var text: []const u8 = "";
    inline for (0..count) |i| {
        text = text ++ std.fmt.comptimePrint(
            \\
            \\dup z{d}.d, #0
        , .{first + i});
    }
    return text;
}

fn sveDgemvTransStep8F64Asm(comptime acc_first: comptime_int) []const u8 {
    comptime var text = ld1Indexed("d", 16, "p0", "x4", "x16");
    inline for (0..8) |col| {
        const base = std.fmt.comptimePrint("x{d}", .{8 + col});
        text = text ++ ld1Indexed("d", 17 + col, "p0", base, "x16");
    }
    inline for (0..8) |col| {
        text = text ++ std.fmt.comptimePrint(
            \\
            \\fmla z{d}.d, p0/m, z{d}.d, z16.d
        , .{ acc_first + col, 17 + col });
    }
    return text ++
        \\
        \\incd x16
    ;
}

fn sveFaddF64RangeAsm(comptime lhs_first: comptime_int, comptime rhs_first: comptime_int, comptime count: comptime_int) []const u8 {
    comptime var text: []const u8 = "";
    inline for (0..count) |i| {
        text = text ++ std.fmt.comptimePrint(
            \\
            \\fadd z{d}.d, z{d}.d, z{d}.d
        , .{ lhs_first + i, lhs_first + i, rhs_first + i });
    }
    return text;
}

fn sveDgemvTransTailF64Asm(comptime labels: anytype) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\{s}:
        \\cbz x1, {s}f
        \\
        \\{s}:
    , .{ labels.tail, labels.done, labels.tail_loop }) ++ sveDupZeroF64RangeAsm(0, 1) ++ std.fmt.comptimePrint(
        \\
        \\mov x16, #0
        \\
        \\{s}:
        \\whilelo p0.d, x16, x0
        \\b.none {s}f
    , .{ labels.tail_k_loop, labels.tail_reduce }) ++ ld1Indexed("d", 16, "p0", "x4", "x16") ++ ld1Indexed("d", 17, "p0", "x2", "x16") ++
        \\
        \\fmla z0.d, p0/m, z17.d, z16.d
        \\incd x16
    ++ std.fmt.comptimePrint(
        \\
        \\b {s}b
        \\
        \\{s}:
        \\faddv d1, p1, z0.d
        \\ldr d2, [x5]
        \\fmadd d2, d0, d1, d2
        \\str d2, [x5]
        \\add x2, x2, x3
        \\add x5, x5, #8
        \\subs x1, x1, #1
        \\b.ne {s}b
        \\
        \\{s}:
        \\ret
    , .{ labels.tail_k_loop, labels.tail_reduce, labels.tail_loop, labels.done });
}

fn sveDgemvTransFullN8TailF64Asm(comptime labels: anytype) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\{s}:
        \\cbz x1, {s}f
        \\b {s}f
        \\
        \\{s}:
        \\ret
    , .{ labels.tail, labels.done, labels.done, labels.done });
}

pub fn sveDgemvTransF64Asm(comptime mode: SveDgemvTransF64Mode) []const u8 {
    const labels = switch (mode) {
        .predicated => .{
            .tail = "14",
            .done = "18",
            .tail_loop = "15",
            .tail_k_loop = "16",
            .tail_reduce = "17",
        },
        .full_n8, .full_n8_acc2 => .{
            .tail = "14",
            .done = "15",
        },
    };
    const pred_setup = if (mode == .predicated) ptrue_p1_d else ptrue_p0_d ++ ptrue_p1_d;
    const accumulators: comptime_int = if (mode == .full_n8_acc2) 16 else 8;
    const loop_test =
        if (mode == .predicated)
            \\
            \\whilelo p0.d, x16, x0
            \\b.none 13f
        else
            \\
            \\cmp x16, x0
            \\b.hs 13f
        ;
    const steps =
        if (mode == .full_n8_acc2)
            sveDgemvTransStep8F64Asm(0) ++ sveDgemvTransStep8F64Asm(8)
        else
            sveDgemvTransStep8F64Asm(0);
    const combine =
        if (mode == .full_n8_acc2)
            sveFaddF64RangeAsm(0, 8, 8)
        else
            "";
    const tail =
        if (mode == .predicated)
            sveDgemvTransTailF64Asm(labels)
        else
            sveDgemvTransFullN8TailF64Asm(labels);
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, {s}f
        \\cbz x1, {s}f
    , .{ labels.done, labels.done }) ++ pred_setup ++
        \\
        \\10:
        \\cmp x1, #8
    ++ std.fmt.comptimePrint(
        \\
        \\b.lo {s}f
        \\
        \\11:
    , .{labels.tail}) ++ sveDgemvTransColumnPointers8Asm() ++ sveDupZeroF64RangeAsm(0, accumulators) ++ std.fmt.comptimePrint(
        \\
        \\mov x16, #0
        \\
        \\12:{s}
    , .{loop_test}) ++ steps ++
        \\
        \\b 12b
        \\
        \\13:
    ++ combine ++ sveDgemvTransReduceStore8F64Asm(0) ++
        \\
        \\
        \\add x2, x15, x3
        \\add x5, x5, #64
        \\sub x1, x1, #8
        \\b 10b
        \\
    ++ tail;
}

pub fn dgemvTransSveF64Asm() []const u8 {
    return sveDgemvTransF64Asm(.predicated);
}

pub fn dgemvTransSveF64FullN8Asm() []const u8 {
    return sveDgemvTransF64Asm(.full_n8);
}

pub fn dgemvTransSveF64FullN8Acc2Asm() []const u8 {
    return sveDgemvTransF64Asm(.full_n8_acc2);
}

fn cgemvTransFcmlaReduceStoreAsm(comptime acc: comptime_int, comptime offset: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ext v0.16b, v{d}.16b, v{d}.16b, #8
        \\fadd v{d}.2s, v{d}.2s, v0.2s
        \\movi v0.4s, #0
        \\fcmla v0.4s, v{d}.4s, v6.4s, #0
        \\fcmla v0.4s, v{d}.4s, v6.4s, #90
        \\ldr d5, [x7, #{d}]
        \\movi v1.4s, #0
        \\fcmla v1.4s, v5.4s, v7.4s, #0
        \\fcmla v1.4s, v5.4s, v7.4s, #90
        \\fadd v0.2s, v0.2s, v1.2s
        \\str d0, [x7, #{d}]
    , .{ acc, acc, acc, acc, acc, acc, offset, offset });
}

pub fn cgemvTransFcmlaF32M128Asm() []const u8 {
    return
    \\
    \\fmov s6, w0
    \\fmov s5, w1
    \\fmov s7, w2
    \\fmov s4, w3
    \\dup v6.4s, v6.s[0]
    \\mov v6.s[1], v5.s[0]
    \\mov v6.s[3], v5.s[0]
    \\dup v7.4s, v7.s[0]
    \\mov v7.s[1], v4.s[0]
    \\mov v7.s[3], v4.s[0]
    \\mov x14, #32
    \\mov x15, x4
    \\
    \\1:
    \\movi v16.4s, #0
    \\movi v17.4s, #0
    \\movi v18.4s, #0
    \\movi v19.4s, #0
    \\movi v20.4s, #0
    \\movi v21.4s, #0
    \\movi v22.4s, #0
    \\movi v23.4s, #0
    \\movi v24.4s, #0
    \\movi v25.4s, #0
    \\movi v26.4s, #0
    \\movi v27.4s, #0
    \\movi v28.4s, #0
    \\movi v29.4s, #0
    \\movi v30.4s, #0
    \\movi v31.4s, #0
    \\mov x8, x15
    \\add x9, x8, x5
    \\add x10, x9, x5
    \\add x11, x10, x5
    \\mov x12, x6
    \\mov x13, #16
    \\
    \\2:
    \\ldr q0, [x12], #16
    \\ldr q1, [x8], #16
    \\ldr q2, [x9], #16
    \\ldr q3, [x10], #16
    \\ldr q4, [x11], #16
    \\fcmla v16.4s, v1.4s, v0.4s, #0
    \\fcmla v17.4s, v2.4s, v0.4s, #0
    \\fcmla v18.4s, v3.4s, v0.4s, #0
    \\fcmla v19.4s, v4.4s, v0.4s, #0
    \\fcmla v16.4s, v1.4s, v0.4s, #90
    \\fcmla v17.4s, v2.4s, v0.4s, #90
    \\fcmla v18.4s, v3.4s, v0.4s, #90
    \\fcmla v19.4s, v4.4s, v0.4s, #90
    \\ldr q0, [x12], #16
    \\ldr q1, [x8], #16
    \\ldr q2, [x9], #16
    \\ldr q3, [x10], #16
    \\ldr q4, [x11], #16
    \\fcmla v20.4s, v1.4s, v0.4s, #0
    \\fcmla v21.4s, v2.4s, v0.4s, #0
    \\fcmla v22.4s, v3.4s, v0.4s, #0
    \\fcmla v23.4s, v4.4s, v0.4s, #0
    \\fcmla v20.4s, v1.4s, v0.4s, #90
    \\fcmla v21.4s, v2.4s, v0.4s, #90
    \\fcmla v22.4s, v3.4s, v0.4s, #90
    \\fcmla v23.4s, v4.4s, v0.4s, #90
    \\ldr q0, [x12], #16
    \\ldr q1, [x8], #16
    \\ldr q2, [x9], #16
    \\ldr q3, [x10], #16
    \\ldr q4, [x11], #16
    \\fcmla v24.4s, v1.4s, v0.4s, #0
    \\fcmla v25.4s, v2.4s, v0.4s, #0
    \\fcmla v26.4s, v3.4s, v0.4s, #0
    \\fcmla v27.4s, v4.4s, v0.4s, #0
    \\fcmla v24.4s, v1.4s, v0.4s, #90
    \\fcmla v25.4s, v2.4s, v0.4s, #90
    \\fcmla v26.4s, v3.4s, v0.4s, #90
    \\fcmla v27.4s, v4.4s, v0.4s, #90
    \\ldr q0, [x12], #16
    \\ldr q1, [x8], #16
    \\ldr q2, [x9], #16
    \\ldr q3, [x10], #16
    \\ldr q4, [x11], #16
    \\fcmla v28.4s, v1.4s, v0.4s, #0
    \\fcmla v29.4s, v2.4s, v0.4s, #0
    \\fcmla v30.4s, v3.4s, v0.4s, #0
    \\fcmla v31.4s, v4.4s, v0.4s, #0
    \\fcmla v28.4s, v1.4s, v0.4s, #90
    \\fcmla v29.4s, v2.4s, v0.4s, #90
    \\fcmla v30.4s, v3.4s, v0.4s, #90
    \\fcmla v31.4s, v4.4s, v0.4s, #90
    \\subs x13, x13, #1
    \\b.ne 2b
    \\fadd v16.4s, v16.4s, v20.4s
    \\fadd v17.4s, v17.4s, v21.4s
    \\fadd v18.4s, v18.4s, v22.4s
    \\fadd v19.4s, v19.4s, v23.4s
    \\fadd v24.4s, v24.4s, v28.4s
    \\fadd v25.4s, v25.4s, v29.4s
    \\fadd v26.4s, v26.4s, v30.4s
    \\fadd v27.4s, v27.4s, v31.4s
    \\fadd v16.4s, v16.4s, v24.4s
    \\fadd v17.4s, v17.4s, v25.4s
    \\fadd v18.4s, v18.4s, v26.4s
    \\fadd v19.4s, v19.4s, v27.4s
    ++ cgemvTransFcmlaReduceStoreAsm(16, 0) ++
        cgemvTransFcmlaReduceStoreAsm(17, 8) ++
        cgemvTransFcmlaReduceStoreAsm(18, 16) ++
        cgemvTransFcmlaReduceStoreAsm(19, 24) ++
        \\
        \\lsl x16, x5, #2
        \\add x15, x15, x16
        \\add x7, x7, #32
        \\subs x14, x14, #1
        \\b.ne 1b
        \\ret
    ;
}

fn cgemvNoTransFcmlaLoadCoeffAsm(comptime coeff: comptime_int, comptime offset: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ldr d0, [x6, #{d}]
        \\mov v0.d[1], v0.d[0]
        \\movi v{d}.4s, #0
        \\fcmla v{d}.4s, v0.4s, v6.4s, #0
        \\fcmla v{d}.4s, v0.4s, v6.4s, #90
    , .{ offset, coeff, coeff, coeff });
}

fn cgemvNoTransFcmlaColumnAsm(comptime ptr: comptime_int, comptime coeff: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ldr q0, [x{d}, x16]
        \\fcmla v16.4s, v0.4s, v{d}.4s, #0
        \\fcmla v16.4s, v0.4s, v{d}.4s, #90
    , .{ ptr, coeff, coeff });
}

pub fn cgemvNoTransFcmlaF32M128Asm() []const u8 {
    return
    \\
    \\fmov s6, w0
    \\fmov s5, w1
    \\fmov s7, w2
    \\fmov s4, w3
    \\dup v6.4s, v6.s[0]
    \\mov v6.s[1], v5.s[0]
    \\mov v6.s[3], v5.s[0]
    \\dup v7.4s, v7.s[0]
    \\mov v7.s[1], v4.s[0]
    \\mov v7.s[3], v4.s[0]
    \\mov x3, #16
    \\
    \\1:
    ++ cgemvNoTransFcmlaLoadCoeffAsm(20, 0) ++
        cgemvNoTransFcmlaLoadCoeffAsm(21, 8) ++
        cgemvNoTransFcmlaLoadCoeffAsm(22, 16) ++
        cgemvNoTransFcmlaLoadCoeffAsm(23, 24) ++
        cgemvNoTransFcmlaLoadCoeffAsm(24, 32) ++
        cgemvNoTransFcmlaLoadCoeffAsm(25, 40) ++
        cgemvNoTransFcmlaLoadCoeffAsm(26, 48) ++
        cgemvNoTransFcmlaLoadCoeffAsm(27, 56) ++
        \\
        \\mov x8, x4
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\add x12, x11, x5
        \\add x13, x12, x5
        \\add x14, x13, x5
        \\add x15, x14, x5
        \\mov x16, #0
        \\mov x17, #64
        \\
        \\2:
        \\ldr q1, [x7, x16]
        \\movi v16.4s, #0
        \\fcmla v16.4s, v1.4s, v7.4s, #0
        \\fcmla v16.4s, v1.4s, v7.4s, #90
    ++ cgemvNoTransFcmlaColumnAsm(8, 20) ++
        cgemvNoTransFcmlaColumnAsm(9, 21) ++
        cgemvNoTransFcmlaColumnAsm(10, 22) ++
        cgemvNoTransFcmlaColumnAsm(11, 23) ++
        cgemvNoTransFcmlaColumnAsm(12, 24) ++
        cgemvNoTransFcmlaColumnAsm(13, 25) ++
        cgemvNoTransFcmlaColumnAsm(14, 26) ++
        cgemvNoTransFcmlaColumnAsm(15, 27) ++
        \\
        \\str q16, [x7, x16]
        \\add x16, x16, #16
        \\subs x17, x17, #1
        \\b.ne 2b
        \\
        \\lsl x2, x5, #3
        \\add x4, x4, x2
        \\add x6, x6, #64
        \\fmov s7, #1.0
        \\movi v4.4s, #0
        \\dup v7.4s, v7.s[0]
        \\mov v7.s[1], v4.s[0]
        \\mov v7.s[3], v4.s[0]
        \\subs x3, x3, #1
        \\b.ne 1b
        \\ret
    ;
}

fn cgemvNoTransFcmlaTaskColumnRows8Asm(comptime ptr: comptime_int, comptime coeff: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ldr q0, [x{d}, x16]
        \\fcmla v16.4s, v0.4s, v{d}.4s, #0
        \\fcmla v16.4s, v0.4s, v{d}.4s, #90
        \\ldr q0, [x{d}, x2]
        \\fcmla v17.4s, v0.4s, v{d}.4s, #0
        \\fcmla v17.4s, v0.4s, v{d}.4s, #90
        \\ldr q0, [x{d}, x0]
        \\fcmla v18.4s, v0.4s, v{d}.4s, #0
        \\fcmla v18.4s, v0.4s, v{d}.4s, #90
        \\ldr q0, [x{d}, x1]
        \\fcmla v19.4s, v0.4s, v{d}.4s, #0
        \\fcmla v19.4s, v0.4s, v{d}.4s, #90
    , .{
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
    });
}

fn cgemvNoTransFcmlaTaskColumnRows16Asm(comptime ptr: comptime_int, comptime coeff: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ldr q0, [x{d}, x16]
        \\fcmla v16.4s, v0.4s, v{d}.4s, #0
        \\fcmla v16.4s, v0.4s, v{d}.4s, #90
        \\ldr q0, [x{d}, x2]
        \\fcmla v17.4s, v0.4s, v{d}.4s, #0
        \\fcmla v17.4s, v0.4s, v{d}.4s, #90
        \\ldr q0, [x{d}, x0]
        \\fcmla v18.4s, v0.4s, v{d}.4s, #0
        \\fcmla v18.4s, v0.4s, v{d}.4s, #90
        \\ldr q0, [x{d}, x1]
        \\fcmla v19.4s, v0.4s, v{d}.4s, #0
        \\fcmla v19.4s, v0.4s, v{d}.4s, #90
        \\ldr q0, [x{d}, x12]
        \\fcmla v20.4s, v0.4s, v{d}.4s, #0
        \\fcmla v20.4s, v0.4s, v{d}.4s, #90
        \\ldr q0, [x{d}, x13]
        \\fcmla v21.4s, v0.4s, v{d}.4s, #0
        \\fcmla v21.4s, v0.4s, v{d}.4s, #90
        \\ldr q0, [x{d}, x14]
        \\fcmla v22.4s, v0.4s, v{d}.4s, #0
        \\fcmla v22.4s, v0.4s, v{d}.4s, #90
        \\ldr q0, [x{d}, x15]
        \\fcmla v23.4s, v0.4s, v{d}.4s, #0
        \\fcmla v23.4s, v0.4s, v{d}.4s, #90
    , .{
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
    });
}

pub fn cgemvNoTransFcmlaF32M512N64TaskAsm() []const u8 {
    return
    \\
    \\fmov s6, w0
    \\fmov s5, w1
    \\dup v6.4s, v6.s[0]
    \\mov v6.s[1], v5.s[0]
    \\mov v6.s[3], v5.s[0]
    \\mov x3, #16
    \\
    \\1:
    ++ cgemvNoTransFcmlaLoadCoeffAsm(24, 0) ++
        cgemvNoTransFcmlaLoadCoeffAsm(25, 8) ++
        cgemvNoTransFcmlaLoadCoeffAsm(26, 16) ++
        cgemvNoTransFcmlaLoadCoeffAsm(27, 24) ++
        \\
        \\mov x8, x4
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\mov x16, #0
        \\mov x17, #64
        \\
        \\2:
        \\add x2, x16, #16
        \\add x0, x16, #32
        \\add x1, x16, #48
        \\ldr q16, [x7, x16]
        \\ldr q17, [x7, x2]
        \\ldr q18, [x7, x0]
        \\ldr q19, [x7, x1]
    ++ cgemvNoTransFcmlaTaskColumnRows8Asm(8, 24) ++
        cgemvNoTransFcmlaTaskColumnRows8Asm(9, 25) ++
        cgemvNoTransFcmlaTaskColumnRows8Asm(10, 26) ++
        cgemvNoTransFcmlaTaskColumnRows8Asm(11, 27) ++
        \\
        \\str q16, [x7, x16]
        \\str q17, [x7, x2]
        \\str q18, [x7, x0]
        \\str q19, [x7, x1]
        \\add x16, x16, #64
        \\subs x17, x17, #1
        \\b.ne 2b
        \\
        \\lsl x0, x5, #2
        \\add x4, x4, x0
        \\add x6, x6, #32
        \\subs x3, x3, #1
        \\b.ne 1b
        \\ret
    ;
}

pub fn cgemvNoTransFcmlaF32M512N64TaskRow16Asm() []const u8 {
    return
    \\
    \\fmov s6, w0
    \\fmov s5, w1
    \\dup v6.4s, v6.s[0]
    \\mov v6.s[1], v5.s[0]
    \\mov v6.s[3], v5.s[0]
    \\mov x3, #16
    \\
    \\1:
    ++ cgemvNoTransFcmlaLoadCoeffAsm(24, 0) ++
        cgemvNoTransFcmlaLoadCoeffAsm(25, 8) ++
        cgemvNoTransFcmlaLoadCoeffAsm(26, 16) ++
        cgemvNoTransFcmlaLoadCoeffAsm(27, 24) ++
        \\
        \\mov x8, x4
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\mov x16, #0
        \\mov x17, #32
        \\
        \\2:
        \\add x2, x16, #16
        \\add x0, x16, #32
        \\add x1, x16, #48
        \\add x12, x16, #64
        \\add x13, x16, #80
        \\add x14, x16, #96
        \\add x15, x16, #112
        \\ldr q16, [x7, x16]
        \\ldr q17, [x7, x2]
        \\ldr q18, [x7, x0]
        \\ldr q19, [x7, x1]
        \\ldr q20, [x7, x12]
        \\ldr q21, [x7, x13]
        \\ldr q22, [x7, x14]
        \\ldr q23, [x7, x15]
    ++ cgemvNoTransFcmlaTaskColumnRows16Asm(8, 24) ++
        cgemvNoTransFcmlaTaskColumnRows16Asm(9, 25) ++
        cgemvNoTransFcmlaTaskColumnRows16Asm(10, 26) ++
        cgemvNoTransFcmlaTaskColumnRows16Asm(11, 27) ++
        \\
        \\str q16, [x7, x16]
        \\str q17, [x7, x2]
        \\str q18, [x7, x0]
        \\str q19, [x7, x1]
        \\str q20, [x7, x12]
        \\str q21, [x7, x13]
        \\str q22, [x7, x14]
        \\str q23, [x7, x15]
        \\add x16, x16, #128
        \\subs x17, x17, #1
        \\b.ne 2b
        \\
        \\lsl x0, x5, #2
        \\add x4, x4, x0
        \\add x6, x6, #32
        \\subs x3, x3, #1
        \\b.ne 1b
        \\ret
    ;
}

fn cgemvNoTransFcmlaF32Rows16LoopAsm(comptime load_existing: bool) []const u8 {
    const init_acc =
        if (load_existing)
            \\
            \\ldr q16, [x7, x16]
            \\ldr q17, [x7, x2]
            \\ldr q18, [x7, x0]
            \\ldr q19, [x7, x1]
            \\ldr q20, [x7, x12]
            \\ldr q21, [x7, x13]
            \\ldr q22, [x7, x14]
            \\ldr q23, [x7, x15]
            \\
        else
            \\
            \\movi v16.4s, #0
            \\movi v17.4s, #0
            \\movi v18.4s, #0
            \\movi v19.4s, #0
            \\movi v20.4s, #0
            \\movi v21.4s, #0
            \\movi v22.4s, #0
            \\movi v23.4s, #0
            \\
        ;
    return
    \\
    \\mov x16, #0
    \\mov x17, #32
    \\
    \\2:
    \\add x2, x16, #16
    \\add x0, x16, #32
    \\add x1, x16, #48
    \\add x12, x16, #64
    \\add x13, x16, #80
    \\add x14, x16, #96
    \\add x15, x16, #112
    ++ init_acc ++
        cgemvNoTransFcmlaTaskColumnRows16Asm(8, 24) ++
        cgemvNoTransFcmlaTaskColumnRows16Asm(9, 25) ++
        cgemvNoTransFcmlaTaskColumnRows16Asm(10, 26) ++
        cgemvNoTransFcmlaTaskColumnRows16Asm(11, 27) ++
        \\
        \\str q16, [x7, x16]
        \\str q17, [x7, x2]
        \\str q18, [x7, x0]
        \\str q19, [x7, x1]
        \\str q20, [x7, x12]
        \\str q21, [x7, x13]
        \\str q22, [x7, x14]
        \\str q23, [x7, x15]
        \\add x16, x16, #128
        \\subs x17, x17, #1
        \\b.ne 2b
    ;
}

fn cgemvNoTransFcmlaF32LoadCoeff4Asm() []const u8 {
    return cgemvNoTransFcmlaLoadCoeffAsm(24, 0) ++
        cgemvNoTransFcmlaLoadCoeffAsm(25, 8) ++
        cgemvNoTransFcmlaLoadCoeffAsm(26, 16) ++
        cgemvNoTransFcmlaLoadCoeffAsm(27, 24) ++
        \\
        \\mov x8, x4
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
    ;
}

fn cgemvNoTransFcmlaF32Advance4ColumnsAsm() []const u8 {
    return
    \\
    \\lsl x0, x5, #2
    \\add x4, x4, x0
    \\add x6, x6, #32
    ;
}

pub fn cgemvNoTransFcmlaF32M512N64TaskNoMemsetAsm() []const u8 {
    return
    \\
    \\fmov s6, w0
    \\fmov s5, w1
    \\dup v6.4s, v6.s[0]
    \\mov v6.s[1], v5.s[0]
    \\mov v6.s[3], v5.s[0]
    ++ cgemvNoTransFcmlaF32LoadCoeff4Asm() ++
        cgemvNoTransFcmlaF32Rows16LoopAsm(false) ++
        cgemvNoTransFcmlaF32Advance4ColumnsAsm() ++
        \\
        \\mov x3, #15
        \\
        \\1:
    ++ cgemvNoTransFcmlaF32LoadCoeff4Asm() ++
        cgemvNoTransFcmlaF32Rows16LoopAsm(true) ++
        cgemvNoTransFcmlaF32Advance4ColumnsAsm() ++
        \\
        \\subs x3, x3, #1
        \\b.ne 1b
        \\ret
    ;
}

fn zgemvNoTransFcmlaLoadCoeffAsm(comptime coeff: comptime_int, comptime offset: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ldr q0, [x6, #{d}]
        \\movi v{d}.2d, #0
        \\fcmla v{d}.2d, v0.2d, v6.2d, #0
        \\fcmla v{d}.2d, v0.2d, v6.2d, #90
    , .{ offset, coeff, coeff, coeff });
}

fn zgemvNoTransFcmlaColumnAsm(comptime ptr: comptime_int, comptime coeff: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ldr q0, [x{d}, x16]
        \\fcmla v16.2d, v0.2d, v{d}.2d, #0
        \\fcmla v16.2d, v0.2d, v{d}.2d, #90
    , .{ ptr, coeff, coeff });
}

fn zgemvNoTransFcmlaF64Rows8M128LoopAsm(comptime beta_scale: bool) []const u8 {
    const init_acc =
        if (beta_scale)
            \\
            \\ldr q0, [x7, x16]
            \\movi v16.2d, #0
            \\fcmla v16.2d, v0.2d, v7.2d, #0
            \\fcmla v16.2d, v0.2d, v7.2d, #90
            \\ldr q0, [x7, x2]
            \\movi v17.2d, #0
            \\fcmla v17.2d, v0.2d, v7.2d, #0
            \\fcmla v17.2d, v0.2d, v7.2d, #90
            \\ldr q0, [x7, x0]
            \\movi v18.2d, #0
            \\fcmla v18.2d, v0.2d, v7.2d, #0
            \\fcmla v18.2d, v0.2d, v7.2d, #90
            \\ldr q0, [x7, x1]
            \\movi v19.2d, #0
            \\fcmla v19.2d, v0.2d, v7.2d, #0
            \\fcmla v19.2d, v0.2d, v7.2d, #90
            \\ldr q0, [x7, x12]
            \\movi v20.2d, #0
            \\fcmla v20.2d, v0.2d, v7.2d, #0
            \\fcmla v20.2d, v0.2d, v7.2d, #90
            \\ldr q0, [x7, x13]
            \\movi v21.2d, #0
            \\fcmla v21.2d, v0.2d, v7.2d, #0
            \\fcmla v21.2d, v0.2d, v7.2d, #90
            \\ldr q0, [x7, x14]
            \\movi v22.2d, #0
            \\fcmla v22.2d, v0.2d, v7.2d, #0
            \\fcmla v22.2d, v0.2d, v7.2d, #90
            \\ldr q0, [x7, x15]
            \\movi v23.2d, #0
            \\fcmla v23.2d, v0.2d, v7.2d, #0
            \\fcmla v23.2d, v0.2d, v7.2d, #90
            \\
        else
            \\
            \\ldr q16, [x7, x16]
            \\ldr q17, [x7, x2]
            \\ldr q18, [x7, x0]
            \\ldr q19, [x7, x1]
            \\ldr q20, [x7, x12]
            \\ldr q21, [x7, x13]
            \\ldr q22, [x7, x14]
            \\ldr q23, [x7, x15]
            \\
        ;
    return
    \\
    \\mov x16, #0
    \\mov x17, #16
    \\
    \\2:
    \\add x2, x16, #16
    \\add x0, x16, #32
    \\add x1, x16, #48
    \\add x12, x16, #64
    \\add x13, x16, #80
    \\add x14, x16, #96
    \\add x15, x16, #112
    ++ init_acc ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(8, 24) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(9, 25) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(10, 26) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(11, 27) ++
        \\
        \\str q16, [x7, x16]
        \\str q17, [x7, x2]
        \\str q18, [x7, x0]
        \\str q19, [x7, x1]
        \\str q20, [x7, x12]
        \\str q21, [x7, x13]
        \\str q22, [x7, x14]
        \\str q23, [x7, x15]
        \\add x16, x16, #128
        \\subs x17, x17, #1
        \\b.ne 2b
    ;
}

pub fn zgemvNoTransFcmlaF64M128Asm() []const u8 {
    return
    \\
    \\fmov d6, x0
    \\fmov d5, x1
    \\fmov d7, x2
    \\fmov d4, x3
    \\mov v6.d[1], v5.d[0]
    \\mov v7.d[1], v4.d[0]
    ++ zgemvNoTransFcmlaF64LoadCoeff4Asm() ++
        zgemvNoTransFcmlaF64Rows8M128LoopAsm(true) ++
        zgemvNoTransFcmlaF64Advance4ColumnsAsm() ++
        \\
        \\mov x3, #31
        \\
        \\1:
    ++ zgemvNoTransFcmlaF64LoadCoeff4Asm() ++
        zgemvNoTransFcmlaF64Rows8M128LoopAsm(false) ++
        zgemvNoTransFcmlaF64Advance4ColumnsAsm() ++
        \\
        \\subs x3, x3, #1
        \\b.ne 1b
        \\ret
    ;
}

fn zgemvNoTransFcmlaTaskColumnRows8Asm(comptime ptr: comptime_int, comptime coeff: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\add x0, x{d}, x16
        \\ldp q0, q1, [x0]
        \\fcmla v16.2d, v0.2d, v{d}.2d, #0
        \\fcmla v16.2d, v0.2d, v{d}.2d, #90
        \\fcmla v17.2d, v1.2d, v{d}.2d, #0
        \\fcmla v17.2d, v1.2d, v{d}.2d, #90
        \\ldp q0, q1, [x0, #32]
        \\fcmla v18.2d, v0.2d, v{d}.2d, #0
        \\fcmla v18.2d, v0.2d, v{d}.2d, #90
        \\fcmla v19.2d, v1.2d, v{d}.2d, #0
        \\fcmla v19.2d, v1.2d, v{d}.2d, #90
        \\ldp q0, q1, [x0, #64]
        \\fcmla v20.2d, v0.2d, v{d}.2d, #0
        \\fcmla v20.2d, v0.2d, v{d}.2d, #90
        \\fcmla v21.2d, v1.2d, v{d}.2d, #0
        \\fcmla v21.2d, v1.2d, v{d}.2d, #90
        \\ldp q0, q1, [x0, #96]
        \\fcmla v22.2d, v0.2d, v{d}.2d, #0
        \\fcmla v22.2d, v0.2d, v{d}.2d, #90
        \\fcmla v23.2d, v1.2d, v{d}.2d, #0
        \\fcmla v23.2d, v1.2d, v{d}.2d, #90
    , .{
        ptr,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
        coeff,
    });
}

pub fn zgemvNoTransFcmlaF64M512N64TaskAsm() []const u8 {
    return
    \\
    \\fmov d6, x0
    \\fmov d5, x1
    \\mov v6.d[1], v5.d[0]
    \\mov x3, #16
    \\
    \\1:
    ++ zgemvNoTransFcmlaLoadCoeffAsm(24, 0) ++
        zgemvNoTransFcmlaLoadCoeffAsm(25, 16) ++
        zgemvNoTransFcmlaLoadCoeffAsm(26, 32) ++
        zgemvNoTransFcmlaLoadCoeffAsm(27, 48) ++
        \\
        \\mov x8, x4
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\mov x16, #0
        \\mov x17, #64
        \\
        \\2:
        \\add x2, x7, x16
        \\ldnp q16, q17, [x2]
        \\ldp q18, q19, [x2, #32]
        \\ldp q20, q21, [x2, #64]
        \\ldp q22, q23, [x2, #96]
    ++ zgemvNoTransFcmlaTaskColumnRows8Asm(8, 24) ++
        zgemvNoTransFcmlaTaskColumnRows8Asm(9, 25) ++
        zgemvNoTransFcmlaTaskColumnRows8Asm(10, 26) ++
        zgemvNoTransFcmlaTaskColumnRows8Asm(11, 27) ++
        \\
        \\stnp q16, q17, [x2]
        \\stp q18, q19, [x2, #32]
        \\stp q20, q21, [x2, #64]
        \\stp q22, q23, [x2, #96]
        \\add x16, x16, #128
        \\subs x17, x17, #1
        \\b.ne 2b
        \\
        \\lsl x0, x5, #2
        \\add x4, x4, x0
        \\add x6, x6, #64
        \\subs x3, x3, #1
        \\b.ne 1b
        \\ret
    ;
}

fn zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(comptime ptr: comptime_int, comptime coeff: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ldr q0, [x{d}, x16]
        \\fcmla v16.2d, v0.2d, v{d}.2d, #0
        \\fcmla v16.2d, v0.2d, v{d}.2d, #90
        \\ldr q0, [x{d}, x2]
        \\fcmla v17.2d, v0.2d, v{d}.2d, #0
        \\fcmla v17.2d, v0.2d, v{d}.2d, #90
        \\ldr q0, [x{d}, x0]
        \\fcmla v18.2d, v0.2d, v{d}.2d, #0
        \\fcmla v18.2d, v0.2d, v{d}.2d, #90
        \\ldr q0, [x{d}, x1]
        \\fcmla v19.2d, v0.2d, v{d}.2d, #0
        \\fcmla v19.2d, v0.2d, v{d}.2d, #90
        \\ldr q0, [x{d}, x12]
        \\fcmla v20.2d, v0.2d, v{d}.2d, #0
        \\fcmla v20.2d, v0.2d, v{d}.2d, #90
        \\ldr q0, [x{d}, x13]
        \\fcmla v21.2d, v0.2d, v{d}.2d, #0
        \\fcmla v21.2d, v0.2d, v{d}.2d, #90
        \\ldr q0, [x{d}, x14]
        \\fcmla v22.2d, v0.2d, v{d}.2d, #0
        \\fcmla v22.2d, v0.2d, v{d}.2d, #90
        \\ldr q0, [x{d}, x15]
        \\fcmla v23.2d, v0.2d, v{d}.2d, #0
        \\fcmla v23.2d, v0.2d, v{d}.2d, #90
    , .{
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
        ptr, coeff, coeff,
    });
}

pub fn zgemvNoTransFcmlaF64M512N64TaskIndexedAsm() []const u8 {
    return
    \\
    \\fmov d6, x0
    \\fmov d5, x1
    \\mov v6.d[1], v5.d[0]
    \\mov x3, #16
    \\
    \\1:
    ++ zgemvNoTransFcmlaLoadCoeffAsm(24, 0) ++
        zgemvNoTransFcmlaLoadCoeffAsm(25, 16) ++
        zgemvNoTransFcmlaLoadCoeffAsm(26, 32) ++
        zgemvNoTransFcmlaLoadCoeffAsm(27, 48) ++
        \\
        \\mov x8, x4
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\mov x16, #0
        \\mov x17, #64
        \\
        \\2:
        \\add x2, x16, #16
        \\add x0, x16, #32
        \\add x1, x16, #48
        \\add x12, x16, #64
        \\add x13, x16, #80
        \\add x14, x16, #96
        \\add x15, x16, #112
        \\ldr q16, [x7, x16]
        \\ldr q17, [x7, x2]
        \\ldr q18, [x7, x0]
        \\ldr q19, [x7, x1]
        \\ldr q20, [x7, x12]
        \\ldr q21, [x7, x13]
        \\ldr q22, [x7, x14]
        \\ldr q23, [x7, x15]
    ++ zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(8, 24) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(9, 25) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(10, 26) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(11, 27) ++
        \\
        \\str q16, [x7, x16]
        \\str q17, [x7, x2]
        \\str q18, [x7, x0]
        \\str q19, [x7, x1]
        \\str q20, [x7, x12]
        \\str q21, [x7, x13]
        \\str q22, [x7, x14]
        \\str q23, [x7, x15]
        \\add x16, x16, #128
        \\subs x17, x17, #1
        \\b.ne 2b
        \\
        \\lsl x0, x5, #2
        \\add x4, x4, x0
        \\add x6, x6, #64
        \\subs x3, x3, #1
        \\b.ne 1b
        \\ret
    ;
}

fn zgemvNoTransFcmlaF64Rows8IndexedLoopAsm(comptime load_existing: bool) []const u8 {
    const init_acc =
        if (load_existing)
            \\
            \\ldr q16, [x7, x16]
            \\ldr q17, [x7, x2]
            \\ldr q18, [x7, x0]
            \\ldr q19, [x7, x1]
            \\ldr q20, [x7, x12]
            \\ldr q21, [x7, x13]
            \\ldr q22, [x7, x14]
            \\ldr q23, [x7, x15]
            \\
        else
            \\
            \\movi v16.2d, #0
            \\movi v17.2d, #0
            \\movi v18.2d, #0
            \\movi v19.2d, #0
            \\movi v20.2d, #0
            \\movi v21.2d, #0
            \\movi v22.2d, #0
            \\movi v23.2d, #0
            \\
        ;
    return
    \\
    \\mov x16, #0
    \\mov x17, #64
    \\
    \\2:
    \\add x2, x16, #16
    \\add x0, x16, #32
    \\add x1, x16, #48
    \\add x12, x16, #64
    \\add x13, x16, #80
    \\add x14, x16, #96
    \\add x15, x16, #112
    ++ init_acc ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(8, 24) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(9, 25) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(10, 26) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(11, 27) ++
        \\
        \\str q16, [x7, x16]
        \\str q17, [x7, x2]
        \\str q18, [x7, x0]
        \\str q19, [x7, x1]
        \\str q20, [x7, x12]
        \\str q21, [x7, x13]
        \\str q22, [x7, x14]
        \\str q23, [x7, x15]
        \\add x16, x16, #128
        \\subs x17, x17, #1
        \\b.ne 2b
    ;
}

fn zgemvNoTransFcmlaF64LoadCoeff4Asm() []const u8 {
    return zgemvNoTransFcmlaLoadCoeffAsm(24, 0) ++
        zgemvNoTransFcmlaLoadCoeffAsm(25, 16) ++
        zgemvNoTransFcmlaLoadCoeffAsm(26, 32) ++
        zgemvNoTransFcmlaLoadCoeffAsm(27, 48) ++
        \\
        \\mov x8, x4
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
    ;
}

fn zgemvNoTransFcmlaF64Advance4ColumnsAsm() []const u8 {
    return
    \\
    \\lsl x0, x5, #2
    \\add x4, x4, x0
    \\add x6, x6, #64
    ;
}

pub fn zgemvNoTransFcmlaF64M512N64TaskNoMemsetAsm() []const u8 {
    return
    \\
    \\fmov d6, x0
    \\fmov d5, x1
    \\mov v6.d[1], v5.d[0]
    ++ zgemvNoTransFcmlaF64LoadCoeff4Asm() ++
        zgemvNoTransFcmlaF64Rows8IndexedLoopAsm(false) ++
        zgemvNoTransFcmlaF64Advance4ColumnsAsm() ++
        \\
        \\mov x3, #15
        \\
        \\1:
    ++ zgemvNoTransFcmlaF64LoadCoeff4Asm() ++
        zgemvNoTransFcmlaF64Rows8IndexedLoopAsm(true) ++
        zgemvNoTransFcmlaF64Advance4ColumnsAsm() ++
        \\
        \\subs x3, x3, #1
        \\b.ne 1b
        \\ret
    ;
}

pub fn zgemvNoTransFcmlaF64M512NTaskNoMemsetAsm() []const u8 {
    return
    \\
    \\fmov d6, x0
    \\fmov d5, x1
    \\mov v6.d[1], v5.d[0]
    \\mov x7, x6
    \\mov x6, x5
    \\mov x5, x4
    \\mov x4, x3
    \\mov x3, x2
    ++ zgemvNoTransFcmlaF64LoadCoeff4Asm() ++
        zgemvNoTransFcmlaF64Rows8IndexedLoopAsm(false) ++
        zgemvNoTransFcmlaF64Advance4ColumnsAsm() ++
        \\
        \\subs x3, x3, #1
        \\b.eq 3f
        \\
        \\1:
    ++ zgemvNoTransFcmlaF64LoadCoeff4Asm() ++
        zgemvNoTransFcmlaF64Rows8IndexedLoopAsm(true) ++
        zgemvNoTransFcmlaF64Advance4ColumnsAsm() ++
        \\
        \\subs x3, x3, #1
        \\b.ne 1b
        \\
        \\3:
        \\ret
    ;
}

fn zgemvNoTransFcmlaTaskColumnRowImmAsm(comptime acc: comptime_int, comptime coeff: comptime_int, comptime offset: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ldr q1, [x0, #{d}]
        \\fcmla v{d}.2d, v1.2d, v{d}.2d, #0
        \\fcmla v{d}.2d, v1.2d, v{d}.2d, #90
    , .{ offset, acc, coeff, acc, coeff });
}

fn zgemvNoTransFcmlaTaskColumnRows16Asm(comptime ptr: comptime_int, comptime coeff: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\add x0, x{d}, x16
    , .{ptr}) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(16, coeff, 0) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(17, coeff, 16) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(18, coeff, 32) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(19, coeff, 48) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(20, coeff, 64) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(21, coeff, 80) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(22, coeff, 96) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(23, coeff, 112) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(24, coeff, 128) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(25, coeff, 144) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(26, coeff, 160) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(27, coeff, 176) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(28, coeff, 192) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(29, coeff, 208) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(30, coeff, 224) ++
        zgemvNoTransFcmlaTaskColumnRowImmAsm(31, coeff, 240);
}

pub fn zgemvNoTransFcmlaF64M512N64TaskRow16Asm() []const u8 {
    return
    \\
    \\fmov d6, x0
    \\fmov d5, x1
    \\mov v6.d[1], v5.d[0]
    \\mov x3, #16
    \\
    \\1:
    ++ zgemvNoTransFcmlaLoadCoeffAsm(2, 0) ++
        zgemvNoTransFcmlaLoadCoeffAsm(3, 16) ++
        zgemvNoTransFcmlaLoadCoeffAsm(4, 32) ++
        zgemvNoTransFcmlaLoadCoeffAsm(5, 48) ++
        \\
        \\mov x8, x4
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\mov x16, #0
        \\mov x17, #32
        \\
        \\2:
        \\add x0, x7, x16
        \\ldr q16, [x0]
        \\ldr q17, [x0, #16]
        \\ldr q18, [x0, #32]
        \\ldr q19, [x0, #48]
        \\ldr q20, [x0, #64]
        \\ldr q21, [x0, #80]
        \\ldr q22, [x0, #96]
        \\ldr q23, [x0, #112]
        \\ldr q24, [x0, #128]
        \\ldr q25, [x0, #144]
        \\ldr q26, [x0, #160]
        \\ldr q27, [x0, #176]
        \\ldr q28, [x0, #192]
        \\ldr q29, [x0, #208]
        \\ldr q30, [x0, #224]
        \\ldr q31, [x0, #240]
    ++ zgemvNoTransFcmlaTaskColumnRows16Asm(8, 2) ++
        zgemvNoTransFcmlaTaskColumnRows16Asm(9, 3) ++
        zgemvNoTransFcmlaTaskColumnRows16Asm(10, 4) ++
        zgemvNoTransFcmlaTaskColumnRows16Asm(11, 5) ++
        \\
        \\add x0, x7, x16
        \\str q16, [x0]
        \\str q17, [x0, #16]
        \\str q18, [x0, #32]
        \\str q19, [x0, #48]
        \\str q20, [x0, #64]
        \\str q21, [x0, #80]
        \\str q22, [x0, #96]
        \\str q23, [x0, #112]
        \\str q24, [x0, #128]
        \\str q25, [x0, #144]
        \\str q26, [x0, #160]
        \\str q27, [x0, #176]
        \\str q28, [x0, #192]
        \\str q29, [x0, #208]
        \\str q30, [x0, #224]
        \\str q31, [x0, #240]
        \\add x16, x16, #256
        \\subs x17, x17, #1
        \\b.ne 2b
        \\
        \\lsl x0, x5, #2
        \\add x4, x4, x0
        \\add x6, x6, #64
        \\subs x3, x3, #1
        \\b.ne 1b
        \\ret
    ;
}

pub fn zgemvNoTransFcmlaF64M64N512RowsAsm() []const u8 {
    return
    \\
    \\fmov d6, x2
    \\fmov d5, x3
    \\mov v6.d[1], v5.d[0]
    \\mov x3, #128
    \\
    \\1:
    ++ zgemvNoTransFcmlaLoadCoeffAsm(24, 0) ++
        zgemvNoTransFcmlaLoadCoeffAsm(25, 16) ++
        zgemvNoTransFcmlaLoadCoeffAsm(26, 32) ++
        zgemvNoTransFcmlaLoadCoeffAsm(27, 48) ++
        \\
        \\mov x8, x4
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\mov x16, #0
        \\mov x17, #8
        \\
        \\2:
        \\add x2, x16, #16
        \\add x0, x16, #32
        \\add x1, x16, #48
        \\add x12, x16, #64
        \\add x13, x16, #80
        \\add x14, x16, #96
        \\add x15, x16, #112
        \\ldr q16, [x7, x16]
        \\ldr q17, [x7, x2]
        \\ldr q18, [x7, x0]
        \\ldr q19, [x7, x1]
        \\ldr q20, [x7, x12]
        \\ldr q21, [x7, x13]
        \\ldr q22, [x7, x14]
        \\ldr q23, [x7, x15]
    ++ zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(8, 24) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(9, 25) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(10, 26) ++
        zgemvNoTransFcmlaTaskColumnRows8IndexedAsm(11, 27) ++
        \\
        \\str q16, [x7, x16]
        \\str q17, [x7, x2]
        \\str q18, [x7, x0]
        \\str q19, [x7, x1]
        \\str q20, [x7, x12]
        \\str q21, [x7, x13]
        \\str q22, [x7, x14]
        \\str q23, [x7, x15]
        \\add x16, x16, #128
        \\subs x17, x17, #1
        \\b.ne 2b
        \\
        \\lsl x0, x5, #2
        \\add x4, x4, x0
        \\add x6, x6, #64
        \\subs x3, x3, #1
        \\b.ne 1b
        \\ret
    ;
}

fn zgemvTransFcmlaReduceStoreAsm(comptime acc: comptime_int, comptime offset: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ldr q5, [x7, #{d}]
        \\movi v0.2d, #0
        \\fcmla v0.2d, v5.2d, v7.2d, #0
        \\fcmla v0.2d, v5.2d, v7.2d, #90
        \\fcmla v0.2d, v{d}.2d, v6.2d, #0
        \\fcmla v0.2d, v{d}.2d, v6.2d, #90
        \\str q0, [x7, #{d}]
    , .{ offset, acc, acc, offset });
}

fn zgemvTransFcmlaReduceAddStoreAsm(comptime acc: comptime_int, comptime offset: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\movi v0.2d, #0
        \\fcmla v0.2d, v{d}.2d, v6.2d, #0
        \\fcmla v0.2d, v{d}.2d, v6.2d, #90
        \\ldr q5, [x7, #{d}]
        \\fadd v0.2d, v0.2d, v5.2d
        \\str q0, [x7, #{d}]
    , .{ acc, acc, offset, offset });
}

fn zgemvTransFcmlaStepAsm(comptime acc0: comptime_int, comptime acc1: comptime_int, comptime acc2: comptime_int, comptime acc3: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ldr q0, [x12], #16
        \\ldr q1, [x8], #16
        \\ldr q2, [x9], #16
        \\ldr q3, [x10], #16
        \\ldr q4, [x11], #16
        \\fcmla v{d}.2d, v1.2d, v0.2d, #0
        \\fcmla v{d}.2d, v2.2d, v0.2d, #0
        \\fcmla v{d}.2d, v3.2d, v0.2d, #0
        \\fcmla v{d}.2d, v4.2d, v0.2d, #0
        \\fcmla v{d}.2d, v1.2d, v0.2d, #90
        \\fcmla v{d}.2d, v2.2d, v0.2d, #90
        \\fcmla v{d}.2d, v3.2d, v0.2d, #90
        \\fcmla v{d}.2d, v4.2d, v0.2d, #90
    , .{ acc0, acc1, acc2, acc3, acc0, acc1, acc2, acc3 });
}

fn zgemvTransFcmlaF64MNColsAsm(comptime row_steps: comptime_int, comptime col_groups: comptime_int) []const u8 {
    const inner = zgemvTransFcmlaStepAsm(16, 17, 18, 19) ++
        zgemvTransFcmlaStepAsm(20, 21, 22, 23) ++
        zgemvTransFcmlaStepAsm(24, 25, 26, 27) ++
        zgemvTransFcmlaStepAsm(28, 29, 30, 31);
    const store_tail = zgemvTransFcmlaReduceStoreAsm(16, 0) ++
        zgemvTransFcmlaReduceStoreAsm(17, 16) ++
        zgemvTransFcmlaReduceStoreAsm(18, 32) ++
        zgemvTransFcmlaReduceStoreAsm(19, 48);
    return
    \\
    \\fmov d6, x0
    \\fmov d5, x1
    \\fmov d7, x2
    \\fmov d4, x3
    \\mov v6.d[1], v5.d[0]
    \\mov v7.d[1], v4.d[0]
    \\
    ++ std.fmt.comptimePrint(
        \\
        \\mov x14, #{d}
        \\
    , .{col_groups}) ++
        \\
        \\mov x15, x4
        \\
        \\1:
        \\movi v16.2d, #0
        \\movi v17.2d, #0
        \\movi v18.2d, #0
        \\movi v19.2d, #0
        \\movi v20.2d, #0
        \\movi v21.2d, #0
        \\movi v22.2d, #0
        \\movi v23.2d, #0
        \\movi v24.2d, #0
        \\movi v25.2d, #0
        \\movi v26.2d, #0
        \\movi v27.2d, #0
        \\movi v28.2d, #0
        \\movi v29.2d, #0
        \\movi v30.2d, #0
        \\movi v31.2d, #0
        \\mov x8, x15
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\mov x12, x6
    ++ std.fmt.comptimePrint(
        \\
        \\mov x13, #{d}
        \\
    , .{row_steps}) ++
        \\
        \\2:
    ++ inner ++
        \\
        \\subs x13, x13, #1
        \\b.ne 2b
        \\fadd v16.2d, v16.2d, v20.2d
        \\fadd v17.2d, v17.2d, v21.2d
        \\fadd v18.2d, v18.2d, v22.2d
        \\fadd v19.2d, v19.2d, v23.2d
        \\fadd v24.2d, v24.2d, v28.2d
        \\fadd v25.2d, v25.2d, v29.2d
        \\fadd v26.2d, v26.2d, v30.2d
        \\fadd v27.2d, v27.2d, v31.2d
        \\fadd v16.2d, v16.2d, v24.2d
        \\fadd v17.2d, v17.2d, v25.2d
        \\fadd v18.2d, v18.2d, v26.2d
        \\fadd v19.2d, v19.2d, v27.2d
    ++ store_tail ++
        \\
        \\lsl x16, x5, #2
        \\add x15, x15, x16
        \\add x7, x7, #64
        \\subs x14, x14, #1
        \\b.ne 1b
        \\ret
    ;
}

pub fn zgemvTransFcmlaF64M128Asm() []const u8 {
    return zgemvTransFcmlaF64MNColsAsm(32, 32);
}

pub fn zgemvTransFcmlaF64M256N128TaskAsm() []const u8 {
    return zgemvTransFcmlaF64MNColsAsm(64, 32);
}

fn zgemvTransFcmlaF64M512N64TaskBodyAsm(comptime with_beta: bool) []const u8 {
    const beta_setup =
        if (with_beta)
            \\
            \\fmov d7, x2
            \\fmov d4, x3
            \\
        else
            "";
    const beta_pack =
        if (with_beta)
            \\
            \\mov v7.d[1], v4.d[0]
            \\
        else
            "";
    const store_tail =
        if (with_beta)
            zgemvTransFcmlaReduceStoreAsm(16, 0) ++
                zgemvTransFcmlaReduceStoreAsm(17, 16) ++
                zgemvTransFcmlaReduceStoreAsm(18, 32) ++
                zgemvTransFcmlaReduceStoreAsm(19, 48)
        else
            zgemvTransFcmlaReduceAddStoreAsm(16, 0) ++
                zgemvTransFcmlaReduceAddStoreAsm(17, 16) ++
                zgemvTransFcmlaReduceAddStoreAsm(18, 32) ++
                zgemvTransFcmlaReduceAddStoreAsm(19, 48);
    return
    \\
    \\fmov d6, x0
    \\fmov d5, x1
    ++ beta_setup ++
        \\
        \\mov v6.d[1], v5.d[0]
    ++ beta_pack ++
        \\
        \\
        \\mov x14, #16
        \\mov x15, x4
        \\
        \\1:
        \\movi v16.2d, #0
        \\movi v17.2d, #0
        \\movi v18.2d, #0
        \\movi v19.2d, #0
        \\movi v20.2d, #0
        \\movi v21.2d, #0
        \\movi v22.2d, #0
        \\movi v23.2d, #0
        \\movi v24.2d, #0
        \\movi v25.2d, #0
        \\movi v26.2d, #0
        \\movi v27.2d, #0
        \\movi v28.2d, #0
        \\movi v29.2d, #0
        \\movi v30.2d, #0
        \\movi v31.2d, #0
        \\mov x8, x15
        \\add x9, x8, x5
        \\add x10, x9, x5
        \\add x11, x10, x5
        \\mov x12, x6
        \\mov x13, #128
        \\
        \\2:
    ++ zgemvTransFcmlaStepAsm(16, 17, 18, 19) ++
        zgemvTransFcmlaStepAsm(20, 21, 22, 23) ++
        zgemvTransFcmlaStepAsm(24, 25, 26, 27) ++
        zgemvTransFcmlaStepAsm(28, 29, 30, 31) ++
        \\
        \\subs x13, x13, #1
        \\b.ne 2b
        \\fadd v16.2d, v16.2d, v20.2d
        \\fadd v17.2d, v17.2d, v21.2d
        \\fadd v18.2d, v18.2d, v22.2d
        \\fadd v19.2d, v19.2d, v23.2d
        \\fadd v24.2d, v24.2d, v28.2d
        \\fadd v25.2d, v25.2d, v29.2d
        \\fadd v26.2d, v26.2d, v30.2d
        \\fadd v27.2d, v27.2d, v31.2d
        \\fadd v16.2d, v16.2d, v24.2d
        \\fadd v17.2d, v17.2d, v25.2d
        \\fadd v18.2d, v18.2d, v26.2d
        \\fadd v19.2d, v19.2d, v27.2d
    ++ store_tail ++
        \\
        \\lsl x16, x5, #2
        \\add x15, x15, x16
        \\add x7, x7, #64
        \\subs x14, x14, #1
        \\b.ne 1b
        \\ret
    ;
}

pub fn zgemvTransFcmlaF64M512N64TaskAsm() []const u8 {
    return zgemvTransFcmlaF64M512N64TaskBodyAsm(false);
}

pub fn zgemvTransFcmlaF64M512N64TaskBetaAsm() []const u8 {
    return zgemvTransFcmlaF64M512N64TaskBodyAsm(true);
}

pub fn smeGemmPanel1mAsm(comptime lane: []const u8) []const u8 {
    const suffix = laneLoadSuffix(lane);
    const shift = laneIndexShift(lane);
    const store_stride_setup = if (std.mem.eql(u8, lane, "s"))
        \\
        \\mov x11, x6
    else
        "";
    const store_stride_reg = if (std.mem.eql(u8, lane, "s")) "x11" else "x6";
    return read_svl_x14 ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\lsl x3, x3, #{d}
        \\mov x7, #0
        \\1:
        \\cmp x7, x3
        \\b.hs 5f
        \\zero {{ za0.{s} }}
        \\
        \\add x10, x0, x7
        \\mov x11, x1
        \\mov x9, x4
        \\2:
        \\cbz x9, 3f
        \\ld1{s} {{ z0.{s} }}, p0/z, [x10]
        \\ld1{s} {{ z1.{s} }}, p0/z, [x11]
        \\fmopa za0.{s}, p0/m, p0/m, z0.{s}, z1.{s}
        \\add x10, x10, x5
        \\add x11, x11, x14
        \\sub x9, x9, #1
        \\b 2b
        \\
        \\3:
        \\add x10, x2, x7{s}
        \\mov w12, #0
        \\lsr x13, x14, #{d}
        \\4:
        \\cbz x13, 6f
        \\st1{s} {{ za0v.{s}[w12, 0] }}, p0, [x10]
        \\add x10, x10, {s}
        \\add w12, w12, #1
        \\sub x13, x13, #1
        \\b 4b
        \\
        \\6:
        \\add x7, x7, x14
        \\b 1b
        \\
        \\5:
        \\ret
    , .{
        shift,  lane,
        suffix, lane,
        suffix, lane,
        lane,   lane,
        lane,   store_stride_setup,
        shift,  suffix,
        lane,   store_stride_reg,
    });
}

pub fn smeGemmPanel4mAsm(comptime lane: []const u8) []const u8 {
    const suffix = laneLoadSuffix(lane);
    const shift = laneIndexShift(lane);
    return read_svl_x14 ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\lsl x3, x3, #{d}
        \\mov x7, #0
        \\13:
        \\cmp x7, x3
        \\b.hs 17f
        \\zero {{ za0.{s}, za1.{s}, za2.{s}, za3.{s} }}
        \\
        \\add x10, x0, x7
        \\mov x11, x1
        \\mov x9, x4
        \\14:
        \\cbz x9, 15f
        \\ld1{s} {{ z0.{s} }}, p0/z, [x10]
        \\add x13, x10, x14
        \\ld1{s} {{ z1.{s} }}, p0/z, [x13]
        \\add x13, x10, x14, lsl #1
        \\ld1{s} {{ z2.{s} }}, p0/z, [x13]
        \\add x13, x10, x14
        \\add x13, x13, x14, lsl #1
        \\ld1{s} {{ z3.{s} }}, p0/z, [x13]
        \\ld1{s} {{ z4.{s} }}, p0/z, [x11]
    , .{
        shift,
        lane,
        lane,
        lane,
        lane,
        suffix,
        lane,
        suffix,
        lane,
        suffix,
        lane,
        suffix,
        lane,
        suffix,
        lane,
    }) ++ std.fmt.comptimePrint(
        \\
        \\fmopa za0.{s}, p0/m, p0/m, z0.{s}, z4.{s}
        \\fmopa za1.{s}, p0/m, p0/m, z1.{s}, z4.{s}
        \\fmopa za2.{s}, p0/m, p0/m, z2.{s}, z4.{s}
        \\fmopa za3.{s}, p0/m, p0/m, z3.{s}, z4.{s}
        \\add x10, x10, x5
        \\add x11, x11, x14
        \\sub x9, x9, #1
        \\b 14b
        \\
    , .{
        lane, lane, lane,
        lane, lane, lane,
        lane, lane, lane,
        lane, lane, lane,
    }) ++ smeZaVectorStoreLoopAsm(lane, "15", "16", "18", .{
        .{ .za = 0, .base = "x10", .address_setup = "" },
        .{
            .za = 1,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x14
            ,
        },
        .{
            .za = 2,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x14, lsl #1
            ,
        },
        .{
            .za = 3,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x14
            \\add x16, x16, x14, lsl #1
            ,
        },
    }) ++
        \\
        \\add x7, x7, x14, lsl #2
        \\b 13b
        \\
        \\17:
        \\ret
    ;
}

fn smeStepLoadAsm(
    comptime lane: []const u8,
    comptime z_reg: comptime_int,
    comptime base: []const u8,
    comptime address_setup: []const u8,
) []const u8 {
    const suffix = laneLoadSuffix(lane);
    return address_setup ++ std.fmt.comptimePrint(
        \\
        \\ld1{s} {{ z{d}.{s} }}, p0/z, [{s}]
    , .{ suffix, z_reg, lane, base });
}

fn smeStepLoadsAsm(comptime lane: []const u8, comptime loads: anytype) []const u8 {
    comptime var text: []const u8 = "";
    inline for (loads) |load| {
        text = text ++ smeStepLoadAsm(lane, load.z, load.base, load.address_setup);
    }
    return text;
}

fn smeStepFmopaAsm(
    comptime lane: []const u8,
    comptime za: comptime_int,
    comptime lhs_z: comptime_int,
    comptime rhs_z: comptime_int,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\fmopa za{d}.{s}, p0/m, p0/m, z{d}.{s}, z{d}.{s}
    , .{ za, lane, lhs_z, lane, rhs_z, lane });
}

fn smeStepFmopasAsm(comptime lane: []const u8, comptime fmopas: anytype) []const u8 {
    comptime var text: []const u8 = "";
    inline for (fmopas) |op| {
        text = text ++ smeStepFmopaAsm(lane, op.za, op.lhs, op.rhs);
    }
    return text;
}

fn smeFmopaStepAsm(comptime lane: []const u8, comptime loads: anytype, comptime fmopas: anytype) []const u8 {
    return smeStepLoadsAsm(lane, loads) ++ smeStepFmopasAsm(lane, fmopas);
}

fn smeGemmPanel2x2StepAsm(comptime lane: []const u8) []const u8 {
    return smeFmopaStepAsm(lane, .{
        .{ .z = 0, .base = "x10", .address_setup = "" },
        .{
            .z = 1,
            .base = "x13",
            .address_setup =
            \\
            \\add x13, x10, x14
            ,
        },
        .{ .z = 2, .base = "x11", .address_setup = "" },
        .{
            .z = 3,
            .base = "x13",
            .address_setup =
            \\
            \\add x13, x11, x14
            ,
        },
    }, .{
        .{ .za = 0, .lhs = 0, .rhs = 2 },
        .{ .za = 1, .lhs = 1, .rhs = 2 },
        .{ .za = 2, .lhs = 0, .rhs = 3 },
        .{ .za = 3, .lhs = 1, .rhs = 3 },
    });
}

pub fn smeGemmPanel2x2Asm(comptime lane: []const u8) []const u8 {
    const shift = laneIndexShift(lane);
    const k_step = smeGemmPanel2x2StepAsm(lane);
    const advance = smeGemmPanelNx2AdvanceAsm();
    return read_svl_x14 ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\lsr x15, x14, #{d}
        \\mul x15, x15, x6
        \\
        \\lsl x3, x3, #{d}
        \\mov x7, #0
        \\31:
        \\cmp x7, x3
        \\b.hs 35f
        \\zero {{ za0.{s}, za1.{s}, za2.{s}, za3.{s} }}
        \\
        \\add x10, x0, x7
        \\mov x11, x1
        \\mov x9, x4
        \\32:
        \\cmp x9, #2
        \\b.lo 43f
    , .{
        shift, shift,
        lane,  lane,
        lane,  lane,
    }) ++ k_step ++ advance ++ k_step ++ advance ++
        \\
        \\sub x9, x9, #2
        \\b 32b
        \\
        \\43:
        \\cbz x9, 33f
    ++ k_step ++ smeGemmPanel2x2StoreAsm(lane, "33", "34", "36") ++
        \\
        \\add x7, x7, x14, lsl #1
        \\b 31b
        \\
        \\35:
        \\ret
    ;
}

fn smeGemmPanelNx2AdvanceAsm() []const u8 {
    return
    \\
    \\add x10, x10, x5
    \\add x11, x11, x14, lsl #1
    ;
}

fn zaVectorStoreAsm(
    comptime lane: []const u8,
    comptime za: comptime_int,
    comptime base: []const u8,
    comptime address_setup: []const u8,
) []const u8 {
    const suffix = laneLoadSuffix(lane);
    return address_setup ++ std.fmt.comptimePrint(
        \\
        \\st1{s} {{ za{d}v.{s}[w12, 0] }}, p0, [{s}]
    , .{ suffix, za, lane, base });
}

fn zaVectorStoreSeqAsm(comptime lane: []const u8, comptime stores: anytype) []const u8 {
    comptime var text: []const u8 = "";
    inline for (stores) |store| {
        text = text ++ zaVectorStoreAsm(lane, store.za, store.base, store.address_setup);
    }
    return text;
}

fn smeZaVectorStoreLoopAsm(
    comptime lane: []const u8,
    comptime start: []const u8,
    comptime loop: []const u8,
    comptime done: []const u8,
    comptime stores: anytype,
) []const u8 {
    const shift = laneIndexShift(lane);
    return std.fmt.comptimePrint(
        \\
        \\{s}:
        \\add x10, x2, x7
        \\mov w12, #0
        \\lsr x13, x14, #{d}
        \\{s}:
        \\cbz x13, {s}f
    , .{ start, shift, loop, done }) ++ zaVectorStoreSeqAsm(lane, stores) ++ std.fmt.comptimePrint(
        \\
        \\add x10, x10, x6
        \\add w12, w12, #1
        \\sub x13, x13, #1
        \\b {s}b
        \\
        \\{s}:
    , .{
        loop,
        done,
    });
}

fn smeGemmPanel2x2StoreAsm(comptime lane: []const u8, comptime start: []const u8, comptime loop: []const u8, comptime done: []const u8) []const u8 {
    return smeZaVectorStoreLoopAsm(lane, start, loop, done, .{
        .{ .za = 0, .base = "x10", .address_setup = "" },
        .{
            .za = 1,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x14
            ,
        },
        .{
            .za = 2,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x15
            ,
        },
        .{
            .za = 3,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x16, x14
            ,
        },
    });
}

fn smeGemmPanel2x2RepeatedStepsAsm(comptime lane: []const u8, comptime count: comptime_int) []const u8 {
    comptime var text: []const u8 = "";
    inline for (0..count) |_| {
        text = text ++ smeGemmPanel2x2StepAsm(lane) ++ smeGemmPanelNx2AdvanceAsm();
    }
    return text;
}

pub fn smeGemmPanels2x2Asm(comptime lane: []const u8, comptime k_unroll: comptime_int, comptime prefetch_b: bool) []const u8 {
    if (k_unroll != 2 and k_unroll != 4) @compileError("SME 2x2 panel batch supports K unroll 2 or 4");
    const shift = laneIndexShift(lane);
    const labels = if (k_unroll == 4)
        .{
            .outer = "90",
            .row_loop = "91",
            .row_done = "98",
            .done = "99",
            .k_loop = "92",
            .tail2 = "93",
            .tail1 = "94",
            .store_start = "95",
            .store_loop = "96",
            .store_done = "97",
        }
    else
        .{
            .outer = "73",
            .row_loop = "74",
            .row_done = "80",
            .done = "81",
            .k_loop = "75",
            .tail2 = "82",
            .tail1 = "",
            .store_start = "76",
            .store_loop = "77",
            .store_done = "79",
        };
    const prefetch = if (prefetch_b)
        \\
        \\prfm pldl1keep, [x11, #512]
    else
        "";
    const tail = if (k_unroll == 4) blk: {
        break :blk std.fmt.comptimePrint(
            \\
            \\cmp x9, #2
            \\b.lo {s}f
        , .{labels.tail1}) ++ smeGemmPanel2x2RepeatedStepsAsm(lane, 2) ++ std.fmt.comptimePrint(
            \\
            \\sub x9, x9, #2
            \\{s}:
            \\cbz x9, {s}f
        , .{ labels.tail1, labels.store_start }) ++ smeGemmPanel2x2StepAsm(lane);
    } else blk: {
        break :blk std.fmt.comptimePrint(
            \\
            \\cbz x9, {s}f
        , .{labels.store_start}) ++ smeGemmPanel2x2StepAsm(lane);
    };
    return
    \\
    \\stp x19, x20, [sp, #-16]!
    ++ read_svl_x14 ++ std.fmt.comptimePrint(
        \\
        \\lsr x15, x14, #{d}
        \\mul x15, x15, x6
        \\lsl x17, x14, #1
        \\mul x17, x17, x4
        \\lsl x8, x15, #1
        \\mov x19, x7
        \\lsl x20, x3, #{d}
        \\
    , .{ shift, shift }) ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\{s}:
        \\cbz x19, {s}f
        \\mov x7, #0
        \\{s}:
        \\cmp x7, x20
        \\b.hs {s}f
        \\zero {{ za0.{s}, za1.{s}, za2.{s}, za3.{s} }}
        \\
        \\add x10, x0, x7
        \\mov x11, x1
        \\mov x9, x4
        \\{s}:
        \\cmp x9, #{d}
        \\b.lo {s}f{s}
    , .{
        labels.outer,    labels.done,
        labels.row_loop, labels.row_done,
        lane,            lane,
        lane,            lane,
        labels.k_loop,   k_unroll,
        labels.tail2,    prefetch,
    }) ++ smeGemmPanel2x2RepeatedStepsAsm(lane, k_unroll) ++ std.fmt.comptimePrint(
        \\
        \\sub x9, x9, #{d}
        \\b {s}b
        \\
        \\{s}:
    , .{ k_unroll, labels.k_loop, labels.tail2 }) ++ tail ++ smeGemmPanel2x2StoreAsm(lane, labels.store_start, labels.store_loop, labels.store_done) ++ std.fmt.comptimePrint(
        \\
        \\add x7, x7, x14, lsl #1
        \\b {s}b
        \\
        \\{s}:
        \\add x1, x1, x17
        \\add x2, x2, x8
        \\sub x19, x19, #1
        \\b {s}b
        \\
        \\{s}:
        \\ldp x19, x20, [sp], #16
        \\ret
    , .{
        labels.row_loop,
        labels.row_done,
        labels.outer,
        labels.done,
    });
}

fn smeGemmPanel1x2StepAsm(comptime lane: []const u8) []const u8 {
    return smeFmopaStepAsm(lane, .{
        .{ .z = 0, .base = "x10", .address_setup = "" },
        .{ .z = 2, .base = "x11", .address_setup = "" },
        .{
            .z = 3,
            .base = "x13",
            .address_setup =
            \\
            \\add x13, x11, x14
            ,
        },
    }, .{
        .{ .za = 0, .lhs = 0, .rhs = 2 },
        .{ .za = 1, .lhs = 0, .rhs = 3 },
    });
}

pub fn smeGemmPanel1x2Asm(comptime lane: []const u8) []const u8 {
    const shift = laneIndexShift(lane);
    const k_step = smeGemmPanel1x2StepAsm(lane);
    const advance = smeGemmPanelNx2AdvanceAsm();
    return read_svl_x14 ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\lsr x15, x14, #{d}
        \\mul x15, x15, x6
        \\
        \\lsl x3, x3, #{d}
        \\mov x7, #0
        \\83:
        \\cmp x7, x3
        \\b.hs 87f
        \\zero {{ za0.{s}, za1.{s} }}
        \\
        \\add x10, x0, x7
        \\mov x11, x1
        \\mov x9, x4
        \\84:
        \\cmp x9, #2
        \\b.lo 88f
    , .{
        shift, shift,
        lane,  lane,
    }) ++ k_step ++ advance ++ k_step ++ advance ++
        \\
        \\sub x9, x9, #2
        \\b 84b
        \\
        \\88:
        \\cbz x9, 85f
    ++ k_step ++ smeZaVectorStoreLoopAsm(lane, "85", "86", "89", .{
        .{ .za = 0, .base = "x10", .address_setup = "" },
        .{
            .za = 1,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x15
            ,
        },
    }) ++
        \\
        \\add x7, x7, x14
        \\b 83b
        \\
        \\87:
        \\ret
    ;
}

fn smeGemmPanel4x2F64StepAsm() []const u8 {
    return smeFmopaStepAsm("d", .{
        .{ .z = 0, .base = "x10", .address_setup = "" },
        .{
            .z = 1,
            .base = "x13",
            .address_setup =
            \\
            \\add x13, x10, x14
            ,
        },
        .{
            .z = 2,
            .base = "x13",
            .address_setup =
            \\
            \\add x13, x10, x14, lsl #1
            ,
        },
        .{
            .z = 3,
            .base = "x13",
            .address_setup =
            \\
            \\add x13, x10, x14
            \\add x13, x13, x14, lsl #1
            ,
        },
        .{ .z = 4, .base = "x11", .address_setup = "" },
        .{
            .z = 5,
            .base = "x13",
            .address_setup =
            \\
            \\add x13, x11, x14
            ,
        },
    }, .{
        .{ .za = 0, .lhs = 0, .rhs = 4 },
        .{ .za = 1, .lhs = 1, .rhs = 4 },
        .{ .za = 2, .lhs = 2, .rhs = 4 },
        .{ .za = 3, .lhs = 3, .rhs = 4 },
        .{ .za = 4, .lhs = 0, .rhs = 5 },
        .{ .za = 5, .lhs = 1, .rhs = 5 },
        .{ .za = 6, .lhs = 2, .rhs = 5 },
        .{ .za = 7, .lhs = 3, .rhs = 5 },
    });
}

fn smeGemmPanel4x2F64RepeatedStepsAsm(comptime count: comptime_int) []const u8 {
    comptime var text: []const u8 = "";
    inline for (0..count) |_| {
        text = text ++ smeGemmPanel4x2F64StepAsm() ++ smeGemmPanelNx2AdvanceAsm();
    }
    return text;
}

fn smeGemmPanel4x2F64StoreAsm(comptime start: []const u8, comptime loop: []const u8, comptime done: []const u8) []const u8 {
    return smeZaVectorStoreLoopAsm("d", start, loop, done, .{
        .{ .za = 0, .base = "x10", .address_setup = "" },
        .{
            .za = 1,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x14
            ,
        },
        .{
            .za = 2,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x14, lsl #1
            ,
        },
        .{
            .za = 3,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x14
            \\add x16, x16, x14, lsl #1
            ,
        },
        .{
            .za = 4,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x15
            ,
        },
        .{
            .za = 5,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x16, x14
            ,
        },
        .{
            .za = 6,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x15
            \\add x16, x16, x14, lsl #1
            ,
        },
        .{
            .za = 7,
            .base = "x16",
            .address_setup =
            \\
            \\add x16, x10, x15
            \\add x16, x16, x14
            \\add x16, x16, x14, lsl #1
            ,
        },
    });
}

fn smeGemmPanel4x2F64RowsAsm(comptime labels: anytype, comptime row_limit: []const u8, comptime after_rows: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\{s}:
        \\cmp x7, {s}
        \\b.hs {s}f
        \\zero {{ za0.d, za1.d, za2.d, za3.d, za4.d, za5.d, za6.d, za7.d }}
        \\
        \\add x10, x0, x7
        \\mov x11, x1
        \\mov x9, x4
        \\{s}:
        \\cmp x9, #2
        \\b.lo {s}f
        \\prfm pldl1keep, [x11, #512]
    , .{
        labels.row_loop,
        row_limit,
        labels.row_done,
        labels.k_loop,
        labels.tail,
    }) ++ smeGemmPanel4x2F64RepeatedStepsAsm(2) ++ std.fmt.comptimePrint(
        \\
        \\sub x9, x9, #2
        \\b {s}b
        \\
        \\{s}:
        \\cbz x9, {s}f
        \\prfm pldl1keep, [x11, #512]
    , .{
        labels.k_loop,
        labels.tail,
        labels.store_start,
    }) ++ smeGemmPanel4x2F64StepAsm() ++ smeGemmPanel4x2F64StoreAsm(labels.store_start, labels.store_loop, labels.store_done) ++ std.fmt.comptimePrint(
        \\
        \\add x7, x7, x14, lsl #2
        \\b {s}b
        \\
        \\{s}:
        \\{s}
    , .{
        labels.row_loop,
        labels.row_done,
        after_rows,
    });
}

pub fn smeGemmPanel4x2F64Asm() []const u8 {
    const labels = .{
        .row_loop = "45",
        .row_done = "49",
        .k_loop = "46",
        .tail = "51",
        .store_start = "47",
        .store_loop = "48",
        .store_done = "50",
    };
    return read_svl_x14 ++ ptrue("p0", "d") ++
        \\
        \\lsr x15, x14, #3
        \\mul x15, x15, x6
        \\
        \\lsl x3, x3, #3
        \\mov x7, #0
    ++ smeGemmPanel4x2F64RowsAsm(labels, "x3",
        \\
        \\ret
    );
}

pub fn smeGemmPanels4x2F64Asm() []const u8 {
    const labels = .{
        .row_loop = "65",
        .row_done = "71",
        .k_loop = "66",
        .tail = "69",
        .store_start = "67",
        .store_loop = "68",
        .store_done = "70",
    };
    return
    \\
    \\stp x19, x20, [sp, #-16]!
    ++ read_svl_x14 ++
        \\
        \\lsr x15, x14, #3
        \\mul x15, x15, x6
        \\lsl x17, x14, #1
        \\mul x17, x17, x4
        \\lsl x8, x15, #1
        \\mov x19, x7
        \\lsl x20, x3, #3
    ++ ptrue("p0", "d") ++
        \\
        \\64:
        \\cbz x19, 72f
        \\mov x7, #0
    ++ smeGemmPanel4x2F64RowsAsm(labels, "x20",
        \\
        \\add x1, x1, x17
        \\add x2, x2, x8
        \\sub x19, x19, #1
        \\b 64b
    ) ++
        \\
        \\72:
        \\ldp x19, x20, [sp], #16
        \\ret
    ;
}

fn ld1Seq(
    comptime lane: []const u8,
    comptime first_z: comptime_int,
    comptime count: comptime_int,
    comptime pred: []const u8,
    comptime base: []const u8,
    comptime first_vl_offset: comptime_int,
) []const u8 {
    comptime var text: []const u8 = "";
    if (first_vl_offset + count > 8) {
        text = text ++ std.fmt.comptimePrint(
            \\
            \\addvl x9, {s}, #8
        , .{base});
    }
    inline for (0..count) |i| {
        const offset = first_vl_offset + i;
        text = text ++ if (offset < 8)
            ld1(lane, first_z + i, pred, base, offset)
        else
            ld1(lane, first_z + i, pred, "x9", offset - 8);
    }
    return text;
}

fn st1Seq(
    comptime lane: []const u8,
    comptime first_z: comptime_int,
    comptime count: comptime_int,
    comptime pred: []const u8,
    comptime base: []const u8,
    comptime first_vl_offset: comptime_int,
) []const u8 {
    comptime var text: []const u8 = "";
    if (first_vl_offset + count > 8) {
        text = text ++ std.fmt.comptimePrint(
            \\
            \\addvl x10, {s}, #8
        , .{base});
    }
    inline for (0..count) |i| {
        const offset = first_vl_offset + i;
        text = text ++ if (offset < 8)
            st1(lane, first_z + i, pred, base, offset)
        else
            st1(lane, first_z + i, pred, "x10", offset - 8);
    }
    return text;
}

fn dupZeroSeq(comptime lane: []const u8, comptime first_z: comptime_int, comptime count: comptime_int) []const u8 {
    comptime var text: []const u8 = "";
    inline for (0..count) |i| {
        text = text ++ std.fmt.comptimePrint(
            \\
            \\dup z{d}.{s}, #0
        , .{ first_z + i, lane });
    }
    return text;
}

fn faddSeq(
    comptime lane: []const u8,
    comptime acc_first: comptime_int,
    comptime src_first: comptime_int,
    comptime count: comptime_int,
) []const u8 {
    comptime var text: []const u8 = "";
    inline for (0..count) |i| {
        text = text ++ std.fmt.comptimePrint(
            \\
            \\fadd z{d}.{s}, z{d}.{s}, z{d}.{s}
        , .{ acc_first + i, lane, acc_first + i, lane, src_first + i, lane });
    }
    return text;
}

fn reduceAccumulators(comptime lane: []const u8, comptime first_z: comptime_int, comptime count: comptime_int) []const u8 {
    comptime var text: []const u8 = "";
    comptime var step: comptime_int = 1;
    while (step < count) : (step *= 2) {
        comptime var i: comptime_int = 0;
        while (i + step < count) : (i += step * 2) {
            text = text ++ std.fmt.comptimePrint(
                \\
                \\fadd z{d}.{s}, z{d}.{s}, z{d}.{s}
            , .{ first_z + i, lane, first_z + i, lane, first_z + i + step, lane });
        }
    }
    return text;
}

fn addvlAdvance(comptime reg: []const u8, comptime count: comptime_int) []const u8 {
    if (count <= 16) {
        return std.fmt.comptimePrint(
            \\
            \\addvl {s}, {s}, #{d}
        , .{ reg, reg, count });
    }
    if (count == 32) {
        return std.fmt.comptimePrint(
            \\
            \\addvl {s}, {s}, #16
            \\addvl {s}, {s}, #16
        , .{ reg, reg, reg, reg });
    }
    @compileError("unsupported addvl advance");
}

fn complexLd2(
    comptime lane: []const u8,
    comptime real_z: comptime_int,
    comptime imag_z: comptime_int,
    comptime pred: []const u8,
    comptime base: []const u8,
    comptime vl_offset: comptime_int,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ld2{s} {{ z{d}.{s}, z{d}.{s} }}, {s}/z, {s}
    , .{ laneLoadSuffix(lane), real_z, lane, imag_z, lane, pred, vlAddr(base, vl_offset) });
}

fn complexSt2(
    comptime lane: []const u8,
    comptime real_z: comptime_int,
    comptime imag_z: comptime_int,
    comptime pred: []const u8,
    comptime base: []const u8,
    comptime vl_offset: comptime_int,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\st2{s} {{ z{d}.{s}, z{d}.{s} }}, {s}, {s}
    , .{ laneLoadSuffix(lane), real_z, lane, imag_z, lane, pred, vlAddr(base, vl_offset) });
}

fn complexFmla(
    comptime lane: []const u8,
    comptime pred: []const u8,
    comptime acc_re: comptime_int,
    comptime acc_im: comptime_int,
    comptime x_re: comptime_int,
    comptime x_im: comptime_int,
    comptime y_re: comptime_int,
    comptime y_im: comptime_int,
    comptime conj_x: bool,
) []const u8 {
    const real_imag_op = if (conj_x) "fmla" else "fmls";
    const imag_imag_op = if (conj_x) "fmls" else "fmla";
    return std.fmt.comptimePrint(
        \\
        \\fmla z{d}.{s}, {s}/m, z{d}.{s}, z{d}.{s}
        \\{s} z{d}.{s}, {s}/m, z{d}.{s}, z{d}.{s}
        \\fmla z{d}.{s}, {s}/m, z{d}.{s}, z{d}.{s}
        \\{s} z{d}.{s}, {s}/m, z{d}.{s}, z{d}.{s}
    , .{
        acc_re,       lane,         pred,   x_re, lane, y_re, lane,
        real_imag_op, acc_re,       lane,   pred, x_im, lane, y_im,
        lane,         acc_im,       lane,   pred, x_re, lane, y_im,
        lane,         imag_imag_op, acc_im, lane, pred, x_im, lane,
        y_re,         lane,
    });
}

fn complexAxpyOp(comptime lane: []const u8, comptime pred: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\fmla z16.{s}, {s}/m, z4.{s}, z0.{s}
        \\fmls z16.{s}, {s}/m, z5.{s}, z1.{s}
        \\fmla z17.{s}, {s}/m, z4.{s}, z1.{s}
        \\fmla z17.{s}, {s}/m, z5.{s}, z0.{s}
    , .{
        lane, pred, lane, lane,
        lane, pred, lane, lane,
        lane, pred, lane, lane,
        lane, pred, lane, lane,
    });
}

fn complexLd2C64Pair(
    comptime real_z: comptime_int,
    comptime imag_z: comptime_int,
    comptime base: []const u8,
    comptime vl_offset: comptime_int,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ld2d {{ z{d}.d, z{d}.d }}, p0/z, {s}
    , .{ real_z, imag_z, vlAddr(base, vl_offset) });
}

fn complexLd2C64ToRegs(
    comptime real_z: comptime_int,
    comptime imag_z: comptime_int,
    comptime base: []const u8,
    comptime vl_offset: comptime_int,
) []const u8 {
    return complexLd2C64Pair(24, 25, base, vl_offset) ++ std.fmt.comptimePrint(
        \\
        \\mov z{d}.d, z24.d
        \\mov z{d}.d, z25.d
    , .{ real_z, imag_z });
}

fn complexSt2C64Pair(
    comptime real_z: comptime_int,
    comptime imag_z: comptime_int,
    comptime base: []const u8,
    comptime vl_offset: comptime_int,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\st2d {{ z{d}.d, z{d}.d }}, p0, {s}
    , .{ real_z, imag_z, vlAddr(base, vl_offset) });
}

fn zgemvNoTransSme2C64LoadAcc(
    comptime vl_offset: comptime_int,
    comptime real_row: comptime_int,
    comptime imag_row: comptime_int,
) []const u8 {
    comptime var text: []const u8 = "";
    const base = if (vl_offset >= 16) "x12" else "x9";
    const base_offset = if (vl_offset >= 16) vl_offset - 16 else vl_offset;
    if (vl_offset >= 16) {
        text = text ++
            \\
            \\addvl x12, x9, #16
        ;
    }
    inline for (0..4) |i| {
        text = text ++ complexLd2C64ToRegs(16 + i, 20 + i, base, base_offset + i * 2);
    }
    return text ++ std.fmt.comptimePrint(
        \\
        \\mov w8, #{d}
        \\mov w11, #{d}
        \\fmla za.d[w8, 0, vgx4], {{ z16.d - z19.d }}, z4.d
        \\fmls za.d[w8, 0, vgx4], {{ z20.d - z23.d }}, z5.d
        \\fmla za.d[w11, 0, vgx4], {{ z16.d - z19.d }}, z5.d
        \\fmla za.d[w11, 0, vgx4], {{ z20.d - z23.d }}, z4.d
    , .{ real_row, imag_row });
}

fn zgemvNoTransSme2C64LoadAcc128() []const u8 {
    return zgemvNoTransSme2C64LoadAcc(0, 0, 2) ++
        zgemvNoTransSme2C64LoadAcc(8, 8, 10) ++
        zgemvNoTransSme2C64LoadAcc(16, 1, 3) ++
        zgemvNoTransSme2C64LoadAcc(24, 9, 11);
}

fn zgemvNoTransSme2C64StoreAcc(
    comptime vl_offset: comptime_int,
    comptime real_row: comptime_int,
    comptime imag_row: comptime_int,
) []const u8 {
    comptime var text: []const u8 = std.fmt.comptimePrint(
        \\
        \\mov w8, #{d}
        \\mov w11, #{d}
        \\mov {{ z16.d - z19.d }}, za.d[w8, 0, vgx4]
        \\mov {{ z20.d - z23.d }}, za.d[w11, 0, vgx4]
    , .{ real_row, imag_row });
    const base = if (vl_offset >= 16) "x12" else "x9";
    const base_offset = if (vl_offset >= 16) vl_offset - 16 else vl_offset;
    if (vl_offset >= 16) {
        text = text ++
            \\
            \\addvl x12, x9, #16
        ;
    }
    inline for (0..4) |i| {
        text = text ++ complexLd2C64Pair(24, 25, base, base_offset + i * 2) ++ std.fmt.comptimePrint(
            \\
            \\fadd z24.d, z24.d, z{d}.d
            \\fadd z25.d, z25.d, z{d}.d
        , .{ 16 + i, 20 + i }) ++ complexSt2C64Pair(24, 25, base, base_offset + i * 2);
    }
    return text;
}

fn zgemvNoTransSme2C64StoreAcc128() []const u8 {
    return zgemvNoTransSme2C64StoreAcc(0, 0, 2) ++
        zgemvNoTransSme2C64StoreAcc(8, 8, 10) ++
        zgemvNoTransSme2C64StoreAcc(16, 1, 3) ++
        zgemvNoTransSme2C64StoreAcc(24, 9, 11);
}

fn zgemvNoTransSme2C64LoadAcc64() []const u8 {
    return zgemvNoTransSme2C64LoadAcc(0, 0, 1) ++
        zgemvNoTransSme2C64LoadAcc(8, 8, 9);
}

fn zgemvNoTransSme2C64StoreAcc64() []const u8 {
    return zgemvNoTransSme2C64StoreAcc(0, 0, 1) ++
        zgemvNoTransSme2C64StoreAcc(8, 8, 9);
}

fn zgemvNoTransSme2C64Asm(
    comptime load_acc: []const u8,
    comptime store_acc: []const u8,
    comptime row_advance_bytes: comptime_int,
    comptime row_count: comptime_int,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, 12f
        \\cbz x1, 12f
    , .{}) ++ ptrue_p0_d ++ std.fmt.comptimePrint(
        \\
        \\fmov d6, x2
        \\fmov d7, x3
        \\mov x13, x4
        \\mov x14, x7
        \\mov x10, x0
        \\
        \\10:
        \\zero {{ za }}
        \\mov x15, x13
        \\mov x17, x1
        \\mov x16, x6
        \\
        \\11:
        \\ldr d0, [x16]
        \\ldr d1, [x16, #8]
        \\add x16, x16, #16
        \\fmul d4, d6, d0
        \\fmul d2, d7, d1
        \\fsub d4, d4, d2
        \\fmul d5, d6, d1
        \\fmadd d5, d7, d0, d5
        \\mov z4.d, d4
        \\mov z5.d, d5
        \\mov x9, x15
    , .{}) ++ load_acc ++ std.fmt.comptimePrint(
        \\
        \\add x15, x15, x5
        \\subs x17, x17, #1
        \\b.ne 11b
        \\
        \\mov x9, x14
    , .{}) ++ store_acc ++ std.fmt.comptimePrint(
        \\
        \\add x13, x13, #{d}
        \\add x14, x14, #{d}
        \\subs x10, x10, #{d}
        \\b.ne 10b
        \\
        \\12:
        \\ret
    , .{ row_advance_bytes, row_advance_bytes, row_count });
}

pub fn zgemvNoTransSme2C64512x1Asm() []const u8 {
    @setEvalBranchQuota(200000);
    return zgemvNoTransSme2C64Asm(zgemvNoTransSme2C64LoadAcc128(), zgemvNoTransSme2C64StoreAcc128(), 2048, 128);
}

pub fn zgemvNoTransSme2C6464x1Asm() []const u8 {
    @setEvalBranchQuota(200000);
    return zgemvNoTransSme2C64Asm(zgemvNoTransSme2C64LoadAcc64(), zgemvNoTransSme2C64StoreAcc64(), 1024, 64);
}

pub fn sveScalAsm(comptime lane: []const u8, comptime unroll: comptime_int) []const u8 {
    @setEvalBranchQuota(100000);
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, 12f
    , .{}) ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\mov z4.{s}, {s}
        \\{s} x6
        \\lsl x7, x6, #{d}
        \\
        \\10:
        \\cmp x0, x7
        \\b.lo 11f
    , .{ lane, scalarReg(lane, 0), laneCountMnemonic(lane), log2Pow2(unroll) }) ++ ld1Seq(lane, 0, unroll, "p0", "x1", 0) ++ blk: {
        comptime var text: []const u8 = "";
        inline for (0..unroll) |i| {
            text = text ++ std.fmt.comptimePrint(
                \\
                \\fmul z{d}.{s}, p0/m, z{d}.{s}, z4.{s}
            , .{ i, lane, i, lane, lane });
        }
        break :blk text;
    } ++ st1Seq(lane, 0, unroll, "p0", "x1", 0) ++ addvlAdvance("x1", unroll) ++ std.fmt.comptimePrint(
        \\
        \\sub x0, x0, x7
        \\b 10b
        \\
        \\11:
        \\cbz x0, 12f
        \\mov x8, #0
        \\
        \\13:
        \\whilelo p1.{s}, x8, x0
        \\b.none 12f
    , .{lane}) ++ ld1Indexed(lane, 0, "p1", "x1", "x8") ++ std.fmt.comptimePrint(
        \\
        \\fmul z0.{s}, p1/m, z0.{s}, z4.{s}
    , .{ lane, lane, lane }) ++ st1Indexed(lane, 0, "p1", "x1", "x8") ++ std.fmt.comptimePrint(
        \\
        \\{s} x8
        \\b 13b
        \\
        \\12:
        \\ret
    , .{laneIncMnemonic(lane)});
}

pub fn sveRealDotAsm(comptime lane: []const u8, comptime unroll: comptime_int) []const u8 {
    @setEvalBranchQuota(100000);
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, 13f
    , .{}) ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\{s} x6
        \\lsl x7, x6, #{d}
    , .{ laneCountMnemonic(lane), log2Pow2(unroll) }) ++ dupZeroSeq(lane, 0, unroll) ++ std.fmt.comptimePrint(
        \\
        \\10:
        \\cmp x0, x7
        \\b.lo 11f
    , .{}) ++ ld1Seq(lane, 4, unroll, "p0", "x1", 0) ++ ld1Seq(lane, 16, unroll, "p0", "x2", 0) ++ blk: {
        comptime var text: []const u8 = "";
        inline for (0..unroll) |i| {
            text = text ++ std.fmt.comptimePrint(
                \\
                \\fmla z{d}.{s}, p0/m, z{d}.{s}, z{d}.{s}
            , .{ i, lane, 4 + i, lane, 16 + i, lane });
        }
        break :blk text;
    } ++ addvlAdvance("x1", unroll) ++ addvlAdvance("x2", unroll) ++ std.fmt.comptimePrint(
        \\
        \\sub x0, x0, x7
        \\b 10b
        \\
        \\11:
        \\cbz x0, 12f
        \\mov x8, #0
        \\
        \\14:
        \\whilelo p1.{s}, x8, x0
        \\b.none 12f
    , .{lane}) ++ ld1Indexed(lane, 4, "p1", "x1", "x8") ++ ld1Indexed(lane, 16, "p1", "x2", "x8") ++ std.fmt.comptimePrint(
        \\
        \\fmla z0.{s}, p1/m, z4.{s}, z16.{s}
        \\{s} x8
        \\b 14b
        \\
        \\12:
    , .{ lane, lane, lane, laneIncMnemonic(lane) }) ++ reduceAccumulators(lane, 0, unroll) ++ std.fmt.comptimePrint(
        \\
        \\faddv {s}, p0, z0.{s}
        \\fmov {s}, {s}
        \\ret
        \\
        \\13:
        \\mov {s}, {s}
        \\ret
    , .{ scalarReg(lane, 16), lane, returnGpr(lane), scalarReg(lane, 16), returnGpr(lane), gprZero(lane) });
}

pub fn sveRealAsumAsm(comptime lane: []const u8, comptime unroll: comptime_int) []const u8 {
    @setEvalBranchQuota(200000);
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, 13f
    , .{}) ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\{s} x6
        \\lsl x7, x6, #{d}
    , .{ laneCountMnemonic(lane), log2Pow2(unroll) }) ++ dupZeroSeq(lane, 0, unroll) ++ std.fmt.comptimePrint(
        \\
        \\10:
        \\cmp x0, x7
        \\b.lo 11f
    , .{}) ++ ld1Seq(lane, 16, unroll, "p0", "x1", 0) ++ blk: {
        comptime var text: []const u8 = "";
        inline for (0..unroll) |i| {
            text = text ++ std.fmt.comptimePrint(
                \\
                \\fabs z{d}.{s}, p0/m, z{d}.{s}
            , .{ 16 + i, lane, 16 + i, lane });
        }
        break :blk text;
    } ++ faddSeq(lane, 0, 16, unroll) ++ addvlAdvance("x1", unroll) ++ std.fmt.comptimePrint(
        \\
        \\sub x0, x0, x7
        \\b 10b
        \\
        \\11:
        \\cbz x0, 12f
        \\mov x8, #0
        \\
        \\14:
        \\whilelo p1.{s}, x8, x0
        \\b.none 12f
    , .{lane}) ++ ld1Indexed(lane, 16, "p1", "x1", "x8") ++ std.fmt.comptimePrint(
        \\
        \\fabs z16.{s}, p1/m, z16.{s}
        \\fadd z0.{s}, z0.{s}, z16.{s}
        \\{s} x8
        \\b 14b
        \\
        \\12:
    , .{ lane, lane, lane, lane, lane, laneIncMnemonic(lane) }) ++ reduceAccumulators(lane, 0, unroll) ++ std.fmt.comptimePrint(
        \\
        \\faddv {s}, p0, z0.{s}
        \\fmov {s}, {s}
        \\ret
        \\
        \\13:
        \\mov {s}, {s}
        \\ret
    , .{ scalarReg(lane, 16), lane, returnGpr(lane), scalarReg(lane, 16), returnGpr(lane), gprZero(lane) });
}

pub fn sveComplexDotAsm(comptime lane: []const u8, comptime conj_x: bool) []const u8 {
    @setEvalBranchQuota(100000);
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, 13f
    , .{}) ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\{s} x6
        \\lsl x7, x6, #2
    , .{laneCountMnemonic(lane)}) ++ dupZeroSeq(lane, 0, 4) ++ dupZeroSeq(lane, 8, 4) ++ std.fmt.comptimePrint(
        \\
        \\10:
        \\cmp x0, x7
        \\b.lo 11f
    , .{}) ++ blk: {
        comptime var text: []const u8 = "";
        inline for (0..4) |i| {
            text = text ++ complexLd2(lane, 4, 5, "p0", "x1", i * 2) ++ complexLd2(lane, 16, 17, "p0", "x2", i * 2) ++
                complexFmla(lane, "p0", i, 8 + i, 4, 5, 16, 17, conj_x);
        }
        break :blk text;
    } ++ std.fmt.comptimePrint(
        \\
        \\addvl x1, x1, #8
        \\addvl x2, x2, #8
        \\sub x0, x0, x7
        \\b 10b
        \\
        \\11:
        \\cbz x0, 12f
        \\
        \\14:
        \\whilelo p1.{s}, xzr, x0
        \\b.none 12f
    , .{lane}) ++ complexLd2(lane, 4, 5, "p1", "x1", 0) ++ complexLd2(lane, 16, 17, "p1", "x2", 0) ++
        complexFmla(lane, "p1", 0, 8, 4, 5, 16, 17, conj_x) ++ std.fmt.comptimePrint(
        \\
        \\cmp x0, x6
        \\b.ls 12f
        \\sub x0, x0, x6
        \\addvl x1, x1, #2
        \\addvl x2, x2, #2
        \\b 14b
        \\
        \\12:
    , .{}) ++ reduceAccumulators(lane, 0, 4) ++ reduceAccumulators(lane, 8, 4) ++ std.fmt.comptimePrint(
        \\
        \\faddv {s}, p0, z0.{s}
        \\faddv {s}, p0, z8.{s}
        \\str {s}, [x3]
        \\str {s}, [x3, #{d}]
        \\ret
        \\
        \\13:
        \\str {s}, [x3]
        \\str {s}, [x3, #{d}]
        \\ret
    , .{
        scalarReg(lane, 0),                           lane,
        scalarReg(lane, 1),                           lane,
        scalarReg(lane, 0),                           scalarReg(lane, 1),
        @as(comptime_int, 1) << laneIndexShift(lane), gprZero(lane),
        gprZero(lane),                                @as(comptime_int, 1) << laneIndexShift(lane),
    });
}

pub fn sveComplexAxpyAsm(comptime lane: []const u8) []const u8 {
    @setEvalBranchQuota(100000);
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, 3f
    , .{}) ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\mov z0.{s}, {s}
        \\mov z1.{s}, {s}
        \\{s} x6
        \\lsl x7, x6, #2
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
    , .{ lane, scalarReg(lane, 0), lane, scalarReg(lane, 1), laneCountMnemonic(lane) }) ++ blk: {
        comptime var text: []const u8 = "";
        inline for (0..4) |i| {
            text = text ++ complexLd2(lane, 4, 5, "p0", "x1", i * 2) ++ complexLd2(lane, 16, 17, "p0", "x2", i * 2) ++
                complexAxpyOp(lane, "p0") ++ complexSt2(lane, 16, 17, "p0", "x2", i * 2);
        }
        break :blk text;
    } ++ std.fmt.comptimePrint(
        \\
        \\addvl x1, x1, #8
        \\addvl x2, x2, #8
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 3f
        \\whilelo p1.{s}, xzr, x0
    , .{lane}) ++ complexLd2(lane, 4, 5, "p1", "x1", 0) ++ complexLd2(lane, 16, 17, "p1", "x2", 0) ++
        complexAxpyOp(lane, "p1") ++ complexSt2(lane, 16, 17, "p1", "x2", 0) ++ std.fmt.comptimePrint(
        \\
        \\cmp x0, x6
        \\b.ls 3f
        \\sub x0, x0, x6
        \\addvl x1, x1, #2
        \\addvl x2, x2, #2
        \\b 1b
        \\
        \\3:
        \\ret
    , .{});
}

fn smeZaSelectPair(comptime row_a: comptime_int, comptime row_b: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\mov w8, #{d}
        \\mov w11, #{d}
    , .{ row_a, row_b });
}

fn smeZaReadPair(comptime lane: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\mov {{ z4.{s} - z7.{s} }}, za.{s}[w8, 0, vgx4]
        \\mov {{ z16.{s} - z19.{s} }}, za.{s}[w11, 0, vgx4]
    , .{ lane, lane, lane, lane, lane, lane });
}

fn smeZaAddPair(comptime lane: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\fadd za.{s}[w8, 0, vgx4], {{ z4.{s} - z7.{s} }}
        \\fadd za.{s}[w11, 0, vgx4], {{ z16.{s} - z19.{s} }}
    , .{ lane, lane, lane, lane, lane, lane });
}

fn smeStreamingBlockElements(comptime lane: []const u8) comptime_int {
    if (std.mem.eql(u8, lane, "s")) return 512;
    if (std.mem.eql(u8, lane, "d")) return 256;
    @compileError("unsupported SME streaming lane");
}

fn smeVgx4Ld1AtVl(
    comptime lane: []const u8,
    comptime first_z: comptime_int,
    comptime pred: []const u8,
    comptime base: []const u8,
    comptime vl_offset: comptime_int,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\ld1{s} {{ z{d}.{s} - z{d}.{s} }}, {s}/z, {s}
    , .{ laneLoadSuffix(lane), first_z, lane, first_z + 3, lane, pred, vlAddr(base, vl_offset) });
}

fn smeVgx4Ld1(comptime lane: []const u8, comptime first_z: comptime_int, comptime pred: []const u8, comptime base: []const u8) []const u8 {
    return smeVgx4Ld1AtVl(lane, first_z, pred, base, 0);
}

fn smeVgx4St1AtVl(
    comptime lane: []const u8,
    comptime first_z: comptime_int,
    comptime pred: []const u8,
    comptime base: []const u8,
    comptime vl_offset: comptime_int,
) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\st1{s} {{ z{d}.{s} - z{d}.{s} }}, {s}, {s}
    , .{ laneLoadSuffix(lane), first_z, lane, first_z + 3, lane, pred, vlAddr(base, vl_offset) });
}

fn smeVgx4St1(comptime lane: []const u8, comptime first_z: comptime_int, comptime pred: []const u8, comptime base: []const u8) []const u8 {
    return smeVgx4St1AtVl(lane, first_z, pred, base, 0);
}

fn smePtrAtVl(comptime dst: []const u8, comptime base: []const u8, comptime offset: comptime_int) []const u8 {
    if (offset == 0) {
        return std.fmt.comptimePrint(
            \\
            \\mov {s}, {s}
        , .{ dst, base });
    }
    return std.fmt.comptimePrint(
        \\
        \\addvl {s}, {s}, #{d}
    , .{ dst, base, offset });
}

fn smeZaFmlaScalarPair(comptime lane: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\fmla za.{s}[w8, 0, vgx4], {{ z4.{s} - z7.{s} }}, z0.{s}
        \\fmla za.{s}[w11, 0, vgx4], {{ z16.{s} - z19.{s} }}, z0.{s}
    , .{
        lane, lane, lane, lane,
        lane, lane, lane, lane,
    });
}

fn smeFabsPair(comptime lane: []const u8) []const u8 {
    comptime var text: []const u8 = "";
    inline for (4..8) |z| {
        text = text ++ std.fmt.comptimePrint(
            \\
            \\fabs z{d}.{s}, p0/m, z{d}.{s}
        , .{ z, lane, z, lane });
    }
    inline for (16..20) |z| {
        text = text ++ std.fmt.comptimePrint(
            \\
            \\fabs z{d}.{s}, p0/m, z{d}.{s}
        , .{ z, lane, z, lane });
    }
    return text;
}

pub fn smeCopyBytesStreamingAsm() []const u8 {
    return ptrue("pn8", "b") ++ std.fmt.comptimePrint(
        \\
        \\cbz x0, 4f
        \\cntb x6
        \\lsl x7, x6, #3
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
    , .{}) ++
        smeVgx4Ld1AtVl("b", 4, "pn8", "x1", 0) ++
        smeVgx4Ld1AtVl("b", 16, "pn8", "x1", 4) ++
        smeVgx4St1AtVl("b", 4, "pn8", "x2", 0) ++
        smeVgx4St1AtVl("b", 16, "pn8", "x2", 4) ++
        std.fmt.comptimePrint(
            \\
            \\addvl x1, x1, #8
            \\addvl x2, x2, #8
            \\sub x0, x0, x7
            \\b 0b
            \\
            \\1:
            \\cbz x0, 4f
            \\lsl x7, x6, #2
            \\cmp x0, x7
            \\b.lo 2f
            \\sub x8, x0, x7
            \\ptrue pn9.b
            \\whilelt pn10.b, xzr, x8, vlx4
        , .{}) ++
        smeVgx4Ld1AtVl("b", 4, "pn9", "x1", 0) ++
        smeVgx4Ld1AtVl("b", 16, "pn10", "x1", 4) ++
        smeVgx4St1AtVl("b", 4, "pn9", "x2", 0) ++
        smeVgx4St1AtVl("b", 16, "pn10", "x2", 4) ++
        std.fmt.comptimePrint(
            \\
            \\b 4f
            \\
            \\2:
            \\whilelt pn9.b, xzr, x0, vlx4
        , .{}) ++
        smeVgx4Ld1AtVl("b", 4, "pn9", "x1", 0) ++
        smeVgx4St1AtVl("b", 4, "pn9", "x2", 0) ++
        std.fmt.comptimePrint(
            \\
            \\4:
            \\ret
        , .{});
}

const SmeUnaryZaOp = enum {
    scal,
    asum,
};

fn smeUnaryZaAccumulatePair(comptime lane: []const u8, comptime op: SmeUnaryZaOp) []const u8 {
    return switch (op) {
        .scal => smeZaFmlaScalarPair(lane),
        .asum => smeFabsPair(lane) ++ smeZaAddPair(lane),
    };
}

fn smeUnaryZaMainBlock(comptime lane: []const u8, comptime op: SmeUnaryZaOp) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\mov x10, x1
        \\addvl x13, x10, #4
        \\addvl x9, x10, #8
        \\addvl x12, x10, #12
    , .{}) ++
        smeZaSelectPair(0, 8) ++ smeVgx4Ld1(lane, 4, "pn8", "x10") ++ smeVgx4Ld1(lane, 16, "pn8", "x13") ++ smeUnaryZaAccumulatePair(lane, op) ++
        smeZaSelectPair(1, 9) ++ smeVgx4Ld1(lane, 4, "pn8", "x9") ++ smeVgx4Ld1(lane, 16, "pn8", "x12") ++ smeUnaryZaAccumulatePair(lane, op) ++
        std.fmt.comptimePrint(
            \\
            \\addvl x10, x10, #16
            \\addvl x13, x13, #16
            \\addvl x9, x9, #16
            \\addvl x12, x12, #16
        , .{}) ++
        smeZaSelectPair(2, 10) ++ smeVgx4Ld1(lane, 4, "pn8", "x10") ++ smeVgx4Ld1(lane, 16, "pn8", "x13") ++ smeUnaryZaAccumulatePair(lane, op) ++
        smeZaSelectPair(3, 11) ++ smeVgx4Ld1(lane, 4, "pn8", "x9") ++ smeVgx4Ld1(lane, 16, "pn8", "x12") ++ smeUnaryZaAccumulatePair(lane, op);
}

fn smeUnaryZaStoreBlock(comptime lane: []const u8) []const u8 {
    comptime var text: []const u8 = std.fmt.comptimePrint(
        \\
        \\mov x10, x1
        \\addvl x13, x10, #4
    , .{});
    inline for (0..4) |row| {
        text = text ++ smeZaSelectPair(row, 8 + row) ++ smeZaReadPair(lane) ++
            smeVgx4St1(lane, 4, "pn8", "x10") ++ smeVgx4St1(lane, 16, "pn8", "x13");
        if (row != 3) {
            text = text ++ std.fmt.comptimePrint(
                \\
                \\addvl x10, x10, #8
                \\addvl x13, x13, #8
            , .{});
        }
    }
    return text;
}

fn smeStreamingAdvanceX1() []const u8 {
    return
    \\
    \\add x1, x1, #2048
    \\
    ;
}

fn smeStreamingAdvanceX1X2() []const u8 {
    return
    \\
    \\add x1, x1, #2048
    \\add x2, x2, #2048
    \\
    ;
}

fn smeAxpyZaAccumulate(comptime lane: []const u8, comptime za_row: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\fadd za.{s}[{s}, 0, vgx4], {{ z16.{s} - z19.{s} }}
        \\fmla za.{s}[{s}, 0, vgx4], {{ z4.{s} - z7.{s} }}, z0.{s}
    , .{
        lane, za_row, lane, lane,
        lane, za_row, lane, lane,
        lane,
    });
}

fn smeAxpyZaMainBlock(comptime lane: []const u8) []const u8 {
    comptime var text: []const u8 = "";
    inline for (0..8) |pair| {
        const row = pair / 2;
        if (pair % 2 == 0) text = text ++ smeZaSelectPair(row, 8 + row);
        text = text ++ smePtrAtVl("x10", "x1", pair * 4) ++ smePtrAtVl("x12", "x2", pair * 4) ++
            smeVgx4Ld1(lane, 4, "pn8", "x10") ++ smeVgx4Ld1(lane, 16, "pn8", "x12") ++
            smeAxpyZaAccumulate(lane, if (pair % 2 == 0) "w8" else "w11");
    }
    return text;
}

fn smeAxpyZaStoreBlock(comptime lane: []const u8) []const u8 {
    comptime var text: []const u8 = "";
    inline for (0..4) |row| {
        text = text ++ smeZaSelectPair(row, 8 + row) ++ smeZaReadPair(lane) ++
            smePtrAtVl("x12", "x2", row * 8) ++ smeVgx4St1(lane, 4, "pn8", "x12") ++
            smePtrAtVl("x12", "x2", row * 8 + 4) ++ smeVgx4St1(lane, 16, "pn8", "x12");
    }
    return text;
}

fn smeStreamingFinalReduce(comptime lane: []const u8, comptime tail_z: comptime_int) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\fadd z4.{s}, z4.{s}, z5.{s}
        \\fadd z6.{s}, z6.{s}, z7.{s}
        \\fadd z16.{s}, z16.{s}, z17.{s}
        \\fadd z18.{s}, z18.{s}, z19.{s}
        \\fadd z4.{s}, z4.{s}, z6.{s}
        \\fadd z16.{s}, z16.{s}, z18.{s}
        \\fadd z4.{s}, z4.{s}, z16.{s}
        \\fadd z4.{s}, z4.{s}, z{d}.{s}
        \\faddv {s}, p0, z4.{s}
        \\fmov {s}, {s}
        \\ret
        \\
        \\4:
        \\mov {s}, {s}
        \\ret
    , .{
        lane,            lane,                lane,
        lane,            lane,                lane,
        lane,            lane,                lane,
        lane,            lane,                lane,
        lane,            lane,                lane,
        lane,            lane,                lane,
        lane,            lane,                lane,
        lane,            lane,                tail_z,
        lane,            scalarReg(lane, 16), lane,
        returnGpr(lane), scalarReg(lane, 16), returnGpr(lane),
        gprZero(lane),
    });
}

fn smeDotFinalReduce(comptime lane: []const u8) []const u8 {
    return smeStreamingFinalReduce(lane, 28);
}

fn smeAsumFinalReduce(comptime lane: []const u8) []const u8 {
    return smeStreamingFinalReduce(lane, 24);
}

fn smeZaReductionAsm(comptime lane: []const u8, comptime final_reduce: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        \\
        \\2:
    , .{}) ++
        smeZaSelectPair(2, 10) ++ smeZaReadPair(lane) ++
        smeZaSelectPair(0, 8) ++ smeZaAddPair(lane) ++
        smeZaSelectPair(3, 11) ++ smeZaReadPair(lane) ++
        smeZaSelectPair(1, 9) ++ smeZaAddPair(lane) ++
        smeZaSelectPair(0, 8) ++ smeZaReadPair(lane) ++
        smeZaSelectPair(1, 9) ++ smeZaAddPair(lane) ++
        smeZaReadPair(lane) ++ final_reduce;
}

fn smeDotZaReductionAsm(comptime lane: []const u8) []const u8 {
    return smeZaReductionAsm(lane, smeDotFinalReduce(lane));
}

fn smeAsumZaReductionAsm(comptime lane: []const u8) []const u8 {
    return smeZaReductionAsm(lane, smeAsumFinalReduce(lane));
}

pub fn smeScalStreamingAsm(comptime lane: []const u8) []const u8 {
    @setEvalBranchQuota(300000);
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, 3f
    , .{}) ++ ptrue("pn8", lane) ++ std.fmt.comptimePrint(
        \\
        \\mov z0.{s}, {s}
        \\mov x7, #{d}
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\zero {{ za }}
    , .{ lane, scalarReg(lane, 0), smeStreamingBlockElements(lane) }) ++
        smeUnaryZaMainBlock(lane, .scal) ++ smeUnaryZaStoreBlock(lane) ++ smeStreamingAdvanceX1() ++ std.fmt.comptimePrint(
        \\
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\mov z0.{s}, {s}
        \\
        \\4:
        \\whilelo p1.{s}, x8, x0
        \\b.none 2f
    , .{ lane, scalarReg(lane, 0), lane }) ++ ld1Indexed(lane, 4, "p1", "x1", "x8") ++ std.fmt.comptimePrint(
        \\
        \\fmul z4.{s}, p1/m, z4.{s}, z0.{s}
    , .{ lane, lane, lane }) ++ st1Indexed(lane, 4, "p1", "x1", "x8") ++ std.fmt.comptimePrint(
        \\
        \\{s} x8
        \\b 4b
        \\
        \\2:
        \\3:
        \\ret
    , .{laneIncMnemonic(lane)});
}

pub fn smeAsumStreamingAsm(comptime lane: []const u8) []const u8 {
    @setEvalBranchQuota(300000);
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, 4f
    , .{}) ++ ptrue("pn8", lane) ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\mov x7, #{d}
        \\zero {{ za }}
        \\dup z24.{s}, #0
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
    , .{ smeStreamingBlockElements(lane), lane }) ++ smeUnaryZaMainBlock(lane, .asum) ++ smeStreamingAdvanceX1() ++ std.fmt.comptimePrint(
        \\
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\
        \\5:
        \\whilelo p1.{s}, x8, x0
        \\b.none 2f
    , .{lane}) ++ ld1Indexed(lane, 25, "p1", "x1", "x8") ++ std.fmt.comptimePrint(
        \\
        \\fabs z25.{s}, p1/m, z25.{s}
        \\fadd z24.{s}, z24.{s}, z25.{s}
        \\{s} x8
        \\b 5b
    , .{ lane, lane, lane, lane, lane, laneIncMnemonic(lane) }) ++ smeAsumZaReductionAsm(lane);
}

pub fn smeAxpyStreamingAsm(comptime lane: []const u8) []const u8 {
    @setEvalBranchQuota(300000);
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, 3f
    , .{}) ++ ptrue("pn8", lane) ++ std.fmt.comptimePrint(
        \\
        \\mov z0.{s}, {s}
        \\mov x7, #{d}
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
        \\zero {{ za }}
    , .{ lane, scalarReg(lane, 0), smeStreamingBlockElements(lane) }) ++
        smeAxpyZaMainBlock(lane) ++ smeAxpyZaStoreBlock(lane) ++ smeStreamingAdvanceX1X2() ++ std.fmt.comptimePrint(
        \\
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\mov z0.{s}, {s}
        \\
        \\4:
        \\whilelo p1.{s}, x8, x0
        \\b.none 2f
    , .{ lane, scalarReg(lane, 0), lane }) ++ ld1Indexed(lane, 4, "p1", "x1", "x8") ++ ld1Indexed(lane, 16, "p1", "x2", "x8") ++ std.fmt.comptimePrint(
        \\
        \\fmla z16.{s}, p1/m, z4.{s}, z0.{s}
    , .{ lane, lane, lane }) ++ st1Indexed(lane, 16, "p1", "x2", "x8") ++ std.fmt.comptimePrint(
        \\
        \\{s} x8
        \\b 4b
        \\
        \\2:
        \\3:
        \\ret
    , .{laneIncMnemonic(lane)});
}

pub fn smeDotStreamingAsm(comptime lane: []const u8) []const u8 {
    @setEvalBranchQuota(200000);
    return std.fmt.comptimePrint(
        \\
        \\cbz x0, 4f
    , .{}) ++ ptrue("pn8", lane) ++ ptrue("p0", lane) ++ std.fmt.comptimePrint(
        \\
        \\{s} x6
        \\lsl x7, x6, #5
        \\zero {{ za }}
        \\dup z28.{s}, #0
        \\
        \\0:
        \\cmp x0, x7
        \\b.lo 1f
    , .{ laneCountMnemonic(lane), lane }) ++ blk: {
        comptime var text: []const u8 = "";
        inline for (0..4) |i| {
            text = text ++ std.fmt.comptimePrint(
                \\
                \\mov w8, #{d}
                \\mov w11, #{d}
                \\ld1{s} {{ z4.{s} - z7.{s} }}, pn8/z, {s}
                \\ld1{s} {{ z16.{s} - z19.{s} }}, pn8/z, {s}
                \\ld1{s} {{ z20.{s} - z23.{s} }}, pn8/z, {s}
                \\ld1{s} {{ z24.{s} - z27.{s} }}, pn8/z, {s}
                \\fmla za.{s}[w8, 0, vgx4], {{ z4.{s} - z7.{s} }}, {{ z16.{s} - z19.{s} }}
                \\fmla za.{s}[w11, 0, vgx4], {{ z20.{s} - z23.{s} }}, {{ z24.{s} - z27.{s} }}
            , .{
                i,                    8 + i,
                laneLoadSuffix(lane), lane,
                lane,                 vlAddr("x1", i * 8),
                laneLoadSuffix(lane), lane,
                lane,                 vlAddr("x2", i * 8),
                laneLoadSuffix(lane), lane,
                lane,                 vlAddr("x1", i * 8 + 4),
                laneLoadSuffix(lane), lane,
                lane,                 vlAddr("x2", i * 8 + 4),
                lane,                 lane,
                lane,                 lane,
                lane,                 lane,
                lane,                 lane,
                lane,                 lane,
            });
        }
        break :blk text;
    } ++ addvlAdvance("x1", 32) ++ addvlAdvance("x2", 32) ++ std.fmt.comptimePrint(
        \\
        \\sub x0, x0, x7
        \\b 0b
        \\
        \\1:
        \\cbz x0, 2f
        \\mov x8, #0
        \\
        \\5:
        \\whilelo p1.{s}, x8, x0
        \\b.none 2f
    , .{lane}) ++ ld1Indexed(lane, 4, "p1", "x1", "x8") ++ ld1Indexed(lane, 16, "p1", "x2", "x8") ++ std.fmt.comptimePrint(
        \\
        \\fmla z28.{s}, p1/m, z4.{s}, z16.{s}
        \\{s} x8
        \\b 5b
    , .{ lane, lane, lane, laneIncMnemonic(lane) }) ++ smeDotZaReductionAsm(lane);
}
