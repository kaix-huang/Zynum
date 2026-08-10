// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 vector whole-function asm entry points.

const builders = @import("builders.zig");

pub noinline fn sscalSveF32(n: usize, alpha: f32, x: [*]f32) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = x;
    asm volatile (builders.sveScalAsm("s", 8) ::: .{ .memory = true });
}

pub noinline fn dscalSveF64(n: usize, alpha: f64, x: [*]f64) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = x;
    asm volatile (builders.sveScalAsm("d", 4) ::: .{ .memory = true });
}

pub noinline fn dasumSveF64Bits(n: usize, x: [*]const f64) callconv(.naked) u64 {
    _ = n;
    _ = x;
    asm volatile (builders.sveRealAsumAsm("d", 16) ::: .{ .memory = true });
}

pub noinline fn ddotSveF64Bits(n: usize, x: [*]const f64, y: [*]const f64) callconv(.naked) u64 {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (builders.sveRealDotAsm("d", 4) ::: .{ .memory = true });
}

pub noinline fn sdotSveF32Bits(n: usize, x: [*]const f32, y: [*]const f32) callconv(.naked) u32 {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (builders.sveRealDotAsm("s", 8) ::: .{ .memory = true });
}

pub noinline fn sdotSveF32AccF64Bits(n: usize, x: [*]const f32, y: [*]const f32) callconv(.naked) u64 {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (builders.sveDotF32AccF64Asm() ::: .{ .memory = true });
}

pub noinline fn copySveBytes(n: usize, x: [*]const u8, y: [*]u8) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (builders.sveCopyAsm("b") ::: .{ .memory = true });
}

pub noinline fn swapSveBytes(n: usize, x: [*]u8, y: [*]u8) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (builders.sveSwapAsm("b") ::: .{ .memory = true });
}

pub noinline fn saxpySveF32(n: usize, alpha: f32, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    asm volatile (builders.sveAxpyAsm("s") ::: .{ .memory = true });
}

pub noinline fn daxpySveF64(n: usize, alpha: f64, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    asm volatile (builders.sveAxpyAsm("d") ::: .{ .memory = true });
}

pub noinline fn saxpbySveF32(n: usize, alpha: f32, beta: f32, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = beta;
    _ = x;
    _ = y;
    asm volatile (builders.sveAxpbyAsm("s") ::: .{ .memory = true });
}

pub noinline fn daxpbySveF64(n: usize, alpha: f64, beta: f64, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = beta;
    _ = x;
    _ = y;
    asm volatile (builders.sveAxpbyAsm("d") ::: .{ .memory = true });
}

pub noinline fn slinearSveF32(n: usize, x: [*]f32, y: [*]f32, a: f32, b: f32, c: f32, d: f32) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = a;
    _ = b;
    _ = c;
    _ = d;
    asm volatile (builders.sveLinearTransformAsm("s") ::: .{ .memory = true });
}

pub noinline fn dlinearSveF64(n: usize, x: [*]f64, y: [*]f64, a: f64, b: f64, c: f64, d: f64) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = a;
    _ = b;
    _ = c;
    _ = d;
    asm volatile (builders.sveLinearTransformAsm("d") ::: .{ .memory = true });
}

pub noinline fn cscalSveC32(n: usize, alpha_re: f32, alpha_im: f32, x: [*]u8) callconv(.naked) void {
    _ = n;
    _ = alpha_re;
    _ = alpha_im;
    _ = x;
    asm volatile (builders.sveComplexScalAsm("s") ::: .{ .memory = true });
}

pub noinline fn zscalSveC64(n: usize, alpha_re: f64, alpha_im: f64, x: [*]u8) callconv(.naked) void {
    _ = n;
    _ = alpha_re;
    _ = alpha_im;
    _ = x;
    asm volatile (builders.sveComplexScalAsm("d") ::: .{ .memory = true });
}

pub noinline fn caxpbySveC32(n: usize, alpha_re: f32, alpha_im: f32, beta_re: f32, beta_im: f32, x: [*]const u8, y: [*]u8) callconv(.naked) void {
    _ = n;
    _ = alpha_re;
    _ = alpha_im;
    _ = beta_re;
    _ = beta_im;
    _ = x;
    _ = y;
    asm volatile (builders.sveComplexAxpbyAsm("s") ::: .{ .memory = true });
}

pub noinline fn zaxpbySveC64(n: usize, alpha_re: f64, alpha_im: f64, beta_re: f64, beta_im: f64, x: [*]const u8, y: [*]u8) callconv(.naked) void {
    _ = n;
    _ = alpha_re;
    _ = alpha_im;
    _ = beta_re;
    _ = beta_im;
    _ = x;
    _ = y;
    asm volatile (builders.sveComplexAxpbyAsm("d") ::: .{ .memory = true });
}

pub noinline fn smaxabsSveF32Bits(n: usize, x: [*]const f32) callconv(.naked) u32 {
    _ = n;
    _ = x;
    asm volatile (builders.sveMaxAbsAsm("s") ::: .{ .memory = true });
}

pub noinline fn dmaxabsSveF64Bits(n: usize, x: [*]const f64) callconv(.naked) u64 {
    _ = n;
    _ = x;
    asm volatile (builders.sveMaxAbsAsm("d") ::: .{ .memory = true });
}

pub noinline fn cmaxabs1SveC32Bits(n: usize, x: [*]const u8) callconv(.naked) u32 {
    _ = n;
    _ = x;
    asm volatile (builders.sveComplexMaxAbs1Asm("s") ::: .{ .memory = true });
}

pub noinline fn zmaxabs1SveC64Bits(n: usize, x: [*]const u8) callconv(.naked) u64 {
    _ = n;
    _ = x;
    asm volatile (builders.sveComplexMaxAbs1Asm("d") ::: .{ .memory = true });
}

pub noinline fn sscaledSsqSveF32Bits(n: usize, scale: f32, x: [*]const f32) callconv(.naked) u32 {
    _ = n;
    _ = scale;
    _ = x;
    asm volatile (builders.sveScaledSumSquaresAsm("s") ::: .{ .memory = true });
}

pub noinline fn dscaledSsqSveF64Bits(n: usize, scale: f64, x: [*]const f64) callconv(.naked) u64 {
    _ = n;
    _ = scale;
    _ = x;
    asm volatile (builders.sveScaledSumSquaresAsm("d") ::: .{ .memory = true });
}
