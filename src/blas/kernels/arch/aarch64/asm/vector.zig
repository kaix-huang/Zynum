// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 vector whole-function asm entry points.

const builders = @import("builders.zig");

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
