// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Single-vector architecture kernel facade.

const builtin = @import("builtin");

const aarch64 = @import("aarch64/vector_unary.zig");

pub fn scalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.scalUnitReal(T, n, alpha, x),
        else => false,
    };
}

pub fn asumUnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.asumUnitReal(T, n, x),
        else => null,
    };
}
