// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Two-vector architecture kernel facade.

const builtin = @import("builtin");

const aarch64 = @import("aarch64/vector_binary.zig");

pub fn copyUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.copyUnitReal(T, n, x, y),
        else => false,
    };
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.axpyUnitReal(T, n, alpha, x, y),
        else => false,
    };
}

pub fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.dotUnitReal(T, n, x, y),
        else => null,
    };
}
