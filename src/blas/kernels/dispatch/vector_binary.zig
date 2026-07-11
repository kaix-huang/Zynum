// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Two-vector architecture kernel facade.

const builtin = @import("builtin");

const aarch64 = @import("../arch/aarch64/vector/binary.zig");
const x86_64 = @import("../arch/x86_64/vector/binary.zig");

pub fn fixedCopyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.fixedCopyBytes(n_bytes, x, y),
        .x86_64 => x86_64.fixedCopyBytes(n_bytes, x, y),
        else => false,
    };
}

pub fn copyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.copyBytes(n_bytes, x, y),
        .x86_64 => x86_64.copyBytes(n_bytes, x, y),
        else => false,
    };
}

pub fn copyUnit(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    return copyBytes(n * @sizeOf(T), @ptrCast(x), @ptrCast(y));
}

pub fn copyUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.copyUnitReal(T, n, x, y),
        .x86_64 => x86_64.copyUnitReal(T, n, x, y),
        else => false,
    };
}

pub fn swapUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.swapUnitReal(T, n, x, y),
        .x86_64 => x86_64.swapUnitReal(T, n, x, y),
        else => false,
    };
}

pub fn swapUnitRealStreaming(comptime T: type, n: usize, x: [*]T, y: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.swapUnitRealStreaming(T, n, x, y),
        else => false,
    };
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.axpyUnitReal(T, n, alpha, x, y),
        .x86_64 => x86_64.axpyUnitReal(T, n, alpha, x, y),
        else => false,
    };
}

pub fn axpyUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.axpyUnitComplex(T, n, alpha, x, y),
        .x86_64 => x86_64.axpyUnitComplex(T, n, alpha, x, y),
        else => false,
    };
}

pub fn axpbyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.axpbyUnitReal(T, n, alpha, x, beta, y),
        .x86_64 => x86_64.axpbyUnitReal(T, n, alpha, x, beta, y),
        else => false,
    };
}

pub fn axpbyUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.axpbyUnitComplex(T, n, alpha, x, beta, y),
        .x86_64 => x86_64.axpbyUnitComplex(T, n, alpha, x, beta, y),
        else => false,
    };
}

pub fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.dotUnitReal(T, n, x, y),
        .x86_64 => x86_64.dotUnitReal(T, n, x, y),
        else => null,
    };
}

pub fn dotUnitComplex(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.dotUnitComplex(T, n, x, y, conjx),
        .x86_64 => x86_64.dotUnitComplex(T, n, x, y, conjx),
        else => null,
    };
}

pub fn rotUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.rotUnitReal(T, n, x, y, c, s),
        .x86_64 => x86_64.rotUnitReal(T, n, x, y, c, s),
        else => false,
    };
}

pub fn rotUnitRealStreaming(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.rotUnitRealStreaming(T, n, x, y, c, s),
        else => false,
    };
}

pub fn rotmUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.rotmUnitReal(T, n, x, y, flag, h11, h21, h12, h22),
        else => false,
    };
}
