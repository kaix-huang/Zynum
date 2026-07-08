// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Single-vector architecture kernel facade.

const builtin = @import("builtin");

const aarch64 = @import("../arch/aarch64/vector/unary.zig");
const types = @import("../../types.zig");
const x86_64 = @import("../arch/x86_64/vector/unary.zig");

pub fn scalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.scalUnitReal(T, n, alpha, x),
        .x86_64 => x86_64.scalUnitReal(T, n, alpha, x),
        else => false,
    };
}

pub fn scalUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.scalUnitComplex(T, n, alpha, x),
        .x86_64 => x86_64.scalUnitComplex(T, n, alpha, x),
        else => false,
    };
}

pub fn asumUnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.asumUnitReal(T, n, x),
        .x86_64 => x86_64.asumUnitReal(T, n, x),
        else => null,
    };
}

pub fn nrm2UnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.nrm2UnitReal(T, n, x),
        .x86_64 => x86_64.nrm2UnitReal(T, n, x),
        else => null,
    };
}

pub fn iamaxUnitReal(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.iamaxUnitReal(T, n, x),
        .x86_64 => x86_64.iamaxUnitReal(T, n, x),
        else => null,
    };
}
