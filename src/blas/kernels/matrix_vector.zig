// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Matrix-vector architecture kernel facade.

const builtin = @import("builtin");

const scalar = @import("../core/scalar.zig");
const aarch64 = @import("aarch64/matrix_vector.zig");

const BlasInt = scalar.BlasInt;

pub fn gemvTransUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransUnitReal(T, m, n, alpha, a, lda, x, y),
        else => false,
    };
}

pub fn gemvTransFullUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransFullUnitReal(T, m, n, alpha, a, lda, x, beta, y),
        else => false,
    };
}

pub fn gemvTransAmxUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransAmxUnitReal(T, m, n, alpha, a, lda, x, y),
        else => false,
    };
}

pub fn gemvNoTransUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransUnitReal(T, m, n, alpha, a, lda, x, y),
        else => false,
    };
}

pub fn gemvNoTransFullUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransFullUnitReal(T, m, n, alpha, a, lda, x, beta, y),
        else => false,
    };
}

pub fn gemvNoTransPackLenUnitReal(comptime T: type, m: usize, n: usize, lda: BlasInt) ?usize {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransPackLenUnitReal(T, m, n, lda),
        else => null,
    };
}

pub fn gemvNoTransPackUnitReal(
    comptime T: type,
    n: usize,
    alpha: T,
    x: [*]const T,
    pack: []T,
) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransPackUnitReal(T, n, alpha, x, pack),
        else => false,
    };
}

pub fn gemvNoTransPackedRowsUnitReal(
    comptime T: type,
    row_count: usize,
    n: usize,
    a: [*]const T,
    lda: BlasInt,
    pack: [*]const T,
    scratch: [*]T,
    y: [*]T,
) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransPackedRowsUnitReal(T, row_count, n, a, lda, pack, scratch, y),
        else => false,
    };
}

pub fn supportsGemvNoTransUnitReal(comptime T: type) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.supportsGemvNoTransUnitReal(T),
        else => false,
    };
}

pub fn gerUnitReal(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    x: [*]const T,
    y: [*]const T,
    a: [*]T,
    lda: BlasInt,
) bool {
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gerUnitReal(T, m, n, alpha, x, y, a, lda),
        else => false,
    };
}
