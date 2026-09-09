// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Matrix-vector architecture kernel facade.

const builtin = @import("builtin");

const scalar = @import("../../core/shared/scalar.zig");
const aarch64 = @import("../arch/aarch64/matrix_vector.zig");
const x86_64 = @import("../arch/x86_64/matrix_vector.zig");

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
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvTransUnitReal, T, .{ m, n, alpha, a, lda, x, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransUnitReal(T, m, n, alpha, a, lda, x, y),
        .x86_64 => x86_64.gemvTransUnitReal(T, m, n, alpha, a, lda, x, y),
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
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvTransFullUnitReal, T, .{ m, n, alpha, a, lda, x, beta, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransFullUnitReal(T, m, n, alpha, a, lda, x, beta, y),
        .x86_64 => x86_64.gemvTransFullUnitReal(T, m, n, alpha, a, lda, x, beta, y),
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
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvTransAmxUnitReal, T, .{ m, n, alpha, a, lda, x, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransAmxUnitReal(T, m, n, alpha, a, lda, x, y),
        .x86_64 => false,
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
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvNoTransUnitReal, T, .{ m, n, alpha, a, lda, x, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransUnitReal(T, m, n, alpha, a, lda, x, y),
        .x86_64 => x86_64.gemvNoTransUnitReal(T, m, n, alpha, a, lda, x, y),
        else => false,
    };
}

pub fn gemvNoTransUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvNoTransUnitComplex, T, .{ m, n, alpha, a, lda, x, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransUnitComplex(T, m, n, alpha, a, lda, x, y),
        .x86_64 => x86_64.gemvNoTransUnitComplex(T, m, n, alpha, a, lda, x, y),
        else => false,
    };
}

pub fn supportsGemvNoTransRowsUnitComplex(comptime T: type, row_count: usize, n: usize, lda: BlasInt) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_supportsGemvNoTransRowsUnitComplex, T, .{ row_count, n, lda });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.supportsGemvNoTransRowsUnitComplex(T, row_count, n, lda),
        .x86_64 => false,
        else => false,
    };
}

pub fn gemvNoTransRowsUnitComplex(
    comptime T: type,
    row_count: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvNoTransRowsUnitComplex, T, .{ row_count, n, alpha, a, lda, x, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransRowsUnitComplex(T, row_count, n, alpha, a, lda, x, y),
        .x86_64 => false,
        else => false,
    };
}

pub fn gemvNoTransFullUnitComplex(
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
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvNoTransFullUnitComplex, T, .{ m, n, alpha, a, lda, x, beta, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransFullUnitComplex(T, m, n, alpha, a, lda, x, beta, y),
        .x86_64 => false,
        else => false,
    };
}

pub fn gemvNoTransTaskUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y_delta: [*]T,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvNoTransTaskUnitComplex, T, .{ m, n, alpha, a, lda, x, y_delta });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransTaskUnitComplex(T, m, n, alpha, a, lda, x, y_delta),
        .x86_64 => x86_64.gemvNoTransTaskUnitComplex(T, m, n, alpha, a, lda, x, y_delta),
        else => false,
    };
}

pub fn supportsGemvNoTransTaskUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_supportsGemvNoTransTaskUnitComplex, T, .{ m, n, lda });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.supportsGemvNoTransTaskUnitComplex(T, m, n, lda),
        .x86_64 => x86_64.supportsGemvNoTransTaskUnitComplex(T, m, n, lda),
        else => false,
    };
}

pub fn supportsGemvNoTransFullUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_supportsGemvNoTransFullUnitComplex, T, .{ m, n, lda });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.supportsGemvNoTransFullUnitComplex(T, m, n, lda),
        .x86_64 => false,
        else => false,
    };
}

pub fn gemvTransUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
    do_conj: bool,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvTransUnitComplex, T, .{ m, n, alpha, a, lda, x, y, do_conj });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransUnitComplex(T, m, n, alpha, a, lda, x, y, do_conj),
        .x86_64 => x86_64.gemvTransUnitComplex(T, m, n, alpha, a, lda, x, y, do_conj),
        else => false,
    };
}

pub fn gemvTransTaskUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
    do_conj: bool,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvTransTaskUnitComplex, T, .{ m, n, alpha, a, lda, x, y, do_conj });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransTaskUnitComplex(T, m, n, alpha, a, lda, x, y, do_conj),
        .x86_64 => x86_64.gemvTransTaskUnitComplex(T, m, n, alpha, a, lda, x, y, do_conj),
        else => false,
    };
}

pub fn supportsGemvTransTaskUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt, do_conj: bool) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_supportsGemvTransTaskUnitComplex, T, .{ m, n, lda, do_conj });
    return switch (builtin.cpu.arch) {
        .aarch64 => false,
        .x86_64 => x86_64.supportsGemvTransTaskUnitComplex(T, m, n, lda, do_conj),
        else => false,
    };
}

pub fn gemvTransTaskFullUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
    do_conj: bool,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvTransTaskFullUnitComplex, T, .{ m, n, alpha, a, lda, x, beta, y, do_conj });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransTaskFullUnitComplex(T, m, n, alpha, a, lda, x, beta, y, do_conj),
        .x86_64 => false,
        else => false,
    };
}

pub fn supportsGemvTransTaskFullUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt, do_conj: bool) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_supportsGemvTransTaskFullUnitComplex, T, .{ m, n, lda, do_conj });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.supportsGemvTransTaskFullUnitComplex(T, m, n, lda, do_conj),
        .x86_64 => false,
        else => false,
    };
}

pub fn gemvTransTaskFullUnitComplexC64M512N64(
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    beta: scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) void {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvTransTaskFullUnitComplexC64M512N64, void, .{ alpha, a, lda, x, beta, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransTaskFullUnitComplexC64M512N64(alpha, a, lda, x, beta, y),
        else => unreachable,
    };
}

pub fn gemvTransTaskFullUnitComplexC64M256N128(
    alpha: scalar.ComplexF64,
    a: [*]const scalar.ComplexF64,
    lda: BlasInt,
    x: [*]const scalar.ComplexF64,
    beta: scalar.ComplexF64,
    y: [*]scalar.ComplexF64,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvTransTaskFullUnitComplexC64M256N128, void, .{ alpha, a, lda, x, beta, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransTaskFullUnitComplexC64M256N128(alpha, a, lda, x, beta, y),
        else => false,
    };
}

pub fn gemvTransFullUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
    do_conj: bool,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvTransFullUnitComplex, T, .{ m, n, alpha, a, lda, x, beta, y, do_conj });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvTransFullUnitComplex(T, m, n, alpha, a, lda, x, beta, y, do_conj),
        .x86_64 => false,
        else => false,
    };
}

pub fn supportsGemvTransFullUnitComplex(comptime T: type, m: usize, n: usize, lda: BlasInt, do_conj: bool) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_supportsGemvTransFullUnitComplex, T, .{ m, n, lda, do_conj });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.supportsGemvTransFullUnitComplex(T, m, n, lda, do_conj),
        .x86_64 => false,
        else => false,
    };
}

pub fn supportsGemvNoTransUnitComplex(comptime T: type) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_supportsGemvNoTransUnitComplex, T, .{});
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.supportsGemvNoTransUnitComplex(T),
        .x86_64 => x86_64.supportsGemvNoTransUnitComplex(T),
        else => false,
    };
}

pub fn supportsGemvTransUnitComplex(comptime T: type) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_supportsGemvTransUnitComplex, T, .{});
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.supportsGemvTransUnitComplex(T),
        .x86_64 => x86_64.supportsGemvTransUnitComplex(T),
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
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvNoTransFullUnitReal, T, .{ m, n, alpha, a, lda, x, beta, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransFullUnitReal(T, m, n, alpha, a, lda, x, beta, y),
        .x86_64 => x86_64.gemvNoTransFullUnitReal(T, m, n, alpha, a, lda, x, beta, y),
        else => false,
    };
}

pub fn gemvNoTransPackLenUnitReal(comptime T: type, m: usize, n: usize, lda: BlasInt) ?usize {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvNoTransPackLenUnitReal, T, .{ m, n, lda });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransPackLenUnitReal(T, m, n, lda),
        .x86_64 => x86_64.gemvNoTransPackLenUnitReal(T, m, n, lda),
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
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvNoTransPackUnitReal, T, .{ n, alpha, x, pack });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransPackUnitReal(T, n, alpha, x, pack),
        .x86_64 => x86_64.gemvNoTransPackUnitReal(T, n, alpha, x, pack),
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
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gemvNoTransPackedRowsUnitReal, T, .{ row_count, n, a, lda, pack, scratch, y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gemvNoTransPackedRowsUnitReal(T, row_count, n, a, lda, pack, scratch, y),
        .x86_64 => x86_64.gemvNoTransPackedRowsUnitReal(T, row_count, n, a, lda, pack, scratch, y),
        else => false,
    };
}

pub fn supportsGemvNoTransUnitReal(comptime T: type) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_supportsGemvNoTransUnitReal, T, .{});
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.supportsGemvNoTransUnitReal(T),
        .x86_64 => x86_64.supportsGemvNoTransUnitReal(T),
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
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gerUnitReal, T, .{ m, n, alpha, x, y, a, lda });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gerUnitReal(T, m, n, alpha, x, y, a, lda),
        .x86_64 => x86_64.gerUnitReal(T, m, n, alpha, x, y, a, lda),
        else => false,
    };
}

pub fn gerUnitComplex(
    comptime T: type,
    m: usize,
    n: usize,
    alpha: T,
    x: [*]const T,
    y: [*]const T,
    a: [*]T,
    lda: BlasInt,
    conjugate_y: bool,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_gerUnitComplex, T, .{ m, n, alpha, x, y, a, lda, conjugate_y });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.gerUnitComplex(T, m, n, alpha, x, y, a, lda, conjugate_y),
        .x86_64 => x86_64.gerUnitComplex(T, m, n, alpha, x, y, a, lda, conjugate_y),
        else => false,
    };
}

pub fn triangularAxpyUnit(
    comptime T: type,
    n: usize,
    alpha: T,
    a: [*]const T,
    x: [*]T,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_triangularAxpyUnit, T, .{ n, alpha, a, x });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.triangularAxpyUnit(T, n, alpha, a, x),
        .x86_64 => x86_64.triangularAxpyUnit(T, n, alpha, a, x),
        else => false,
    };
}

pub fn triangularDotUnit(
    comptime T: type,
    n: usize,
    a: [*]const T,
    x: [*]const T,
    conjugate_a: bool,
    result: *T,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_triangularDotUnit, T, .{ n, a, x, conjugate_a, result });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.triangularDotUnit(T, n, a, x, conjugate_a, result),
        .x86_64 => x86_64.triangularDotUnit(T, n, a, x, conjugate_a, result),
        else => false,
    };
}

pub fn symmetricColumnsUnit(
    comptime T: type,
    upper: bool,
    hermitian: bool,
    n: usize,
    j0: usize,
    j1: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y_delta: [*]T,
) bool {
    if (comptime @import("zynum-build-options").dynamic_dispatch) return @import("../multiversion/client.zig").call(.matrix_vector_symmetricColumnsUnit, T, .{ upper, hermitian, n, j0, j1, alpha, a, lda, x, y_delta });
    return switch (builtin.cpu.arch) {
        .aarch64 => aarch64.symmetricColumnsUnit(T, upper, hermitian, n, j0, j1, alpha, a, lda, x, y_delta),
        .x86_64 => x86_64.symmetricColumnsUnit(T, upper, hermitian, n, j0, j1, alpha, a, lda, x, y_delta),
        else => false,
    };
}
