// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const matrix_vector_ops = @import("../matrix_vector.zig");
const gemm_impl = @import("gemm.zig");

const BlasInt = scalar.BlasInt;
const Uplo = scalar.Uplo;
const Side = scalar.Side;

const zero = scalar.zero;
const one = scalar.one;
const add = scalar.add;
const mul = scalar.mul;
const isComplex = scalar.isComplex;
const isOne = scalar.isOne;
const isZero = scalar.isZero;

const toUsize = indexing.toUsize;
const matIndex = indexing.matIndex;
const symValue = matrix_vector_ops.symValue;

// Keep smaller controls allocation-free; all current Level 3 target shapes meet this gate.
const dense_symm_min_dimension = 128;
const dense_symm_max_workspace_bytes = 64 * 1024 * 1024;

fn useDenseSymmGemm(comptime T: type, m: usize, n: usize, order: usize) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    if (comptime T == scalar.ComplexF64) {
        if (m != n and order < 256) return false;
    }
    return m >= dense_symm_min_dimension and n >= dense_symm_min_dimension and order >= dense_symm_min_dimension;
}

fn scaleSymmOutput(comptime T: type, m: usize, n: usize, beta: T, c: [*]T, ldc: BlasInt) void {
    if (isOne(T, beta)) return;
    for (0..n) |j| {
        for (0..m) |i| {
            const index = matIndex(ldc, i, j);
            c[index] = if (isZero(T, beta)) zero(T) else mul(T, beta, c[index]);
        }
    }
}

pub noinline fn trySymm(comptime T: type, side: Side, uplo: Uplo, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) bool {
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (isZero(T, alpha)) {
        scaleSymmOutput(T, m, n, beta, c, ldc);
        return true;
    }
    const order = if (side == .left) m else n;
    if (!useDenseSymmGemm(T, m, n, order)) return false;

    const dense_len = std.math.mul(usize, order, order) catch return false;
    const saved_c_len = if (comptime isComplex(T))
        if (isZero(T, beta)) 0 else std.math.mul(usize, m, n) catch return false
    else
        0;
    const workspace_len = std.math.add(usize, dense_len, saved_c_len) catch return false;
    const workspace_bytes = std.math.mul(usize, workspace_len, @sizeOf(T)) catch return false;
    if (workspace_bytes > dense_symm_max_workspace_bytes) return false;

    const workspace = std.heap.c_allocator.alloc(T, workspace_len) catch return false;
    defer std.heap.c_allocator.free(workspace);
    const dense_a = workspace[0..dense_len];
    const saved_c = workspace[dense_len..];

    for (0..order) |j| {
        for (0..order) |i| {
            dense_a[i + j * order] = symValue(T, uplo, a, lda, i, j, herm);
        }
    }

    if (comptime isComplex(T)) {
        // Complex GEMM's packed paths require unit alpha and zero beta, so preserve logical C and apply both scalars after the product.
        if (saved_c.len != 0) {
            for (0..n) |j| {
                for (0..m) |i| saved_c[i + j * m] = c[matIndex(ldc, i, j)];
            }
        }
        if (side == .left) {
            gemm_impl.gemm(T, .no_trans, .no_trans, m_, n_, m_, one(T), dense_a.ptr, m_, b, ldb, zero(T), c, ldc);
        } else {
            gemm_impl.gemm(T, .no_trans, .no_trans, m_, n_, n_, one(T), b, ldb, dense_a.ptr, n_, zero(T), c, ldc);
        }
        for (0..n) |j| {
            for (0..m) |i| {
                const index = matIndex(ldc, i, j);
                const product = if (isOne(T, alpha)) c[index] else mul(T, alpha, c[index]);
                c[index] = if (saved_c.len == 0) product else add(T, product, mul(T, beta, saved_c[i + j * m]));
            }
        }
    } else if (side == .left) {
        gemm_impl.gemm(T, .no_trans, .no_trans, m_, n_, m_, alpha, dense_a.ptr, m_, b, ldb, beta, c, ldc);
    } else {
        gemm_impl.gemm(T, .no_trans, .no_trans, m_, n_, n_, alpha, b, ldb, dense_a.ptr, n_, beta, c, ldc);
    }
    return true;
}
