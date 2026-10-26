// Copyright (C) 2026 Zynum contributors
// Copyright (C) 2026 Scientific Platform contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Adapted from gemmWithResources/gemmComplexPortable in upstream
//! src/blas/core/matrix_matrix/gemm.zig at c5cb123e55eaae683de09b0c8030837f5600f140.
//! Changes: retain only serial reference loop, parameterize element access for
//! Heteris byte strides and transpose/conjugation, and move admission to dense.zig.
//! Removed optimized dispatch, pool, global/TLS scratch and BLAS integer limits.
//! Arithmetic order and alpha/beta zero branches match the reference path.
const scalar = @import("scientific_sources").scalar;
pub fn gemm(comptime T: type, comptime Matrix: type, m: usize, n: usize, k: usize, alpha: T, a: Matrix, b: Matrix, beta: T, out: Matrix) void {
    if (k == 0 or scalar.isZero(T, alpha)) {
        for (0..n) |j| for (0..m) |i| {
            const dest = out.at(i, j);
            dest.* = if (scalar.isZero(T, beta)) scalar.zero(T) else scalar.mul(T, beta, dest.*);
        };
        return;
    }
    for (0..n) |j| {
        for (0..m) |i| {
            var sum = scalar.zero(T);
            for (0..k) |p| sum = scalar.add(T, sum, scalar.mul(T, a.get(i, p), b.get(p, j)));
            const dest = out.at(i, j);
            dest.* = scalar.add(T, scalar.mul(T, alpha, sum), if (scalar.isZero(T, beta)) scalar.zero(T) else scalar.mul(T, beta, dest.*));
        }
    }
}
