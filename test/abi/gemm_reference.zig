// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pure test-only oracle extracted from fortran_compat_test.zig. It imports no
//! ABI frontend, execution service, reference Context or default resource owner.
fn matIndex(lda: i32, row: usize, col: usize) usize {
    return row + col * @as(usize, @intCast(lda));
}

pub fn referenceGemmTransposedA(comptime T: type, transposed_b: bool, m: usize, n: usize, k: usize, alpha: T, a: []const T, lda: i32, b: []const T, ldb: i32, beta: T, c: []T, ldc: i32) void {
    for (0..n) |j| {
        for (0..m) |i| {
            var sum: T = 0;
            for (0..k) |p| {
                const bv = if (transposed_b) b[matIndex(ldb, j, p)] else b[matIndex(ldb, p, j)];
                sum = @mulAdd(T, a[matIndex(lda, p, i)], bv, sum);
            }
            const idxc = matIndex(ldc, i, j);
            c[idxc] = @mulAdd(T, alpha, sum, if (beta == 0) 0 else beta * c[idxc]);
        }
    }
}
