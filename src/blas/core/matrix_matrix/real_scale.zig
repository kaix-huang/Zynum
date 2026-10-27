// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared original GEMM scaling branch, including beta-zero no-read semantics.
const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
pub fn columns(comptime T: type, m: usize, n0: usize, n1: usize, beta: T, c: [*]T, ldc: scalar.BlasInt) void {
    for (n0..n1) |j| {
        for (0..m) |i| {
            const index = indexing.matIndex(ldc, i, j);
            c[index] = if (scalar.isZero(T, beta)) scalar.zero(T) else scalar.mul(T, beta, c[index]);
        }
    }
}
