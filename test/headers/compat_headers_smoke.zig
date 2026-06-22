// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const c = @cImport({
    @cInclude("zynum/blas/cblas.h");
    @cInclude("zynum/blas/blas.h");
});

test "compatibility headers import through Zig" {
    comptime {
        _ = c.cblas_saxpy;
        _ = c.cblas_dgemv;
        _ = c.cblas_cgemm;
        _ = c.cblas_dsyr2k;
        _ = c.cblas_zher2k;
        _ = c.saxpy_;
        _ = c.dgemv_;
        _ = c.cgemm_;
        _ = c.dsyr2k_;
        _ = c.zher2k_;
    }

    const length: c.zynum_blas_int = 1;
    const source: f32 = 1.0;
    const destination: f32 = 0.0;
    const z: c.zynum_blas_complex_float = .{ .real = 1.0, .imag = 0.0 };

    try std.testing.expectEqual(@as(c.zynum_blas_int, 1), length);
    try std.testing.expectEqual(@as(f32, 1.0), source);
    try std.testing.expectEqual(@as(f32, 0.0), destination);
    try std.testing.expectEqual(@as(f32, 1.0), z.real);
    try std.testing.expectEqual(@as(f32, 0.0), z.imag);
    try std.testing.expectEqual(@as(c_int, 101), c.CblasRowMajor);
    try std.testing.expectEqual(@as(c_int, 111), c.CblasNoTrans);
}
