// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const c = @cImport({
    @cInclude("zynum/blas/cblas.h");
    @cInclude("zynum/blas/blas.h");
});

test "compatibility headers import through Zig" {
    comptime {
        if (!@hasDecl(c, "cblas_saxpy")) @compileError("missing cblas_saxpy");
        if (!@hasDecl(c, "cblas_dgemv")) @compileError("missing cblas_dgemv");
        if (!@hasDecl(c, "cblas_cgemm")) @compileError("missing cblas_cgemm");
        if (!@hasDecl(c, "cblas_dsyr2k")) @compileError("missing cblas_dsyr2k");
        if (!@hasDecl(c, "cblas_zher2k")) @compileError("missing cblas_zher2k");
        if (!@hasDecl(c, "saxpy_")) @compileError("missing saxpy_");
        if (!@hasDecl(c, "dgemv_")) @compileError("missing dgemv_");
        if (!@hasDecl(c, "cgemm_")) @compileError("missing cgemm_");
        if (!@hasDecl(c, "dsyr2k_")) @compileError("missing dsyr2k_");
        if (!@hasDecl(c, "zher2k_")) @compileError("missing zher2k_");
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
