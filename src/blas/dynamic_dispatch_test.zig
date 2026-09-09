// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Correctness checks for a baseline dispatcher linked with ISA objects.
//! This root is deliberately separate from canonical inventory evidence.
const std = @import("std");
const builtin = @import("builtin");
const hardware = @import("hardware.zig");
const client = @import("kernels/multiversion/client.zig");
const protocol = @import("kernels/multiversion/protocol.zig");
const runtime = @import("runtime.zig");
const api = @import("api.zig");
const fortran = @import("abi/fortran.zig");

const scalars = .{ f32, f64, api.ComplexF32, api.ComplexF64 };

fn allCapabilities() hardware.Capabilities {
    var cap: hardware.Capabilities = .{};
    inline for (std.meta.fields(hardware.Capabilities)) |field| @field(cap, field.name) = true;
    return cap;
}

fn maximumTier() client.Tier {
    return switch (builtin.cpu.arch) {
        .aarch64 => .aarch64_sme2p1,
        .x86_64 => .x86_avx512,
        else => .baseline,
    };
}

fn value(comptime T: type, real: f64, imaginary: f64) T {
    if (T == f32 or T == f64) return @floatCast(real);
    return .{ .re = @floatCast(real), .im = @floatCast(imaginary) };
}

fn expectValue(comptime T: type, expected_real: f64, expected_imaginary: f64, actual: T) !void {
    if (T == f32 or T == f64) {
        try std.testing.expectApproxEqAbs(value(T, expected_real, 0), actual, value(T, 0.0001, 0));
    } else {
        try std.testing.expectApproxEqAbs(@as(f64, expected_real), @as(f64, actual.re), 0.0001);
        try std.testing.expectApproxEqAbs(@as(f64, expected_imaginary), @as(f64, actual.im), 0.0001);
    }
}

fn prefix(comptime T: type) []const u8 {
    return if (T == f32) "s" else if (T == f64) "d" else if (T == api.ComplexF32) "c" else "z";
}

test "ISA ceilings never grant features and baseline always stays baseline" {
    const full = allCapabilities();
    inline for (std.meta.tags(client.Tier)) |ceiling| {
        try std.testing.expectEqual(client.Tier.baseline, client.select(.{}, ceiling));
    }
    try std.testing.expectEqual(client.Tier.baseline, client.select(full, .baseline));
    try std.testing.expectEqual(maximumTier(), client.select(full, maximumTier()));
    if (builtin.cpu.arch == .aarch64) {
        try std.testing.expectEqual(client.Tier.aarch64_sve2, client.select(full, .aarch64_sve2));
        try std.testing.expectEqual(client.Tier.aarch64_sme, client.select(full, .aarch64_sme));
        try std.testing.expectEqual(client.Tier.aarch64_sme2, client.select(full, .aarch64_sme2));
    } else if (builtin.cpu.arch == .x86_64) {
        try std.testing.expectEqual(client.Tier.x86_avx, client.select(full, .x86_avx));
        try std.testing.expectEqual(client.Tier.x86_avx2_fma, client.select(full, .x86_avx2_fma));
    }
}

test "missing compiler prerequisites downgrade an otherwise capable host" {
    var cap = allCapabilities();
    if (builtin.cpu.arch == .aarch64) {
        cap.sme_f64f64 = false;
        try std.testing.expectEqual(client.Tier.aarch64_sme, client.select(cap, maximumTier()));
        cap.bf16 = false;
        try std.testing.expectEqual(client.Tier.aarch64_sve2, client.select(cap, maximumTier()));
        cap.fullfp16 = false;
        try std.testing.expectEqual(client.Tier.baseline, client.select(cap, maximumTier()));
    } else if (builtin.cpu.arch == .x86_64) {
        cap.f16c = false;
        try std.testing.expectEqual(client.Tier.x86_avx2_fma, client.select(cap, maximumTier()));
        cap.fma = false;
        try std.testing.expectEqual(client.Tier.x86_avx, client.select(cap, maximumTier()));
        cap.avx = false;
        try std.testing.expectEqual(client.Tier.baseline, client.select(cap, maximumTier()));
    }
}

test "selected tier agrees with host evidence and fresh process baseline ceiling" {
    const selected = client.selectedTier();
    std.debug.print("dynamic dispatch selected tier: {s}\n", .{@tagName(selected)});
    try std.testing.expectEqual(selected, client.select(hardware.detected(), selected));
    try std.testing.expectEqualStrings(@tagName(selected), runtime.selectedKernelTier());
    try std.testing.expectEqual(selected, client.selectedTier());
    if (std.c.getenv("ZYNUM_MAX_ISA")) |raw| {
        if (std.mem.eql(u8, std.mem.span(raw), "baseline")) {
            try std.testing.expectEqual(client.Tier.baseline, selected);
        }
    }
}

test "private packets keep arguments and results within aligned storage" {
    @setEvalBranchQuota(10_000);
    inline for (@typeInfo(protocol.Operation).@"enum".fields) |field| {
        const operation = comptime @field(protocol.Operation, field.name);
        inline for (.{ void, f32, f64, api.ComplexF32, api.ComplexF64 }) |T| {
            if (comptime protocol.validScalar(operation, T)) {
                const layout = protocol.packetLayout(operation, T);
                const Args = protocol.Args(operation, T);
                const Result = protocol.Result(operation, T);
                try std.testing.expect(layout.alignment >= @alignOf(Args));
                try std.testing.expect(layout.alignment >= @alignOf(Result));
                try std.testing.expectEqual(@as(usize, 0), layout.args_offset % @alignOf(Args));
                try std.testing.expectEqual(@as(usize, 0), layout.result_offset % @alignOf(Result));
                try std.testing.expect(layout.args_offset + @sizeOf(Args) <= layout.size);
                try std.testing.expect(layout.result_offset + @sizeOf(Result) <= layout.size);
                try std.testing.expectEqual(@as(usize, 0), layout.size % layout.alignment);
            }
        }
    }
}

test "ABI scaling and checked dot preserve vector tails for every scalar" {
    runtime.setMaxThreads(2);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    inline for (scalars) |T| {
        var x: [133]T = undefined;
        const y = [_]T{value(T, 1, 0)} ** 131;
        for (x[0..131], 0..) |*element, index| {
            element.* = value(T, @as(f64, @floatFromInt(index % 7)) - 3, 1);
        }
        x[131] = value(T, 777, -9);
        x[132] = value(T, 888, -8);
        const n: api.BlasInt = 131;
        const stride: api.BlasInt = 1;
        const scale = value(T, 2, 0);
        @field(fortran, prefix(T) ++ "scal_")(&n, &scale, &x, &stride);
        const dot = try api.dotProduct(.{
            .left_vector = try api.constVector(T, x[0..131], .{}),
            .right_vector = try api.constVector(T, &y, .{}),
        });
        try expectValue(T, -10, 262, dot);
        try expectValue(T, -6, 2, x[0]);
        try expectValue(T, 2, 2, x[130]);
        try expectValue(T, 777, -9, x[131]);
        try expectValue(T, 888, -8, x[132]);
    }
}

test "ABI GEMV handles padded leading dimensions and complex coefficients" {
    runtime.setMaxThreads(2);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    inline for (scalars) |T| {
        const numbers = [_]f64{ 1, 2, 3, 4, 5, 999, 999, 6, 7, 8, 9, 10, 999, 999, 11, 12, 13, 14, 15, 999, 999 };
        var a: [21]T = undefined;
        for (&a, numbers) |*element, number| element.* = value(T, number, 0);
        const x = [_]T{ value(T, 2, 1), value(T, -1, 0), value(T, 3, -1) };
        var y = [_]T{value(T, 1, 1)} ** 6;
        const m: api.BlasInt = 5;
        const n: api.BlasInt = 3;
        const lda: api.BlasInt = 7;
        const stride: api.BlasInt = 1;
        const alpha = value(T, 1, 0);
        const beta = value(T, 2, 0);
        @field(fortran, prefix(T) ++ "gemv_")("N", &m, &n, &alpha, &a, &lda, &x, &stride, &beta, &y, &stride);
        for ([_]f64{ 31, 35, 39, 43, 47 }, 0..) |expected, index| try expectValue(T, expected, -8, y[index]);
        try expectValue(T, 1, 1, y[5]);
    }
}

test "ABI GEMM matches explicit fringe products for all four scalar types" {
    runtime.setMaxThreads(2);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    inline for (scalars) |T| {
        var a = [_]T{value(T, -999, -999)} ** 49;
        var b = [_]T{value(T, -999, -999)} ** 24;
        var c = [_]T{value(T, -999, -999)} ** 18;
        for (0..7) |col| {
            for (0..5) |row| a[col * 7 + row] = value(T, @floatFromInt(row + 1), 1);
        }
        for (0..3) |col| {
            for (0..7) |row| b[col * 8 + row] = value(T, @floatFromInt(col + 1), 0);
            for (0..5) |row| c[col * 6 + row] = value(T, 1, -1);
        }
        const m: api.BlasInt = 5;
        const n: api.BlasInt = 3;
        const k: api.BlasInt = 7;
        const lda: api.BlasInt = 7;
        const ldb: api.BlasInt = 8;
        const ldc: api.BlasInt = 6;
        const alpha = value(T, 1, 0);
        const beta = value(T, 2, 0);
        @field(fortran, prefix(T) ++ "gemm_")("N", "N", &m, &n, &k, &alpha, &a, &lda, &b, &ldb, &beta, &c, &ldc);
        const expected = [_][5]f64{ .{ 9, 16, 23, 30, 37 }, .{ 16, 30, 44, 58, 72 }, .{ 23, 44, 65, 86, 107 } };
        for (expected, [_]f64{ 5, 12, 19 }, 0..) |column, imaginary, col| {
            for (column, 0..) |real, row| try expectValue(T, real, imaginary, c[col * 6 + row]);
            try expectValue(T, -999, -999, c[col * 6 + 5]);
        }
    }
}

test "checked workspace aliasing works through the selected object" {
    runtime.setMaxThreads(2);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    var a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 2, 0, 0, 3 };
    var workspace: [4]f64 = undefined;
    const shape = .{ .row_count = 2, .column_count = 2 };
    try api.matrixMultiplyWithWorkspace(.{
        .left_matrix = try api.constMatrix(f64, &a, shape),
        .right_matrix = try api.constMatrix(f64, &b, shape),
        .result_matrix = try api.matrix(f64, &a, shape),
        .workspace = &workspace,
    });
    try std.testing.expectEqualSlices(f64, &.{ 2, 4, 9, 12 }, &a);
    const x = [_]f64{1};
    var y = [_]f64{ 55, 66 };
    try std.testing.expectError(error.DimensionMismatch, api.matrixVectorMultiply(.{
        .matrix = try api.constMatrix(f64, &a, shape),
        .input_vector = try api.constVector(f64, &x, .{}),
        .result_vector = try api.vector(f64, &y, .{}),
    }));
    try std.testing.expectEqualSlices(f64, &.{ 55, 66 }, &y);
}

test "parallel fringe GEMM remains usable after repeated shutdown" {
    runtime.setMaxThreads(2);
    try std.testing.expect(runtime.maxThreads() <= @min(@as(usize, 2), runtime.totalThreadCount()));
    if (comptime @import("zynum-build-options").thread_limit != 0) {
        try std.testing.expect(runtime.maxThreads() <= @import("zynum-build-options").thread_limit);
        try std.testing.expect(runtime.hasExplicitThreadLimit());
    }
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(f64, 193 * 191);
    defer allocator.free(a);
    const b = try allocator.alloc(f64, 191 * 197);
    defer allocator.free(b);
    const c = try allocator.alloc(f64, 193 * 197);
    defer allocator.free(c);
    @memset(a, 1);
    @memset(b, 1);
    for (0..2) |_| {
        @memset(c, -123);
        try api.matrixMultiply(.{
            .left_matrix = try api.constMatrix(f64, a, .{ .row_count = 193, .column_count = 191 }),
            .right_matrix = try api.constMatrix(f64, b, .{ .row_count = 191, .column_count = 197 }),
            .result_matrix = try api.matrix(f64, c, .{ .row_count = 193, .column_count = 197 }),
        });
        for (c) |actual| try std.testing.expectEqual(@as(f64, 191), actual);
        fortran.zynum_blas_shutdown();
        fortran.zynum_blas_shutdown();
    }
}
