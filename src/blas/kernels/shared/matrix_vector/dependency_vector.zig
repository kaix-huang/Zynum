// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Allocation-free serial SIMD leaves for one triangular dependency step.
//!
//! This module deliberately has no runtime, task-pool, or architecture-dispatch
//! dependency so compact triangular arithmetic can live in a true link-isolated
//! object without instantiating a second execution runtime.

const std = @import("std");
const builtin = @import("builtin");

const types = @import("../../../types.zig");
const fixed_simd = @import("../vector/fixed_simd.zig");

fn isReal(comptime T: type) bool {
    return T == f32 or T == f64;
}

fn isComplex(comptime T: type) bool {
    return T == types.ComplexF32 or T == types.ComplexF64;
}

fn Real(comptime T: type) type {
    if (T == f32 or T == types.ComplexF32) return f32;
    if (T == f64 or T == types.ComplexF64) return f64;
    @compileError("triangular dependency vectors support BLAS real and complex scalars");
}

fn complexMul(comptime T: type, a: T, b: T) T {
    return .{
        .re = a.re * b.re - a.im * b.im,
        .im = a.re * b.im + a.im * b.re,
    };
}

fn config(comptime T: type) fixed_simd.Config {
    const R = Real(T);
    if (comptime builtin.cpu.arch == .x86_64) {
        const has_avx512 = std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f);
        const has_avx = std.Target.x86.featureSetHas(builtin.cpu.features, .avx);
        return .{
            .lane_count = if (R == f32)
                (if (has_avx512) 16 else if (has_avx) 8 else 4)
            else if (has_avx512)
                8
            else if (has_avx)
                4
            else
                2,
            .unroll_vectors = if (has_avx512) 6 else 4,
        };
    }
    return .{
        .lane_count = if (R == f32) 4 else 2,
        .unroll_vectors = 4,
    };
}

pub fn axpy(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) void {
    if (n == 0) return;
    const handled = if (comptime isReal(T))
        fixed_simd.axpyUnitReal(T, config(T), n, alpha, x, y)
    else if (comptime isComplex(T))
        fixed_simd.axpyUnitComplex(T, config(T), n, alpha, x, y)
    else
        false;
    if (handled) return;

    for (0..n) |i| {
        if (comptime isReal(T)) {
            y[i] = @mulAdd(T, alpha, x[i], y[i]);
        } else {
            const product = complexMul(T, alpha, x[i]);
            y[i] = .{ .re = y[i].re + product.re, .im = y[i].im + product.im };
        }
    }
}

pub fn dot(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjugate_x: bool) T {
    if (comptime isReal(T)) {
        return fixed_simd.dotUnitReal(T, config(T), n, x, y) orelse blk: {
            var sum: T = 0;
            for (0..n) |i| sum = @mulAdd(T, x[i], y[i], sum);
            break :blk sum;
        };
    }
    if (comptime isComplex(T)) {
        return fixed_simd.dotUnitComplex(T, config(T), n, x, y, conjugate_x) orelse blk: {
            var sum: T = .{ .re = 0, .im = 0 };
            for (0..n) |i| {
                const xv: T = if (conjugate_x) .{ .re = x[i].re, .im = -x[i].im } else x[i];
                const product = complexMul(T, xv, y[i]);
                sum = .{ .re = sum.re + product.re, .im = sum.im + product.im };
            }
            break :blk sum;
        };
    }
    @compileError("triangular dependency vectors support BLAS real and complex scalars");
}

test "dependency vectors cover short real and conjugated complex tails" {
    var real_x = [_]f64{ 1, 2, 3 };
    const real_a = [_]f64{ 4, 5, 6 };
    axpy(f64, real_x.len, 0.5, &real_a, &real_x);
    try std.testing.expectEqualSlices(f64, &.{ 3, 4.5, 6 }, &real_x);
    try std.testing.expectEqual(@as(f64, 32), dot(f64, real_x.len, &real_a, &.{ 1, 2, 3 }, false));

    const complex_x = [_]types.ComplexF64{
        .{ .re = 1, .im = 2 },
        .{ .re = -3, .im = 4 },
    };
    const complex_y = [_]types.ComplexF64{
        .{ .re = 5, .im = -1 },
        .{ .re = 2, .im = 3 },
    };
    const result = dot(types.ComplexF64, complex_x.len, &complex_x, &complex_y, true);
    try std.testing.expectEqual(@as(f64, 9), result.re);
    try std.testing.expectEqual(@as(f64, -28), result.im);
}
