// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const dispatch = @import("core/vector/stride2_dispatch.zig");
const ops = @import("core/vector/operations.zig");
const runtime = @import("runtime.zig");
const types = @import("types.zig");

const BlasInt = types.BlasInt;
const ComplexF32 = types.ComplexF32;
const ComplexF64 = types.ComplexF64;
const large_odd_n: usize = 512 * 1024 + 17;

const RealUpdate = enum { scal, swap, axpy, axpby, rot, rotm };
const ComplexUpdate = enum { scal, rscal, swap, axpy_real, axpy, axpby, rot };

fn asBlasInt(value: usize) BlasInt {
    return @intCast(value);
}

fn targetAvailable() bool {
    return builtin.cpu.arch == .x86_64 and runtime.totalThreadCount() > 1;
}

fn Real(comptime T: type) type {
    return if (T == ComplexF32) f32 else f64;
}

fn realInitialX(comptime T: type, i: usize) T {
    return @floatCast((@as(f64, @floatFromInt(i % 31)) - 15) / 16);
}

fn realInitialY(comptime T: type, i: usize) T {
    return @floatCast((@as(f64, @floatFromInt(i % 29)) - 14) / 15);
}

fn resetReal(comptime T: type, x: []T, y: []T) void {
    for (0..large_odd_n) |i| {
        const physical = 2 * i;
        x[physical] = realInitialX(T, i);
        y[physical] = realInitialY(T, i);
        if (physical + 1 < x.len) {
            x[physical + 1] = 12345;
            y[physical + 1] = -23456;
        }
    }
}

fn runRealReference(comptime T: type, update: RealUpdate, x: []T, y: []T) void {
    const n = asBlasInt(large_odd_n);
    const inc: BlasInt = 2;
    switch (update) {
        .scal => ops.scal(T, n, @as(T, 0.625), x.ptr, inc),
        .swap => ops.swap(T, n, x.ptr, inc, y.ptr, inc),
        .axpy => ops.axpy(T, n, @as(T, -0.75), x.ptr, inc, y.ptr, inc),
        .axpby => ops.axpby(T, n, @as(T, -0.75), x.ptr, inc, @as(T, 0.375), y.ptr, inc),
        .rot => ops.rot(T, n, x.ptr, inc, y.ptr, inc, @as(T, 0.8), @as(T, -0.6)),
        .rotm => {
            const param = [_]T{ -1, 0.75, -0.25, 0.5, 0.625 };
            ops.rotm(T, n, x.ptr, inc, y.ptr, inc, &param);
        },
    }
}

fn runRealUpdate(comptime T: type, update: RealUpdate, x: []T, y: []T) bool {
    const n = asBlasInt(large_odd_n);
    const inc: BlasInt = 2;
    return switch (update) {
        .scal => dispatch.scal(T, n, @as(T, 0.625), x.ptr, inc),
        .swap => dispatch.swap(T, n, x.ptr, inc, y.ptr, inc),
        .axpy => dispatch.axpy(T, n, @as(T, -0.75), x.ptr, inc, y.ptr, inc),
        .axpby => dispatch.axpby(T, n, @as(T, -0.75), x.ptr, inc, @as(T, 0.375), y.ptr, inc),
        .rot => dispatch.rot(T, n, x.ptr, inc, y.ptr, inc, @as(T, 0.8), @as(T, -0.6)),
        .rotm => blk: {
            const param = [_]T{ -1, 0.75, -0.25, 0.5, 0.625 };
            break :blk dispatch.rotm(T, n, x.ptr, inc, y.ptr, inc, &param);
        },
    };
}

fn expectedRealUpdate(comptime T: type, update: RealUpdate, i: usize) struct { x: T, y: T } {
    const x = realInitialX(T, i);
    const y = realInitialY(T, i);
    return switch (update) {
        .scal => .{ .x = x * 0.625, .y = y },
        .swap => .{ .x = y, .y = x },
        .axpy => .{ .x = x, .y = @mulAdd(T, -0.75, x, y) },
        .axpby => .{ .x = x, .y = @mulAdd(T, -0.75, x, 0.375 * y) },
        .rot => .{
            .x = @mulAdd(T, x, 0.8, y * -0.6),
            .y = @mulAdd(T, -x, -0.6, y * 0.8),
        },
        .rotm => .{
            .x = @mulAdd(T, x, 0.75, y * 0.5),
            .y = @mulAdd(T, x, -0.25, y * 0.625),
        },
    };
}

fn expectRealUpdate(comptime T: type, update: RealUpdate, single_x: []const T, single_y: []const T, multi_x: []const T, multi_y: []const T) !void {
    const tolerance: T = if (T == f32) 2e-6 else 2e-14;
    for (0..large_odd_n) |i| {
        const physical = 2 * i;
        const expected = expectedRealUpdate(T, update, i);
        try std.testing.expectApproxEqAbs(expected.x, single_x[physical], tolerance);
        try std.testing.expectApproxEqAbs(expected.y, single_y[physical], tolerance);
        try std.testing.expectEqual(single_x[physical], multi_x[physical]);
        try std.testing.expectEqual(single_y[physical], multi_y[physical]);
        if (physical + 1 < single_x.len) {
            try std.testing.expectEqual(@as(T, 12345), single_x[physical + 1]);
            try std.testing.expectEqual(@as(T, -23456), single_y[physical + 1]);
            try std.testing.expectEqual(single_x[physical + 1], multi_x[physical + 1]);
            try std.testing.expectEqual(single_y[physical + 1], multi_y[physical + 1]);
        }
    }
}

fn checkRealUpdates(comptime T: type) !void {
    const allocator = std.testing.allocator;
    const storage_len = 2 * large_odd_n - 1;
    const single_x = try allocator.alloc(T, storage_len);
    defer allocator.free(single_x);
    const single_y = try allocator.alloc(T, storage_len);
    defer allocator.free(single_y);
    const multi_x = try allocator.alloc(T, storage_len);
    defer allocator.free(multi_x);
    const multi_y = try allocator.alloc(T, storage_len);
    defer allocator.free(multi_y);
    defer runtime.setMaxThreads(0);

    inline for (std.meta.tags(RealUpdate)) |update| {
        resetReal(T, single_x, single_y);
        resetReal(T, multi_x, multi_y);
        runtime.setMaxThreads(1);
        runRealReference(T, update, single_x, single_y);
        runtime.setMaxThreads(2);
        try std.testing.expect(runRealUpdate(T, update, multi_x, multi_y));
        try expectRealUpdate(T, update, single_x, single_y, multi_x, multi_y);
    }
}

fn complexValue(comptime T: type, re: f64, im: f64) T {
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}

fn complexAdd(comptime T: type, a: T, b: T) T {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}

fn complexSub(comptime T: type, a: T, b: T) T {
    return .{ .re = a.re - b.re, .im = a.im - b.im };
}

fn complexMul(comptime T: type, a: T, b: T) T {
    return .{
        .re = a.re * b.re - a.im * b.im,
        .im = a.re * b.im + a.im * b.re,
    };
}

fn complexScale(comptime T: type, a: T, value: anytype) T {
    return .{ .re = a.re * value, .im = a.im * value };
}

fn complexInitialX(comptime T: type, i: usize) T {
    return complexValue(T, (@as(f64, @floatFromInt(i % 23)) - 11) / 12, (@as(f64, @floatFromInt(i % 19)) - 9) / 10);
}

fn complexInitialY(comptime T: type, i: usize) T {
    return complexValue(T, (@as(f64, @floatFromInt(i % 17)) - 8) / 9, (@as(f64, @floatFromInt(i % 13)) - 6) / 7);
}

fn resetComplex(comptime T: type, x: []T, y: []T) void {
    for (0..large_odd_n) |i| {
        const physical = 2 * i;
        x[physical] = complexInitialX(T, i);
        y[physical] = complexInitialY(T, i);
        if (physical + 1 < x.len) {
            x[physical + 1] = complexValue(T, 12345, -12345);
            y[physical + 1] = complexValue(T, -23456, 23456);
        }
    }
}

fn runComplexReference(comptime T: type, update: ComplexUpdate, x: []T, y: []T) void {
    const n = asBlasInt(large_odd_n);
    const inc: BlasInt = 2;
    const alpha = complexValue(T, -0.75, 0.5);
    const beta = complexValue(T, 0.375, -0.25);
    switch (update) {
        .scal => ops.scal(T, n, alpha, x.ptr, inc),
        .rscal => ops.rscal(T, n, @as(Real(T), 0.625), x.ptr, inc),
        .swap => ops.swap(T, n, x.ptr, inc, y.ptr, inc),
        .axpy_real => ops.axpy(T, n, complexValue(T, -0.75, 0), x.ptr, inc, y.ptr, inc),
        .axpy => ops.axpy(T, n, alpha, x.ptr, inc, y.ptr, inc),
        .axpby => ops.axpby(T, n, alpha, x.ptr, inc, beta, y.ptr, inc),
        .rot => ops.rot(T, n, x.ptr, inc, y.ptr, inc, @as(Real(T), 0.8), complexValue(T, -0.6, 0)),
    }
}

fn runComplexUpdate(comptime T: type, update: ComplexUpdate, x: []T, y: []T) bool {
    const n = asBlasInt(large_odd_n);
    const inc: BlasInt = 2;
    const alpha = complexValue(T, -0.75, 0.5);
    const beta = complexValue(T, 0.375, -0.25);
    return switch (update) {
        .scal => dispatch.scal(T, n, alpha, x.ptr, inc),
        .rscal => dispatch.rscal(T, n, @as(Real(T), 0.625), x.ptr, inc),
        .swap => dispatch.swap(T, n, x.ptr, inc, y.ptr, inc),
        .axpy_real => dispatch.axpy(T, n, complexValue(T, -0.75, 0), x.ptr, inc, y.ptr, inc),
        .axpy => dispatch.axpy(T, n, alpha, x.ptr, inc, y.ptr, inc),
        .axpby => dispatch.axpby(T, n, alpha, x.ptr, inc, beta, y.ptr, inc),
        .rot => dispatch.rot(T, n, x.ptr, inc, y.ptr, inc, @as(Real(T), 0.8), complexValue(T, -0.6, 0)),
    };
}

fn expectedComplexUpdate(comptime T: type, update: ComplexUpdate, i: usize) struct { x: T, y: T } {
    const x = complexInitialX(T, i);
    const y = complexInitialY(T, i);
    const alpha = complexValue(T, -0.75, 0.5);
    const beta = complexValue(T, 0.375, -0.25);
    return switch (update) {
        .scal => .{ .x = complexMul(T, alpha, x), .y = y },
        .rscal => .{ .x = complexScale(T, x, 0.625), .y = y },
        .swap => .{ .x = y, .y = x },
        .axpy_real => .{ .x = x, .y = complexAdd(T, y, complexScale(T, x, -0.75)) },
        .axpy => .{ .x = x, .y = complexAdd(T, y, complexMul(T, alpha, x)) },
        .axpby => .{ .x = x, .y = complexAdd(T, complexMul(T, alpha, x), complexMul(T, beta, y)) },
        .rot => .{
            .x = complexAdd(T, complexScale(T, x, 0.8), complexScale(T, y, -0.6)),
            .y = complexSub(T, complexScale(T, y, 0.8), complexScale(T, x, -0.6)),
        },
    };
}

fn expectComplexApprox(comptime T: type, expected: T, actual: T) !void {
    const tolerance: Real(T) = if (T == ComplexF32) 3e-6 else 3e-14;
    try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
}

fn FloatBits(comptime R: type) type {
    return if (R == f32) u32 else u64;
}

fn negativeZero(comptime R: type) R {
    const Bits = FloatBits(R);
    const sign_mask: Bits = @as(Bits, 1) << (@bitSizeOf(Bits) - 1);
    return @bitCast(sign_mask);
}

fn resetComplexRealScale(comptime T: type, x: []T) void {
    const R = Real(T);
    const neg_zero = negativeZero(R);
    for (0..large_odd_n) |i| {
        const physical = 2 * i;
        x[physical] = .{ .re = 0, .im = neg_zero };
        if (physical + 1 < x.len) x[physical + 1] = complexValue(T, 12345, -12345);
    }
    x[0] = .{ .re = std.math.inf(R), .im = 1 };
    x[2] = .{ .re = -std.math.inf(R), .im = -1 };
    x[4] = .{ .re = 0, .im = neg_zero };
    x[6] = .{ .re = neg_zero, .im = 0 };
}

fn expectComplexRealScale(comptime T: type, x: []const T) !void {
    const R = Real(T);
    const Bits = FloatBits(R);
    const sign_mask: Bits = @as(Bits, 1) << (@bitSizeOf(Bits) - 1);
    try std.testing.expect(std.math.isInf(x[0].re) and x[0].re > 0);
    try std.testing.expectEqual(@as(R, 2), x[0].im);
    try std.testing.expect(std.math.isInf(x[2].re) and x[2].re < 0);
    try std.testing.expectEqual(@as(R, -2), x[2].im);
    try std.testing.expectEqual(@as(Bits, 0), @as(Bits, @bitCast(x[4].re)));
    try std.testing.expectEqual(sign_mask, @as(Bits, @bitCast(x[4].im)));
    try std.testing.expectEqual(sign_mask, @as(Bits, @bitCast(x[6].re)));
    try std.testing.expectEqual(@as(Bits, 0), @as(Bits, @bitCast(x[6].im)));
    for (0..large_odd_n - 1) |i| {
        try std.testing.expectEqual(complexValue(T, 12345, -12345), x[2 * i + 1]);
    }
}

fn checkComplexRealScale(comptime T: type) !void {
    const R = Real(T);
    const allocator = std.testing.allocator;
    const storage_len = 2 * large_odd_n - 1;
    const x = try allocator.alloc(T, storage_len);
    defer allocator.free(x);
    const n = asBlasInt(large_odd_n);
    const inc: BlasInt = 2;
    const alpha: R = 2;
    defer runtime.setMaxThreads(0);
    runtime.setMaxThreads(2);

    resetComplexRealScale(T, x);
    try std.testing.expect(dispatch.rscal(T, n, alpha, x.ptr, inc));
    try expectComplexRealScale(T, x);
}

fn highPrecisionNrm2Expected(comptime R: type, component_count: usize) R {
    const W = if (R == f32) f64 else f128;
    const scale: W = @floatCast(std.math.floatTrueMin(R));
    const count: W = @floatFromInt(component_count);
    return @floatCast(scale * @sqrt(count));
}

fn checkSubnormalRealNrm2(comptime T: type) !void {
    const allocator = std.testing.allocator;
    const storage_len = 2 * large_odd_n - 1;
    const x = try allocator.alloc(T, storage_len);
    defer allocator.free(x);
    const tiny = std.math.floatTrueMin(T);
    for (0..large_odd_n) |i| {
        x[2 * i] = if (i % 2 == 0) tiny else -tiny;
        if (2 * i + 1 < storage_len) x[2 * i + 1] = 1;
    }
    const n = asBlasInt(large_odd_n);
    const inc: BlasInt = 2;
    defer runtime.setMaxThreads(0);
    runtime.setMaxThreads(2);
    const candidate = dispatch.nrm2(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
    const expected = highPrecisionNrm2Expected(T, large_odd_n);
    const tolerance = tiny * @as(T, 32);
    try std.testing.expect(std.math.isFinite(candidate) and candidate > 0);
    try std.testing.expectApproxEqAbs(expected, candidate, tolerance);
}

fn checkSubnormalComplexNrm2(comptime T: type) !void {
    const R = Real(T);
    const allocator = std.testing.allocator;
    const storage_len = 2 * large_odd_n - 1;
    const x = try allocator.alloc(T, storage_len);
    defer allocator.free(x);
    const tiny = std.math.floatTrueMin(R);
    for (0..large_odd_n) |i| {
        x[2 * i] = .{ .re = tiny, .im = -tiny };
        if (2 * i + 1 < storage_len) x[2 * i + 1] = complexValue(T, 1, 1);
    }
    const n = asBlasInt(large_odd_n);
    const inc: BlasInt = 2;
    defer runtime.setMaxThreads(0);
    runtime.setMaxThreads(2);
    const candidate = dispatch.nrm2(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
    const expected = highPrecisionNrm2Expected(R, 2 * large_odd_n);
    const tolerance = tiny * @as(R, 32);
    try std.testing.expect(std.math.isFinite(candidate) and candidate > 0);
    try std.testing.expectApproxEqAbs(expected, candidate, tolerance);
}

fn checkComplexUpdates(comptime T: type) !void {
    const allocator = std.testing.allocator;
    const storage_len = 2 * large_odd_n - 1;
    const single_x = try allocator.alloc(T, storage_len);
    defer allocator.free(single_x);
    const single_y = try allocator.alloc(T, storage_len);
    defer allocator.free(single_y);
    const multi_x = try allocator.alloc(T, storage_len);
    defer allocator.free(multi_x);
    const multi_y = try allocator.alloc(T, storage_len);
    defer allocator.free(multi_y);
    defer runtime.setMaxThreads(0);

    inline for (std.meta.tags(ComplexUpdate)) |update| {
        resetComplex(T, single_x, single_y);
        resetComplex(T, multi_x, multi_y);
        runtime.setMaxThreads(1);
        runComplexReference(T, update, single_x, single_y);
        runtime.setMaxThreads(2);
        try std.testing.expect(runComplexUpdate(T, update, multi_x, multi_y));
        for (0..large_odd_n) |i| {
            const physical = 2 * i;
            const expected = expectedComplexUpdate(T, update, i);
            try expectComplexApprox(T, expected.x, single_x[physical]);
            try expectComplexApprox(T, expected.y, single_y[physical]);
            try expectComplexApprox(T, single_x[physical], multi_x[physical]);
            try expectComplexApprox(T, single_y[physical], multi_y[physical]);
            if (physical + 1 < storage_len) {
                try std.testing.expectEqual(complexValue(T, 12345, -12345), single_x[physical + 1]);
                try std.testing.expectEqual(complexValue(T, -23456, 23456), single_y[physical + 1]);
                try std.testing.expectEqual(single_x[physical + 1], multi_x[physical + 1]);
                try std.testing.expectEqual(single_y[physical + 1], multi_y[physical + 1]);
            }
        }
    }
}

fn checkRealReductions(comptime T: type) !void {
    const allocator = std.testing.allocator;
    const storage_len = 2 * large_odd_n - 1;
    const x = try allocator.alloc(T, storage_len);
    defer allocator.free(x);
    const y = try allocator.alloc(T, storage_len);
    defer allocator.free(y);
    for (0..large_odd_n) |i| {
        x[2 * i] = @floatFromInt(@as(i32, @intCast(i % 3)) - 1);
        y[2 * i] = @floatFromInt(@as(i32, @intCast(i % 5)) - 2);
        if (2 * i + 1 < storage_len) {
            x[2 * i + 1] = 12345;
            y[2 * i + 1] = -23456;
        }
    }
    const n = asBlasInt(large_odd_n);
    const inc: BlasInt = 2;
    defer runtime.setMaxThreads(0);
    runtime.setMaxThreads(1);
    const single_dot = ops.dot(T, n, x.ptr, inc, y.ptr, inc, false);
    const single_asum = ops.asum(T, n, x.ptr, inc);
    const single_nrm2 = ops.nrm2(T, n, x.ptr, inc);
    const single_mixed = if (T == f32) ops.dotF32AccF64(n, x.ptr, inc, y.ptr, inc) else 0;
    runtime.setMaxThreads(2);
    const multi_dot = dispatch.dot(T, n, x.ptr, inc, y.ptr, inc, false) orelse return error.UnexpectedStride2DispatchMiss;
    const multi_asum = dispatch.asum(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
    const multi_nrm2 = dispatch.nrm2(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
    const multi_mixed = if (T == f32)
        dispatch.dotF32AccF64(n, x.ptr, inc, y.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss
    else
        0;
    const tolerance: T = if (T == f32) 0.01 else 1e-10;
    try std.testing.expectApproxEqAbs(single_dot, multi_dot, tolerance);
    try std.testing.expectApproxEqAbs(single_asum, multi_asum, tolerance);
    try std.testing.expectApproxEqAbs(single_nrm2, multi_nrm2, tolerance);
    if (T == f32) try std.testing.expectApproxEqAbs(single_mixed, multi_mixed, 1e-10);

    const nan_index = large_odd_n / 2;
    const first_max = nan_index + 1;
    const second_max = large_odd_n - 9;
    x[2 * nan_index] = std.math.nan(T);
    x[2 * first_max] = -100;
    x[2 * second_max] = 100;
    runtime.setMaxThreads(1);
    const single_iamax = ops.iamax(T, n, x.ptr, inc);
    runtime.setMaxThreads(2);
    const multi_iamax = dispatch.iamax(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
    try std.testing.expectEqual(@as(BlasInt, @intCast(first_max + 1)), single_iamax);
    try std.testing.expectEqual(single_iamax, multi_iamax);
    const nan_dot = dispatch.dot(T, n, x.ptr, inc, y.ptr, inc, false) orelse return error.UnexpectedStride2DispatchMiss;
    const nan_asum = dispatch.asum(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
    const nan_nrm2 = dispatch.nrm2(T, n, x.ptr, inc) orelse ops.nrm2(T, n, x.ptr, inc);
    try std.testing.expect(std.math.isNan(nan_dot));
    try std.testing.expect(std.math.isNan(nan_asum));
    try std.testing.expect(std.math.isNan(nan_nrm2));
    if (T == f32) {
        const nan_mixed = dispatch.dotF32AccF64(n, x.ptr, inc, y.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
        try std.testing.expect(std.math.isNan(nan_mixed));
    }
    x[0] = std.math.nan(T);
    try std.testing.expectEqual(@as(BlasInt, 1), dispatch.iamax(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss);
}

fn checkComplexReductions(comptime T: type) !void {
    const R = Real(T);
    const allocator = std.testing.allocator;
    const storage_len = 2 * large_odd_n - 1;
    const x = try allocator.alloc(T, storage_len);
    defer allocator.free(x);
    const y = try allocator.alloc(T, storage_len);
    defer allocator.free(y);
    for (0..large_odd_n) |i| {
        x[2 * i] = complexValue(T, @floatFromInt(@as(i32, @intCast(i % 3)) - 1), @floatFromInt(@as(i32, @intCast(i % 5)) - 2));
        y[2 * i] = complexValue(T, @floatFromInt(@as(i32, @intCast(i % 7)) - 3), @floatFromInt(@as(i32, @intCast(i % 3)) - 1));
        if (2 * i + 1 < storage_len) {
            x[2 * i + 1] = complexValue(T, 12345, -12345);
            y[2 * i + 1] = complexValue(T, -23456, 23456);
        }
    }
    const n = asBlasInt(large_odd_n);
    const inc: BlasInt = 2;
    defer runtime.setMaxThreads(0);
    runtime.setMaxThreads(1);
    const single_dotu = ops.dot(T, n, x.ptr, inc, y.ptr, inc, false);
    const single_dotc = ops.dot(T, n, x.ptr, inc, y.ptr, inc, true);
    const single_asum = ops.asum(T, n, x.ptr, inc);
    const single_nrm2 = ops.nrm2(T, n, x.ptr, inc);
    runtime.setMaxThreads(2);
    const multi_dotu = dispatch.dot(T, n, x.ptr, inc, y.ptr, inc, false) orelse return error.UnexpectedStride2DispatchMiss;
    const multi_dotc = dispatch.dot(T, n, x.ptr, inc, y.ptr, inc, true) orelse return error.UnexpectedStride2DispatchMiss;
    const multi_asum = dispatch.asum(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
    const multi_nrm2 = dispatch.nrm2(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
    const tolerance: R = if (T == ComplexF32) 0.1 else 1e-9;
    try std.testing.expectApproxEqAbs(single_dotu.re, multi_dotu.re, tolerance);
    try std.testing.expectApproxEqAbs(single_dotu.im, multi_dotu.im, tolerance);
    try std.testing.expectApproxEqAbs(single_dotc.re, multi_dotc.re, tolerance);
    try std.testing.expectApproxEqAbs(single_dotc.im, multi_dotc.im, tolerance);
    try std.testing.expectApproxEqAbs(single_asum, multi_asum, tolerance);
    try std.testing.expectApproxEqAbs(single_nrm2, multi_nrm2, tolerance);

    const nan_index = large_odd_n / 2;
    const first_max = nan_index + 1;
    const second_max = large_odd_n - 9;
    x[2 * nan_index] = complexValue(T, std.math.nan(R), 0);
    x[2 * first_max] = complexValue(T, -60, 40);
    x[2 * second_max] = complexValue(T, 50, -50);
    runtime.setMaxThreads(1);
    const single_iamax = ops.iamax(T, n, x.ptr, inc);
    runtime.setMaxThreads(2);
    const multi_iamax = dispatch.iamax(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
    try std.testing.expectEqual(@as(BlasInt, @intCast(first_max + 1)), single_iamax);
    try std.testing.expectEqual(single_iamax, multi_iamax);
    const nan_dot = dispatch.dot(T, n, x.ptr, inc, y.ptr, inc, false) orelse return error.UnexpectedStride2DispatchMiss;
    const nan_asum = dispatch.asum(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss;
    const nan_nrm2 = dispatch.nrm2(T, n, x.ptr, inc) orelse ops.nrm2(T, n, x.ptr, inc);
    try std.testing.expect(std.math.isNan(nan_dot.re));
    try std.testing.expect(std.math.isNan(nan_dot.im));
    try std.testing.expect(std.math.isNan(nan_asum));
    try std.testing.expect(std.math.isNan(nan_nrm2));
    x[0] = complexValue(T, std.math.nan(R), 0);
    try std.testing.expectEqual(@as(BlasInt, 1), dispatch.iamax(T, n, x.ptr, inc) orelse return error.UnexpectedStride2DispatchMiss);
}

test "ABI stride-two dispatcher gate misses have no side effects" {
    const below_gate = asBlasInt(dispatch.minimum_elements - 1);
    const target_n = asBlasInt(dispatch.minimum_elements);
    const inc1: BlasInt = 1;
    const inc2: BlasInt = 2;
    var x = [_]f64{ 1, 2, 3, 4 };
    var y = [_]f64{ 5, 6, 7, 8 };
    const original_x = x;
    const original_y = y;
    const param = [_]f64{ -1, 0.75, -0.25, 0.5, 0.625 };

    try std.testing.expect(!dispatch.scal(f64, below_gate, 2, &x, inc2));
    try std.testing.expect(!dispatch.swap(f64, below_gate, &x, inc2, &y, inc2));
    try std.testing.expect(!dispatch.axpy(f64, below_gate, 2, &x, inc2, &y, inc2));
    try std.testing.expect(!dispatch.axpby(f64, below_gate, 2, &x, inc2, 3, &y, inc2));
    try std.testing.expect(!dispatch.rot(f64, below_gate, &x, inc2, &y, inc2, 0.8, -0.6));
    try std.testing.expect(!dispatch.rotm(f64, below_gate, &x, inc2, &y, inc2, &param));
    try std.testing.expect(dispatch.dot(f64, below_gate, &x, inc2, &y, inc2, false) == null);
    try std.testing.expect(dispatch.dotF32AccF64(below_gate, @ptrCast(&x), inc2, @ptrCast(&y), inc2) == null);
    try std.testing.expect(dispatch.asum(f64, below_gate, &x, inc2) == null);
    try std.testing.expect(dispatch.nrm2(f64, below_gate, &x, inc2) == null);
    try std.testing.expect(dispatch.iamax(f64, below_gate, &x, inc2) == null);
    try std.testing.expect(!dispatch.scal(f64, target_n, 2, &x, inc1));
    try std.testing.expect(dispatch.dot(f64, target_n, &x, inc1, &y, inc2, false) == null);

    defer runtime.setMaxThreads(0);
    runtime.setMaxThreads(1);
    try std.testing.expect(!dispatch.scal(f64, target_n, 2, &x, inc2));
    try std.testing.expectEqualSlices(f64, &original_x, &x);
    try std.testing.expectEqualSlices(f64, &original_y, &y);
}

test "ABI stride-two dispatcher rejects overlapping binary updates" {
    const n = asBlasInt(dispatch.minimum_elements);
    const inc: BlasInt = 2;
    var values = [_]f64{ 1, 2, 3, 4 };
    const original = values;
    const param = [_]f64{ -1, 0.75, -0.25, 0.5, 0.625 };
    defer runtime.setMaxThreads(0);
    runtime.setMaxThreads(2);

    try std.testing.expect(!dispatch.swap(f64, n, &values, inc, &values, inc));
    try std.testing.expectEqualSlices(f64, &original, &values);
    try std.testing.expect(!dispatch.axpy(f64, n, 2, &values, inc, &values, inc));
    try std.testing.expectEqualSlices(f64, &original, &values);
    try std.testing.expect(!dispatch.axpby(f64, n, 2, &values, inc, 3, &values, inc));
    try std.testing.expectEqualSlices(f64, &original, &values);
    try std.testing.expect(!dispatch.rot(f64, n, &values, inc, &values, inc, 0.8, -0.6));
    try std.testing.expectEqualSlices(f64, &original, &values);
    try std.testing.expect(!dispatch.rotm(f64, n, &values, inc, &values, inc, &param));
    try std.testing.expectEqualSlices(f64, &original, &values);
}

test "stride-two real updates match single-thread fallback with odd exact storage" {
    if (!targetAvailable()) return;
    try checkRealUpdates(f32);
    try checkRealUpdates(f64);
}

test "stride-two complex updates match single-thread fallback with odd exact storage" {
    if (!targetAvailable()) return;
    try checkComplexUpdates(ComplexF32);
    try checkComplexUpdates(ComplexF64);
}

test "stride-two real reductions preserve ordered results and iamax semantics" {
    if (!targetAvailable()) return;
    try checkRealReductions(f32);
    try checkRealReductions(f64);
}

test "stride-two complex reductions preserve ordered results and iamax semantics" {
    if (!targetAvailable()) return;
    try checkComplexReductions(ComplexF32);
    try checkComplexReductions(ComplexF64);
}

test "stride-two subnormal real nrm2 stays finite" {
    if (!targetAvailable()) return;
    try checkSubnormalRealNrm2(f32);
    try checkSubnormalRealNrm2(f64);
}

test "stride-two subnormal complex nrm2 stays finite" {
    if (!targetAvailable()) return;
    try checkSubnormalComplexNrm2(ComplexF32);
    try checkSubnormalComplexNrm2(ComplexF64);
}

test "CSSCAL and ZDSCAL keep complex components independent" {
    if (!targetAvailable()) return;
    try checkComplexRealScale(ComplexF32);
    try checkComplexRealScale(ComplexF64);
}
