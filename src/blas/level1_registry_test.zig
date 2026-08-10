// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Forced-path Level 1 contract tests.
//!
//! The shared SIMD calls bypass architecture preference dispatch. Non-unit
//! stride calls bypass contiguous paths. The ABI-only stride-two family is
//! exercised separately by `vector_stride2_parallel_test.zig`.

const std = @import("std");
const builtin = @import("builtin");

const aarch_binary = @import("kernels/arch/aarch64/vector/binary.zig");
const aarch_features = @import("kernels/arch/aarch64/features.zig");
const aarch_unary = @import("kernels/arch/aarch64/vector/unary.zig");
const ops = @import("core/vector/operations.zig");
const catalog = @import("kernels/shared/vector/catalog.zig");
const fixed_simd = @import("kernels/shared/vector/fixed_simd.zig");
const isolated_abi = @import("kernels/isolated/x86_64_stride2_abi.zig");
const types = @import("types.zig");
const x86_binary = @import("kernels/arch/x86_64/vector/binary.zig");
const x86_unary = @import("kernels/arch/x86_64/vector/unary.zig");

const cfg: fixed_simd.Config = .{
    .lane_count = 8,
    .unroll_vectors = 2,
    .copy_lane_count = 16,
};

const fixed_width_configs = .{
    fixed_simd.Config{ .lane_count = 2, .unroll_vectors = 4, .copy_lane_count = 16 },
    fixed_simd.Config{ .lane_count = 4, .unroll_vectors = 4, .copy_lane_count = 32 },
    fixed_simd.Config{ .lane_count = 8, .unroll_vectors = 6, .copy_lane_count = 64 },
};

fn fillReal(x: []f64, y: []f64) void {
    for (x, y, 0..) |*xv, *yv, i| {
        xv.* = @as(f64, @floatFromInt(i)) * 0.25 - 2;
        yv.* = @as(f64, @floatFromInt(i)) * -0.125 + 1;
    }
}

fn expectStreamingCall(before: aarch_features.TestStreamingDepths, uses_za: bool) !void {
    const after = aarch_features.testStreamingEntries();
    try std.testing.expect(after.sm > before.sm);
    if (uses_za) {
        try std.testing.expect(after.za > before.za);
    } else {
        try std.testing.expectEqual(before.za, after.za);
    }
    try std.testing.expectEqual(aarch_features.TestStreamingDepths{ .sm = 0, .za = 0 }, aarch_features.testStreamingDepths());
    try std.testing.expectEqual(@as(u2, 0), aarch_features.streamingModeBits());
}

test "forced contiguous real kernel covers exact width, tail, and misalignment" {
    inline for (.{ @as(usize, 8), @as(usize, 13) }) |n| {
        var x_storage: [24]f64 align(64) = undefined;
        var y_storage: [24]f64 align(64) = undefined;
        fillReal(&x_storage, &y_storage);
        const before = y_storage;
        const x = x_storage[1..].ptr;
        const y = y_storage[1..].ptr;

        try std.testing.expect(fixed_simd.axpyUnitReal(f64, cfg, n, -0.75, x, y));
        for (0..n) |i| {
            const expected = @mulAdd(f64, -0.75, x[i], before[i + 1]);
            try std.testing.expectApproxEqAbs(expected, y[i], 1e-14);
        }
        try std.testing.expectEqualSlices(f64, before[n + 1 ..], y_storage[n + 1 ..]);
    }
}

test "native SME2 Level 1 catalog cells enter and balance declared state" {
    if (comptime builtin.cpu.arch != .aarch64 or !aarch_features.has_sme2) return;
    if (aarch_features.streamingVectorBytes() != 64) return;

    const allocator = std.testing.allocator;
    const n: usize = 64 * 1024;
    const nn: types.BlasInt = @intCast(n);
    const xf32 = try allocator.alloc(f32, n);
    defer allocator.free(xf32);
    const yf32 = try allocator.alloc(f32, n);
    defer allocator.free(yf32);
    const xf64 = try allocator.alloc(f64, n);
    defer allocator.free(xf64);
    const yf64 = try allocator.alloc(f64, n);
    defer allocator.free(yf64);
    const xc32 = try allocator.alloc(types.ComplexF32, n);
    defer allocator.free(xc32);
    const yc32 = try allocator.alloc(types.ComplexF32, n);
    defer allocator.free(yc32);
    const xc64 = try allocator.alloc(types.ComplexF64, n);
    defer allocator.free(xc64);
    const yc64 = try allocator.alloc(types.ComplexF64, n);
    defer allocator.free(yc64);

    for (xf32, yf32, xf64, yf64, xc32, yc32, xc64, yc64, 0..) |*a, *b, *c, *d, *e, *f, *g, *h, i| {
        const small: f32 = @floatFromInt(i % 17 + 1);
        a.* = small * 0.03125;
        b.* = small * -0.015625;
        c.* = @as(f64, small) * 0.03125;
        d.* = @as(f64, small) * -0.015625;
        e.* = .{ .re = a.*, .im = b.* };
        f.* = .{ .re = b.*, .im = a.* };
        g.* = .{ .re = c.*, .im = d.* };
        h.* = .{ .re = d.*, .im = c.* };
    }

    var calls: usize = 0;
    var before = aarch_features.testStreamingEntries();
    ops.copy(f32, 2048, xf32.ptr, 1, yf32.ptr, 1);
    try expectStreamingCall(before, false);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    ops.copy(f64, 1024, xf64.ptr, 1, yf64.ptr, 1);
    try expectStreamingCall(before, false);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    ops.copy(types.ComplexF32, 1024, xc32.ptr, 1, yc32.ptr, 1);
    try expectStreamingCall(before, false);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    ops.copy(types.ComplexF64, 512, xc64.ptr, 1, yc64.ptr, 1);
    try expectStreamingCall(before, false);
    calls += 1;

    inline for (.{
        .{ f32, @as(usize, 16 * 1024), xf32.ptr, yf32.ptr },
        .{ f64, @as(usize, 8 * 1024), xf64.ptr, yf64.ptr },
        .{ types.ComplexF32, @as(usize, 8 * 1024), xc32.ptr, yc32.ptr },
        .{ types.ComplexF64, @as(usize, 4 * 1024), xc64.ptr, yc64.ptr },
    }) |case| {
        before = aarch_features.testStreamingEntries();
        ops.swap(case[0], @intCast(case[1]), case[2], 1, case[3], 1);
        try expectStreamingCall(before, false);
        calls += 1;
    }

    before = aarch_features.testStreamingEntries();
    ops.scal(f32, nn, 0.75, xf32.ptr, 1);
    try expectStreamingCall(before, true);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    ops.rscal(types.ComplexF32, nn, 0.75, xc32.ptr, 1);
    try expectStreamingCall(before, true);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    ops.axpy(f32, nn, -0.25, xf32.ptr, 1, yf32.ptr, 1);
    try expectStreamingCall(before, true);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    ops.axpby(f32, nn, 0.625, xf32.ptr, 1, -0.375, yf32.ptr, 1);
    try expectStreamingCall(before, true);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    _ = ops.dot(f32, nn, xf32.ptr, 1, yf32.ptr, 1, false);
    try expectStreamingCall(before, true);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    _ = ops.asum(f32, nn, xf32.ptr, 1);
    try expectStreamingCall(before, true);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    _ = ops.asum(types.ComplexF32, nn, xc32.ptr, 1);
    try expectStreamingCall(before, true);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    ops.rot(f32, nn, xf32.ptr, 1, yf32.ptr, 1, 0.8, 0.6);
    try expectStreamingCall(before, true);
    calls += 1;
    before = aarch_features.testStreamingEntries();
    const param = [_]f32{ -1, 0.8, -0.6, 0.6, 0.8 };
    ops.rotm(f32, nn, xf32.ptr, 1, yf32.ptr, 1, &param);
    try expectStreamingCall(before, true);
    calls += 1;

    if (comptime aarch_features.has_sme_f64f64) {
        before = aarch_features.testStreamingEntries();
        ops.scal(f64, nn, 0.75, xf64.ptr, 1);
        try expectStreamingCall(before, true);
        calls += 1;
        before = aarch_features.testStreamingEntries();
        ops.rscal(types.ComplexF64, nn, 0.75, xc64.ptr, 1);
        try expectStreamingCall(before, true);
        calls += 1;
        before = aarch_features.testStreamingEntries();
        ops.axpy(f64, nn, -0.25, xf64.ptr, 1, yf64.ptr, 1);
        try expectStreamingCall(before, true);
        calls += 1;
        before = aarch_features.testStreamingEntries();
        _ = ops.dot(f64, nn, xf64.ptr, 1, yf64.ptr, 1, false);
        try expectStreamingCall(before, true);
        calls += 1;
        before = aarch_features.testStreamingEntries();
        _ = ops.asum(f64, nn, xf64.ptr, 1);
        try expectStreamingCall(before, true);
        calls += 1;
        before = aarch_features.testStreamingEntries();
        _ = ops.asum(types.ComplexF64, nn, xc64.ptr, 1);
        try expectStreamingCall(before, true);
        calls += 1;
    }

    try std.testing.expectEqual(@as(usize, if (aarch_features.has_sme_f64f64) 23 else 17), calls);
}

test "forced contiguous kernel rejects short calls without side effects" {
    var x: [16]f64 = undefined;
    var y: [16]f64 = undefined;
    fillReal(&x, &y);
    const before = y;

    try std.testing.expect(!fixed_simd.axpyUnitReal(f64, cfg, 0, 2, &x, &y));
    try std.testing.expect(!fixed_simd.axpyUnitReal(f64, cfg, cfg.lane_count - 1, 2, &x, &y));
    try std.testing.expectEqualSlices(f64, &before, &y);

    ops.axpy(f64, 0, 2, &x, 1, &y, 1);
    try std.testing.expectEqualSlices(f64, &before, &y);
}

fn complexMul(a: types.ComplexF64, b: types.ComplexF64) types.ComplexF64 {
    return .{
        .re = a.re * b.re - a.im * b.im,
        .im = a.re * b.im + a.im * b.re,
    };
}

fn complexAdd(a: types.ComplexF64, b: types.ComplexF64) types.ComplexF64 {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}

fn expectComplex(expected: types.ComplexF64, actual: types.ComplexF64) !void {
    try std.testing.expectApproxEqAbs(expected.re, actual.re, 2e-13);
    try std.testing.expectApproxEqAbs(expected.im, actual.im, 2e-13);
}

test "forced contiguous complex update and reduction cover odd tails" {
    const n: usize = 7;
    const alpha: types.ComplexF64 = .{ .re = -0.75, .im = 0.5 };
    const beta: types.ComplexF64 = .{ .re = 0.375, .im = -0.25 };
    var x_storage: [12]types.ComplexF64 align(64) = undefined;
    var y_storage: [12]types.ComplexF64 align(64) = undefined;
    for (&x_storage, &y_storage, 0..) |*xv, *yv, i| {
        const value: f64 = @floatFromInt(i);
        xv.* = .{ .re = value * 0.25 - 1, .im = value * -0.125 + 0.5 };
        yv.* = .{ .re = value * -0.2 + 0.75, .im = value * 0.1 - 0.25 };
    }
    const before = y_storage;
    const x = x_storage[1..].ptr;
    const y = y_storage[1..].ptr;

    try std.testing.expect(fixed_simd.axpbyUnitComplex(types.ComplexF64, cfg, n, alpha, x, beta, y));
    for (0..n) |i| {
        const expected = complexAdd(complexMul(alpha, x[i]), complexMul(beta, before[i + 1]));
        try expectComplex(expected, y[i]);
    }

    var expected_dot: types.ComplexF64 = .{ .re = 0, .im = 0 };
    for (0..n) |i| expected_dot = complexAdd(expected_dot, complexMul(x[i], y[i]));
    const actual_dot = fixed_simd.dotUnitComplex(types.ComplexF64, cfg, n, x, y, false).?;
    try expectComplex(expected_dot, actual_dot);
}

test "forced reductions preserve IAMAX first-index ties" {
    const n: usize = 13;
    var storage: [20]f64 align(64) = [_]f64{0} ** 20;
    const x = storage[1..].ptr;
    for (0..n) |i| x[i] = @as(f64, @floatFromInt(i)) * 0.25 - 1;
    x[2] = 9;
    x[10] = -9;

    try std.testing.expectEqual(@as(types.BlasInt, 3), fixed_simd.iamaxUnitReal(f64, cfg, n, x).?);
    try std.testing.expectEqual(@as(types.BlasInt, 0), fixed_simd.iamaxUnitReal(f64, cfg, 0, x).?);
    try std.testing.expect(fixed_simd.iamaxUnitReal(f64, cfg, cfg.lane_count - 1, x) == null);

    var expected_asum: f64 = 0;
    var expected_squares: f64 = 0;
    for (0..n) |i| {
        expected_asum += @abs(x[i]);
        expected_squares = @mulAdd(f64, x[i], x[i], expected_squares);
    }
    try std.testing.expectApproxEqAbs(expected_asum, fixed_simd.asumUnitReal(f64, cfg, n, x).?, 1e-13);
    try std.testing.expectApproxEqAbs(@sqrt(expected_squares), fixed_simd.nrm2UnitReal(f64, cfg, n, x).?, 1e-13);
}

test "forced mixed precision dot accumulates in f64 across tails" {
    const n: usize = 13;
    var x_storage: [20]f32 align(64) = undefined;
    var y_storage: [20]f32 align(64) = undefined;
    for (&x_storage, &y_storage, 0..) |*xv, *yv, i| {
        const value: f32 = @floatFromInt(i);
        xv.* = value * 0.375 - 2;
        yv.* = value * -0.1875 + 1;
    }
    const x = x_storage[1..].ptr;
    const y = y_storage[1..].ptr;
    var expected: f64 = 0;
    for (0..n) |i| expected = @mulAdd(f64, @as(f64, x[i]), @as(f64, y[i]), expected);

    try std.testing.expectApproxEqAbs(expected, fixed_simd.dotF32AccF64Unit(cfg, n, x, y).?, 1e-13);
    try std.testing.expect(fixed_simd.dotF32AccF64Unit(cfg, cfg.lane_count - 1, x, y) == null);
}

test "forced complex IAMAX uses abs1 and preserves first-index ties" {
    const n: usize = 13;
    var storage: [20]types.ComplexF64 align(64) = [_]types.ComplexF64{.{ .re = 0, .im = 0 }} ** 20;
    const x = storage[1..].ptr;
    for (0..n) |i| {
        const value: f64 = @floatFromInt(i);
        x[i] = .{ .re = value * 0.25 - 1, .im = value * -0.125 + 0.5 };
    }
    x[2] = .{ .re = 4, .im = -5 };
    x[10] = .{ .re = -8, .im = 1 };

    try std.testing.expectEqual(@as(types.BlasInt, 3), fixed_simd.iamaxUnitComplex(types.ComplexF64, cfg, n, x).?);
    try std.testing.expectEqual(@as(types.BlasInt, 0), fixed_simd.iamaxUnitComplex(types.ComplexF64, cfg, 0, x).?);
    try std.testing.expect(fixed_simd.iamaxUnitComplex(types.ComplexF64, cfg, 3, x) == null);
}

test "forced ROTM covers all flags, tails, and the identity flag" {
    inline for (.{ @as(f64, -1), @as(f64, 0), @as(f64, 1) }) |flag| {
        const n: usize = 13;
        var x_storage: [20]f64 align(64) = undefined;
        var y_storage: [20]f64 align(64) = undefined;
        fillReal(&x_storage, &y_storage);
        const before_x = x_storage;
        const before_y = y_storage;
        const x = x_storage[1..].ptr;
        const y = y_storage[1..].ptr;
        const h11: f64 = 0.75;
        const h21: f64 = -0.5;
        const h12: f64 = 0.25;
        const h22: f64 = 1.25;

        try std.testing.expect(fixed_simd.rotmUnitReal(f64, cfg, n, x, y, flag, h11, h21, h12, h22));
        for (0..n) |i| {
            const w = before_x[i + 1];
            const z = before_y[i + 1];
            const expected_x = if (flag < 0)
                @mulAdd(f64, w, h11, z * h12)
            else if (flag == 0)
                @mulAdd(f64, z, h12, w)
            else
                @mulAdd(f64, w, h11, z);
            const expected_y = if (flag < 0)
                @mulAdd(f64, w, h21, z * h22)
            else if (flag == 0)
                @mulAdd(f64, w, h21, z)
            else
                z * h22 - w;
            try std.testing.expectApproxEqAbs(expected_x, x[i], 1e-14);
            try std.testing.expectApproxEqAbs(expected_y, y[i], 1e-14);
        }
        try std.testing.expectEqualSlices(f64, before_x[n + 1 ..], x_storage[n + 1 ..]);
        try std.testing.expectEqualSlices(f64, before_y[n + 1 ..], y_storage[n + 1 ..]);
    }

    var x: [8]f64 = [_]f64{1} ** 8;
    var y: [8]f64 = [_]f64{2} ** 8;
    const before_x = x;
    const before_y = y;
    try std.testing.expect(fixed_simd.rotmUnitReal(f64, cfg, x.len, &x, &y, -2, 0, 0, 0, 0));
    try std.testing.expectEqualSlices(f64, &before_x, &x);
    try std.testing.expectEqualSlices(f64, &before_y, &y);
}

test "forced SVE f32 counterparts cover empty exact-VL and predicated tails" {
    if (comptime builtin.cpu.arch != .aarch64 or !aarch_features.has_sve) return;

    const lane_count = aarch_features.sveVectorBytes() / @sizeOf(f32);
    const cases = [_]usize{ 0, 1, lane_count, lane_count + 3, lane_count * 4 + 1 };
    const alpha: types.ComplexF32 = .{ .re = -0.75, .im = 0.375 };

    for (cases) |n| {
        var real_x_storage: [300]f32 align(64) = undefined;
        var real_y_storage: [300]f32 align(64) = undefined;
        var complex_x_storage: [300]types.ComplexF32 align(64) = undefined;
        var complex_y_storage: [300]types.ComplexF32 align(64) = undefined;
        for (&real_x_storage, &real_y_storage, &complex_x_storage, &complex_y_storage, 0..) |*rx, *ry, *cx, *cy, i| {
            const value: f32 = @floatFromInt(i);
            rx.* = value * 0.03125 - 1.5;
            ry.* = value * -0.015625 + 0.75;
            cx.* = .{ .re = value * 0.0234375 - 0.8, .im = value * -0.01171875 + 0.4 };
            cy.* = .{ .re = value * -0.01953125 + 0.6, .im = value * 0.0078125 - 0.2 };
        }

        const real_x = real_x_storage[1..].ptr;
        const real_y = real_y_storage[1..].ptr;
        const complex_x = complex_x_storage[1..].ptr;
        const complex_y = complex_y_storage[1..].ptr;
        const real_before = real_x_storage;
        const complex_before = complex_y_storage;

        var expected_dot: f32 = 0;
        var expected_mixed_dot: f64 = 0;
        var expected_dotu: types.ComplexF32 = .{ .re = 0, .im = 0 };
        var expected_dotc: types.ComplexF32 = .{ .re = 0, .im = 0 };
        for (0..n) |i| {
            // Keep the scalar oracle independent of the candidate's fused
            // instruction sequence and of target-libc `fmaf` lowering.
            expected_dot += real_x[i] * real_y[i];
            expected_mixed_dot += @as(f64, real_x[i]) * @as(f64, real_y[i]);
            expected_dotu.re += complex_x[i].re * complex_y[i].re - complex_x[i].im * complex_y[i].im;
            expected_dotu.im += complex_x[i].re * complex_y[i].im + complex_x[i].im * complex_y[i].re;
            expected_dotc.re += complex_x[i].re * complex_y[i].re + complex_x[i].im * complex_y[i].im;
            expected_dotc.im += complex_x[i].re * complex_y[i].im - complex_x[i].im * complex_y[i].re;
        }

        const actual_dot = aarch_binary.sveDotF32Candidate(n, real_x, real_y).?;
        std.testing.expectApproxEqRel(expected_dot, actual_dot, 2e-5) catch |err| {
            std.debug.print("SVE f32 dot mismatch at n={d}: expected={d}, actual={d}\n", .{ n, expected_dot, actual_dot });
            return err;
        };
        try std.testing.expectApproxEqRel(expected_mixed_dot, aarch_binary.sveDotF32AccF64Candidate(n, real_x, real_y).?, 1e-13);
        const actual_dotu = aarch_binary.sveDotComplexF32Candidate(n, complex_x, complex_y, false).?;
        const actual_dotc = aarch_binary.sveDotComplexF32Candidate(n, complex_x, complex_y, true).?;
        try std.testing.expectApproxEqAbs(expected_dotu.re, actual_dotu.re, 2e-4);
        try std.testing.expectApproxEqAbs(expected_dotu.im, actual_dotu.im, 2e-4);
        try std.testing.expectApproxEqAbs(expected_dotc.re, actual_dotc.re, 2e-4);
        try std.testing.expectApproxEqAbs(expected_dotc.im, actual_dotc.im, 2e-4);

        try std.testing.expect(aarch_binary.sveAxpyComplexF32Candidate(n, alpha, complex_x, complex_y));
        for (0..n) |i| {
            const expected_re = complex_before[i + 1].re + alpha.re * complex_x[i].re - alpha.im * complex_x[i].im;
            const expected_im = complex_before[i + 1].im + alpha.re * complex_x[i].im + alpha.im * complex_x[i].re;
            try std.testing.expectApproxEqAbs(expected_re, complex_y[i].re, 2e-6);
            try std.testing.expectApproxEqAbs(expected_im, complex_y[i].im, 2e-6);
        }
        try std.testing.expectEqualSlices(types.ComplexF32, complex_before[n + 1 ..], complex_y_storage[n + 1 ..]);

        try std.testing.expect(aarch_unary.sveScalF32Candidate(n, -0.625, real_x));
        for (0..n) |i| try std.testing.expectApproxEqAbs(real_before[i + 1] * -0.625, real_x[i], 1e-6);
        try std.testing.expectEqualSlices(f32, real_before[n + 1 ..], real_x_storage[n + 1 ..]);
    }
}

test "forced SVE extended update reduction copy and tie families use predicated tails" {
    if (comptime builtin.cpu.arch != .aarch64 or !aarch_features.has_sve) return;

    const n: usize = 19;
    var x: [32]f64 align(64) = undefined;
    var y: [32]f64 align(64) = undefined;
    for (&x, &y, 0..) |*xv, *yv, i| {
        const value: f64 = @floatFromInt(i);
        xv.* = value * 0.25 - 2;
        yv.* = value * -0.125 + 1;
    }
    const original_x = x;
    const original_y = y;

    try std.testing.expect(aarch_binary.sveCopyBytesCandidate(n * @sizeOf(f64), @ptrCast(&x), @ptrCast(&y)));
    try std.testing.expectEqualSlices(f64, x[0..n], y[0..n]);
    try std.testing.expectEqualSlices(f64, original_y[n..], y[n..]);

    y = original_y;
    try std.testing.expect(aarch_binary.sveSwapBytesCandidate(n * @sizeOf(f64), @ptrCast(&x), @ptrCast(&y)));
    try std.testing.expectEqualSlices(f64, original_y[0..n], x[0..n]);
    try std.testing.expectEqualSlices(f64, original_x[0..n], y[0..n]);
    x = original_x;
    y = original_y;

    try std.testing.expect(aarch_binary.sveAxpyRealCandidate(f64, n, -0.75, &x, &y));
    for (0..n) |i| try std.testing.expectApproxEqAbs(original_y[i] - 0.75 * original_x[i], y[i], 1e-13);
    y = original_y;
    try std.testing.expect(aarch_binary.sveAxpbyRealCandidate(f64, n, -0.75, &x, 0.375, &y));
    for (0..n) |i| try std.testing.expectApproxEqAbs(-0.75 * original_x[i] + 0.375 * original_y[i], y[i], 1e-13);

    x = original_x;
    y = original_y;
    try std.testing.expect(aarch_binary.sveRotUnitRealCandidate(f64, n, &x, &y, 0.75, -0.5));
    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(0.75 * original_x[i] - 0.5 * original_y[i], x[i], 1e-13);
        try std.testing.expectApproxEqAbs(0.5 * original_x[i] + 0.75 * original_y[i], y[i], 1e-13);
    }
    x = original_x;
    y = original_y;
    try std.testing.expect(aarch_binary.sveRotmUnitRealCandidate(f64, n, &x, &y, 0, 0, -0.5, 0.25, 0));
    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(original_x[i] + 0.25 * original_y[i], x[i], 1e-13);
        try std.testing.expectApproxEqAbs(-0.5 * original_x[i] + original_y[i], y[i], 1e-13);
    }

    var norm_expected: f64 = 0;
    for (original_x[0..n]) |value| norm_expected += value * value;
    try std.testing.expectApproxEqRel(@sqrt(norm_expected), aarch_unary.sveNrm2UnitRealCandidate(f64, n, &original_x).?, 2e-13);
    x = original_x;
    x[3] = -11;
    x[17] = 11;
    try std.testing.expectEqual(@as(types.BlasInt, 4), aarch_unary.sveIamaxUnitRealCandidate(f64, n, &x).?);

    const complex_n: usize = 11;
    const alpha: types.ComplexF64 = .{ .re = -0.75, .im = 0.5 };
    const beta: types.ComplexF64 = .{ .re = 0.25, .im = -0.125 };
    var cx: [16]types.ComplexF64 align(64) = undefined;
    var cy: [16]types.ComplexF64 align(64) = undefined;
    for (&cx, &cy, 0..) |*xv, *yv, i| {
        const value: f64 = @floatFromInt(i);
        xv.* = .{ .re = value * 0.25 - 1, .im = value * -0.125 + 0.5 };
        yv.* = .{ .re = value * -0.2 + 0.75, .im = value * 0.1 - 0.25 };
    }
    const original_cx = cx;
    const original_cy = cy;
    try std.testing.expect(aarch_unary.sveScalComplexCandidate(types.ComplexF64, complex_n, alpha, &cx));
    for (0..complex_n) |i| try expectComplex(complexMul(alpha, original_cx[i]), cx[i]);

    cx = original_cx;
    cy = original_cy;
    try std.testing.expect(aarch_binary.sveAxpbyComplexCandidate(types.ComplexF64, complex_n, alpha, &cx, beta, &cy));
    for (0..complex_n) |i| {
        try expectComplex(complexAdd(complexMul(alpha, original_cx[i]), complexMul(beta, original_cy[i])), cy[i]);
    }
    cx = original_cx;
    cx[2] = .{ .re = 6, .im = -4 };
    cx[9] = .{ .re = -9, .im = 1 };
    try std.testing.expectEqual(@as(types.BlasInt, 3), aarch_unary.sveIamaxUnitComplexCandidate(types.ComplexF64, complex_n, &cx).?);

    // Instantiate the f32 forms independently; their arithmetic is covered by
    // the existing f32 SVE test and the same formulas above.
    var f32_x: [1]f32 = .{1};
    var f32_y: [1]f32 = .{2};
    var c32_x: [1]types.ComplexF32 = .{.{ .re = 1, .im = -1 }};
    var c32_y: [1]types.ComplexF32 = .{.{ .re = 2, .im = 1 }};
    try std.testing.expect(aarch_binary.sveAxpyRealCandidate(f32, 0, 1, &f32_x, &f32_y));
    try std.testing.expect(aarch_binary.sveAxpbyRealCandidate(f32, 0, 1, &f32_x, 1, &f32_y));
    try std.testing.expect(aarch_binary.sveRotUnitRealCandidate(f32, 0, &f32_x, &f32_y, 1, 0));
    try std.testing.expect(aarch_binary.sveRotmUnitRealCandidate(f32, 0, &f32_x, &f32_y, -2, 0, 0, 0, 0));
    try std.testing.expect(aarch_unary.sveScalComplexCandidate(types.ComplexF32, 0, .{ .re = 1, .im = 0 }, &c32_x));
    try std.testing.expect(aarch_binary.sveAxpbyComplexCandidate(types.ComplexF32, 0, .{ .re = 1, .im = 0 }, &c32_x, .{ .re = 1, .im = 0 }, &c32_y));
    try std.testing.expectEqual(@as(f32, 0), aarch_unary.sveNrm2UnitRealCandidate(f32, 0, &f32_x).?);
    try std.testing.expectEqual(@as(types.BlasInt, 0), aarch_unary.sveIamaxUnitRealCandidate(f32, 0, &f32_x).?);
    try std.testing.expectEqual(@as(types.BlasInt, 0), aarch_unary.sveIamaxUnitComplexCandidate(types.ComplexF32, 0, &c32_x).?);
}

test "fixed candidates instantiate 128 256 and 512 bit accumulator geometries" {
    inline for (fixed_width_configs) |candidate_cfg| {
        const n: usize = 21;
        var dot_x: [24]f32 align(64) = undefined;
        var dot_y: [24]f32 align(64) = undefined;
        var expected_dot: f64 = 0;
        for (&dot_x, &dot_y, 0..) |*xv, *yv, i| {
            const value: f32 = @floatFromInt(i);
            xv.* = value * 0.125 - 1;
            yv.* = value * -0.0625 + 0.75;
        }
        for (0..n) |i| expected_dot = @mulAdd(f64, @as(f64, dot_x[i]), @as(f64, dot_y[i]), expected_dot);
        try std.testing.expectApproxEqAbs(expected_dot, fixed_simd.dotF32AccF64Unit(candidate_cfg, n, &dot_x, &dot_y).?, 1e-13);

        var complex_x: [24]types.ComplexF32 align(64) = [_]types.ComplexF32{.{ .re = 0, .im = 0 }} ** 24;
        for (0..n) |i| {
            const value: f32 = @floatFromInt(i);
            complex_x[i] = .{ .re = value * 0.25 - 1, .im = value * -0.125 + 0.5 };
        }
        complex_x[3] = .{ .re = 6, .im = -4 };
        complex_x[17] = .{ .re = -9, .im = 1 };
        try std.testing.expectEqual(@as(types.BlasInt, 4), fixed_simd.iamaxUnitComplex(types.ComplexF32, candidate_cfg, n, &complex_x).?);

        inline for (.{ @as(f32, -1), @as(f32, 0), @as(f32, 1) }) |flag| {
            var rotm_x: [24]f32 align(64) = undefined;
            var rotm_y: [24]f32 align(64) = undefined;
            for (&rotm_x, &rotm_y, 0..) |*xv, *yv, i| {
                const value: f32 = @floatFromInt(i);
                xv.* = value * 0.25 - 2;
                yv.* = value * -0.125 + 1;
            }
            const before_x = rotm_x;
            const before_y = rotm_y;
            const h11: f32 = 0.75;
            const h21: f32 = -0.5;
            const h12: f32 = 0.25;
            const h22: f32 = 1.25;
            try std.testing.expect(fixed_simd.rotmUnitReal(f32, candidate_cfg, n, &rotm_x, &rotm_y, flag, h11, h21, h12, h22));
            for (0..n) |i| {
                const w = before_x[i];
                const z = before_y[i];
                const expected_x = if (flag < 0)
                    @mulAdd(f32, w, h11, z * h12)
                else if (flag == 0)
                    @mulAdd(f32, z, h12, w)
                else
                    @mulAdd(f32, w, h11, z);
                const expected_y = if (flag < 0)
                    @mulAdd(f32, w, h21, z * h22)
                else if (flag == 0)
                    @mulAdd(f32, w, h21, z)
                else
                    z * h22 - w;
                try std.testing.expectApproxEqAbs(expected_x, rotm_x[i], 1e-6);
                try std.testing.expectApproxEqAbs(expected_y, rotm_y[i], 1e-6);
            }
        }
    }
}

test "native architecture candidate entrypoints preserve fixed kernel semantics" {
    if (comptime builtin.cpu.arch != .aarch64 and builtin.cpu.arch != .x86_64) return;
    const n: usize = 21;
    var dot_x: [24]f32 align(64) = undefined;
    var dot_y: [24]f32 align(64) = undefined;
    var expected_dot: f64 = 0;
    for (&dot_x, &dot_y, 0..) |*xv, *yv, i| {
        const value: f32 = @floatFromInt(i);
        xv.* = value * 0.125 - 1;
        yv.* = value * -0.0625 + 0.75;
    }
    for (0..n) |i| expected_dot = @mulAdd(f64, @as(f64, dot_x[i]), @as(f64, dot_y[i]), expected_dot);

    var complex_x: [24]types.ComplexF32 align(64) = [_]types.ComplexF32{.{ .re = 0, .im = 0 }} ** 24;
    for (0..n) |i| {
        const value: f32 = @floatFromInt(i);
        complex_x[i] = .{ .re = value * 0.25 - 1, .im = value * -0.125 + 0.5 };
    }
    complex_x[3] = .{ .re = 6, .im = -4 };
    complex_x[17] = .{ .re = -9, .im = 1 };

    var rotm_x: [24]f32 align(64) = undefined;
    var rotm_y: [24]f32 align(64) = undefined;
    for (&rotm_x, &rotm_y, 0..) |*xv, *yv, i| {
        const value: f32 = @floatFromInt(i);
        xv.* = value * 0.25 - 2;
        yv.* = value * -0.125 + 1;
    }

    if (comptime builtin.cpu.arch == .aarch64) {
        try std.testing.expectApproxEqAbs(expected_dot, aarch_binary.fixedDotF32AccF64UnitCandidate(n, &dot_x, &dot_y).?, 1e-13);
        try std.testing.expectEqual(@as(types.BlasInt, 4), aarch_unary.fixedIamaxUnitComplexCandidate(types.ComplexF32, n, &complex_x).?);
        try std.testing.expect(aarch_binary.fixedRotmUnitRealCandidate(f32, n, &rotm_x, &rotm_y, 0, 0.75, -0.5, 0.25, 1.25));
    } else {
        try std.testing.expectApproxEqAbs(expected_dot, x86_binary.fixedDotF32AccF64UnitCandidate(n, &dot_x, &dot_y).?, 1e-13);
        try std.testing.expectEqual(@as(types.BlasInt, 4), x86_unary.fixedIamaxUnitComplexCandidate(types.ComplexF32, n, &complex_x).?);
        try std.testing.expect(x86_binary.fixedRotmUnitRealCandidate(f32, n, &rotm_x, &rotm_y, 0, 0.75, -0.5, 0.25, 1.25));
    }
}

test "native production complex IAMAX selection respects measured profile" {
    var x: [257]types.ComplexF32 align(64) = [_]types.ComplexF32{.{ .re = 1, .im = -1 }} ** 257;
    x[17] = .{ .re = 8, .im = -7 };
    if (comptime builtin.cpu.arch == .aarch64) {
        try std.testing.expectEqual(@as(types.BlasInt, 18), aarch_unary.iamaxUnitComplex(types.ComplexF32, 256, &x).?);
        try std.testing.expect(aarch_unary.iamaxUnitComplex(types.ComplexF32, 257, &x) == null);
    } else if (comptime builtin.cpu.arch == .x86_64) {
        try std.testing.expect(x86_unary.iamaxUnitComplex(types.ComplexF32, 256, &x) == null);
    }
}

fn checkScalarStride(comptime increment: types.BlasInt) !void {
    const indices = if (increment > 0)
        [_]usize{ 0, 3, 6, 9, 12 }
    else
        [_]usize{ 12, 9, 6, 3, 0 };
    var x: [13]f64 = undefined;
    var y: [13]f64 = undefined;
    for (&x, &y, 0..) |*xv, *yv, i| {
        xv.* = @floatFromInt(i + 1);
        yv.* = -@as(f64, @floatFromInt(i + 1));
    }
    const before = y;

    ops.axpy(f64, 5, 2, &x, increment, &y, increment);
    for (0..y.len) |i| {
        var selected = false;
        for (indices) |index| selected = selected or i == index;
        const expected = if (selected) before[i] + 2 * x[i] else before[i];
        try std.testing.expectEqual(expected, y[i]);
    }
}

test "terminal scalar fallback covers positive and negative non-unit strides" {
    try checkScalarStride(3);
    try checkScalarStride(-3);

    // The registry itself is part of this test root, so compile-time validation
    // proves every nonterminal descriptor resolves to a terminal fallback.
    try std.testing.expect(catalog.registry.len > 0);
}

test "isolated stride-two request ABI has a stable fixed layout" {
    if (@sizeOf(usize) != 8) return;
    try std.testing.expectEqual(@as(usize, 1), @sizeOf(isolated_abi.Operation));
    try std.testing.expectEqual(@as(usize, 1), @sizeOf(isolated_abi.Scalar));
    try std.testing.expectEqual(@as(usize, 120), @sizeOf(isolated_abi.Request));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(isolated_abi.Request));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(isolated_abi.Request, "n"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(isolated_abi.Request, "args"));
    try std.testing.expectEqual(@as(usize, 96), @offsetOf(isolated_abi.Request, "result"));
    try std.testing.expectEqual(@as(usize, 112), @offsetOf(isolated_abi.Request, "result_index"));
}
