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

test "prefetched reductions preserve ordinary values across cache-policy boundaries" {
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    const inc: i32 = 1;
    inline for (.{ f32, f64 }) |T| {
        const norm = if (T == f32) fortran.snrm2_ else fortran.dnrm2_;
        const abs_sum = if (T == f32) fortran.sasum_ else fortran.dasum_;
        for ([_]usize{ 256 * 1024, 2 * 1024 * 1024, 32 * 1024 * 1024 }) |bytes| {
            for ([_]isize{ -1, 0, 1 }) |offset| {
                const count: usize = @intCast(@as(isize, @intCast(bytes / @sizeOf(T))) + offset);
                const x = try std.testing.allocator.alloc(T, count);
                defer std.testing.allocator.free(x);
                @memset(x, 1);
                const n: i32 = @intCast(count);
                const expected = @sqrt(@as(f64, @floatFromInt(count)));
                try std.testing.expectApproxEqRel(expected, @as(f64, norm(&n, x.ptr, &inc)), 1e-6);
                try std.testing.expectEqual(@as(T, @floatFromInt(count)), abs_sum(&n, x.ptr, &inc));
            }
        }
    }
}

test "streaming copy preserves unaligned edges and declines unsupported tiers" {
    const binary = @import("kernels/dispatch/vector_binary.zig");
    var source: [8320]u8 align(64) = undefined;
    var target: [8320]u8 align(64) = undefined;
    for (&source, 0..) |*v, i| v.* = @intCast(i % 251);
    for ([_]usize{ 0, 1, 7, 31, 63 }) |offset| {
        for ([_]usize{ 256, 257, 511, 512, 513, 8193 }) |n| {
            @memset(&target, 255);
            if (binary.streamCopyBytes(n, source[3..].ptr, target[offset..].ptr)) {
                try std.testing.expectEqualSlices(u8, source[3 .. 3 + n], target[offset .. offset + n]);
                for (target[0..offset]) |v| try std.testing.expectEqual(@as(u8, 255), v);
                for (target[offset + n ..]) |v| try std.testing.expectEqual(@as(u8, 255), v);
            } else {
                for (target) |v| try std.testing.expectEqual(@as(u8, 255), v);
            }
        }
    }
}

test "large public copy orders streaming workers and retains overlap semantics" {
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    const count: usize = 8 * 1024 * 1024 + 17;
    const x = try std.testing.allocator.alloc(f32, count + 32);
    defer std.testing.allocator.free(x);
    const y = try std.testing.allocator.alloc(f32, count + 32);
    defer std.testing.allocator.free(y);
    for (x, 0..) |*v, i| v.* = @floatFromInt(i % 251);
    const n: i32 = @intCast(count);
    const inc: i32 = 1;
    for ([_]usize{ 1, 4 }) |threads| {
        runtime.setMaxThreads(threads);
        @memset(y, -1);
        fortran.scopy_(&n, x[1..].ptr, &inc, y[3..].ptr, &inc);
        try std.testing.expectEqualSlices(f32, x[1 .. count + 1], y[3 .. count + 3]);
        for (y[0..3]) |v| try std.testing.expectEqual(@as(f32, -1), v);
        for (y[count + 3 ..]) |v| try std.testing.expectEqual(@as(f32, -1), v);
    }
    fortran.scopy_(&n, x.ptr, &inc, x[1..].ptr, &inc);
    for (x[1 .. count + 1], 0..) |v, i| try std.testing.expectEqual(@as(f32, @floatFromInt(i % 251)), v);
}

test "complex DOT fused products preserve conjugation across tails and tasks" {
    defer fortran.zynum_blas_shutdown();
    const inc: i32 = 1;
    inline for (.{ api.ComplexF32, api.ComplexF64 }) |T| {
        const R = if (T == api.ComplexF32) f32 else f64;
        const dotu = if (T == api.ComplexF32) fortran.cdotu_sub_ else fortran.zdotu_sub_;
        const dotc = if (T == api.ComplexF32) fortran.cdotc_sub_ else fortran.zdotc_sub_;
        for ([_]usize{ 31, 32, 33, 1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192, 8193, 32768, 1048576 }) |n| {
            const x = try std.testing.allocator.alloc(T, n);
            defer std.testing.allocator.free(x);
            const y = try std.testing.allocator.alloc(T, n);
            defer std.testing.allocator.free(y);
            var ur: f64 = 0;
            var ui: f64 = 0;
            var cr: f64 = 0;
            var ci: f64 = 0;
            for (0..n) |i| {
                x[i] = .{ .re = @as(R, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 16, .im = @as(R, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 8 };
                y[i] = .{ .re = @as(R, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 4, .im = @as(R, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 16 };
                const xr: f64 = x[i].re;
                const xi: f64 = x[i].im;
                const yr: f64 = y[i].re;
                const yi: f64 = y[i].im;
                ur += xr * yr - xi * yi;
                ui += xr * yi + xi * yr;
                cr += xr * yr + xi * yi;
                ci += xr * yi - xi * yr;
            }
            const ni: i32 = @intCast(n);
            var u: T = undefined;
            var c: T = undefined;
            dotu(&ni, x.ptr, &inc, y.ptr, &inc, &u);
            dotc(&ni, x.ptr, &inc, y.ptr, &inc, &c);
            try std.testing.expectApproxEqAbs(@as(R, @floatCast(ur)), u.re, 0.0001);
            try std.testing.expectApproxEqAbs(@as(R, @floatCast(ui)), u.im, 0.0001);
            try std.testing.expectApproxEqAbs(@as(R, @floatCast(cr)), c.re, 0.0001);
            try std.testing.expectApproxEqAbs(@as(R, @floatCast(ci)), c.im, 0.0001);
        }
    }
}

test "IAMAX preserves first ties for monotone and repeated data across batching boundaries" {
    defer fortran.zynum_blas_shutdown();
    const inc: i32 = 1;
    inline for (scalars) |T| {
        const R = if (T == f32 or T == api.ComplexF32) f32 else f64;
        const complex = T == api.ComplexF32 or T == api.ComplexF64;
        const iamax = if (T == f32) fortran.isamax_ else if (T == f64) fortran.idamax_ else if (T == api.ComplexF32) fortran.icamax_ else fortran.izamax_;
        for ([_]usize{ 1, 7, 8, 15, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 4097, 1048576 }) |n| {
            const x = try std.testing.allocator.alloc(T, n + 2);
            defer std.testing.allocator.free(x);
            const guard: T = if (complex) .{ .re = 1e20, .im = 1e20 } else 1e20;
            @memset(x, guard);
            for (0..5) |pattern| {
                var best: usize = 0;
                var best_abs: R = -1;
                for (0..n) |i| {
                    const magnitude: R = @floatFromInt(switch (pattern) {
                        0 => i,
                        1 => n - i,
                        2 => (i * 17) % 251,
                        3 => if (i == n / 2 or i == n - 1) @as(usize, 1000) else 1,
                        else => if (i == 1 or i == 2 or i == n - 1) @as(usize, 1000) else 1,
                    });
                    const signed = if (i % 2 == 0) magnitude else -magnitude;
                    x[i + 1] = if (complex) .{ .re = signed, .im = magnitude / 2 } else signed;
                    const ax = if (complex) @abs(x[i + 1].re) + @abs(x[i + 1].im) else @abs(x[i + 1]);
                    if (ax > best_abs) {
                        best_abs = ax;
                        best = i;
                    }
                }
                const ni: i32 = @intCast(n);
                try std.testing.expectEqual(@as(i32, @intCast(best + 1)), iamax(&ni, x.ptr + 1, &inc));
                try std.testing.expectEqual(guard, x[0]);
                try std.testing.expectEqual(guard, x[n + 1]);
            }
        }
    }
}

test "NRM2 preserves finite tiny and huge inputs across vector and task boundaries" {
    defer fortran.zynum_blas_shutdown();
    const inc: i32 = 1;
    inline for (scalars) |T| {
        const R = if (T == f32 or T == api.ComplexF32) f32 else f64;
        const complex = T == api.ComplexF32 or T == api.ComplexF64;
        const norm = if (T == f32) fortran.snrm2_ else if (T == f64) fortran.dnrm2_ else if (T == api.ComplexF32) fortran.scnrm2_ else fortran.dznrm2_;
        const magnitudes = if (R == f32) [_]R{ 0, 1e-40, 1e-30, 1e-20, 1, 1e30 } else [_]R{ 0, 1e-310, 1e-200, 1e-150, 1, 1e300 };
        for ([_]usize{ 8, 31, 32, 33, 4096, 524289 }) |n| {
            const x = try std.testing.allocator.alloc(T, n);
            defer std.testing.allocator.free(x);
            for (magnitudes) |magnitude| {
                const v: T = if (complex) .{ .re = magnitude, .im = -magnitude } else magnitude;
                @memset(x, v);
                const ni: i32 = @intCast(n);
                const expected: R = @floatCast(@as(f64, magnitude) * @sqrt(@as(f64, @floatFromInt(n * (if (complex) @as(usize, 2) else 1)))));
                const tolerance = @max(@abs(expected) * 0.00003, 2 * std.math.floatTrueMin(R));
                try std.testing.expectApproxEqAbs(expected, norm(&ni, x.ptr, &inc), tolerance);
            }
        }
    }
}

test "NRM2 reciprocal overflow falls back for nonunit tiny inputs" {
    defer fortran.zynum_blas_shutdown();
    const n: i32 = 33;
    inline for (scalars) |T| {
        const R = if (T == f32 or T == api.ComplexF32) f32 else f64;
        const complex = T == api.ComplexF32 or T == api.ComplexF64;
        const norm = if (T == f32) fortran.snrm2_ else if (T == f64) fortran.dnrm2_ else if (T == api.ComplexF32) fortran.scnrm2_ else fortran.dznrm2_;
        const magnitude: R = if (R == f32) 1e-40 else 1e-310;
        const v: T = if (complex) .{ .re = magnitude, .im = -magnitude } else magnitude;
        const x = [_]T{v} ** 66;
        const expected: R = @floatCast(@as(f64, magnitude) * @sqrt(@as(f64, if (complex) 66 else 33)));
        for ([_]i32{ 2, -1 }) |inc| {
            try std.testing.expectApproxEqAbs(expected, norm(&n, &x, &inc), @max(expected * 0.00003, 2 * std.math.floatTrueMin(R)));
        }
    }
}

test "mixed DOT stays correct across SIMD and parallel partition boundaries" {
    defer fortran.zynum_blas_shutdown();
    const inc: i32 = 1;
    const bias: f32 = 0.5;
    for ([_]usize{ 4095, 4096, 4097, 32767, 32768, 65535, 65536, 65537, 1048576 }) |n| {
        const x = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(x);
        const y = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(y);
        var expected: f64 = 0;
        for (0..n) |i| {
            x[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 8;
            y[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 4;
            expected += @as(f64, x[i]) * @as(f64, y[i]);
        }
        const ni: i32 = @intCast(n);
        try std.testing.expectEqual(expected, fortran.dsdot_(&ni, x.ptr, &inc, y.ptr, &inc));
        try std.testing.expectEqual(@as(f32, @floatCast(expected + bias)), fortran.sdsdot_(&ni, &bias, x.ptr, &inc, y.ptr, &inc));
    }
}

test "ROTM preserves guards and all elements across SIMD and task boundaries" {
    defer fortran.zynum_blas_shutdown();
    const inc: i32 = 1;
    inline for (.{ f32, f64 }) |T| {
        const transform = if (T == f32) fortran.srotm_ else fortran.drotm_;
        for ([_]usize{ 4095, 4096, 4097, 65535, 65536, 65537, 1048576 }) |n| {
            const x = try std.testing.allocator.alloc(T, n + 2);
            defer std.testing.allocator.free(x);
            const y = try std.testing.allocator.alloc(T, n + 2);
            defer std.testing.allocator.free(y);
            for ([_]T{ -1, 0, 1 }) |flag| {
                @memset(x, 123);
                @memset(y, 456);
                for (0..n) |i| {
                    x[i + 1] = @as(T, @floatFromInt(i % 17)) / 8;
                    y[i + 1] = @as(T, @floatFromInt(i % 13)) / 4;
                }
                const param = [_]T{ flag, 0.5, -0.25, 0.125, 0.75 };
                const ni: i32 = @intCast(n);
                transform(&ni, x.ptr + 1, &inc, y.ptr + 1, &inc, &param);
                for (0..n) |i| {
                    const w = @as(T, @floatFromInt(i % 17)) / 8;
                    const z = @as(T, @floatFromInt(i % 13)) / 4;
                    const expected_x = if (flag < 0) 0.5 * w + 0.125 * z else if (flag == 0) w + 0.125 * z else 0.5 * w + z;
                    const expected_y = if (flag < 0) -0.25 * w + 0.75 * z else if (flag == 0) -0.25 * w + z else -w + 0.75 * z;
                    try std.testing.expectEqual(expected_x, x[i + 1]);
                    try std.testing.expectEqual(expected_y, y[i + 1]);
                }
                try std.testing.expectEqual(@as(T, 123), x[0]);
                try std.testing.expectEqual(@as(T, 123), x[n + 1]);
                try std.testing.expectEqual(@as(T, 456), y[0]);
                try std.testing.expectEqual(@as(T, 456), y[n + 1]);
            }
        }
    }
}

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

test "paired complex transposed GEMV preserves odd rows columns and unaligned guards" {
    const scalar = @import("core/shared/scalar.zig");
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    for ([_]usize{ 1, 4 }) |cap| {
        runtime.setMaxThreads(cap);
        inline for (.{ api.ComplexF32, api.ComplexF64 }) |T| {
            inline for (.{ .{ 131, 137 }, .{ 256, 256 }, .{ 259, 257 } }) |shape| {
                const m: api.BlasInt = shape[0];
                const n: api.BlasInt = shape[1];
                const lda: api.BlasInt = m + 3;
                const inc: api.BlasInt = 1;
                var a: [(@as(usize, shape[0]) + 3) * @as(usize, shape[1]) + 2]T = undefined;
                var x: [shape[0] + 2]T = undefined;
                for (&a, 0..) |*v, i| v.* = value(T, (@as(f64, @floatFromInt(i % 13)) - 6) / 16, (@as(f64, @floatFromInt(i % 11)) - 5) / 32);
                for (&x, 0..) |*v, i| v.* = value(T, (@as(f64, @floatFromInt(i % 7)) - 3) / 8, (@as(f64, @floatFromInt(i % 5)) - 2) / 16);
                const alpha = value(T, 0.625, -0.375);
                const beta = value(T, -0.25, 0.125);
                for ([_]u8{ 'T', 'C' }) |trans| {
                    const flag = [_]u8{trans};
                    var y = [_]T{value(T, 0.5, -0.25)} ** (shape[1] + 2);
                    y[0] = value(T, 999, -999);
                    y[shape[1] + 1] = y[0];
                    @field(fortran, prefix(T) ++ "gemv_")(&flag, &m, &n, &alpha, a[1..].ptr, &lda, x[1..].ptr, &inc, &beta, y[1..].ptr, &inc);
                    for (0..shape[1]) |j| {
                        var sum = scalar.zero(T);
                        for (0..shape[0]) |i| {
                            const av = a[1 + i + j * (shape[0] + 3)];
                            sum = scalar.add(T, sum, scalar.mul(T, if (trans == 'C') scalar.conj(T, av) else av, x[1 + i]));
                        }
                        const expected = scalar.add(T, scalar.mul(T, alpha, sum), scalar.mul(T, beta, value(T, 0.5, -0.25)));
                        try expectValue(T, expected.re, expected.im, y[1 + j]);
                    }
                    try expectValue(T, 999, -999, y[0]);
                    try expectValue(T, 999, -999, y[shape[1] + 1]);
                }
            }
        }
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

test "packed A GEMM preserves padded fringes and alpha beta in transpose modes" {
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    inline for (.{ f32, f64 }) |T| {
        const m: i32 = 65;
        const n: i32 = 67;
        const k: i32 = 129;
        const ld: i32 = 132;
        const alpha: T = 0.75;
        const beta: T = -0.5;
        const a = try std.testing.allocator.alloc(T, 132 * 129);
        defer std.testing.allocator.free(a);
        const b = try std.testing.allocator.alloc(T, 132 * 129);
        defer std.testing.allocator.free(b);
        const c = try std.testing.allocator.alloc(T, 132 * 67);
        defer std.testing.allocator.free(c);
        inline for (.{ false, true }) |trans_a| {
            inline for (.{ false, true }) |trans_b| {
                @memset(a, -999);
                @memset(b, -999);
                @memset(c, 3);
                for (0..129) |p| {
                    for (0..65) |row| a[if (trans_a) p + row * 132 else row + p * 132] = @as(T, @floatFromInt(@as(i32, @intCast((row + p) % 7)) - 3)) * 0.125;
                    for (0..67) |col| b[if (trans_b) col + p * 132 else p + col * 132] = @as(T, @floatFromInt(@as(i32, @intCast((col + p) % 5)) - 2)) * 0.25;
                }
                @field(fortran, prefix(T) ++ "gemm_")(if (trans_a) "T" else "N", if (trans_b) "T" else "N", &m, &n, &k, &alpha, a.ptr, &ld, b.ptr, &ld, &beta, c.ptr, &ld);
                for (0..67) |col| {
                    for (0..65) |row| {
                        var sum: f64 = 0;
                        for (0..129) |p| sum += @as(f64, a[if (trans_a) p + row * 132 else row + p * 132]) * @as(f64, b[if (trans_b) col + p * 132 else p + col * 132]);
                        try std.testing.expectApproxEqAbs(0.75 * sum - 1.5, @as(f64, c[row + col * 132]), 1e-5);
                    }
                    for (65..132) |row| try std.testing.expectEqual(@as(T, 3), c[row + col * 132]);
                }
            }
        }
    }
}

test "K blocked packed A GEMM preserves padded fringes and alpha beta in transpose modes" {
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    inline for (.{ f32, f64 }) |T| {
        const m: i32 = 65;
        const n: i32 = 67;
        const k: i32 = 513;
        const ld: i32 = 516;
        const alpha: T = 0.75;
        const beta: T = -0.5;
        const a = try std.testing.allocator.alloc(T, 516 * 513);
        defer std.testing.allocator.free(a);
        const b = try std.testing.allocator.alloc(T, 516 * 513);
        defer std.testing.allocator.free(b);
        const c = try std.testing.allocator.alloc(T, 516 * 67);
        defer std.testing.allocator.free(c);
        inline for (.{ false, true }) |trans_a| {
            inline for (.{ false, true }) |trans_b| {
                @memset(a, -999);
                @memset(b, -999);
                @memset(c, 3);
                for (0..513) |p| {
                    for (0..65) |row| a[if (trans_a) p + row * 516 else row + p * 516] = @as(T, @floatFromInt(@as(i32, @intCast((row + p) % 7)) - 3)) * 0.125;
                    for (0..67) |col| b[if (trans_b) col + p * 516 else p + col * 516] = @as(T, @floatFromInt(@as(i32, @intCast((col + p) % 5)) - 2)) * 0.25;
                }
                @field(fortran, prefix(T) ++ "gemm_")(if (trans_a) "T" else "N", if (trans_b) "T" else "N", &m, &n, &k, &alpha, a.ptr, &ld, b.ptr, &ld, &beta, c.ptr, &ld);
                for (0..67) |col| {
                    for (0..65) |row| {
                        var sum: f64 = 0;
                        for (0..513) |p| sum += @as(f64, a[if (trans_a) p + row * 516 else row + p * 516]) * @as(f64, b[if (trans_b) col + p * 516 else p + col * 516]);
                        try std.testing.expectApproxEqAbs(0.75 * sum - 1.5, @as(f64, c[row + col * 516]), 1e-5);
                    }
                    for (65..516) |row| try std.testing.expectEqual(@as(T, 3), c[row + col * 516]);
                }
            }
        }
    }
}

test "complex 3M general coefficients preserve conjugation tails and beta-zero semantics" {
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    inline for (.{ api.ComplexF32, api.ComplexF64 }) |T| {
        const R = if (T == api.ComplexF32) f32 else f64;
        const m: i32 = 65;
        const n: i32 = 67;
        const k: i32 = 65;
        const ld: i32 = 70;
        const a = try std.testing.allocator.alloc(T, 70 * 70);
        defer std.testing.allocator.free(a);
        const b = try std.testing.allocator.alloc(T, 70 * 70);
        defer std.testing.allocator.free(b);
        const c = try std.testing.allocator.alloc(T, 70 * 67 + 2);
        defer std.testing.allocator.free(c);
        for (a, 0..) |*v, i| v.* = value(T, @as(f64, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 16, @as(f64, @floatFromInt(i % 7)) / 32);
        for (b, 0..) |*v, i| v.* = value(T, @as(f64, @floatFromInt(i % 11)) / 32, @as(f64, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) / 16);
        for ([_][*:0]const u8{ "N", "T", "C" }) |ta| {
            for ([_][*:0]const u8{ "N", "T", "C" }) |tb| {
                for ([_]bool{ false, true }) |beta_zero| {
                    const alpha = value(T, 0.75, 0.125);
                    const beta = if (beta_zero) value(T, 0, 0) else value(T, -0.25, 0.125);
                    @memset(c, value(T, 777, 777));
                    for (0..67) |j| for (0..65) |i| {
                        c[1 + i + j * 70] = if (beta_zero) .{ .re = std.math.nan(R), .im = std.math.nan(R) } else value(T, 0.5, -0.25);
                    };
                    @field(fortran, prefix(T) ++ "gemm_")(ta, tb, &m, &n, &k, &alpha, a.ptr, &ld, b.ptr, &ld, &beta, c[1..].ptr, &ld);
                    for (0..67) |j| {
                        for (0..65) |i| {
                            var re: f64 = 0;
                            var im: f64 = 0;
                            for (0..65) |p| {
                                const av = a[if (ta[0] == 'N') i + p * 70 else p + i * 70];
                                const bv = b[if (tb[0] == 'N') p + j * 70 else j + p * 70];
                                const ai: f64 = if (ta[0] == 'C') -av.im else av.im;
                                const bi: f64 = if (tb[0] == 'C') -bv.im else bv.im;
                                re += @as(f64, av.re) * bv.re - ai * bi;
                                im += @as(f64, av.re) * bi + ai * bv.re;
                            }
                            const old_re: f64 = if (beta_zero) 0 else -0.09375;
                            const old_im: f64 = if (beta_zero) 0 else 0.125;
                            try expectValue(T, 0.75 * re - 0.125 * im + old_re, 0.75 * im + 0.125 * re + old_im, c[1 + i + j * 70]);
                        }
                        for (65..70) |i| try std.testing.expectEqual(value(T, 777, 777), c[1 + i + j * 70]);
                    }
                    try std.testing.expectEqual(value(T, 777, 777), c[0]);
                    try std.testing.expectEqual(value(T, 777, 777), c[c.len - 1]);
                }
            }
        }
    }
}

test "parallel complex 3M packing preserves uneven task fringes and coefficients" {
    runtime.setMaxThreads(3);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    inline for (.{ api.ComplexF32, api.ComplexF64 }) |T| {
        const R = if (T == api.ComplexF32) f32 else f64;
        const m: i32 = 257;
        const n: i32 = 259;
        const k: i32 = 257;
        const ld: i32 = 264;
        const a = try std.testing.allocator.alloc(T, 264 * 264);
        defer std.testing.allocator.free(a);
        const b = try std.testing.allocator.alloc(T, 264 * 264);
        defer std.testing.allocator.free(b);
        const c = try std.testing.allocator.alloc(T, 264 * 259 + 2);
        defer std.testing.allocator.free(c);
        for (a, 0..) |*v, i| v.* = value(T, @as(f64, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 16, @as(f64, @floatFromInt(i % 7)) / 32);
        for (b, 0..) |*v, i| v.* = value(T, @as(f64, @floatFromInt(i % 11)) / 32, @as(f64, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) / 16);
        for ([_][*:0]const u8{ "N", "T", "C" }) |ta| {
            for ([_][*:0]const u8{ "N", "T", "C" }) |tb| {
                for ([_]bool{ false, true }) |beta_zero| {
                    const alpha = value(T, 0.75, 0.125);
                    const beta = if (beta_zero) value(T, 0, 0) else value(T, -0.25, 0.125);
                    @memset(c, value(T, 777, 777));
                    for (0..259) |j| for (0..257) |i| {
                        c[1 + i + j * 264] = if (beta_zero) .{ .re = std.math.nan(R), .im = std.math.nan(R) } else value(T, 0.5, -0.25);
                    };
                    @field(fortran, prefix(T) ++ "gemm_")(ta, tb, &m, &n, &k, &alpha, a.ptr, &ld, b.ptr, &ld, &beta, c[1..].ptr, &ld);
                    for (0..259) |j| {
                        for (0..257) |i| {
                            var re: f64 = 0;
                            var im: f64 = 0;
                            for (0..257) |p| {
                                const av = a[if (ta[0] == 'N') i + p * 264 else p + i * 264];
                                const bv = b[if (tb[0] == 'N') p + j * 264 else j + p * 264];
                                const ai: f64 = if (ta[0] == 'C') -av.im else av.im;
                                const bi: f64 = if (tb[0] == 'C') -bv.im else bv.im;
                                re += @as(f64, av.re) * bv.re - ai * bi;
                                im += @as(f64, av.re) * bi + ai * bv.re;
                            }
                            const old_re: f64 = if (beta_zero) 0 else -0.09375;
                            const old_im: f64 = if (beta_zero) 0 else 0.125;
                            try expectValue(T, 0.75 * re - 0.125 * im + old_re, 0.75 * im + 0.125 * re + old_im, c[1 + i + j * 264]);
                        }
                        for (257..264) |i| try std.testing.expectEqual(value(T, 777, 777), c[1 + i + j * 264]);
                    }
                    try std.testing.expectEqual(value(T, 777, 777), c[0]);
                    try std.testing.expectEqual(value(T, 777, 777), c[c.len - 1]);
                }
            }
        }
    }
}
