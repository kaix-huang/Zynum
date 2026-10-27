// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Correctness checks for a baseline dispatcher linked with ISA objects.
//! This root is deliberately separate from canonical inventory evidence.
const std = @import("std");
const builtin = @import("builtin");
const hardware = @import("zynum-core").internal.testing.hardware;
const client = @import("zynum-core").internal.testing.client;
const protocol = @import("zynum-core").internal.testing.protocol;
const runtime = @import("zynum-core").blas.runtime;
const api = @import("zynum-core").blas.api;
const fortran = @import("zynum-blas-abi").fortran;

const scalars = .{ f32, f64, api.ComplexF32, api.ComplexF64 };

test "prefetched reductions preserve ordinary values across cache-policy boundaries" {
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    const inc: i32 = 1;
    inline for (.{ f32, f64 }) |T| {
        const norm = if (T == f32) fortran.snrm2_ else fortran.dnrm2_;
        const abs_sum = if (T == f32) fortran.sasum_ else fortran.dasum_;
        for ([_]usize{ 4 * 1024, 8 * 1024, 256 * 1024, 2 * 1024 * 1024, 32 * 1024 * 1024 }) |bytes| {
            for ([_]isize{ -1, 0, 1 }) |offset| {
                const count: usize = @intCast(@as(isize, @intCast(bytes / @sizeOf(T))) + offset);
                const x = try std.testing.allocator.alloc(T, count);
                defer std.testing.allocator.free(x);
                @memset(x, 1);
                const n: i32 = @intCast(count);
                const expected = @sqrt(@as(f64, @floatFromInt(count)));
                try std.testing.expectApproxEqRel(expected, @as(f64, norm(&n, x.ptr, &inc)), 1e-6);
                try std.testing.expectEqual(@as(T, @floatFromInt(count)), abs_sum(&n, x.ptr, &inc));
                if (count <= 4096) {
                    x[count / 2] = std.math.nan(T);
                    try std.testing.expect(std.math.isNan(abs_sum(&n, x.ptr, &inc)));
                    x[count / 2] = std.math.inf(T);
                    try std.testing.expectEqual(std.math.inf(T), abs_sum(&n, x.ptr, &inc));
                }
            }
        }
    }
}

test "streaming copy preserves unaligned edges and declines unsupported tiers" {
    const binary = @import("zynum-core").internal.testing.vector_binary;
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
        for ([_]usize{ 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 4096, 524289 }) |n| {
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

test "NRM2 exceptional lanes preserve NaN and infinity across ASIMD tails" {
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    const inc: i32 = 1;
    inline for (scalars) |T| {
        const R = if (T == f32 or T == api.ComplexF32) f32 else f64;
        const complex = T == api.ComplexF32 or T == api.ComplexF64;
        const norm = if (T == f32) fortran.snrm2_ else if (T == f64) fortran.dnrm2_ else if (T == api.ComplexF32) fortran.scnrm2_ else fortran.dznrm2_;
        for ([_]usize{ 7, 8, 9, 15, 16, 17, 33 }) |count| {
            const x = try std.testing.allocator.alloc(T, count);
            defer std.testing.allocator.free(x);
            const n: i32 = @intCast(count);
            for (0..count) |lane| {
                @memset(x, if (complex) .{ .re = 1, .im = -1 } else 1);
                x[lane] = if (complex) .{ .re = 1, .im = std.math.inf(R) } else std.math.inf(R);
                try std.testing.expectEqual(std.math.inf(R), norm(&n, x.ptr, &inc));
                x[lane] = if (complex) .{ .re = 1, .im = std.math.nan(R) } else std.math.nan(R);
                try std.testing.expect(std.math.isNan(norm(&n, x.ptr, &inc)));
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
        const x: [66]T = @splat(v);
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
    inline for (@typeInfo(hardware.Capabilities).@"struct".field_names) |field| @field(cap, field) = true;
    return cap;
}

fn maximumTier() client.Tier {
    return switch (builtin.target.cpu.arch) {
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
    if (builtin.target.cpu.arch == .aarch64) {
        try std.testing.expectEqual(client.Tier.aarch64_sve2, client.select(full, .aarch64_sve2));
        try std.testing.expectEqual(client.Tier.aarch64_sme, client.select(full, .aarch64_sme));
        try std.testing.expectEqual(client.Tier.aarch64_sme2, client.select(full, .aarch64_sme2));
    } else if (builtin.target.cpu.arch == .x86_64) {
        try std.testing.expectEqual(client.Tier.x86_avx, client.select(full, .x86_avx));
        try std.testing.expectEqual(client.Tier.x86_avx2_fma, client.select(full, .x86_avx2_fma));
    }
}

test "missing compiler prerequisites downgrade an otherwise capable host" {
    var cap = allCapabilities();
    if (builtin.target.cpu.arch == .aarch64) {
        cap.sme_f64f64 = false;
        try std.testing.expectEqual(client.Tier.aarch64_sme, client.select(cap, maximumTier()));
        cap.bf16 = false;
        try std.testing.expectEqual(client.Tier.aarch64_sve2, client.select(cap, maximumTier()));
        cap.fullfp16 = false;
        try std.testing.expectEqual(client.Tier.baseline, client.select(cap, maximumTier()));
    } else if (builtin.target.cpu.arch == .x86_64) {
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
    inline for (@typeInfo(protocol.Operation).@"enum".field_names) |field| {
        const operation = comptime @field(protocol.Operation, field);
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

fn checkAxpyStreamingBoundaries() !void {
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    inline for (scalars) |T| {
        for ([_]usize{ 65535, 65536, 65537, 66047, 66049, 262145 }) |count| {
            const x = try std.testing.allocator.alloc(T, count + 2);
            defer std.testing.allocator.free(x);
            const y = try std.testing.allocator.alloc(T, count + 2);
            defer std.testing.allocator.free(y);
            @memset(x, value(T, 777, -9));
            @memset(y, value(T, 888, -8));
            for (0..count) |i| {
                x[i + 1] = value(T, @as(f64, @floatFromInt(i % 17)) - 8, 2);
                y[i + 1] = value(T, @as(f64, @floatFromInt(i % 7)) - 3, -1);
            }
            const n: api.BlasInt = @intCast(count);
            const stride: api.BlasInt = 1;
            const alpha = value(T, 0.5, 0);
            @field(fortran, prefix(T) ++ "axpy_")(&n, &alpha, x.ptr + 1, &stride, y.ptr + 1, &stride);
            for (0..count) |i| {
                const re = @as(f64, @floatFromInt(i % 17)) - 8;
                try expectValue(T, re, 2, x[i + 1]);
                try expectValue(T, (@as(f64, @floatFromInt(i % 7)) - 3) + 0.5 * re, 0, y[i + 1]);
            }
            try expectValue(T, 777, -9, x[0]);
            try expectValue(T, 777, -9, x[count + 1]);
            try expectValue(T, 888, -8, y[0]);
            try expectValue(T, 888, -8, y[count + 1]);
        }
    }
}

test "ABI scaling and checked dot preserve vector tails for every scalar" {
    try checkAxpyStreamingBoundaries();
    runtime.setMaxThreads(2);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    inline for (scalars) |T| {
        var x: [133]T = undefined;
        const y: [131]T = @splat(value(T, 1, 0));
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

const ComplexGemvCalleeSavedProbe = struct {
    // Use an assembly caller so LLVM cannot hide a callee violation by
    // spilling these registers around an ordinary numerical assertion.
    noinline fn invoke(target: usize, args: *const [8]usize) callconv(.naked) usize {
        _ = target;
        _ = args;
        asm volatile (
            \\stp x29, x30, [sp, #-48]!
            \\stp d8, d9, [sp, #16]
            \\stp d10, d11, [sp, #32]
            \\mov x16, x0
            \\mov x17, x1
            \\mov x9, #0x5a58
            \\fmov d8, x9
            \\mov x9, #0x5a59
            \\fmov d9, x9
            \\mov x9, #0x5a5a
            \\fmov d10, x9
            \\mov x9, #0x5a5b
            \\fmov d11, x9
            \\ldp x0, x1, [x17]
            \\ldp x2, x3, [x17, #16]
            \\ldp x4, x5, [x17, #32]
            \\ldp x6, x7, [x17, #48]
            \\blr x16
            \\fmov x10, d8
            \\mov x9, #0x5a58
            \\cmp x10, x9
            \\b.ne 1f
            \\fmov x10, d9
            \\mov x9, #0x5a59
            \\cmp x10, x9
            \\b.ne 1f
            \\fmov x10, d10
            \\mov x9, #0x5a5a
            \\cmp x10, x9
            \\b.ne 1f
            \\fmov x10, d11
            \\mov x9, #0x5a5b
            \\cmp x10, x9
            \\b.ne 1f
            \\mov x0, #0
            \\b 2f
            \\1:
            \\mov x0, #1
            \\2:
            \\ldp d10, d11, [sp, #32]
            \\ldp d8, d9, [sp, #16]
            \\ldp x29, x30, [sp], #48
            \\ret
            ::: .{ .memory = true });
    }

    fn check() !void {
        if (comptime builtin.target.cpu.arch != .aarch64) return;
        if (comptime !builtin.target.cpu.has(.aarch64, .complxnum)) return;
        const kernels = @import("zynum-core").internal.testing.aarch64_matrix_vector;
        const T = api.ComplexF64;
        var a: [259 * 128]T = @splat(.{ .re = 1, .im = 2 });
        var x: [128]T = @splat(.{ .re = 1, .im = -1 });
        var y: [128]T = @splat(.{ .re = 0, .im = 0 });
        const args = [8]usize{
            @bitCast(@as(f64, 1)), 0,                0,               0,
            @intFromPtr(&a),       259 * @sizeOf(T), @intFromPtr(&x), @intFromPtr(&y),
        };
        const Probe = *const fn (usize, *const [8]usize) callconv(.c) usize;
        const probe: Probe = @ptrCast(&invoke);
        inline for (.{ kernels.zgemvTransFcmlaF64M128Cols8, kernels.zgemvConjTransFcmlaF64M128Cols8 }) |kernel| {
            try std.testing.expectEqual(@as(usize, 0), probe(@intFromPtr(&kernel), &args));
        }
    }
};

test "paired complex transposed GEMV preserves odd rows columns and unaligned guards" {
    try ComplexGemvCalleeSavedProbe.check();
    const scalar = @import("zynum-core").internal.blas_unchecked.scalar;
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
                    var y: [(shape[1] + 2)]T = @splat(value(T, 0.5, -0.25));
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
        var y: [6]T = @splat(value(T, 1, 1));
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

fn checkGemmDoubleRowTails() !void {
    const allocator = std.testing.allocator;
    for ([_]usize{ 128, 129, 143, 144, 254, 255, 256 }) |rows| {
        const m: api.BlasInt = @intCast(rows);
        const n: api.BlasInt = 33;
        const k: api.BlasInt = 129;
        const lda: api.BlasInt = @intCast(rows + 3);
        const ldb: api.BlasInt = 132;
        const ldc: api.BlasInt = @intCast(rows + 5);
        const a = try allocator.alloc(f64, (rows + 3) * 129 + 2);
        defer allocator.free(a);
        const b = try allocator.alloc(f64, 132 * 33 + 2);
        defer allocator.free(b);
        const c = try allocator.alloc(f64, (rows + 5) * 33 + 2);
        defer allocator.free(c);
        @memset(a, -999);
        @memset(b, -999);
        @memset(c, -999);
        for (0..129) |p| {
            for (0..rows) |row| a[1 + row + p * (rows + 3)] = @as(f64, @floatFromInt(@as(i32, @intCast((row + p) % 7)) - 3)) * 0.125;
            for (0..33) |col| b[1 + p + col * 132] = @as(f64, @floatFromInt(@as(i32, @intCast((col + p) % 5)) - 2)) * 0.25;
        }
        const a_copy = try allocator.dupe(f64, a);
        defer allocator.free(a_copy);
        const b_copy = try allocator.dupe(f64, b);
        defer allocator.free(b_copy);
        const alpha: f64 = 1;
        const beta: f64 = 0;
        fortran.dgemm_("N", "N", &m, &n, &k, &alpha, a.ptr + 1, &lda, b.ptr + 1, &ldb, &beta, c.ptr + 1, &ldc);
        for (0..33) |col| {
            for (0..rows) |row| {
                var sum: f64 = 0;
                for (0..129) |p| sum = @mulAdd(f64, a[1 + row + p * (rows + 3)], b[1 + p + col * 132], sum);
                try std.testing.expectEqual(sum, c[1 + row + col * (rows + 5)]);
            }
            for (rows..rows + 5) |row| try std.testing.expectEqual(@as(f64, -999), c[1 + row + col * (rows + 5)]);
        }
        try std.testing.expectEqual(@as(f64, -999), c[0]);
        try std.testing.expectEqual(@as(f64, -999), c[c.len - 1]);
        try std.testing.expectEqualSlices(f64, a_copy, a);
        try std.testing.expectEqualSlices(f64, b_copy, b);
    }
}

test "ABI GEMM matches explicit fringe products for all four scalar types" {
    try checkGemmDoubleRowTails();
    runtime.setMaxThreads(2);
    defer runtime.setMaxThreads(0);
    defer fortran.zynum_blas_shutdown();
    inline for (scalars) |T| {
        var a: [49]T = @splat(value(T, -999, -999));
        var b: [24]T = @splat(value(T, -999, -999));
        var c: [18]T = @splat(value(T, -999, -999));
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

// Cross-object ownership coverage uses the same Core-owned Context as admitted
// BLAS, with a private allocator and a one-thread policy. Buffer fixtures are
// allocated separately, so the observed allocations are numerical scratch only.
const DynamicResourceExecution = @import("zynum-core").internal.numerics.Execution;
const DynamicResourceContext = DynamicResourceExecution.context.Context;
const DynamicReferenceContext = DynamicResourceExecution.reference.ReferenceContext;
fn dynamicProgress(ctx: *DynamicResourceContext) DynamicResourceExecution.context.CloseResult {
    var report: DynamicResourceExecution.CallReport = .{};
    return ctx.tryProgressClose(&report);
}
fn dynamicExecutor(ctx: *DynamicResourceContext) *DynamicResourceExecution.owner.Executor {
    const wrapper: *DynamicReferenceContext = @fieldParentPtr("context", ctx);
    return &wrapper.executor;
}

const DynamicResourceAllocations = DynamicResourceExecution.test_support.Allocations;

const DynamicResourceProbe = struct {
    var context: ?*DynamicResourceContext = null;
    var allocations: ?*DynamicResourceAllocations = null;
    var actual: []const u8 = &.{};
    var expected: []const u8 = &.{};
    var admitted: usize = 0;
    var finished: usize = 0;
    var watch: bool = false;

    fn maxThreads() usize {
        return 1;
    }
    fn helpers(_: usize) usize {
        return 0;
    }
    fn configure(_: ?usize) void {
        @panic("serial dynamic GEMM unexpectedly created a worker");
    }
    const policy: DynamicResourceExecution.owner.Policy = .{
        .max_threads = maxThreads,
        .helper_count = helpers,
        .configure_thread = configure,
        .worker_stack_size = 2 * 1024 * 1024,
    };
    fn setup(ctx: *DynamicResourceContext, log: *DynamicResourceAllocations) void {
        context = ctx;
        allocations = log;
        admitted = 0;
        finished = 0;
        watch = false;
        actual = &.{};
        expected = &.{};
        DynamicResourceContext.testing.configure(ctx, std.math.maxInt(usize), observe);
    }
    fn observe(ctx: *DynamicResourceContext, event: DynamicResourceExecution.context.TestEvent) void {
        if (ctx != context.? or ctx == DynamicResourceExecution.getContext())
            @panic("dynamic GEMM changed its admitted Context");
        const state = ctx.snapshot();
        if (state.active_operations != 1 or state.resource_pins != 0)
            @panic("dynamic GEMM nested admission or retained a scratch pin");
        switch (event) {
            .admitted => admitted += 1,
            .finishing => {
                if (watch) {
                    if (!std.mem.eql(u8, actual, expected))
                        @panic("dynamic GEMM ended before complete output or damaged padding");
                    const stats = allocations.?.stats();
                    if (stats.accepted != stats.freed)
                        @panic("dynamic object retained the borrowed allocator's scratch");
                }
                finished += 1;
            },
        }
    }
};

const DynamicResourceDefaultGuard = struct {
    const State = struct {
        context: DynamicResourceExecution.context.Snapshot,
        busy: u8,
        io_init: u8,
        persistent_init: u8,
        workers: u32,
        job_count: u32,
        generations: [DynamicResourceExecution.owner.max_tasks]u32,
        cache_domain: ?*const DynamicResourceExecution.owner.WorkerCacheDomain,
        max_threads_override: usize,
    };
    before: State,
    fn state() State {
        const executor = DynamicResourceExecution.get();
        var generations: [DynamicResourceExecution.owner.max_tasks]u32 = undefined;
        for (&generations, 0..) |*generation, i|
            generation.* = executor.persistent_worker_generation[i].load(.acquire);
        return .{
            .context = DynamicResourceExecution.getContext().snapshot(),
            .busy = executor.io_busy.load(.acquire),
            .io_init = executor.io_init_state.load(.acquire),
            .persistent_init = executor.persistent_init_state.load(.acquire),
            .workers = executor.persistent_worker_count.load(.acquire),
            .job_count = executor.persistent_job_count.load(.acquire),
            .generations = generations,
            .cache_domain = executor.persistent_cache_domain,
            .max_threads_override = DynamicResourceExecution.policy.maxThreadsOverride(),
        };
    }
    fn unexpected(_: *DynamicResourceContext, _: DynamicResourceExecution.context.TestEvent) void {
        @panic("dynamic borrowed GEMM used default Context admission");
    }
    fn init() !DynamicResourceDefaultGuard {
        const before = state();
        try std.testing.expectEqual(@as(usize, 0), before.context.active_operations);
        try std.testing.expectEqual(@as(usize, 0), before.context.resource_pins);
        try std.testing.expectEqual(@as(u8, 0), before.busy);
        DynamicResourceContext.testing.configure(DynamicResourceExecution.getContext(), std.math.maxInt(usize), unexpected);
        return .{ .before = before };
    }
    fn check(self: DynamicResourceDefaultGuard) !void {
        try std.testing.expectEqualDeep(self.before, state());
    }
    fn deinit(_: DynamicResourceDefaultGuard) void {
        DynamicResourceContext.testing.configure(DynamicResourceExecution.getContext(), std.math.maxInt(usize), null);
    }
};

fn dynamicResourceValue(index: usize, modulus: usize, bias: i64) i64 {
    return @as(i64, @intCast(index % modulus)) - bias;
}

fn dynamicResourceGemm(comptime T: type, m: usize, n: usize, k: usize, budget: usize) !DynamicResourceAllocations.Stats {
    try std.testing.expect(@import("zynum-build-options").dynamic_dispatch);
    const buffers = std.testing.allocator;
    const lda = m + 3;
    const ldb = k + 5;
    const ldc = m + 7;
    const a = try buffers.alloc(T, lda * k + 2);
    defer buffers.free(a);
    const b = try buffers.alloc(T, ldb * n + 2);
    defer buffers.free(b);
    const c = try buffers.alloc(T, ldc * n + 2);
    defer buffers.free(c);
    @memset(a, 203);
    @memset(b, 211);
    @memset(c, 137);
    for (0..k) |p| for (0..m) |i| {
        a[1 + i + p * lda] = @floatFromInt(dynamicResourceValue(i, 5, 2) + dynamicResourceValue(p, 7, 3));
    };
    for (0..n) |j| for (0..k) |p| {
        b[1 + p + j * ldb] = @floatFromInt(dynamicResourceValue(p, 3, 1) + dynamicResourceValue(j, 11, 5));
    };
    for (0..n) |j| for (0..m) |i| {
        c[1 + i + j * ldc] = @floatFromInt(2 * ((i + j) % 5) + 1);
    };
    const a_before = try buffers.dupe(T, a);
    defer buffers.free(a_before);
    const b_before = try buffers.dupe(T, b);
    defer buffers.free(b_before);
    const expected = try buffers.dupe(T, c);
    defer buffers.free(expected);
    // Independent rank-two formula, not a second GEMM loop. Small integral
    // inputs/intermediates and odd nonzero outputs are exact in both types.
    var sum_u: i64 = 0;
    var sum_v: i64 = 0;
    var sum_uv: i64 = 0;
    for (0..k) |p| {
        const u = dynamicResourceValue(p, 7, 3);
        const v = dynamicResourceValue(p, 3, 1);
        sum_u += u;
        sum_v += v;
        sum_uv += u * v;
    }
    for (0..n) |j| for (0..m) |i| {
        const r = dynamicResourceValue(i, 5, 2);
        const s = dynamicResourceValue(j, 11, 5);
        const product = r * sum_v + r * s * @as(i64, @intCast(k)) + sum_uv + s * sum_u;
        const old: i64 = @intCast(2 * ((i + j) % 5) + 1);
        expected[1 + i + j * ldc] = @floatFromInt(2 * product - 3 * old);
    };

    const default_guard = try DynamicResourceDefaultGuard.init();
    defer default_guard.deinit();
    var allocations: DynamicResourceAllocations = .{ .budget = budget };
    var reference_ctx: DynamicReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), DynamicResourceProbe.policy, .owned_closable, DynamicResourceExecution.reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    DynamicResourceProbe.setup(ctx, &allocations);
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) {
        DynamicResourceProbe.watch = false;
        operation.end();
    };
    DynamicResourceProbe.actual = std.mem.sliceAsBytes(c);
    DynamicResourceProbe.expected = std.mem.sliceAsBytes(expected);
    DynamicResourceProbe.watch = true;
    // The original admitted Fortran body reaches gemmRealAdmitted, its actual
    // CallResources/planner, and the linked object's packet dispatch.
    const mi: fortran.BlasInt = @intCast(m);
    const ni: fortran.BlasInt = @intCast(n);
    const ki: fortran.BlasInt = @intCast(k);
    const ldai: fortran.BlasInt = @intCast(lda);
    const ldbi: fortran.BlasInt = @intCast(ldb);
    const ldci: fortran.BlasInt = @intCast(ldc);
    const alpha: T = 2;
    const beta: T = -3;
    @field(fortran.internal_admitted, prefix(T) ++ "gemm_")(&operation, "N", "N", &mi, &ni, &ki, &alpha, a.ptr + 1, &ldai, b.ptr + 1, &ldbi, &beta, c.ptr + 1, &ldci);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(c));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a_before), std.mem.sliceAsBytes(a));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b_before), std.mem.sliceAsBytes(b));
    // No Executor allocation can explain observed scratch traffic in this
    // one-thread fixture. This also guards unintended default-pool fallback.
    try std.testing.expectEqual(@as(u8, 0), dynamicExecutor(ctx).io_init_state.load(.acquire));
    try std.testing.expectEqual(@as(u8, 0), dynamicExecutor(ctx).persistent_init_state.load(.acquire));
    try std.testing.expectEqual(@as(u8, 0), dynamicExecutor(ctx).io_busy.load(.acquire));
    const stats = allocations.stats();
    try std.testing.expectEqual(stats.accepted, stats.freed);
    operation.end();
    ended = true;
    DynamicResourceProbe.watch = false;
    try std.testing.expectEqual(@as(usize, 1), DynamicResourceProbe.admitted);
    try std.testing.expectEqual(@as(usize, 1), DynamicResourceProbe.finished);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
    try std.testing.expectEqual(DynamicResourceExecution.context.RequestResult.pending, ctx.requestClose());
    try std.testing.expectEqual(DynamicResourceExecution.context.CloseResult.closed, dynamicProgress(ctx));
    try default_guard.check();
    return stats;
}

test "dynamic GEMM heap packs use the borrowed Context allocator and preserve OOM fallback" {
    inline for (.{ f32, f64 }) |T| {
        // Arbitrary coefficients exclude SME/AMX direct GEMM. 13 rows disable
        // x86 K blocking/packed A; 4097*tile_n elements exceed the 64 KiB SIMD
        // stack pack. NN plus one task has no host-side allocation.
        const success = try dynamicResourceGemm(T, 13, 19, 4097, std.math.maxInt(usize));
        try std.testing.expectEqual(@as(usize, 1), success.attempts);
        try std.testing.expectEqual(@as(usize, 1), success.accepted);
        try std.testing.expectEqual(@as(usize, 1), success.freed);
        try std.testing.expectEqual(@as(usize, 0), success.rejected);
        const failed = try dynamicResourceGemm(T, 13, 19, 4097, 0);
        try std.testing.expectEqual(@as(usize, 1), failed.attempts);
        try std.testing.expectEqual(@as(usize, 0), failed.accepted);
        try std.testing.expectEqual(@as(usize, 0), failed.freed);
        try std.testing.expectEqual(@as(usize, 1), failed.rejected);
    }
}

test "dynamic GEMM vector shortcut carries borrowed allocation ownership across object tiers" {
    inline for (.{ f32, f64 }) |T| {
        // GEMM n=1 enters GEMV before planner. At these shapes an SME2 object
        // with 64-byte streaming vectors uses an allocation-free fused kernel;
        // Apple AMX otherwise owns two call-local B/C buffers. Baseline runs
        // with apple_amx enabled must exercise both actual allocations.
        const rows: usize = if (T == f32) 256 else 512;
        const stats = try dynamicResourceGemm(T, rows, 1, 257, std.math.maxInt(usize));
        const apple_amx = builtin.target.cpu.arch == .aarch64 and builtin.target.os.tag == .macos and @import("zynum-build-options").apple_amx;
        const tier = client.selectedTier();
        const may_use_sme2 = tier == .aarch64_sme2 or tier == .aarch64_sme2p1;
        if (!apple_amx) {
            try std.testing.expectEqual(@as(usize, 0), stats.attempts);
        } else if (!may_use_sme2) {
            try std.testing.expectEqual(@as(usize, 2), stats.attempts);
        } else {
            try std.testing.expect(stats.attempts == 0 or stats.attempts == 2);
        }
        try std.testing.expectEqual(@as(usize, 0), stats.rejected);
        try std.testing.expectEqual(stats.attempts, stats.accepted);
        try std.testing.expectEqual(stats.accepted, stats.freed);
        std.debug.print("dynamic GEMM shortcut {s} tier={s} scratch allocations={d} frees={d}\n", .{ @typeName(T), @tagName(tier), stats.accepted, stats.freed });
    }
}

fn dynamicComplexResourceGemm(comptime T: type, budget: usize) !DynamicResourceAllocations.Stats {
    try std.testing.expect(@import("zynum-build-options").dynamic_dispatch);
    const m = 13;
    const n = 19;
    const k = 4097;
    const lda = m + 3;
    const ldb = k + 5;
    const ldc = m + 7;
    const buffers = std.testing.allocator;
    const a = try buffers.alloc(T, lda * k + 2);
    defer buffers.free(a);
    const b = try buffers.alloc(T, ldb * n + 2);
    defer buffers.free(b);
    const c = try buffers.alloc(T, ldc * n + 2);
    defer buffers.free(c);
    @memset(a, .{ .re = 203, .im = -207 });
    @memset(b, .{ .re = 211, .im = -219 });
    @memset(c, .{ .re = 137, .im = -149 });
    for (0..k) |p| for (0..m) |i| {
        a[1 + i + p * lda] = .{ .re = 2, .im = 3 };
    };
    for (0..n) |j| for (0..k) |p| {
        b[1 + p + j * ldb] = .{ .re = 4, .im = -1 };
    };
    for (0..n) |j| for (0..m) |i| {
        c[1 + i + j * ldc] = .{ .re = 3, .im = 1 };
    };
    const a_before = try buffers.dupe(T, a);
    defer buffers.free(a_before);
    const b_before = try buffers.dupe(T, b);
    defer buffers.free(b_before);
    const expected = try buffers.dupe(T, c);
    defer buffers.free(expected);
    // (2+3i)(4-i)=11+10i; (2+i)k(11+10i)+(-3+2i)(3+i).
    // This closed-form integer expectation is independent of 3M and the real
    // object kernels and fits exactly in both scalar formats.
    for (0..n) |j| for (0..m) |i| {
        expected[1 + i + j * ldc] = .{ .re = 12 * k - 11, .im = 31 * k + 3 };
    };
    const guard = try DynamicResourceDefaultGuard.init();
    defer guard.deinit();
    var allocations: DynamicResourceAllocations = .{ .budget = budget };
    var reference_ctx: DynamicReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), DynamicResourceProbe.policy, .owned_closable, DynamicResourceExecution.reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    DynamicResourceProbe.setup(ctx, &allocations);
    defer {
        DynamicResourceProbe.watch = false;
        _ = ctx.requestClose();
        _ = dynamicProgress(ctx);
    }
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) {
        DynamicResourceProbe.watch = false;
        operation.end();
    };
    DynamicResourceProbe.actual = std.mem.sliceAsBytes(c);
    DynamicResourceProbe.expected = std.mem.sliceAsBytes(expected);
    DynamicResourceProbe.watch = true;
    const mi: fortran.BlasInt = m;
    const ni: fortran.BlasInt = n;
    const ki: fortran.BlasInt = k;
    const ldai: fortran.BlasInt = lda;
    const ldbi: fortran.BlasInt = ldb;
    const ldci: fortran.BlasInt = ldc;
    const alpha: T = .{ .re = 2, .im = 1 };
    const beta: T = .{ .re = -3, .im = 2 };
    @field(fortran.internal_admitted, prefix(T) ++ "gemm_")(&operation, "N", "N", &mi, &ni, &ki, &alpha, a.ptr + 1, &ldai, b.ptr + 1, &ldbi, &beta, c.ptr + 1, &ldci);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(c));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a_before), std.mem.sliceAsBytes(a));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b_before), std.mem.sliceAsBytes(b));
    const stats = allocations.stats();
    try std.testing.expectEqual(stats.accepted, stats.freed);
    try std.testing.expectEqual(@as(u8, 0), dynamicExecutor(ctx).io_init_state.load(.acquire));
    try std.testing.expectEqual(@as(u8, 0), dynamicExecutor(ctx).persistent_init_state.load(.acquire));
    operation.end();
    ended = true;
    DynamicResourceProbe.watch = false;
    try std.testing.expectEqual(@as(usize, 1), DynamicResourceProbe.admitted);
    try std.testing.expectEqual(@as(usize, 1), DynamicResourceProbe.finished);
    _ = ctx.requestClose();
    try std.testing.expectEqual(DynamicResourceExecution.context.CloseResult.closed, dynamicProgress(ctx));
    try guard.check();
    return stats;
}

test "dynamic complex decomposition carries borrowed allocation into real object packs" {
    inline for (.{ api.ComplexF32, api.ComplexF64 }) |T| {
        // The first allocation is the host 3M workspace. Thirteen real rows
        // exclude packed-A plans; long K exceeds the real SIMD stack pack.
        // Normal and forced-baseline invocations are separate actual runs.
        const success = try dynamicComplexResourceGemm(T, std.math.maxInt(usize));
        try std.testing.expect(success.accepted > 1);
        try std.testing.expectEqual(@as(usize, 0), success.rejected);
        const outer_oom = try dynamicComplexResourceGemm(T, 0);
        try std.testing.expectEqual(@as(usize, 0), outer_oom.accepted);
        try std.testing.expect(outer_oom.rejected > 0);
        const inner_oom = try dynamicComplexResourceGemm(T, 1);
        try std.testing.expectEqual(@as(usize, 1), inner_oom.accepted);
        try std.testing.expect(inner_oom.rejected > 0);
    }
}

const ComplexUnitResourceProbe = struct {
    var queries: std.atomic.Value(usize) = .init(0);
    var configurations: std.atomic.Value(usize) = .init(0);
    var expected: []const u8 = &.{};
    var actual: []const u8 = &.{};
    var context: ?*DynamicResourceContext = null;
    var watching: bool = false;
    var finishes: usize = 0;
    fn maxThreads() usize {
        _ = queries.fetchAdd(1, .monotonic);
        return 2;
    }
    fn helpers(limit: usize) usize {
        return @min(limit, 1);
    }
    fn configure(_: ?usize) void {
        _ = configurations.fetchAdd(1, .monotonic);
    }
    const policy: DynamicResourceExecution.owner.Policy = .{
        .max_threads = maxThreads,
        .helper_count = helpers,
        .configure_thread = configure,
        .worker_stack_size = 2 * 1024 * 1024,
    };
    fn observe(ctx: *DynamicResourceContext, event: DynamicResourceExecution.context.TestEvent) void {
        if (ctx != context.? or ctx == DynamicResourceExecution.getContext()) @panic("unit helper changed Context");
        if (ctx.snapshot().active_operations != 1 or ctx.snapshot().resource_pins != 0) @panic("unit helper lost operation ownership");
        if (event == .finishing and watching) {
            if (!std.mem.eql(u8, actual, expected)) @panic("unit helper finished before final stores or damaged a guard");
            finishes += 1;
        }
    }
};

test "complex GEMV unit helpers retain resources at actual Level 1 scheduling boundaries" {
    const core = @import("zynum-core").internal.blas_unchecked;
    const guard = try DynamicResourceDefaultGuard.init();
    defer guard.deinit();
    var allocations: DynamicResourceAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: DynamicReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), ComplexUnitResourceProbe.policy, .owned_closable, DynamicResourceExecution.reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    ComplexUnitResourceProbe.context = ctx;
    ComplexUnitResourceProbe.queries.store(0, .release);
    ComplexUnitResourceProbe.configurations.store(0, .release);
    ComplexUnitResourceProbe.finishes = 0;
    DynamicResourceContext.testing.configure(ctx, std.math.maxInt(usize), ComplexUnitResourceProbe.observe);
    defer {
        ComplexUnitResourceProbe.watching = false;
        _ = ctx.requestClose();
        _ = dynamicProgress(ctx);
    }
    var calls: usize = 0;
    inline for (.{ api.ComplexF32, api.ComplexF64 }) |T| {
        // Independent helper boundary tests; these do not claim a giant GEMM
        // end-to-end run. 2*n crosses the AArch64 real-f32 scaling cutoff too.
        for ([_]usize{ 512 * 1024 - 1, 512 * 1024, 1024 * 1024 + 1 }) |n| {
            for ([_]bool{ false, true }) |real_coefficient| {
                const x = try std.testing.allocator.alloc(T, n + 2);
                defer std.testing.allocator.free(x);
                const y = try std.testing.allocator.alloc(T, n + 2);
                defer std.testing.allocator.free(y);
                const expected = try std.testing.allocator.alloc(T, n + 2);
                defer std.testing.allocator.free(expected);
                @memset(x, .{ .re = 101, .im = -103 });
                @memset(y, .{ .re = 107, .im = -109 });
                @memset(expected, .{ .re = 107, .im = -109 });
                @memset(x[1 .. n + 1], .{ .re = 3, .im = 2 });
                @memset(y[1 .. n + 1], .{ .re = 5, .im = 7 });
                @memset(expected[1 .. n + 1], if (real_coefficient) T{ .re = 16, .im = 18 } else T{ .re = 7, .im = 26 });
                const alpha: T = .{ .re = 2, .im = if (real_coefficient) 0 else 1 };
                var operation = try ctx.begin();
                var ended = false;
                defer if (!ended) {
                    ComplexUnitResourceProbe.watching = false;
                    operation.end();
                };
                const resources: core.execution.CallResources = .{ .admitted = &operation };
                ComplexUnitResourceProbe.actual = std.mem.sliceAsBytes(y);
                ComplexUnitResourceProbe.expected = std.mem.sliceAsBytes(expected);
                ComplexUnitResourceProbe.watching = true;
                core.vector.scalUnitWithResources(T, resources, @intCast(n), alpha, y.ptr + 1);
                core.vector.axpyUnitWithResources(T, resources, @intCast(n), alpha, x.ptr + 1, y.ptr + 1);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(y));
                for (x, 0..) |element, i| {
                    const wanted: T = if (i == 0 or i == n + 1) .{ .re = 101, .im = -103 } else .{ .re = 3, .im = 2 };
                    try std.testing.expectEqualDeep(wanted, element);
                }
                operation.end();
                ended = true;
                ComplexUnitResourceProbe.watching = false;
                calls += 1;
                try guard.check();
            }
        }
    }
    try std.testing.expectEqual(calls, ComplexUnitResourceProbe.finishes);
    try std.testing.expect(ComplexUnitResourceProbe.queries.load(.acquire) > 0);
    try std.testing.expect(ComplexUnitResourceProbe.configurations.load(.acquire) > 0);
    _ = ctx.requestClose();
    try std.testing.expectEqual(DynamicResourceExecution.context.CloseResult.closed, dynamicProgress(ctx));
    try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
    try std.testing.expectEqual(@as(usize, 0), allocations.stats().rejected);
    try guard.check();
}

fn dynamicIndependentGemv(comptime T: type, budget: usize) !DynamicResourceAllocations.Stats {
    const rows: usize = if (T == f32) 256 else 512;
    const cols: usize = 257;
    const lda_u = rows + 3;
    const buffers = std.testing.allocator;
    const a = try buffers.alloc(T, lda_u * cols + 2);
    defer buffers.free(a);
    const x = try buffers.alloc(T, cols + 2);
    defer buffers.free(x);
    const y = try buffers.alloc(T, rows + 2);
    defer buffers.free(y);
    @memset(a, 101);
    @memset(x, 103);
    @memset(y, 107);
    for (0..cols) |j| for (0..rows) |i| {
        a[1 + i + j * lda_u] = 2;
    };
    @memset(x[1 .. cols + 1], 3);
    @memset(y[1 .. rows + 1], 5);
    const a_before = try buffers.dupe(T, a);
    defer buffers.free(a_before);
    const x_before = try buffers.dupe(T, x);
    defer buffers.free(x_before);
    const expected = try buffers.dupe(T, y);
    defer buffers.free(expected);
    // Closed-form independent reference; a second GEMV implementation is not used.
    @memset(expected[1 .. rows + 1], @as(T, @floatFromInt(12 * cols - 15)));
    const guard = try DynamicResourceDefaultGuard.init();
    defer guard.deinit();
    var allocations: DynamicResourceAllocations = .{ .budget = budget };
    var reference_ctx: DynamicReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), DynamicResourceProbe.policy, .owned_closable, DynamicResourceExecution.reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    defer {
        DynamicResourceProbe.watch = false;
        _ = ctx.requestClose();
        _ = dynamicProgress(ctx);
    }
    DynamicResourceProbe.setup(ctx, &allocations);
    var operation = try ctx.begin();
    var ended = false;
    defer if (!ended) {
        DynamicResourceProbe.watch = false;
        operation.end();
    };
    DynamicResourceProbe.actual = std.mem.sliceAsBytes(y);
    DynamicResourceProbe.expected = std.mem.sliceAsBytes(expected);
    DynamicResourceProbe.watch = true;
    try std.testing.expectEqual(DynamicResourceExecution.context.RequestResult.pending, ctx.requestClose());
    try std.testing.expectEqual(DynamicResourceExecution.context.CloseResult.busy, dynamicProgress(ctx));
    const m: i32 = @intCast(rows);
    const n: i32 = @intCast(cols);
    const lda: i32 = @intCast(lda_u);
    const inc: i32 = 1;
    const alpha: T = 2;
    const beta: T = -3;
    @field(fortran.internal_admitted, prefix(T) ++ "gemv_")(&operation, "N", &m, &n, &alpha, a.ptr + 1, &lda, x.ptr + 1, &inc, &beta, y.ptr + 1, &inc);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(y));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a_before), std.mem.sliceAsBytes(a));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(x_before), std.mem.sliceAsBytes(x));
    try std.testing.expectEqual(@as(u8, 0), dynamicExecutor(ctx).io_init_state.load(.acquire));
    try std.testing.expectEqual(@as(u8, 0), dynamicExecutor(ctx).persistent_init_state.load(.acquire));
    const stats = allocations.stats();
    try std.testing.expectEqual(stats.accepted, stats.freed);
    operation.end();
    ended = true;
    DynamicResourceProbe.watch = false;
    try std.testing.expectEqual(@as(usize, 1), DynamicResourceProbe.admitted);
    try std.testing.expectEqual(@as(usize, 1), DynamicResourceProbe.finished);
    try std.testing.expectEqual(DynamicResourceExecution.context.CloseResult.closed, dynamicProgress(ctx));
    try guard.check();
    return stats;
}

test "dynamic independent GEMV forwards borrowed allocator to the selected object and its OOM fallback" {
    try std.testing.expect(@import("zynum-build-options").dynamic_dispatch);
    inline for (.{ f32, f64 }) |T| {
        const success = try dynamicIndependentGemv(T, std.math.maxInt(usize));
        const apple_amx = builtin.target.cpu.arch == .aarch64 and builtin.target.os.tag == .macos and @import("zynum-build-options").apple_amx;
        const tier = client.selectedTier();
        const may_use_sme2 = tier == .aarch64_sme2 or tier == .aarch64_sme2p1;
        if (!apple_amx) {
            try std.testing.expectEqual(@as(usize, 0), success.attempts);
        } else if (!may_use_sme2) {
            try std.testing.expectEqual(@as(usize, 2), success.attempts);
        } else {
            try std.testing.expect(success.attempts == 0 or success.attempts == 2);
        }
        try std.testing.expectEqual(@as(usize, 0), success.rejected);
        try std.testing.expectEqual(success.attempts, success.accepted);
        try std.testing.expectEqual(success.accepted, success.freed);
        const refused = try dynamicIndependentGemv(T, 0);
        try std.testing.expectEqual(@as(usize, 0), refused.accepted);
        try std.testing.expectEqual(@as(usize, 0), refused.freed);
        try std.testing.expectEqual(refused.attempts, refused.rejected);
        if (success.attempts != 0) try std.testing.expect(refused.rejected > 0);
    }
}

const DynamicWorkspaceProbe = struct {
    var context: ?*DynamicResourceContext = null;
    var allocation_log: ?*DynamicResourceAllocations = null;
    var live_allocations: usize = 0;
    var result: []const f64 = &.{};
    var workspace: []const f64 = &.{};
    var expected_result: []const f64 = &.{};
    var expected_workspace: []const f64 = &.{};
    var watch: bool = false;
    var finishes: usize = 0;
    fn maxThreads() usize {
        return 4;
    }
    fn helpers(limit: usize) usize {
        return @min(limit, 3);
    }
    fn configure(_: ?usize) void {}
    const policy: DynamicResourceExecution.owner.Policy = .{
        .max_threads = maxThreads,
        .helper_count = helpers,
        .configure_thread = configure,
        .worker_stack_size = 2 * 1024 * 1024,
    };
    fn observe(ctx: *DynamicResourceContext, event: DynamicResourceExecution.context.TestEvent) void {
        if (ctx != context.? or ctx == DynamicResourceExecution.getContext()) @panic("workspace COPY changed its owner");
        if (ctx.snapshot().active_operations == 0 or ctx.snapshot().resource_pins != 0) @panic("workspace COPY lost admission");
        if (event == .finishing and watch) {
            if (!std.mem.eql(f64, result, expected_result) or !std.mem.eql(f64, workspace, expected_workspace))
                @panic("workspace operation ended before final copyback or damaged padding");
            const stats = allocation_log.?.stats();
            if (stats.accepted - stats.freed != live_allocations) @panic("workspace call retained scratch at finishing");
            finishes += 1;
        }
    }
};

const DynamicWorkspaceBatch = struct {
    context: *DynamicResourceContext,
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    finished: std.atomic.Value(bool) = .init(false),
    visits: [4]std.atomic.Value(u32) = @splat(.init(0)),
    accepted: bool = false,
    fn execute(raw: *const anyopaque, index: usize) void {
        const self: *@This() = @ptrCast(@alignCast(@constCast(raw)));
        _ = self.visits[index].fetchAdd(1, .monotonic);
        if (index == 0) {
            self.entered.store(true, .release);
            while (!self.release.load(.acquire)) std.Thread.yield() catch {};
        }
    }
    fn run(self: *@This()) void {
        defer self.finished.store(true, .release);
        var operation = self.context.begin() catch return;
        defer operation.end();
        self.accepted = dynamicExecutor(self.context).runLowLatency(.{ .task_fn = execute, .tasks = self, .count = 4 });
    }
};

test "checked GEMV workspace COPY uses both batches of the borrowed executor and finishes after closing" {
    // Four MiB crosses both existing architecture COPY gates, with only one
    // matrix column. These are valid prechecked views; the test-only alias runs
    // the exact public post-validation body, without another admission.
    const length = 4 * 1024 * 1024 / @sizeOf(f64);
    const buffers = std.testing.allocator;
    const a = try buffers.alloc(f64, length + 2);
    defer buffers.free(a);
    const y = try buffers.alloc(f64, length + 2);
    defer buffers.free(y);
    const scratch = try buffers.alloc(f64, length + 2);
    defer buffers.free(scratch);
    const expected = try buffers.alloc(f64, length + 2);
    defer buffers.free(expected);
    const expected_scratch = try buffers.alloc(f64, length + 2);
    defer buffers.free(expected_scratch);
    @memset(a, 97);
    const x = [_]f64{2};
    const matrix_view = try api.constMatrix(f64, a[1 .. length + 1], .{ .row_count = length, .column_count = 1 });
    const input_view = try api.constVector(f64, &x, .{});
    const result_view = try api.vector(f64, y[1 .. length + 1], .{});
    const guard = try DynamicResourceDefaultGuard.init();
    defer guard.deinit();
    for ([_]bool{ false, true }) |busy| {
        @memset(y, 101);
        @memset(scratch, 103);
        @memset(expected, 101);
        @memset(expected_scratch, 103);
        for (0..length) |i| {
            y[1 + i] = @floatFromInt(1 + i % 7);
            expected[1 + i] = -3 * y[1 + i];
            expected_scratch[1 + i] = expected[1 + i];
        }
        var allocations: DynamicResourceAllocations = .{ .budget = std.math.maxInt(usize) };
        var reference_ctx: DynamicReferenceContext = undefined;
        reference_ctx.initInPlace(allocations.allocator(), DynamicWorkspaceProbe.policy, .owned_closable, DynamicResourceExecution.reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        defer {
            DynamicWorkspaceProbe.watch = false;
            _ = ctx.requestClose();
            _ = dynamicProgress(ctx);
        }
        // Warm the actual four-task persistent executor, then record its exact
        // generation counters; COPY must issue two jobs, or none while busy.
        var warm: DynamicWorkspaceBatch = .{ .context = ctx };
        warm.release.store(true, .release);
        DynamicWorkspaceBatch.run(&warm);
        try std.testing.expect(warm.accepted);
        for (&warm.visits) |*v| try std.testing.expectEqual(@as(u32, 1), v.load(.acquire));
        try std.testing.expectEqual(@as(u32, 3), dynamicExecutor(ctx).persistent_worker_count.load(.acquire));
        DynamicWorkspaceProbe.context = ctx;
        DynamicWorkspaceProbe.allocation_log = &allocations;
        const before_allocations = allocations.stats();
        DynamicWorkspaceProbe.live_allocations = before_allocations.accepted - before_allocations.freed;
        DynamicWorkspaceProbe.result = y;
        DynamicWorkspaceProbe.workspace = scratch;
        DynamicWorkspaceProbe.expected_result = expected;
        DynamicWorkspaceProbe.expected_workspace = expected_scratch;
        DynamicWorkspaceProbe.watch = false;
        DynamicWorkspaceProbe.finishes = 0;
        DynamicResourceContext.testing.configure(ctx, std.math.maxInt(usize), DynamicWorkspaceProbe.observe);
        var active: DynamicWorkspaceBatch = .{ .context = ctx };
        var caller: ?std.Thread = null;
        defer {
            DynamicWorkspaceProbe.watch = false;
            active.release.store(true, .release);
            if (caller) |thread| thread.join();
        }
        if (busy) {
            caller = try std.Thread.spawn(.{}, DynamicWorkspaceBatch.run, .{&active});
            while (!active.entered.load(.acquire) and !active.finished.load(.acquire)) std.Thread.yield() catch {};
            try std.testing.expect(active.entered.load(.acquire));
        }
        var generations: [3]u32 = undefined;
        for (&generations, 0..) |*generation, i| generation.* = dynamicExecutor(ctx).persistent_worker_generation[i].load(.acquire);
        var operation = try ctx.begin();
        var ended = false;
        defer if (!ended) {
            DynamicWorkspaceProbe.watch = false;
            operation.end();
        };
        DynamicWorkspaceProbe.watch = true;
        try std.testing.expectEqual(DynamicResourceExecution.context.RequestResult.pending, ctx.requestClose());
        try std.testing.expectEqual(DynamicResourceExecution.context.CloseResult.busy, dynamicProgress(ctx));
        api.operations.testing.executeMatrixVectorWithWorkspaceAdmitted(f64, &operation, matrix_view, input_view, result_view, scratch[1 .. length + 1], length, 0, -3);
        for (generations, 0..) |generation, i|
            try std.testing.expectEqual(generation +% @as(u32, if (busy) 0 else 2), dynamicExecutor(ctx).persistent_worker_generation[i].load(.acquire));
        if (busy) {
            try std.testing.expect(!active.finished.load(.acquire));
            try std.testing.expectEqual(@as(u8, 1), dynamicExecutor(ctx).io_busy.load(.acquire));
            try std.testing.expect(dynamicExecutor(ctx).persistent_task_fn == DynamicWorkspaceBatch.execute);
        }
        try std.testing.expectEqualSlices(f64, expected, y);
        try std.testing.expectEqualSlices(f64, expected_scratch, scratch);
        for (a) |element| try std.testing.expectEqual(@as(f64, 97), element);
        try std.testing.expectEqual(@as(f64, 2), x[0]);
        operation.end();
        ended = true;
        DynamicWorkspaceProbe.watch = false;
        try std.testing.expectEqual(@as(usize, 1), DynamicWorkspaceProbe.finishes);
        if (caller) |thread| {
            active.release.store(true, .release);
            thread.join();
            caller = null;
            try std.testing.expect(active.accepted);
            for (&active.visits) |*v| try std.testing.expectEqual(@as(u32, 1), v.load(.acquire));
        }
        try std.testing.expectEqual(DynamicResourceExecution.context.CloseResult.closed, dynamicProgress(ctx));
        try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
        try guard.check();
    }
}

test "checked Level 1 Into COPY and combine retain one closing Context across object dispatch" {
    const n = 512 * 1024; // four MiB per vector: actual COPY and f64 AXPBY gates
    const buffers = std.testing.allocator;
    const x = try buffers.alloc(f64, n + 2);
    defer buffers.free(x);
    const input = try buffers.alloc(f64, n + 2);
    defer buffers.free(input);
    const output = try buffers.alloc(f64, n + 2);
    defer buffers.free(output);
    const expected = try buffers.alloc(f64, n + 2);
    defer buffers.free(expected);
    @memset(x, 97);
    @memset(input, 101);
    for (0..n) |i| {
        x[1 + i] = @floatFromInt(i % 7 + 1);
        input[1 + i] = @floatFromInt(i % 5 + 4);
    }
    const x_before = try buffers.dupe(f64, x);
    defer buffers.free(x_before);
    const input_before = try buffers.dupe(f64, input);
    defer buffers.free(input_before);
    const xv = try api.constVector(f64, x[1 .. n + 1], .{});
    const iv = try api.constVector(f64, input[1 .. n + 1], .{});
    const ov = try api.vector(f64, output[1 .. n + 1], .{});
    const guard = try DynamicResourceDefaultGuard.init();
    defer guard.deinit();
    for ([_]bool{ false, true }) |busy| {
        @memset(output, 103);
        @memset(expected, 103);
        for (0..n) |i| expected[1 + i] = 2 * x_before[1 + i] - 3 * input_before[1 + i];
        var allocations: DynamicResourceAllocations = .{ .budget = std.math.maxInt(usize) };
        var reference_ctx: DynamicReferenceContext = undefined;
        reference_ctx.initInPlace(allocations.allocator(), DynamicWorkspaceProbe.policy, .owned_closable, DynamicResourceExecution.reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        defer {
            DynamicWorkspaceProbe.watch = false;
            _ = ctx.requestClose();
            _ = dynamicProgress(ctx);
        }
        var warm: DynamicWorkspaceBatch = .{ .context = ctx };
        warm.release.store(true, .release);
        DynamicWorkspaceBatch.run(&warm);
        try std.testing.expect(warm.accepted);
        try std.testing.expectEqual(@as(u32, 3), dynamicExecutor(ctx).persistent_worker_count.load(.acquire));
        DynamicWorkspaceProbe.context = ctx;
        DynamicWorkspaceProbe.allocation_log = &allocations;
        DynamicWorkspaceProbe.result = output;
        DynamicWorkspaceProbe.expected_result = expected;
        DynamicWorkspaceProbe.workspace = input;
        DynamicWorkspaceProbe.expected_workspace = input_before;
        DynamicWorkspaceProbe.finishes = 0;
        DynamicResourceContext.testing.configure(ctx, std.math.maxInt(usize), DynamicWorkspaceProbe.observe);
        var active: DynamicWorkspaceBatch = .{ .context = ctx };
        var caller: ?std.Thread = null;
        defer {
            DynamicWorkspaceProbe.watch = false;
            active.release.store(true, .release);
            if (caller) |thread| thread.join();
        }
        if (busy) {
            caller = try std.Thread.spawn(.{}, DynamicWorkspaceBatch.run, .{&active});
            while (!active.entered.load(.acquire) and !active.finished.load(.acquire)) std.Thread.yield() catch {};
            try std.testing.expect(active.entered.load(.acquire));
        }
        var generations: [3]u32 = undefined;
        for (&generations, 0..) |*generation, i| generation.* = dynamicExecutor(ctx).persistent_worker_generation[i].load(.acquire);
        const allocation_state = allocations.stats();
        DynamicWorkspaceProbe.live_allocations = allocation_state.accepted - allocation_state.freed;
        var operation = try ctx.begin();
        var ended = false;
        defer if (!ended) {
            DynamicWorkspaceProbe.watch = false;
            operation.end();
        };
        DynamicWorkspaceProbe.watch = true;
        try std.testing.expectEqual(DynamicResourceExecution.context.RequestResult.pending, ctx.requestClose());
        try std.testing.expectEqual(DynamicResourceExecution.context.CloseResult.busy, dynamicProgress(ctx));
        // This alias is the exact checked post-validation helper. Public
        // validation/alias handling has separate modern-root coverage.
        api.operations.testing.executeCombineVectorsIntoAdmitted(f64, &operation, xv, iv, ov, 2, -3);
        if (comptime builtin.target.cpu.arch == .aarch64 or builtin.target.cpu.arch == .x86_64) {
            for (generations, 0..) |generation, i|
                try std.testing.expectEqual(generation +% @as(u32, if (busy) 0 else 2), dynamicExecutor(ctx).persistent_worker_generation[i].load(.acquire));
        }
        try std.testing.expectEqualSlices(f64, expected, output);
        try std.testing.expectEqualSlices(f64, x_before, x);
        try std.testing.expectEqualSlices(f64, input_before, input);
        operation.end();
        ended = true;
        DynamicWorkspaceProbe.watch = false;
        try std.testing.expectEqual(@as(usize, 1), DynamicWorkspaceProbe.finishes);
        if (caller) |thread| {
            try std.testing.expect(!active.finished.load(.acquire));
            try std.testing.expect(dynamicExecutor(ctx).persistent_task_fn == DynamicWorkspaceBatch.execute);
            active.release.store(true, .release);
            thread.join();
            caller = null;
            try std.testing.expect(active.accepted);
            for (&active.visits) |*visits| try std.testing.expectEqual(@as(u32, 1), visits.load(.acquire));
        }
        try std.testing.expectEqual(DynamicResourceExecution.context.CloseResult.closed, dynamicProgress(ctx));
        try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
        try guard.check();
    }
}

test "dynamic Level 1 reductions retain borrowed policy across real complex and mixed accumulation" {
    const guard = try DynamicResourceDefaultGuard.init();
    defer guard.deinit();
    var allocations: DynamicResourceAllocations = .{ .budget = std.math.maxInt(usize) };
    var reference_ctx: DynamicReferenceContext = undefined;
    reference_ctx.initInPlace(allocations.allocator(), ComplexUnitResourceProbe.policy, .owned_closable, DynamicResourceExecution.reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    ComplexUnitResourceProbe.context = ctx;
    ComplexUnitResourceProbe.queries.store(0, .release);
    ComplexUnitResourceProbe.configurations.store(0, .release);
    ComplexUnitResourceProbe.watching = false;
    DynamicResourceContext.testing.configure(ctx, std.math.maxInt(usize), ComplexUnitResourceProbe.observe);
    defer {
        ComplexUnitResourceProbe.watching = false;
        _ = ctx.requestClose();
        _ = dynamicProgress(ctx);
    }
    inline for (.{ f32, f64, api.ComplexF32, api.ComplexF64 }) |T| {
        const complex = T == api.ComplexF32 or T == api.ComplexF64;
        const n: usize = if (complex) 512 * 1024 else 2 * 1024 * 1024 + 1;
        const x = try std.testing.allocator.alloc(T, n + 2);
        defer std.testing.allocator.free(x);
        const y = try std.testing.allocator.alloc(T, n + 2);
        defer std.testing.allocator.free(y);
        @memset(x, value(T, 97, 43));
        @memset(y, value(T, 101, 47));
        @memset(x[1 .. n + 1], value(T, 1, 2));
        @memset(y[1 .. n + 1], value(T, 2, -1));
        // Equal largest values in distinct worker partitions: first tie wins.
        x[2] = value(T, 3, 4);
        x[n] = value(T, 3, 4);
        const ni: i32 = @intCast(n);
        const inc: i32 = 1;
        const nf: f64 = @floatFromInt(n);
        var operation = try ctx.begin();
        var ended = false;
        defer if (!ended) operation.end();
        const queries_before = ComplexUnitResourceProbe.queries.load(.acquire);
        if (comptime complex) {
            var result = value(T, 0, 0);
            @field(fortran.internal_admitted, prefix(T) ++ "dotu_sub_")(&operation, &ni, x.ptr + 1, &inc, y.ptr + 1, &inc, &result);
            try expectValue(T, 4 * nf + 12, 3 * nf + 4, result);
            @field(fortran.internal_admitted, prefix(T) ++ "dotc_sub_")(&operation, &ni, x.ptr + 1, &inc, y.ptr + 1, &inc, &result);
            try expectValue(T, 4, -5 * nf - 12, result);
        } else {
            const result = @field(fortran.internal_admitted, prefix(T) ++ "dot_")(&operation, &ni, x.ptr + 1, &inc, y.ptr + 1, &inc);
            try std.testing.expectEqual(value(T, 2 * nf + 8, 0), result);
        }
        if (T == f32) {
            const offset: f32 = 1;
            try std.testing.expectEqual(2 * nf + 8, fortran.internal_admitted.dsdot_(&operation, &ni, x.ptr + 1, &inc, y.ptr + 1, &inc));
            try std.testing.expectEqual(@as(f32, @floatCast(2 * nf + 9)), fortran.internal_admitted.sdsdot_(&operation, &ni, &offset, x.ptr + 1, &inc, y.ptr + 1, &inc));
        }
        const ap = comptime if (T == api.ComplexF32) "sc" else if (T == api.ComplexF64) "dz" else prefix(T);
        const sum = @field(fortran.internal_admitted, ap ++ "asum_")(&operation, &ni, x.ptr + 1, &inc);
        try std.testing.expectEqual(@as(f64, if (complex) 3 * nf + 8 else nf + 4), @as(f64, sum));
        const norm = @field(fortran.internal_admitted, ap ++ "nrm2_")(&operation, &ni, x.ptr + 1, &inc);
        try std.testing.expectApproxEqRel(@sqrt(if (complex) 5 * nf + 40 else nf + 16), @as(f64, norm), 0.000002);
        try std.testing.expectEqual(@as(i32, 2), @field(fortran.internal_admitted, "i" ++ prefix(T) ++ "amax_")(&operation, &ni, x.ptr + 1, &inc));
        try std.testing.expect(ComplexUnitResourceProbe.queries.load(.acquire) > queries_before);
        for (x[1 .. n + 1], 0..) |element, i| try std.testing.expectEqualDeep(value(T, if (i == 1 or i == n - 1) 3 else 1, if (i == 1 or i == n - 1) 4 else 2), element);
        for (y[1 .. n + 1]) |element| try std.testing.expectEqualDeep(value(T, 2, -1), element);
        try std.testing.expectEqualDeep(value(T, 97, 43), x[0]);
        try std.testing.expectEqualDeep(x[0], x[n + 1]);
        try std.testing.expectEqualDeep(value(T, 101, 47), y[0]);
        try std.testing.expectEqualDeep(y[0], y[n + 1]);
        operation.end();
        ended = true;
    }
    try std.testing.expect(ComplexUnitResourceProbe.configurations.load(.acquire) > 0);
    try std.testing.expect(allocations.stats().accepted > 0);
    _ = ctx.requestClose();
    try std.testing.expectEqual(DynamicResourceExecution.context.CloseResult.closed, dynamicProgress(ctx));
    try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
    try guard.check();
}
