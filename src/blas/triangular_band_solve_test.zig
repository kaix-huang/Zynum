// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const matrix_vector = @import("core/matrix_vector.zig");
const triangular_band_solve = @import("core/matrix_vector/triangular_band_solve.zig");
const scalar = @import("core/shared/scalar.zig");

const ComplexF32 = scalar.ComplexF32;
const ComplexF64 = scalar.ComplexF64;
const Diag = scalar.Diag;
const Order = scalar.Order;
const Uplo = scalar.Uplo;

fn value(comptime T: type, re: f64, im: f64) T {
    if (T == f32 or T == f64) return @floatCast(re);
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}

fn nanValue(comptime T: type) T {
    if (T == f32 or T == f64) return std.math.nan(T);
    const R = scalar.Real(T);
    return .{ .re = std.math.nan(R), .im = std.math.nan(R) };
}

fn matrixValue(comptime T: type, row: usize, column: usize) T {
    const re_seed = @as(f64, @floatFromInt((row * 17 + column * 11 + 5) % 37)) - 18.0;
    const im_seed = @as(f64, @floatFromInt((row * 7 + column * 19 + 3) % 31)) - 15.0;
    if (row == column) return value(T, 1.625 + re_seed / 512.0, im_seed / 768.0);
    return value(T, re_seed / 192.0, im_seed / 224.0);
}

fn solutionValue(comptime T: type, index: usize) T {
    const re_seed = @as(f64, @floatFromInt((index * 13 + 7) % 41)) - 20.0;
    const im_seed = @as(f64, @floatFromInt((index * 5 + 9) % 37)) - 18.0;
    return value(T, re_seed / 23.0, im_seed / 29.0);
}

fn fillBandMatrix(comptime T: type, a: []T, uplo: Uplo, diag: Diag, n: usize, k: usize, lda: usize) void {
    @memset(a, nanValue(T));
    for (0..n) |column| {
        if (uplo == .upper) {
            const first_row = column - @min(column, k);
            var band_row = k - (column - first_row);
            for (first_row..column + 1) |row| {
                if (diag == .non_unit or row != column) a[band_row + column * lda] = matrixValue(T, row, column);
                band_row += 1;
            }
        } else {
            const row_end = @min(n, column + k + 1);
            var band_row: usize = 0;
            for (column..row_end) |row| {
                if (diag == .non_unit or row != column) a[band_row + column * lda] = matrixValue(T, row, column);
                band_row += 1;
            }
        }
    }
}

fn accumulateReferenceEntry(
    comptime T: type,
    trans: Order,
    diag: Diag,
    row: usize,
    column: usize,
    matrix_index: usize,
    a: []const T,
    solution: []const T,
    rhs: []T,
) void {
    const stored = if (diag == .unit and row == column) scalar.one(T) else a[matrix_index];
    if (trans == .no_trans) {
        rhs[row] = scalar.add(T, rhs[row], scalar.mul(T, stored, solution[column]));
    } else {
        const av = if (trans == .conj_trans) scalar.conj(T, stored) else stored;
        rhs[column] = scalar.add(T, rhs[column], scalar.mul(T, av, solution[row]));
    }
}

// Build b = op(A) * solution out of place and by stored columns. This keeps
// the reference traversal independent from the in-place substitution kernels.
fn buildReferenceRhs(comptime T: type, uplo: Uplo, trans: Order, diag: Diag, n: usize, k: usize, a: []const T, lda: usize, solution: []const T, rhs: []T) void {
    @memset(rhs, scalar.zero(T));
    for (0..n) |column| {
        if (uplo == .upper) {
            const first_row = column - @min(column, k);
            var band_row = k - (column - first_row);
            for (first_row..column + 1) |row| {
                accumulateReferenceEntry(T, trans, diag, row, column, band_row + column * lda, a, solution, rhs);
                band_row += 1;
            }
        } else {
            const row_end = @min(n, column + k + 1);
            var band_row: usize = 0;
            for (column..row_end) |row| {
                accumulateReferenceEntry(T, trans, diag, row, column, band_row + column * lda, a, solution, rhs);
                band_row += 1;
            }
        }
    }
}

fn expectApprox(comptime T: type, expected: T, actual: T) !void {
    const tolerance = if (T == f32 or T == ComplexF32) @as(f32, 2e-3) else @as(f64, 2e-11);
    if (T == f32 or T == f64) {
        try std.testing.expect(std.math.isFinite(actual));
        try std.testing.expectApproxEqAbs(expected, actual, tolerance);
    } else {
        try std.testing.expect(std.math.isFinite(actual.re));
        try std.testing.expect(std.math.isFinite(actual.im));
        try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
        try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
    }
}

fn expectNan(comptime T: type, actual: T) !void {
    if (T == f32 or T == f64) {
        try std.testing.expect(std.math.isNan(actual));
    } else {
        try std.testing.expect(std.math.isNan(actual.re));
        try std.testing.expect(std.math.isNan(actual.im));
    }
}

fn expectFinite(comptime T: type, actual: T) !void {
    if (T == f32 or T == f64) {
        try std.testing.expect(std.math.isFinite(actual));
    } else {
        try std.testing.expect(std.math.isFinite(actual.re));
        try std.testing.expect(std.math.isFinite(actual.im));
    }
}

fn expectBandSentinels(comptime T: type, a: []const T, uplo: Uplo, diag: Diag, n: usize, k: usize, lda: usize) !void {
    for (0..n) |column| {
        for (0..lda) |band_row| {
            const is_logical = if (uplo == .upper)
                band_row >= k - @min(column, k) and band_row <= k
            else
                band_row <= @min(k, n - column - 1);
            const is_diagonal = band_row == (if (uplo == .upper) k else 0);
            if (!is_logical or (diag == .unit and is_diagonal)) {
                try expectNan(T, a[band_row + column * lda]);
            } else {
                try expectFinite(T, a[band_row + column * lda]);
            }
        }
    }
}

fn expectBytesEqual(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}

fn runCase(comptime T: type, uplo: Uplo, trans: Order, diag: Diag, n: usize, k: usize, check_facade: bool) !void {
    const lda = k + 4;
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, lda * n);
    defer allocator.free(a);
    const solution = try allocator.alloc(T, n);
    defer allocator.free(solution);
    const rhs = try allocator.alloc(T, n);
    defer allocator.free(rhs);
    const x_storage = try allocator.alloc(T, n + 2);
    defer allocator.free(x_storage);

    fillBandMatrix(T, a, uplo, diag, n, k, lda);
    const a_before = try allocator.dupe(T, a);
    defer allocator.free(a_before);
    for (solution, 0..) |*entry, i| entry.* = solutionValue(T, i);
    buildReferenceRhs(T, uplo, trans, diag, n, k, a, lda, solution, rhs);

    const x = x_storage[1 .. n + 1];
    @memcpy(x, rhs);
    x_storage[0] = value(T, 123.0, -77.0);
    x_storage[n + 1] = value(T, -91.0, 55.0);

    const forced_hit = triangular_band_solve.testing.tryTbsvForX86(
        T,
        uplo,
        trans,
        diag,
        @intCast(n),
        @intCast(k),
        a.ptr,
        @intCast(lda),
        x.ptr,
        1,
    );
    try std.testing.expect(forced_hit);
    for (solution, x) |want, got| try expectApprox(T, want, got);
    try expectApprox(T, value(T, 123.0, -77.0), x_storage[0]);
    try expectApprox(T, value(T, -91.0, 55.0), x_storage[n + 1]);

    @memcpy(x, rhs);
    const production_hit = triangular_band_solve.tryTbsv(
        T,
        uplo,
        trans,
        diag,
        @intCast(n),
        @intCast(k),
        a.ptr,
        @intCast(lda),
        x.ptr,
        1,
    );
    if (builtin.cpu.arch == .x86_64) {
        try std.testing.expect(production_hit);
        for (solution, x) |want, got| try expectApprox(T, want, got);
    } else {
        try std.testing.expect(!production_hit);
        try expectBytesEqual(T, rhs, x);
    }
    if (check_facade) {
        @memcpy(x, rhs);
        matrix_vector.tbsv(T, uplo, trans, diag, @intCast(n), @intCast(k), a.ptr, @intCast(lda), x.ptr, 1);
        for (solution, x) |want, got| try expectApprox(T, want, got);
    }

    try expectApprox(T, value(T, 123.0, -77.0), x_storage[0]);
    try expectApprox(T, value(T, -91.0, 55.0), x_storage[n + 1]);
    try expectBytesEqual(T, a_before, a);
    try expectBandSentinels(T, a, uplo, diag, n, k, lda);
}

fn checkType(comptime T: type) !void {
    for ([_]usize{ 0, 1, 8 }) |k| {
        for ([_]Uplo{ .upper, .lower }) |uplo| {
            for ([_]Order{ .no_trans, .trans, .conj_trans }) |trans| {
                for ([_]Diag{ .unit, .non_unit }) |diag| {
                    try runCase(T, uplo, trans, diag, 512, k, false);
                }
            }
        }
    }
}

fn checkComplexVectorHelpers(comptime T: type) !void {
    try runCase(T, .upper, .no_trans, .non_unit, 1024, 64, false);
    try runCase(T, .lower, .no_trans, .unit, 1024, 64, false);
    try runCase(T, .upper, .conj_trans, .unit, 1024, 64, false);
    try runCase(T, .lower, .conj_trans, .non_unit, 1024, 64, false);
}

fn expectGateMissUnchanged(comptime T: type, n: scalar.BlasInt, k: scalar.BlasInt, incx: scalar.BlasInt) !void {
    var a = [1]T{nanValue(T)};
    var x = [4]T{
        value(T, 1.0, -2.0),
        value(T, 3.0, -4.0),
        value(T, 5.0, -6.0),
        value(T, 7.0, -8.0),
    };
    const before = x;

    try std.testing.expect(!triangular_band_solve.testing.tryTbsvForX86(T, .upper, .no_trans, .unit, n, k, &a, 1, &x, incx));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&x));
    try std.testing.expect(!triangular_band_solve.tryTbsv(T, .upper, .no_trans, .unit, n, k, &a, 1, &x, incx));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&x));
}

fn checkGateMisses(comptime T: type) !void {
    try std.testing.expect(triangular_band_solve.testing.gateAllowsForX86(T, 512, 0, 1));
    try std.testing.expect(triangular_band_solve.testing.gateAllowsForX86(T, 512, 32, 1));
    try std.testing.expect(!triangular_band_solve.testing.gateAllowsForX86(T, 511, 0, 1));
    try std.testing.expect(!triangular_band_solve.testing.gateAllowsForX86(T, 512, -1, 1));
    try std.testing.expect(!triangular_band_solve.testing.gateAllowsForX86(T, 512, 33, 1));
    try std.testing.expect(!triangular_band_solve.testing.gateAllowsForX86(T, 512, 0, 2));
    try std.testing.expectEqual(
        builtin.cpu.arch == .x86_64,
        triangular_band_solve.testing.productionGateAllows(T, 512, 8, 1),
    );

    try expectGateMissUnchanged(T, 511, 0, 1);
    try expectGateMissUnchanged(T, 512, -1, 1);
    try expectGateMissUnchanged(T, 512, 33, 1);
    try expectGateMissUnchanged(T, 512, 0, 2);
}

test "TBSV band-window solve matches independent s/d/c/z references" {
    try runFiniteCandidate(f32);
    try runFiniteCandidate(f64);
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| try checkType(T);
    inline for (.{ ComplexF32, ComplexF64 }) |T| try checkComplexVectorHelpers(T);
}

test "matrix-vector facade routes compact TBSV" {
    try runCase(f64, .lower, .conj_trans, .unit, 512, 8, true);
}

test "TBSV band-window gate misses leave x unchanged" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| try checkGateMisses(T);
}

fn candidateReference(comptime T: type, uplo: Uplo, trans: Order, diag: Diag, n: usize, k: usize, a: [*]const T, lda: usize, x: [*]T, inc: i32) void {
    @setFloatMode(.strict);
    const stride: usize = @intCast(if (inc < 0) -inc else inc);
    const upper = (trans == .no_trans and uplo == .upper) or (trans != .no_trans and uplo == .lower);
    for (0..n) |iteration| {
        const i = if (upper) n - iteration - 1 else iteration;
        const pi = if (inc > 0) i * stride else (n - i - 1) * stride;
        var result = x[pi];
        const begin = if (upper) i + 1 else 0;
        const end = if (upper) n else i;
        for (begin..end) |j| {
            const row = if (trans == .no_trans) i else j;
            const col = if (trans == .no_trans) j else i;
            const stored = if (uplo == .upper) row <= col and col - row <= k else row >= col and row - col <= k;
            const av: T = if (!stored) 0 else if (uplo == .upper) a[k + row - col + col * lda] else a[row - col + col * lda];
            const pj = if (inc > 0) j * stride else (n - j - 1) * stride;
            const product = av * x[pj];
            result = result - product;
        }
        if (diag == .non_unit) result = result / a[i * lda + (if (uplo == .upper) k else @as(usize, 0))];
        x[pi] = result;
    }
}

fn runFiniteCandidate(comptime T: type) !void {
    const entry = @import("core/matrix_vector/compact_triangular_entry.zig");
    const catalog = @import("kernels/shared/matrix_vector/catalog.zig");
    const tuning = @import("kernels/shared/matrix_vector/tuning.zig");
    const id = catalog.Implementation.compact_triangular_band_finite;
    const scalar_kind: catalog.ScalarKind = if (T == f32) .f32 else .f64;
    const descriptor = catalog.findImplementation(.tbsv, scalar_kind, id).?;
    try std.testing.expectEqual(catalog.Lifecycle.experimental, descriptor.lifecycle);
    try std.testing.expectEqual(catalog.Implementation.portable_scalar, descriptor.fallback.?.implementation);
    try std.testing.expectEqual(catalog.CompletionScope.whole_operation, descriptor.completion);
    try std.testing.expect(descriptor.workspace.private_output);
    var profile = tuning.production_2026_07_17.triangular;
    profile.enable_finite_tbsv = true;
    try std.testing.expectEqual(if (builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos) id else .portable_scalar, profile.selectFiniteTbsv(T, 128, 32));
    try std.testing.expectEqual(catalog.Implementation.portable_scalar, profile.selectFiniteTbsv(T, 127, 8));
    try std.testing.expectEqual(catalog.Implementation.portable_scalar, profile.selectFiniteTbsv(T, 128, 33));
    profile.enable_finite_tbsv = false;
    try std.testing.expectEqual(catalog.Implementation.portable_scalar, profile.selectFiniteTbsv(T, 128, 8));
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .macos) return;
    var a: [5000]T = undefined;
    var x: [260]T = undefined;
    for ([_]usize{ 128, 129 }) |n| {
        for ([_]usize{ 0, 1, 8, 15, 16, 17, 32 }) |k| {
            const lda = k + 3;
            for ([_]Uplo{ .upper, .lower }) |uplo| {
                for ([_]Order{ .no_trans, .trans, .conj_trans }) |trans| {
                    for ([_]Diag{ .non_unit, .unit }) |diag| {
                        @memset(&a, std.math.nan(T));
                        for (0..n) |col| {
                            const first = if (uplo == .upper) col - @min(col, k) else col;
                            const last = if (uplo == .upper) col + 1 else @min(n, col + k + 1);
                            for (first..last) |row| {
                                const pos = if (uplo == .upper) k + row - col + col * lda else row - col + col * lda;
                                a[pos] = if (row == col) (if (diag == .unit) std.math.nan(T) else 2) else @as(T, 1.0 / 128.0);
                            }
                        }
                        for ([_]i32{ 1, 2, -1, -2 }) |inc| {
                            @memset(&x, -123);
                            const stride: usize = @intCast(if (inc < 0) -inc else inc);
                            for (0..n) |i| x[1 + i * stride] = 1 + @as(T, @floatFromInt(i % 5)) / 8;
                            var expected = x;
                            candidateReference(T, uplo, trans, diag, n, k, &a, lda, expected[1..].ptr, inc);
                            try std.testing.expect(entry.testing.forceTbsv(T, id, std.testing.allocator, 64 * 1024 * 1024, uplo, trans, diag, @intCast(n), @intCast(k), &a, @intCast(lda), x[1..].ptr, inc));
                            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&expected), std.mem.asBytes(&x));
                        }
                    }
                }
            }
        }
    }
    // A finite nonzero subnormal solution is eligible; poisoned unit diagonals stay unread.
    @memset(&a, std.math.nan(T));
    @memset(&x, std.math.floatMin(T) / 2);
    const subnormal_before = x;
    try std.testing.expect(entry.testing.forceTbsv(T, id, std.testing.allocator, 64 * 1024 * 1024, .lower, .no_trans, .unit, 128, 0, &a, 1, x[1..].ptr, 1));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&subnormal_before), std.mem.asBytes(&x));
    // Every zero case must complete through the O(n*k) candidate, bit for bit.
    // Vary sign representatives across skipped-prefix/suffix boundaries.
    for ([_]usize{ 0, 1, 8 }) |k| {
        const lda = k + 3;
        for ([_]Uplo{ .upper, .lower }) |uplo| {
            for ([_]Order{ .no_trans, .trans, .conj_trans }) |trans| {
                for ([_]Diag{ .non_unit, .unit }) |diag| {
                    @memset(&a, std.math.nan(T));
                    for (0..128) |col| {
                        const begin = if (uplo == .upper) col - @min(col, k) else col;
                        const end = if (uplo == .upper) col + 1 else @min(@as(usize, 128), col + k + 1);
                        for (begin..end) |row| {
                            const pos = if (uplo == .upper) k + row - col + col * lda else row - col + col * lda;
                            a[pos] = if (row == col)
                                (if (diag == .unit) std.math.nan(T) else if (row % 2 == 0) @as(T, 2) else -@as(T, 2))
                            else if ((row + col) % 2 == 0) @as(T, 0) else -@as(T, 0);
                        }
                    }
                    for ([_]i32{ 1, 2, -1, -2 }) |inc| {
                        const stride: usize = @intCast(if (inc < 0) -inc else inc);
                        for (0..8) |mode| {
                            @memset(&x, std.math.nan(T));
                            for (0..128) |j| {
                                const v: T = switch (mode) {
                                    0 => 0,
                                    1 => -@as(T, 0),
                                    2 => if (j % 2 == 0) @as(T, 0) else -@as(T, 0),
                                    3 => if (j <= k) -@as(T, 0) else @as(T, 0),
                                    4 => if (j == k) @as(T, 1) else -@as(T, 0),
                                    5 => if (j == 127 - k) -@as(T, 1) else @as(T, 0),
                                    6 => if (j == k or j == k + 1) @as(T, 1) else if (j == 126 - k or j == 127 - k) -@as(T, 1) else @as(T, 0),
                                    else => if (j % 3 == 0) @as(T, 1) else if (j % 3 == 1) -@as(T, 1) else -@as(T, 0),
                                };
                                x[1 + (if (inc > 0) j else 127 - j) * stride] = v;
                            }
                            var expected = x;
                            candidateReference(T, uplo, trans, diag, 128, k, &a, lda, expected[1..].ptr, inc);
                            try std.testing.expect(entry.testing.forceTbsv(T, id, std.testing.allocator, 64 * 1024 * 1024, uplo, trans, diag, 128, @intCast(k), &a, @intCast(lda), x[1..].ptr, inc));
                            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&expected), std.mem.asBytes(&x));
                        }
                    }
                }
            }
        }
    }
    // Refusals and accepted exceptional finite zeros compare every byte.
    for (0..13) |mode| {
        @memset(&a, 0);
        @memset(&x, 1);
        for (0..128) |i| a[i * 3] = 2;
        switch (mode) {
            0 => x[128] = std.math.nan(T), // fails after earlier private rows succeeded
            1 => x[128] = std.math.inf(T),
            2 => a[127 * 3] = 0,
            3 => a[127 * 3] = std.math.inf(T),
            4 => x[128] = 0,
            5 => a[127 * 3] = std.math.nan(T),
            6 => {
                a[126 * 3 + 1] = std.math.floatMax(T);
                x[127] = std.math.floatMax(T);
            },
            7 => {
                a[127 * 3] = std.math.floatMax(T);
                x[128] = std.math.floatMin(T);
            },
            8 => x[128] = -@as(T, 0),
            9 => {
                a[126 * 3 + 1] = 1;
                x[128] = 0.5;
            },
            10 => a[126 * 3 + 1] = std.math.inf(T),
            11 => {
                @memset(&x, 0);
                x[128] = std.math.nan(T);
            },
            else => {
                @memset(&x, 0);
                x[128] = std.math.inf(T);
            },
        }
        const before = x;
        const accepts_zero = mode == 3 or mode == 4 or mode == 7 or mode == 8 or mode == 9;
        try std.testing.expectEqual(accepts_zero, entry.testing.forceTbsv(T, id, std.testing.allocator, 64 * 1024 * 1024, .lower, .no_trans, .non_unit, 128, 1, &a, 3, x[1..].ptr, 1));
        var expected = before;
        candidateReference(T, .lower, .no_trans, .non_unit, 128, 1, &a, 3, expected[1..].ptr, 1);
        if (accepts_zero) {
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&expected), std.mem.asBytes(&x));
        } else {
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&x));
        }
        x = before;
        try std.testing.expect(entry.testing.forceTbsv(T, .portable_scalar, std.testing.allocator, 0, .lower, .no_trans, .non_unit, 128, 1, &a, 3, x[1..].ptr, 1));
        for (expected, x) |want, got| {
            if (std.math.isNan(want)) try std.testing.expect(std.math.isNan(got)) else try std.testing.expectEqualSlices(u8, std.mem.asBytes(&want), std.mem.asBytes(&got));
        }
    }
    @memset(&a, 1);
    @memset(&x, 1);
    const before = x;
    var empty: [0]u8 = .{};
    var failing = std.heap.FixedBufferAllocator.init(&empty);
    try std.testing.expect(!entry.testing.forceTbsv(T, id, failing.allocator(), 64 * 1024 * 1024, .lower, .no_trans, .unit, 128, 1, &a, 3, x[1..].ptr, 1));
    try std.testing.expect(!entry.testing.forceTbsv(T, id, std.testing.allocator, 0, .lower, .no_trans, .unit, 128, 1, &a, 3, x[1..].ptr, 1));
    try std.testing.expect(!entry.testing.forceTbsv(T, id, std.testing.allocator, 64 * 1024 * 1024, .lower, .no_trans, .unit, 127, 1, &a, 3, x[1..].ptr, 1));
    try std.testing.expect(!entry.testing.forceTbsv(T, id, std.testing.allocator, 64 * 1024 * 1024, .lower, .no_trans, .unit, 128, 33, &a, 35, x[1..].ptr, 1));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&x));
    const matrix_before = a;
    try std.testing.expect(!entry.testing.forceTbsv(T, id, std.testing.allocator, 64 * 1024 * 1024, .lower, .no_trans, .unit, 128, 1, &a, 3, &a, 1));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&matrix_before), std.mem.asBytes(&a));
}
