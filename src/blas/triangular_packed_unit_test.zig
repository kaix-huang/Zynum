// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const matrix_vector = @import("core/matrix_vector.zig");
const packed_unit = @import("core/matrix_vector/triangular_packed_unit.zig");
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
    if (row == column) return value(T, 1.5 + re_seed / 512.0, im_seed / 640.0);
    return value(T, re_seed / 4096.0, im_seed / 5120.0);
}

fn vectorValue(comptime T: type, index: usize) T {
    const re_seed = @as(f64, @floatFromInt((index * 13 + 7) % 41)) - 20.0;
    const im_seed = @as(f64, @floatFromInt((index * 5 + 9) % 37)) - 18.0;
    return value(T, re_seed / 23.0, im_seed / 29.0);
}

fn fillPacked(comptime T: type, ap: []T, uplo: Uplo, diag: Diag, n: usize) void {
    @memset(ap, nanValue(T));
    var packed_index: usize = 0;
    for (0..n) |column| {
        if (uplo == .upper) {
            for (0..column + 1) |row| {
                if (diag == .non_unit or row != column) ap[packed_index] = matrixValue(T, row, column);
                packed_index += 1;
            }
        } else {
            for (column..n) |row| {
                if (diag == .non_unit or row != column) ap[packed_index] = matrixValue(T, row, column);
                packed_index += 1;
            }
        }
    }
}

fn unpackDense(comptime T: type, dense: []T, ap: []const T, uplo: Uplo, diag: Diag, n: usize) void {
    @memset(dense, scalar.zero(T));
    var packed_index: usize = 0;
    for (0..n) |column| {
        if (uplo == .upper) {
            for (0..column + 1) |row| {
                dense[row + column * n] = if (diag == .unit and row == column) scalar.one(T) else ap[packed_index];
                packed_index += 1;
            }
        } else {
            for (column..n) |row| {
                dense[row + column * n] = if (diag == .unit and row == column) scalar.one(T) else ap[packed_index];
                packed_index += 1;
            }
        }
    }
}

fn opValue(comptime T: type, dense: []const T, trans_: Order, n: usize, row: usize, column: usize) T {
    const stored_row = if (trans_ == .no_trans) row else column;
    const stored_column = if (trans_ == .no_trans) column else row;
    const entry = dense[stored_row + stored_column * n];
    return if (trans_ == .conj_trans) scalar.conj(T, entry) else entry;
}

// TPMV is referenced out of place through a dense matrix, independently of
// the packed-column in-place traversal used by the production leaf.
fn referenceTpmv(comptime T: type, dense: []const T, trans_: Order, n: usize, input: []const T, output: []T) void {
    for (0..n) |row| {
        var sum = scalar.zero(T);
        for (0..n) |column| {
            sum = scalar.add(T, sum, scalar.mul(T, opValue(T, dense, trans_, n, row, column), input[column]));
        }
        output[row] = sum;
    }
}

// TPSV uses row substitution over the dense operator. It neither shares packed
// offsets nor the column-update solve order used by no-transpose production.
fn referenceTpsv(comptime T: type, dense: []const T, uplo: Uplo, trans_: Order, diag: Diag, n: usize, rhs: []const T, output: []T) void {
    @memcpy(output, rhs);
    const op_is_upper = (uplo == .upper and trans_ == .no_trans) or (uplo == .lower and trans_ != .no_trans);
    if (op_is_upper) {
        var row = n;
        while (row > 0) {
            row -= 1;
            var solved = output[row];
            for (row + 1..n) |column| {
                solved = scalar.sub(T, solved, scalar.mul(T, opValue(T, dense, trans_, n, row, column), output[column]));
            }
            if (diag == .non_unit) solved = scalar.divv(T, solved, opValue(T, dense, trans_, n, row, row));
            output[row] = solved;
        }
    } else {
        for (0..n) |row| {
            var solved = output[row];
            for (0..row) |column| {
                solved = scalar.sub(T, solved, scalar.mul(T, opValue(T, dense, trans_, n, row, column), output[column]));
            }
            if (diag == .non_unit) solved = scalar.divv(T, solved, opValue(T, dense, trans_, n, row, row));
            output[row] = solved;
        }
    }
}

fn expectApprox(comptime T: type, expected: T, actual: T) !void {
    const tolerance = if (T == f32 or T == ComplexF32) @as(f32, 2e-3) else @as(f64, 5e-11);
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

fn expectPaddingNan(comptime T: type, storage: []const T, active_start: usize, active_len: usize) !void {
    for (storage[0..active_start]) |entry| try expectNan(T, entry);
    for (storage[active_start + active_len ..]) |entry| try expectNan(T, entry);
}

fn invokeTpmv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, ap: []const T, input: []const T, x: []T) !void {
    if (comptime builtin.cpu.arch == .x86_64) {
        try std.testing.expect(packed_unit.tryTpmv(T, uplo, trans_, diag, n, ap.ptr, x.ptr, 1));
    } else {
        try std.testing.expect(!packed_unit.tryTpmv(T, uplo, trans_, diag, n, ap.ptr, x.ptr, 1));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(input), std.mem.sliceAsBytes(x));
        packed_unit.testing.runTpmvUnit(T, uplo, trans_, diag, n, ap.ptr, x.ptr);
    }
}

fn invokeTpsv(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, n: usize, ap: []const T, input: []const T, x: []T) !void {
    if (comptime builtin.cpu.arch == .x86_64) {
        try std.testing.expect(packed_unit.tryTpsv(T, uplo, trans_, diag, n, ap.ptr, x.ptr, 1));
    } else {
        try std.testing.expect(!packed_unit.tryTpsv(T, uplo, trans_, diag, n, ap.ptr, x.ptr, 1));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(input), std.mem.sliceAsBytes(x));
        packed_unit.testing.runTpsvUnit(T, uplo, trans_, diag, n, ap.ptr, x.ptr);
    }
}

fn runCase(comptime T: type, uplo: Uplo, trans_: Order, diag: Diag, check_facade: bool) !void {
    const n = 131;
    const packed_len = n * (n + 1) / 2;
    const guard = 4;
    const allocator = std.testing.allocator;

    const ap_storage = try allocator.alloc(T, packed_len + 2 * guard);
    defer allocator.free(ap_storage);
    const ap = ap_storage[guard .. guard + packed_len];
    @memset(ap_storage, nanValue(T));
    fillPacked(T, ap, uplo, diag, n);
    const ap_before = try allocator.dupe(T, ap_storage);
    defer allocator.free(ap_before);

    const dense = try allocator.alloc(T, n * n);
    defer allocator.free(dense);
    unpackDense(T, dense, ap, uplo, diag, n);

    const input = try allocator.alloc(T, n);
    defer allocator.free(input);
    for (input, 0..) |*entry, i| entry.* = vectorValue(T, i);

    const expected = try allocator.alloc(T, n);
    defer allocator.free(expected);
    const x_storage = try allocator.alloc(T, n + 2 * guard);
    defer allocator.free(x_storage);
    const x = x_storage[guard .. guard + n];

    referenceTpmv(T, dense, trans_, n, input, expected);
    @memset(x_storage, nanValue(T));
    @memcpy(x, input);
    try invokeTpmv(T, uplo, trans_, diag, n, ap, input, x);
    for (expected, x) |want, got| try expectApprox(T, want, got);
    try expectPaddingNan(T, x_storage, guard, n);
    if (check_facade) {
        @memset(x_storage, nanValue(T));
        @memcpy(x, input);
        matrix_vector.tpmv(T, uplo, trans_, diag, @intCast(n), ap.ptr, x.ptr, 1);
        for (expected, x) |want, got| try expectApprox(T, want, got);
        try expectPaddingNan(T, x_storage, guard, n);
    }

    referenceTpsv(T, dense, uplo, trans_, diag, n, input, expected);
    @memset(x_storage, nanValue(T));
    @memcpy(x, input);
    try invokeTpsv(T, uplo, trans_, diag, n, ap, input, x);
    for (expected, x) |want, got| try expectApprox(T, want, got);
    try expectPaddingNan(T, x_storage, guard, n);
    if (check_facade) {
        @memset(x_storage, nanValue(T));
        @memcpy(x, input);
        matrix_vector.tpsv(T, uplo, trans_, diag, @intCast(n), ap.ptr, x.ptr, 1);
        for (expected, x) |want, got| try expectApprox(T, want, got);
        try expectPaddingNan(T, x_storage, guard, n);
    }

    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ap_before), std.mem.sliceAsBytes(ap_storage));
    try expectPaddingNan(T, ap_storage, guard, packed_len);
}

fn runAllCases(comptime T: type) !void {
    inline for (.{ Uplo.upper, Uplo.lower }) |uplo| {
        inline for (.{ Order.no_trans, Order.trans, Order.conj_trans }) |trans_| {
            inline for (.{ Diag.non_unit, Diag.unit }) |diag| try runCase(T, uplo, trans_, diag, false);
        }
    }
}

fn expectGateMissUnchanged(comptime T: type, n: usize, incx: scalar.BlasInt) !void {
    const allocator = std.testing.allocator;
    const packed_len = n * (n + 1) / 2;
    const ap = try allocator.alloc(T, packed_len);
    defer allocator.free(ap);
    for (ap, 0..) |*entry, i| entry.* = matrixValue(T, i % n, i % n);

    const x = try allocator.alloc(T, 2 * n + 8);
    defer allocator.free(x);
    for (x, 0..) |*entry, i| entry.* = vectorValue(T, i);
    const before = try allocator.dupe(T, x);
    defer allocator.free(before);

    try std.testing.expect(!packed_unit.tryTpmv(T, .upper, .no_trans, .non_unit, n, ap.ptr, x.ptr, incx));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before), std.mem.sliceAsBytes(x));
    try std.testing.expect(!packed_unit.tryTpsv(T, .lower, .conj_trans, .unit, n, ap.ptr, x.ptr, incx));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(before), std.mem.sliceAsBytes(x));
}

test "packed-column unit TPMV and TPSV match independent references" {
    try finitePackedCases(f32);
    try finitePackedCases(f64);
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| try runAllCases(T);
}

test "matrix-vector facade routes compact TPMV and TPSV" {
    try runCase(ComplexF64, .lower, .conj_trans, .unit, true);
}

test "packed-column production gates fail without modifying x" {
    inline for (.{ f32, f64, ComplexF32, ComplexF64 }) |T| {
        try std.testing.expectEqual(builtin.cpu.arch == .x86_64, packed_unit.testing.gateAllows(T, 128, 1));
        try std.testing.expect(!packed_unit.testing.gateAllows(T, 127, 1));
        try std.testing.expect(!packed_unit.testing.gateAllows(T, 128, 2));
        try expectGateMissUnchanged(T, 127, 1);
        try expectGateMissUnchanged(T, 128, 2);
    }
}

fn finitePackedReference(comptime T: type, uplo: Uplo, trans: Order, diag: Diag, n: usize, ap: [*]const T, x: [*]T, inc: i32) void {
    @setFloatMode(.strict);
    const stride: usize = @intCast(if (inc < 0) -inc else inc);
    const upper = (trans == .no_trans and uplo == .upper) or (trans != .no_trans and uplo == .lower);
    for (0..n) |iteration| {
        const i = if (upper) iteration else n - iteration - 1;
        var sum: T = 0;
        for (0..n) |j| {
            const row = if (trans == .no_trans) i else j;
            const col = if (trans == .no_trans) j else i;
            const av: T = if (row == col and diag == .unit) 1 else if (uplo == .upper)
                (if (row > col) 0 else ap[col * (col + 1) / 2 + row])
            else
                (if (row < col) 0 else ap[col * (2 * n - col + 1) / 2 + row - col]);
            const product = av * x[if (inc > 0) j * stride else (n - j - 1) * stride];
            sum = sum + product;
        }
        x[if (inc > 0) i * stride else (n - i - 1) * stride] = sum;
    }
}

fn finitePackedCases(comptime T: type) !void {
    const entry = @import("core/matrix_vector/compact_triangular_entry.zig");
    const catalog = @import("kernels/shared/matrix_vector/catalog.zig");
    const tuning = @import("kernels/shared/matrix_vector/tuning.zig");
    const id = catalog.Implementation.compact_triangular_packed_finite;
    const descriptor = catalog.findImplementation(.tpmv, if (T == f32) .f32 else .f64, id).?;
    try std.testing.expectEqual(catalog.Lifecycle.experimental, descriptor.lifecycle);
    try std.testing.expectEqual(catalog.Implementation.portable_scalar, descriptor.fallback.?.implementation);
    try std.testing.expectEqual(catalog.CompletionScope.whole_operation, descriptor.completion);
    try std.testing.expect(descriptor.workspace.private_output);
    var profile = tuning.production_2026_07_17.triangular;
    profile.enable_finite_tpmv = true;
    try std.testing.expectEqual(catalog.Implementation.portable_scalar, profile.selectFiniteTpmv(T, 63));
    try std.testing.expectEqual(if (builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos) id else .portable_scalar, profile.selectFiniteTpmv(T, 64));
    profile.enable_finite_tpmv = false;
    try std.testing.expectEqual(catalog.Implementation.portable_scalar, profile.selectFiniteTpmv(T, 128));
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .macos) return;
    var ap: [140000]T = undefined;
    var x: [1060]T = undefined;
    for ([_]usize{ 64, 65, 66, 67, 68, 69, 70, 71, 96, 97, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527 }) |n| {
        for ([_]Uplo{ .upper, .lower }) |uplo| {
            for ([_]Order{ .no_trans, .trans, .conj_trans }) |trans| {
                for ([_]Diag{ .non_unit, .unit }) |diag| {
                    @memset(&ap, std.math.nan(T));
                    for (0..n) |col| {
                        const begin = if (uplo == .upper) 0 else col;
                        const end = if (uplo == .upper) col + 1 else n;
                        for (begin..end) |row| {
                            const pos = if (uplo == .upper) col * (col + 1) / 2 + row else col * (2 * n - col + 1) / 2 + row - col;
                            // Mixed signs and nonbinary fractions expose an
                            // accidental reassociation when peeling diagonals.
                            const off_diagonal = @as(T, @floatFromInt(1 + row % 3)) / 997;
                            ap[pos] = if (row == col) (if (diag == .unit) std.math.nan(T) else 2) else if ((row + col) % 2 == 0) off_diagonal else -off_diagonal;
                        }
                    }
                    for ([_]i32{ 1, 2, -1, -2 }) |inc| {
                        @memset(&x, std.math.nan(T));
                        const stride: usize = @intCast(if (inc < 0) -inc else inc);
                        for (0..n) |j| {
                            const magnitude: T = switch (j % 3) {
                                0 => 1.0 / 16.0,
                                1 => 1,
                                else => 16,
                            };
                            const value_ = magnitude * (1 + @as(T, @floatFromInt(j % 5)) / 8);
                            x[1 + j * stride] = if (j % 2 == 0) value_ else -value_;
                        }
                        var expected = x;
                        finitePackedReference(T, uplo, trans, diag, n, &ap, expected[1..].ptr, inc);
                        try std.testing.expect(entry.testing.forceTpmv(T, id, std.testing.allocator, 64 * 1024 * 1024, uplo, trans, diag, @intCast(n), &ap, x[1..].ptr, inc));
                        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&expected), std.mem.asBytes(&x));
                    }
                }
            }
        }
    }
    // Row blocks and their tails must refuse without committing earlier rows,
    // including an invalid value in any of the sixteen sums or the final row.
    for ([_]usize{ 65, 129, 130, 131, 132, 133, 134, 135, 512, 527 }) |n| {
        for ([_]Uplo{ .upper, .lower }) |uplo| {
            for ([_]Order{ .no_trans, .trans, .conj_trans }) |trans| {
                if (n >= 512 and trans != .no_trans) continue;
                for ([_]i32{ 1, 2, -1, -2 }) |inc| {
                    for ([_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, n - 15, n - 8, n - 1 }) |bad_row| {
                        for ([_]T{ 0, std.math.nan(T), std.math.inf(T), std.math.floatMax(T) }) |bad| {
                            @memset(&ap, 0);
                            @memset(&x, 2);
                            for (0..n) |i| {
                                const offset = if (uplo == .upper) i * (i + 1) / 2 + i else i * (2 * n + 1 - i) / 2;
                                ap[offset] = if (i == bad_row) bad else 1;
                            }
                            const unchanged = x;
                            try std.testing.expect(!entry.testing.forceTpmv(T, id, std.testing.allocator, 64 * 1024 * 1024, uplo, trans, .non_unit, @intCast(n), &ap, x[1..].ptr, inc));
                            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&unchanged), std.mem.asBytes(&x));
                        }
                    }
                }
            }
        }
    }
    // Nonzero subnormal outputs remain eligible in both transpose block widths.
    // Unit diagonals must remain unread even on the exceptional-exponent path.
    for ([_]usize{ 64, 128 }) |n| {
        for ([_]Uplo{ .upper, .lower }) |uplo| {
            for ([_]Order{ .no_trans, .trans, .conj_trans }) |trans| {
                @memset(&ap, 0);
                for (0..n) |i| {
                    const diagonal = if (uplo == .upper) i * (i + 1) / 2 + i else i * (2 * n + 1 - i) / 2;
                    ap[diagonal] = std.math.nan(T);
                }
                @memset(&x, std.math.floatMin(T) / 2);
                const small_before = x;
                try std.testing.expect(entry.testing.forceTpmv(T, id, std.testing.allocator, 64 * 1024 * 1024, uplo, trans, .unit, @intCast(n), &ap, x[1..].ptr, 1));
                try std.testing.expectEqualSlices(u8, std.mem.asBytes(&small_before), std.mem.asBytes(&x));
            }
        }
    }
    for (0..10) |mode| {
        @memset(&ap, 0);
        @memset(&x, 1);
        for (0..128) |i| ap[i * (257 - i) / 2] = 2;
        switch (mode) {
            0 => x[1] = std.math.nan(T),
            1 => x[1] = std.math.inf(T),
            2 => ap[0] = std.math.nan(T), // last private row fails after earlier rows succeed
            3 => ap[0] = std.math.inf(T),
            4 => {
                ap[0] = std.math.inf(T);
                x[1] = 0;
            },
            5 => {
                ap[0] = std.math.floatMax(T);
                x[1] = 2;
            },
            6 => x[1] = -@as(T, 0),
            7 => {
                ap[1] = -2;
            }, // row1 cancels its diagonal
            8 => {
                ap[0] = std.math.floatMin(T);
                x[1] = std.math.floatMin(T);
            },
            else => {
                ap[0] = std.math.floatMax(T);
                ap[1] = std.math.floatMax(T);
                x[1] = 1;
                ap[128] = std.math.floatMax(T);
                x[2] = 1;
            },
        }
        const before = x;
        try std.testing.expect(!entry.testing.forceTpmv(T, id, std.testing.allocator, 64 * 1024 * 1024, .lower, .no_trans, .non_unit, 128, &ap, x[1..].ptr, 1));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&x));
        var expected = before;
        finitePackedReference(T, .lower, .no_trans, .non_unit, 128, &ap, expected[1..].ptr, 1);
        try std.testing.expect(entry.testing.forceTpmv(T, .portable_scalar, std.testing.allocator, 0, .lower, .no_trans, .non_unit, 128, &ap, x[1..].ptr, 1));
        for (expected, x) |want, got| {
            if (std.math.isNan(want)) try std.testing.expect(std.math.isNan(got)) else try std.testing.expectEqualSlices(u8, std.mem.asBytes(&want), std.mem.asBytes(&got));
        }
    }
    @memset(&ap, 1);
    @memset(&x, 1);
    const before = x;
    var empty: [0]u8 = .{};
    var failing = std.heap.FixedBufferAllocator.init(&empty);
    try std.testing.expect(!entry.testing.forceTpmv(T, id, failing.allocator(), 64 * 1024 * 1024, .upper, .no_trans, .unit, 128, &ap, x[1..].ptr, 1));
    try std.testing.expect(!entry.testing.forceTpmv(T, id, std.testing.allocator, 0, .upper, .no_trans, .unit, 128, &ap, x[1..].ptr, 1));
    try std.testing.expect(!entry.testing.forceTpmv(T, id, std.testing.allocator, 64 * 1024 * 1024, .upper, .no_trans, .unit, 63, &ap, x[1..].ptr, 1));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&x));
    const matrix_before = ap;
    try std.testing.expect(!entry.testing.forceTpmv(T, id, std.testing.allocator, 64 * 1024 * 1024, .upper, .no_trans, .unit, 128, &ap, &ap, 1));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&matrix_before), std.mem.asBytes(&ap));
}

test "finite f64 transpose preserves exact results across large prefetch threshold" {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .macos) return;
    const entry = @import("core/matrix_vector/compact_triangular_entry.zig");
    const allocator = std.testing.allocator;
    for ([_]usize{ 4095, 4096, 4097 }) |n| {
        const ap = try allocator.alloc(f64, n * (n + 1) / 2);
        defer allocator.free(ap);
        const x = try allocator.alloc(f64, n + 2);
        defer allocator.free(x);
        for ([_]Uplo{ .upper, .lower }) |uplo| {
            for ([_]Diag{ .unit, .non_unit }) |diag| {
                @memset(ap, 0.125);
                for (0..n) |j| {
                    const pos = if (uplo == .upper) j * (j + 1) / 2 + j else j * (2 * n - j + 1) / 2;
                    ap[pos] = if (diag == .unit) std.math.nan(f64) else 2;
                }
                for ([_]Order{ .trans, .conj_trans }) |trans| {
                    @memset(x, 1);
                    x[0] = -123;
                    x[n + 1] = -123;
                    try std.testing.expect(entry.testing.forceTpmv(f64, .compact_triangular_packed_finite, allocator, 64 * 1024 * 1024, uplo, trans, diag, @intCast(n), ap.ptr, x[1..].ptr, 1));
                    // With unit input and binary coefficients, counting the
                    // off-diagonal terms gives an exact independent oracle.
                    for (0..n) |i| {
                        const terms = if (uplo == .upper) i else n - i - 1;
                        const expected: f64 = (if (diag == .unit) @as(f64, 1) else 2) + @as(f64, @floatFromInt(terms)) * 0.125;
                        try std.testing.expectEqual(expected, x[i + 1]);
                    }
                    try std.testing.expectEqual(@as(f64, -123), x[0]);
                    try std.testing.expectEqual(@as(f64, -123), x[n + 1]);
                }
            }
        }
    }
}

test "production TPMV preserves staged results across small workspace threshold" {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .macos) return;
    const entry = @import("core/matrix_vector/compact_triangular_entry.zig");
    inline for (.{ f32, f64 }) |T| {
        for ([_]usize{ 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513 }) |n| {
            const ap = try std.testing.allocator.alloc(T, n * (n + 1) / 2);
            defer std.testing.allocator.free(ap);
            var storage: [1542]T = undefined;
            for ([_]Uplo{ .upper, .lower }) |uplo| {
                for ([_]Order{ .no_trans, .trans, .conj_trans }) |trans_| {
                    for ([_]i32{ 1, -1, 2, -2, 3, -3 }) |inc| {
                        const stride: usize = @intCast(if (inc < 0) -inc else inc);
                        const last = (n - 1) * stride;
                        const effective_upper = (uplo == .upper) == (trans_ == .no_trans);
                        // The final case forces a late zero-result refusal, so
                        // a partial write before scalar fallback changes results.
                        for (0..3) |mode| {
                            const diag: Diag = if (mode == 0) .unit else .non_unit;
                            @memset(ap, 0.125);
                            for (0..n) |i| {
                                const index = if (uplo == .upper) i * (i + 1) / 2 + i else i * (2 * n - i + 1) / 2;
                                ap[index] = if (diag == .unit) std.math.nan(T) else 2;
                                if (mode == 2 and i == n - 1) {
                                    const terms = if (effective_upper) n - i - 1 else i;
                                    ap[index] = -@as(T, @floatFromInt(terms)) * 0.125;
                                }
                            }
                            @memset(&storage, -123);
                            const x = storage[1..].ptr;
                            for (0..n) |i| x[if (inc > 0) i * stride else last - i * stride] = 1;
                            entry.tpmv(T, uplo, trans_, diag, @intCast(n), ap.ptr, x, inc);
                            for (0..n) |i| {
                                const terms = if (effective_upper) n - i - 1 else i;
                                const expected: T = if (mode == 2 and i == n - 1) 0 else (if (diag == .unit) @as(T, 1) else 2) + @as(T, @floatFromInt(terms)) * 0.125;
                                try std.testing.expectEqual(expected, x[if (inc > 0) i * stride else last - i * stride]);
                            }
                            try std.testing.expectEqual(@as(T, -123), storage[0]);
                            for (1..storage.len) |i| {
                                const offset = i - 1;
                                if (offset > last or offset % stride != 0) try std.testing.expectEqual(@as(T, -123), storage[i]);
                            }
                        }
                    }
                }
            }
        }
    }
}

test "finite TPMV rejects exceptional inputs before allocating across scan tails" {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .macos) return;
    const entry = @import("core/matrix_vector/compact_triangular_entry.zig");
    inline for (.{ f32, f64 }) |T| {
        var ap: [8385]T = undefined;
        @memset(&ap, 0.125);
        for ([_]usize{ 64, 65, 79, 80, 129 }) |n| {
            for ([_]i32{ 1, -1, 2, -2 }) |inc| {
                const stride: usize = @intCast(if (inc < 0) -inc else inc);
                for (0..n) |position| {
                    for ([_]T{ std.math.inf(T), std.math.nan(T) }) |exceptional| {
                        var storage: [260]T = @splat(-123);
                        for (0..n) |i| storage[1 + i * stride] = 1;
                        storage[1 + position * stride] = exceptional;
                        const before = storage;
                        var tracked = std.testing.FailingAllocator.init(std.testing.allocator, .{});
                        try std.testing.expect(!entry.testing.forceTpmv(T, .compact_triangular_packed_finite, tracked.allocator(), 64 * 1024 * 1024, .upper, .trans, .unit, @intCast(n), &ap, storage[1..].ptr, inc));
                        try std.testing.expectEqual(@as(usize, 0), tracked.allocated_bytes);
                        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&storage));
                    }
                }
            }
        }
    }
}
