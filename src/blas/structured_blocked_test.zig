// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const scalar = @import("core/shared/scalar.zig");
const indexing = @import("core/shared/indexing.zig");
const blocked = @import("core/matrix_matrix/structured_blocked.zig");
const symmetric = @import("core/matrix_matrix/symmetric.zig");
const triangular = @import("core/matrix_matrix/triangular.zig");
const packing = @import("kernels/shared/matrix_matrix/structured_packing.zig");

const matIndex = indexing.matIndex;

fn value(comptime T: type, real: f64, imaginary: f64) T {
    if (comptime scalar.isComplex(T)) return .{ .re = @floatCast(real), .im = @floatCast(imaginary) };
    return @floatCast(real);
}

fn sample(comptime T: type, index: usize, salt: usize) T {
    const re = (@as(f64, @floatFromInt((index * 17 + salt * 11) % 29)) - 14) / 16;
    const im = (@as(f64, @floatFromInt((index * 7 + salt * 13) % 31)) - 15) / 24;
    return value(T, re, im);
}

fn expectClose(comptime T: type, expected: T, actual: T) !void {
    const Real = scalar.Real(T);
    const absolute: Real = if (T == f32 or T == scalar.ComplexF32) 5e-3 else 5e-11;
    const relative: Real = if (T == f32 or T == scalar.ComplexF32) 5e-3 else 5e-11;
    if (comptime scalar.isComplex(T)) {
        const scale = @max(@abs(expected.re), @abs(expected.im));
        const tolerance = absolute + relative * scale;
        try std.testing.expect(@abs(expected.re - actual.re) <= tolerance);
        try std.testing.expect(@abs(expected.im - actual.im) <= tolerance);
    } else {
        try std.testing.expect(@abs(expected - actual) <= absolute + relative * @abs(expected));
    }
}

fn expectSlicesClose(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |want, got| try expectClose(T, want, got);
}

fn runSymmCase(comptime T: type, side: scalar.Side, uplo: scalar.Uplo, hermitian: bool) !void {
    return runSymmCaseSized(T, 9, 7, side, uplo, hermitian);
}

noinline fn runSymmCaseSized(comptime T: type, comptime m: usize, comptime n: usize, side: scalar.Side, uplo: scalar.Uplo, hermitian: bool) !void {
    const order = if (side == .left) m else n;
    const lda: scalar.BlasInt = @intCast(order + 2);
    const ldb: scalar.BlasInt = @intCast(m + 3);
    const ldc: scalar.BlasInt = @intCast(m + 4);
    var a: [(@as(usize, @max(m, n)) + 2) * @as(usize, @max(m, n))]T = undefined;
    var b: [(m + 3) * n]T = undefined;
    var expected: [(m + 4) * n]T = undefined;
    for (&a, 0..) |*entry, index| entry.* = sample(T, index, 1);
    for (&b, 0..) |*entry, index| entry.* = sample(T, index, 2);
    for (&expected, 0..) |*entry, index| entry.* = sample(T, index, 3);
    var actual = expected;
    const alpha = value(T, 0.75, if (hermitian) 0 else -0.1875);
    const beta = value(T, -0.25, if (hermitian) 0 else 0.125);

    if (m >= 128) {
        for (0..n) |j| {
            for (0..m) |i| {
                var sum = scalar.zero(T);
                for (0..order) |p| {
                    const row = if (side == .left) i else p;
                    const col = if (side == .left) p else j;
                    const stored = if (uplo == .upper) row <= col else row >= col;
                    var av = a[if (stored) matIndex(lda, row, col) else matIndex(lda, col, row)];
                    if (hermitian) {
                        if (!stored) av = scalar.conj(T, av);
                        if (comptime scalar.isComplex(T)) {
                            if (row == col) av.im = 0;
                        }
                    }
                    const bv = b[if (side == .left) matIndex(ldb, p, j) else matIndex(ldb, i, p)];
                    sum = scalar.add(T, sum, scalar.mul(T, av, bv));
                }
                const index = matIndex(ldc, i, j);
                expected[index] = scalar.add(T, scalar.mul(T, alpha, sum), scalar.mul(T, beta, expected[index]));
            }
        }
    } else {
        symmetric.symm(T, side, uplo, @intCast(m), @intCast(n), alpha, &a, lda, &b, ldb, beta, &expected, ldc, hermitian);
    }
    try std.testing.expect(blocked.trySymm(T, .{ .block_size = if (m >= 128) 64 else 4 }, side, uplo, @intCast(m), @intCast(n), alpha, &a, lda, &b, ldb, beta, &actual, ldc, hermitian));
    if (m >= 128) {
        // Exercise the public core selection against the independent scalar
        // reference too, including the AArch64 active-panel candidate.
        var selected: [(m + 4) * n]T = undefined;
        for (&selected, 0..) |*entry, index| entry.* = sample(T, index, 3);
        symmetric.symm(T, side, uplo, @intCast(m), @intCast(n), alpha, &a, lda, &b, ldb, beta, &selected, ldc, hermitian);
        try expectSlicesClose(T, &expected, &selected);
    }
    expectSlicesClose(T, &expected, &actual) catch |err| {
        std.debug.print("SYMM mismatch type={s} side={s} uplo={s} hermitian={}\n", .{ @typeName(T), @tagName(side), @tagName(uplo), hermitian });
        return err;
    };
}

test "full structured materialization matches scalar rectangular reference" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(4);
    defer {
        @import("core/execution/thread_pool.zig").shutdown();
        runtime.setMaxThreads(0);
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Side{ .left, .right }) |side| {
            for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
                try runSymmCaseSized(T, 263, 257, side, uplo, false);
                if (comptime scalar.isComplex(T)) try runSymmCaseSized(T, 263, 257, side, uplo, true);
            }
        }
    }
}

noinline fn runSymmZeroCase(comptime T: type, dimension: usize, side: scalar.Side, uplo: scalar.Uplo, hermitian: bool) !void {
    const ld: usize = dimension + 3;
    const elements: usize = (128 + 3) * 128;
    std.debug.assert(dimension > 0 and dimension <= 128);
    const dim_: scalar.BlasInt = @intCast(dimension);
    const ld_: scalar.BlasInt = @intCast(ld);
    var a: [elements]T = @splat(value(T, std.math.nan(f64), std.math.nan(f64)));
    var b: [elements]T = @splat(value(T, std.math.nan(f64), std.math.nan(f64)));
    var initial: [elements]T = undefined;
    for (&initial, 0..) |*entry, index| entry.* = sample(T, index, 19);
    for (0..dimension) |j| {
        for (0..dimension) |i| initial[i + j * ld] = if ((i + j) % 2 == 0) value(T, std.math.inf(f64), 1) else value(T, std.math.nan(f64), std.math.inf(f64));
    }
    for ([_]f64{ 0, 1, 2 }) |coefficient| {
        const beta = value(T, coefficient, 0);
        var actual = initial;
        var selected = initial;
        try std.testing.expect(@call(.never_inline, blocked.trySymm, .{ T, blocked.Options{ .block_size = 64 }, side, uplo, dim_, dim_, scalar.zero(T), &a, ld_, &b, ld_, beta, &actual, ld_, hermitian }));
        @call(.never_inline, symmetric.symm, .{ T, side, uplo, dim_, dim_, scalar.zero(T), &a, ld_, &b, ld_, beta, &selected, ld_, hermitian });
        // Runtime dimensions share a maximal allocation; preserve the complete
        // suffix beyond this case's logical matrix, including the small case.
        const logical_end = ld * dimension;
        try std.testing.expectEqualSlices(T, initial[logical_end..], actual[logical_end..]);
        try std.testing.expectEqualSlices(T, initial[logical_end..], selected[logical_end..]);
        for (0..dimension) |j| {
            for (0..ld) |i| {
                const index = i + j * ld;
                const want = if (i >= dimension or coefficient == 1) initial[index] else if (coefficient == 0) scalar.zero(T) else scalar.mul(T, beta, initial[index]);
                if (i >= dimension) {
                    try std.testing.expectEqual(want, actual[index]);
                    try std.testing.expectEqual(want, selected[index]);
                } else if (comptime scalar.isComplex(T)) {
                    try expectSpecialComponent(scalar.Real(T), want.re, actual[index].re);
                    try expectSpecialComponent(scalar.Real(T), want.im, actual[index].im);
                    try expectSpecialComponent(scalar.Real(T), want.re, selected[index].re);
                    try expectSpecialComponent(scalar.Real(T), want.im, selected[index].im);
                } else {
                    try expectSpecialComponent(T, want, actual[index]);
                    try expectSpecialComponent(T, want, selected[index]);
                }
            }
        }
    }
}

test "blocked SYMM and HEMM pack only active structured panels" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]usize{ 1, 128 }) |dimension| {
            for ([_]scalar.Side{ .left, .right }) |side| {
                for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
                    try runSymmZeroCase(T, dimension, side, uplo, false);
                    if (comptime scalar.isComplex(T)) try runSymmZeroCase(T, dimension, side, uplo, true);
                }
            }
        }
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Side{ .left, .right }) |side| {
            for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
                try runSymmCaseSized(T, 128, 128, side, uplo, false);
                try runSymmCaseSized(T, 129, 131, side, uplo, false);
                if (comptime scalar.isComplex(T)) try runSymmCaseSized(T, 129, 131, side, uplo, true);
            }
        }
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Side{ scalar.Side.left, scalar.Side.right }) |side| {
            for ([_]scalar.Uplo{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| try runSymmCase(T, side, uplo, false);
        }
    }
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Side{ scalar.Side.left, scalar.Side.right }) |side| {
            for ([_]scalar.Uplo{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| try runSymmCase(T, side, uplo, true);
        }
    }
}

fn runRankCase(comptime T: type, uplo: scalar.Uplo, trans: scalar.Order, hermitian: bool, rank2: bool) !void {
    return runRankCaseSized(T, 9, 7, 4, uplo, trans, hermitian, rank2);
}

noinline fn runRankCaseSized(comptime T: type, comptime n: usize, comptime k: usize, comptime block_size: usize, uplo: scalar.Uplo, trans: scalar.Order, hermitian: bool, rank2: bool) !void {
    const rows = if (trans == .no_trans) n else k;
    const lda: scalar.BlasInt = @intCast(rows + 2);
    const ldb: scalar.BlasInt = @intCast(rows + 3);
    const ldc: scalar.BlasInt = @intCast(n + 4);
    var a: [(@as(usize, @max(n, k)) + 3) * @as(usize, @max(n, k))]T = undefined;
    var b: [(@as(usize, @max(n, k)) + 3) * @as(usize, @max(n, k))]T = undefined;
    var expected: [(n + 4) * n]T = undefined;
    for (&a, 0..) |*entry, index| entry.* = sample(T, index, 4);
    for (&b, 0..) |*entry, index| entry.* = sample(T, index, 5);
    for (&expected, 0..) |*entry, index| entry.* = sample(T, index, 6);
    var actual = expected;
    const alpha = value(T, 0.625, if (hermitian and !rank2) 0 else -0.25);
    const beta = value(T, -0.375, 0);

    if (n >= 128) {
        const Reference = struct {
            fn op(matrix: [*]const T, ld: scalar.BlasInt, tr: scalar.Order, i: usize, p: usize) T {
                const v = matrix[if (tr == .no_trans) matIndex(ld, i, p) else matIndex(ld, p, i)];
                return if (tr == .conj_trans) scalar.conj(T, v) else v;
            }
        };
        for (0..n) |j| {
            for (0..n) |i| {
                if (if (uplo == .upper) i > j else i < j) continue;
                var first = scalar.zero(T);
                var second = scalar.zero(T);
                for (0..k) |p| {
                    const ai = Reference.op(&a, lda, trans, i, p);
                    const aj = Reference.op(&a, lda, trans, j, p);
                    const bi = Reference.op(&b, ldb, trans, i, p);
                    const bj = Reference.op(&b, ldb, trans, j, p);
                    const right = if (rank2) bj else aj;
                    first = scalar.add(T, first, scalar.mul(T, ai, if (hermitian) scalar.conj(T, right) else right));
                    if (rank2) second = scalar.add(T, second, scalar.mul(T, bi, if (hermitian) scalar.conj(T, aj) else aj));
                }
                const index = matIndex(ldc, i, j);
                expected[index] = scalar.add(T, scalar.mul(T, alpha, first), scalar.mul(T, beta, expected[index]));
                if (rank2) expected[index] = scalar.add(T, expected[index], scalar.mul(T, if (hermitian) scalar.conj(T, alpha) else alpha, second));
                if (hermitian and i == j) {
                    if (comptime scalar.isComplex(T)) expected[index].im = 0;
                }
            }
        }
    } else if (rank2) {
        symmetric.syr2k(T, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, &b, ldb, beta, &expected, ldc, hermitian);
    } else {
        symmetric.syrk(T, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, beta, &expected, ldc, hermitian);
    }
    if (rank2) {
        try std.testing.expect(blocked.trySyr2k(T, .{ .block_size = block_size }, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, &b, ldb, beta, &actual, ldc, hermitian));
    } else {
        try std.testing.expect(blocked.trySyrk(T, .{ .block_size = block_size }, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, beta, &actual, ldc, hermitian));
    }
    if (n >= 128 and (n <= 131 or n == 257) and block_size == 64) {
        var packed_output: [(n + 4) * n]T = undefined;
        for (&packed_output, 0..) |*entry, index| entry.* = sample(T, index, 6);
        for (0..4) |plan| {
            const parallel = plan != 0;
            for (&packed_output, 0..) |*entry, index| entry.* = sample(T, index, 6);
            const options = blocked.Options{ .block_size = 64, .packed_rank_panels = true, .parallel_packed_rank_panels = parallel, .packed_rank_pool_available = plan != 2, .workspace_byte_limit = if (plan == 3) 3 * 64 * 64 * @sizeOf(T) else std.math.maxInt(usize) };
            const operation: @import("kernels/shared/matrix_matrix/structured_catalog.zig").StructuredOperation = if (rank2) (if (hermitian) .her2k else .syr2k) else (if (hermitian) .herk else .syrk);
            const kernel = if (parallel) blocked.parallelPackedRankKernel(T, operation) else blocked.packedRankKernel(T, operation);
            if (plan == 1) {
                const runtime = @import("runtime.zig");
                var direct_options = options;
                direct_options.packed_rank_worker_limit = @min(@as(usize, 8), runtime.helperThreadCount(7) + 1);
                const completed = if (rank2)
                    blocked.tryParallelPackedRank(T, true, direct_options, uplo, trans, n, k, alpha, &a, lda, &b, ldb, beta, &packed_output, ldc, hermitian)
                else
                    blocked.tryParallelPackedRank(T, false, direct_options, uplo, trans, n, k, alpha, &a, lda, &b, ldb, beta, &packed_output, ldc, hermitian);
                if (runtime.maxThreads() > 1 and runtime.helperThreadCount(7) > 0) {
                    // No fallback wrapper: success proves the parallel body ran.
                    try std.testing.expect(completed);
                    try expectSlicesClose(T, &expected, &packed_output);
                } else {
                    try std.testing.expect(!completed);
                    for (packed_output, 0..) |entry, index| try std.testing.expectEqual(sample(T, index, 6), entry);
                }
                for (&packed_output, 0..) |*entry, index| entry.* = sample(T, index, 6);
            }
            try std.testing.expect(blocked.executePackedRank(T, kernel, options, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, &b, ldb, beta, &packed_output, ldc));
            try expectSlicesClose(T, &expected, &packed_output);
            for (0..n) |j| {
                for (0..n + 4) |i| {
                    if (i >= n or (if (uplo == .upper) i > j else i < j)) try std.testing.expectEqual(sample(T, i + j * (n + 4), 6), packed_output[i + j * (n + 4)]);
                }
                if (comptime scalar.isComplex(T)) {
                    if (hermitian) try std.testing.expectEqual(@as(scalar.Real(T), 0), packed_output[j + j * (n + 4)].im);
                }
            }
        }
    }
    if (n >= 128 and n <= 131) {
        var selected: [(n + 4) * n]T = undefined;
        for (&selected, 0..) |*entry, index| entry.* = sample(T, index, 6);
        if (rank2) {
            symmetric.syr2k(T, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, &b, ldb, beta, &selected, ldc, hermitian);
        } else {
            symmetric.syrk(T, uplo, trans, @intCast(n), @intCast(k), alpha, &a, lda, beta, &selected, ldc, hermitian);
        }
        try expectSlicesClose(T, &expected, &selected);
        for (0..n) |j| {
            for (0..n + 4) |i| {
                if (i >= n or (if (uplo == .upper) i > j else i < j)) {
                    try std.testing.expectEqual(sample(T, i + j * (n + 4), 6), selected[i + j * (n + 4)]);
                }
            }
            if (comptime scalar.isComplex(T)) {
                if (hermitian) try std.testing.expectEqual(@as(scalar.Real(T), 0), selected[j + j * (n + 4)].im);
            }
        }
    }
    try expectSlicesClose(T, &expected, &actual);
    for (0..n) |j| {
        if (comptime scalar.isComplex(T)) {
            if (hermitian) try std.testing.expectEqual(@as(scalar.Real(T), 0), actual[j + j * (n + 4)].im);
        }
        for (0..n + 4) |i| {
            if (i >= n or (if (uplo == .upper) i > j else i < j)) try std.testing.expectEqual(expected[i + j * (n + 4)], actual[i + j * (n + 4)]);
        }
    }
}

test "queued rank tiles match independent scalar reference including partial last tiles" {
    const rank_profile = @import("kernels/tuning/structured.zig").aarch64_macos_rank_tile_candidate;
    try std.testing.expect(rank_profile.candidate(64, 32, false));
    try std.testing.expect(!rank_profile.candidate(63, 1000, false));
    try std.testing.expect(!rank_profile.candidate(65, 33, true));
    try std.testing.expect(!rank_profile.candidate(96, 56, true));
    try std.testing.expect(rank_profile.candidate(96, 57, true));
    try std.testing.expect(rank_profile.candidate(127, 64, true));
    try std.testing.expect(rank_profile.candidate(128, 63, true));
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(8);
    defer {
        @import("core/execution/thread_pool.zig").shutdown();
        runtime.setMaxThreads(0);
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
            for ([_]scalar.Order{ .no_trans, .trans }) |trans| {
                try runRankCaseSized(T, 128, 64, 64, uplo, trans, false, false);
                try runRankCaseSized(T, 129, 65, 64, uplo, trans, false, true);
                try runRankCaseSized(T, 257, 73, 64, uplo, trans, false, false);
                try runRankCaseSized(T, 257, 73, 64, uplo, trans, false, true);
            }
            if (comptime scalar.isComplex(T)) {
                for ([_]scalar.Order{ .no_trans, .conj_trans }) |trans| {
                    try runRankCaseSized(T, 128, 64, 64, uplo, trans, true, false);
                    try runRankCaseSized(T, 129, 65, 64, uplo, trans, true, true);
                    try runRankCaseSized(T, 257, 73, 64, uplo, trans, true, false);
                    try runRankCaseSized(T, 257, 73, 64, uplo, trans, true, true);
                }
            }
        }
    }
}

test "rank panels and split diagonal preserve full general-coefficient results and storage fringes" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Uplo{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| {
            for ([_]scalar.Order{ scalar.Order.no_trans, scalar.Order.trans }) |trans| {
                try runRankCaseSized(T, 128, 64, 64, uplo, trans, false, false);
                try runRankCaseSized(T, 128, 64, 64, uplo, trans, false, true);
                try runRankCaseSized(T, 129, 65, 64, uplo, trans, false, false);
                try runRankCaseSized(T, 129, 65, 64, uplo, trans, false, true);
                try runRankCaseSized(T, 97, 73, 96, uplo, trans, false, false);
                try runRankCaseSized(T, 97, 73, 96, uplo, trans, false, true);
            }
            if (comptime scalar.isComplex(T)) {
                for ([_]scalar.Order{ scalar.Order.no_trans, scalar.Order.conj_trans }) |trans| {
                    try runRankCaseSized(T, 128, 64, 64, uplo, trans, true, false);
                    try runRankCaseSized(T, 128, 64, 64, uplo, trans, true, true);
                    try runRankCaseSized(T, 129, 65, 64, uplo, trans, true, false);
                    try runRankCaseSized(T, 129, 65, 64, uplo, trans, true, true);
                    try runRankCaseSized(T, 97, 73, 96, uplo, trans, true, false);
                    try runRankCaseSized(T, 97, 73, 96, uplo, trans, true, true);
                }
            }
        }
    }
}

test "single-thread rank transpose reuse preserves odd full output and padding" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(1);
    defer runtime.setMaxThreads(0);
    inline for (.{ f32, f64 }) |T| {
        for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
            try runRankCaseSized(T, 129, 67, 64, uplo, .trans, false, false);
            try runRankCaseSized(T, 129, 67, 64, uplo, .trans, false, true);
            try runRankCaseSized(T, 257, 73, 64, uplo, .trans, false, false);
            try runRankCaseSized(T, 257, 73, 64, uplo, .trans, false, true);
        }
    }
}

test "queued real transpose panels preserve partial final blocks at larger depth" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(4);
    defer {
        @import("core/execution/thread_pool.zig").shutdown();
        runtime.setMaxThreads(0);
    }
    inline for (.{ f32, f64 }) |T| {
        for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
            try runRankCaseSized(T, 385, 257, 64, uplo, .trans, false, false);
            try runRankCaseSized(T, 385, 257, 64, uplo, .trans, false, true);
        }
    }
}

test "tiled real transpose packing preserves odd extents and output guards" {
    inline for (.{ f32, f64 }) |T| {
        const m = 65;
        const k = 1027;
        const lda = k + 3;
        var input: [lda * m]T = undefined;
        for (&input, 0..) |*element, i| element.* = @as(T, @floatFromInt(i)) / 137;
        var output: [m * k + 2]T = @splat(-999);
        @import("kernels/shared/matrix_matrix/packing.zig").packTransposedA(T, m, k, &input, lda, output[1 .. m * k + 1]);
        for (0..k) |p| {
            for (0..m) |i| try std.testing.expectEqual(input[p + i * lda], output[1 + p * m + i]);
        }
        try std.testing.expectEqual(@as(T, -999), output[0]);
        try std.testing.expectEqual(@as(T, -999), output[m * k + 1]);
    }
}

noinline fn runRankZeroProductEpilogue(comptime T: type, uplo: scalar.Uplo, hermitian: bool, rank_two: bool) !void {
    const n: usize = 3;
    const ld: usize = 5;
    const a: [ld * n]T = @splat(scalar.zero(T));
    const b = a;
    var initial: [ld * n]T = undefined;
    for (&initial, 0..) |*entry, index| entry.* = sample(T, index, 23);
    for (0..n) |j| {
        for (0..n) |i| {
            if (if (uplo == .upper) i <= j else i >= j) {
                initial[i + j * ld] = if (hermitian and i == j)
                    value(T, 3, std.math.nan(f64))
                else if ((i + j) % 2 == 0)
                    value(T, std.math.inf(f64), 2)
                else
                    value(T, std.math.nan(f64), std.math.inf(f64));
            }
        }
    }
    for ([_]f64{ 0, 1, 2 }) |coefficient| {
        var actual = initial;
        const beta = value(T, coefficient, 0);
        // alpha1,k2 executes the old column product path even though A/B=0.
        if (rank_two) {
            @call(.never_inline, symmetric.syr2k, .{ T, uplo, scalar.Order.no_trans, @as(scalar.BlasInt, 3), @as(scalar.BlasInt, 2), value(T, 1, 0), &a, @as(scalar.BlasInt, 5), &b, @as(scalar.BlasInt, 5), beta, &actual, @as(scalar.BlasInt, 5), hermitian });
        } else {
            @call(.never_inline, symmetric.syrk, .{ T, uplo, scalar.Order.no_trans, @as(scalar.BlasInt, 3), @as(scalar.BlasInt, 2), value(T, 1, 0), &a, @as(scalar.BlasInt, 5), beta, &actual, @as(scalar.BlasInt, 5), hermitian });
        }
        for (0..n) |j| {
            for (0..ld) |i| {
                const index = i + j * ld;
                const stored = i < n and (if (uplo == .upper) i <= j else i >= j);
                if (!stored) {
                    try std.testing.expectEqual(initial[index], actual[index]);
                    continue;
                }
                var expected = initial[index];
                if (comptime scalar.isComplex(T)) {
                    if (coefficient == 0) {
                        expected = .{ .re = 0, .im = 0 };
                    } else if (hermitian and i == j) {
                        expected = .{ .re = @as(scalar.Real(T), @floatCast(coefficient)) * initial[index].re, .im = 0 };
                    } else if (coefficient != 1) {
                        // Independent complex multiplication, including 0*Inf.
                        expected = .{ .re = 2 * initial[index].re - 0 * initial[index].im, .im = 2 * initial[index].im + 0 * initial[index].re };
                    }
                    if (std.math.isNan(expected.re)) try std.testing.expect(std.math.isNan(actual[index].re)) else try std.testing.expectEqual(expected.re, actual[index].re);
                    if (std.math.isNan(expected.im)) try std.testing.expect(std.math.isNan(actual[index].im)) else try std.testing.expectEqual(expected.im, actual[index].im);
                } else {
                    expected = if (coefficient == 0) 0 else if (coefficient == 1) initial[index] else 2 * initial[index];
                    if (std.math.isNan(expected)) try std.testing.expect(std.math.isNan(actual[index])) else try std.testing.expectEqual(expected, actual[index]);
                }
            }
        }
    }
}

test "blocked rank-k and rank-2k preserve the unstored triangle and Hermitian diagonal" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
            for ([_]bool{ false, true }) |rank_two| {
                try runRankZeroProductEpilogue(T, uplo, false, rank_two);
                if (comptime scalar.isComplex(T)) try runRankZeroProductEpilogue(T, uplo, true, rank_two);
            }
        }
    }

    // The imaginary part of an existing Hermitian diagonal is not an input.
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        inline for (.{ false, true }) |rank_two| {
            for ([_]f64{ 0, 1 }) |alpha_real| {
                for ([_]f64{ 0, 1, 2 }) |beta_real| {
                    for ([_]f64{ std.math.nan(f64), std.math.inf(f64) }) |ignored_imaginary| {
                        const a = [_]T{if (alpha_real == 0) value(T, std.math.nan(f64), std.math.nan(f64)) else value(T, 1, 0)} ** 4;
                        for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
                            const prior = value(T, if (beta_real == 0) std.math.nan(f64) else 3, ignored_imaginary);
                            var actual = [_]T{prior} ** 4;
                            var ordinary = actual;
                            const alpha = value(T, alpha_real, 0);
                            const beta = value(T, beta_real, 0);
                            if (rank_two) {
                                try std.testing.expect(blocked.trySyr2k(T, .{ .block_size = 2, .direct_rank_panels = true }, uplo, .no_trans, 2, 2, alpha, &a, 2, &a, 2, beta, &actual, 2, true));
                                symmetric.syr2k(T, uplo, .no_trans, 2, 2, alpha, &a, 2, &a, 2, beta, &ordinary, 2, true);
                            } else {
                                try std.testing.expect(blocked.trySyrk(T, .{ .block_size = 2, .direct_rank_panels = true }, uplo, .no_trans, 2, 2, alpha, &a, 2, beta, &actual, 2, true));
                                symmetric.syrk(T, uplo, .no_trans, 2, 2, alpha, &a, 2, beta, &ordinary, 2, true);
                            }
                            const wanted = value(T, alpha_real * (if (rank_two) @as(f64, 4) else 2) + beta_real * 3, 0);
                            for ([_]usize{ 0, 3 }) |i| {
                                try std.testing.expectEqual(wanted.re, actual[i].re);
                                try std.testing.expectEqual(wanted.im, actual[i].im);
                                try std.testing.expectEqual(wanted.re, ordinary[i].re);
                                try std.testing.expectEqual(wanted.im, ordinary[i].im);
                            }
                        }
                    }
                }
            }
        }
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Uplo{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| {
            for ([_]scalar.Order{ scalar.Order.no_trans, scalar.Order.trans }) |trans| {
                try runRankCase(T, uplo, trans, false, false);
                try runRankCase(T, uplo, trans, false, true);
            }
        }
    }
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Uplo{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| {
            for ([_]scalar.Order{ scalar.Order.no_trans, scalar.Order.conj_trans }) |trans| {
                try runRankCase(T, uplo, trans, true, false);
                try runRankCase(T, uplo, trans, true, true);
            }
        }
    }
}

fn runTriangularCase(comptime T: type, solve: bool, side: scalar.Side, uplo: scalar.Uplo, trans: scalar.Order, diag: scalar.Diag) !void {
    return runTriangularCaseSized(T, 9, 7, solve, side, uplo, trans, diag);
}

noinline fn runTriangularCaseSized(comptime T: type, comptime m: usize, comptime n: usize, solve: bool, side: scalar.Side, uplo: scalar.Uplo, trans: scalar.Order, diag: scalar.Diag) !void {
    const order = if (side == .left) m else n;
    const lda: scalar.BlasInt = @intCast(order + 2);
    const ldb: scalar.BlasInt = @intCast(m + 3);
    var a: [(@as(usize, @max(m, n)) + 2) * @as(usize, @max(m, n))]T = undefined;
    var initial: [(m + 3) * n]T = undefined;
    for (&a, 0..) |*entry, index| entry.* = sample(T, index, 7);
    if (m >= 64) {
        for (&a) |*entry| entry.* = scalar.mul(T, entry.*, value(T, 1.0 / @as(f64, @floatFromInt(m)), 0));
    }
    for (0..order) |j| {
        const index = matIndex(lda, j, j);
        a[index] = if (m >= 64 and diag == .unit) value(T, std.math.nan(f64), std.math.nan(f64)) else value(T, 2.5 + @as(f64, @floatFromInt(j)) / 16, 0.0625);
    }
    for (&initial, 0..) |*entry, index| entry.* = sample(T, index, 8);
    var expected = initial;
    var actual = initial;
    const alpha = value(T, -0.625, 0.1875);

    if (solve) {
        if (m < 127) triangular.trsm(T, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &expected, ldb);
        try std.testing.expect(blocked.tryTrsm(T, .{ .block_size = if (m >= 64) 64 else 4 }, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &actual, ldb));
    } else {
        if (m < 127) triangular.trmm(T, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &expected, ldb);
        try std.testing.expect(blocked.tryTrmm(T, .{ .block_size = if (m >= 64) 64 else 4 }, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &actual, ldb));
    }
    if (m < 127) return expectSlicesClose(T, &expected, &actual);
    const check_selected = m <= 131;
    if (check_selected) {
        if (solve) {
            triangular.trsm(T, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &expected, ldb);
        } else {
            triangular.trmm(T, side, uplo, trans, diag, @intCast(m), @intCast(n), alpha, &a, lda, &expected, ldb);
        }
    }
    for (0..n) |j| {
        for (0..m) |i| {
            var sum = scalar.zero(T);
            var selected_sum = scalar.zero(T);
            for (0..order) |p| {
                const row = if (side == .left) i else p;
                const col = if (side == .left) p else j;
                const source_row = if (trans == .no_trans) row else col;
                const source_col = if (trans == .no_trans) col else row;
                if (if (uplo == .upper) source_row > source_col else source_row < source_col) continue;
                var av = if (row == col and diag == .unit) scalar.one(T) else a[matIndex(lda, source_row, source_col)];
                if (trans == .conj_trans) av = scalar.conj(T, av);
                const bindex = if (side == .left) matIndex(ldb, p, j) else matIndex(ldb, i, p);
                sum = scalar.add(T, sum, scalar.mul(T, av, if (solve) actual[bindex] else initial[bindex]));
                if (check_selected and solve) selected_sum = scalar.add(T, selected_sum, scalar.mul(T, av, expected[bindex]));
            }
            const index = matIndex(ldb, i, j);
            if (solve) {
                try expectClose(T, scalar.mul(T, alpha, initial[index]), sum);
                if (check_selected) try expectClose(T, scalar.mul(T, alpha, initial[index]), selected_sum);
            } else {
                try expectClose(T, scalar.mul(T, alpha, sum), actual[index]);
                if (check_selected) try expectClose(T, scalar.mul(T, alpha, sum), expected[index]);
            }
        }
        for (m..m + 3) |i| {
            try std.testing.expectEqual(initial[matIndex(ldb, i, j)], actual[matIndex(ldb, i, j)]);
            if (check_selected) try std.testing.expectEqual(initial[matIndex(ldb, i, j)], expected[matIndex(ldb, i, j)]);
        }
    }
}

test "queued triangular RHS panels preserve full scalar products and solve residuals" {
    const tuning = @import("kernels/tuning/structured.zig");
    const multiply = tuning.aarch64_macos_trmm_candidate;
    const solve = tuning.aarch64_macos_trsm_candidate;
    try std.testing.expect(!multiply.candidate(.f32, .right, false, 63, 128));
    try std.testing.expect(multiply.candidate(.f32, .right, true, 64, 64));
    try std.testing.expect(multiply.candidate(.f32, .right, false, 127, 128));
    try std.testing.expect(multiply.candidate(.f32, .left, true, 128, 511));
    try std.testing.expect(!multiply.candidate(.f32, .left, true, 128, 512));
    try std.testing.expect(multiply.candidate(.f32, .left, false, 128, 512));
    try std.testing.expect(solve.candidate(.f32, .right, true, 128, 512));
    try std.testing.expect(!solve.candidate(.f32, .left, false, 128, 128));
    try std.testing.expect(solve.candidate(.complex_f64, .left, true, 128, 128));
    try std.testing.expect(!solve.candidate(.complex_f64, .left, true, 128, 512));
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(4);
    defer {
        @import("core/execution/thread_pool.zig").shutdown();
        runtime.setMaxThreads(0);
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Side{ .left, .right }) |side| {
            for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
                for ([_]scalar.Order{ .no_trans, .trans, .conj_trans }) |trans| {
                    for ([_]scalar.Diag{ .unit, .non_unit }) |diag| {
                        try runTriangularCaseSized(T, 259, 257, false, side, uplo, trans, diag);
                        try runTriangularCaseSized(T, 259, 257, true, side, uplo, trans, diag);
                    }
                }
            }
        }
    }
}

fn testTriangularType(comptime T: type) !void {
    for ([_]scalar.Side{ scalar.Side.left, scalar.Side.right }) |side| {
        for ([_]scalar.Uplo{ scalar.Uplo.upper, scalar.Uplo.lower }) |uplo| {
            for ([_]scalar.Order{ scalar.Order.no_trans, scalar.Order.trans }) |trans| {
                for ([_]scalar.Diag{ scalar.Diag.non_unit, scalar.Diag.unit }) |diag| {
                    try runTriangularCase(T, false, side, uplo, trans, diag);
                    try runTriangularCase(T, true, side, uplo, trans, diag);
                }
            }
            if (comptime scalar.isComplex(T)) {
                for ([_]scalar.Diag{ scalar.Diag.non_unit, scalar.Diag.unit }) |diag| {
                    try runTriangularCase(T, false, side, uplo, .conj_trans, diag);
                    try runTriangularCase(T, true, side, uplo, .conj_trans, diag);
                }
            }
        }
    }
}

noinline fn runTriangularZeroCase(comptime T: type, comptime dimension: usize, solve: bool, side: scalar.Side) !void {
    const m: usize = dimension;
    const n: usize = dimension;
    const ldb: usize = m + 3;
    var a: [m * m]T = @splat(value(T, std.math.nan(f64), std.math.nan(f64)));
    var initial: [ldb * n]T = undefined;
    for (&initial, 0..) |*entry, index| entry.* = sample(T, index, 12);
    for (0..n) |j| {
        for (0..m) |i| initial[i + j * ldb] = value(T, std.math.nan(f64), std.math.nan(f64));
    }
    var actual = initial;
    var selected = initial;
    const zero = scalar.zero(T);
    if (solve) {
        try std.testing.expect(blocked.tryTrsm(T, .{ .block_size = 64 }, side, .upper, .conj_trans, .non_unit, m, n, zero, &a, m, &actual, ldb));
        triangular.trsm(T, side, .upper, .conj_trans, .non_unit, m, n, zero, &a, m, &selected, ldb);
    } else {
        try std.testing.expect(blocked.tryTrmm(T, .{ .block_size = 64 }, side, .upper, .conj_trans, .non_unit, m, n, zero, &a, m, &actual, ldb));
        triangular.trmm(T, side, .upper, .conj_trans, .non_unit, m, n, zero, &a, m, &selected, ldb);
    }
    for (0..n) |j| {
        for (0..ldb) |i| {
            const index = i + j * ldb;
            const want = if (i < m) zero else initial[index];
            try std.testing.expectEqual(want, actual[index]);
            try std.testing.expectEqual(want, selected[index]);
        }
    }
}

fn expectSpecialComponent(comptime R: type, expected: R, actual: R) !void {
    if (std.math.isNan(expected)) return std.testing.expect(std.math.isNan(actual));
    if (std.math.isInf(expected)) return std.testing.expectEqual(expected, actual);
    try expectClose(R, expected, actual);
}

noinline fn runTrmmNonfiniteCase(comptime T: type, side: scalar.Side, uplo: scalar.Uplo, trans: scalar.Order, diag: scalar.Diag, imaginary: bool) !void {
    const size: usize = 3;
    const ld: usize = 5;
    var a: [ld * size]T = @splat(value(T, 0.25, 0.125));
    for (0..size) |i| a[i + i * ld] = if (diag == .unit) value(T, std.math.nan(f64), std.math.nan(f64)) else value(T, 2, 0.125);
    var input: [ld * size]T = @splat(value(T, 2, 1));
    input[2 + 2 * ld] = if (imaginary) value(T, 1, std.math.inf(f64)) else value(T, std.math.inf(f64), 1);
    var output = input;
    try std.testing.expect(blocked.tryTrmm(T, .{ .block_size = 64 }, side, uplo, trans, diag, size, size, scalar.one(T), &a, ld, &output, ld));
    for (0..size) |j| {
        for (0..size) |i| {
            const pivot = if (side == .left) i else j;
            var expected = if (diag == .unit) input[i + j * ld] else scalar.zero(T);
            for (0..size) |p| {
                const row = if (side == .left) i else p;
                const col = if (side == .left) p else j;
                const ar = if (trans == .no_trans) row else col;
                const ac = if (trans == .no_trans) col else row;
                if (if (uplo == .upper) ar > ac else ar < ac) continue;
                if (diag == .unit and p == pivot) continue;
                const av = if (trans == .conj_trans) scalar.conj(T, a[ar + ac * ld]) else a[ar + ac * ld];
                const bv = input[if (side == .left) p + j * ld else i + p * ld];
                expected = scalar.add(T, expected, scalar.mul(T, av, bv));
            }
            if (comptime scalar.isComplex(T)) {
                try expectSpecialComponent(scalar.Real(T), expected.re, output[i + j * ld].re);
                try expectSpecialComponent(scalar.Real(T), expected.im, output[i + j * ld].im);
            } else try expectSpecialComponent(T, expected, output[i + j * ld]);
        }
        for (size..ld) |i| try std.testing.expectEqual(input[i + j * ld], output[i + j * ld]);
    }
}

test "blocked TRMM and TRSM preserve every side triangle transpose and diagonal mode" {
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Side{ .left, .right }) |side| {
            for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
                for ([_]scalar.Order{ .no_trans, .trans, .conj_trans }) |trans| {
                    for ([_]scalar.Diag{ .unit, .non_unit }) |diag| {
                        try runTrmmNonfiniteCase(T, side, uplo, trans, diag, false);
                        if (comptime scalar.isComplex(T)) try runTrmmNonfiniteCase(T, side, uplo, trans, diag, true);
                    }
                }
            }
        }
    }
    // A 1x1 unit solve must neither multiply by alpha=1 nor divide by a
    // synthesized complex one: either operation can introduce 0 * infinity.
    inline for (.{ scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        const a = [_]T{value(T, std.math.nan(f64), std.math.nan(f64))};
        for ([_]scalar.Side{ .left, .right }) |side| {
            for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
                for ([_]scalar.Order{ .no_trans, .trans, .conj_trans }) |trans| {
                    for ([_]bool{ false, true }) |imaginary_infinity| {
                        const initial = if (imaginary_infinity) value(T, 1, std.math.inf(f64)) else value(T, std.math.inf(f64), 1);
                        var b = [_]T{initial};
                        try std.testing.expect(blocked.tryTrsm(T, .{ .block_size = 2 }, side, uplo, trans, .unit, 1, 1, scalar.one(T), &a, 1, &b, 1));
                        try std.testing.expectEqual(initial.re, b[0].re);
                        try std.testing.expectEqual(initial.im, b[0].im);
                    }
                }
            }
        }
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Side{ .left, .right }) |side| {
            inline for (.{ @as(usize, 127), @as(usize, 128) }) |dimension| {
                try runTriangularZeroCase(T, dimension, false, side);
                try runTriangularZeroCase(T, dimension, true, side);
            }
        }
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| {
        for ([_]scalar.Side{ .left, .right }) |side| {
            for ([_]scalar.Uplo{ .upper, .lower }) |uplo| {
                for ([_]scalar.Order{ .no_trans, .trans, .conj_trans }) |trans| {
                    for ([_]scalar.Diag{ .unit, .non_unit }) |diag| {
                        if (side == .right) {
                            try runTriangularCaseSized(T, 64, 64, false, side, uplo, trans, diag);
                            try runTriangularCaseSized(T, 65, 65, false, side, uplo, trans, diag);
                        }
                        try runTriangularCaseSized(T, 129, 131, false, side, uplo, trans, diag);
                        try runTriangularCaseSized(T, 129, 131, true, side, uplo, trans, diag);
                    }
                }
            }
        }
        inline for (.{ false, true }) |solve| {
            try runTriangularCaseSized(T, 128, 128, solve, .left, .upper, .no_trans, .non_unit);
            try runTriangularCaseSized(T, 127, 129, solve, .right, .lower, .conj_trans, .unit);
        }
    }
    inline for (.{ f32, f64, scalar.ComplexF32, scalar.ComplexF64 }) |T| try testTriangularType(T);
}

test "blocked workspace rejection occurs before caller output changes" {
    const runtime = @import("runtime.zig");
    runtime.setMaxThreads(4);
    defer runtime.setMaxThreads(0);
    var a: [9 * 9]f64 = undefined;
    var b: [9 * 7]f64 = undefined;
    var c: [9 * 7]f64 = undefined;
    for (&a, 0..) |*entry, index| entry.* = sample(f64, index, 9);
    for (&b, 0..) |*entry, index| entry.* = sample(f64, index, 10);
    for (&c, 0..) |*entry, index| entry.* = sample(f64, index, 11);
    const original_b = b;
    const original_c = c;
    const parallel_options = blocked.Options{ .packed_rank_panels = true, .parallel_packed_rank_panels = true };
    try std.testing.expect(!blocked.tryParallelPackedRank(f64, false, parallel_options, .upper, .no_trans, 129, 0, 1, &a, 9, &b, 9, 0, &c, 9, false));
    try std.testing.expect(!blocked.tryParallelPackedRank(f64, false, parallel_options, .upper, .no_trans, 129, 7, 0, &a, 9, &b, 9, 0, &c, 9, false));
    try std.testing.expectEqualSlices(f64, &original_c, &c);
    for ([_]bool{ false, true }) |pool_available| {
        const denied_parallel = blocked.Options{ .block_size = 64, .packed_rank_panels = true, .parallel_packed_rank_panels = true, .workspace_available = !pool_available, .packed_rank_pool_available = pool_available };
        try std.testing.expect(!blocked.tryParallelPackedRank(f64, false, denied_parallel, .upper, .no_trans, 129, 7, 1, &a, 9, &b, 9, 1, &c, 9, false));
        try std.testing.expectEqualSlices(f64, &original_c, &c);
    }

    const denied = blocked.Options{ .block_size = 4, .workspace_available = false };
    const forced_options = blocked.Options{ .block_size = 64, .packed_rank_panels = true };
    // A disabled forced identity must reject even operations that only scale C.
    for ([_]scalar.BlasInt{ 0, 7 }) |reduction| {
        try std.testing.expect(!blocked.executePackedRank(f64, .dsyrk_packed_nn_parallel, forced_options, .upper, .no_trans, 7, reduction, 0, &a, 9, &b, 9, 0, &c, 9));
        try std.testing.expectEqualSlices(f64, &original_c, &c);
    }

    try std.testing.expect(!blocked.executePackedRank(f64, .ssyrk_packed_nn_panels, forced_options, .upper, .no_trans, 9, 7, 1, &a, 9, &b, 9, 1, &c, 9));
    try std.testing.expectEqualSlices(f64, &original_c, &c);
    try std.testing.expect(!blocked.executePackedRank(f64, .dsyrk_packed_nn_panels, forced_options, .upper, .conj_trans, 9, 7, 1, &a, 9, &b, 9, 1, &c, 9));
    try std.testing.expectEqualSlices(f64, &original_c, &c);
    const conflicting = blocked.Options{ .block_size = 64, .direct_rank_panels = true, .packed_rank_panels = true };
    try std.testing.expect(!blocked.trySyrk(f64, conflicting, .upper, .no_trans, 9, 7, 1, &a, 9, 1, &c, 9, false));
    try std.testing.expectEqualSlices(f64, &original_c, &c);
    const packed_denied = blocked.Options{ .block_size = 64, .packed_rank_panels = true, .workspace_available = false };
    try std.testing.expect(!blocked.trySyr2k(f64, packed_denied, .upper, .no_trans, 9, 7, 1, &a, 9, &b, 9, 1, &c, 9, false));
    try std.testing.expectEqualSlices(f64, &original_c, &c);
    var store_only = original_c;
    for (&store_only) |*entry| entry.* = std.math.nan(f64);
    try std.testing.expect(blocked.trySyrk(f64, packed_denied, .upper, .no_trans, 7, 7, 0, &a, 9, 0, &store_only, 9, false));
    for (0..7) |j| {
        for (0..9) |i| {
            if (i <= j) try std.testing.expectEqual(@as(f64, 0), store_only[i + j * 9]) else try std.testing.expect(std.math.isNan(store_only[i + j * 9]));
        }
    }
    try std.testing.expect(!blocked.trySymm(f64, denied, .left, .upper, 9, 7, 1, &a, 9, &b, 9, 1, &c, 9, false));
    try std.testing.expectEqualSlices(f64, &original_c, &c);
    try std.testing.expect(!blocked.trySyrk(f64, denied, .upper, .no_trans, 9, 7, 1, &a, 9, 1, &c, 9, false));
    try std.testing.expectEqualSlices(f64, &original_c, &c);
    try std.testing.expect(!blocked.tryTrmm(f64, denied, .left, .upper, .no_trans, .non_unit, 9, 7, 1, &a, 9, &b, 9));
    try std.testing.expectEqualSlices(f64, &original_b, &b);
    try std.testing.expect(!blocked.tryTrsm(f64, denied, .right, .lower, .trans, .non_unit, 9, 7, 1, &a, 9, &b, 9));
    try std.testing.expectEqualSlices(f64, &original_b, &b);
}

test "packing effective triangle follows transpose" {
    try std.testing.expectEqual(packing.Triangle.upper, packing.effectiveTriangle(.upper, .no_trans));
    try std.testing.expectEqual(packing.Triangle.lower, packing.effectiveTriangle(.upper, .trans));
    try std.testing.expectEqual(packing.Triangle.upper, packing.effectiveTriangle(.lower, .conj_trans));
}
