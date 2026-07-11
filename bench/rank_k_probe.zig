// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const BlasInt = i32;

const ComplexF32 = extern struct {
    re: f32,
    im: f32,
};

const ComplexF64 = extern struct {
    re: f64,
    im: f64,
};

fn RankKFn(comptime T: type, comptime Scalar: type) type {
    return *const fn ([*]const u8, [*]const u8, *const BlasInt, *const BlasInt, *const Scalar, [*]const T, *const BlasInt, *const Scalar, [*]T, *const BlasInt) callconv(.c) void;
}

fn Rank2KFn(comptime T: type, comptime AlphaScalar: type, comptime BetaScalar: type) type {
    return *const fn ([*]const u8, [*]const u8, *const BlasInt, *const BlasInt, *const AlphaScalar, [*]const T, *const BlasInt, [*]const T, *const BlasInt, *const BetaScalar, [*]T, *const BlasInt) callconv(.c) void;
}

const Routine = enum {
    ssyrk,
    dsyrk,
    csyrk,
    zsyrk,
    cherk,
    zherk,
    ssyr2k,
    dsyr2k,
    csyr2k,
    zsyr2k,
    cher2k,
    zher2k,

    fn parse(value: []const u8) !Routine {
        inline for (std.meta.tags(Routine)) |routine| {
            if (std.ascii.eqlIgnoreCase(value, @tagName(routine))) return routine;
        }
        return error.BadRoutine;
    }

    fn kind(self: Routine) []const u8 {
        return switch (self) {
            .ssyrk, .ssyr2k => "f32",
            .dsyrk, .dsyr2k => "f64",
            .csyrk, .cherk, .csyr2k, .cher2k => "c32",
            .zsyrk, .zherk, .zsyr2k, .zher2k => "c64",
        };
    }

    fn isHermitian(self: Routine) bool {
        return switch (self) {
            .cherk, .zherk, .cher2k, .zher2k => true,
            else => false,
        };
    }

    fn isRank2K(self: Routine) bool {
        return switch (self) {
            .ssyr2k, .dsyr2k, .csyr2k, .zsyr2k, .cher2k, .zher2k => true,
            else => false,
        };
    }

    fn isComplex(self: Routine) bool {
        return switch (self) {
            .ssyrk, .dsyrk, .ssyr2k, .dsyr2k => false,
            .csyrk, .zsyrk, .cherk, .zherk, .csyr2k, .zsyr2k, .cher2k, .zher2k => true,
        };
    }

    fn supportsTranspose(self: Routine, trans: u8) bool {
        return if (self.isHermitian())
            trans == 'N' or trans == 'C'
        else
            trans == 'N' or trans == 'T';
    }
};

const ScalarSpec = struct {
    re: f64,
    im: f64 = 0,
};

const Options = struct {
    blas_path: []const u8,
    library: []const u8,
    shape: []const u8,
    routine: Routine,
    n: usize,
    k: usize,
    uplo: u8,
    trans: u8,
    alpha: ScalarSpec,
    beta: ScalarSpec,
    reps: usize,
};

const CheckResult = struct {
    status: []const u8,
    max_abs_error: f64,
    max_rel_error: f64,
    samples: usize,
    raw_output: []const u8,
};

const BenchResult = struct {
    best_ns: i96,
    median_ns: i96,
    p95_ns: i96,
    max_ns: i96,
    check: CheckResult,
};

fn usage() void {
    std.debug.print(
        "usage: rank-k-probe --blas path --library label --routine ssyrk|dsyrk|csyrk|zsyrk|cherk|zherk|ssyr2k|dsyr2k|csyr2k|zsyr2k|cher2k|zher2k --n N --k K --uplo U|L --trans N|T|C --alpha RE[,IM] --beta RE[,IM] [--shape label] [--reps count]\n",
        .{},
    );
}

fn parseScalar(value: []const u8) !ScalarSpec {
    var parts = std.mem.splitScalar(u8, value, ',');
    const real = parts.next() orelse return error.BadScalar;
    if (real.len == 0) return error.BadScalar;
    const imaginary = parts.next();
    if (parts.next() != null) return error.BadScalar;
    return .{
        .re = try std.fmt.parseFloat(f64, real),
        .im = if (imaginary) |part| blk: {
            if (part.len == 0) return error.BadScalar;
            break :blk try std.fmt.parseFloat(f64, part);
        } else 0,
    };
}

fn parseChar(value: []const u8) !u8 {
    if (value.len != 1) return error.BadCharacter;
    return std.ascii.toUpper(value[0]);
}

fn checkedDimension(value: []const u8) !usize {
    const result = try std.fmt.parseInt(usize, value, 10);
    if (result == 0 or result > std.math.maxInt(BlasInt)) return error.BadDimension;
    return result;
}

fn parseOptions(init: std.process.Init, allocator: std.mem.Allocator) !Options {
    var blas_path: ?[]const u8 = null;
    var library: ?[]const u8 = null;
    var shape: ?[]const u8 = null;
    var routine: ?Routine = null;
    var n: ?usize = null;
    var k: ?usize = null;
    var uplo: ?u8 = null;
    var trans: ?u8 = null;
    var alpha: ?ScalarSpec = null;
    var beta: ?ScalarSpec = null;
    var reps: usize = 5;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--blas")) {
            blas_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--library")) {
            library = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--shape")) {
            shape = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--routine")) {
            routine = try Routine.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--n")) {
            n = try checkedDimension(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--k")) {
            k = try checkedDimension(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--uplo")) {
            uplo = try parseChar(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--trans")) {
            trans = try parseChar(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            alpha = try parseScalar(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--beta")) {
            beta = try parseScalar(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--reps")) {
            reps = try std.fmt.parseInt(usize, args.next() orelse return error.MissingValue, 10);
        } else {
            usage();
            return error.BadArgument;
        }
    }

    const selected_routine = routine orelse return error.MissingRoutine;
    const selected_uplo = uplo orelse return error.MissingUplo;
    const selected_trans = trans orelse return error.MissingTranspose;
    const selected_alpha = alpha orelse return error.MissingAlpha;
    const selected_beta = beta orelse return error.MissingBeta;
    if (selected_uplo != 'U' and selected_uplo != 'L') return error.BadUplo;
    if (!selected_routine.supportsTranspose(selected_trans)) return error.BadTranspose;
    if (reps == 0) return error.InvalidRepetitions;
    const alpha_must_be_real = !selected_routine.isComplex() or
        (selected_routine.isHermitian() and !selected_routine.isRank2K());
    const beta_must_be_real = !selected_routine.isComplex() or selected_routine.isHermitian();
    if ((alpha_must_be_real and selected_alpha.im != 0) or
        (beta_must_be_real and selected_beta.im != 0))
    {
        return error.ComplexScalarNotAllowed;
    }

    return .{
        .blas_path = blas_path orelse return error.MissingBlas,
        .library = library orelse return error.MissingLibrary,
        .shape = shape orelse "custom",
        .routine = selected_routine,
        .n = n orelse return error.MissingN,
        .k = k orelse return error.MissingK,
        .uplo = selected_uplo,
        .trans = selected_trans,
        .alpha = selected_alpha,
        .beta = selected_beta,
        .reps = reps,
    };
}

fn nextFillValue(seed: *u64) f64 {
    seed.* = seed.* *% 6364136223846793005 +% 1442695040888963407;
    const bits: u32 = @truncate(seed.* >> 32);
    return @as(f64, @floatFromInt(bits % 1000)) / 1000.0 - 0.5;
}

fn fill(comptime T: type, values: []T, seed_value: u64) void {
    var seed = seed_value;
    for (values) |*value| {
        if (T == ComplexF32 or T == ComplexF64) {
            value.* = .{
                .re = @floatCast(nextFillValue(&seed)),
                .im = @floatCast(nextFillValue(&seed)),
            };
        } else {
            value.* = @floatCast(nextFillValue(&seed));
        }
    }
}

fn scalarValue(comptime T: type, spec: ScalarSpec) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{ .re = @floatCast(spec.re), .im = @floatCast(spec.im) }
    else
        @floatCast(spec.re);
}

fn zero(comptime T: type) T {
    return if (T == ComplexF32 or T == ComplexF64) .{ .re = 0, .im = 0 } else 0;
}

fn add(comptime T: type, lhs: T, rhs: T) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{ .re = lhs.re + rhs.re, .im = lhs.im + rhs.im }
    else
        lhs + rhs;
}

fn mul(comptime T: type, lhs: T, rhs: T) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{
            .re = lhs.re * rhs.re - lhs.im * rhs.im,
            .im = lhs.re * rhs.im + lhs.im * rhs.re,
        }
    else
        lhs * rhs;
}

fn conjugate(comptime T: type, value: T) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{ .re = value.re, .im = -value.im }
    else
        value;
}

fn absValue(comptime T: type, value: T) f64 {
    if (T == ComplexF32 or T == ComplexF64) {
        const re: f64 = @floatCast(value.re);
        const im: f64 = @floatCast(value.im);
        return @sqrt(re * re + im * im);
    }
    return @abs(@as(f64, @floatCast(value)));
}

fn absDiff(comptime T: type, lhs: T, rhs: T) f64 {
    return absValue(T, add(T, lhs, if (T == ComplexF32 or T == ComplexF64)
        .{ .re = -rhs.re, .im = -rhs.im }
    else
        -rhs));
}

fn storedLeadingDimension(n: usize, k: usize, trans: u8) usize {
    return if (trans == 'N') n else k;
}

fn storedColumnCount(n: usize, k: usize, trans: u8) usize {
    return if (trans == 'N') k else n;
}

fn opElement(comptime T: type, matrix: []const T, lda: usize, trans: u8, row: usize, col: usize) T {
    const value = if (trans == 'N')
        matrix[row + col * lda]
    else
        matrix[col + row * lda];
    return if (trans == 'C') conjugate(T, value) else value;
}

fn expectedElement(comptime T: type, a: []const T, c0: []const T, options: Options, lda: usize, row: usize, col: usize) T {
    var sum = zero(T);
    for (0..options.k) |inner| {
        const left = opElement(T, a, lda, options.trans, row, inner);
        var right = opElement(T, a, lda, options.trans, col, inner);
        if (options.routine.isHermitian()) right = conjugate(T, right);
        sum = add(T, sum, mul(T, left, right));
    }
    const alpha = scalarValue(T, options.alpha);
    const beta = scalarValue(T, options.beta);
    var expected = add(T, mul(T, alpha, sum), mul(T, beta, c0[row + col * options.n]));
    if ((T == ComplexF32 or T == ComplexF64) and options.routine.isHermitian() and row == col) expected.im = 0;
    return expected;
}

fn expectedRank2KElement(comptime T: type, a: []const T, b: []const T, c0: []const T, options: Options, lda: usize, ldb: usize, row: usize, col: usize) T {
    var first_sum = zero(T);
    var second_sum = zero(T);
    for (0..options.k) |inner| {
        const a_row = opElement(T, a, lda, options.trans, row, inner);
        var a_col = opElement(T, a, lda, options.trans, col, inner);
        const b_row = opElement(T, b, ldb, options.trans, row, inner);
        var b_col = opElement(T, b, ldb, options.trans, col, inner);
        if (options.routine.isHermitian()) {
            a_col = conjugate(T, a_col);
            b_col = conjugate(T, b_col);
        }
        first_sum = add(T, first_sum, mul(T, a_row, b_col));
        second_sum = add(T, second_sum, mul(T, b_row, a_col));
    }

    const alpha = scalarValue(T, options.alpha);
    const second_alpha = if (options.routine.isHermitian()) conjugate(T, alpha) else alpha;
    const beta = scalarValue(T, options.beta);
    var expected = add(T, add(T, mul(T, alpha, first_sum), mul(T, second_alpha, second_sum)), mul(T, beta, c0[row + col * options.n]));
    if ((T == ComplexF32 or T == ComplexF64) and options.routine.isHermitian() and row == col) expected.im = 0;
    return expected;
}

fn isStored(uplo: u8, row: usize, col: usize) bool {
    return if (uplo == 'U') row <= col else row >= col;
}

fn sampleIndex(slot: usize, len: usize) usize {
    return switch (slot) {
        0 => 0,
        1 => len / 4,
        2 => len / 2,
        3 => (len - 1) - (len - 1) / 4,
        else => len - 1,
    };
}

fn checkElement(comptime T: type, a: []const T, b: ?[]const T, c0: []const T, actual: []const T, options: Options, lda: usize, ldb: usize, row: usize, col: usize, result: *CheckResult) void {
    const actual_value = actual[row + col * options.n];
    const expected = if (isStored(options.uplo, row, col))
        if (b) |matrix_b|
            expectedRank2KElement(T, a, matrix_b, c0, options, lda, ldb, row, col)
        else
            expectedElement(T, a, c0, options, lda, row, col)
    else
        c0[row + col * options.n];
    const difference = absDiff(T, actual_value, expected);
    const expected_abs = absValue(T, expected);
    const relative = difference / @max(expected_abs, std.math.floatMin(f64));
    result.max_abs_error = @max(result.max_abs_error, difference);
    result.max_rel_error = @max(result.max_rel_error, relative);
    result.samples += 1;

    const is_low_precision = T == f32 or T == ComplexF32;
    const alpha_scale = @max(1.0, @sqrt(options.alpha.re * options.alpha.re + options.alpha.im * options.alpha.im));
    const operation_factor: f64 = if (options.routine.isRank2K()) 2.0 else 1.0;
    const absolute_limit = (if (is_low_precision) 2e-5 else 2e-13) *
        @as(f64, @floatFromInt(@max(@as(usize, 1), options.k))) * alpha_scale * operation_factor;
    const relative_limit: f64 = if (is_low_precision) 5e-4 else 5e-12;
    const failed = if (isStored(options.uplo, row, col))
        difference > absolute_limit + relative_limit * expected_abs
    else
        difference != 0;
    if (failed) {
        result.status = "correctness_failed";
        result.raw_output = if (isStored(options.uplo, row, col))
            "reference tolerance exceeded"
        else
            "unstored triangle was modified";
    }
}

fn checkResultImpl(comptime T: type, a: []const T, b: ?[]const T, c0: []const T, actual: []const T, options: Options, lda: usize, ldb: usize) CheckResult {
    const full_check = options.n *| options.n <= 4096;
    var result = CheckResult{
        .status = if (full_check) "checked-ok" else "sampled-ok",
        .max_abs_error = 0,
        .max_rel_error = 0,
        .samples = 0,
        .raw_output = "",
    };
    if (full_check) {
        for (0..options.n) |col| {
            for (0..options.n) |row| {
                checkElement(T, a, b, c0, actual, options, lda, ldb, row, col, &result);
            }
        }
        return result;
    }

    var seen: [25]struct { row: usize, col: usize } = undefined;
    var seen_count: usize = 0;
    for (0..5) |row_slot| {
        const row = sampleIndex(row_slot, options.n);
        for (0..5) |col_slot| {
            const col = sampleIndex(col_slot, options.n);
            var duplicate = false;
            for (seen[0..seen_count]) |entry| {
                if (entry.row == row and entry.col == col) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            seen[seen_count] = .{ .row = row, .col = col };
            seen_count += 1;
            checkElement(T, a, b, c0, actual, options, lda, ldb, row, col, &result);
        }
    }
    return result;
}

fn checkResult(comptime T: type, a: []const T, c0: []const T, actual: []const T, options: Options, lda: usize) CheckResult {
    return checkResultImpl(T, a, null, c0, actual, options, lda, lda);
}

fn checkRank2KResult(comptime T: type, a: []const T, b: []const T, c0: []const T, actual: []const T, options: Options, lda: usize, ldb: usize) CheckResult {
    return checkResultImpl(T, a, b, c0, actual, options, lda, ldb);
}

fn sortTimings(values: []i96) void {
    var index: usize = 1;
    while (index < values.len) : (index += 1) {
        const value = values[index];
        var insertion = index;
        while (insertion > 0 and values[insertion - 1] > value) : (insertion -= 1) {
            values[insertion] = values[insertion - 1];
        }
        values[insertion] = value;
    }
}

fn benchRankK(comptime T: type, comptime Scalar: type, function: RankKFn(T, Scalar), allocator: std.mem.Allocator, io: std.Io, options: Options) !BenchResult {
    const lda = storedLeadingDimension(options.n, options.k, options.trans);
    const a_len = try std.math.mul(usize, lda, storedColumnCount(options.n, options.k, options.trans));
    const c_len = try std.math.mul(usize, options.n, options.n);
    const a = try allocator.alloc(T, a_len);
    defer allocator.free(a);
    const c0 = try allocator.alloc(T, c_len);
    defer allocator.free(c0);
    const c = try allocator.alloc(T, c_len);
    defer allocator.free(c);
    const timings = try allocator.alloc(i96, options.reps);
    defer allocator.free(timings);

    fill(T, a, 0x3141_5926_5358_9793);
    fill(T, c0, 0x2718_2818_2845_9045);
    @memcpy(c, c0);

    var uplo = [_]u8{options.uplo};
    var trans = [_]u8{options.trans};
    const n: BlasInt = @intCast(options.n);
    const k: BlasInt = @intCast(options.k);
    const lda_i: BlasInt = @intCast(lda);
    const ldc_i: BlasInt = @intCast(options.n);
    var alpha = scalarValue(Scalar, options.alpha);
    var beta = scalarValue(Scalar, options.beta);

    function(&uplo, &trans, &n, &k, &alpha, a.ptr, &lda_i, &beta, c.ptr, &ldc_i);
    const check = checkResult(T, a, c0, c, options, lda);

    for (0..options.reps) |rep| {
        @memcpy(c, c0);
        const start = std.Io.Clock.awake.now(io).nanoseconds;
        function(&uplo, &trans, &n, &k, &alpha, a.ptr, &lda_i, &beta, c.ptr, &ldc_i);
        const end = std.Io.Clock.awake.now(io).nanoseconds;
        timings[rep] = if (end > start) end - start else 1;
    }
    sortTimings(timings);
    const p95_index = @min(timings.len - 1, ((timings.len * 95) + 99) / 100 - 1);
    return .{
        .best_ns = timings[0],
        .median_ns = timings[timings.len / 2],
        .p95_ns = timings[p95_index],
        .max_ns = timings[timings.len - 1],
        .check = check,
    };
}

fn benchRank2K(comptime T: type, comptime AlphaScalar: type, comptime BetaScalar: type, function: Rank2KFn(T, AlphaScalar, BetaScalar), allocator: std.mem.Allocator, io: std.Io, options: Options) !BenchResult {
    const lda = storedLeadingDimension(options.n, options.k, options.trans);
    const ldb = storedLeadingDimension(options.n, options.k, options.trans);
    const matrix_len = try std.math.mul(usize, lda, storedColumnCount(options.n, options.k, options.trans));
    const c_len = try std.math.mul(usize, options.n, options.n);
    const a = try allocator.alloc(T, matrix_len);
    defer allocator.free(a);
    const b = try allocator.alloc(T, matrix_len);
    defer allocator.free(b);
    const c0 = try allocator.alloc(T, c_len);
    defer allocator.free(c0);
    const c = try allocator.alloc(T, c_len);
    defer allocator.free(c);
    const timings = try allocator.alloc(i96, options.reps);
    defer allocator.free(timings);

    fill(T, a, 0x3141_5926_5358_9793);
    fill(T, b, 0x1618_0339_8874_9894);
    fill(T, c0, 0x2718_2818_2845_9045);
    @memcpy(c, c0);

    var uplo = [_]u8{options.uplo};
    var trans = [_]u8{options.trans};
    const n: BlasInt = @intCast(options.n);
    const k: BlasInt = @intCast(options.k);
    const lda_i: BlasInt = @intCast(lda);
    const ldb_i: BlasInt = @intCast(ldb);
    const ldc_i: BlasInt = @intCast(options.n);
    var alpha = scalarValue(AlphaScalar, options.alpha);
    var beta = scalarValue(BetaScalar, options.beta);

    function(&uplo, &trans, &n, &k, &alpha, a.ptr, &lda_i, b.ptr, &ldb_i, &beta, c.ptr, &ldc_i);
    const check = checkRank2KResult(T, a, b, c0, c, options, lda, ldb);

    for (0..options.reps) |rep| {
        @memcpy(c, c0);
        const start = std.Io.Clock.awake.now(io).nanoseconds;
        function(&uplo, &trans, &n, &k, &alpha, a.ptr, &lda_i, b.ptr, &ldb_i, &beta, c.ptr, &ldc_i);
        const end = std.Io.Clock.awake.now(io).nanoseconds;
        timings[rep] = if (end > start) end - start else 1;
    }
    sortTimings(timings);
    const p95_index = @min(timings.len - 1, ((timings.len * 95) + 99) / 100 - 1);
    return .{
        .best_ns = timings[0],
        .median_ns = timings[timings.len / 2],
        .p95_ns = timings[p95_index],
        .max_ns = timings[timings.len - 1],
        .check = check,
    };
}

fn flopCount(options: Options) u128 {
    const triangle = @as(u128, options.n) * @as(u128, options.n + 1) / 2;
    const rank_factor: u128 = if (options.routine.isRank2K()) 2 else 1;
    const arithmetic_factor: u128 = if (options.routine.isComplex()) 8 else 2;
    const factor = rank_factor * arithmetic_factor;
    return factor * triangle * @as(u128, options.k);
}

fn gflops(flops: u128, elapsed_ns: i96) f64 {
    if (elapsed_ns <= 0) return 0;
    return @as(f64, @floatFromInt(flops)) / @as(f64, @floatFromInt(elapsed_ns));
}

fn csvEscape(writer: *std.Io.Writer, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |character| {
        if (character == '"') try writer.writeByte('"');
        try writer.writeByte(character);
    }
    try writer.writeByte('"');
}

fn writeRow(writer: *std.Io.Writer, options: Options, result: BenchResult) !void {
    const lda = storedLeadingDimension(options.n, options.k, options.trans);
    const flops = flopCount(options);
    try writer.writeAll("level3,");
    try writer.writeAll(@tagName(options.routine));
    try writer.writeByte(',');
    try writer.writeAll(options.routine.kind());
    try writer.writeByte(',');
    try csvEscape(writer, options.library);
    try writer.writeByte(',');
    try csvEscape(writer, options.blas_path);
    try writer.writeByte(',');
    try csvEscape(writer, options.shape);
    try writer.print(",{d},{d},{c},{c},{d:.17},{d:.17},{d:.17},{d:.17},{d},", .{
        options.n,
        options.k,
        options.uplo,
        options.trans,
        options.alpha.re,
        options.alpha.im,
        options.beta.re,
        options.beta.im,
        lda,
    });
    if (options.routine.isRank2K()) try writer.print("{d}", .{lda});
    try writer.print(",{d},{d},{d},{d},{d},{d},{d},{d:.9},{d:.9},gflops,", .{
        options.n,
        options.reps,
        flops,
        result.best_ns,
        result.median_ns,
        result.p95_ns,
        result.max_ns,
        gflops(flops, result.best_ns),
        gflops(flops, result.median_ns),
    });
    try writer.writeAll(if (std.mem.eql(u8, result.check.status, "correctness_failed")) "correctness_failed," else "ok,");
    try writer.writeAll(result.check.status);
    try writer.print(",{d:.9},{d:.9},{d},", .{
        result.check.max_abs_error,
        result.check.max_rel_error,
        result.check.samples,
    });
    try csvEscape(writer, result.check.raw_output);
    try writer.writeByte('\n');
}

fn runSelected(dyn: *std.DynLib, allocator: std.mem.Allocator, io: std.Io, options: Options) !BenchResult {
    return switch (options.routine) {
        .ssyrk => benchRankK(f32, f32, dyn.lookup(RankKFn(f32, f32), "ssyrk_") orelse return error.MissingSymbol, allocator, io, options),
        .dsyrk => benchRankK(f64, f64, dyn.lookup(RankKFn(f64, f64), "dsyrk_") orelse return error.MissingSymbol, allocator, io, options),
        .csyrk => benchRankK(ComplexF32, ComplexF32, dyn.lookup(RankKFn(ComplexF32, ComplexF32), "csyrk_") orelse return error.MissingSymbol, allocator, io, options),
        .zsyrk => benchRankK(ComplexF64, ComplexF64, dyn.lookup(RankKFn(ComplexF64, ComplexF64), "zsyrk_") orelse return error.MissingSymbol, allocator, io, options),
        .cherk => benchRankK(ComplexF32, f32, dyn.lookup(RankKFn(ComplexF32, f32), "cherk_") orelse return error.MissingSymbol, allocator, io, options),
        .zherk => benchRankK(ComplexF64, f64, dyn.lookup(RankKFn(ComplexF64, f64), "zherk_") orelse return error.MissingSymbol, allocator, io, options),
        .ssyr2k => benchRank2K(f32, f32, f32, dyn.lookup(Rank2KFn(f32, f32, f32), "ssyr2k_") orelse return error.MissingSymbol, allocator, io, options),
        .dsyr2k => benchRank2K(f64, f64, f64, dyn.lookup(Rank2KFn(f64, f64, f64), "dsyr2k_") orelse return error.MissingSymbol, allocator, io, options),
        .csyr2k => benchRank2K(ComplexF32, ComplexF32, ComplexF32, dyn.lookup(Rank2KFn(ComplexF32, ComplexF32, ComplexF32), "csyr2k_") orelse return error.MissingSymbol, allocator, io, options),
        .zsyr2k => benchRank2K(ComplexF64, ComplexF64, ComplexF64, dyn.lookup(Rank2KFn(ComplexF64, ComplexF64, ComplexF64), "zsyr2k_") orelse return error.MissingSymbol, allocator, io, options),
        .cher2k => benchRank2K(ComplexF32, ComplexF32, f32, dyn.lookup(Rank2KFn(ComplexF32, ComplexF32, f32), "cher2k_") orelse return error.MissingSymbol, allocator, io, options),
        .zher2k => benchRank2K(ComplexF64, ComplexF64, f64, dyn.lookup(Rank2KFn(ComplexF64, ComplexF64, f64), "zher2k_") orelse return error.MissingSymbol, allocator, io, options),
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const options = parseOptions(init, allocator) catch |err| {
        usage();
        return err;
    };
    var dyn = try std.DynLib.open(options.blas_path);
    const result = try runSelected(&dyn, allocator, init.io, options);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout_writer.interface.writeAll("level,routine,kind,library,library_path,shape,n,k,uplo,trans,alpha_re,alpha_im,beta_re,beta_im,lda,ldb,ldc,reps,flop_count,best_ns,median_ns,p95_ns,max_ns,gflops,median_gflops,metric,status,check_status,check_max_abs_error,check_max_rel_error,check_samples,check_raw_output\n");
    try writeRow(&stdout_writer.interface, options, result);
    try stdout_writer.flush();
}

test "routine transpose restrictions follow SYRK and HERK contracts" {
    try std.testing.expect(Routine.ssyrk.supportsTranspose('N'));
    try std.testing.expect(Routine.ssyrk.supportsTranspose('T'));
    try std.testing.expect(!Routine.ssyrk.supportsTranspose('C'));
    try std.testing.expect(Routine.zherk.supportsTranspose('N'));
    try std.testing.expect(Routine.zherk.supportsTranspose('C'));
    try std.testing.expect(!Routine.zherk.supportsTranspose('T'));
    try std.testing.expect(Routine.dsyr2k.supportsTranspose('T'));
    try std.testing.expect(!Routine.dsyr2k.supportsTranspose('C'));
    try std.testing.expect(Routine.zher2k.supportsTranspose('C'));
    try std.testing.expect(!Routine.zher2k.supportsTranspose('T'));
}

test "scalar parser accepts real and complex values" {
    try std.testing.expectEqual(ScalarSpec{ .re = 0.75 }, try parseScalar("0.75"));
    try std.testing.expectEqual(ScalarSpec{ .re = -0.5, .im = 0.125 }, try parseScalar("-0.5,0.125"));
    try std.testing.expectError(error.BadScalar, parseScalar("1,2,3"));
}

test "rank-k reference distinguishes symmetric and Hermitian products" {
    const a = [_]ComplexF64{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = -1 },
    };
    const c0 = [_]ComplexF64{
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
    };
    const base = Options{
        .blas_path = "unused",
        .library = "unused",
        .shape = "tiny",
        .routine = .zsyrk,
        .n = 2,
        .k = 1,
        .uplo = 'U',
        .trans = 'N',
        .alpha = .{ .re = 1 },
        .beta = .{ .re = 0 },
        .reps = 1,
    };
    try std.testing.expectEqual(ComplexF64{ .re = 3, .im = 1 }, expectedElement(ComplexF64, &a, &c0, base, 2, 0, 1));
    var hermitian = base;
    hermitian.routine = .zherk;
    try std.testing.expectEqual(ComplexF64{ .re = 1, .im = 3 }, expectedElement(ComplexF64, &a, &c0, hermitian, 2, 0, 1));
}

test "rank-2k reference uses independent B and conjugated HER2K alpha" {
    const a = [_]ComplexF64{
        .{ .re = 1, .im = 1 },
        .{ .re = 2, .im = -1 },
    };
    const b = [_]ComplexF64{
        .{ .re = 3, .im = 2 },
        .{ .re = -1, .im = 4 },
    };
    const c0 = [_]ComplexF64{
        .{ .re = 0, .im = 5 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = 0 },
        .{ .re = 0, .im = -7 },
    };
    const base = Options{
        .blas_path = "unused",
        .library = "unused",
        .shape = "tiny",
        .routine = .zsyr2k,
        .n = 2,
        .k = 1,
        .uplo = 'U',
        .trans = 'N',
        .alpha = .{ .re = 1 },
        .beta = .{ .re = 0 },
        .reps = 1,
    };
    try std.testing.expectEqual(ComplexF64{ .re = 3, .im = 4 }, expectedRank2KElement(ComplexF64, &a, &b, &c0, base, 2, 2, 0, 1));

    var hermitian = base;
    hermitian.routine = .zher2k;
    hermitian.alpha = .{ .re = 2, .im = 1 };
    try std.testing.expectEqual(ComplexF64{ .re = 26, .im = 3 }, expectedRank2KElement(ComplexF64, &a, &b, &c0, hermitian, 2, 2, 0, 1));
    const diagonal = expectedRank2KElement(ComplexF64, &a, &b, &c0, hermitian, 2, 2, 0, 0);
    try std.testing.expectEqual(@as(f64, 0), diagonal.im);
}
