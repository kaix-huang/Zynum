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

fn TriangularMatrixFn(comptime T: type) type {
    return *const fn ([*]const u8, [*]const u8, [*]const u8, [*]const u8, *const BlasInt, *const BlasInt, *const T, [*]const T, *const BlasInt, [*]T, *const BlasInt) callconv(.c) void;
}

const Family = enum {
    trmm,
    trsm,
};

const Routine = enum {
    strmm,
    dtrmm,
    ctrmm,
    ztrmm,
    strsm,
    dtrsm,
    ctrsm,
    ztrsm,

    fn parse(value: []const u8) !Routine {
        inline for (std.meta.tags(Routine)) |routine| {
            if (std.ascii.eqlIgnoreCase(value, @tagName(routine))) return routine;
        }
        return error.BadRoutine;
    }

    fn kind(self: Routine) []const u8 {
        return switch (self) {
            .strmm, .strsm => "f32",
            .dtrmm, .dtrsm => "f64",
            .ctrmm, .ctrsm => "c32",
            .ztrmm, .ztrsm => "c64",
        };
    }

    fn family(self: Routine) Family {
        return switch (self) {
            .strmm, .dtrmm, .ctrmm, .ztrmm => .trmm,
            .strsm, .dtrsm, .ctrsm, .ztrsm => .trsm,
        };
    }

    fn isComplex(self: Routine) bool {
        return switch (self) {
            .strmm, .dtrmm, .strsm, .dtrsm => false,
            .ctrmm, .ztrmm, .ctrsm, .ztrsm => true,
        };
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
    m: usize,
    n: usize,
    side: u8,
    uplo: u8,
    trans: u8,
    diag: u8,
    alpha: ScalarSpec,
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
        "usage: triangular-matrix-probe --blas path --library label --routine strmm|dtrmm|ctrmm|ztrmm|strsm|dtrsm|ctrsm|ztrsm --m M --n N --side L|R --uplo U|L --trans N|T|C --diag N|U --alpha RE[,IM] [--shape label] [--reps count]\n",
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
    var m: ?usize = null;
    var n: ?usize = null;
    var side: ?u8 = null;
    var uplo: ?u8 = null;
    var trans: ?u8 = null;
    var diag: ?u8 = null;
    var alpha: ?ScalarSpec = null;
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
        } else if (std.mem.eql(u8, arg, "--m")) {
            m = try checkedDimension(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--n")) {
            n = try checkedDimension(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--side")) {
            side = try parseChar(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--uplo")) {
            uplo = try parseChar(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--trans")) {
            trans = try parseChar(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--diag")) {
            diag = try parseChar(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            alpha = try parseScalar(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--reps")) {
            reps = try std.fmt.parseInt(usize, args.next() orelse return error.MissingValue, 10);
        } else {
            usage();
            return error.BadArgument;
        }
    }

    const selected_routine = routine orelse return error.MissingRoutine;
    const selected_side = side orelse return error.MissingSide;
    const selected_uplo = uplo orelse return error.MissingUplo;
    const selected_trans = trans orelse return error.MissingTrans;
    const selected_diag = diag orelse return error.MissingDiag;
    const selected_alpha = alpha orelse return error.MissingAlpha;
    if (selected_side != 'L' and selected_side != 'R') return error.BadSide;
    if (selected_uplo != 'U' and selected_uplo != 'L') return error.BadUplo;
    if (selected_trans != 'N' and selected_trans != 'T' and selected_trans != 'C') return error.BadTrans;
    if (selected_diag != 'N' and selected_diag != 'U') return error.BadDiag;
    if (!selected_routine.isComplex() and selected_trans == 'C') return error.ConjugateTransposeNotAllowed;
    if (!selected_routine.isComplex() and selected_alpha.im != 0) return error.ComplexScalarNotAllowed;
    if (reps == 0) return error.InvalidRepetitions;

    return .{
        .blas_path = blas_path orelse return error.MissingBlas,
        .library = library orelse return error.MissingLibrary,
        .shape = shape orelse "custom",
        .routine = selected_routine,
        .m = m orelse return error.MissingM,
        .n = n orelse return error.MissingN,
        .side = selected_side,
        .uplo = selected_uplo,
        .trans = selected_trans,
        .diag = selected_diag,
        .alpha = selected_alpha,
        .reps = reps,
    };
}

fn nextFillValue(seed: *u64) f64 {
    seed.* = seed.* *% 6364136223846793005 +% 1442695040888963407;
    const bits: u32 = @truncate(seed.* >> 32);
    return @as(f64, @floatFromInt(bits % 1000)) / 1000.0 - 0.5;
}

fn isComplex(comptime T: type) bool {
    return T == ComplexF32 or T == ComplexF64;
}

fn fill(comptime T: type, values: []T, seed_value: u64) void {
    var seed = seed_value;
    for (values) |*value| {
        if (comptime isComplex(T)) {
            value.* = .{
                .re = @floatCast(nextFillValue(&seed)),
                .im = @floatCast(nextFillValue(&seed)),
            };
        } else {
            value.* = @floatCast(nextFillValue(&seed));
        }
    }
}

fn isStored(uplo: u8, row: usize, col: usize) bool {
    return if (uplo == 'U') row <= col else row >= col;
}

fn realValue(comptime T: type, value: f64) T {
    return if (comptime isComplex(T)) .{ .re = @floatCast(value), .im = 0 } else @floatCast(value);
}

fn poisonValue(comptime T: type, row: usize, col: usize) T {
    const value = 29.0 + @as(f64, @floatFromInt((row * 17 + col * 31) % 23));
    return if (comptime isComplex(T))
        .{ .re = @floatCast(value), .im = @floatCast(-value - 5.0) }
    else
        @floatCast(value);
}

fn prepareTriangularInput(comptime T: type, values: []T, order: usize, uplo: u8, diag: u8) void {
    fill(T, values, 0x3141_5926_5358_9793);
    for (0..order) |col| {
        for (0..order) |row| {
            const index = row + col * order;
            if (!isStored(uplo, row, col)) {
                values[index] = poisonValue(T, row, col);
            } else if (row == col) {
                if (diag == 'U') {
                    values[index] = poisonValue(T, row, col);
                } else if (comptime isComplex(T)) {
                    values[index] = .{
                        .re = @floatCast(1.25 + 0.03 * @as(f64, @floatFromInt(row % 7))),
                        .im = @floatCast(0.02 * @as(f64, @floatFromInt(@as(i32, @intCast(row % 5)) - 2))),
                    };
                } else {
                    values[index] = @floatCast(1.25 + 0.03 * @as(f64, @floatFromInt(row % 7)));
                }
            } else if (comptime isComplex(T)) {
                values[index].re *= 0.08;
                values[index].im *= 0.08;
            } else {
                values[index] *= 0.08;
            }
        }
    }
}

fn scalarValue(comptime T: type, spec: ScalarSpec) T {
    return if (comptime isComplex(T))
        .{ .re = @floatCast(spec.re), .im = @floatCast(spec.im) }
    else
        @floatCast(spec.re);
}

fn zero(comptime T: type) T {
    return realValue(T, 0);
}

fn one(comptime T: type) T {
    return realValue(T, 1);
}

fn add(comptime T: type, lhs: T, rhs: T) T {
    return if (comptime isComplex(T))
        .{ .re = lhs.re + rhs.re, .im = lhs.im + rhs.im }
    else
        lhs + rhs;
}

fn sub(comptime T: type, lhs: T, rhs: T) T {
    return if (comptime isComplex(T))
        .{ .re = lhs.re - rhs.re, .im = lhs.im - rhs.im }
    else
        lhs - rhs;
}

fn mul(comptime T: type, lhs: T, rhs: T) T {
    return if (comptime isComplex(T))
        .{
            .re = lhs.re * rhs.re - lhs.im * rhs.im,
            .im = lhs.re * rhs.im + lhs.im * rhs.re,
        }
    else
        lhs * rhs;
}

fn div(comptime T: type, lhs: T, rhs: T) T {
    if (comptime isComplex(T)) {
        const denominator = rhs.re * rhs.re + rhs.im * rhs.im;
        return .{
            .re = (lhs.re * rhs.re + lhs.im * rhs.im) / denominator,
            .im = (lhs.im * rhs.re - lhs.re * rhs.im) / denominator,
        };
    }
    return lhs / rhs;
}

fn conjugate(comptime T: type, value: T) T {
    return if (comptime isComplex(T)) .{ .re = value.re, .im = -value.im } else value;
}

fn absValue(comptime T: type, value: T) f64 {
    if (comptime isComplex(T)) {
        const re: f64 = @floatCast(value.re);
        const im: f64 = @floatCast(value.im);
        return @sqrt(re * re + im * im);
    }
    return @abs(@as(f64, @floatCast(value)));
}

fn absDiff(comptime T: type, lhs: T, rhs: T) f64 {
    return absValue(T, sub(T, lhs, rhs));
}

fn opElement(comptime T: type, a: []const T, lda: usize, options: Options, row: usize, col: usize) T {
    if (row == col and options.diag == 'U') return one(T);
    const transposed = options.trans != 'N';
    const source_row = if (transposed) col else row;
    const source_col = if (transposed) row else col;
    if (!isStored(options.uplo, source_row, source_col)) return zero(T);
    const value = a[source_row + source_col * lda];
    return if (options.trans == 'C') conjugate(T, value) else value;
}

fn opIsUpper(options: Options) bool {
    return (options.uplo == 'U') == (options.trans == 'N');
}

fn referenceTrmm(comptime T: type, a: []const T, b0: []const T, expected: []T, options: Options, lda: usize, ldb: usize) void {
    const order = if (options.side == 'L') options.m else options.n;
    const alpha = scalarValue(T, options.alpha);
    for (0..options.n) |col| {
        for (0..options.m) |row| {
            var sum = zero(T);
            for (0..order) |inner| {
                const term = if (options.side == 'L')
                    mul(T, opElement(T, a, lda, options, row, inner), b0[inner + col * ldb])
                else
                    mul(T, b0[row + inner * ldb], opElement(T, a, lda, options, inner, col));
                sum = add(T, sum, term);
            }
            expected[row + col * ldb] = mul(T, alpha, sum);
        }
    }
}

fn referenceTrsmLeft(comptime T: type, a: []const T, b0: []const T, expected: []T, options: Options, lda: usize, ldb: usize) void {
    const alpha = scalarValue(T, options.alpha);
    for (0..options.n) |col| {
        if (opIsUpper(options)) {
            var reverse = options.m;
            while (reverse > 0) {
                reverse -= 1;
                var value = mul(T, alpha, b0[reverse + col * ldb]);
                for (reverse + 1..options.m) |known| {
                    value = sub(T, value, mul(T, opElement(T, a, lda, options, reverse, known), expected[known + col * ldb]));
                }
                expected[reverse + col * ldb] = div(T, value, opElement(T, a, lda, options, reverse, reverse));
            }
        } else {
            for (0..options.m) |row| {
                var value = mul(T, alpha, b0[row + col * ldb]);
                for (0..row) |known| {
                    value = sub(T, value, mul(T, opElement(T, a, lda, options, row, known), expected[known + col * ldb]));
                }
                expected[row + col * ldb] = div(T, value, opElement(T, a, lda, options, row, row));
            }
        }
    }
}

fn referenceTrsmRight(comptime T: type, a: []const T, b0: []const T, expected: []T, options: Options, lda: usize, ldb: usize) void {
    const alpha = scalarValue(T, options.alpha);
    for (0..options.m) |row| {
        if (opIsUpper(options)) {
            for (0..options.n) |col| {
                var value = mul(T, alpha, b0[row + col * ldb]);
                for (0..col) |known| {
                    value = sub(T, value, mul(T, expected[row + known * ldb], opElement(T, a, lda, options, known, col)));
                }
                expected[row + col * ldb] = div(T, value, opElement(T, a, lda, options, col, col));
            }
        } else {
            var reverse = options.n;
            while (reverse > 0) {
                reverse -= 1;
                var value = mul(T, alpha, b0[row + reverse * ldb]);
                for (reverse + 1..options.n) |known| {
                    value = sub(T, value, mul(T, expected[row + known * ldb], opElement(T, a, lda, options, known, reverse)));
                }
                expected[row + reverse * ldb] = div(T, value, opElement(T, a, lda, options, reverse, reverse));
            }
        }
    }
}

fn referenceResult(comptime T: type, a: []const T, b0: []const T, expected: []T, options: Options, lda: usize, ldb: usize) void {
    switch (options.routine.family()) {
        .trmm => referenceTrmm(T, a, b0, expected, options, lda, ldb),
        .trsm => if (options.side == 'L')
            referenceTrsmLeft(T, a, b0, expected, options, lda, ldb)
        else
            referenceTrsmRight(T, a, b0, expected, options, lda, ldb),
    }
}

fn checkResult(comptime T: type, expected: []const T, actual: []const T, options: Options) CheckResult {
    var result = CheckResult{
        .status = "checked-ok",
        .max_abs_error = 0,
        .max_rel_error = 0,
        .samples = 0,
        .raw_output = "",
    };
    const order = if (options.side == 'L') options.m else options.n;
    const low_precision = T == f32 or T == ComplexF32;
    const alpha_scale = @max(1.0, @sqrt(options.alpha.re * options.alpha.re + options.alpha.im * options.alpha.im));
    const absolute_limit = (if (low_precision) 6e-5 else 8e-13) *
        @as(f64, @floatFromInt(order)) * alpha_scale;
    const relative_limit: f64 = if (low_precision) 2e-3 else 2e-11;

    for (expected, actual) |expected_value, actual_value| {
        const difference = absDiff(T, actual_value, expected_value);
        const expected_abs = absValue(T, expected_value);
        const relative = difference / @max(expected_abs, std.math.floatMin(f64));
        result.max_abs_error = if (std.math.isFinite(difference)) @max(result.max_abs_error, difference) else std.math.inf(f64);
        result.max_rel_error = if (std.math.isFinite(relative)) @max(result.max_rel_error, relative) else std.math.inf(f64);
        result.samples += 1;
        if (!std.math.isFinite(difference) or difference > absolute_limit + relative_limit * expected_abs) {
            result.status = "correctness_failed";
            result.raw_output = "full B reference tolerance exceeded; triangular, transpose, or unit-diagonal semantics may have been violated";
        }
    }
    return result;
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

fn benchTriangularMatrix(comptime T: type, function: TriangularMatrixFn(T), allocator: std.mem.Allocator, io: std.Io, options: Options) !BenchResult {
    const order = if (options.side == 'L') options.m else options.n;
    const lda = order;
    const ldb = options.m;
    const a_len = try std.math.mul(usize, lda, order);
    const matrix_len = try std.math.mul(usize, options.m, options.n);
    const a = try allocator.alloc(T, a_len);
    defer allocator.free(a);
    const b0 = try allocator.alloc(T, matrix_len);
    defer allocator.free(b0);
    const b = try allocator.alloc(T, matrix_len);
    defer allocator.free(b);
    const expected = try allocator.alloc(T, matrix_len);
    defer allocator.free(expected);
    const timings = try allocator.alloc(i96, options.reps);
    defer allocator.free(timings);

    prepareTriangularInput(T, a, order, options.uplo, options.diag);
    fill(T, b0, 0x1618_0339_8874_9894);
    referenceResult(T, a, b0, expected, options, lda, ldb);
    @memcpy(b, b0);

    var side = [_]u8{options.side};
    var uplo = [_]u8{options.uplo};
    var trans = [_]u8{options.trans};
    var diag = [_]u8{options.diag};
    const m: BlasInt = @intCast(options.m);
    const n: BlasInt = @intCast(options.n);
    const lda_i: BlasInt = @intCast(lda);
    const ldb_i: BlasInt = @intCast(ldb);
    var alpha = scalarValue(T, options.alpha);

    function(&side, &uplo, &trans, &diag, &m, &n, &alpha, a.ptr, &lda_i, b.ptr, &ldb_i);
    const check = checkResult(T, expected, b, options);

    for (0..options.reps) |rep| {
        @memcpy(b, b0);
        const start = std.Io.Clock.awake.now(io).nanoseconds;
        function(&side, &uplo, &trans, &diag, &m, &n, &alpha, a.ptr, &lda_i, b.ptr, &ldb_i);
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
    const order: u128 = if (options.side == 'L') options.m else options.n;
    const factor: u128 = if (options.routine.isComplex()) 4 else 1;
    return factor * @as(u128, options.m) * @as(u128, options.n) * order;
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
    const order = if (options.side == 'L') options.m else options.n;
    const flops = flopCount(options);
    try writer.writeAll("level3,");
    try writer.writeAll(@tagName(options.routine));
    try writer.writeByte(',');
    try writer.writeAll(@tagName(options.routine.family()));
    try writer.writeByte(',');
    try writer.writeAll(options.routine.kind());
    try writer.writeByte(',');
    try csvEscape(writer, options.library);
    try writer.writeByte(',');
    try csvEscape(writer, options.blas_path);
    try writer.writeByte(',');
    try csvEscape(writer, options.shape);
    try writer.print(",{d},{d},{c},{c},{c},{c},{d:.17},{d:.17},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d:.9},{d:.9},gflops,", .{
        options.m,
        options.n,
        options.side,
        options.uplo,
        options.trans,
        options.diag,
        options.alpha.re,
        options.alpha.im,
        order,
        order,
        options.m,
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
        .strmm => benchTriangularMatrix(f32, dyn.lookup(TriangularMatrixFn(f32), "strmm_") orelse return error.MissingSymbol, allocator, io, options),
        .dtrmm => benchTriangularMatrix(f64, dyn.lookup(TriangularMatrixFn(f64), "dtrmm_") orelse return error.MissingSymbol, allocator, io, options),
        .ctrmm => benchTriangularMatrix(ComplexF32, dyn.lookup(TriangularMatrixFn(ComplexF32), "ctrmm_") orelse return error.MissingSymbol, allocator, io, options),
        .ztrmm => benchTriangularMatrix(ComplexF64, dyn.lookup(TriangularMatrixFn(ComplexF64), "ztrmm_") orelse return error.MissingSymbol, allocator, io, options),
        .strsm => benchTriangularMatrix(f32, dyn.lookup(TriangularMatrixFn(f32), "strsm_") orelse return error.MissingSymbol, allocator, io, options),
        .dtrsm => benchTriangularMatrix(f64, dyn.lookup(TriangularMatrixFn(f64), "dtrsm_") orelse return error.MissingSymbol, allocator, io, options),
        .ctrsm => benchTriangularMatrix(ComplexF32, dyn.lookup(TriangularMatrixFn(ComplexF32), "ctrsm_") orelse return error.MissingSymbol, allocator, io, options),
        .ztrsm => benchTriangularMatrix(ComplexF64, dyn.lookup(TriangularMatrixFn(ComplexF64), "ztrsm_") orelse return error.MissingSymbol, allocator, io, options),
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const options = parseOptions(init, allocator) catch |err| {
        usage();
        return err;
    };
    // Keep short-lived threaded BLAS libraries mapped until process exit.
    var dyn = try std.DynLib.open(options.blas_path);
    const result = try runSelected(&dyn, allocator, init.io, options);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout_writer.interface.writeAll("level,routine,family,kind,library,library_path,shape,m,n,side,uplo,trans,diag,alpha_re,alpha_im,order,lda,ldb,reps,flop_count,best_ns,median_ns,p95_ns,max_ns,gflops,median_gflops,metric,status,check_status,check_max_abs_error,check_max_rel_error,check_samples,check_raw_output\n");
    try writeRow(&stdout_writer.interface, options, result);
    try stdout_writer.flush();
}

test "scalar parser accepts real and complex values" {
    try std.testing.expectEqual(ScalarSpec{ .re = 0.75 }, try parseScalar("0.75"));
    try std.testing.expectEqual(ScalarSpec{ .re = -0.5, .im = 0.125 }, try parseScalar("-0.5,0.125"));
    try std.testing.expectError(error.BadScalar, parseScalar("1,2,3"));
}

test "op element honors transpose conjugation and unit diagonal" {
    const matrix = [_]ComplexF64{
        .{ .re = 11, .im = 12 },
        .{ .re = 99, .im = 98 },
        .{ .re = 2, .im = 3 },
        .{ .re = 21, .im = 22 },
    };
    const options = Options{
        .blas_path = "unused",
        .library = "test",
        .shape = "tiny",
        .routine = .ztrmm,
        .m = 2,
        .n = 1,
        .side = 'L',
        .uplo = 'U',
        .trans = 'C',
        .diag = 'U',
        .alpha = .{ .re = 1 },
        .reps = 1,
    };
    try std.testing.expectEqual(ComplexF64{ .re = 1, .im = 0 }, opElement(ComplexF64, &matrix, 2, options, 0, 0));
    try std.testing.expectEqual(ComplexF64{ .re = 2, .im = -3 }, opElement(ComplexF64, &matrix, 2, options, 1, 0));
    try std.testing.expectEqual(ComplexF64{ .re = 0, .im = 0 }, opElement(ComplexF64, &matrix, 2, options, 0, 1));
}

test "right-side triangular solve reference satisfies a small upper system" {
    const a = [_]f64{ 2, 99, 3, 4 };
    const b = [_]f64{ 8, 22 };
    var expected = [_]f64{ 0, 0 };
    const options = Options{
        .blas_path = "unused",
        .library = "test",
        .shape = "tiny",
        .routine = .dtrsm,
        .m = 1,
        .n = 2,
        .side = 'R',
        .uplo = 'U',
        .trans = 'N',
        .diag = 'N',
        .alpha = .{ .re = 1 },
        .reps = 1,
    };
    referenceTrsmRight(f64, &a, &b, &expected, options, 2, 1);
    try std.testing.expectApproxEqAbs(4.0, expected[0], 1e-14);
    try std.testing.expectApproxEqAbs(2.5, expected[1], 1e-14);
}
