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

fn SymmFn(comptime T: type) type {
    return *const fn ([*]const u8, [*]const u8, *const BlasInt, *const BlasInt, *const T, [*]const T, *const BlasInt, [*]const T, *const BlasInt, *const T, [*]T, *const BlasInt) callconv(.c) void;
}

const Routine = enum {
    ssymm,
    dsymm,
    csymm,
    zsymm,
    chemm,
    zhemm,

    fn parse(value: []const u8) !Routine {
        inline for (std.meta.tags(Routine)) |routine| {
            if (std.ascii.eqlIgnoreCase(value, @tagName(routine))) return routine;
        }
        return error.BadRoutine;
    }

    fn kind(self: Routine) []const u8 {
        return switch (self) {
            .ssymm => "f32",
            .dsymm => "f64",
            .csymm, .chemm => "c32",
            .zsymm, .zhemm => "c64",
        };
    }

    fn isComplex(self: Routine) bool {
        return switch (self) {
            .ssymm, .dsymm => false,
            .csymm, .zsymm, .chemm, .zhemm => true,
        };
    }

    fn isHermitian(self: Routine) bool {
        return self == .chemm or self == .zhemm;
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
        "usage: symm-probe --blas path --library label --routine ssymm|dsymm|csymm|zsymm|chemm|zhemm --m M --n N --side L|R --uplo U|L --alpha RE[,IM] --beta RE[,IM] [--shape label] [--reps count]\n",
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
        } else if (std.mem.eql(u8, arg, "--m")) {
            m = try checkedDimension(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--n")) {
            n = try checkedDimension(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--side")) {
            side = try parseChar(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--uplo")) {
            uplo = try parseChar(args.next() orelse return error.MissingValue);
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
    const selected_side = side orelse return error.MissingSide;
    const selected_uplo = uplo orelse return error.MissingUplo;
    const selected_alpha = alpha orelse return error.MissingAlpha;
    const selected_beta = beta orelse return error.MissingBeta;
    if (selected_side != 'L' and selected_side != 'R') return error.BadSide;
    if (selected_uplo != 'U' and selected_uplo != 'L') return error.BadUplo;
    if (reps == 0) return error.InvalidRepetitions;
    if (!selected_routine.isComplex() and (selected_alpha.im != 0 or selected_beta.im != 0)) {
        return error.ComplexScalarNotAllowed;
    }

    return .{
        .blas_path = blas_path orelse return error.MissingBlas,
        .library = library orelse return error.MissingLibrary,
        .shape = shape orelse "custom",
        .routine = selected_routine,
        .m = m orelse return error.MissingM,
        .n = n orelse return error.MissingN,
        .side = selected_side,
        .uplo = selected_uplo,
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

fn isStored(uplo: u8, row: usize, col: usize) bool {
    return if (uplo == 'U') row <= col else row >= col;
}

fn poisonValue(comptime T: type, row: usize, col: usize) T {
    const real = 32.0 + @as(f64, @floatFromInt((row * 17 + col * 29) % 31));
    if (T == ComplexF32 or T == ComplexF64) {
        return .{
            .re = @floatCast(real),
            .im = @floatCast(-real - 7.0),
        };
    }
    return @floatCast(real);
}

fn prepareStructuredInput(comptime T: type, values: []T, order: usize, uplo: u8, hermitian: bool) void {
    fill(T, values, 0x3141_5926_5358_9793);
    for (0..order) |col| {
        for (0..order) |row| {
            if (!isStored(uplo, row, col)) values[row + col * order] = poisonValue(T, row, col);
            if ((T == ComplexF32 or T == ComplexF64) and hermitian and row == col) {
                values[row + col * order].im = @floatCast(71.0 + @as(f64, @floatFromInt(row % 13)));
            }
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

fn structuredElement(comptime T: type, a: []const T, lda: usize, options: Options, row: usize, col: usize) T {
    const direct = isStored(options.uplo, row, col);
    var value = if (direct) a[row + col * lda] else a[col + row * lda];
    if ((T == ComplexF32 or T == ComplexF64) and options.routine.isHermitian()) {
        if (row == col) {
            value.im = 0;
        } else if (!direct) {
            value = conjugate(T, value);
        }
    }
    return value;
}

fn expectedElement(comptime T: type, a: []const T, b: []const T, c0: []const T, options: Options, lda: usize, ldb: usize, ldc: usize, row: usize, col: usize) T {
    const order = if (options.side == 'L') options.m else options.n;
    var sum = zero(T);
    for (0..order) |inner| {
        const product = if (options.side == 'L')
            mul(T, structuredElement(T, a, lda, options, row, inner), b[inner + col * ldb])
        else
            mul(T, b[row + inner * ldb], structuredElement(T, a, lda, options, inner, col));
        sum = add(T, sum, product);
    }
    const alpha = scalarValue(T, options.alpha);
    const beta = scalarValue(T, options.beta);
    return add(T, mul(T, alpha, sum), mul(T, beta, c0[row + col * ldc]));
}

fn checkResult(comptime T: type, a: []const T, b: []const T, c0: []const T, actual: []const T, options: Options, lda: usize, ldb: usize, ldc: usize) CheckResult {
    var result = CheckResult{
        .status = "checked-ok",
        .max_abs_error = 0,
        .max_rel_error = 0,
        .samples = 0,
        .raw_output = "",
    };
    const order = if (options.side == 'L') options.m else options.n;
    const is_low_precision = T == f32 or T == ComplexF32;
    const alpha_scale = @max(1.0, @sqrt(options.alpha.re * options.alpha.re + options.alpha.im * options.alpha.im));
    const absolute_limit = (if (is_low_precision) 3e-5 else 3e-13) *
        @as(f64, @floatFromInt(order)) * alpha_scale;
    const relative_limit: f64 = if (is_low_precision) 8e-4 else 8e-12;

    for (0..options.n) |col| {
        for (0..options.m) |row| {
            const expected = expectedElement(T, a, b, c0, options, lda, ldb, ldc, row, col);
            const actual_value = actual[row + col * ldc];
            const difference = absDiff(T, actual_value, expected);
            const expected_abs = absValue(T, expected);
            const relative = difference / @max(expected_abs, std.math.floatMin(f64));
            if (std.math.isFinite(difference)) result.max_abs_error = @max(result.max_abs_error, difference) else result.max_abs_error = std.math.inf(f64);
            if (std.math.isFinite(relative)) result.max_rel_error = @max(result.max_rel_error, relative) else result.max_rel_error = std.math.inf(f64);
            result.samples += 1;
            if (!std.math.isFinite(difference) or difference > absolute_limit + relative_limit * expected_abs) {
                result.status = "correctness_failed";
                result.raw_output = "full C reference tolerance exceeded; structured A semantics may have been violated";
            }
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

fn benchSymm(comptime T: type, function: SymmFn(T), allocator: std.mem.Allocator, io: std.Io, options: Options) !BenchResult {
    const order = if (options.side == 'L') options.m else options.n;
    const lda = order;
    const ldb = options.m;
    const ldc = options.m;
    const a_len = try std.math.mul(usize, lda, order);
    const matrix_len = try std.math.mul(usize, options.m, options.n);
    const a = try allocator.alloc(T, a_len);
    defer allocator.free(a);
    const b = try allocator.alloc(T, matrix_len);
    defer allocator.free(b);
    const c0 = try allocator.alloc(T, matrix_len);
    defer allocator.free(c0);
    const c = try allocator.alloc(T, matrix_len);
    defer allocator.free(c);
    const timings = try allocator.alloc(i96, options.reps);
    defer allocator.free(timings);

    prepareStructuredInput(T, a, order, options.uplo, options.routine.isHermitian());
    fill(T, b, 0x1618_0339_8874_9894);
    fill(T, c0, 0x2718_2818_2845_9045);
    @memcpy(c, c0);

    var side = [_]u8{options.side};
    var uplo = [_]u8{options.uplo};
    const m: BlasInt = @intCast(options.m);
    const n: BlasInt = @intCast(options.n);
    const lda_i: BlasInt = @intCast(lda);
    const ldb_i: BlasInt = @intCast(ldb);
    const ldc_i: BlasInt = @intCast(ldc);
    var alpha = scalarValue(T, options.alpha);
    var beta = scalarValue(T, options.beta);

    function(&side, &uplo, &m, &n, &alpha, a.ptr, &lda_i, b.ptr, &ldb_i, &beta, c.ptr, &ldc_i);
    const check = checkResult(T, a, b, c0, c, options, lda, ldb, ldc);

    for (0..options.reps) |rep| {
        @memcpy(c, c0);
        const start = std.Io.Clock.awake.now(io).nanoseconds;
        function(&side, &uplo, &m, &n, &alpha, a.ptr, &lda_i, b.ptr, &ldb_i, &beta, c.ptr, &ldc_i);
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
    const factor: u128 = if (options.routine.isComplex()) 8 else 2;
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
    try writer.writeAll(options.routine.kind());
    try writer.writeByte(',');
    try csvEscape(writer, options.library);
    try writer.writeByte(',');
    try csvEscape(writer, options.blas_path);
    try writer.writeByte(',');
    try csvEscape(writer, options.shape);
    try writer.print(",{d},{d},{c},{c},{d:.17},{d:.17},{d:.17},{d:.17},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d:.9},{d:.9},gflops,", .{
        options.m,
        options.n,
        options.side,
        options.uplo,
        options.alpha.re,
        options.alpha.im,
        options.beta.re,
        options.beta.im,
        order,
        order,
        options.m,
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
        .ssymm => benchSymm(f32, dyn.lookup(SymmFn(f32), "ssymm_") orelse return error.MissingSymbol, allocator, io, options),
        .dsymm => benchSymm(f64, dyn.lookup(SymmFn(f64), "dsymm_") orelse return error.MissingSymbol, allocator, io, options),
        .csymm => benchSymm(ComplexF32, dyn.lookup(SymmFn(ComplexF32), "csymm_") orelse return error.MissingSymbol, allocator, io, options),
        .zsymm => benchSymm(ComplexF64, dyn.lookup(SymmFn(ComplexF64), "zsymm_") orelse return error.MissingSymbol, allocator, io, options),
        .chemm => benchSymm(ComplexF32, dyn.lookup(SymmFn(ComplexF32), "chemm_") orelse return error.MissingSymbol, allocator, io, options),
        .zhemm => benchSymm(ComplexF64, dyn.lookup(SymmFn(ComplexF64), "zhemm_") orelse return error.MissingSymbol, allocator, io, options),
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const options = parseOptions(init, allocator) catch |err| {
        usage();
        return err;
    };
    // A short worker leaves BLAS mapped until process exit. Some threaded BLAS
    // libraries have destructors that are unsafe after an explicit dlclose.
    var dyn = try std.DynLib.open(options.blas_path);
    const result = try runSelected(&dyn, allocator, init.io, options);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout_writer.interface.writeAll("level,routine,kind,library,library_path,shape,m,n,side,uplo,alpha_re,alpha_im,beta_re,beta_im,order,lda,ldb,ldc,reps,flop_count,best_ns,median_ns,p95_ns,max_ns,gflops,median_gflops,metric,status,check_status,check_max_abs_error,check_max_rel_error,check_samples,check_raw_output\n");
    try writeRow(&stdout_writer.interface, options, result);
    try stdout_writer.flush();
}

test "scalar parser accepts real and complex values" {
    try std.testing.expectEqual(ScalarSpec{ .re = 0.75 }, try parseScalar("0.75"));
    try std.testing.expectEqual(ScalarSpec{ .re = -0.5, .im = 0.125 }, try parseScalar("-0.5,0.125"));
    try std.testing.expectError(error.BadScalar, parseScalar("1,2,3"));
}

test "structured lookup ignores unstored triangle and Hermitian diagonal imaginary part" {
    const matrix = [_]ComplexF64{
        .{ .re = 1, .im = 9 },
        .{ .re = 99, .im = 88 },
        .{ .re = 2, .im = 3 },
        .{ .re = 4, .im = 7 },
    };
    const base = Options{
        .blas_path = "unused",
        .library = "unused",
        .shape = "tiny",
        .routine = .zsymm,
        .m = 2,
        .n = 1,
        .side = 'L',
        .uplo = 'U',
        .alpha = .{ .re = 1 },
        .beta = .{ .re = 0 },
        .reps = 1,
    };
    try std.testing.expectEqual(ComplexF64{ .re = 2, .im = 3 }, structuredElement(ComplexF64, &matrix, 2, base, 1, 0));
    var hermitian = base;
    hermitian.routine = .zhemm;
    try std.testing.expectEqual(ComplexF64{ .re = 2, .im = -3 }, structuredElement(ComplexF64, &matrix, 2, hermitian, 1, 0));
    try std.testing.expectEqual(ComplexF64{ .re = 1, .im = 0 }, structuredElement(ComplexF64, &matrix, 2, hermitian, 0, 0));
}

test "independent reference handles left and right side products" {
    const a = [_]f64{ 2, 99, 3, 4 };
    const c0 = [_]f64{ 0, 0 };
    const left_b = [_]f64{ 5, 7 };
    const base = Options{
        .blas_path = "unused",
        .library = "unused",
        .shape = "tiny",
        .routine = .dsymm,
        .m = 2,
        .n = 1,
        .side = 'L',
        .uplo = 'U',
        .alpha = .{ .re = 1 },
        .beta = .{ .re = 0 },
        .reps = 1,
    };
    try std.testing.expectEqual(@as(f64, 31), expectedElement(f64, &a, &left_b, &c0, base, 2, 2, 2, 0, 0));
    try std.testing.expectEqual(@as(f64, 43), expectedElement(f64, &a, &left_b, &c0, base, 2, 2, 2, 1, 0));

    var right = base;
    right.m = 1;
    right.n = 2;
    right.side = 'R';
    const right_b = [_]f64{ 5, 7 };
    try std.testing.expectEqual(@as(f64, 31), expectedElement(f64, &a, &right_b, &c0, right, 2, 1, 1, 0, 0));
    try std.testing.expectEqual(@as(f64, 43), expectedElement(f64, &a, &right_b, &c0, right, 2, 1, 1, 0, 1));
}
