// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! One backend, precision, operation, and shape per process. Stdout contains
//! exactly one CSV data row on success; errors produce no row and a nonzero exit.
//! Every published duration is a measured batch duration / actual batch calls.
//! Each of the 30 samples starts with distinct pristine input copies prepared
//! outside the clock. Solve factors and native pivots are prepared once outside
//! timing. Timed cost includes API entry/validation, adapter dispatch, and info
//! checking; excludes allocation, copies, factorization for solves, and checks.
//! Zynum uses its checked public API; competitors use the named native interface.
//! Results describe a warm process and restored inputs, not cold-memory latency.
//! The 128 MiB cap covers worker numerical buffers, not shared libraries or their
//! internal workspaces. Batches target >=100 us; memory/call caps are disclosed
//! through batch_calls, min_batch_ns, and batch_target_met, never time floors.

const std = @import("std");
const lapack = @import("zynum-lapack");
const DynLib = @import("benchmark-dynamic-library").DynLib;
const Int = i32;
const sample_count = 30;
const target_batch_ns = 100_000;
const memory_limit = 128 * 1024 * 1024;
const call_limit = 1 << 20;
const padding_value = -9876.25;
const residual_limit = 100.0;

const Backend = enum { zynum, openblas, accelerate };
const Operation = enum { potrf, potrs, getrf, getrs };
const Kind = enum {
    spotrf,
    dpotrf,
    spotrs,
    dpotrs,
    sgetrf,
    dgetrf,
    sgetrs,
    dgetrs,

    fn operation(self: Kind) Operation {
        return switch (self) {
            .spotrf, .dpotrf => .potrf,
            .spotrs, .dpotrs => .potrs,
            .sgetrf, .dgetrf => .getrf,
            .sgetrs, .dgetrs => .getrs,
        };
    }

    fn single(self: Kind) bool {
        return switch (self) {
            .spotrf, .spotrs, .sgetrf, .sgetrs => true,
            else => false,
        };
    }
};

const Options = struct {
    backend: Backend,
    kind: Kind,
    path: ?[]const u8,
    m: usize,
    n: usize,
    nrhs: usize,
    uplo: u8,
    lda: usize,
    ldb: usize,

    fn solving(self: Options) bool {
        return self.kind.operation() == .potrs or self.kind.operation() == .getrs;
    }

    fn cholesky(self: Options) bool {
        return self.kind.operation() == .potrf or self.kind.operation() == .potrs;
    }

    fn triangle(self: Options) lapack.Triangle {
        return if (self.uplo == 'L') .lower else .upper;
    }

    fn backendLabel(self: Options) []const u8 {
        return switch (self.backend) {
            .zynum => "zynum-checked",
            .openblas => "openblas-lapacke-work-lp64",
            .accelerate => "accelerate-new-lp64",
        };
    }
};

fn usage() void {
    std.debug.print("usage: lapack-sweep --backend zynum|openblas|accelerate --kind spotrf|dpotrf|spotrs|dpotrs|sgetrf|dgetrf|sgetrs|dgetrs --m N --n N [--nrhs N] [--uplo L|U] [--pad N] [--library path]\n", .{});
}

fn parseOptions(init: std.process.Init) !Options {
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, std.heap.page_allocator);
    defer args.deinit();
    // Argument slices must survive iterator deinit; duplicate only the path.
    var backend: ?Backend = null;
    var kind: ?Kind = null;
    var path: ?[]const u8 = null;
    errdefer if (path) |p| std.heap.page_allocator.free(p);
    var m: ?usize = null;
    var n: ?usize = null;
    var nrhs: usize = 0;
    var pad: usize = 0;
    var uplo: ?u8 = null;
    var seen: u8 = 0;
    _ = args.next();
    while (args.next()) |arg| {
        const slot: u3 = if (std.mem.eql(u8, arg, "--backend")) 0 else if (std.mem.eql(u8, arg, "--kind")) 1 else if (std.mem.eql(u8, arg, "--library")) 2 else if (std.mem.eql(u8, arg, "--m")) 3 else if (std.mem.eql(u8, arg, "--n")) 4 else if (std.mem.eql(u8, arg, "--nrhs")) 5 else if (std.mem.eql(u8, arg, "--uplo")) 6 else if (std.mem.eql(u8, arg, "--pad")) 7 else return error.BadArgument;
        const bit = @as(u8, 1) << slot;
        if (seen & bit != 0) return error.DuplicateArgument;
        seen |= bit;
        const value = args.next() orelse return error.MissingValue;
        switch (slot) {
            0 => backend = std.meta.stringToEnum(Backend, value) orelse return error.BadBackend,
            1 => kind = std.meta.stringToEnum(Kind, value) orelse return error.BadKind,
            2 => path = try std.heap.page_allocator.dupe(u8, value),
            3 => m = try std.fmt.parseInt(usize, value, 10),
            4 => n = try std.fmt.parseInt(usize, value, 10),
            5 => nrhs = try std.fmt.parseInt(usize, value, 10),
            6 => {
                if (value.len != 1 or (value[0] != 'L' and value[0] != 'U')) return error.BadTriangle;
                uplo = value[0];
            },
            7 => pad = try std.fmt.parseInt(usize, value, 10),
        }
    }
    const selected = backend orelse return error.MissingBackend;
    if ((selected == .zynum) == (path != null)) return error.BadLibraryArgument;
    if (path) |p| if (p.len == 0) return error.BadLibraryArgument;
    const rows = m orelse return error.MissingShape;
    const columns = n orelse return error.MissingShape;
    const lda = try std.math.add(usize, rows, pad);
    const ldb = try std.math.add(usize, columns, pad);
    for ([_]usize{ rows, columns, nrhs, lda, ldb }) |value| {
        if (value > std.math.maxInt(Int)) return error.DimensionTooLarge;
    }
    if (rows == 0 or columns == 0) return error.EmptyTimingShape;
    const options = Options{ .backend = selected, .kind = kind orelse return error.MissingKind, .path = path, .m = rows, .n = columns, .nrhs = nrhs, .uplo = uplo orelse 'L', .lda = lda, .ldb = ldb };
    if ((options.cholesky() or options.solving()) and rows != columns) return error.ExpectedSquare;
    if (options.solving() != (nrhs > 0)) return error.BadRhsCount;
    if (!options.cholesky() and uplo != null) return error.UnusedTriangle;
    return options;
}

fn Adapter(comptime T: type) type {
    return struct {
        const Self = @This();
        const prefix = if (T == f32) "s" else "d";
        const Open = struct {
            potrf: *const fn (c_int, u8, Int, [*]T, Int) callconv(.c) Int,
            potrs: *const fn (c_int, u8, Int, Int, [*]const T, Int, [*]T, Int) callconv(.c) Int,
            getrf: *const fn (c_int, Int, Int, [*]T, Int, [*]Int) callconv(.c) Int,
            getrs: *const fn (c_int, u8, Int, Int, [*]const T, Int, [*]const Int, [*]T, Int) callconv(.c) Int,
        };
        const Apple = struct {
            potrf: *const fn (*const u8, *const Int, [*]T, *const Int, *Int) callconv(.c) void,
            potrs: *const fn (*const u8, *const Int, *const Int, [*]const T, *const Int, [*]T, *const Int, *Int) callconv(.c) void,
            getrf: *const fn (*const Int, *const Int, [*]T, *const Int, [*]Int, *Int) callconv(.c) void,
            getrs: *const fn (*const u8, *const Int, *const Int, [*]const T, *const Int, [*]const Int, [*]T, *const Int, *Int) callconv(.c) void,
        };
        functions: union(Backend) { zynum: void, openblas: Open, accelerate: Apple },
        dyn: ?DynLib,
        options: Options,

        fn init(options: Options) !Self {
            if (options.backend == .zynum) return .{ .functions = .{ .zynum = {} }, .dyn = null, .options = options };
            var dyn = try DynLib.open(options.path.?);
            errdefer dyn.close();
            if (options.backend == .openblas) {
                const ConfigFn = *const fn () callconv(.c) ?[*:0]const u8;
                const get_config = dyn.lookup(ConfigFn, "openblas_get_config") orelse return error.MissingOpenBlasConfig;
                const config = std.mem.span(get_config() orelse return error.InvalidOpenBlasConfig);
                if (std.mem.indexOf(u8, config, "OpenBLAS") == null) return error.InvalidOpenBlasConfig;
                // A library path supplied at runtime must not change the LP64
                // contract silently, even when its symbol names are identical.
                if (std.mem.indexOf(u8, config, "USE64BITINT") != null) return error.UnsupportedIntegerAbi;
            }
            const functions: @FieldType(Self, "functions") = switch (options.backend) {
                .openblas => .{ .openblas = .{
                    .potrf = dyn.lookup(@FieldType(Open, "potrf"), "LAPACKE_" ++ prefix ++ "potrf_work") orelse return error.MissingPotrf,
                    .potrs = dyn.lookup(@FieldType(Open, "potrs"), "LAPACKE_" ++ prefix ++ "potrs_work") orelse return error.MissingPotrs,
                    .getrf = dyn.lookup(@FieldType(Open, "getrf"), "LAPACKE_" ++ prefix ++ "getrf_work") orelse return error.MissingGetrf,
                    .getrs = dyn.lookup(@FieldType(Open, "getrs"), "LAPACKE_" ++ prefix ++ "getrs_work") orelse return error.MissingGetrs,
                } },
                .accelerate => .{ .accelerate = .{
                    .potrf = dyn.lookup(@FieldType(Apple, "potrf"), prefix ++ "potrf$NEWLAPACK") orelse return error.MissingPotrf,
                    .potrs = dyn.lookup(@FieldType(Apple, "potrs"), prefix ++ "potrs$NEWLAPACK") orelse return error.MissingPotrs,
                    .getrf = dyn.lookup(@FieldType(Apple, "getrf"), prefix ++ "getrf$NEWLAPACK") orelse return error.MissingGetrf,
                    .getrs = dyn.lookup(@FieldType(Apple, "getrs"), prefix ++ "getrs$NEWLAPACK") orelse return error.MissingGetrs,
                } },
                .zynum => unreachable,
            };
            return .{ .functions = functions, .dyn = dyn, .options = options };
        }

        fn deinit(self: *Self) void {
            // This wrapper deliberately retains external handles until exit.
            if (self.dyn) |*dyn| dyn.close();
        }

        fn factorize(self: *const Self, a: []T, pivots: []Int) !void {
            const o = self.options;
            const m: Int = @intCast(o.m);
            const n: Int = @intCast(o.n);
            const lda: Int = @intCast(o.lda);
            var info: Int = 0;
            switch (self.functions) {
                .zynum => {
                    const view = lapack.Matrix(T){ .values = a, .row_count = m, .column_count = n, .leading_dimension = lda };
                    info = @intCast(if (o.cholesky()) try lapack.choleskyFactorize(T, view, o.triangle()) else try lapack.luFactorize(T, view, pivots));
                },
                .openblas => |f| info = if (o.cholesky()) f.potrf(102, o.uplo, n, a.ptr, lda) else f.getrf(102, m, n, a.ptr, lda, pivots.ptr),
                .accelerate => |f| if (o.cholesky()) f.potrf(&o.uplo, &n, a.ptr, &lda, &info) else f.getrf(&m, &n, a.ptr, &lda, pivots.ptr, &info),
            }
            if (info != 0) {
                std.debug.print("factorization failed: info={d}\n", .{info});
                return error.NumericalFailure;
            }
        }

        fn solve(self: *const Self, factor: []const T, pivots: []const Int, rhs: []T) !void {
            const o = self.options;
            const n: Int = @intCast(o.n);
            const nrhs: Int = @intCast(o.nrhs);
            const lda: Int = @intCast(o.lda);
            const ldb: Int = @intCast(o.ldb);
            const trans: u8 = 'N';
            var info: Int = 0;
            switch (self.functions) {
                .zynum => {
                    const a = lapack.ConstMatrix(T){ .values = factor, .row_count = n, .column_count = n, .leading_dimension = lda };
                    const b = lapack.Matrix(T){ .values = rhs, .row_count = n, .column_count = nrhs, .leading_dimension = ldb };
                    info = @intCast(if (o.cholesky()) try lapack.choleskySolve(T, a, b, o.triangle()) else try lapack.luSolve(T, a, pivots, b));
                },
                .openblas => |f| info = if (o.cholesky()) f.potrs(102, o.uplo, n, nrhs, factor.ptr, lda, rhs.ptr, ldb) else f.getrs(102, trans, n, nrhs, factor.ptr, lda, pivots.ptr, rhs.ptr, ldb),
                .accelerate => |f| if (o.cholesky()) f.potrs(&o.uplo, &n, &nrhs, factor.ptr, &lda, rhs.ptr, &ldb, &info) else f.getrs(&trans, &n, &nrhs, factor.ptr, &lda, pivots.ptr, rhs.ptr, &ldb, &info),
            }
            if (info != 0) {
                std.debug.print("solve failed: info={d}\n", .{info});
                return error.NumericalFailure;
            }
        }
    };
}

fn randomValue(seed: *u64) f64 {
    seed.* = seed.* *% 6364136223846793005 +% 1442695040888963407;
    const bits: u32 = @truncate(seed.* >> 32);
    return @as(f64, @floatFromInt(bits % 1024)) / 1024.0 - 0.5;
}

fn fillMatrix(comptime T: type, o: Options, a: []T) void {
    @memset(a, padding_value);
    var seed: u64 = 0x3141592653589793;
    const scale: f64 = @floatFromInt(@max(o.m, o.n));
    if (o.cholesky()) {
        // Symmetric strict diagonal dominance with a positive diagonal gives
        // SPD, with eigenvalues in [1.5,2.5] before negligible input rounding.
        for (0..o.n) |j| {
            for (0..j) |i| {
                const value: T = @floatCast(randomValue(&seed) / scale);
                a[i + j * o.lda] = value;
                a[j + i * o.lda] = value;
            }
            a[j + j * o.lda] = 2;
        }
    } else {
        // Reverse the rows of a diagonally dominant square/rectangular core.
        // The first pivot is m-1 when m>1; the min(m,n) core has full rank.
        for (0..o.n) |j| {
            for (0..o.m) |i| {
                a[o.m - 1 - i + j * o.lda] = @floatCast(if (i == j) 2.0 else randomValue(&seed) / scale);
            }
        }
    }
}

fn exactSolution(row: usize, rhs: usize) f64 {
    return @as(f64, @floatFromInt((row % 17 + 3 * (rhs % 17)) % 17)) / 17.0 - 0.5;
}

fn fillRhs(comptime T: type, o: Options, a: []const T, b: []T) void {
    @memset(b, padding_value);
    for (0..o.nrhs) |j| {
        for (0..o.n) |i| {
            var sum: f64 = 0;
            for (0..o.n) |k| sum += @as(f64, @floatCast(a[i + k * o.lda])) * exactSolution(k, j);
            b[i + j * o.ldb] = @floatCast(sum);
        }
    }
}

fn checkedPivot(o: Options, pivots: []const Int, step: usize) !usize {
    // Never mutate native pivots: GETRS receives exactly GETRF's native output.
    const base: Int = if (o.backend == .zynum) 0 else 1;
    if (pivots[step] < base) return error.InvalidNativePivot;
    const row: usize = @intCast(pivots[step] - base);
    if (row < step or row >= o.m) return error.InvalidNativePivot;
    return row;
}

fn checkStorage(comptime T: type, values: []const T, rows: usize, columns: usize, ld: usize) !void {
    for (0..columns) |j| {
        for (0..rows) |i| if (!std.math.isFinite(values[i + j * ld])) return error.NonFiniteResult;
        for (rows..ld) |i| if (values[i + j * ld] != padding_value) return error.ModifiedPadding;
    }
}

// A known Accelerate upper-POTRF path writes part of the unused triangle.
// Record that storage-convention deviation separately from numerical validity.
// All other backends/triangles retain a hard preservation check. Solves still
// verify the entire input factor and pivots remain byte-identical.
fn checkFactor(comptime T: type, o: Options, original: []const T, factor: []const T, pivots: []const Int, unused_modified_max: *usize) !void {
    if (o.cholesky()) {
        var modified: usize = 0;
        for (0..o.n) |j| {
            for (0..o.n) |i| {
                const index = i + j * o.lda;
                const unused = if (o.uplo == 'L') i < j else i > j;
                if (unused) {
                    if (!std.mem.eql(u8, std.mem.asBytes(&factor[index]), std.mem.asBytes(&original[index]))) modified += 1;
                } else if (!std.math.isFinite(factor[index])) return error.NonFiniteResult;
            }
            for (o.n..o.lda) |i| if (factor[i + j * o.lda] != padding_value) return error.ModifiedPadding;
            if (!(factor[j + j * o.lda] > 0)) return error.InvalidFactorDiagonal;
        }
        if (modified != 0 and !(o.backend == .accelerate and o.uplo == 'U')) return error.ModifiedUnusedTriangle;
        unused_modified_max.* = @max(unused_modified_max.*, modified);
    } else {
        try checkStorage(T, factor, o.m, o.n, o.lda);
        for (0..@min(o.m, o.n)) |k| {
            _ = try checkedPivot(o, pivots, k);
            if (factor[k + k * o.lda] == 0) return error.InvalidFactorDiagonal;
        }
        if (o.m > 1 and try checkedPivot(o, pivots, 0) != o.m - 1) return error.ExpectedRowPivot;
    }
}

fn scaledResidual(comptime T: type, residual: f64, scale: f64, dimension: usize) !f64 {
    if (!std.math.isFinite(residual) or !std.math.isFinite(scale)) return error.NonFiniteResidual;
    const epsilon: f64 = if (T == f32) 0x1p-23 else 0x1p-52;
    const denominator = @as(f64, @floatFromInt(@max(@as(usize, 1), dimension))) * epsilon * scale;
    const result = if (denominator == 0) (if (residual == 0) @as(f64, 0) else return error.InvalidResidualScale) else residual / denominator;
    if (!std.math.isFinite(result) or result > residual_limit) {
        std.debug.print("scaled residual failed: {d} (limit {d})\n", .{ result, residual_limit });
        return error.ResidualTooLarge;
    }
    return result;
}

fn factorResidual(comptime T: type, o: Options, original: []const T, factor: []const T, pivots: []const Int, permutation: []usize, unused_modified_max: *usize) !f64 {
    try checkFactor(T, o, original, factor, pivots, unused_modified_max);
    for (permutation, 0..) |*p, i| p.* = i;
    const steps = @min(o.m, o.n);
    if (!o.cholesky()) {
        for (0..steps) |k| {
            const row = try checkedPivot(o, pivots, k);
            if (row != k) std.mem.swap(usize, &permutation[k], &permutation[row]);
        }
    }
    var residual: f64 = 0;
    var norm: f64 = 0;
    for (0..o.m) |i| {
        var error_sum: f64 = 0;
        var original_sum: f64 = 0;
        for (0..o.n) |j| {
            var reconstructed: f64 = 0;
            for (0..@min(@min(i + 1, j + 1), steps)) |k| {
                if (o.cholesky()) {
                    const left = if (o.uplo == 'L') factor[i + k * o.lda] else factor[k + i * o.lda];
                    const right = if (o.uplo == 'L') factor[j + k * o.lda] else factor[k + j * o.lda];
                    reconstructed += @as(f64, @floatCast(left)) * @as(f64, @floatCast(right));
                } else {
                    const left: f64 = if (i == k) 1 else @floatCast(factor[i + k * o.lda]);
                    reconstructed += left * @as(f64, @floatCast(factor[k + j * o.lda]));
                }
            }
            const expected: f64 = @floatCast(original[permutation[i] + j * o.lda]);
            error_sum += @abs(reconstructed - expected);
            original_sum += @abs(expected);
        }
        if (!std.math.isFinite(error_sum) or !std.math.isFinite(original_sum)) return error.NonFiniteResidual;
        residual = @max(residual, error_sum);
        norm = @max(norm, original_sum);
    }
    return scaledResidual(T, residual, norm, steps);
}

fn solveResidual(comptime T: type, o: Options, a: []const T, b: []const T, x: []const T) !f64 {
    try checkStorage(T, x, o.n, o.nrhs, o.ldb);
    var residual: f64 = 0;
    var norm_a: f64 = 0;
    var norm_b: f64 = 0;
    var norm_x: f64 = 0;
    for (0..o.n) |i| {
        var a_sum: f64 = 0;
        var b_sum: f64 = 0;
        var x_sum: f64 = 0;
        var error_sum: f64 = 0;
        for (0..o.n) |k| a_sum += @abs(@as(f64, @floatCast(a[i + k * o.lda])));
        for (0..o.nrhs) |j| {
            var reconstructed: f64 = 0;
            for (0..o.n) |k| reconstructed += @as(f64, @floatCast(a[i + k * o.lda])) * @as(f64, @floatCast(x[k + j * o.ldb]));
            const expected: f64 = @floatCast(b[i + j * o.ldb]);
            error_sum += @abs(reconstructed - expected);
            b_sum += @abs(expected);
            x_sum += @abs(@as(f64, @floatCast(x[i + j * o.ldb])));
        }
        for ([_]f64{ error_sum, a_sum, b_sum, x_sum }) |value| if (!std.math.isFinite(value)) return error.NonFiniteResidual;
        residual = @max(residual, error_sum);
        norm_a = @max(norm_a, a_sum);
        norm_b = @max(norm_b, b_sum);
        norm_x = @max(norm_x, x_sum);
    }
    return scaledResidual(T, residual, norm_a * norm_x + norm_b, o.n);
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

const Result = struct {
    batch_calls: usize,
    best_ns: f64,
    median_ns: f64,
    p95_ns: f64,
    min_batch_ns: i96,
    scaled_residual: f64,
    factor_unused_triangle_modified_max: usize,
    // Final 30 measured per-call durations, retained in acquisition order.
    // Calibration and discarded sample sets are not included.
    sample_ns: [sample_count]f64,
};

fn sortTimings(values: []f64) void {
    for (1..values.len) |i| {
        const value = values[i];
        var j = i;
        while (j > 0 and values[j - 1] > value) : (j -= 1) values[j] = values[j - 1];
        values[j] = value;
    }
}

fn Buffers(comptime T: type) type {
    return struct {
        original: []const T,
        factor: []const T,
        pivots: []const Int,
        rhs: []const T,
        factor_guard: []const T,
        pivot_guard: []const Int,
        work: []T,
        batch_pivots: []Int,
        slot_length: usize,
        pivot_length: usize,
        unused_modified_max: *usize,

        fn batch(self: @This(), adapter: *const Adapter(T), io: std.Io, calls: usize) !i96 {
            const o = adapter.options;
            const source = if (o.solving()) self.rhs else self.original;
            // Each call receives its own input. No input copy is timed.
            for (0..calls) |i| {
                @memcpy(self.work[i * self.slot_length ..][0..self.slot_length], source);
            }
            @memset(self.batch_pivots[0 .. calls * self.pivot_length], -1);
            const start = nowNs(io);
            for (0..calls) |i| {
                const values = self.work[i * self.slot_length ..][0..self.slot_length];
                if (o.solving()) {
                    try adapter.solve(self.factor, self.pivots, values);
                } else {
                    try adapter.factorize(values, self.batch_pivots[i * self.pivot_length ..][0..self.pivot_length]);
                }
            }
            const elapsed = nowNs(io) - start;
            // Validate all outputs outside timing, including calibration runs.
            // Independent full residual checks in run() cover the warmup and
            // the first/last output slots of the final measured batch.
            for (0..calls) |i| {
                const values = self.work[i * self.slot_length ..][0..self.slot_length];
                if (o.solving()) {
                    try checkStorage(T, values, o.n, o.nrhs, o.ldb);
                } else {
                    try checkFactor(T, o, self.original, values, self.batch_pivots[i * self.pivot_length ..][0..self.pivot_length], self.unused_modified_max);
                }
            }
            if (o.solving()) {
                if (!std.mem.eql(u8, std.mem.sliceAsBytes(self.factor), std.mem.sliceAsBytes(self.factor_guard)) or !std.mem.eql(Int, self.pivots, self.pivot_guard)) return error.ModifiedSolveFactor;
            }
            if (elapsed < 0) return error.InvalidTiming;
            return elapsed;
        }
    };
}

fn run(comptime T: type, options: Options, io: std.Io) !Result {
    const allocator = std.heap.page_allocator;
    const o = options;
    const a_len = try std.math.mul(usize, o.lda, o.n);
    const b_len = if (o.solving()) try std.math.mul(usize, o.ldb, o.nrhs) else 0;
    const k = if (o.cholesky()) 0 else @min(o.m, o.n);
    const guard_a_len = if (o.solving()) a_len else 0;
    const guard_p_len = if (o.solving()) k else 0;
    const slot_length = if (o.solving()) b_len else a_len;
    const pivot_length = if (!o.solving()) k else 0;
    var base_bytes: usize = 0;
    for ([_]usize{ a_len, a_len, b_len, b_len, guard_a_len }) |count| base_bytes = try std.math.add(usize, base_bytes, try std.math.mul(usize, count, @sizeOf(T)));
    for ([_]usize{ k, guard_p_len }) |count| base_bytes = try std.math.add(usize, base_bytes, try std.math.mul(usize, count, @sizeOf(Int)));
    base_bytes = try std.math.add(usize, base_bytes, try std.math.mul(usize, o.m, @sizeOf(usize)));
    const slot_bytes = try std.math.add(usize, try std.math.mul(usize, slot_length, @sizeOf(T)), try std.math.mul(usize, pivot_length, @sizeOf(Int)));
    if (base_bytes >= memory_limit or slot_bytes > memory_limit - base_bytes) return error.MemoryBudgetExceeded;
    const max_calls = @min(call_limit, (memory_limit - base_bytes) / slot_bytes);

    const original = try allocator.alloc(T, a_len);
    defer allocator.free(original);
    const factor = try allocator.alloc(T, a_len);
    defer allocator.free(factor);
    const pivots = try allocator.alloc(Int, k);
    defer allocator.free(pivots);
    const rhs = try allocator.alloc(T, b_len);
    defer allocator.free(rhs);
    const check_rhs = try allocator.alloc(T, b_len);
    defer allocator.free(check_rhs);
    const factor_guard = try allocator.alloc(T, guard_a_len);
    defer allocator.free(factor_guard);
    const pivot_guard = try allocator.alloc(Int, guard_p_len);
    defer allocator.free(pivot_guard);
    const permutation = try allocator.alloc(usize, o.m);
    defer allocator.free(permutation);

    fillMatrix(T, o, original);
    @memcpy(factor, original);
    @memset(pivots, -1);
    var adapter = try Adapter(T).init(o);
    defer adapter.deinit();
    try adapter.factorize(factor, pivots);
    var unused_modified_max: usize = 0;
    const factor_error = try factorResidual(T, o, original, factor, pivots, permutation, &unused_modified_max);
    var residual = factor_error;
    if (o.solving()) {
        @memcpy(factor_guard, factor);
        @memcpy(pivot_guard, pivots);
        fillRhs(T, o, original, rhs);
        @memcpy(check_rhs, rhs);
        try adapter.solve(factor, pivots, check_rhs);
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(factor), std.mem.sliceAsBytes(factor_guard)) or !std.mem.eql(Int, pivots, pivot_guard)) return error.ModifiedSolveFactor;
        // The CSV metric for solve kinds is the solve backward error; the
        // independent factor reconstruction must also pass before timing.
        residual = try solveResidual(T, o, original, rhs, check_rhs);
    }

    const work = try allocator.alloc(T, try std.math.mul(usize, slot_length, max_calls));
    defer allocator.free(work);
    const batch_pivots = try allocator.alloc(Int, try std.math.mul(usize, pivot_length, max_calls));
    defer allocator.free(batch_pivots);
    const buffers = Buffers(T){ .original = original, .factor = factor, .pivots = pivots, .rhs = rhs, .factor_guard = factor_guard, .pivot_guard = pivot_guard, .work = work, .batch_pivots = batch_pivots, .slot_length = slot_length, .pivot_length = pivot_length, .unused_modified_max = &unused_modified_max };
    var calls: usize = 1;
    var timings: [sample_count]f64 = undefined;
    var minimum: i96 = 0;
    while (true) {
        // Require two calibration batches to avoid selecting a batch from a
        // single scheduling outlier; repeat on restored inputs each time.
        while (true) {
            const first = try buffers.batch(&adapter, io, calls);
            const second = try buffers.batch(&adapter, io, calls);
            if (@min(first, second) >= target_batch_ns or calls == max_calls) break;
            calls = @min(max_calls, calls * 2);
        }
        minimum = std.math.maxInt(i96);
        for (&timings) |*timing| {
            const elapsed = try buffers.batch(&adapter, io, calls);
            if (elapsed <= 0) return error.TimerResolutionTooLow;
            minimum = @min(minimum, elapsed);
            timing.* = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(calls));
        }
        if (minimum >= target_batch_ns or calls == max_calls) break;
        // If the final samples become faster, recalibrate and collect a fresh
        // set of 30. Only that final set contributes to the reported quantiles.
        calls = @min(max_calls, calls * 2);
    }
    // The final sample's output slots are still present. Check entire results
    // from actual timed calls, outside every clock, reusing permutation scratch.
    for ([_]usize{ 0, calls - 1 }, 0..) |slot, position| {
        if (position == 1 and calls == 1) continue;
        const values = work[slot * slot_length ..][0..slot_length];
        const measured_residual = if (o.solving())
            try solveResidual(T, o, original, rhs, values)
        else
            try factorResidual(T, o, original, values, batch_pivots[slot * pivot_length ..][0..pivot_length], permutation, &unused_modified_max);
        residual = @max(residual, measured_residual);
    }
    const sample_ns = timings;
    sortTimings(&timings);
    return .{ .batch_calls = calls, .best_ns = timings[0], .median_ns = (timings[14] + timings[15]) / 2, .p95_ns = timings[28], .min_batch_ns = minimum, .scaled_residual = residual, .factor_unused_triangle_modified_max = unused_modified_max, .sample_ns = sample_ns };
}

pub fn main(init: std.process.Init) !void {
    const options = parseOptions(init) catch |err| {
        usage();
        return err;
    };
    defer if (options.path) |path| std.heap.page_allocator.free(path);
    const result = if (options.kind.single()) try run(f32, options, init.io) else try run(f64, options, init.io);
    var stdout_buffer: [2048]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    // Header is documented in NOTES.md; no CSV output is emitted until every
    // required check and every timing sample has succeeded.
    try stdout_writer.interface.print("{s},{s},{d},{d},{d},{c},{d},{d},{d},{d},{d:.6},{d:.6},{d:.6},{e},numerics-padding-ok,{d},{s},{d},\"", .{
        options.backendLabel(),                                          @tagName(options.kind), options.m,      options.n,              options.nrhs,
        if (options.cholesky()) options.uplo else @as(u8, 'N'),          options.lda,            options.ldb,    result.batch_calls,     sample_count,
        result.median_ns,                                                result.p95_ns,          result.best_ns, result.scaled_residual, result.factor_unused_triangle_modified_max,
        if (result.min_batch_ns >= target_batch_ns) "true" else "false", result.min_batch_ns,
    });
    for (result.sample_ns, 0..) |sample, i| {
        if (i != 0) try stdout_writer.interface.writeByte(';');
        try stdout_writer.interface.print("{d}", .{sample});
    }
    try stdout_writer.interface.writeAll("\"\n");
    try stdout_writer.interface.flush();
}
