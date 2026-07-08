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

fn GemmFn(comptime T: type) type {
    return *const fn ([*]const u8, [*]const u8, *const BlasInt, *const BlasInt, *const BlasInt, *const T, [*]const T, *const BlasInt, [*]const T, *const BlasInt, *const T, [*]T, *const BlasInt) callconv(.c) void;
}

const SgemmFn = GemmFn(f32);
const DgemmFn = GemmFn(f64);
const CgemmFn = GemmFn(ComplexF32);
const ZgemmFn = GemmFn(ComplexF64);

const Lib = struct {
    name: []const u8,
    dyn: std.DynLib,
    sgemm: SgemmFn,
    dgemm: DgemmFn,
    cgemm: CgemmFn,
    zgemm: ZgemmFn,
};

const LibSpec = struct {
    name: []const u8,
    path: []const u8,
};

const Shape = struct {
    label: []const u8,
    m: usize,
    n: usize,
    k: usize,
};

const max_custom_shapes = 64;

const default_shapes = [_]Shape{
    .{ .label = "m1_n1_k1", .m = 1, .n = 1, .k = 1 },
    .{ .label = "m8_n8_k8", .m = 8, .n = 8, .k = 8 },
    .{ .label = "m31_n31_k31", .m = 31, .n = 31, .k = 31 },
    .{ .label = "m33_n33_k33", .m = 33, .n = 33, .k = 33 },
    .{ .label = "sq64", .m = 64, .n = 64, .k = 64 },
    .{ .label = "sq96", .m = 96, .n = 96, .k = 96 },
    .{ .label = "sq128", .m = 128, .n = 128, .k = 128 },
    .{ .label = "sq192", .m = 192, .n = 192, .k = 192 },
    .{ .label = "sq256", .m = 256, .n = 256, .k = 256 },
    .{ .label = "sq384", .m = 384, .n = 384, .k = 384 },
    .{ .label = "sq512", .m = 512, .n = 512, .k = 512 },
    .{ .label = "sq768", .m = 768, .n = 768, .k = 768 },
    .{ .label = "sq1024", .m = 1024, .n = 1024, .k = 1024 },
    .{ .label = "m63_n65_k17", .m = 63, .n = 65, .k = 17 },
    .{ .label = "m65_n63_k33", .m = 65, .n = 63, .k = 33 },
    .{ .label = "m127_n129_k31", .m = 127, .n = 129, .k = 31 },
    .{ .label = "m129_n127_k33", .m = 129, .n = 127, .k = 33 },
    .{ .label = "m1_n4096_k256", .m = 1, .n = 4096, .k = 256 },
    .{ .label = "m4096_n1_k256", .m = 4096, .n = 1, .k = 256 },
    .{ .label = "m1024_n64_k1024", .m = 1024, .n = 64, .k = 1024 },
    .{ .label = "m2048_n64_k512", .m = 2048, .n = 64, .k = 512 },
    .{ .label = "m4096_n32_k256", .m = 4096, .n = 32, .k = 256 },
    .{ .label = "m2048_n17_k257", .m = 2048, .n = 17, .k = 257 },
    .{ .label = "m512_n64_k2048", .m = 512, .n = 64, .k = 2048 },
    .{ .label = "m64_n1024_k1024", .m = 64, .n = 1024, .k = 1024 },
    .{ .label = "m64_n2048_k512", .m = 64, .n = 2048, .k = 512 },
    .{ .label = "m32_n4096_k256", .m = 32, .n = 4096, .k = 256 },
    .{ .label = "m17_n2048_k257", .m = 17, .n = 2048, .k = 257 },
    .{ .label = "m64_n512_k2048", .m = 64, .n = 512, .k = 2048 },
    .{ .label = "m1024_n1024_k64", .m = 1024, .n = 1024, .k = 64 },
    .{ .label = "m1024_n1024_k128", .m = 1024, .n = 1024, .k = 128 },
    .{ .label = "m1024_n1024_k256", .m = 1024, .n = 1024, .k = 256 },
    .{ .label = "m256_n256_k2048", .m = 256, .n = 256, .k = 2048 },
    .{ .label = "m128_n128_k4096", .m = 128, .n = 128, .k = 4096 },
    .{ .label = "m1536_n256_k256", .m = 1536, .n = 256, .k = 256 },
    .{ .label = "m256_n1536_k256", .m = 256, .n = 1536, .k = 256 },
    .{ .label = "m512_n256_k768", .m = 512, .n = 256, .k = 768 },
    .{ .label = "m256_n512_k768", .m = 256, .n = 512, .k = 768 },
    .{ .label = "m768_n512_k256", .m = 768, .n = 512, .k = 256 },
    .{ .label = "m512_n768_k256", .m = 512, .n = 768, .k = 256 },
    .{ .label = "m384_n640_k96", .m = 384, .n = 640, .k = 96 },
    .{ .label = "m640_n384_k96", .m = 640, .n = 384, .k = 96 },
};

fn usage() void {
    std.debug.print("usage: gemm-sweep --zynum-blas path [--accelerate path] [--openblas path] [--mkl path] [--aocl-blis path] [--reps n] [--csv path] [--check] [--kind sgemm|dgemm|cgemm|zgemm] [--shape label:m:n:k]\n", .{});
}

fn parseShape(spec: []const u8) !Shape {
    var parts: [4][]const u8 = undefined;
    var count: usize = 0;
    var it = std.mem.splitScalar(u8, spec, ':');
    while (it.next()) |part| {
        if (count == parts.len) return error.BadShape;
        parts[count] = part;
        count += 1;
    }

    const dims = switch (count) {
        3 => parts[0..3],
        4 => parts[1..4],
        else => return error.BadShape,
    };
    const label = if (count == 4) parts[0] else spec;
    return .{
        .label = label,
        .m = try std.fmt.parseInt(usize, dims[0], 10),
        .n = try std.fmt.parseInt(usize, dims[1], 10),
        .k = try std.fmt.parseInt(usize, dims[2], 10),
    };
}

fn loadLib(name: []const u8, path: []const u8) !Lib {
    var dyn = try std.DynLib.open(path);
    errdefer dyn.close();
    return .{
        .name = name,
        .dyn = dyn,
        .sgemm = dyn.lookup(SgemmFn, "sgemm_") orelse return error.MissingSgemm,
        .dgemm = dyn.lookup(DgemmFn, "dgemm_") orelse return error.MissingDgemm,
        .cgemm = dyn.lookup(CgemmFn, "cgemm_") orelse return error.MissingCgemm,
        .zgemm = dyn.lookup(ZgemmFn, "zgemm_") orelse return error.MissingZgemm,
    };
}

fn nextFillValue(seed: *u64) f64 {
    seed.* = seed.* *% 6364136223846793005 +% 1442695040888963407;
    const bits: u32 = @truncate(seed.* >> 32);
    return @as(f64, @floatFromInt(bits % 1000)) / 1000.0 - 0.5;
}

fn fill(comptime T: type, x: []T) void {
    var seed: u64 = 0x3141_5926_5358_9793;
    for (x) |*v| {
        if (T == ComplexF32 or T == ComplexF64) {
            v.* = .{
                .re = @floatCast(nextFillValue(&seed)),
                .im = @floatCast(nextFillValue(&seed)),
            };
        } else {
            v.* = @floatCast(nextFillValue(&seed));
        }
    }
}

fn zero(comptime T: type) T {
    return if (T == ComplexF32 or T == ComplexF64) .{ .re = 0, .im = 0 } else 0;
}

fn one(comptime T: type) T {
    return if (T == ComplexF32 or T == ComplexF64) .{ .re = 1, .im = 0 } else 1;
}

fn add(comptime T: type, a: T, b: T) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{ .re = a.re + b.re, .im = a.im + b.im }
    else
        a + b;
}

fn mul(comptime T: type, a: T, b: T) T {
    return if (T == ComplexF32 or T == ComplexF64)
        .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re }
    else
        a * b;
}

fn absDiff(comptime T: type, a: T, b: T) f64 {
    if (T == ComplexF32 or T == ComplexF64) {
        const dr = @as(f64, @floatCast(a.re)) - @as(f64, @floatCast(b.re));
        const di = @as(f64, @floatCast(a.im)) - @as(f64, @floatCast(b.im));
        return @sqrt(dr * dr + di * di);
    }
    return @abs(@as(f64, @floatCast(a)) - @as(f64, @floatCast(b)));
}

fn sampleIndex(index: usize, len: usize) ?usize {
    if (len == 0) return null;
    return switch (index) {
        0 => 0,
        1 => len / 4,
        2 => len / 2,
        3 => (len - 1) - (len - 1) / 4,
        4 => len - 1,
        else => null,
    };
}

fn checkGemmElement(comptime T: type, a: []const T, b: []const T, c: []const T, shape: Shape, row: usize, col: usize) !void {
    const tolerance: f64 = if (T == f32 or T == ComplexF32) 1e-3 else 1e-9;
    var expected = zero(T);
    for (0..shape.k) |p| {
        expected = add(T, expected, mul(T, a[row + p * shape.m], b[p + col * shape.k]));
    }
    const actual = c[row + col * shape.m];
    if (absDiff(T, expected, actual) > tolerance * @as(f64, @floatFromInt(@max(@as(usize, 1), shape.k)))) {
        return error.GemmCheckFailed;
    }
}

fn checkGemmSamples(comptime T: type, a: []const T, b: []const T, c: []const T, shape: Shape) !void {
    if (shape.m *| shape.n <= 4096) {
        for (0..shape.n) |col| {
            for (0..shape.m) |row| try checkGemmElement(T, a, b, c, shape, row, col);
        }
        return;
    }

    var checked: [25]struct { row: usize, col: usize } = undefined;
    var checked_count: usize = 0;
    for (0..5) |row_slot| {
        const row = sampleIndex(row_slot, shape.m) orelse continue;
        for (0..5) |col_slot| {
            const col = sampleIndex(col_slot, shape.n) orelse continue;
            var duplicate = false;
            for (checked[0..checked_count]) |item| {
                if (item.row == row and item.col == col) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            checked[checked_count] = .{ .row = row, .col = col };
            checked_count += 1;
            try checkGemmElement(T, a, b, c, shape, row, col);
        }
    }
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn gflops(elapsed_ns: i96, flop_factor: f64, m: usize, n: usize, k: usize) f64 {
    if (elapsed_ns <= 0) return 0;
    const ops = flop_factor * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    return ops / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9) / 1e9;
}

const BenchResult = struct {
    best_ns: i96,
    median_ns: i96,
    p95_ns: i96,
    max_ns: i96,
    check: []const u8,
};

fn sortTimings(values: []i96) void {
    var i: usize = 1;
    while (i < values.len) : (i += 1) {
        const value = values[i];
        var j = i;
        while (j > 0 and values[j - 1] > value) : (j -= 1) {
            values[j] = values[j - 1];
        }
        values[j] = value;
    }
}

fn benchGemm(comptime T: type, gemm: GemmFn(T), allocator: std.mem.Allocator, io: std.Io, shape: Shape, reps: usize, check: bool) !BenchResult {
    const m_i: BlasInt = @intCast(shape.m);
    const n_i: BlasInt = @intCast(shape.n);
    const k_i: BlasInt = @intCast(shape.k);
    const a = try allocator.alloc(T, shape.m * shape.k);
    defer allocator.free(a);
    const b = try allocator.alloc(T, shape.k * shape.n);
    defer allocator.free(b);
    const c = try allocator.alloc(T, shape.m * shape.n);
    defer allocator.free(c);
    fill(T, a);
    fill(T, b);
    @memset(c, zero(T));

    var ta = [_]u8{'N'};
    var tb = [_]u8{'N'};
    var alpha: T = one(T);
    var beta: T = zero(T);
    gemm(&ta, &tb, &m_i, &n_i, &k_i, &alpha, a.ptr, &m_i, b.ptr, &k_i, &beta, c.ptr, &m_i);
    if (check) try checkGemmSamples(T, a, b, c, shape);

    if (reps == 0) return error.InvalidRepetitions;
    const timings = try allocator.alloc(i96, reps);
    defer allocator.free(timings);

    for (0..reps) |rep| {
        @memset(c, zero(T));
        const start = nowNs(io);
        gemm(&ta, &tb, &m_i, &n_i, &k_i, &alpha, a.ptr, &m_i, b.ptr, &k_i, &beta, c.ptr, &m_i);
        const end = nowNs(io);
        timings[rep] = if (end >= start) @max(@as(i96, 1), end - start) else 1;
    }
    sortTimings(timings);
    const p95_index = @min(timings.len - 1, ((timings.len * 95) + 99) / 100 - 1);
    return .{
        .best_ns = timings[0],
        .median_ns = timings[timings.len / 2],
        .p95_ns = timings[p95_index],
        .max_ns = timings[timings.len - 1],
        .check = if (check) "checked-ok" else "unchecked",
    };
}

fn csvEscape(writer: *std.Io.Writer, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |c| {
        if (c == '"') try writer.writeByte('"');
        try writer.writeByte(c);
    }
    try writer.writeByte('"');
}

fn flopFactorForKind(kind: []const u8) f64 {
    return if (std.mem.eql(u8, kind, "cgemm") or std.mem.eql(u8, kind, "zgemm")) 8.0 else 2.0;
}

fn writeCsvRow(writer: *std.Io.Writer, kind: []const u8, shape_index: usize, shape: Shape, lib_name: []const u8, result: BenchResult, reps: usize) !void {
    try writer.writeAll(kind);
    try writer.writeByte(',');
    try writer.print("{d},", .{shape_index});
    try csvEscape(writer, shape.label);
    try writer.print(",{d},{d},{d},", .{ shape.m, shape.n, shape.k });
    try csvEscape(writer, lib_name);
    const measured_gflops = gflops(result.best_ns, flopFactorForKind(kind), shape.m, shape.n, shape.k);
    try writer.print(",{d:.6},{d},{d},{d},{d},{d},", .{
        measured_gflops,
        result.best_ns,
        result.median_ns,
        result.p95_ns,
        result.max_ns,
        reps,
    });
    try csvEscape(writer, result.check);
    try writer.writeByte('\n');
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    var zynum_blas_path: ?[]const u8 = null;
    var accel_path: ?[]const u8 = null;
    var openblas_path: ?[]const u8 = null;
    var mkl_path: ?[]const u8 = null;
    var aocl_blis_path: ?[]const u8 = null;
    var csv_path: ?[]const u8 = null;
    var reps: usize = 5;
    var custom_shapes: [max_custom_shapes]Shape = undefined;
    var custom_shape_count: usize = 0;
    var kind_filter_set = false;
    var run_sgemm = true;
    var run_dgemm = true;
    var run_cgemm = true;
    var run_zgemm = true;
    var check = false;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--zynum-blas") or std.mem.eql(u8, arg, "--zynum") or std.mem.eql(u8, arg, "--zig")) {
            zynum_blas_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--accelerate")) {
            accel_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--openblas")) {
            openblas_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--mkl")) {
            mkl_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--aocl-blis") or std.mem.eql(u8, arg, "--aocl")) {
            aocl_blis_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--csv")) {
            csv_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--reps")) {
            reps = try std.fmt.parseInt(usize, args.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--check")) {
            check = true;
        } else if (std.mem.eql(u8, arg, "--kind")) {
            const kind = args.next() orelse return error.MissingValue;
            if (!kind_filter_set) {
                kind_filter_set = true;
                run_sgemm = false;
                run_dgemm = false;
                run_cgemm = false;
                run_zgemm = false;
            }
            if (std.mem.eql(u8, kind, "sgemm")) {
                run_sgemm = true;
            } else if (std.mem.eql(u8, kind, "dgemm")) {
                run_dgemm = true;
            } else if (std.mem.eql(u8, kind, "cgemm")) {
                run_cgemm = true;
            } else if (std.mem.eql(u8, kind, "zgemm")) {
                run_zgemm = true;
            } else {
                usage();
                return error.BadKind;
            }
        } else if (std.mem.eql(u8, arg, "--shape")) {
            if (custom_shape_count == custom_shapes.len) return error.TooManyShapes;
            custom_shapes[custom_shape_count] = try parseShape(args.next() orelse return error.MissingValue);
            custom_shape_count += 1;
        } else {
            usage();
            return error.BadArgument;
        }
    }
    if (zynum_blas_path == null or csv_path == null) {
        usage();
        return error.BadArgument;
    }
    if (!run_sgemm and !run_dgemm and !run_cgemm and !run_zgemm) return error.BadKind;

    var libs: [5]LibSpec = undefined;
    var lib_count: usize = 0;
    libs[lib_count] = .{ .name = "zynum-blas", .path = zynum_blas_path.? };
    lib_count += 1;
    if (accel_path) |path| {
        libs[lib_count] = .{ .name = "Accelerate", .path = path };
        lib_count += 1;
    }
    if (openblas_path) |path| {
        libs[lib_count] = .{ .name = "OpenBLAS", .path = path };
        lib_count += 1;
    }
    if (mkl_path) |path| {
        libs[lib_count] = .{ .name = "MKL", .path = path };
        lib_count += 1;
    }
    if (aocl_blis_path) |path| {
        libs[lib_count] = .{ .name = "AOCL-BLIS", .path = path };
        lib_count += 1;
    }

    var csv_file = try std.Io.Dir.cwd().createFile(init.io, csv_path.?, .{ .truncate = true });
    defer csv_file.close(init.io);
    var buf: [8192]u8 = undefined;
    var writer = csv_file.writer(init.io, &buf);
    try writer.interface.writeAll("kind,shape_index,label,m,n,k,library,gflops,best_ns,median_ns,p95_ns,max_ns,reps,check\n");

    const shapes: []const Shape = if (custom_shape_count == 0) default_shapes[0..] else custom_shapes[0..custom_shape_count];
    // Keep benchmark libraries loaded until process exit. Zynum and comparator
    // BLAS implementations may own process-lifetime worker threads after use.
    for (libs[0..lib_count]) |spec| {
        const lib = try loadLib(spec.name, spec.path);
        std.debug.print("[lib {s}] {d} shapes\n", .{ lib.name, shapes.len });
        for (shapes, 0..) |shape, shape_index| {
            std.debug.print("[{d}/{d}] {s} m={d} n={d} k={d}\n", .{ shape_index + 1, shapes.len, shape.label, shape.m, shape.n, shape.k });
            if (run_sgemm) {
                const sg = try benchGemm(f32, lib.sgemm, allocator, init.io, shape, reps, check);
                try writeCsvRow(&writer.interface, "sgemm", shape_index, shape, lib.name, sg, reps);
            }

            if (run_dgemm) {
                const dg = try benchGemm(f64, lib.dgemm, allocator, init.io, shape, reps, check);
                try writeCsvRow(&writer.interface, "dgemm", shape_index, shape, lib.name, dg, reps);
            }

            if (run_cgemm) {
                const cg = try benchGemm(ComplexF32, lib.cgemm, allocator, init.io, shape, reps, check);
                try writeCsvRow(&writer.interface, "cgemm", shape_index, shape, lib.name, cg, reps);
            }

            if (run_zgemm) {
                const zg = try benchGemm(ComplexF64, lib.zgemm, allocator, init.io, shape, reps, check);
                try writeCsvRow(&writer.interface, "zgemm", shape_index, shape, lib.name, zg, reps);
            }
        }
    }
    try writer.interface.flush();
}
