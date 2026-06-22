// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const BlasInt = i32;
const DgemmFn = *const fn ([*]const u8, [*]const u8, *const BlasInt, *const BlasInt, *const BlasInt, *const f64, [*]const f64, *const BlasInt, [*]const f64, *const BlasInt, *const f64, [*]f64, *const BlasInt) callconv(.c) void;
const SgemmFn = *const fn ([*]const u8, [*]const u8, *const BlasInt, *const BlasInt, *const BlasInt, *const f32, [*]const f32, *const BlasInt, [*]const f32, *const BlasInt, *const f32, [*]f32, *const BlasInt) callconv(.c) void;
const DaxpyFn = *const fn (*const BlasInt, *const f64, [*]const f64, *const BlasInt, [*]f64, *const BlasInt) callconv(.c) void;

const Lib = struct {
    name: []const u8,
    path: []const u8,
    dyn: std.DynLib,
    dgemm: DgemmFn,
    sgemm: SgemmFn,
    daxpy: DaxpyFn,
};

fn usage() void {
    std.debug.print("usage: bench-zynum-blas --zynum-blas path [--accelerate path] [--openblas path] [--mkl path] [--size n] [--reps n]\n", .{});
}

fn loadLib(name: []const u8, path: []const u8) !Lib {
    var dyn = try std.DynLib.open(path);
    errdefer dyn.close();
    return .{
        .name = name,
        .path = path,
        .dyn = dyn,
        .dgemm = dyn.lookup(DgemmFn, "dgemm_") orelse return error.MissingDgemm,
        .sgemm = dyn.lookup(SgemmFn, "sgemm_") orelse return error.MissingSgemm,
        .daxpy = dyn.lookup(DaxpyFn, "daxpy_") orelse return error.MissingDaxpy,
    };
}

fn fill(comptime T: type, x: []T) void {
    var seed: u64 = 0x1234_5678_9abc_def0;
    for (x) |*v| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        const bits: u32 = @truncate(seed >> 32);
        const r = @as(f64, @floatFromInt(bits % 1000)) / 1000.0 - 0.5;
        v.* = @floatCast(r);
    }
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn gflops(elapsed_ns: i96, flops: f64) f64 {
    return flops / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9) / 1e9;
}

fn benchDgemm(allocator: std.mem.Allocator, io: std.Io, lib: Lib, size: usize, reps: usize) !f64 {
    const n_i: BlasInt = @intCast(size);
    const len = size * size;
    const a = try allocator.alloc(f64, len);
    defer allocator.free(a);
    const b = try allocator.alloc(f64, len);
    defer allocator.free(b);
    const c = try allocator.alloc(f64, len);
    defer allocator.free(c);
    fill(f64, a);
    fill(f64, b);
    @memset(c, 0);
    var ta = [_]u8{'N'};
    var tb = [_]u8{'N'};
    var alpha: f64 = 1;
    var beta: f64 = 0;
    lib.dgemm(&ta, &tb, &n_i, &n_i, &n_i, &alpha, a.ptr, &n_i, b.ptr, &n_i, &beta, c.ptr, &n_i);
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        @memset(c, 0);
        const start = nowNs(io);
        lib.dgemm(&ta, &tb, &n_i, &n_i, &n_i, &alpha, a.ptr, &n_i, b.ptr, &n_i, &beta, c.ptr, &n_i);
        const end = nowNs(io);
        best = @min(best, end - start);
    }
    return gflops(best, 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)));
}

fn benchSgemm(allocator: std.mem.Allocator, io: std.Io, lib: Lib, size: usize, reps: usize) !f64 {
    const n_i: BlasInt = @intCast(size);
    const len = size * size;
    const a = try allocator.alloc(f32, len);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, len);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, len);
    defer allocator.free(c);
    fill(f32, a);
    fill(f32, b);
    @memset(c, 0);
    var ta = [_]u8{'N'};
    var tb = [_]u8{'N'};
    var alpha: f32 = 1;
    var beta: f32 = 0;
    lib.sgemm(&ta, &tb, &n_i, &n_i, &n_i, &alpha, a.ptr, &n_i, b.ptr, &n_i, &beta, c.ptr, &n_i);
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        @memset(c, 0);
        const start = nowNs(io);
        lib.sgemm(&ta, &tb, &n_i, &n_i, &n_i, &alpha, a.ptr, &n_i, b.ptr, &n_i, &beta, c.ptr, &n_i);
        const end = nowNs(io);
        best = @min(best, end - start);
    }
    return gflops(best, 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)));
}

fn benchDaxpy(allocator: std.mem.Allocator, io: std.Io, lib: Lib, size: usize, reps: usize) !f64 {
    const n_i: BlasInt = @intCast(size);
    const x = try allocator.alloc(f64, size);
    defer allocator.free(x);
    const y = try allocator.alloc(f64, size);
    defer allocator.free(y);
    fill(f64, x);
    fill(f64, y);
    var inc: BlasInt = 1;
    var alpha: f64 = 0.7;
    lib.daxpy(&n_i, &alpha, x.ptr, &inc, y.ptr, &inc);
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        fill(f64, y);
        const start = nowNs(io);
        lib.daxpy(&n_i, &alpha, x.ptr, &inc, y.ptr, &inc);
        const end = nowNs(io);
        best = @min(best, end - start);
    }
    return gflops(best, 2.0 * @as(f64, @floatFromInt(size)));
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;

    var zynum_blas_path: ?[]const u8 = null;
    var accel_path: ?[]const u8 = null;
    var openblas_path: ?[]const u8 = null;
    var mkl_path: ?[]const u8 = null;
    var size: usize = 512;
    var reps: usize = 3;

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
        } else if (std.mem.eql(u8, arg, "--size")) {
            size = try std.fmt.parseInt(usize, args.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--reps")) {
            reps = try std.fmt.parseInt(usize, args.next() orelse return error.MissingValue, 10);
        } else {
            usage();
            return error.BadArgument;
        }
    }
    if (zynum_blas_path == null) {
        usage();
        return error.BadArgument;
    }

    var libs: [4]Lib = undefined;
    var lib_count: usize = 0;
    libs[lib_count] = try loadLib("zynum-blas", zynum_blas_path.?);
    lib_count += 1;
    if (accel_path) |path| {
        libs[lib_count] = try loadLib("Accelerate", path);
        lib_count += 1;
    }
    if (openblas_path) |path| {
        libs[lib_count] = try loadLib("OpenBLAS", path);
        lib_count += 1;
    }
    if (mkl_path) |path| {
        libs[lib_count] = try loadLib("MKL", path);
        lib_count += 1;
    }
    defer for (libs[0..lib_count]) |*lib| lib.dyn.close();

    std.debug.print("size={d} reps={d}\n", .{ size, reps });
    std.debug.print("library       dgemm GF/s   sgemm GF/s   daxpy GF/s\n", .{});
    for (libs[0..lib_count]) |lib| {
        const dg = try benchDgemm(allocator, init.io, lib, size, reps);
        const sg = try benchSgemm(allocator, init.io, lib, size, reps);
        const ax = try benchDaxpy(allocator, init.io, lib, size * size, reps);
        std.debug.print("{s:<12} {d:>10.2}   {d:>10.2}   {d:>10.2}\n", .{ lib.name, dg, sg, ax });
    }
}
