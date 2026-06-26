// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const BlasInt = i32;

const DcopyFn = *const fn (*const BlasInt, [*]const f64, *const BlasInt, [*]f64, *const BlasInt) callconv(.c) void;
const DscalFn = *const fn (*const BlasInt, *const f64, [*]f64, *const BlasInt) callconv(.c) void;
const DaxpyFn = *const fn (*const BlasInt, *const f64, [*]const f64, *const BlasInt, [*]f64, *const BlasInt) callconv(.c) void;
const DdotFn = *const fn (*const BlasInt, [*]const f64, *const BlasInt, [*]const f64, *const BlasInt) callconv(.c) f64;
const DasumFn = *const fn (*const BlasInt, [*]const f64, *const BlasInt) callconv(.c) f64;
const Dnrm2Fn = *const fn (*const BlasInt, [*]const f64, *const BlasInt) callconv(.c) f64;
const DgemvFn = *const fn ([*]const u8, *const BlasInt, *const BlasInt, *const f64, [*]const f64, *const BlasInt, [*]const f64, *const BlasInt, *const f64, [*]f64, *const BlasInt) callconv(.c) void;
const DsymvFn = *const fn ([*]const u8, *const BlasInt, *const f64, [*]const f64, *const BlasInt, [*]const f64, *const BlasInt, *const f64, [*]f64, *const BlasInt) callconv(.c) void;
const DgerFn = *const fn (*const BlasInt, *const BlasInt, *const f64, [*]const f64, *const BlasInt, [*]const f64, *const BlasInt, [*]f64, *const BlasInt) callconv(.c) void;

const Lib = struct {
    name: []const u8,
    dyn: std.DynLib,
    dcopy: DcopyFn,
    dscal: DscalFn,
    daxpy: DaxpyFn,
    ddot: DdotFn,
    dasum: DasumFn,
    dnrm2: Dnrm2Fn,
    dgemv: DgemvFn,
    dsymv: DsymvFn,
    dger: DgerFn,
};

const Result = struct {
    ns: i96,
    work: f64,
};

fn usage() void {
    std.debug.print("usage: level12-sweep --zynum-blas path [--accelerate path] [--openblas path] [--reps n] [--size n] [--case name]\n", .{});
}

fn loadLib(name: []const u8, path: []const u8) !Lib {
    var dyn = try std.DynLib.open(path);
    errdefer dyn.close();
    return .{
        .name = name,
        .dyn = dyn,
        .dcopy = dyn.lookup(DcopyFn, "dcopy_") orelse return error.MissingDcopy,
        .dscal = dyn.lookup(DscalFn, "dscal_") orelse return error.MissingDscal,
        .daxpy = dyn.lookup(DaxpyFn, "daxpy_") orelse return error.MissingDaxpy,
        .ddot = dyn.lookup(DdotFn, "ddot_") orelse return error.MissingDdot,
        .dasum = dyn.lookup(DasumFn, "dasum_") orelse return error.MissingDasum,
        .dnrm2 = dyn.lookup(Dnrm2Fn, "dnrm2_") orelse return error.MissingDnrm2,
        .dgemv = dyn.lookup(DgemvFn, "dgemv_") orelse return error.MissingDgemv,
        .dsymv = dyn.lookup(DsymvFn, "dsymv_") orelse return error.MissingDsymv,
        .dger = dyn.lookup(DgerFn, "dger_") orelse return error.MissingDger,
    };
}

fn fill(x: []f64) void {
    var seed: u64 = 0x1234_5678_9abc_def0;
    for (x) |*v| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        const bits: u32 = @truncate(seed >> 32);
        v.* = @as(f64, @floatFromInt(bits % 1000)) / 1000.0 - 0.5;
    }
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn copySlice(dst: []f64, src: []const f64) void {
    @memcpy(dst, src);
}

fn benchDcopy(allocator: std.mem.Allocator, io: std.Io, lib: Lib, n: usize, reps: usize) !Result {
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    const y = try allocator.alloc(f64, n);
    defer allocator.free(y);
    fill(x);
    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    lib.dcopy(&ni, x.ptr, &inc, y.ptr, &inc);
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        const start = nowNs(io);
        lib.dcopy(&ni, x.ptr, &inc, y.ptr, &inc);
        best = @min(best, nowNs(io) - start);
    }
    return .{ .ns = best, .work = @floatFromInt(n) };
}

fn benchDscal(allocator: std.mem.Allocator, io: std.Io, lib: Lib, n: usize, reps: usize) !Result {
    const orig = try allocator.alloc(f64, n);
    defer allocator.free(orig);
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    fill(orig);
    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    var alpha: f64 = 0.7;
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        copySlice(x, orig);
        const start = nowNs(io);
        lib.dscal(&ni, &alpha, x.ptr, &inc);
        best = @min(best, nowNs(io) - start);
    }
    return .{ .ns = best, .work = @floatFromInt(n) };
}

fn benchDaxpy(allocator: std.mem.Allocator, io: std.Io, lib: Lib, n: usize, reps: usize) !Result {
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    const orig_y = try allocator.alloc(f64, n);
    defer allocator.free(orig_y);
    const y = try allocator.alloc(f64, n);
    defer allocator.free(y);
    fill(x);
    fill(orig_y);
    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    var alpha: f64 = 0.7;
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        copySlice(y, orig_y);
        const start = nowNs(io);
        lib.daxpy(&ni, &alpha, x.ptr, &inc, y.ptr, &inc);
        best = @min(best, nowNs(io) - start);
    }
    return .{ .ns = best, .work = 2.0 * @as(f64, @floatFromInt(n)) };
}

fn benchDdot(allocator: std.mem.Allocator, io: std.Io, lib: Lib, n: usize, reps: usize) !Result {
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    const y = try allocator.alloc(f64, n);
    defer allocator.free(y);
    fill(x);
    fill(y);
    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    var sink: f64 = 0;
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        const start = nowNs(io);
        sink += lib.ddot(&ni, x.ptr, &inc, y.ptr, &inc);
        best = @min(best, nowNs(io) - start);
    }
    std.mem.doNotOptimizeAway(sink);
    return .{ .ns = best, .work = 2.0 * @as(f64, @floatFromInt(n)) };
}

fn benchDasum(allocator: std.mem.Allocator, io: std.Io, lib: Lib, n: usize, reps: usize) !Result {
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    fill(x);
    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    var sink: f64 = 0;
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        const start = nowNs(io);
        sink += lib.dasum(&ni, x.ptr, &inc);
        best = @min(best, nowNs(io) - start);
    }
    std.mem.doNotOptimizeAway(sink);
    return .{ .ns = best, .work = @floatFromInt(n) };
}

fn benchDnrm2(allocator: std.mem.Allocator, io: std.Io, lib: Lib, n: usize, reps: usize) !Result {
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    fill(x);
    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    var sink: f64 = 0;
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        const start = nowNs(io);
        sink += lib.dnrm2(&ni, x.ptr, &inc);
        best = @min(best, nowNs(io) - start);
    }
    std.mem.doNotOptimizeAway(sink);
    return .{ .ns = best, .work = 2.0 * @as(f64, @floatFromInt(n)) };
}

fn benchDgemv(allocator: std.mem.Allocator, io: std.Io, lib: Lib, n: usize, reps: usize, trans: u8) !Result {
    const a = try allocator.alloc(f64, n * n);
    defer allocator.free(a);
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    const orig_y = try allocator.alloc(f64, n);
    defer allocator.free(orig_y);
    const y = try allocator.alloc(f64, n);
    defer allocator.free(y);
    fill(a);
    fill(x);
    fill(orig_y);
    var trans_buf = [_]u8{trans};
    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    var alpha: f64 = 0.7;
    var beta: f64 = 0.3;
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        copySlice(y, orig_y);
        const start = nowNs(io);
        lib.dgemv(&trans_buf, &ni, &ni, &alpha, a.ptr, &ni, x.ptr, &inc, &beta, y.ptr, &inc);
        best = @min(best, nowNs(io) - start);
    }
    return .{ .ns = best, .work = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) };
}

fn benchDsymv(allocator: std.mem.Allocator, io: std.Io, lib: Lib, n: usize, reps: usize) !Result {
    const a = try allocator.alloc(f64, n * n);
    defer allocator.free(a);
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    const orig_y = try allocator.alloc(f64, n);
    defer allocator.free(orig_y);
    const y = try allocator.alloc(f64, n);
    defer allocator.free(y);
    fill(a);
    fill(x);
    fill(orig_y);
    var uplo = [_]u8{'U'};
    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    var alpha: f64 = 0.7;
    var beta: f64 = 0.3;
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        copySlice(y, orig_y);
        const start = nowNs(io);
        lib.dsymv(&uplo, &ni, &alpha, a.ptr, &ni, x.ptr, &inc, &beta, y.ptr, &inc);
        best = @min(best, nowNs(io) - start);
    }
    return .{ .ns = best, .work = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) };
}

fn benchDger(allocator: std.mem.Allocator, io: std.Io, lib: Lib, n: usize, reps: usize) !Result {
    const x = try allocator.alloc(f64, n);
    defer allocator.free(x);
    const y = try allocator.alloc(f64, n);
    defer allocator.free(y);
    const orig_a = try allocator.alloc(f64, n * n);
    defer allocator.free(orig_a);
    const a = try allocator.alloc(f64, n * n);
    defer allocator.free(a);
    fill(x);
    fill(y);
    fill(orig_a);
    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    var alpha: f64 = 0.7;
    var best: i96 = std.math.maxInt(i96);
    for (0..reps) |_| {
        copySlice(a, orig_a);
        const start = nowNs(io);
        lib.dger(&ni, &ni, &alpha, x.ptr, &inc, y.ptr, &inc, a.ptr, &ni);
        best = @min(best, nowNs(io) - start);
    }
    return .{ .ns = best, .work = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) };
}

fn emitResult(writer: *std.Io.Writer, name: []const u8, lib_name: []const u8, result: Result) !void {
    const seconds = @as(f64, @floatFromInt(result.ns)) / 1e9;
    const rate = result.work / seconds / 1e9;
    try writer.print("{s},{s},{d},{d:.6}\n", .{ name, lib_name, result.ns, rate });
}

fn shouldRun(case_filter: ?[]const u8, name: []const u8) bool {
    return case_filter == null or std.mem.eql(u8, case_filter.?, name);
}

fn knownCase(name: []const u8) bool {
    const cases = [_][]const u8{
        "dcopy",
        "dscal",
        "daxpy",
        "ddot",
        "dasum",
        "dnrm2",
        "dgemv_n",
        "dgemv_t",
        "dsymv",
        "dger",
    };
    for (cases) |case| {
        if (std.mem.eql(u8, name, case)) return true;
    }
    return false;
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    var zynum_blas_path: ?[]const u8 = null;
    var accel_path: ?[]const u8 = null;
    var openblas_path: ?[]const u8 = null;
    var reps: usize = 20;
    var size: usize = 1024;
    var case_filter: ?[]const u8 = null;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--zynum-blas") or std.mem.eql(u8, arg, "--zynum")) {
            zynum_blas_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--accelerate")) {
            accel_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--openblas")) {
            openblas_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--reps")) {
            reps = try std.fmt.parseInt(usize, args.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--size")) {
            size = try std.fmt.parseInt(usize, args.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--case")) {
            case_filter = args.next() orelse return error.MissingValue;
        } else {
            usage();
            return error.BadArgument;
        }
    }
    if (zynum_blas_path == null) {
        usage();
        return error.BadArgument;
    }
    if (case_filter) |case| {
        if (!knownCase(case)) {
            usage();
            return error.BadArgument;
        }
    }

    var libs: [3]Lib = undefined;
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
    // Keep benchmark libraries loaded until process exit. Zynum and comparator
    // BLAS implementations may own process-lifetime worker threads after use.

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout_writer.interface.print("case,library,best_ns,rate_Gops\n", .{});
    for (libs[0..lib_count]) |lib| {
        if (shouldRun(case_filter, "dcopy")) try emitResult(&stdout_writer.interface, "dcopy", lib.name, try benchDcopy(allocator, init.io, lib, size * size, reps));
        if (shouldRun(case_filter, "dscal")) try emitResult(&stdout_writer.interface, "dscal", lib.name, try benchDscal(allocator, init.io, lib, size * size, reps));
        if (shouldRun(case_filter, "daxpy")) try emitResult(&stdout_writer.interface, "daxpy", lib.name, try benchDaxpy(allocator, init.io, lib, size * size, reps));
        if (shouldRun(case_filter, "ddot")) try emitResult(&stdout_writer.interface, "ddot", lib.name, try benchDdot(allocator, init.io, lib, size * size, reps));
        if (shouldRun(case_filter, "dasum")) try emitResult(&stdout_writer.interface, "dasum", lib.name, try benchDasum(allocator, init.io, lib, size * size, reps));
        if (shouldRun(case_filter, "dnrm2")) try emitResult(&stdout_writer.interface, "dnrm2", lib.name, try benchDnrm2(allocator, init.io, lib, size * size, reps));
        if (shouldRun(case_filter, "dgemv_n")) try emitResult(&stdout_writer.interface, "dgemv_n", lib.name, try benchDgemv(allocator, init.io, lib, size, reps, 'N'));
        if (shouldRun(case_filter, "dgemv_t")) try emitResult(&stdout_writer.interface, "dgemv_t", lib.name, try benchDgemv(allocator, init.io, lib, size, reps, 'T'));
        if (shouldRun(case_filter, "dsymv")) try emitResult(&stdout_writer.interface, "dsymv", lib.name, try benchDsymv(allocator, init.io, lib, size, reps));
        if (shouldRun(case_filter, "dger")) try emitResult(&stdout_writer.interface, "dger", lib.name, try benchDger(allocator, init.io, lib, size, reps));
    }
    try stdout_writer.flush();
}
