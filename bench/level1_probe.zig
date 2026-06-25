// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const BlasInt = i32;
const C32 = extern struct { re: f32, im: f32 };
const C64 = extern struct { re: f64, im: f64 };

const SscalFn = *const fn (*const BlasInt, *const f32, [*]f32, *const BlasInt) callconv(.c) void;
const DscalFn = *const fn (*const BlasInt, *const f64, [*]f64, *const BlasInt) callconv(.c) void;
const CsscalFn = *const fn (*const BlasInt, *const f32, [*]C32, *const BlasInt) callconv(.c) void;
const ZdscalFn = *const fn (*const BlasInt, *const f64, [*]C64, *const BlasInt) callconv(.c) void;
const CscalFn = *const fn (*const BlasInt, *const C32, [*]C32, *const BlasInt) callconv(.c) void;
const ZscalFn = *const fn (*const BlasInt, *const C64, [*]C64, *const BlasInt) callconv(.c) void;

const SaxpyFn = *const fn (*const BlasInt, *const f32, [*]const f32, *const BlasInt, [*]f32, *const BlasInt) callconv(.c) void;
const DaxpyFn = *const fn (*const BlasInt, *const f64, [*]const f64, *const BlasInt, [*]f64, *const BlasInt) callconv(.c) void;
const CaxpyFn = *const fn (*const BlasInt, *const C32, [*]const C32, *const BlasInt, [*]C32, *const BlasInt) callconv(.c) void;
const ZaxpyFn = *const fn (*const BlasInt, *const C64, [*]const C64, *const BlasInt, [*]C64, *const BlasInt) callconv(.c) void;
const CaxpbyFn = *const fn (*const BlasInt, *const C32, [*]const C32, *const BlasInt, *const C32, [*]C32, *const BlasInt) callconv(.c) void;
const ZaxpbyFn = *const fn (*const BlasInt, *const C64, [*]const C64, *const BlasInt, *const C64, [*]C64, *const BlasInt) callconv(.c) void;

const SdotFn = *const fn (*const BlasInt, [*]const f32, *const BlasInt, [*]const f32, *const BlasInt) callconv(.c) f32;
const DdotFn = *const fn (*const BlasInt, [*]const f64, *const BlasInt, [*]const f64, *const BlasInt) callconv(.c) f64;
const CdotSubFn = *const fn (*const BlasInt, [*]const C32, *const BlasInt, [*]const C32, *const BlasInt, *C32) callconv(.c) void;
const ZdotSubFn = *const fn (*const BlasInt, [*]const C64, *const BlasInt, [*]const C64, *const BlasInt, *C64) callconv(.c) void;
const CblasCdotSubFn = *const fn (BlasInt, [*]const C32, BlasInt, [*]const C32, BlasInt, *C32) callconv(.c) void;
const CblasZdotSubFn = *const fn (BlasInt, [*]const C64, BlasInt, [*]const C64, BlasInt, *C64) callconv(.c) void;

const SasumFn = *const fn (*const BlasInt, [*]const f32, *const BlasInt) callconv(.c) f32;
const DasumFn = *const fn (*const BlasInt, [*]const f64, *const BlasInt) callconv(.c) f64;
const ScasumFn = *const fn (*const BlasInt, [*]const C32, *const BlasInt) callconv(.c) f32;
const DzasumFn = *const fn (*const BlasInt, [*]const C64, *const BlasInt) callconv(.c) f64;

const Snrm2Fn = *const fn (*const BlasInt, [*]const f32, *const BlasInt) callconv(.c) f32;
const Dnrm2Fn = *const fn (*const BlasInt, [*]const f64, *const BlasInt) callconv(.c) f64;
const Scnrm2Fn = *const fn (*const BlasInt, [*]const C32, *const BlasInt) callconv(.c) f32;
const Dznrm2Fn = *const fn (*const BlasInt, [*]const C64, *const BlasInt) callconv(.c) f64;

const Op = enum {
    sscal,
    dscal,
    csscal,
    zdscal,
    cscal,
    zscal,
    saxpy,
    daxpy,
    caxpy,
    zaxpy,
    caxpby,
    zaxpby,
    sdot,
    ddot,
    cdotu,
    zdotu,
    cdotc,
    zdotc,
    sasum,
    dasum,
    scasum,
    dzasum,
    snrm2,
    dnrm2,
    scnrm2,
    dznrm2,
};

fn usage() void {
    std.debug.print(
        \\usage: level1-probe --lib path --op OP [--n elems] [--seconds s]
        \\OP: sscal dscal csscal zdscal cscal zscal saxpy daxpy caxpy zaxpy caxpby zaxpby
        \\    sdot ddot cdotu zdotu cdotc zdotc sasum dasum scasum dzasum
        \\    snrm2 dnrm2 scnrm2 dznrm2
        \\
    , .{});
}

fn parseOp(name: []const u8) !Op {
    inline for (@typeInfo(Op).@"enum".fields) |field| {
        if (std.mem.eql(u8, name, field.name)) return @enumFromInt(field.value);
    }
    return error.InvalidOp;
}

fn fillF32(x: []f32) void {
    var seed: u64 = 0x1234_5678_9abc_def0;
    for (x) |*v| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        const bits: u32 = @truncate(seed >> 32);
        v.* = @as(f32, @floatFromInt(bits % 1000)) / 1000.0 - 0.5;
    }
}

fn fillF64(x: []f64) void {
    var seed: u64 = 0x1234_5678_9abc_def0;
    for (x) |*v| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        const bits: u32 = @truncate(seed >> 32);
        v.* = @as(f64, @floatFromInt(bits % 1000)) / 1000.0 - 0.5;
    }
}

fn fillC32(x: []C32) void {
    const raw: []f32 = @as([*]f32, @ptrCast(x.ptr))[0 .. 2 * x.len];
    fillF32(raw);
}

fn fillC64(x: []C64) void {
    const raw: []f64 = @as([*]f64, @ptrCast(x.ptr))[0 .. 2 * x.len];
    fillF64(raw);
}

fn workPerIter(op: Op, n: usize) f64 {
    const nf: f64 = @floatFromInt(n);
    return switch (op) {
        .sscal, .dscal, .sasum, .dasum => nf,
        .csscal, .zdscal, .scasum, .dzasum => 2.0 * nf,
        .saxpy, .daxpy, .sdot, .ddot, .snrm2, .dnrm2 => 2.0 * nf,
        .cscal, .zscal => 6.0 * nf,
        .caxpy, .zaxpy, .cdotu, .zdotu, .cdotc, .zdotc => 8.0 * nf,
        .caxpby, .zaxpby => 14.0 * nf,
        .scnrm2, .dznrm2 => 4.0 * nf,
    };
}

fn bytesPerIter(op: Op, n: usize) f64 {
    const nf: f64 = @floatFromInt(n);
    return switch (op) {
        .sscal => 2.0 * nf * @sizeOf(f32),
        .dscal => 2.0 * nf * @sizeOf(f64),
        .csscal, .cscal => 2.0 * nf * @sizeOf(C32),
        .zdscal, .zscal => 2.0 * nf * @sizeOf(C64),
        .saxpy, .sdot => 2.0 * nf * @sizeOf(f32),
        .daxpy, .ddot => 2.0 * nf * @sizeOf(f64),
        .caxpy, .cdotu, .cdotc => 2.0 * nf * @sizeOf(C32),
        .zaxpy, .zdotu, .zdotc => 2.0 * nf * @sizeOf(C64),
        .caxpby => 3.0 * nf * @sizeOf(C32),
        .zaxpby => 3.0 * nf * @sizeOf(C64),
        .sasum, .snrm2 => nf * @sizeOf(f32),
        .dasum, .dnrm2 => nf * @sizeOf(f64),
        .scasum, .scnrm2 => nf * @sizeOf(C32),
        .dzasum, .dznrm2 => nf * @sizeOf(C64),
    };
}

fn checksumF32(x: []const f32) f64 {
    var sum: f64 = 0;
    for (x[0..@min(x.len, 16)]) |v| sum += v;
    return sum;
}

fn checksumF64(x: []const f64) f64 {
    var sum: f64 = 0;
    for (x[0..@min(x.len, 16)]) |v| sum += v;
    return sum;
}

fn checksumC32(x: []const C32) f64 {
    var sum: f64 = 0;
    for (x[0..@min(x.len, 16)]) |v| sum += v.re + v.im;
    return sum;
}

fn checksumC64(x: []const C64) f64 {
    var sum: f64 = 0;
    for (x[0..@min(x.len, 16)]) |v| sum += v.re + v.im;
    return sum;
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    var lib_path: ?[]const u8 = null;
    var op: ?Op = null;
    var n: usize = 1024 * 1024;
    var seconds: u64 = 10;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--lib")) {
            lib_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--op")) {
            op = try parseOp(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--n")) {
            n = try std.fmt.parseInt(usize, args.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--seconds")) {
            seconds = try std.fmt.parseInt(u64, args.next() orelse return error.MissingValue, 10);
        } else {
            usage();
            return error.InvalidArgument;
        }
    }
    const path = lib_path orelse {
        usage();
        return error.MissingLib;
    };
    const selected = op orelse {
        usage();
        return error.MissingOp;
    };

    var dyn = try std.DynLib.open(path);
    defer dyn.close();

    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    var sink: f64 = 0;
    var checksum: f64 = 0;

    const pid = std.c.getpid();
    std.debug.print("pid={d} op={s} n={d} seconds={d} lib={s}\n", .{ pid, @tagName(selected), n, seconds, path });
    const start = std.Io.Clock.awake.now(init.io).nanoseconds;
    const deadline = start + @as(i128, seconds) * std.time.ns_per_s;
    var iters: u64 = 0;

    switch (selected) {
        .sscal, .saxpy, .sdot, .sasum, .snrm2 => {
            const x = try allocator.alloc(f32, n);
            const y = try allocator.alloc(f32, n);
            fillF32(x);
            fillF32(y);
            var alpha: f32 = 1.0000001;
            switch (selected) {
                .sscal => {
                    const f = dyn.lookup(SscalFn, "sscal_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha, x.ptr, &inc);
                },
                .saxpy => {
                    const f = dyn.lookup(SaxpyFn, "saxpy_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha, x.ptr, &inc, y.ptr, &inc);
                },
                .sdot => {
                    const f = dyn.lookup(SdotFn, "sdot_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.ptr, &inc, y.ptr, &inc);
                },
                .sasum => {
                    const f = dyn.lookup(SasumFn, "sasum_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.ptr, &inc);
                },
                .snrm2 => {
                    const f = dyn.lookup(Snrm2Fn, "snrm2_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.ptr, &inc);
                },
                else => unreachable,
            }
            checksum = sink + checksumF32(x) + checksumF32(y);
        },
        .dscal, .daxpy, .ddot, .dasum, .dnrm2 => {
            const x = try allocator.alloc(f64, n);
            const y = try allocator.alloc(f64, n);
            fillF64(x);
            fillF64(y);
            var alpha: f64 = 1.0000000000000002;
            switch (selected) {
                .dscal => {
                    const f = dyn.lookup(DscalFn, "dscal_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha, x.ptr, &inc);
                },
                .daxpy => {
                    const f = dyn.lookup(DaxpyFn, "daxpy_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha, x.ptr, &inc, y.ptr, &inc);
                },
                .ddot => {
                    const f = dyn.lookup(DdotFn, "ddot_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.ptr, &inc, y.ptr, &inc);
                },
                .dasum => {
                    const f = dyn.lookup(DasumFn, "dasum_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.ptr, &inc);
                },
                .dnrm2 => {
                    const f = dyn.lookup(Dnrm2Fn, "dnrm2_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.ptr, &inc);
                },
                else => unreachable,
            }
            checksum = sink + checksumF64(x) + checksumF64(y);
        },
        .csscal, .cscal, .caxpy, .caxpby, .cdotu, .cdotc, .scasum, .scnrm2 => {
            const x = try allocator.alloc(C32, n);
            const y = try allocator.alloc(C32, n);
            fillC32(x);
            fillC32(y);
            var alpha_r: f32 = 1.0000001;
            var alpha_c: C32 = .{ .re = 1.0000001, .im = 0.125 };
            var beta_c: C32 = .{ .re = 0.875, .im = -0.0625 };
            switch (selected) {
                .csscal => {
                    const f = dyn.lookup(CsscalFn, "csscal_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_r, x.ptr, &inc);
                },
                .cscal => {
                    const f = dyn.lookup(CscalFn, "cscal_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.ptr, &inc);
                },
                .caxpy => {
                    const f = dyn.lookup(CaxpyFn, "caxpy_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.ptr, &inc, y.ptr, &inc);
                },
                .caxpby => {
                    const f = dyn.lookup(CaxpbyFn, "caxpby_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.ptr, &inc, &beta_c, y.ptr, &inc);
                },
                .cdotu, .cdotc => {
                    const sym = if (selected == .cdotu) "cdotu_sub_" else "cdotc_sub_";
                    const cblas_sym = if (selected == .cdotu) "cblas_cdotu_sub" else "cblas_cdotc_sub";
                    var out: C32 = undefined;
                    if (dyn.lookup(CdotSubFn, sym)) |f| {
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) {
                            f(&ni, x.ptr, &inc, y.ptr, &inc, &out);
                            sink += out.re + out.im;
                        }
                    } else if (dyn.lookup(CblasCdotSubFn, cblas_sym)) |f| {
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) {
                            f(ni, x.ptr, inc, y.ptr, inc, &out);
                            sink += out.re + out.im;
                        }
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .scasum => {
                    const f = dyn.lookup(ScasumFn, "scasum_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.ptr, &inc);
                },
                .scnrm2 => {
                    const f = dyn.lookup(Scnrm2Fn, "scnrm2_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.ptr, &inc);
                },
                else => unreachable,
            }
            checksum = sink + checksumC32(x) + checksumC32(y);
        },
        .zdscal, .zscal, .zaxpy, .zaxpby, .zdotu, .zdotc, .dzasum, .dznrm2 => {
            const x = try allocator.alloc(C64, n);
            const y = try allocator.alloc(C64, n);
            fillC64(x);
            fillC64(y);
            var alpha_r: f64 = 1.0000000000000002;
            var alpha_c: C64 = .{ .re = 1.0000000000000002, .im = 0.125 };
            var beta_c: C64 = .{ .re = 0.875, .im = -0.0625 };
            switch (selected) {
                .zdscal => {
                    const f = dyn.lookup(ZdscalFn, "zdscal_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_r, x.ptr, &inc);
                },
                .zscal => {
                    const f = dyn.lookup(ZscalFn, "zscal_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.ptr, &inc);
                },
                .zaxpy => {
                    const f = dyn.lookup(ZaxpyFn, "zaxpy_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.ptr, &inc, y.ptr, &inc);
                },
                .zaxpby => {
                    const f = dyn.lookup(ZaxpbyFn, "zaxpby_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.ptr, &inc, &beta_c, y.ptr, &inc);
                },
                .zdotu, .zdotc => {
                    const sym = if (selected == .zdotu) "zdotu_sub_" else "zdotc_sub_";
                    const cblas_sym = if (selected == .zdotu) "cblas_zdotu_sub" else "cblas_zdotc_sub";
                    var out: C64 = undefined;
                    if (dyn.lookup(ZdotSubFn, sym)) |f| {
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) {
                            f(&ni, x.ptr, &inc, y.ptr, &inc, &out);
                            sink += out.re + out.im;
                        }
                    } else if (dyn.lookup(CblasZdotSubFn, cblas_sym)) |f| {
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) {
                            f(ni, x.ptr, inc, y.ptr, inc, &out);
                            sink += out.re + out.im;
                        }
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .dzasum => {
                    const f = dyn.lookup(DzasumFn, "dzasum_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.ptr, &inc);
                },
                .dznrm2 => {
                    const f = dyn.lookup(Dznrm2Fn, "dznrm2_") orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.ptr, &inc);
                },
                else => unreachable,
            }
            checksum = sink + checksumC64(x) + checksumC64(y);
        },
    }

    const elapsed_ns = std.Io.Clock.awake.now(init.io).nanoseconds - start;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);
    const gops = @as(f64, @floatFromInt(iters)) * workPerIter(selected, n) / elapsed_s / 1.0e9;
    const gbps = @as(f64, @floatFromInt(iters)) * bytesPerIter(selected, n) / elapsed_s / 1.0e9;
    std.debug.print("iters={d} elapsed_ns={d} rate_Gops={d:.3} bandwidth_GBps={d:.3} checksum={d:.6}\n", .{ iters, elapsed_ns, gops, gbps, checksum });
}
