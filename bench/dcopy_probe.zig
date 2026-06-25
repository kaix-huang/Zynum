// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const BlasInt = i32;
const CopyFn = *const fn (*const BlasInt, [*]const u8, *const BlasInt, [*]u8, *const BlasInt) callconv(.c) void;

fn usage() void {
    std.debug.print("usage: dcopy-probe --lib path [--kind s|d|c|z] [--n elems] [--seconds s]\n", .{});
}

fn fill(x: []u8) void {
    var seed: u64 = 0x1234_5678_9abc_def0;
    for (x) |*v| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        v.* = @truncate(seed >> 32);
    }
}

const CopyKind = struct {
    symbol: [:0]const u8,
    elem_size: usize,
};

fn copyKind(kind: []const u8) !CopyKind {
    if (std.mem.eql(u8, kind, "s")) return .{ .symbol = "scopy_", .elem_size = @sizeOf(f32) };
    if (std.mem.eql(u8, kind, "d")) return .{ .symbol = "dcopy_", .elem_size = @sizeOf(f64) };
    if (std.mem.eql(u8, kind, "c")) return .{ .symbol = "ccopy_", .elem_size = 2 * @sizeOf(f32) };
    if (std.mem.eql(u8, kind, "z")) return .{ .symbol = "zcopy_", .elem_size = 2 * @sizeOf(f64) };
    return error.InvalidKind;
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    var lib_path: ?[]const u8 = null;
    var kind = try copyKind("d");
    var n: usize = 1024 * 1024;
    var seconds: u64 = 20;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--lib")) {
            lib_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--kind")) {
            kind = try copyKind(args.next() orelse return error.MissingValue);
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

    var dyn = try std.DynLib.open(path);
    defer dyn.close();
    const copy_fn = dyn.lookup(CopyFn, kind.symbol) orelse return error.MissingCopy;

    const byte_count = n * kind.elem_size;
    const x = try allocator.alloc(u8, byte_count);
    const y = try allocator.alloc(u8, byte_count);
    fill(x);
    @memset(y, 0);

    var ni: BlasInt = @intCast(n);
    var inc: BlasInt = 1;
    copy_fn(&ni, x.ptr, &inc, y.ptr, &inc);

    const pid = std.c.getpid();
    std.debug.print("pid={d} symbol={s} n={d} elem_size={d} seconds={d} lib={s}\n", .{ pid, kind.symbol, n, kind.elem_size, seconds, path });
    const start = std.Io.Clock.awake.now(init.io).nanoseconds;
    const deadline = start + @as(i128, seconds) * std.time.ns_per_s;
    var iters: u64 = 0;
    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) {
        copy_fn(&ni, x.ptr, &inc, y.ptr, &inc);
    }
    const elapsed_ns = std.Io.Clock.awake.now(init.io).nanoseconds - start;
    var checksum: u64 = 0;
    for (y[0..@min(y.len, 64)]) |v| checksum +%= v;
    const bytes = @as(f64, @floatFromInt(iters)) * @as(f64, @floatFromInt(byte_count)) * 2;
    const gbps = bytes / (@as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s)) / 1.0e9;
    std.debug.print("iters={d} elapsed_ns={d} bandwidth_GBps={d:.3} checksum={d}\n", .{ iters, elapsed_ns, gbps, checksum });
}
