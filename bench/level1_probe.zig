// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const BlasInt = i32;
const C32 = extern struct { re: f32, im: f32 };
const C64 = extern struct { re: f64, im: f64 };

const ShutdownFn = *const fn () callconv(.c) void;
const SscalFn = *const fn (*const BlasInt, *const f32, [*]f32, *const BlasInt) callconv(.c) void;
const DscalFn = *const fn (*const BlasInt, *const f64, [*]f64, *const BlasInt) callconv(.c) void;
const CsscalFn = *const fn (*const BlasInt, *const f32, [*]C32, *const BlasInt) callconv(.c) void;
const ZdscalFn = *const fn (*const BlasInt, *const f64, [*]C64, *const BlasInt) callconv(.c) void;
const CscalFn = *const fn (*const BlasInt, *const C32, [*]C32, *const BlasInt) callconv(.c) void;
const ZscalFn = *const fn (*const BlasInt, *const C64, [*]C64, *const BlasInt) callconv(.c) void;

const ScopyFn = *const fn (*const BlasInt, [*]const f32, *const BlasInt, [*]f32, *const BlasInt) callconv(.c) void;
const DcopyFn = *const fn (*const BlasInt, [*]const f64, *const BlasInt, [*]f64, *const BlasInt) callconv(.c) void;
const CcopyFn = *const fn (*const BlasInt, [*]const C32, *const BlasInt, [*]C32, *const BlasInt) callconv(.c) void;
const ZcopyFn = *const fn (*const BlasInt, [*]const C64, *const BlasInt, [*]C64, *const BlasInt) callconv(.c) void;

const SswapFn = *const fn (*const BlasInt, [*]f32, *const BlasInt, [*]f32, *const BlasInt) callconv(.c) void;
const DswapFn = *const fn (*const BlasInt, [*]f64, *const BlasInt, [*]f64, *const BlasInt) callconv(.c) void;
const CswapFn = *const fn (*const BlasInt, [*]C32, *const BlasInt, [*]C32, *const BlasInt) callconv(.c) void;
const ZswapFn = *const fn (*const BlasInt, [*]C64, *const BlasInt, [*]C64, *const BlasInt) callconv(.c) void;

const SaxpyFn = *const fn (*const BlasInt, *const f32, [*]const f32, *const BlasInt, [*]f32, *const BlasInt) callconv(.c) void;
const DaxpyFn = *const fn (*const BlasInt, *const f64, [*]const f64, *const BlasInt, [*]f64, *const BlasInt) callconv(.c) void;
const CaxpyFn = *const fn (*const BlasInt, *const C32, [*]const C32, *const BlasInt, [*]C32, *const BlasInt) callconv(.c) void;
const ZaxpyFn = *const fn (*const BlasInt, *const C64, [*]const C64, *const BlasInt, [*]C64, *const BlasInt) callconv(.c) void;
const SaxpbyFn = *const fn (*const BlasInt, *const f32, [*]const f32, *const BlasInt, *const f32, [*]f32, *const BlasInt) callconv(.c) void;
const DaxpbyFn = *const fn (*const BlasInt, *const f64, [*]const f64, *const BlasInt, *const f64, [*]f64, *const BlasInt) callconv(.c) void;
const CaxpbyFn = *const fn (*const BlasInt, *const C32, [*]const C32, *const BlasInt, *const C32, [*]C32, *const BlasInt) callconv(.c) void;
const ZaxpbyFn = *const fn (*const BlasInt, *const C64, [*]const C64, *const BlasInt, *const C64, [*]C64, *const BlasInt) callconv(.c) void;
const CblasSaxpbyFn = *const fn (BlasInt, f32, [*]const f32, BlasInt, f32, [*]f32, BlasInt) callconv(.c) void;
const CblasDaxpbyFn = *const fn (BlasInt, f64, [*]const f64, BlasInt, f64, [*]f64, BlasInt) callconv(.c) void;

const SdotFn = *const fn (*const BlasInt, [*]const f32, *const BlasInt, [*]const f32, *const BlasInt) callconv(.c) f32;
const DdotFn = *const fn (*const BlasInt, [*]const f64, *const BlasInt, [*]const f64, *const BlasInt) callconv(.c) f64;
const SdsdotFn = *const fn (*const BlasInt, *const f32, [*]const f32, *const BlasInt, [*]const f32, *const BlasInt) callconv(.c) f32;
const DsdotFn = *const fn (*const BlasInt, [*]const f32, *const BlasInt, [*]const f32, *const BlasInt) callconv(.c) f64;
const CblasSdotFn = *const fn (BlasInt, [*]const f32, BlasInt, [*]const f32, BlasInt) callconv(.c) f32;
const CblasDdotFn = *const fn (BlasInt, [*]const f64, BlasInt, [*]const f64, BlasInt) callconv(.c) f64;
const CblasSdsdotFn = *const fn (BlasInt, f32, [*]const f32, BlasInt, [*]const f32, BlasInt) callconv(.c) f32;
const CblasDsdotFn = *const fn (BlasInt, [*]const f32, BlasInt, [*]const f32, BlasInt) callconv(.c) f64;
const CdotSubFn = *const fn (*const BlasInt, [*]const C32, *const BlasInt, [*]const C32, *const BlasInt, *C32) callconv(.c) void;
const ZdotSubFn = *const fn (*const BlasInt, [*]const C64, *const BlasInt, [*]const C64, *const BlasInt, *C64) callconv(.c) void;
const CblasCdotSubFn = *const fn (BlasInt, [*]const C32, BlasInt, [*]const C32, BlasInt, *C32) callconv(.c) void;
const CblasZdotSubFn = *const fn (BlasInt, [*]const C64, BlasInt, [*]const C64, BlasInt, *C64) callconv(.c) void;

const SasumFn = *const fn (*const BlasInt, [*]const f32, *const BlasInt) callconv(.c) f32;
const DasumFn = *const fn (*const BlasInt, [*]const f64, *const BlasInt) callconv(.c) f64;
const ScasumFn = *const fn (*const BlasInt, [*]const C32, *const BlasInt) callconv(.c) f32;
const DzasumFn = *const fn (*const BlasInt, [*]const C64, *const BlasInt) callconv(.c) f64;
const CblasSasumFn = *const fn (BlasInt, [*]const f32, BlasInt) callconv(.c) f32;
const CblasDasumFn = *const fn (BlasInt, [*]const f64, BlasInt) callconv(.c) f64;
const CblasScasumFn = *const fn (BlasInt, [*]const C32, BlasInt) callconv(.c) f32;
const CblasDzasumFn = *const fn (BlasInt, [*]const C64, BlasInt) callconv(.c) f64;

const Snrm2Fn = *const fn (*const BlasInt, [*]const f32, *const BlasInt) callconv(.c) f32;
const Dnrm2Fn = *const fn (*const BlasInt, [*]const f64, *const BlasInt) callconv(.c) f64;
const Scnrm2Fn = *const fn (*const BlasInt, [*]const C32, *const BlasInt) callconv(.c) f32;
const Dznrm2Fn = *const fn (*const BlasInt, [*]const C64, *const BlasInt) callconv(.c) f64;
const CblasSnrm2Fn = *const fn (BlasInt, [*]const f32, BlasInt) callconv(.c) f32;
const CblasDnrm2Fn = *const fn (BlasInt, [*]const f64, BlasInt) callconv(.c) f64;
const CblasScnrm2Fn = *const fn (BlasInt, [*]const C32, BlasInt) callconv(.c) f32;
const CblasDznrm2Fn = *const fn (BlasInt, [*]const C64, BlasInt) callconv(.c) f64;

const IsamaxFn = *const fn (*const BlasInt, [*]const f32, *const BlasInt) callconv(.c) BlasInt;
const IdamaxFn = *const fn (*const BlasInt, [*]const f64, *const BlasInt) callconv(.c) BlasInt;
const IcamaxFn = *const fn (*const BlasInt, [*]const C32, *const BlasInt) callconv(.c) BlasInt;
const IzamaxFn = *const fn (*const BlasInt, [*]const C64, *const BlasInt) callconv(.c) BlasInt;

const SrotFn = *const fn (*const BlasInt, [*]f32, *const BlasInt, [*]f32, *const BlasInt, *const f32, *const f32) callconv(.c) void;
const DrotFn = *const fn (*const BlasInt, [*]f64, *const BlasInt, [*]f64, *const BlasInt, *const f64, *const f64) callconv(.c) void;
const CsrotFn = *const fn (*const BlasInt, [*]C32, *const BlasInt, [*]C32, *const BlasInt, *const f32, *const f32) callconv(.c) void;
const ZdrotFn = *const fn (*const BlasInt, [*]C64, *const BlasInt, [*]C64, *const BlasInt, *const f64, *const f64) callconv(.c) void;
const SrotmFn = *const fn (*const BlasInt, [*]f32, *const BlasInt, [*]f32, *const BlasInt, [*]const f32) callconv(.c) void;
const DrotmFn = *const fn (*const BlasInt, [*]f64, *const BlasInt, [*]f64, *const BlasInt, [*]const f64) callconv(.c) void;

const Op = enum {
    scopy,
    dcopy,
    ccopy,
    zcopy,
    sscal,
    dscal,
    csscal,
    zdscal,
    cscal,
    zscal,
    sswap,
    dswap,
    cswap,
    zswap,
    saxpy,
    daxpy,
    caxpy,
    zaxpy,
    saxpby,
    daxpby,
    caxpby,
    zaxpby,
    sdot,
    ddot,
    sdsdot,
    dsdot,
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
    isamax,
    idamax,
    icamax,
    izamax,
    srot,
    drot,
    csrot,
    zdrot,
    srotm,
    drotm,
};

fn usage() void {
    std.debug.print(
        \\usage: level1-probe --lib path --op OP [--variant name] [--inc stride | --incx stride --incy stride] [--n elems] [--seconds s]
        \\OP: scopy dcopy ccopy zcopy sscal dscal csscal zdscal cscal zscal
        \\    saxpy daxpy caxpy zaxpy
        \\    saxpby daxpby caxpby zaxpby sdsdot dsdot
        \\    sswap dswap cswap zswap isamax idamax icamax izamax
        \\    sdot ddot cdotu zdotu cdotc zdotc sasum dasum scasum dzasum
        \\    snrm2 dnrm2 scnrm2 dznrm2
        \\    srot drot csrot zdrot srotm drotm
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

fn workPerIter(op: Op, n: usize, variant: []const u8) f64 {
    const nf: f64 = @floatFromInt(n);
    return switch (op) {
        .scopy, .dcopy, .ccopy, .zcopy => nf,
        .sscal, .dscal, .sasum, .dasum, .sswap, .dswap, .cswap, .zswap, .isamax, .idamax, .icamax, .izamax => nf,
        .csscal, .zdscal, .scasum, .dzasum => 2.0 * nf,
        .saxpy, .daxpy, .sdot, .ddot, .sdsdot, .dsdot, .snrm2, .dnrm2 => 2.0 * nf,
        .saxpby, .daxpby => 3.0 * nf,
        .cscal, .zscal => 6.0 * nf,
        .caxpy, .zaxpy, .cdotu, .zdotu, .cdotc, .zdotc => 8.0 * nf,
        .caxpby, .zaxpby => 14.0 * nf,
        .scnrm2, .dznrm2 => 4.0 * nf,
        .srot, .drot => 6.0 * nf,
        .srotm, .drotm => if (std.mem.eql(u8, variant, "flag_0") or std.mem.eql(u8, variant, "flag_p1")) 4.0 * nf else 6.0 * nf,
        .csrot, .zdrot => 12.0 * nf,
    };
}

fn rotmParam(comptime T: type, variant: []const u8) ![5]T {
    const nan = std.math.nan(T);
    const a: T = 1.0 / 1024.0;
    if (std.mem.eql(u8, variant, "default") or std.mem.eql(u8, variant, "flag_m1")) return .{ -1.0, 0.8, -0.6, 0.6, 0.8 };
    if (std.mem.eql(u8, variant, "flag_0")) return .{ 0.0, nan, -a, a, nan };
    if (std.mem.eql(u8, variant, "flag_p1")) return .{ 1.0, a, nan, nan, -a };
    return error.InvalidVariant;
}

const VectorLayout = struct {
    n: usize,
    inc: BlasInt,
    magnitude: usize,
    span: usize,
    start: usize,

    fn init(n: usize, inc: BlasInt) !VectorLayout {
        if (inc == 0) return error.InvalidStride;
        const wide: i64 = inc;
        const magnitude: usize = @intCast(if (wide < 0) -wide else wide);
        if (n == 0) return .{
            .n = 0,
            .inc = inc,
            .magnitude = magnitude,
            .span = 0,
            .start = 0,
        };
        const distance = try std.math.mul(usize, n - 1, magnitude);
        const span = try std.math.add(usize, distance, 1);
        return .{
            .n = n,
            .inc = inc,
            .magnitude = magnitude,
            .span = span,
            .start = if (inc < 0) distance else 0,
        };
    }

    fn index(self: VectorLayout, logical_index: usize) usize {
        std.debug.assert(logical_index < self.n);
        const offset = logical_index * self.magnitude;
        return if (self.inc < 0) self.start - offset else self.start + offset;
    }

    fn isActive(self: VectorLayout, physical_index: usize) bool {
        if (self.n == 0 or physical_index >= self.span) return false;
        const distance = if (self.inc < 0) blk: {
            if (physical_index > self.start) return false;
            break :blk self.start - physical_index;
        } else physical_index;
        return distance % self.magnitude == 0 and distance / self.magnitude < self.n;
    }
};

const guard_elements = 8;

fn GuardedVector(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        allocation: []T,
        data: []T,
        snapshot: []u8,

        fn init(allocator: std.mem.Allocator, span: usize) !Self {
            const allocation_len = try std.math.add(usize, span, 2 * guard_elements);
            const allocation = try allocator.alloc(T, allocation_len);
            errdefer allocator.free(allocation);
            @memset(std.mem.sliceAsBytes(allocation), 0xa5);
            const snapshot = try allocator.alloc(u8, std.mem.sliceAsBytes(allocation).len);
            return .{
                .allocator = allocator,
                .allocation = allocation,
                .data = allocation[guard_elements .. guard_elements + span],
                .snapshot = snapshot,
            };
        }

        fn deinit(self: *Self) void {
            self.allocator.free(self.snapshot);
            self.allocator.free(self.allocation);
        }

        fn capture(self: *Self) void {
            @memcpy(self.snapshot, std.mem.sliceAsBytes(self.allocation));
        }

        fn verify(self: Self, layout: VectorLayout, active_may_change: bool) !void {
            const current = std.mem.sliceAsBytes(self.allocation);
            const element_size = @sizeOf(T);
            for (0..self.allocation.len) |allocation_index| {
                const in_data = allocation_index >= guard_elements and
                    allocation_index < guard_elements + self.data.len;
                const data_index = if (in_data) allocation_index - guard_elements else 0;
                if (active_may_change and in_data and layout.isActive(data_index)) continue;
                const byte_start = allocation_index * element_size;
                if (!std.mem.eql(
                    u8,
                    self.snapshot[byte_start .. byte_start + element_size],
                    current[byte_start .. byte_start + element_size],
                )) return error.GuardOrGapModified;
            }
        }
    };
}

fn mutatesX(op: Op) bool {
    return switch (op) {
        .sscal,
        .dscal,
        .csscal,
        .zdscal,
        .cscal,
        .zscal,
        .sswap,
        .dswap,
        .cswap,
        .zswap,
        .srot,
        .drot,
        .csrot,
        .zdrot,
        .srotm,
        .drotm,
        => true,
        else => false,
    };
}

fn mutatesY(op: Op) bool {
    return switch (op) {
        .scopy,
        .dcopy,
        .ccopy,
        .zcopy,
        .sswap,
        .dswap,
        .cswap,
        .zswap,
        .saxpy,
        .daxpy,
        .caxpy,
        .zaxpy,
        .saxpby,
        .daxpby,
        .caxpby,
        .zaxpby,
        .srot,
        .drot,
        .csrot,
        .zdrot,
        .srotm,
        .drotm,
        => true,
        else => false,
    };
}

fn bytesPerIter(op: Op, n: usize) f64 {
    const nf: f64 = @floatFromInt(n);
    return switch (op) {
        .scopy => 2.0 * nf * @sizeOf(f32),
        .dcopy => 2.0 * nf * @sizeOf(f64),
        .ccopy => 2.0 * nf * @sizeOf(C32),
        .zcopy => 2.0 * nf * @sizeOf(C64),
        .sscal => 2.0 * nf * @sizeOf(f32),
        .dscal => 2.0 * nf * @sizeOf(f64),
        .csscal, .cscal => 2.0 * nf * @sizeOf(C32),
        .zdscal, .zscal => 2.0 * nf * @sizeOf(C64),
        .saxpy, .sdot, .sdsdot, .dsdot => 2.0 * nf * @sizeOf(f32),
        .daxpy, .ddot => 2.0 * nf * @sizeOf(f64),
        .saxpby => 3.0 * nf * @sizeOf(f32),
        .daxpby => 3.0 * nf * @sizeOf(f64),
        .caxpy, .cdotu, .cdotc => 2.0 * nf * @sizeOf(C32),
        .zaxpy, .zdotu, .zdotc => 2.0 * nf * @sizeOf(C64),
        .caxpby => 3.0 * nf * @sizeOf(C32),
        .zaxpby => 3.0 * nf * @sizeOf(C64),
        .sasum, .snrm2 => nf * @sizeOf(f32),
        .dasum, .dnrm2 => nf * @sizeOf(f64),
        .scasum, .scnrm2 => nf * @sizeOf(C32),
        .dzasum, .dznrm2 => nf * @sizeOf(C64),
        .sswap => 4.0 * nf * @sizeOf(f32),
        .dswap => 4.0 * nf * @sizeOf(f64),
        .cswap => 4.0 * nf * @sizeOf(C32),
        .zswap => 4.0 * nf * @sizeOf(C64),
        .isamax => nf * @sizeOf(f32),
        .idamax => nf * @sizeOf(f64),
        .icamax => nf * @sizeOf(C32),
        .izamax => nf * @sizeOf(C64),
        .srot, .srotm => 4.0 * nf * @sizeOf(f32),
        .drot, .drotm => 4.0 * nf * @sizeOf(f64),
        .csrot => 4.0 * nf * @sizeOf(C32),
        .zdrot => 4.0 * nf * @sizeOf(C64),
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
    var variant: []const u8 = "default";
    var n: usize = 1024 * 1024;
    var legacy_inc: BlasInt = 1;
    var incx_override: ?BlasInt = null;
    var incy_override: ?BlasInt = null;
    var seconds: u64 = 10;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--lib")) {
            lib_path = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--op")) {
            op = try parseOp(args.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--variant")) {
            variant = args.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--n")) {
            n = try std.fmt.parseInt(usize, args.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--inc")) {
            legacy_inc = try std.fmt.parseInt(BlasInt, args.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--incx")) {
            incx_override = try std.fmt.parseInt(BlasInt, args.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--incy")) {
            incy_override = try std.fmt.parseInt(BlasInt, args.next() orelse return error.MissingValue, 10);
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
    if (selected != .srotm and selected != .drotm and !std.mem.eql(u8, variant, "default")) return error.InvalidVariant;
    if (n > std.math.maxInt(BlasInt)) return error.InvalidLength;
    var incx: BlasInt = incx_override orelse legacy_inc;
    var incy: BlasInt = incy_override orelse legacy_inc;
    const x_layout = try VectorLayout.init(n, incx);
    const y_layout = try VectorLayout.init(n, incy);

    var dyn = try std.DynLib.open(path);
    defer {
        if (dyn.lookup(ShutdownFn, "zynum_blas_shutdown")) |shutdown| shutdown();
        dyn.close();
    }

    var ni: BlasInt = @intCast(n);
    var sink: f64 = 0;
    var checksum: f64 = 0;
    var actual_symbol: [:0]const u8 = "";
    var abi_surface: []const u8 = "fortran";

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    const pid = std.c.getpid();
    if (incx == incy) {
        try stdout_writer.interface.print("pid={d} op={s} variant={s} inc={d} incx={d} incy={d} n={d} seconds={d} lib={s}\n", .{ pid, @tagName(selected), variant, incx, incx, incy, n, seconds, path });
    } else {
        try stdout_writer.interface.print("pid={d} op={s} variant={s} incx={d} incy={d} n={d} seconds={d} lib={s}\n", .{ pid, @tagName(selected), variant, incx, incy, n, seconds, path });
    }
    const start = std.Io.Clock.awake.now(init.io).nanoseconds;
    const deadline = start + @as(i128, seconds) * std.time.ns_per_s;
    var iters: u64 = 0;
    var elapsed_ns: i128 = 0;

    switch (selected) {
        .scopy, .sscal, .sswap, .saxpy, .saxpby, .sdot, .sdsdot, .dsdot, .sasum, .snrm2, .isamax, .srot, .srotm => {
            var x = try GuardedVector(f32).init(allocator, x_layout.span);
            defer x.deinit();
            var y = try GuardedVector(f32).init(allocator, y_layout.span);
            defer y.deinit();
            fillF32(x.data);
            fillF32(y.data);
            x.capture();
            y.capture();
            var alpha: f32 = 1.0000001;
            var beta: f32 = 0.875;
            var sb: f32 = 0.125;
            var c: f32 = 0.8;
            var s: f32 = 0.6;
            var param = try rotmParam(f32, variant);
            switch (selected) {
                .scopy => {
                    actual_symbol = "scopy_";
                    const f = dyn.lookup(ScopyFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .sscal => {
                    actual_symbol = "sscal_";
                    const f = dyn.lookup(SscalFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha, x.data.ptr, &incx);
                },
                .sswap => {
                    actual_symbol = "sswap_";
                    const f = dyn.lookup(SswapFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .saxpy => {
                    actual_symbol = "saxpy_";
                    const f = dyn.lookup(SaxpyFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .saxpby => {
                    alpha = 0.125;
                    if (dyn.lookup(CblasSaxpbyFn, "cblas_saxpby")) |f| {
                        actual_symbol = "cblas_saxpby";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(ni, alpha, x.data.ptr, incx, beta, y.data.ptr, incy);
                    } else if (dyn.lookup(CblasSaxpbyFn, "catlas_saxpby")) |f| {
                        actual_symbol = "catlas_saxpby";
                        abi_surface = "catlas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(ni, alpha, x.data.ptr, incx, beta, y.data.ptr, incy);
                    } else if (dyn.lookup(SaxpbyFn, "saxpby_")) |f| {
                        actual_symbol = "saxpby_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha, x.data.ptr, &incx, &beta, y.data.ptr, &incy);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .sdot => {
                    if (dyn.lookup(CblasSdotFn, "cblas_sdot")) |f| {
                        actual_symbol = "cblas_sdot";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx, y.data.ptr, incy);
                    } else if (dyn.lookup(SdotFn, "sdot_")) |f| {
                        actual_symbol = "sdot_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .sdsdot => {
                    if (dyn.lookup(CblasSdsdotFn, "cblas_sdsdot")) |f| {
                        actual_symbol = "cblas_sdsdot";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, sb, x.data.ptr, incx, y.data.ptr, incy);
                    } else if (dyn.lookup(SdsdotFn, "sdsdot_")) |f| {
                        actual_symbol = "sdsdot_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, &sb, x.data.ptr, &incx, y.data.ptr, &incy);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .dsdot => {
                    if (dyn.lookup(CblasDsdotFn, "cblas_dsdot")) |f| {
                        actual_symbol = "cblas_dsdot";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx, y.data.ptr, incy);
                    } else if (dyn.lookup(DsdotFn, "dsdot_")) |f| {
                        actual_symbol = "dsdot_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .sasum => {
                    if (dyn.lookup(CblasSasumFn, "cblas_sasum")) |f| {
                        actual_symbol = "cblas_sasum";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx);
                    } else if (dyn.lookup(SasumFn, "sasum_")) |f| {
                        actual_symbol = "sasum_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .snrm2 => {
                    if (dyn.lookup(CblasSnrm2Fn, "cblas_snrm2")) |f| {
                        actual_symbol = "cblas_snrm2";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx);
                    } else if (dyn.lookup(Snrm2Fn, "snrm2_")) |f| {
                        actual_symbol = "snrm2_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .isamax => {
                    actual_symbol = "isamax_";
                    const f = dyn.lookup(IsamaxFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += @floatFromInt(f(&ni, x.data.ptr, &incx));
                },
                .srot => {
                    actual_symbol = "srot_";
                    const f = dyn.lookup(SrotFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy, &c, &s);
                },
                .srotm => {
                    actual_symbol = "srotm_";
                    const f = dyn.lookup(SrotmFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy, param[0..].ptr);
                },
                else => unreachable,
            }
            elapsed_ns = std.Io.Clock.awake.now(init.io).nanoseconds - start;
            try x.verify(x_layout, mutatesX(selected));
            try y.verify(y_layout, mutatesY(selected));
            checksum = sink + checksumF32(x.data) + checksumF32(y.data);
        },
        .dcopy, .dscal, .dswap, .daxpy, .daxpby, .ddot, .dasum, .dnrm2, .idamax, .drot, .drotm => {
            var x = try GuardedVector(f64).init(allocator, x_layout.span);
            defer x.deinit();
            var y = try GuardedVector(f64).init(allocator, y_layout.span);
            defer y.deinit();
            fillF64(x.data);
            fillF64(y.data);
            x.capture();
            y.capture();
            var alpha: f64 = 1.0000000000000002;
            var beta: f64 = 0.875;
            var c: f64 = 0.8;
            var s: f64 = 0.6;
            var param = try rotmParam(f64, variant);
            switch (selected) {
                .dcopy => {
                    actual_symbol = "dcopy_";
                    const f = dyn.lookup(DcopyFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .dscal => {
                    actual_symbol = "dscal_";
                    const f = dyn.lookup(DscalFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha, x.data.ptr, &incx);
                },
                .dswap => {
                    actual_symbol = "dswap_";
                    const f = dyn.lookup(DswapFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .daxpy => {
                    actual_symbol = "daxpy_";
                    const f = dyn.lookup(DaxpyFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .daxpby => {
                    alpha = 0.125;
                    if (dyn.lookup(CblasDaxpbyFn, "cblas_daxpby")) |f| {
                        actual_symbol = "cblas_daxpby";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(ni, alpha, x.data.ptr, incx, beta, y.data.ptr, incy);
                    } else if (dyn.lookup(CblasDaxpbyFn, "catlas_daxpby")) |f| {
                        actual_symbol = "catlas_daxpby";
                        abi_surface = "catlas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(ni, alpha, x.data.ptr, incx, beta, y.data.ptr, incy);
                    } else if (dyn.lookup(DaxpbyFn, "daxpby_")) |f| {
                        actual_symbol = "daxpby_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha, x.data.ptr, &incx, &beta, y.data.ptr, &incy);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .ddot => {
                    if (dyn.lookup(CblasDdotFn, "cblas_ddot")) |f| {
                        actual_symbol = "cblas_ddot";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx, y.data.ptr, incy);
                    } else if (dyn.lookup(DdotFn, "ddot_")) |f| {
                        actual_symbol = "ddot_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .dasum => {
                    if (dyn.lookup(CblasDasumFn, "cblas_dasum")) |f| {
                        actual_symbol = "cblas_dasum";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx);
                    } else if (dyn.lookup(DasumFn, "dasum_")) |f| {
                        actual_symbol = "dasum_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .dnrm2 => {
                    if (dyn.lookup(CblasDnrm2Fn, "cblas_dnrm2")) |f| {
                        actual_symbol = "cblas_dnrm2";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx);
                    } else if (dyn.lookup(Dnrm2Fn, "dnrm2_")) |f| {
                        actual_symbol = "dnrm2_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .idamax => {
                    actual_symbol = "idamax_";
                    const f = dyn.lookup(IdamaxFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += @floatFromInt(f(&ni, x.data.ptr, &incx));
                },
                .drot => {
                    actual_symbol = "drot_";
                    const f = dyn.lookup(DrotFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy, &c, &s);
                },
                .drotm => {
                    actual_symbol = "drotm_";
                    const f = dyn.lookup(DrotmFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy, param[0..].ptr);
                },
                else => unreachable,
            }
            elapsed_ns = std.Io.Clock.awake.now(init.io).nanoseconds - start;
            try x.verify(x_layout, mutatesX(selected));
            try y.verify(y_layout, mutatesY(selected));
            checksum = sink + checksumF64(x.data) + checksumF64(y.data);
        },
        .ccopy, .csscal, .cscal, .cswap, .caxpy, .caxpby, .cdotu, .cdotc, .scasum, .scnrm2, .icamax, .csrot => {
            var x = try GuardedVector(C32).init(allocator, x_layout.span);
            defer x.deinit();
            var y = try GuardedVector(C32).init(allocator, y_layout.span);
            defer y.deinit();
            fillC32(x.data);
            fillC32(y.data);
            x.capture();
            y.capture();
            var alpha_r: f32 = 1.0000001;
            var alpha_c: C32 = .{ .re = 1.0000001, .im = 0.125 };
            var beta_c: C32 = .{ .re = 0.875, .im = -0.0625 };
            var c: f32 = 0.8;
            var s: f32 = 0.6;
            switch (selected) {
                .ccopy => {
                    actual_symbol = "ccopy_";
                    const f = dyn.lookup(CcopyFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .csscal => {
                    actual_symbol = "csscal_";
                    const f = dyn.lookup(CsscalFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_r, x.data.ptr, &incx);
                },
                .cscal => {
                    actual_symbol = "cscal_";
                    const f = dyn.lookup(CscalFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.data.ptr, &incx);
                },
                .cswap => {
                    actual_symbol = "cswap_";
                    const f = dyn.lookup(CswapFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .caxpy => {
                    actual_symbol = "caxpy_";
                    const f = dyn.lookup(CaxpyFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .caxpby => {
                    actual_symbol = "caxpby_";
                    const f = dyn.lookup(CaxpbyFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.data.ptr, &incx, &beta_c, y.data.ptr, &incy);
                },
                .cdotu, .cdotc => {
                    const sym = if (selected == .cdotu) "cdotu_sub_" else "cdotc_sub_";
                    const cblas_sym = if (selected == .cdotu) "cblas_cdotu_sub" else "cblas_cdotc_sub";
                    var out: C32 = undefined;
                    if (dyn.lookup(CdotSubFn, sym)) |f| {
                        actual_symbol = sym;
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) {
                            f(&ni, x.data.ptr, &incx, y.data.ptr, &incy, &out);
                            sink += out.re + out.im;
                        }
                    } else if (dyn.lookup(CblasCdotSubFn, cblas_sym)) |f| {
                        actual_symbol = cblas_sym;
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) {
                            f(ni, x.data.ptr, incx, y.data.ptr, incy, &out);
                            sink += out.re + out.im;
                        }
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .scasum => {
                    if (dyn.lookup(CblasScasumFn, "cblas_scasum")) |f| {
                        actual_symbol = "cblas_scasum";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx);
                    } else if (dyn.lookup(ScasumFn, "scasum_")) |f| {
                        actual_symbol = "scasum_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .scnrm2 => {
                    if (dyn.lookup(CblasScnrm2Fn, "cblas_scnrm2")) |f| {
                        actual_symbol = "cblas_scnrm2";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx);
                    } else if (dyn.lookup(Scnrm2Fn, "scnrm2_")) |f| {
                        actual_symbol = "scnrm2_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .icamax => {
                    actual_symbol = "icamax_";
                    const f = dyn.lookup(IcamaxFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += @floatFromInt(f(&ni, x.data.ptr, &incx));
                },
                .csrot => {
                    actual_symbol = "csrot_";
                    const f = dyn.lookup(CsrotFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy, &c, &s);
                },
                else => unreachable,
            }
            elapsed_ns = std.Io.Clock.awake.now(init.io).nanoseconds - start;
            try x.verify(x_layout, mutatesX(selected));
            try y.verify(y_layout, mutatesY(selected));
            checksum = sink + checksumC32(x.data) + checksumC32(y.data);
        },
        .zcopy, .zdscal, .zscal, .zswap, .zaxpy, .zaxpby, .zdotu, .zdotc, .dzasum, .dznrm2, .izamax, .zdrot => {
            var x = try GuardedVector(C64).init(allocator, x_layout.span);
            defer x.deinit();
            var y = try GuardedVector(C64).init(allocator, y_layout.span);
            defer y.deinit();
            fillC64(x.data);
            fillC64(y.data);
            x.capture();
            y.capture();
            var alpha_r: f64 = 1.0000000000000002;
            var alpha_c: C64 = .{ .re = 1.0000000000000002, .im = 0.125 };
            var beta_c: C64 = .{ .re = 0.875, .im = -0.0625 };
            var c: f64 = 0.8;
            var s: f64 = 0.6;
            switch (selected) {
                .zcopy => {
                    actual_symbol = "zcopy_";
                    const f = dyn.lookup(ZcopyFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .zdscal => {
                    actual_symbol = "zdscal_";
                    const f = dyn.lookup(ZdscalFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_r, x.data.ptr, &incx);
                },
                .zscal => {
                    actual_symbol = "zscal_";
                    const f = dyn.lookup(ZscalFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.data.ptr, &incx);
                },
                .zswap => {
                    actual_symbol = "zswap_";
                    const f = dyn.lookup(ZswapFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .zaxpy => {
                    actual_symbol = "zaxpy_";
                    const f = dyn.lookup(ZaxpyFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.data.ptr, &incx, y.data.ptr, &incy);
                },
                .zaxpby => {
                    actual_symbol = "zaxpby_";
                    const f = dyn.lookup(ZaxpbyFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, &alpha_c, x.data.ptr, &incx, &beta_c, y.data.ptr, &incy);
                },
                .zdotu, .zdotc => {
                    const sym = if (selected == .zdotu) "zdotu_sub_" else "zdotc_sub_";
                    const cblas_sym = if (selected == .zdotu) "cblas_zdotu_sub" else "cblas_zdotc_sub";
                    var out: C64 = undefined;
                    if (dyn.lookup(ZdotSubFn, sym)) |f| {
                        actual_symbol = sym;
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) {
                            f(&ni, x.data.ptr, &incx, y.data.ptr, &incy, &out);
                            sink += out.re + out.im;
                        }
                    } else if (dyn.lookup(CblasZdotSubFn, cblas_sym)) |f| {
                        actual_symbol = cblas_sym;
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) {
                            f(ni, x.data.ptr, incx, y.data.ptr, incy, &out);
                            sink += out.re + out.im;
                        }
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .dzasum => {
                    if (dyn.lookup(CblasDzasumFn, "cblas_dzasum")) |f| {
                        actual_symbol = "cblas_dzasum";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx);
                    } else if (dyn.lookup(DzasumFn, "dzasum_")) |f| {
                        actual_symbol = "dzasum_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .dznrm2 => {
                    if (dyn.lookup(CblasDznrm2Fn, "cblas_dznrm2")) |f| {
                        actual_symbol = "cblas_dznrm2";
                        abi_surface = "cblas";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(ni, x.data.ptr, incx);
                    } else if (dyn.lookup(Dznrm2Fn, "dznrm2_")) |f| {
                        actual_symbol = "dznrm2_";
                        while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += f(&ni, x.data.ptr, &incx);
                    } else {
                        return error.MissingSymbol;
                    }
                },
                .izamax => {
                    actual_symbol = "izamax_";
                    const f = dyn.lookup(IzamaxFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) sink += @floatFromInt(f(&ni, x.data.ptr, &incx));
                },
                .zdrot => {
                    actual_symbol = "zdrot_";
                    const f = dyn.lookup(ZdrotFn, actual_symbol) orelse return error.MissingSymbol;
                    while (std.Io.Clock.awake.now(init.io).nanoseconds < deadline) : (iters += 1) f(&ni, x.data.ptr, &incx, y.data.ptr, &incy, &c, &s);
                },
                else => unreachable,
            }
            elapsed_ns = std.Io.Clock.awake.now(init.io).nanoseconds - start;
            try x.verify(x_layout, mutatesX(selected));
            try y.verify(y_layout, mutatesY(selected));
            checksum = sink + checksumC64(x.data) + checksumC64(y.data);
        },
    }

    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);
    const gops = @as(f64, @floatFromInt(iters)) * workPerIter(selected, n, variant) / elapsed_s / 1.0e9;
    const gbps = @as(f64, @floatFromInt(iters)) * bytesPerIter(selected, n) / elapsed_s / 1.0e9;
    try stdout_writer.interface.print("iters={d} elapsed_ns={d} rate_Gops={d:.3} bandwidth_GBps={d:.3} checksum={d:.6} symbol={s} abi_surface={s}\n", .{ iters, elapsed_ns, gops, gbps, checksum, actual_symbol, abi_surface });
    try stdout_writer.flush();
}
