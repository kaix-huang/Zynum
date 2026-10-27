// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const api = @import("test_api.zig");
const c = api.abi;
const expectEqual = std.testing.expectEqual;
const Buffer = struct {
    storage: c.heteris_storage,
    data: []f64,
    fn init(r: *c.zynum_resources, count: usize) !Buffer {
        var storage: c.heteris_storage = undefined;
        try expectEqual(@as(i32, c.HETERIS_OK), r.allocate.?(r.user_data, count * 8, 64, &storage));
        var ptr: ?*anyopaque = null;
        var bytes: u64 = 0;
        try expectEqual(@as(i32, c.HETERIS_OK), r.map.?(r.user_data, storage, &ptr, &bytes));
        const data: [*]f64 = @ptrCast(@alignCast(ptr.?));
        return .{ .storage = storage, .data = data[0..count] };
    }
    fn release(self: Buffer, r: *c.zynum_resources) void {
        std.debug.assert(r.release.?(r.user_data, self.storage) == c.HETERIS_OK);
    }
    fn view(self: Buffer, n: u64, offset: u64, stride: i64, flags: u32) c.heteris_view {
        var v = std.mem.zeroes(c.heteris_view);
        v.abi_version = c.HETERIS_ABI_VERSION;
        v.struct_size = @sizeOf(c.heteris_view);
        v.dtype = c.HETERIS_F64;
        v.rank = 1;
        v.flags = flags;
        v.storage = self.storage;
        v.storage_bytes = self.data.len * 8;
        v.byte_offset = offset;
        v.shape[0] = n;
        v.strides[0] = stride;
        return v;
    }
};

test "naive provider: strided axpy and signed-stride dot match scalar reference" {
    var r: c.zynum_resources = undefined;
    try expectEqual(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const x = try Buffer.init(&r, 9);
    defer x.release(&r);
    const y = try Buffer.init(&r, 9);
    defer y.release(&r);
    for (x.data, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    @memset(y.data, 10);
    var xv = x.view(5, 64, -16, c.HETERIS_VIEW_READ);
    var yv = y.view(5, 0, 16, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    try expectEqual(@as(i32, 0), api.axpy(&r, &xv, &yv, 2));
    var expected: f64 = 0;
    for (0..5) |i| {
        const ref = 10 + 2 * x.data[8 - 2 * i];
        try expectEqual(ref, y.data[2 * i]);
        expected += x.data[8 - 2 * i] * ref;
    }
    for (0..4) |i| try expectEqual(@as(f64, 10), y.data[2 * i + 1]);
    var dot: f64 = -1;
    try expectEqual(@as(i32, 0), api.dot(&r, &xv, &yv, &dot));
    try expectEqual(expected, dot);
    xv.shape[0] = 0;
    yv.shape[0] = 0;
    try expectEqual(@as(i32, 0), api.dot(&r, &xv, &yv, &dot));
    try expectEqual(@as(f64, 0), dot);
}

test "prevalidation, nonfinite scalars, aliases, stale handles and provider lifetime" {
    var r: c.zynum_resources = undefined;
    try expectEqual(@as(i32, 0), api.create(&r));
    const x = try Buffer.init(&r, 8);
    @memset(x.data, 3);
    var v = x.view(8, 0, 8, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    try expectEqual(@as(i32, c.HETERIS_BUSY), api.destroy(&r));
    try expectEqual(@as(i32, c.HETERIS_INVALID_ARGUMENT), api.axpy(&r, &v, &v, std.math.nan(f64)));
    try expectEqual(@as(f64, 3), x.data[0]);
    try expectEqual(@as(i32, 0), api.axpy(&r, &v, &v, 2));
    for (x.data) |value| try expectEqual(@as(f64, 9), value);
    var bad = v;
    bad.shape[0] = 7;
    bad.byte_offset = 8;
    var short = v;
    short.shape[0] = 7;
    try expectEqual(@as(i32, c.HETERIS_OVERLAPPING_WRITE), api.axpy(&r, &short, &bad, 2));
    try expectEqual(@as(f64, 9), x.data[1]);
    bad = v;
    bad.storage_bytes = 4;
    try expectEqual(@as(i32, c.HETERIS_OUT_OF_BOUNDS), api.axpy(&r, &v, &bad, 2));
    bad = v;
    bad.strides[0] = 0;
    try expectEqual(@as(i32, c.HETERIS_OVERLAPPING_WRITE), api.axpy(&r, &v, &bad, 2));
    try expectEqual(@as(i32, c.HETERIS_OVERLAPPING_WRITE), api.dot(&r, &v, &v, &x.data[0]));
    x.release(&r);
    var result: f64 = 42;
    try expectEqual(@as(i32, c.HETERIS_INVALID_HANDLE), api.dot(&r, &v, &v, &result));
    try expectEqual(@as(f64, 42), result);
    try expectEqual(@as(i32, 0), api.destroy(&r));
}

test "periodic 3D Laplacian: padded signed strides, Fourier eigenvalue, no partial writes" {
    var r: c.zynum_resources = undefined;
    try expectEqual(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const x = try Buffer.init(&r, 4 * 3 * 5 * 2);
    defer x.release(&r);
    const y = try Buffer.init(&r, x.data.len);
    defer y.release(&r);
    @memset(x.data, -99);
    @memset(y.data, 123);
    var xv = x.view(4, 3 * 3 * 5 * 16, -3 * 5 * 16, c.HETERIS_VIEW_READ);
    xv.rank = 3;
    xv.shape[1] = 3;
    xv.shape[2] = 5;
    xv.strides[1] = 5 * 16;
    xv.strides[2] = 16;
    var yv = xv;
    yv.storage = y.storage;
    yv.flags = c.HETERIS_VIEW_WRITE;
    const h = [3]f64{ 2, 3, 4 };
    for (0..4) |i| for (0..3) |j| for (0..5) |k| {
        x.data[((3 - i) * 3 * 5 + j * 5 + k) * 2] = @import("periodic_reference.zig").fourierInput(i, k, 4, 5);
    };
    const bad_h = [3]f64{ 1, std.math.inf(f64), 1 };
    try expectEqual(@as(i32, c.HETERIS_INVALID_ARGUMENT), api.laplacian(&r, &xv, &yv, &bad_h));
    for (y.data) |value| try expectEqual(@as(f64, 123), value);
    try expectEqual(@as(i32, 0), api.laplacian(&r, &xv, &yv, &h));
    for (0..4) |i| for (0..3) |j| for (0..5) |k| {
        const index = ((3 - i) * 3 * 5 + j * 5 + k) * 2;
        const expected = @import("periodic_reference.zig").fourierExpected(i, k, 4, 5, h);
        try std.testing.expectApproxEqAbs(expected, y.data[index], 2e-14);
        try expectEqual(@as(f64, 123), y.data[index + 1]);
    };
    yv.storage = x.storage;
    try expectEqual(@as(i32, c.HETERIS_OVERLAPPING_WRITE), api.laplacian(&r, &xv, &yv, &h));
    // Any empty dimension must be a quick no-op, even when another is huge.
    xv.shape[0] = std.math.maxInt(i64);
    xv.shape[1] = 0;
    yv.shape = xv.shape;
    try expectEqual(@as(i32, 0), api.laplacian(&r, &xv, &yv, &h));
}

fn unavailableMap(_: ?*anyopaque, _: c.heteris_storage, _: [*c]?*anyopaque, _: [*c]u64) callconv(.c) i32 {
    return c.HETERIS_UNAVAILABLE;
}
test "integrated provider error propagates without fallback" {
    var r: c.zynum_resources = undefined;
    try expectEqual(@as(i32, 0), api.create(&r));
    const x = try Buffer.init(&r, 1);
    @memset(x.data, 7);
    const v = x.view(1, 0, 8, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    var integrated = r;
    integrated.mode = c.ZYNUM_MODE_INTEGRATED;
    integrated.map = unavailableMap;
    try expectEqual(@as(i32, c.HETERIS_UNAVAILABLE), api.axpy(&integrated, &v, &v, 2));
    try expectEqual(@as(f64, 7), x.data[0]);
    x.release(&r);
    try expectEqual(@as(i32, 0), api.destroy(&r));
}

test "output may contain its descriptor: kernel snapshots metadata before writes" {
    var r: c.zynum_resources = undefined;
    try expectEqual(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const count = @sizeOf(c.heteris_view) / @sizeOf(f64);
    const x = try Buffer.init(&r, count);
    defer x.release(&r);
    const y = try Buffer.init(&r, count);
    defer y.release(&r);
    @memset(x.data, 1);
    @memset(y.data, 0);
    const xv = x.view(count, 0, 8, c.HETERIS_VIEW_READ);
    const yv: *c.heteris_view = @ptrCast(y.data.ptr);
    yv.* = y.view(count, 0, 8, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    var expected: [count]f64 = undefined;
    for (y.data, &expected) |value, *ref| ref.* = value + 1;
    try expectEqual(@as(i32, 0), api.axpy(&r, &xv, yv, 1));
    for (y.data, expected) |actual, ref| try expectEqual(ref, actual);
}
