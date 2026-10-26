// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const api = @import("test_api.zig");
const c = api.abi;
const eq = std.testing.expectEqual;
const R = c.HETERIS_VIEW_READ;
const W = c.HETERIS_VIEW_WRITE;

const Buffer = struct {
    storage: c.heteris_storage,
    data: []f64,
    fn init(r: *c.zynum_resources, count: usize) !Buffer {
        var storage: c.heteris_storage = undefined;
        try eq(@as(i32, 0), r.allocate.?(r.user_data, count * 8, 64, &storage));
        var raw: ?*anyopaque = null;
        var bytes: u64 = 0;
        try eq(@as(i32, 0), r.map.?(r.user_data, storage, &raw, &bytes));
        const p: [*]f64 = @ptrCast(@alignCast(raw.?));
        return .{ .storage = storage, .data = p[0..count] };
    }
    fn release(self: Buffer, r: *c.zynum_resources) void {
        std.debug.assert(r.release.?(r.user_data, self.storage) == 0);
    }
    fn view(self: Buffer, dtype: u32, rows: u64, cols: u64, offset: u64, rs: i64, cs: i64, flags: u32) c.heteris_view {
        var result = std.mem.zeroes(c.heteris_view);
        result.abi_version = c.HETERIS_ABI_VERSION;
        result.struct_size = @sizeOf(c.heteris_view);
        result.dtype = dtype;
        result.rank = 2;
        result.flags = flags;
        result.storage = self.storage;
        result.storage_bytes = self.data.len * 8;
        result.byte_offset = offset;
        result.shape[0] = rows;
        result.shape[1] = cols;
        result.strides[0] = rs;
        result.strides[1] = cs;
        return result;
    }
};

test "GEMM rectangular known answer, all transpose modes and mixed signed strides" {
    var r: c.zynum_resources = undefined;
    try eq(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const a = try Buffer.init(&r, 8);
    defer a.release(&r);
    const b = try Buffer.init(&r, 8);
    defer b.release(&r);
    const out = try Buffer.init(&r, 6);
    defer out.release(&r);
    // A=[1 2 3;4 5 6], B=[7 8;9 10;11 12]. Expected AB=[58 64;139 154].
    // Store A with negative column stride and B in column-major order.
    a.data[0..8].* = .{ 3, 2, 1, -99, 6, 5, 4, -99 };
    b.data[0..8].* = .{ 7, 9, 11, -99, 8, 10, 12, -99 };
    for (0..3) |oa| for (0..3) |ob| {
        var av = a.view(c.HETERIS_F64, 2, 3, 16, 32, -8, R);
        var bv = b.view(c.HETERIS_F64, 3, 2, 0, 8, 32, R);
        if (oa != 0) {
            std.mem.swap(u64, &av.shape[0], &av.shape[1]);
            std.mem.swap(i64, &av.strides[0], &av.strides[1]);
        }
        if (ob != 0) {
            std.mem.swap(u64, &bv.shape[0], &bv.shape[1]);
            std.mem.swap(i64, &bv.strides[0], &bv.strides[1]);
        }
        @memset(out.data, 2);
        const cv = out.view(c.HETERIS_F64, 2, 2, 24, -24, 8, R | W);
        try eq(@as(i32, 0), api.gemm(&r, @intCast(oa), @intCast(ob), &av, &bv, &cv, 2, -3));
        try eq(@as(f64, 110), out.data[3]);
        try eq(@as(f64, 122), out.data[4]);
        try eq(@as(f64, 272), out.data[0]);
        try eq(@as(f64, 302), out.data[1]);
        try eq(@as(f64, 2), out.data[2]);
        try eq(@as(f64, 2), out.data[5]);
    };
    // Read broadcasts and active beta-zero branch both avoid the old NaNs in C.
    const av = a.view(c.HETERIS_F64, 2, 3, 16, 0, -8, R);
    const bv = b.view(c.HETERIS_F64, 3, 2, 0, 8, 0, R);
    const cv = out.view(c.HETERIS_F64, 2, 2, 0, 16, 8, W);
    @memset(out.data, std.math.nan(f64));
    try eq(@as(i32, 0), api.gemm(&r, 0, 0, &av, &bv, &cv, 1, 0));
    for (out.data[0..4]) |value| try eq(@as(f64, 58), value);
}

test "GEMM zero branches, empty/invalid dimensions, alias errors before writes" {
    var r: c.zynum_resources = undefined;
    try eq(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const a = try Buffer.init(&r, 6);
    defer a.release(&r);
    const b = try Buffer.init(&r, 6);
    defer b.release(&r);
    const out = try Buffer.init(&r, 4);
    defer out.release(&r);
    @memset(a.data, std.math.nan(f64));
    @memset(b.data, std.math.inf(f64));
    @memset(out.data, std.math.nan(f64));
    var av = a.view(c.HETERIS_F64, 2, 3, 0, 24, 8, R);
    var bv = b.view(c.HETERIS_F64, 3, 2, 0, 16, 8, R);
    var cv = out.view(c.HETERIS_F64, 2, 2, 0, 16, 8, W);
    try eq(@as(i32, 0), api.gemm(&r, 0, 0, &av, &bv, &cv, 0, 0));
    for (out.data) |x| try eq(@as(f64, 0), x);
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), api.gemm(&r, 0, 0, &av, &bv, &cv, 1, 1));
    cv.flags = R | W;
    @memset(out.data, 7);
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), api.gemm(&r, 3, 0, &av, &bv, &cv, 1, 0));
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), api.gemm(&r, 0, 0, &av, &bv, &cv, std.math.inf(f64), 0));
    var alias = a.view(c.HETERIS_F64, 2, 2, 0, 24, 8, W);
    try eq(@as(i32, c.HETERIS_OVERLAPPING_WRITE), api.gemm(&r, 0, 0, &av, &bv, &alias, 0, 0));
    alias = b.view(c.HETERIS_F64, 2, 2, 0, 16, 8, W);
    try eq(@as(i32, c.HETERIS_OVERLAPPING_WRITE), api.gemm(&r, 0, 0, &av, &bv, &alias, 0, 0));
    cv.shape[0] = 1;
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), api.gemm(&r, 0, 0, &av, &bv, &cv, 0, 0));
    for (out.data) |x| try eq(@as(f64, 7), x);
    cv.shape[0] = 2;
    av.shape[1] = 0;
    bv.shape[0] = 0;
    try eq(@as(i32, 0), api.gemm(&r, 0, 0, &av, &bv, &cv, 1, -2));
    for (out.data) |x| try eq(@as(f64, -14), x);
    // Empty results still validate/map; stale storage is never ignored.
    av.shape[0] = 0;
    cv.shape[0] = 0;
    try eq(@as(i32, 0), api.gemm(&r, 0, 0, &av, &bv, &cv, 1, 0));
    av.storage.generation += 1;
    try eq(@as(i32, c.HETERIS_INVALID_HANDLE), api.gemm(&r, 0, 0, &av, &bv, &cv, 0, 0));
    alias = a.view(c.HETERIS_F64, 2, 2, 0, 0, 8, W);
    av = a.view(c.HETERIS_F64, 2, 3, 0, 24, 8, R);
    bv = b.view(c.HETERIS_F64, 3, 2, 0, 16, 8, R);
    try eq(@as(i32, c.HETERIS_OVERLAPPING_WRITE), api.gemm(&r, 0, 0, &av, &bv, &alias, 0, 0));
}

test "complex GEMM distinguishes transpose and conjugation with nonreal scalars" {
    var r: c.zynum_resources = undefined;
    try eq(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const a = try Buffer.init(&r, 12);
    defer a.release(&r);
    const b = try Buffer.init(&r, 12);
    defer b.release(&r);
    const out = try Buffer.init(&r, 8);
    defer out.release(&r);
    // Rectangular A=[1+i,2-i,0+2i;3, -1+i,2], B=[1,2i;2-i,1; i,3-i].
    a.data[0..12].* = .{ 1, 1, 2, -1, 0, 2, 3, 0, -1, 1, 2, 0 };
    b.data[0..12].* = .{ 1, 0, 0, 2, 2, -1, 1, 0, 0, 1, 3, -1 };
    const av = a.view(c.HETERIS_COMPLEX128, 2, 3, 0, 48, 16, R);
    const bv = b.view(c.HETERIS_COMPLEX128, 3, 2, 0, 32, 16, R);
    const cv = out.view(c.HETERIS_COMPLEX128, 2, 2, 0, 32, 16, W);
    try eq(@as(i32, 0), api.gemm_complex(&r, 0, 0, &av, &bv, &cv, .{ .real = 1, .imag = 0 }, .{ .real = 0, .imag = 0 }));
    const expected = [_]f64{ 2, -3, 2, 7, 2, 5, 5, 5 };
    for (out.data, expected) |x, y| try eq(y, x);
    // A physical transpose, interpreted as conjugate transpose => conj(original A).
    var at = av;
    std.mem.swap(u64, &at.shape[0], &at.shape[1]);
    std.mem.swap(i64, &at.strides[0], &at.strides[1]);
    try eq(@as(i32, 0), api.gemm_complex(&r, 2, 0, &at, &bv, &cv, .{ .real = 0, .imag = 1 }, .{ .real = 0, .imag = 0 }));
    const conjugated = [_]f64{ 1, 8, 3, 2, -1, 0, -3, 5 };
    for (out.data, conjugated) |x, y| try eq(y, x);
    var cv_read = cv;
    cv_read.flags = R | W;
    try eq(@as(i32, 0), api.gemm_complex(&r, 0, 0, &av, &bv, &cv_read, .{ .real = 0, .imag = 0 }, .{ .real = 1, .imag = 1 }));
    const scaled = [_]f64{ -7, 9, 1, 5, -1, -1, -8, 2 };
    for (out.data, scaled) |x, y| try eq(y, x);
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), api.gemm_complex(&r, 0, 0, &av, &bv, &cv, .{ .real = 1, .imag = std.math.nan(f64) }, .{ .real = 0, .imag = 0 }));
    // Reverse physical rows of A; store B column-major with negative columns.
    a.data[0..12].* = .{ 3, 0, -1, 1, 2, 0, 1, 1, 2, -1, 0, 2 };
    b.data[0..12].* = .{ 0, 2, 1, 0, 3, -1, 1, 0, 2, -1, 0, 1 };
    const products = [4][8]f64{
        .{ 2, -3, 2, 7, 2, 5, 5, 5 },
        .{ 8, -1, 2, -3, 0, 1, 5, 3 },
        .{ 8, 1, 2, 3, 0, -1, 5, -3 },
        .{ 2, 3, 2, -7, 2, -5, 5, -5 },
    };
    for (0..3) |oa| for (0..3) |ob| {
        var left = a.view(c.HETERIS_COMPLEX128, 2, 3, 48, -48, 16, R);
        var right = b.view(c.HETERIS_COMPLEX128, 3, 2, 48, 16, -48, R);
        if (oa != 0) {
            std.mem.swap(u64, &left.shape[0], &left.shape[1]);
            std.mem.swap(i64, &left.strides[0], &left.strides[1]);
        }
        if (ob != 0) {
            std.mem.swap(u64, &right.shape[0], &right.shape[1]);
            std.mem.swap(i64, &right.strides[0], &right.strides[1]);
        }
        @memset(out.data, std.math.nan(f64));
        try eq(@as(i32, 0), api.gemm_complex(&r, @intCast(oa), @intCast(ob), &left, &right, &cv, .{ .real = 1, .imag = 0 }, .{ .real = 0, .imag = 0 }));
        const which: usize = (if (oa == 2) @as(usize, 1) else 0) + (if (ob == 2) @as(usize, 2) else 0);
        for (out.data, products[which]) |x, y| try eq(y, x);
    };
}

test "GEMM maps validate address overflow and propagate provider failures" {
    const Provider = struct {
        status: i32 = 0,
        base: usize = 8,
        bytes: u64 = std.math.maxInt(u64),
        fn allocate(_: ?*anyopaque, _: u64, _: u64, _: [*c]c.heteris_storage) callconv(.c) i32 {
            return c.HETERIS_UNAVAILABLE;
        }
        fn release(_: ?*anyopaque, _: c.heteris_storage) callconv(.c) i32 {
            return c.HETERIS_UNAVAILABLE;
        }
        fn map(user: ?*anyopaque, _: c.heteris_storage, ptr: [*c]?*anyopaque, bytes: [*c]u64) callconv(.c) i32 {
            const self: *@This() = @ptrCast(@alignCast(user.?));
            ptr.* = @ptrFromInt(self.base);
            bytes.* = self.bytes;
            return self.status;
        }
    };
    var state = Provider{};
    var r = std.mem.zeroes(c.zynum_resources);
    r.abi_version = c.ZYNUM_ABI_VERSION;
    r.struct_size = @sizeOf(c.zynum_resources);
    r.mode = c.ZYNUM_MODE_INTEGRATED;
    r.user_data = &state;
    r.allocate = Provider.allocate;
    r.release = Provider.release;
    r.map = Provider.map;
    // No fake pointer can be dereferenced: all cases fail during admission.
    const fake = Buffer{ .storage = .{ .provider_id = 1, .id = 1, .generation = 1 }, .data = &.{} };
    var av = fake.view(c.HETERIS_F64, 1, 1, 0, 8, 8, R);
    av.storage_bytes = 8;
    var cv = av;
    cv.flags = W;
    try eq(@as(i32, c.HETERIS_OVERFLOW), api.gemm(&r, 0, 0, &av, &av, &cv, 0, 0));
    state.status = c.HETERIS_BUDGET_EXCEEDED;
    try eq(@as(i32, c.HETERIS_BUDGET_EXCEEDED), api.gemm(&r, 0, 0, &av, &av, &cv, 0, 0));
    state.status = 0;
    state.bytes = 8;
    state.base = 1;
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), api.gemm(&r, 0, 0, &av, &av, &cv, 0, 0));
    av.shape[0] = std.math.maxInt(u64);
    try eq(@as(i32, c.HETERIS_OVERFLOW), api.gemm(&r, 0, 0, &av, &av, &cv, 0, 0));
}

test "GEMM snapshots resource callbacks and all view metadata before mapping" {
    const Mutator = struct {
        source: c.zynum_resources,
        table: *c.zynum_resources,
        descriptors: [3]*c.heteris_view,
        count: usize = 0,
        fn map(user: ?*anyopaque, storage: c.heteris_storage, ptr: [*c]?*anyopaque, bytes: [*c]u64) callconv(.c) i32 {
            const self: *@This() = @ptrCast(@alignCast(user.?));
            self.count += 1;
            self.table.map = null;
            for (self.descriptors) |descriptor| {
                descriptor.shape[0] = 0;
                descriptor.strides[0] = 0;
                descriptor.storage.generation = 0;
            }
            return self.source.map.?(self.source.user_data, storage, ptr, bytes);
        }
    };
    var r: c.zynum_resources = undefined;
    try eq(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const a = try Buffer.init(&r, 1);
    defer a.release(&r);
    const b = try Buffer.init(&r, 1);
    defer b.release(&r);
    const out = try Buffer.init(&r, 1);
    defer out.release(&r);
    a.data[0] = 3;
    b.data[0] = 7;
    out.data[0] = -1;
    var av = a.view(c.HETERIS_F64, 1, 1, 0, 8, 8, R);
    var bv = b.view(c.HETERIS_F64, 1, 1, 0, 8, 8, R);
    var cv = out.view(c.HETERIS_F64, 1, 1, 0, 8, 8, W);
    var external = r;
    var state = Mutator{ .source = r, .table = &external, .descriptors = .{ &av, &bv, &cv } };
    external.user_data = &state;
    external.map = Mutator.map;
    try eq(@as(i32, 0), api.gemm(&external, 0, 0, &av, &bv, &cv, 1, 0));
    try eq(@as(usize, 3), state.count);
    try eq(@as(f64, 21), out.data[0]);
}
