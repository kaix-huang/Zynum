// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const V = @import("views.zig");
const c = V.c;
const S = @import("spectral.zig");
const api = @import("root.zig").testing;
const eq = std.testing.expectEqual;
const Buffer = struct {
    storage: c.heteris_storage,
    data: []f64,
    fn init(r: *V.Resources, count: usize) !Buffer {
        var storage: c.heteris_storage = undefined;
        try eq(@as(i32, 0), r.allocate.?(r.user_data, count * 8, 64, &storage));
        var raw: ?*anyopaque = null;
        var bytes: u64 = 0;
        try eq(@as(i32, 0), r.map.?(r.user_data, storage, &raw, &bytes));
        const ptr: [*]f64 = @ptrCast(@alignCast(raw.?));
        return .{ .storage = storage, .data = ptr[0..count] };
    }
    fn close(self: Buffer, r: *V.Resources) void {
        std.debug.assert(r.release.?(r.user_data, self.storage) == 0);
    }
    fn view(self: Buffer, rows: u64, columns: u64, flags: u32) V.View {
        var out = std.mem.zeroes(V.View);
        out.abi_version = c.HETERIS_ABI_VERSION;
        out.struct_size = @sizeOf(V.View);
        out.dtype = c.HETERIS_F64;
        out.rank = 2;
        out.flags = flags;
        out.storage = self.storage;
        out.storage_bytes = self.data.len * 8;
        out.shape[0] = rows;
        out.shape[1] = columns;
        out.strides[0] = 8;
        out.strides[1] = @intCast(rows * 8);
        return out;
    }
};
const Backend = struct {
    calls: usize = 0,
    fail: usize = 0,
    nonfinite: bool = false,
    fn apply(raw: ?*anyopaque, resources: ?*const V.Resources, xv: ?*const V.View, yv: ?*const V.View) callconv(.c) V.Status {
        const self: *Backend = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        var xm: V.Mapping = undefined;
        var ym: V.Mapping = undefined;
        var status = V.mapped(resources.?, xv.?, &xm);
        if (status != 0) return status;
        status = V.mapped(resources.?, yv.?, &ym);
        if (status != 0) return status;
        const diagonal = [_]f64{ -2, 0, 6 };
        for (0..@intCast(xv.?.shape[1])) |j| for (diagonal, 0..) |lambda, i| {
            V.at(yv.?, ym, .{ i, j, 0 }).* = lambda * V.at(xv.?, xm, .{ i, j, 0 }).*;
        };
        if (self.nonfinite) V.at(yv.?, ym, .{ 0, 0, 0 }).* = std.math.inf(f64);
        if (self.calls == self.fail) return c.HETERIS_UNAVAILABLE;
        return 0;
    }
};

const MomentBackend = struct {
    calls: usize = 0,
    fail: usize = 0,
    nonfinite: bool = false,
    corrupt: ?*[4]V.View = null,
    resources_to_corrupt: ?*V.Resources = null,
    fn apply(raw: ?*anyopaque, resources: ?*const V.Resources, xv: ?*const V.View, yv: ?*const V.View) callconv(.c) V.Status {
        const self: *MomentBackend = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        var xm: V.Mapping = undefined;
        var ym: V.Mapping = undefined;
        var status = V.mapped(resources.?, xv.?, &xm);
        if (status != 0) return status;
        status = V.mapped(resources.?, yv.?, &ym);
        if (status != 0) return status;
        const diagonal = [_]f64{ -2, 0, 6 };
        if (xv.?.shape[0] != 0 and xv.?.shape[1] != 0) {
            for (0..@intCast(xv.?.shape[1])) |j| for (0..@intCast(xv.?.shape[0])) |i| {
                V.at(yv.?, ym, .{ i, j, 0 }).* = diagonal[i % 3] * V.at(xv.?, xm, .{ i, j, 0 }).*;
            };
            if (self.nonfinite) V.at(yv.?, ym, .{ 0, 0, 0 }).* = std.math.inf(f64);
        }
        if (self.corrupt) |views| for (views) |*view| {
            view.* = std.mem.zeroes(V.View);
        };
        if (self.resources_to_corrupt) |r| r.* = std.mem.zeroes(V.Resources);
        if (self.calls == self.fail) return c.HETERIS_UNAVAILABLE;
        return 0;
    }
};

const RemappingProvider = struct {
    original: V.Resources,
    source: c.heteris_storage,
    target: c.heteris_storage,
    fn map(raw: ?*anyopaque, storage: c.heteris_storage, out: [*c]?*anyopaque, bytes: [*c]u64) callconv(.c) V.Status {
        const self: *RemappingProvider = @ptrCast(@alignCast(raw.?));
        const actual = if (storage.id == self.source.id) self.target else storage;
        return self.original.map.?(self.original.user_data, actual, out, bytes);
    }
};

test "raw block moments: independent diagonal orders, negative strides, uneven block, series duality" {
    var r: V.Resources = undefined;
    try eq(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const input = try Buffer.init(&r, 9);
    defer input.close(&r);
    const output = try Buffer.init(&r, 57);
    defer output.close(&r);
    const scratch = try Buffer.init(&r, 29);
    defer scratch.close(&r);
    const staged = try Buffer.init(&r, 53);
    defer staged.close(&r);
    const coefficients = try Buffer.init(&r, 17);
    defer coefficients.close(&r);
    const series_output = try Buffer.init(&r, 9);
    defer series_output.close(&r);
    const series_scratch = try Buffer.init(&r, 36);
    defer series_scratch.close(&r);
    for (input.data, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i + 1)) / 3;
    @memset(output.data, 97);
    @memset(scratch.data, 81);
    @memset(staged.data, 82);
    var x = input.view(3, 3, c.HETERIS_VIEW_READ);
    x.byte_offset = 16;
    x.strides[0] = -8;
    var moments = output.view(17, 3, c.HETERIS_VIEW_WRITE);
    moments.byte_offset = 16 * 8;
    moments.strides[0] = -8;
    moments.strides[1] = 19 * 8;
    const work = scratch.view(3, 9, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    const stage = staged.view(17, 3, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    var backend = MomentBackend{};
    try eq(@as(i32, 0), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 17, -2, 6, &moments, &work, &stage));
    try eq(@as(usize, 16), backend.calls);
    const scaled = [_]f64{ -1, -0.5, 1 };
    for (0..3) |j| {
        for (0..17) |k| {
            var expected: f64 = 0;
            for (scaled, 0..) |a, i| {
                const initial = input.data[2 - i + 3 * j];
                // Closed diagonal eigensystem oracle, independent of recurrence.
                expected += initial * initial * @cos(@as(f64, @floatFromInt(k)) * std.math.acos(a));
            }
            try std.testing.expectApproxEqAbs(expected, output.data[16 - k + 19 * j], 3e-12);
        }
        try eq(@as(f64, 97), output.data[17 + 19 * j]);
        try eq(@as(f64, 97), output.data[18 + 19 * j]);
    }
    try eq(@as(f64, 81), scratch.data[27]);
    try eq(@as(f64, 82), staged.data[51]);
    for (coefficients.data, 0..) |*value, k| value.* = (if (k % 2 == 0) @as(f64, 1) else -1) / @as(f64, @floatFromInt(k + 1));
    var coeff = coefficients.view(17, 1, c.HETERIS_VIEW_READ);
    coeff.rank = 1;
    coeff.shape[1] = 0;
    coeff.strides[1] = 0;
    const y = series_output.view(3, 3, c.HETERIS_VIEW_WRITE);
    const series_work = series_scratch.view(3, 12, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    backend = .{};
    try eq(@as(i32, 0), S.zynum_chebyshev_series_f64(&r, MomentBackend.apply, &backend, &x, &coeff, -2, 6, &y, &series_work));
    for (0..3) |j| {
        var moment_series: f64 = 0;
        var inner_product: f64 = 0;
        for (0..17) |k| moment_series += coefficients.data[k] * output.data[16 - k + 19 * j];
        for (0..3) |i| inner_product += input.data[2 - i + 3 * j] * series_output.data[i + 3 * j];
        try std.testing.expectApproxEqAbs(moment_series, inner_product, 2e-12);
    }
}

test "moments validate, propagate failures atomically, allow X output alias and snapshot metadata" {
    var r: V.Resources = undefined;
    try eq(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const input = try Buffer.init(&r, 6);
    defer input.close(&r);
    const output = try Buffer.init(&r, 6);
    defer output.close(&r);
    const scratch = try Buffer.init(&r, 18);
    defer scratch.close(&r);
    const staged = try Buffer.init(&r, 6);
    defer staged.close(&r);
    const initial = [_]f64{ 1, 2, 3, 4, 5, 6 };
    @memcpy(input.data, &initial);
    @memset(output.data, 97);
    const x = input.view(3, 2, c.HETERIS_VIEW_READ);
    const moments = output.view(3, 2, c.HETERIS_VIEW_WRITE);
    const work = scratch.view(3, 6, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    const stage = staged.view(3, 2, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    var backend = MomentBackend{ .fail = 2 };
    try eq(@as(i32, c.HETERIS_UNAVAILABLE), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -2, 6, &moments, &work, &stage));
    for (output.data) |value| try eq(@as(f64, 97), value);
    backend = .{ .nonfinite = true };
    try eq(@as(i32, c.HETERIS_NUMERICAL_FAILURE), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -2, 6, &moments, &work, &stage));
    for (output.data) |value| try eq(@as(f64, 97), value);
    backend = .{};
    try eq(@as(i32, c.HETERIS_NUMERICAL_FAILURE), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -1e-310, 1e-310, &moments, &work, &stage));
    try eq(@as(usize, 1), backend.calls);
    for (output.data) |value| try eq(@as(f64, 97), value);
    backend = .{};
    input.data[0] = std.math.nan(f64);
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -2, 6, &moments, &work, &stage));
    input.data[0] = std.math.floatMax(f64);
    try eq(@as(i32, c.HETERIS_NUMERICAL_FAILURE), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -2, 6, &moments, &work, &stage));
    input.data[0] = 1;
    try eq(@as(usize, 0), backend.calls);
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 0, -2, 6, &moments, &work, &stage));
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, 6, -2, &moments, &work, &stage));
    var alias_stage = moments;
    alias_stage.flags = c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE;
    try eq(@as(i32, c.HETERIS_OVERLAPPING_WRITE), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -2, 6, &moments, &work, &alias_stage));
    var bad = stage;
    bad.strides[1] = 16;
    try std.testing.expect(S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -2, 6, &moments, &work, &bad) != 0);
    for (output.data) |value| try eq(@as(f64, 97), value);
    // Different handles whose providers expose the same bytes must still fail.
    var remap = RemappingProvider{ .original = r, .source = stage.storage, .target = moments.storage };
    var remapped_resources = r;
    remapped_resources.mode = c.ZYNUM_MODE_INTEGRATED;
    remapped_resources.user_data = &remap;
    remapped_resources.map = RemappingProvider.map;
    backend = .{};
    try eq(@as(i32, c.HETERIS_OVERLAPPING_WRITE), S.zynum_chebyshev_moments_f64(&remapped_resources, MomentBackend.apply, &backend, &x, 3, -2, 6, &moments, &work, &stage));
    try eq(@as(usize, 0), backend.calls);
    // Original descriptors and resources can reside in mutable caller memory.
    var views = [4]V.View{ x, moments, work, stage };
    var mutable_resources = r;
    backend = .{ .corrupt = &views, .resources_to_corrupt = &mutable_resources };
    try eq(@as(i32, 0), S.zynum_chebyshev_moments_f64(&mutable_resources, MomentBackend.apply, &backend, &views[0], 3, -2, 6, &views[1], &views[2], &views[3]));
    try eq(@as(usize, 2), backend.calls);
    try std.testing.expectEqualSlices(f64, &[_]f64{ 14, 6, 8, 77, 7.5, 39.5 }, output.data);
    backend = .{};
    var inplace = input.view(3, 2, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    try eq(@as(i32, 0), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &inplace, 3, -2, 6, &inplace, &work, &stage));
    try std.testing.expectEqualSlices(f64, output.data, input.data);
    // A singleton order computes the raw squared norm with no action.
    @memcpy(input.data, &initial);
    var singleton = moments;
    singleton.shape[0] = 1;
    var singleton_stage = stage;
    singleton_stage.shape[0] = 1;
    singleton_stage.strides[1] = 8;
    backend = .{};
    try eq(@as(i32, 0), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 1, -2, 6, &singleton, &work, &singleton_stage));
    try eq(@as(usize, 0), backend.calls);
    try eq(@as(f64, 14), output.data[0]);
    try eq(@as(f64, 77), output.data[3]);
}

test "moments empty panels, huge empty dimensions and finite extreme intervals" {
    var r: V.Resources = undefined;
    try eq(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const empty = try Buffer.init(&r, 0);
    defer empty.close(&r);
    const output = try Buffer.init(&r, 6);
    defer output.close(&r);
    const staged = try Buffer.init(&r, 6);
    defer staged.close(&r);
    var x = empty.view(0, 2, c.HETERIS_VIEW_READ);
    var moments = output.view(3, 2, c.HETERIS_VIEW_WRITE);
    var work = empty.view(0, 6, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    var stage = staged.view(3, 2, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    @memset(output.data, 97);
    var backend = MomentBackend{};
    try eq(@as(i32, 0), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -2, 6, &moments, &work, &stage));
    try eq(@as(usize, 2), backend.calls);
    for (output.data) |value| try eq(@as(f64, 0), value);
    const huge = @as(u64, std.math.maxInt(i64)) / 8;
    x = empty.view(huge, 0, c.HETERIS_VIEW_READ);
    moments = empty.view(3, 0, c.HETERIS_VIEW_WRITE);
    work = empty.view(huge, 0, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    stage = empty.view(3, 0, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    backend = .{};
    try eq(@as(i32, 0), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -2, 6, &moments, &work, &stage));
    try eq(@as(usize, 0), backend.calls);
    x = empty.view(0, 0, c.HETERIS_VIEW_READ);
    work = empty.view(0, 0, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    moments = empty.view(huge, 0, c.HETERIS_VIEW_WRITE);
    stage = empty.view(huge, 0, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    try eq(@as(i32, 0), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, huge, -2, 6, &moments, &work, &stage));
    moments.shape[0] = huge + 1;
    stage.shape[0] = huge + 1;
    try eq(@as(i32, c.HETERIS_OVERFLOW), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, huge + 1, -2, 6, &moments, &work, &stage));
    x.shape[1] = huge;
    work.shape[1] = 0; // admitted empty descriptor, incompatible workspace width
    moments.shape[0] = 3;
    moments.shape[1] = 0;
    stage = moments;
    stage.flags = c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE;
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -2, 6, &moments, &work, &stage));
    // Widening the interval subtraction prevents finite extreme bounds from
    // turning into an infinite half-width or losing a representable moment.
    const input = try Buffer.init(&r, 3);
    defer input.close(&r);
    const recurrence = try Buffer.init(&r, 9);
    defer recurrence.close(&r);
    @memcpy(input.data, &[_]f64{ 1, 2, 3 });
    x = input.view(3, 1, c.HETERIS_VIEW_READ);
    moments = output.view(3, 1, c.HETERIS_VIEW_WRITE);
    work = recurrence.view(3, 3, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    stage = staged.view(3, 1, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    try eq(@as(i32, 0), S.zynum_chebyshev_moments_f64(&r, MomentBackend.apply, &backend, &x, 3, -std.math.floatMax(f64), std.math.floatMax(f64), &moments, &work, &stage));
    try eq(@as(f64, 14), output.data[0]);
    try std.testing.expectApproxEqAbs(@as(f64, -14), output.data[2], 1e-14);
}

test "matrix-free block polynomial: independent quadratic oracle, in-place commit and failure atomicity" {
    var r: V.Resources = undefined;
    try eq(@as(i32, 0), api.create(&r));
    defer std.debug.assert(api.destroy(&r) == 0);
    const input = try Buffer.init(&r, 6);
    defer input.close(&r);
    const output = try Buffer.init(&r, 8);
    defer output.close(&r);
    const coefficients = try Buffer.init(&r, 3);
    defer coefficients.close(&r);
    const scratch = try Buffer.init(&r, 26);
    defer scratch.close(&r);
    for (input.data, 0..) |*value, i| value.* = @floatFromInt(i + 1);
    @memcpy(coefficients.data, &[_]f64{ 1, 2, 3 });
    @memset(output.data, 97);
    @memset(scratch.data, 81);
    const x = input.view(3, 2, c.HETERIS_VIEW_READ);
    var y = output.view(3, 2, c.HETERIS_VIEW_WRITE);
    y.strides[1] = 32; // padding must remain untouched
    var coeff = coefficients.view(3, 1, c.HETERIS_VIEW_READ);
    coeff.rank = 1;
    coeff.shape[1] = 0;
    coeff.strides[1] = 0;
    const work = scratch.view(3, 8, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    var backend = Backend{};
    try eq(@as(i32, 0), S.zynum_chebyshev_series_f64(&r, Backend.apply, &backend, &x, &coeff, -2, 6, &y, &work));
    try eq(@as(usize, 2), backend.calls);
    const scaled = [_]f64{ -1, -0.5, 1 };
    for (0..2) |j| for (scaled, 0..) |a, i| {
        const expected = (1 + 2 * a + 3 * (2 * a * a - 1)) * input.data[i + 3 * j];
        try std.testing.expectApproxEqAbs(expected, output.data[i + 4 * j], 2e-13);
    };
    try eq(@as(f64, 97), output.data[3]);
    try eq(@as(f64, 97), output.data[7]);
    try eq(@as(f64, 81), scratch.data[24]);
    try eq(@as(f64, 81), scratch.data[25]);
    const before = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var inplace = input.view(3, 2, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE);
    backend = .{ .fail = 2 };
    try eq(@as(i32, c.HETERIS_UNAVAILABLE), S.zynum_chebyshev_series_f64(&r, Backend.apply, &backend, &inplace, &coeff, -2, 6, &inplace, &work));
    try std.testing.expectEqualSlices(f64, &before, input.data);
    backend = .{ .nonfinite = true };
    try eq(@as(i32, c.HETERIS_NUMERICAL_FAILURE), S.zynum_chebyshev_series_f64(&r, Backend.apply, &backend, &inplace, &coeff, -2, 6, &inplace, &work));
    try std.testing.expectEqualSlices(f64, &before, input.data);
    backend = .{};
    try eq(@as(i32, 0), S.zynum_chebyshev_series_f64(&r, Backend.apply, &backend, &inplace, &coeff, -2, 6, &inplace, &work));
    for (0..2) |j| for (scaled, 0..) |a, i| {
        try std.testing.expectApproxEqAbs((1 + 2 * a + 3 * (2 * a * a - 1)) * before[i + 3 * j], input.data[i + 3 * j], 2e-13);
    };
    backend = .{};
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), S.zynum_chebyshev_series_f64(&r, Backend.apply, &backend, &x, &coeff, 6, -2, &y, &work));
    try eq(@as(usize, 0), backend.calls);
    // Constant polynomial performs no callback, including overlapping X/Y.
    coeff.shape[0] = 1;
    backend = .{};
    try eq(@as(i32, 0), S.zynum_chebyshev_series_f64(&r, Backend.apply, &backend, &inplace, &coeff, -2, 6, &inplace, &work));
    try eq(@as(usize, 0), backend.calls);
    coefficients.data[0] = std.math.nan(f64);
    try eq(@as(i32, c.HETERIS_INVALID_ARGUMENT), S.zynum_chebyshev_series_f64(&r, Backend.apply, &backend, &x, &coeff, -2, 6, &y, &work));
    coefficients.data[0] = 1;
    var bad_work = work;
    bad_work.strides[1] = 32;
    try std.testing.expect(S.zynum_chebyshev_series_f64(&r, Backend.apply, &backend, &x, &coeff, -2, 6, &y, &bad_work) != 0);
    // Alias detection resolves mapped addresses, including coefficient output.
    var aliased_coeff = y;
    aliased_coeff.rank = 1;
    aliased_coeff.shape[0] = 1;
    aliased_coeff.shape[1] = 0;
    aliased_coeff.strides[1] = 0;
    aliased_coeff.flags = c.HETERIS_VIEW_READ;
    try eq(@as(i32, c.HETERIS_OVERLAPPING_WRITE), S.zynum_chebyshev_series_f64(&r, Backend.apply, &backend, &x, &aliased_coeff, -2, 6, &y, &work));
    try eq(@as(usize, 0), backend.calls);
}
