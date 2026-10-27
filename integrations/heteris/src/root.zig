// Copyright (C) 2026 Scientific Platform contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const scalar = @import("scientific_sources").scalar;
const c = @import("views.zig").c;
const Status = c.heteris_status;
const View = c.heteris_view;
const Resources = c.zynum_resources;

const views = @import("views.zig");
const Mapping = views.Mapping;
const pair = views.pair;
const intersects = views.intersects;
const sameLayout = views.sameLayout;
const at = views.at;
export fn zynum_axpy_f64(r_arg: ?*const Resources, x_arg: ?*const View, y_arg: ?*const View, alpha: f64) Status {
    const resource_copy = (r_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const x_copy = (x_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const y_copy = (y_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const r = &resource_copy;
    const x = &x_copy;
    const y = &y_copy;
    if (!std.math.isFinite(alpha)) return c.HETERIS_INVALID_ARGUMENT;
    var xm: Mapping = undefined;
    var ym: Mapping = undefined;
    const status = pair(r, x, y, 1, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE, &xm, &ym);
    if (status != c.HETERIS_OK) return status;
    if (intersects(xm, ym) and !sameLayout(x, xm, y, ym)) return c.HETERIS_OVERLAPPING_WRITE;
    for (0..@intCast(x.shape[0])) |i| {
        const index: [3]u64 = .{ i, 0, 0 };
        const dest = at(y, ym, index);
        dest.* = scalar.add(f64, scalar.mul(f64, alpha, at(x, xm, index).*), dest.*);
    }
    return c.HETERIS_OK;
}

export fn zynum_dot_f64(r_arg: ?*const Resources, x_arg: ?*const View, y_arg: ?*const View, result_arg: ?*f64) Status {
    const resource_copy = (r_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const x_copy = (x_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const y_copy = (y_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const r = &resource_copy;
    const x = &x_copy;
    const y = &y_copy;
    const result = result_arg orelse return c.HETERIS_INVALID_ARGUMENT;
    var xm: Mapping = undefined;
    var ym: Mapping = undefined;
    const status = pair(r, x, y, 1, c.HETERIS_VIEW_READ, &xm, &ym);
    if (status != c.HETERIS_OK) return status;
    const address = @intFromPtr(result);
    if (address > std.math.maxInt(usize) - @sizeOf(f64)) return c.HETERIS_OVERFLOW;
    const output: Mapping = .{ .base = address, .begin = address, .end = address + @sizeOf(f64) };
    if (intersects(output, xm) or intersects(output, ym)) return c.HETERIS_OVERLAPPING_WRITE;
    var sum = scalar.zero(f64);
    for (0..@intCast(x.shape[0])) |i| {
        const index: [3]u64 = .{ i, 0, 0 };
        sum = scalar.add(f64, sum, scalar.mul(f64, at(x, xm, index).*, at(y, ym, index).*));
    }
    result.* = sum;
    return c.HETERIS_OK;
}

export fn zynum_laplacian_periodic_3d_f64(r_arg: ?*const Resources, x_arg: ?*const View, y_arg: ?*const View, h_arg: [*c]const f64) Status {
    const resource_copy = (r_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const x_copy = (x_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const y_copy = (y_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const r = &resource_copy;
    const x = &x_copy;
    const y = &y_copy;
    if (h_arg == null) return c.HETERIS_INVALID_ARGUMENT;
    // Copy scalar arguments before any output writes, including when they alias y.
    const inv_h2 = [3]f64{ h_arg[0], h_arg[1], h_arg[2] };
    for (inv_h2) |h| if (!std.math.isFinite(h) or h <= 0) return c.HETERIS_INVALID_ARGUMENT;
    var xm: Mapping = undefined;
    var ym: Mapping = undefined;
    const status = pair(r, x, y, 3, c.HETERIS_VIEW_WRITE, &xm, &ym);
    if (status != c.HETERIS_OK) return status;
    if (intersects(xm, ym)) return c.HETERIS_OVERLAPPING_WRITE;
    for (x.shape[0..3]) |length| if (length == 0) return c.HETERIS_OK;
    var index: [3]u64 = .{ 0, 0, 0 };
    while (index[0] < x.shape[0]) : (index[0] += 1) {
        index[1] = 0;
        while (index[1] < x.shape[1]) : (index[1] += 1) {
            index[2] = 0;
            while (index[2] < x.shape[2]) : (index[2] += 1) {
                const shape: [3]usize = .{ @intCast(x.shape[0]), @intCast(x.shape[1]), @intCast(x.shape[2]) };
                const point: [3]usize = .{ @intCast(index[0]), @intCast(index[1]), @intCast(index[2]) };
                const Reader = struct {
                    view: *const View,
                    mapping: Mapping,
                    pub fn read(self: @This(), position: [3]usize) f64 {
                        return at(self.view, self.mapping, .{ @intCast(position[0]), @intCast(position[1]), @intCast(position[2]) }).*;
                    }
                };
                at(y, ym, index).* = @import("scientific_sources").periodic_laplacian.valueAt(shape, point, inv_h2, Reader{ .view = x, .mapping = xm });
            }
        }
    }
    return c.HETERIS_OK;
}

comptime {
    _ = @import("dense.zig");
    _ = @import("spectral.zig");
}

test {
    _ = @import("dense_tests.zig");
    _ = @import("spectral_tests.zig");
    _ = @import("tests.zig");
}
pub const testing = struct {
    pub const gemm = @import("dense.zig").zynum_gemm_f64;
    pub const gemm_complex = @import("dense.zig").zynum_gemm_c128;
    pub const abi = c;
    pub const axpy = zynum_axpy_f64;
    pub const dot = zynum_dot_f64;
    pub const laplacian = zynum_laplacian_periodic_3d_f64;
};
