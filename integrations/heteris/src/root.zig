// Copyright (C) 2026 Scientific Platform contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const scalar = @import("scientific_sources").scalar;
const c = @import("views.zig").c;
extern "c" fn malloc(usize) ?*anyopaque;
extern "c" fn free(?*anyopaque) void;
const Status = c.heteris_status;
const View = c.heteris_view;
const Resources = c.zynum_resources;
const Storage = c.heteris_storage;

const views = @import("views.zig");
const resourcesValid = views.resourcesValid;
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
                const center = at(x, xm, index).*;
                var value: f64 = 0;
                for (0..3) |d| {
                    var left = index;
                    var right = index;
                    left[d] = if (index[d] == 0) x.shape[d] - 1 else index[d] - 1;
                    right[d] = if (index[d] + 1 == x.shape[d]) 0 else index[d] + 1;
                    value += inv_h2[d] * (at(x, xm, left).* - 2 * center + at(x, xm, right).*);
                }
                at(y, ym, index).* = value;
            }
        }
    }
    return c.HETERIS_OK;
}

const Node = struct { next: ?*Node, raw: *anyopaque, address: usize, bytes: u64, id: u64 };
const Provider = struct { first: ?*Node, provider_id: u64, next_id: u64 };
var provider_sequence = std.atomic.Value(u64).init(1);
fn nextProviderId() ?u64 {
    const high_bit = @as(u64, 1) << 63;
    var sequence = provider_sequence.load(.monotonic);
    while (sequence < high_bit) {
        if (provider_sequence.cmpxchgWeak(sequence, sequence + 1, .monotonic, .monotonic)) |actual| {
            sequence = actual;
        } else return high_bit | sequence;
    }
    return null;
}
fn provider(user: ?*anyopaque) ?*Provider {
    return if (user) |p| @ptrCast(@alignCast(p)) else null;
}
fn naiveAllocate(user: ?*anyopaque, bytes: u64, alignment: u64, out: [*c]Storage) callconv(.c) Status {
    const p = provider(user) orelse return c.HETERIS_INVALID_ARGUMENT;
    if (out == null or alignment < 8 or !std.math.isPowerOfTwo(alignment)) return c.HETERIS_INVALID_ARGUMENT;
    if (bytes > std.math.maxInt(usize) or alignment > std.math.maxInt(usize) or bytes > std.math.maxInt(usize) - alignment) return c.HETERIS_OVERFLOW;
    if (p.next_id == std.math.maxInt(u64)) return c.HETERIS_OVERFLOW;
    const raw = malloc(@intCast(bytes + alignment)) orelse return c.HETERIS_OUT_OF_MEMORY;
    const node_memory = malloc(@sizeOf(Node)) orelse {
        free(raw);
        return c.HETERIS_OUT_OF_MEMORY;
    };
    const node: *Node = @ptrCast(@alignCast(node_memory));
    node.* = .{ .next = p.first, .raw = raw, .address = std.mem.alignForward(usize, @intFromPtr(raw), @intCast(alignment)), .bytes = bytes, .id = p.next_id };
    p.first = node;
    p.next_id += 1;
    out.* = .{ .provider_id = p.provider_id, .id = node.id, .generation = 1 };
    return c.HETERIS_OK;
}
fn naiveRelease(user: ?*anyopaque, storage: Storage) callconv(.c) Status {
    const p = provider(user) orelse return c.HETERIS_INVALID_ARGUMENT;
    if (storage.provider_id != p.provider_id or storage.generation != 1) return c.HETERIS_INVALID_HANDLE;
    var link = &p.first;
    while (link.*) |node| {
        if (node.id == storage.id) {
            link.* = node.next;
            free(node.raw);
            free(node);
            return c.HETERIS_OK;
        }
        link = &node.next;
    }
    return c.HETERIS_INVALID_HANDLE;
}
fn naiveMap(user: ?*anyopaque, storage: Storage, out: [*c]?*anyopaque, bytes: [*c]u64) callconv(.c) Status {
    const p = provider(user) orelse return c.HETERIS_INVALID_ARGUMENT;
    if (out == null or bytes == null) return c.HETERIS_INVALID_ARGUMENT;
    if (storage.provider_id != p.provider_id or storage.generation != 1) return c.HETERIS_INVALID_HANDLE;
    var link = p.first;
    while (link) |node| : (link = node.next) {
        if (node.id == storage.id) {
            out.* = @ptrFromInt(node.address);
            bytes.* = node.bytes;
            return c.HETERIS_OK;
        }
    }
    return c.HETERIS_INVALID_HANDLE;
}
export fn zynum_resources_create_naive(out_arg: ?*Resources) Status {
    const out = out_arg orelse return c.HETERIS_INVALID_ARGUMENT;
    const provider_id = nextProviderId() orelse return c.HETERIS_OVERFLOW;
    const memory = malloc(@sizeOf(Provider)) orelse return c.HETERIS_OUT_OF_MEMORY;
    const p: *Provider = @ptrCast(@alignCast(memory));
    p.* = .{ .first = null, .provider_id = provider_id, .next_id = 1 };
    out.* = .{ .abi_version = c.ZYNUM_ABI_VERSION, .struct_size = @sizeOf(Resources), .mode = c.ZYNUM_MODE_NAIVE, .reserved = 0, .user_data = p, .allocate = naiveAllocate, .release = naiveRelease, .map = naiveMap };
    return c.HETERIS_OK;
}
export fn zynum_resources_destroy_naive(r_arg: ?*Resources) Status {
    const r = r_arg orelse return c.HETERIS_INVALID_ARGUMENT;
    if (!resourcesValid(r) or r.mode != c.ZYNUM_MODE_NAIVE or r.map != naiveMap or r.allocate != naiveAllocate or r.release != naiveRelease) return c.HETERIS_INVALID_ARGUMENT;
    const p = provider(r.user_data) orelse return c.HETERIS_INVALID_ARGUMENT;
    if (p.first != null) return c.HETERIS_BUSY;
    free(p);
    r.* = std.mem.zeroes(Resources);
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
    pub const create = zynum_resources_create_naive;
    pub const destroy = zynum_resources_destroy_naive;
};
