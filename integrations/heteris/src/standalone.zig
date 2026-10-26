// Copyright (C) 2026 Scientific Platform contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Explicit standalone provider; never imported by the production math module.
const std = @import("std");
const views = @import("views.zig");
const c = views.c;
const Status = c.heteris_status;
const Resources = c.zynum_resources;
const Storage = c.heteris_storage;
const resourcesValid = views.resourcesValid;
extern "c" fn malloc(usize) ?*anyopaque;
extern "c" fn free(?*anyopaque) void;

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
pub export fn zynum_resources_create_naive(out_arg: ?*Resources) Status {
    const out = out_arg orelse return c.HETERIS_INVALID_ARGUMENT;
    const provider_id = nextProviderId() orelse return c.HETERIS_OVERFLOW;
    const memory = malloc(@sizeOf(Provider)) orelse return c.HETERIS_OUT_OF_MEMORY;
    const p: *Provider = @ptrCast(@alignCast(memory));
    p.* = .{ .first = null, .provider_id = provider_id, .next_id = 1 };
    out.* = .{ .abi_version = c.ZYNUM_ABI_VERSION, .struct_size = @sizeOf(Resources), .mode = c.ZYNUM_MODE_NAIVE, .reserved = 0, .user_data = p, .allocate = naiveAllocate, .release = naiveRelease, .map = naiveMap };
    return c.HETERIS_OK;
}
pub export fn zynum_resources_destroy_naive(r_arg: ?*Resources) Status {
    const r = r_arg orelse return c.HETERIS_INVALID_ARGUMENT;
    if (!resourcesValid(r) or r.mode != c.ZYNUM_MODE_NAIVE or r.map != naiveMap or r.allocate != naiveAllocate or r.release != naiveRelease) return c.HETERIS_INVALID_ARGUMENT;
    const p = provider(r.user_data) orelse return c.HETERIS_INVALID_ARGUMENT;
    if (p.first != null) return c.HETERIS_BUSY;
    free(p);
    r.* = std.mem.zeroes(Resources);
    return c.HETERIS_OK;
}
