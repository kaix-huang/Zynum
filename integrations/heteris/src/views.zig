// Copyright (C) 2026 Scientific Platform contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
pub const c = @import("zynum_c");
pub const Status = c.heteris_status;
pub const View = c.heteris_view;
pub const Resources = c.zynum_resources;
extern "c" fn zynum_validate_view(*const View, [*c]u64, [*c]u64) Status;
pub fn resourcesValid(r: *const Resources) bool {
    return r.abi_version == c.ZYNUM_ABI_VERSION and r.struct_size >= @sizeOf(Resources) and
        (r.mode == c.ZYNUM_MODE_NAIVE or r.mode == c.ZYNUM_MODE_INTEGRATED) and r.reserved == 0 and
        r.allocate != null and r.release != null and r.map != null;
}

pub const Mapping = struct { base: usize, begin: usize, end: usize };
pub fn validate(v: *const View, rank: u32, flags: u32) Status {
    return validateType(v, rank, c.HETERIS_F64, flags);
}
pub fn validateType(v: *const View, rank: u32, dtype: u32, flags: u32) Status {
    const status = zynum_validate_view(v, null, null);
    if (status != c.HETERIS_OK) return status;
    if (v.dtype != dtype or v.rank != rank) return c.HETERIS_UNSUPPORTED;
    if ((v.flags & flags) != flags) return c.HETERIS_INVALID_ARGUMENT;
    return c.HETERIS_OK;
}
pub fn mapped(r: *const Resources, v: *const View, out: *Mapping) Status {
    var raw: ?*anyopaque = null;
    var bytes: u64 = 0;
    const status = r.map.?(r.user_data, v.storage, &raw, &bytes);
    if (status != c.HETERIS_OK) return status;
    if (bytes < v.storage_bytes) return c.HETERIS_OUT_OF_BOUNDS;
    const base = @intFromPtr(raw);
    if ((bytes != 0 and raw == null) or base % @alignOf(f64) != 0) return c.HETERIS_INVALID_ARGUMENT;
    if (bytes > std.math.maxInt(usize) - base) return c.HETERIS_OVERFLOW;
    var begin: u64 = 0;
    var end: u64 = 0;
    const metadata_status = zynum_validate_view(v, &begin, &end);
    if (metadata_status != c.HETERIS_OK) return metadata_status;
    out.* = .{ .base = base, .begin = base + @as(usize, @intCast(begin)), .end = base + @as(usize, @intCast(end)) };
    return c.HETERIS_OK;
}
pub fn intersects(x: Mapping, y: Mapping) bool {
    return x.begin != x.end and y.begin != y.end and x.begin < y.end and y.begin < x.end;
}
pub fn sameLayout(x: *const View, xm: Mapping, y: *const View, ym: Mapping) bool {
    if (xm.base + x.byte_offset != ym.base + y.byte_offset or x.rank != y.rank) return false;
    for (0..x.rank) |d| if (x.shape[d] != y.shape[d] or x.strides[d] != y.strides[d]) return false;
    return true;
}
pub fn at(v: *const View, m: Mapping, indices: [3]u64) *f64 {
    var offset: i128 = v.byte_offset;
    for (0..v.rank) |d| offset += @as(i128, indices[d]) * v.strides[d];
    return @ptrFromInt(m.base + @as(usize, @intCast(offset)));
}
pub fn pair(r: *const Resources, x: *const View, y: *const View, rank: u32, yflags: u32, xm: *Mapping, ym: *Mapping) Status {
    if (!resourcesValid(r)) return c.HETERIS_INVALID_ARGUMENT;
    var status = validate(x, rank, c.HETERIS_VIEW_READ);
    if (status != c.HETERIS_OK) return status;
    status = validate(y, rank, yflags);
    if (status != c.HETERIS_OK) return status;
    for (0..rank) |d| if (x.shape[d] != y.shape[d]) return c.HETERIS_INVALID_ARGUMENT;
    status = mapped(r, x, xm);
    if (status != c.HETERIS_OK) return status;
    return mapped(r, y, ym);
}
