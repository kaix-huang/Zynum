// Copyright (C) 2026 Scientific Platform contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Adapted checked panel recurrence from Zynum c5cb123e55eaae683de09b0c8030837f5600f140,
// src/numerics/chebyshev.zig: panelAdmitted/seriesArithmetic. Execution admission
// is replaced by the explicit synchronous Heteris resources/callback ABI. Four
// vectors, recurrence order, no endpoint halving, and final commit are preserved.
// The checked loops now live in the ownerless checked_spectral Core engine,
// using the exact upstream recurrence_arithmetic.zig unchanged.
// The moments entry point below is new bridge code using that same checked
// recurrence; the pinned upstream revision has no Chebyshev moments API.
const std = @import("std");
const V = @import("views.zig");
const Engine = @import("scientific_sources").checked_spectral;
const A = @import("scientific_sources").recurrence_arithmetic;
const c = V.c;
pub const Apply = *const fn (?*anyopaque, ?*const V.Resources, ?*const V.View, ?*const V.View) callconv(.c) V.Status;

fn cellAt(view: *const V.View, mapping: V.Mapping, i: u64, j: u64) *f64 {
    return V.at(view, mapping, .{ i, j, 0 });
}
fn panelView(work: V.View, start: u64, columns: u64, flags: u32) V.View {
    var out = work;
    out.byte_offset += start * @as(u64, @intCast(work.strides[1]));
    out.shape[1] = columns;
    out.flags = flags;
    return out;
}

const TrustedPanel = struct {
    view: V.View,
    mapping: V.Mapping,

    pub fn cell(self: TrustedPanel, i: u64, j: u64) *f64 {
        return cellAt(&self.view, self.mapping, i, j);
    }
    pub fn panel(self: TrustedPanel, start: u64, columns: u64) TrustedPanel {
        return .{ .view = panelView(self.view, start, columns, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE), .mapping = self.mapping };
    }
};
const Callback = struct {
    apply_fn: Apply,
    user: ?*anyopaque,
    resources: *const V.Resources,

    pub fn apply(self: Callback, source: TrustedPanel, destination: TrustedPanel) i32 {
        var source_view = source.view;
        source_view.flags = c.HETERIS_VIEW_READ;
        var destination_view = destination.view;
        destination_view.flags = c.HETERIS_VIEW_WRITE;
        return self.apply_fn(self.user, self.resources, &source_view, &destination_view);
    }
};
fn legacyStatus(result: Engine.Result) V.Status {
    return switch (result) {
        .ok => c.HETERIS_OK,
        .callback_failure => |status| status,
        .numerical_failure => c.HETERIS_NUMERICAL_FAILURE,
    };
}

pub export fn zynum_chebyshev_series_f64(r_arg: ?*const V.Resources, apply_arg: ?Apply, user: ?*anyopaque, x_arg: ?*const V.View, coeff_arg: ?*const V.View, lower: f64, upper: f64, y_arg: ?*const V.View, work_arg: ?*const V.View) V.Status {
    const r = (r_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const apply = apply_arg orelse return c.HETERIS_INVALID_ARGUMENT;
    const x = (x_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const coeff = (coeff_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const y = (y_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const work = (work_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    if (!V.resourcesValid(&r) or !std.math.isFinite(lower) or !std.math.isFinite(upper) or !(lower < upper)) return c.HETERIS_INVALID_ARGUMENT;
    inline for (.{ .{ &x, @as(u32, 2), @as(u32, c.HETERIS_VIEW_READ) }, .{ &y, @as(u32, 2), @as(u32, c.HETERIS_VIEW_WRITE) }, .{ &coeff, @as(u32, 1), @as(u32, c.HETERIS_VIEW_READ) }, .{ &work, @as(u32, 2), @as(u32, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE) } }) |item| {
        const status = V.validate(item[0], item[1], item[2]);
        if (status != c.HETERIS_OK) return status;
    }
    const rows = x.shape[0];
    const columns = x.shape[1];
    const four_columns = std.math.mul(u64, columns, 4) catch return c.HETERIS_OVERFLOW;
    const stride = std.math.mul(u64, rows, 8) catch return c.HETERIS_OVERFLOW;
    if (stride > std.math.maxInt(i64)) return c.HETERIS_OVERFLOW;
    if (y.shape[0] != rows or y.shape[1] != columns or coeff.shape[0] == 0 or
        work.shape[0] != rows or work.shape[1] != four_columns or
        work.strides[0] != 8 or work.strides[1] != @as(i64, @intCast(stride))) return c.HETERIS_INVALID_ARGUMENT;
    var xm: V.Mapping = undefined;
    var ym: V.Mapping = undefined;
    var cm: V.Mapping = undefined;
    var wm: V.Mapping = undefined;
    const operands = [_]*const V.View{ &x, &y, &coeff, &work };
    const mappings = [_]*V.Mapping{ &xm, &ym, &cm, &wm };
    for (operands, mappings) |operand, mapping| {
        const status = V.mapped(&r, operand, mapping);
        if (status != c.HETERIS_OK) return status;
    }
    if (V.intersects(wm, xm) or V.intersects(wm, ym) or V.intersects(wm, cm) or V.intersects(cm, ym)) return c.HETERIS_OVERLAPPING_WRITE;
    for (0..@intCast(coeff.shape[0])) |k| if (!std.math.isFinite(cellAt(&coeff, cm, k, 0).*)) return c.HETERIS_INVALID_ARGUMENT;
    for (0..@intCast(columns)) |j| for (0..@intCast(rows)) |i| {
        if (!std.math.isFinite(cellAt(&x, xm, i, j).*)) return c.HETERIS_INVALID_ARGUMENT;
    };
    var progress: Engine.Progress = .{};
    return legacyStatus(Engine.series(A, Callback{ .apply_fn = apply, .user = user, .resources = &r }, TrustedPanel{ .view = x, .mapping = xm }, TrustedPanel{ .view = coeff, .mapping = cm }, TrustedPanel{ .view = y, .mapping = ym }, TrustedPanel{ .view = work, .mapping = wm }, rows, columns, coeff.shape[0], lower, upper, &progress));
}

pub export fn zynum_chebyshev_moments_f64(r_arg: ?*const V.Resources, apply_arg: ?Apply, user: ?*anyopaque, x_arg: ?*const V.View, count: u64, lower: f64, upper: f64, moments_arg: ?*const V.View, work_arg: ?*const V.View, stage_arg: ?*const V.View) V.Status {
    const r = (r_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const apply = apply_arg orelse return c.HETERIS_INVALID_ARGUMENT;
    const x = (x_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const moments = (moments_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const work = (work_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const stage = (stage_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    if (!V.resourcesValid(&r) or count == 0 or !std.math.isFinite(lower) or
        !std.math.isFinite(upper) or !(lower < upper)) return c.HETERIS_INVALID_ARGUMENT;
    inline for (.{ .{ &x, @as(u32, c.HETERIS_VIEW_READ) }, .{ &moments, @as(u32, c.HETERIS_VIEW_WRITE) }, .{ &work, @as(u32, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE) }, .{ &stage, @as(u32, c.HETERIS_VIEW_READ | c.HETERIS_VIEW_WRITE) } }) |item| {
        const status = V.validate(item[0], 2, item[1]);
        if (status != c.HETERIS_OK) return status;
    }
    const rows = x.shape[0];
    const columns = x.shape[1];
    const three_columns = std.math.mul(u64, columns, 3) catch return c.HETERIS_OVERFLOW;
    const work_stride = std.math.mul(u64, rows, 8) catch return c.HETERIS_OVERFLOW;
    const stage_stride = std.math.mul(u64, count, 8) catch return c.HETERIS_OVERFLOW;
    if (work_stride > std.math.maxInt(i64) or stage_stride > std.math.maxInt(i64)) return c.HETERIS_OVERFLOW;
    if (moments.shape[0] != count or moments.shape[1] != columns or
        work.shape[0] != rows or work.shape[1] != three_columns or work.strides[0] != 8 or
        work.strides[1] != @as(i64, @intCast(work_stride)) or
        stage.shape[0] != count or stage.shape[1] != columns or stage.strides[0] != 8 or
        stage.strides[1] != @as(i64, @intCast(stage_stride))) return c.HETERIS_INVALID_ARGUMENT;
    var xm: V.Mapping = undefined;
    var mm: V.Mapping = undefined;
    var wm: V.Mapping = undefined;
    var sm: V.Mapping = undefined;
    const operands = [_]*const V.View{ &x, &moments, &work, &stage };
    const mappings = [_]*V.Mapping{ &xm, &mm, &wm, &sm };
    for (operands, mappings) |operand, mapping| {
        const status = V.mapped(&r, operand, mapping);
        if (status != c.HETERIS_OK) return status;
    }
    if (V.intersects(wm, xm) or V.intersects(wm, mm) or V.intersects(wm, sm) or
        V.intersects(sm, xm) or V.intersects(sm, mm)) return c.HETERIS_OVERLAPPING_WRITE;
    // Avoid a huge empty dimension causing an outer loop with no work.
    if (columns == 0) return c.HETERIS_OK;
    if (rows != 0) for (0..@intCast(columns)) |j| for (0..@intCast(rows)) |i| {
        if (!std.math.isFinite(cellAt(&x, xm, i, j).*)) return c.HETERIS_INVALID_ARGUMENT;
    };
    var progress: Engine.Progress = .{};
    return legacyStatus(Engine.rawMoments(A, Callback{ .apply_fn = apply, .user = user, .resources = &r }, TrustedPanel{ .view = x, .mapping = xm }, TrustedPanel{ .view = moments, .mapping = mm }, TrustedPanel{ .view = work, .mapping = wm }, TrustedPanel{ .view = stage, .mapping = sm }, rows, columns, count, lower, upper, &progress));
}
