// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! C controls, primitive marshalling and report projection only. No math loop.
const std = @import("std");
const execution = @import("zynum-execution");
const native = @import("zynum-native").spectral;
const c = @import("heteris-provider-c");
const Context = execution.Context;
const Report = c.zynum_native_spectral_report_v1;
const Range = struct {
    begin: usize,
    end: usize,
    fn overlaps(a: Range, b: Range) bool {
        return a.begin != a.end and b.begin != b.end and a.begin < b.end and b.begin < a.end;
    }
};
fn range(address: usize, bytes: u64) ?Range {
    const length = std.math.cast(usize, bytes) orelse return null;
    const end = std.math.add(usize, address, length) catch return null;
    return .{ .begin = address, .end = end };
}
fn control(address: usize, bytes: u64, alignment: usize) ?Range {
    if (address == 0 or address % alignment != 0) return null;
    return range(address, bytes);
}
fn payload(buffer: native.ConstBuffer) ?Range {
    if (buffer.address == 0 and buffer.backing != 0) return null;
    const count = std.math.cast(usize, buffer.backing) orelse return null;
    const length = std.math.mul(usize, count, @sizeOf(f64)) catch return null;
    return range(buffer.address, length);
}
const Controls = struct { context: *Context, guard: *native.Guard, report: *Report };
const RawControls = struct {
    context: ?*anyopaque,
    user: ?*anyopaque,
    user_bytes: u64,
    guard: ?*anyopaque,
    guard_bytes: u64,
    report: ?*align(1) Report,
};
fn preflight(raw: RawControls, args: native.Args) ?Controls {
    // Address arithmetic only: no dereference, aligned payload pointer, slice,
    // report initialization, CAS or admission before every full range passes.
    const ctx = @intFromPtr(raw.context);
    const rpt = @intFromPtr(raw.report);
    const grd = @intFromPtr(raw.guard);
    const usr = @intFromPtr(raw.user);
    const context_range = control(ctx, @sizeOf(Context), @alignOf(Context)) orelse return null;
    const report_range = control(rpt, @sizeOf(Report), @alignOf(Report)) orelse return null;
    const guard_range = control(grd, raw.guard_bytes, @alignOf(native.Guard)) orelse return null;
    if (raw.guard_bytes < @sizeOf(native.Guard)) return null;
    if ((usr == 0) != (raw.user_bytes == 0)) return null;
    const user_range = range(usr, raw.user_bytes) orelse return null;
    const controls = [_]Range{ context_range, report_range, guard_range, user_range };
    for (controls, 0..) |a, i| for (controls[i + 1 ..]) |b| {
        if (a.overlaps(b)) return null;
    };
    const buffers = [_]native.ConstBuffer{ args.input, args.output, args.coefficients, args.work };
    for (buffers, 0..) |buffer, i| {
        if (args.kind == .moments and i == 2) continue;
        const full = payload(buffer) orelse return null;
        for (controls) |item| if (item.overlaps(full)) return null;
    }
    return .{ .context = @ptrCast(@alignCast(raw.context.?)), .guard = @ptrCast(@alignCast(raw.guard.?)), .report = @ptrCast(@alignCast(raw.report.?)) };
}
fn initReport(report: *Report) void {
    report.* = std.mem.zeroes(Report);
    report.abi_version = c.ZYNUM_NATIVE_ABI_VERSION;
    report.struct_size = @sizeOf(Report);
}
fn project(report: *Report, result: native.Report) void {
    report.phase = @backingInt(result.phase);
    report.output_state = if (result.output_complete) c.ZYNUM_NATIVE_OUTPUT_COMPLETE else c.ZYNUM_NATIVE_OUTPUT_UNCHANGED;
    report.admission_attempted = @intFromBool(result.admission_attempted);
    report.admission_succeeded = @intFromBool(result.admission_succeeded);
    report.admission_status = switch (result.admission) {
        .ok => c.ZYNUM_NATIVE_OK,
        .closing => c.ZYNUM_NATIVE_CONTEXT_CLOSING,
        .counter_overflow => c.ZYNUM_NATIVE_COUNTER_OVERFLOW,
    };
    report.order_valid = @intFromBool(result.progress.order_valid);
    report.order = result.progress.order;
    report.callback_returned = @intFromBool(result.progress.callback_returned);
    report.callback_status = result.progress.callback_status;
    report.callback_order = result.progress.callback_order;
    report.callbacks_returned = result.progress.callbacks_returned;
}
const Callback = struct {
    apply: c.zynum_native_spectral_apply_f64_v1,
    user: ?*anyopaque,
    fn invoke(raw: ?*anyopaque, context: *Context, input: native.ConstPanel, output: native.Panel) i32 {
        const self: *const Callback = @ptrCast(@alignCast(raw.?));
        return self.apply.?(self.user, context, input.rows, input.columns, if (input.data) |p| p else null, input.rows * input.columns, input.ld, if (output.data) |p| p else null, output.rows * output.columns, output.ld);
    }
};
fn call(raw: RawControls, args: native.Args) i32 {
    const checked = preflight(raw, args) orelse return c.ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL;
    initReport(checked.report);
    if (!checked.guard.acquire()) return c.ZYNUM_NATIVE_SPECTRAL_BUSY;
    defer checked.guard.release();
    var result: native.Report = .{};
    const status = native.invoke(checked.context, args, &result);
    project(checked.report, result);
    return @backingInt(status);
}
export fn zynum_native_spectral_guard_query_v1(bytes_raw: ?*align(1) u64, alignment_raw: ?*align(1) u64) i32 {
    const a = control(@intFromPtr(bytes_raw), @sizeOf(u64), @alignOf(u64)) orelse return c.ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL;
    const b = control(@intFromPtr(alignment_raw), @sizeOf(u64), @alignOf(u64)) orelse return c.ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL;
    if (a.overlaps(b)) return c.ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL;
    const out_bytes: *u64 = @ptrCast(@alignCast(bytes_raw.?));
    const out_alignment: *u64 = @ptrCast(@alignCast(alignment_raw.?));
    out_bytes.* = @sizeOf(native.Guard);
    out_alignment.* = @alignOf(native.Guard);
    return c.ZYNUM_NATIVE_SPECTRAL_OK;
}
export fn zynum_native_spectral_guard_init_v1(storage: ?*anyopaque, bytes: u64) i32 {
    _ = control(@intFromPtr(storage), bytes, @alignOf(native.Guard)) orelse return c.ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL;
    if (bytes < @sizeOf(native.Guard)) return c.ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL;
    const guard: *native.Guard = @ptrCast(@alignCast(storage.?));
    guard.* = .{};
    return c.ZYNUM_NATIVE_SPECTRAL_OK;
}
export fn zynum_native_spectral_workspace_elements_v1(kind: u32, rows: u64, columns: u64, orders: u64, elements_raw: ?*align(1) u64) i32 {
    _ = control(@intFromPtr(elements_raw), @sizeOf(u64), @alignOf(u64)) orelse return c.ZYNUM_NATIVE_SPECTRAL_INVALID_CONTROL;
    const selected = std.enums.fromInt(native.Kind, kind) orelse return c.ZYNUM_NATIVE_SPECTRAL_INVALID_ARGUMENT;
    const elements = native.workspaceElements(selected, rows, columns, orders) catch |err| return switch (err) {
        error.InvalidArgument => c.ZYNUM_NATIVE_SPECTRAL_INVALID_ARGUMENT,
        else => c.ZYNUM_NATIVE_SPECTRAL_OVERFLOW,
    };
    const output: *u64 = @ptrCast(@alignCast(elements_raw.?));
    output.* = elements;
    return c.ZYNUM_NATIVE_SPECTRAL_OK;
}
export fn zynum_native_chebyshev_series_f64_v1(context: ?*anyopaque, apply: c.zynum_native_spectral_apply_f64_v1, user: ?*anyopaque, user_bytes: u64, rows: u64, columns: u64, x: ?*align(1) const f64, x_backing: u64, ldx: u64, coefficients: ?*align(1) const f64, coefficients_backing: u64, coefficient_count: u64, lower: f64, upper: f64, y: ?*align(1) f64, y_backing: u64, ldy: u64, work: ?*align(1) f64, work_backing: u64, guard: ?*anyopaque, guard_bytes: u64, report: ?*align(1) Report) i32 {
    var callback: Callback = .{ .apply = apply, .user = user };
    return call(.{ .context = context, .user = user, .user_bytes = user_bytes, .guard = guard, .guard_bytes = guard_bytes, .report = report }, .{
        .kind = .series,
        .rows = rows,
        .columns = columns,
        .orders = coefficient_count,
        .lower = lower,
        .upper = upper,
        .input = .{ .address = @intFromPtr(x), .backing = x_backing, .ld = ldx },
        .coefficients = .{ .address = @intFromPtr(coefficients), .backing = coefficients_backing },
        .output = .{ .address = @intFromPtr(y), .backing = y_backing, .ld = ldy },
        .work = .{ .address = @intFromPtr(work), .backing = work_backing },
        .operator = .{ .apply = if (apply == null) null else Callback.invoke, .data = &callback },
    });
}
export fn zynum_native_chebyshev_moments_f64_v1(context: ?*anyopaque, apply: c.zynum_native_spectral_apply_f64_v1, user: ?*anyopaque, user_bytes: u64, rows: u64, columns: u64, x: ?*align(1) const f64, x_backing: u64, ldx: u64, orders: u64, lower: f64, upper: f64, moments: ?*align(1) f64, moments_backing: u64, ldm: u64, work: ?*align(1) f64, work_backing: u64, guard: ?*anyopaque, guard_bytes: u64, report: ?*align(1) Report) i32 {
    var callback: Callback = .{ .apply = apply, .user = user };
    return call(.{ .context = context, .user = user, .user_bytes = user_bytes, .guard = guard, .guard_bytes = guard_bytes, .report = report }, .{
        .kind = .moments,
        .rows = rows,
        .columns = columns,
        .orders = orders,
        .lower = lower,
        .upper = upper,
        .input = .{ .address = @intFromPtr(x), .backing = x_backing, .ld = ldx },
        .output = .{ .address = @intFromPtr(moments), .backing = moments_backing, .ld = ldm },
        .work = .{ .address = @intFromPtr(work), .backing = work_backing },
        .operator = .{ .apply = if (apply == null) null else Callback.invoke, .data = &callback },
    });
}
comptime {
    // One authoritative translation supplies callback/report and status values.
    if (@sizeOf(Report) != 80 or @offsetOf(Report, "order") != 40) @compileError("spectral report target layout differs from supported x86-64 ABI");
    if (@backingInt(native.Status.numerical_failure) != c.ZYNUM_NATIVE_SPECTRAL_NUMERICAL_FAILURE or @backingInt(native.Phase.complete) != c.ZYNUM_NATIVE_SPECTRAL_COMPLETE) @compileError("spectral ABI projection mismatch");
}
