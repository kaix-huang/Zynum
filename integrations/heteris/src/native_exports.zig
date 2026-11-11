// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Fixed-width validation and conversion only. Mathematical work is forwarded
//! into the one configured original Core; this file owns no numerical loops.
const std = @import("std");
const execution = @import("zynum-execution");
const native = @import("zynum-native");
const c = @import("heteris-provider-c");
const options = @import("zynum-build-options");
const Context = execution.Context;
const Report = c.zynum_native_report_v1;
const Policy = c.zynum_native_policy_v1;
const Requirements = c.zynum_native_context_requirements_v1;
comptime {
    _ = @import("adapter-root");
    _ = @import("native_spectral_exports.zig");
}
fn validAddress(address: usize, bytes: usize, alignment: usize) bool {
    if (address == 0 or address % alignment != 0) return false;
    _ = std.math.add(usize, address, bytes) catch return false;
    return true;
}
fn overlaps(a: usize, a_bytes: usize, b: usize, b_bytes: usize) bool {
    if (a_bytes == 0 or b_bytes == 0) return false;
    const a_end = std.math.add(usize, a, a_bytes) catch return true;
    const b_end = std.math.add(usize, b, b_bytes) catch return true;
    return a < b_end and b < a_end;
}
fn reportPointer(raw: ?*align(1) Report, storage: ?*anyopaque, storage_bytes: usize) ?*Report {
    const pointer = raw orelse return null;
    const address = @intFromPtr(pointer);
    if (!validAddress(address, @sizeOf(Report), @alignOf(Report))) return null;
    if (storage) |memory| if (overlaps(address, @sizeOf(Report), @intFromPtr(memory), storage_bytes)) return null;
    const report: *Report = @ptrCast(@alignCast(pointer));
    report.* = std.mem.zeroes(Report);
    report.abi_version = 1;
    report.struct_size = @sizeOf(Report);
    return report;
}
fn reportResult(report: *Report, result: execution.CallReport) void {
    report.resource_stage = @backingInt(result.provider.stage);
    report.submission_state = @backingInt(result.provider.state);
    report.output_state = @backingInt(result.output);
    report.provider_status = result.provider.returned_status;
    report.execution_status = result.provider.execution_status;
    report.callback_status = result.provider.callback_status;
    report.cleanup_status = result.provider.cleanup_status;
    report.callbacks_started = result.provider.callbacks_started;
    report.callbacks_completed = result.provider.callbacks_completed;
}
fn errorStatus(err: execution.ExecutionError) i32 {
    return switch (err) {
        error.InvalidProvider => c.ZYNUM_NATIVE_PROVIDER_CONTRACT_FAILURE,
        error.Unsupported => c.ZYNUM_NATIVE_UNSUPPORTED,
        error.ContextClosing => c.ZYNUM_NATIVE_CONTEXT_CLOSING,
        error.CounterOverflow => c.ZYNUM_NATIVE_COUNTER_OVERFLOW,
        error.ResourceBudgetExceeded => c.ZYNUM_NATIVE_BUDGET_EXCEEDED,
        error.OutOfMemory => c.ZYNUM_NATIVE_OUT_OF_MEMORY,
        error.ProviderBusy => c.ZYNUM_NATIVE_BUSY,
        error.ProviderClosed => c.ZYNUM_NATIVE_CLOSED,
        error.WouldDeadlock => c.ZYNUM_NATIVE_WOULD_DEADLOCK,
        error.ProviderRejected => c.ZYNUM_NATIVE_PROVIDER_REJECTED,
        error.ProviderExecutionFailed => c.ZYNUM_NATIVE_EXECUTION_FAILURE,
        error.ProviderCallbackFailed => c.ZYNUM_NATIVE_CALLBACK_FAILURE,
        error.ExtentOverflow => c.ZYNUM_NATIVE_OVERFLOW,
    };
}
fn contextPointer(raw: ?*anyopaque) ?*Context {
    const memory = raw orelse return null;
    if (!validAddress(@intFromPtr(memory), @sizeOf(Context), @alignOf(Context))) return null;
    return @ptrCast(@alignCast(memory));
}
export fn zynum_native_context_query_v1(raw: ?*align(1) Requirements) i32 {
    const memory = raw orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    if (!validAddress(@intFromPtr(memory), @sizeOf(Requirements), @alignOf(Requirements))) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const out: *Requirements = @ptrCast(@alignCast(memory));
    out.* = std.mem.zeroes(Requirements);
    out.abi_version = 1;
    out.struct_size = @sizeOf(Requirements);
    out.bytes = @sizeOf(Context);
    out.alignment = @alignOf(Context);
    return c.ZYNUM_NATIVE_OK;
}
export fn zynum_native_context_init_v1(storage: ?*anyopaque, bytes: u64, provider_arg: ?*align(1) const c.heteris_runtime_provider_v1, policy_arg: ?*align(1) const Policy, report_arg: ?*align(1) Report) i32 {
    const byte_count = std.math.cast(usize, bytes) orelse {
        _ = reportPointer(report_arg, storage, @sizeOf(Context)) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
        return c.ZYNUM_NATIVE_OVERFLOW;
    };
    // Check descriptor/report separation before initializing any output bytes.
    if (report_arg) |raw_report| {
        if (policy_arg) |raw_policy| if (overlaps(@intFromPtr(raw_report), @sizeOf(Report), @intFromPtr(raw_policy), @sizeOf(Policy))) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
        if (provider_arg) |raw_provider| if (overlaps(@intFromPtr(raw_report), @sizeOf(Report), @intFromPtr(raw_provider), @sizeOf(c.heteris_runtime_provider_v1))) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    }
    const report = reportPointer(report_arg, storage, byte_count) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const ctx = contextPointer(storage) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    if (byte_count < @sizeOf(Context)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    _ = std.math.add(usize, @intFromPtr(ctx), byte_count) catch return c.ZYNUM_NATIVE_OVERFLOW;
    const policy_raw = policy_arg orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const provider_raw = provider_arg orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    if (!validAddress(@intFromPtr(policy_raw), @sizeOf(Policy), @alignOf(Policy)) or
        !validAddress(@intFromPtr(provider_raw), @sizeOf(c.heteris_runtime_provider_v1), @alignOf(c.heteris_runtime_provider_v1))) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    if (overlaps(@intFromPtr(report), @sizeOf(Report), @intFromPtr(policy_raw), @sizeOf(Policy)) or
        overlaps(@intFromPtr(report), @sizeOf(Report), @intFromPtr(provider_raw), @sizeOf(c.heteris_runtime_provider_v1))) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const policy: Policy = @as(*const Policy, @ptrCast(@alignCast(policy_raw))).*;
    if (policy.abi_version != 1 or policy.struct_size < @sizeOf(Policy) or
        policy.reserved[0] != 0 or policy.reserved[1] != 0) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    if (policy.algorithm != 1) return c.ZYNUM_NATIVE_UNSUPPORTED;
    const provider: *const c.heteris_runtime_provider_v1 = @ptrCast(@alignCast(provider_raw));
    var result: execution.CallReport = .{};
    const initialized = Context.initIntegrated(provider, .{ .algorithm = .portable_generic, .max_parallel_tasks = policy.max_parallel_tasks }, &result) catch |err| {
        reportResult(report, result);
        return errorStatus(err);
    };
    ctx.* = initialized;
    reportResult(report, result);
    report.close_state = c.ZYNUM_NATIVE_CLOSE_OPEN;
    return c.ZYNUM_NATIVE_OK;
}
export fn zynum_native_context_request_close_v1(storage: ?*anyopaque, report_arg: ?*align(1) Report) i32 {
    const report = reportPointer(report_arg, storage, @sizeOf(Context)) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const ctx = contextPointer(storage) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    switch (ctx.requestClose()) {
        .immortal => return c.ZYNUM_NATIVE_UNSUPPORTED,
        .pending => report.close_state = c.ZYNUM_NATIVE_CLOSE_CLOSING,
        .closed => report.close_state = c.ZYNUM_NATIVE_CLOSE_CLOSED,
    }
    return c.ZYNUM_NATIVE_OK;
}
export fn zynum_native_context_progress_close_v1(storage: ?*anyopaque, report_arg: ?*align(1) Report) i32 {
    const report = reportPointer(report_arg, storage, @sizeOf(Context)) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const ctx = contextPointer(storage) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    var result: execution.CallReport = .{};
    const state = ctx.tryProgressClose(&result);
    reportResult(report, result);
    return switch (state) {
        .immortal => c.ZYNUM_NATIVE_UNSUPPORTED,
        .busy => blk: {
            report.close_state = c.ZYNUM_NATIVE_CLOSE_BUSY;
            break :blk c.ZYNUM_NATIVE_BUSY;
        },
        .waiting_resources => blk: {
            report.close_state = c.ZYNUM_NATIVE_CLOSE_WAITING_RESOURCES;
            break :blk c.ZYNUM_NATIVE_BUSY;
        },
        .release_failed => blk: {
            report.close_state = c.ZYNUM_NATIVE_CLOSE_RELEASE_FAILED;
            break :blk c.ZYNUM_NATIVE_RELEASE_FAILED;
        },
        .closed => blk: {
            report.close_state = c.ZYNUM_NATIVE_CLOSE_CLOSED;
            break :blk c.ZYNUM_NATIVE_OK;
        },
    };
}
fn coreIdentity() callconv(.c) *const anyopaque {
    return native.coreIdentity();
}
fn executionIdentity() callconv(.c) *const anyopaque {
    return execution.executionIdentity();
}
comptime {
    if (options.native_identity_tests) {
        @export(&coreIdentity, .{ .name = "zynum_native_core_identity_v1" });
        @export(&executionIdentity, .{ .name = "zynum_native_execution_identity_v1" });
    }
}

const BufferError = error{ InvalidView, Overflow };
const BufferRange = struct { address: usize, backing: usize, backing_bytes: usize, required_bytes: usize };
fn dimension(value: u64) BufferError!i32 {
    return std.math.cast(i32, value) orelse error.Overflow;
}
fn matrixRange(address: usize, backing64: u64, rows: i32, columns: i32, leading: i32) BufferError!BufferRange {
    _ = try dimension(backing64);
    const backing = std.math.cast(usize, backing64) orelse return error.Overflow;
    const row_count: usize = @intCast(rows);
    const column_count: usize = @intCast(columns);
    const stride: usize = @intCast(leading);
    var required: usize = 0;
    if (row_count != 0 and column_count != 0) {
        if (stride < row_count) return error.InvalidView;
        const last_column = std.math.mul(usize, column_count - 1, stride) catch return error.Overflow;
        required = std.math.add(usize, last_column, row_count) catch return error.Overflow;
    }
    if (required > backing) return error.InvalidView;
    const bytes = std.math.mul(usize, backing, @sizeOf(f64)) catch return error.Overflow;
    const required_bytes = std.math.mul(usize, required, @sizeOf(f64)) catch return error.Overflow;
    if (address == 0) {
        if (backing != 0 or required != 0) return error.InvalidView;
    } else {
        if (address % @alignOf(f64) != 0) return error.InvalidView;
        _ = std.math.add(usize, address, bytes) catch return error.Overflow;
    }
    return .{ .address = address, .backing = backing, .backing_bytes = bytes, .required_bytes = required_bytes };
}
fn bufferStatus(err: BufferError) i32 {
    return switch (err) {
        error.InvalidView => c.ZYNUM_NATIVE_VIEW_INVALID,
        error.Overflow => c.ZYNUM_NATIVE_OVERFLOW,
    };
}
fn precheckMetadata(report_address: usize, storage_address: usize, address: usize, backing64: u64) bool {
    // Invalid unrepresentable extents are diagnosed after report initialization.
    // Representable metadata aliases are rejected before any report bytes change.
    const count = std.math.cast(usize, backing64) orelse return false;
    const bytes = std.math.mul(usize, count, @sizeOf(f64)) catch return false;
    _ = std.math.add(usize, address, bytes) catch return false;
    return overlaps(report_address, @sizeOf(Report), address, bytes) or overlaps(storage_address, @sizeOf(Context), address, bytes);
}
const empty_f64: [0]f64 = .{};
fn constSlice(range: BufferRange) []const f64 {
    if (range.address == 0) return &empty_f64;
    const pointer: [*]const f64 = @ptrFromInt(range.address);
    return pointer[0..range.backing];
}
fn mutableSlice(range: BufferRange) []f64 {
    if (range.address == 0) return @constCast(&empty_f64);
    const pointer: [*]f64 = @ptrFromInt(range.address);
    return pointer[0..range.backing];
}
fn mathematicalStatus(err: native.NativeError) i32 {
    return switch (err) {
        error.DimensionMismatch, error.InvalidLength, error.InvalidStride, error.InvalidLeadingDimension, error.BufferTooSmall, error.WorkspaceTooSmall, error.AliasingNotAllowed => c.ZYNUM_NATIVE_VIEW_INVALID,
        error.InvalidProvider => c.ZYNUM_NATIVE_PROVIDER_CONTRACT_FAILURE,
        error.Unsupported => c.ZYNUM_NATIVE_UNSUPPORTED,
        error.ContextClosing => c.ZYNUM_NATIVE_CONTEXT_CLOSING,
        error.CounterOverflow => c.ZYNUM_NATIVE_COUNTER_OVERFLOW,
        error.ResourceBudgetExceeded => c.ZYNUM_NATIVE_BUDGET_EXCEEDED,
        error.OutOfMemory => c.ZYNUM_NATIVE_OUT_OF_MEMORY,
        error.ProviderBusy => c.ZYNUM_NATIVE_BUSY,
        error.ProviderClosed => c.ZYNUM_NATIVE_CLOSED,
        error.WouldDeadlock => c.ZYNUM_NATIVE_WOULD_DEADLOCK,
        error.ProviderRejected => c.ZYNUM_NATIVE_PROVIDER_REJECTED,
        error.ProviderExecutionFailed => c.ZYNUM_NATIVE_EXECUTION_FAILURE,
        error.ProviderCallbackFailed => c.ZYNUM_NATIVE_CALLBACK_FAILURE,
        error.ExtentOverflow => c.ZYNUM_NATIVE_OVERFLOW,
    };
}
export fn zynum_native_gemm_f64_nn_v1(storage: ?*anyopaque, m64: u64, n64: u64, k64: u64, alpha: f64, a_raw: ?*align(1) const f64, a_backing: u64, lda64: u64, b_raw: ?*align(1) const f64, b_backing: u64, ldb64: u64, beta: f64, c_raw: ?*align(1) f64, c_backing: u64, ldc64: u64, report_arg: ?*align(1) Report) i32 {
    const a_address = if (a_raw) |pointer| @intFromPtr(pointer) else 0;
    const b_address = if (b_raw) |pointer| @intFromPtr(pointer) else 0;
    const c_address = if (c_raw) |pointer| @intFromPtr(pointer) else 0;
    const storage_address = if (storage) |pointer| @intFromPtr(pointer) else 0;
    if (report_arg) |raw_report| {
        const report_address = @intFromPtr(raw_report);
        if (precheckMetadata(report_address, storage_address, a_address, a_backing) or
            precheckMetadata(report_address, storage_address, b_address, b_backing) or
            precheckMetadata(report_address, storage_address, c_address, c_backing)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    }
    const report = reportPointer(report_arg, storage, @sizeOf(Context)) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const ctx = contextPointer(storage) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const m = dimension(m64) catch |err| return bufferStatus(err);
    const n = dimension(n64) catch |err| return bufferStatus(err);
    const k = dimension(k64) catch |err| return bufferStatus(err);
    const lda = dimension(lda64) catch |err| return bufferStatus(err);
    const ldb = dimension(ldb64) catch |err| return bufferStatus(err);
    const ldc = dimension(ldc64) catch |err| return bufferStatus(err);
    const a = matrixRange(a_address, a_backing, m, k, lda) catch |err| return bufferStatus(err);
    const b = matrixRange(b_address, b_backing, k, n, ldb) catch |err| return bufferStatus(err);
    const output = matrixRange(c_address, c_backing, m, n, ldc) catch |err| return bufferStatus(err);
    if (!std.math.isFinite(alpha) or !std.math.isFinite(beta)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    if (overlaps(output.address, output.required_bytes, a.address, a.required_bytes) or
        overlaps(output.address, output.required_bytes, b.address, b.required_bytes)) return c.ZYNUM_NATIVE_VIEW_INVALID;
    // All dimensions, capacities, byte/address arithmetic, alignments and
    // metadata/output aliases are proven before forming any borrowed slice.
    const args: native.MatrixProductF64 = .{
        .left_matrix = .{ .values = constSlice(a), .row_count = m, .column_count = k, .leading_dimension = lda },
        .right_matrix = .{ .values = constSlice(b), .row_count = k, .column_count = n, .leading_dimension = ldb },
        .result_matrix = .{ .values = mutableSlice(output), .row_count = m, .column_count = n, .leading_dimension = ldc },
        .product_scale = alpha,
        .result_scale = beta,
    };
    var result: execution.CallReport = .{};
    native.matrixMultiplyF64NN(ctx, args, &result) catch |err| {
        reportResult(report, result);
        return mathematicalStatus(err);
    };
    reportResult(report, result);
    return c.ZYNUM_NATIVE_OK;
}

export fn zynum_native_gemm_f64_tn_v1(storage: ?*anyopaque, m64: u64, n64: u64, k64: u64, alpha: f64, a_raw: ?*align(1) const f64, a_backing: u64, lda64: u64, b_raw: ?*align(1) const f64, b_backing: u64, ldb64: u64, beta: f64, c_raw: ?*align(1) f64, c_backing: u64, ldc64: u64, report_arg: ?*align(1) Report) i32 {
    const a_address = if (a_raw) |pointer| @intFromPtr(pointer) else 0;
    const b_address = if (b_raw) |pointer| @intFromPtr(pointer) else 0;
    const c_address = if (c_raw) |pointer| @intFromPtr(pointer) else 0;
    const storage_address = if (storage) |pointer| @intFromPtr(pointer) else 0;
    if (report_arg) |raw_report| {
        const report_address = @intFromPtr(raw_report);
        if (precheckMetadata(report_address, storage_address, a_address, a_backing) or
            precheckMetadata(report_address, storage_address, b_address, b_backing) or
            precheckMetadata(report_address, storage_address, c_address, c_backing)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    }
    const report = reportPointer(report_arg, storage, @sizeOf(Context)) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const ctx = contextPointer(storage) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const m = dimension(m64) catch |err| return bufferStatus(err);
    const n = dimension(n64) catch |err| return bufferStatus(err);
    const k = dimension(k64) catch |err| return bufferStatus(err);
    const lda = dimension(lda64) catch |err| return bufferStatus(err);
    const ldb = dimension(ldb64) catch |err| return bufferStatus(err);
    const ldc = dimension(ldc64) catch |err| return bufferStatus(err);
    const a = matrixRange(a_address, a_backing, k, m, lda) catch |err| return bufferStatus(err);
    const b = matrixRange(b_address, b_backing, k, n, ldb) catch |err| return bufferStatus(err);
    const output = matrixRange(c_address, c_backing, m, n, ldc) catch |err| return bufferStatus(err);
    if (!std.math.isFinite(alpha) or !std.math.isFinite(beta)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    if (overlaps(output.address, output.required_bytes, a.address, a.required_bytes) or
        overlaps(output.address, output.required_bytes, b.address, b.required_bytes)) return c.ZYNUM_NATIVE_VIEW_INVALID;
    // All dimensions, capacities, byte/address arithmetic, alignments and
    // metadata/output aliases are proven before forming any borrowed slice.
    const args: native.MatrixProductF64 = .{
        .left_matrix = .{ .values = constSlice(a), .row_count = k, .column_count = m, .leading_dimension = lda, .operation = .transposed },
        .right_matrix = .{ .values = constSlice(b), .row_count = k, .column_count = n, .leading_dimension = ldb },
        .result_matrix = .{ .values = mutableSlice(output), .row_count = m, .column_count = n, .leading_dimension = ldc },
        .product_scale = alpha,
        .result_scale = beta,
    };
    var result: execution.CallReport = .{};
    native.matrixMultiplyF64TN(ctx, args, &result) catch |err| {
        reportResult(report, result);
        return mathematicalStatus(err);
    };
    reportResult(report, result);
    return c.ZYNUM_NATIVE_OK;
}

// Heteris Data COMPLEX128 is interleaved real/imaginary binary64, size 16,
// alignment 8. Prove the one original Core scalar representation before casts.
const C128 = native.ComplexF64;
comptime {
    if (@sizeOf(f64) != 8 or @alignOf(f64) != 8 or @sizeOf(C128) != 16 or @alignOf(C128) != 8 or
        @offsetOf(C128, "re") != 0 or @offsetOf(C128, "im") != 8)
        @compileError("Core ComplexF64 does not match Heteris Data COMPLEX128");
    if (c.heteris_dtype_size(c.HETERIS_COMPLEX128) != @sizeOf(C128) or
        c.heteris_dtype_alignment(c.HETERIS_COMPLEX128) != @alignOf(C128))
        @compileError("Heteris Data COMPLEX128 layout differs from the Core");
}
fn complexDeclaredBytes(address: usize, capacity64: u64) ?usize {
    const capacity = std.math.cast(usize, capacity64) orelse return null;
    const bytes = std.math.mul(usize, capacity, @sizeOf(C128)) catch return null;
    _ = std.math.add(usize, address, bytes) catch return null;
    return bytes;
}
fn complexMatrixRange(address: usize, capacity64: u64, bytes: usize, rows: i32, columns: i32, leading: i32) BufferError!BufferRange {
    const capacity = std.math.cast(usize, capacity64) orelse return error.Overflow;
    var required: usize = 0;
    if (rows != 0 and columns != 0) {
        if (leading < rows) return error.InvalidView;
        const offset = std.math.mul(usize, @intCast(columns - 1), @intCast(leading)) catch return error.Overflow;
        required = std.math.add(usize, offset, @intCast(rows)) catch return error.Overflow;
    }
    const required_bytes = std.math.mul(usize, required, @sizeOf(C128)) catch return error.Overflow;
    if (required > capacity) return error.InvalidView;
    if (address == 0) {
        if (capacity != 0 or required != 0) return error.InvalidView;
    } else if (address % @alignOf(C128) != 0) return error.InvalidView;
    return .{ .address = address, .backing = capacity, .backing_bytes = bytes, .required_bytes = required_bytes };
}
const empty_c128: [0]C128 = .{};
fn constComplexSlice(range: BufferRange) []const C128 {
    if (range.address == 0) return &empty_c128;
    const values: [*]const C128 = @ptrFromInt(range.address);
    return values[0..range.backing];
}
fn mutableComplexSlice(range: BufferRange) []C128 {
    if (range.address == 0) return @constCast(&empty_c128);
    const values: [*]C128 = @ptrFromInt(range.address);
    return values[0..range.backing];
}
export fn zynum_native_gemm_c128_row_major_cn_v1(storage: ?*anyopaque, m64: u64, n64: u64, k64: u64, alpha_re: f64, alpha_im: f64, a_raw: ?*align(1) const f64, a_capacity: u64, lda64: u64, b_raw: ?*align(1) const f64, b_capacity: u64, ldb64: u64, beta_re: f64, beta_im: f64, c_raw: ?*align(1) f64, c_capacity: u64, ldc64: u64, report_raw: ?*align(1) Report) i32 {
    // Prove every complete declared range and control separation before a
    // report write, borrowed slice, Context access, allocation or submission.
    const ctx = contextPointer(storage) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const report_pointer = report_raw orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const report_address = @intFromPtr(report_pointer);
    const context_address = @intFromPtr(ctx);
    if (!validAddress(report_address, @sizeOf(Report), @alignOf(Report)) or
        overlaps(report_address, @sizeOf(Report), context_address, @sizeOf(Context))) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const addresses = [3]usize{
        if (a_raw) |value| @intFromPtr(value) else 0,
        if (b_raw) |value| @intFromPtr(value) else 0,
        if (c_raw) |value| @intFromPtr(value) else 0,
    };
    const capacities = [3]u64{ a_capacity, b_capacity, c_capacity };
    var bytes: [3]usize = undefined;
    for (addresses, capacities, 0..) |address, capacity, index| {
        bytes[index] = complexDeclaredBytes(address, capacity) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
        if (overlaps(report_address, @sizeOf(Report), address, bytes[index]) or
            overlaps(context_address, @sizeOf(Context), address, bytes[index])) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    }
    const m = dimension(m64) catch |err| return bufferStatus(err);
    const n = dimension(n64) catch |err| return bufferStatus(err);
    const k = dimension(k64) catch |err| return bufferStatus(err);
    const lda = dimension(lda64) catch |err| return bufferStatus(err);
    const ldb = dimension(ldb64) catch |err| return bufferStatus(err);
    const ldc = dimension(ldc64) catch |err| return bufferStatus(err);
    // Row-major physical A(k,m), B(k,n), C(m,n) are borrowed without
    // copies as column-major A(m,k), B(n,k), C(n,m).
    const a = complexMatrixRange(addresses[0], a_capacity, bytes[0], m, k, lda) catch |err| return bufferStatus(err);
    const b = complexMatrixRange(addresses[1], b_capacity, bytes[1], n, k, ldb) catch |err| return bufferStatus(err);
    const output = complexMatrixRange(addresses[2], c_capacity, bytes[2], n, m, ldc) catch |err| return bufferStatus(err);
    if (!std.math.isFinite(alpha_re) or !std.math.isFinite(alpha_im) or
        !std.math.isFinite(beta_re) or !std.math.isFinite(beta_im)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    if (overlaps(output.address, output.backing_bytes, a.address, a.backing_bytes) or
        overlaps(output.address, output.backing_bytes, b.address, b.backing_bytes)) return c.ZYNUM_NATIVE_VIEW_INVALID;
    const report = reportPointer(report_raw, storage, @sizeOf(Context)) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    // C^T = alpha * B^T * conjugate(A) + beta*C^T: NC with dimensions
    // (n,m,k). Neither alpha nor beta is conjugated by this storage mapping.
    const args: native.MatrixProductC128 = .{
        .left_matrix = .{ .values = constComplexSlice(b), .row_count = n, .column_count = k, .leading_dimension = ldb },
        .right_matrix = .{ .values = constComplexSlice(a), .row_count = m, .column_count = k, .leading_dimension = lda, .operation = .adjoint },
        .result_matrix = .{ .values = mutableComplexSlice(output), .row_count = n, .column_count = m, .leading_dimension = ldc },
        .product_scale = .{ .re = alpha_re, .im = alpha_im },
        .result_scale = .{ .re = beta_re, .im = beta_im },
    };
    var result: execution.CallReport = .{};
    native.matrixMultiplyC128NC(ctx, args, &result) catch |err| {
        reportResult(report, result);
        return mathematicalStatus(err);
    };
    reportResult(report, result);
    return c.ZYNUM_NATIVE_OK;
}

// Full declared ranges, including empty-grid tails, must be computable before
// any control bytes change. This intentionally does not alter GEMM precedence.
fn gridDeclaredBytes(address: usize, backing: u64) ?usize {
    const count = std.math.cast(usize, backing) orelse return null;
    const bytes = std.math.mul(usize, count, @sizeOf(f64)) catch return null;
    _ = std.math.add(usize, address, bytes) catch return null;
    return bytes;
}
fn gridErrorStatus(err: native.PeriodicLaplacianError) i32 {
    return switch (err) {
        error.InvalidView, error.AliasingNotAllowed => c.ZYNUM_NATIVE_VIEW_INVALID,
        error.InvalidCoefficient => c.ZYNUM_NATIVE_INVALID_ARGUMENT,
        else => |execution_error| errorStatus(execution_error),
    };
}
export fn zynum_native_laplacian_periodic_3d_f64_v1(storage: ?*anyopaque, n0: u64, n1: u64, n2: u64, x_raw: ?*align(1) const f64, x_backing: u64, y_raw: ?*align(1) f64, y_backing: u64, h0: f64, h1: f64, h2: f64, report_raw: ?*align(1) Report) i32 {
    const storage_address = if (storage) |pointer| @intFromPtr(pointer) else 0;
    const report_address = if (report_raw) |pointer| @intFromPtr(pointer) else 0;
    const x_address = if (x_raw) |pointer| @intFromPtr(pointer) else 0;
    const y_address = if (y_raw) |pointer| @intFromPtr(pointer) else 0;
    // Step 0: no dereference, report initialization, admission or buffer slice.
    if (!validAddress(storage_address, @sizeOf(Context), @alignOf(Context)) or
        !validAddress(report_address, @sizeOf(Report), @alignOf(Report)) or
        overlaps(storage_address, @sizeOf(Context), report_address, @sizeOf(Report))) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const x_bytes = gridDeclaredBytes(x_address, x_backing) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const y_bytes = gridDeclaredBytes(y_address, y_backing) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    if (overlaps(storage_address, @sizeOf(Context), x_address, x_bytes) or
        overlaps(storage_address, @sizeOf(Context), y_address, y_bytes) or
        overlaps(report_address, @sizeOf(Report), x_address, x_bytes) or
        overlaps(report_address, @sizeOf(Report), y_address, y_bytes)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const report: *Report = @ptrCast(@alignCast(report_raw.?));
    report.* = std.mem.zeroes(Report);
    report.abi_version = 1;
    report.struct_size = @sizeOf(Report);
    // Step 2: independently representable axes; zero precedes multiplication.
    const shape: [3]usize = .{
        std.math.cast(usize, n0) orelse return c.ZYNUM_NATIVE_OVERFLOW,
        std.math.cast(usize, n1) orelse return c.ZYNUM_NATIVE_OVERFLOW,
        std.math.cast(usize, n2) orelse return c.ZYNUM_NATIVE_OVERFLOW,
    };
    const count = if (shape[0] == 0 or shape[1] == 0 or shape[2] == 0) 0 else blk: {
        const plane = std.math.mul(usize, shape[0], shape[1]) catch return c.ZYNUM_NATIVE_OVERFLOW;
        break :blk std.math.mul(usize, plane, shape[2]) catch return c.ZYNUM_NATIVE_OVERFLOW;
    };
    const required_bytes = std.math.mul(usize, count, @sizeOf(f64)) catch return c.ZYNUM_NATIVE_OVERFLOW;
    // Step 3: X then Y address/null/capacity, including empty geometry.
    if ((x_address == 0 and (x_backing != 0 or count != 0)) or
        x_address % @alignOf(f64) != 0 or x_backing < count) return c.ZYNUM_NATIVE_VIEW_INVALID;
    if ((y_address == 0 and (y_backing != 0 or count != 0)) or
        y_address % @alignOf(f64) != 0 or y_backing < count) return c.ZYNUM_NATIVE_VIEW_INVALID;
    const h: [3]f64 = .{ h0, h1, h2 };
    for (h) |value| if (!std.math.isFinite(value) or value <= 0) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    if (overlaps(x_address, x_bytes, y_address, y_bytes)) return c.ZYNUM_NATIVE_VIEW_INVALID;
    const x: BufferRange = .{ .address = x_address, .backing = @intCast(x_backing), .backing_bytes = x_bytes, .required_bytes = required_bytes };
    const y: BufferRange = .{ .address = y_address, .backing = @intCast(y_backing), .backing_bytes = y_bytes, .required_bytes = required_bytes };
    const ctx: *Context = @ptrCast(@alignCast(storage.?));
    var result: execution.CallReport = .{};
    native.periodicLaplacianF64(ctx, .{ .shape = shape, .input = constSlice(x), .output = mutableSlice(y), .inv_h2 = h }, &result) catch |err| {
        reportResult(report, result);
        return gridErrorStatus(err);
    };
    reportResult(report, result);
    return c.ZYNUM_NATIVE_OK;
}

const solve = native.periodic_neg_laplacian;
const SolveProblem = c.zynum_native_periodic_neg_laplacian_problem_v1;
const SolveRequirements = c.zynum_native_periodic_neg_laplacian_requirements_v1;
const SolveReport = c.zynum_native_periodic_neg_laplacian_report_v1;

fn solveProblem(problem: *const SolveProblem) error{ InvalidArgument, ExtentOverflow }!solve.Problem {
    if (problem.abi_version != 1 or problem.reserved0 != 0 or problem.reserved[0] != 0 or problem.reserved[1] != 0)
        return error.InvalidArgument;
    var shape: [3]usize = undefined;
    for (problem.shape, &shape) |axis, *length| length.* = std.math.cast(usize, axis) orelse return error.ExtentOverflow;
    return .{
        .shape = shape,
        .inv_h2 = problem.inv_h2,
        .max_iterations = problem.max_iterations,
        .absolute_tolerance_l2 = problem.absolute_tolerance_l2,
        .relative_tolerance = problem.relative_tolerance,
    };
}

fn solveErrorStatus(err: solve.Error) i32 {
    return switch (err) {
        error.InvalidArgument => c.ZYNUM_NATIVE_INVALID_ARGUMENT,
        else => |grid_error| gridErrorStatus(grid_error),
    };
}

export fn zynum_native_periodic_neg_laplacian_query_f64_v1(problem_arg: ?*align(1) const SolveProblem, out_arg: ?*align(1) SolveRequirements) i32 {
    const raw_problem = problem_arg orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const raw_out = out_arg orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const problem_address = @intFromPtr(raw_problem);
    const out_address = @intFromPtr(raw_out);
    if (!validAddress(problem_address, @sizeOf(SolveProblem), @alignOf(SolveProblem)) or
        !validAddress(out_address, @sizeOf(SolveRequirements), @alignOf(SolveRequirements)) or
        overlaps(problem_address, @sizeOf(SolveProblem), out_address, @sizeOf(SolveRequirements))) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const input: *const SolveProblem = @alignCast(raw_problem);
    if (input.struct_size != @sizeOf(SolveProblem)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const problem = solveProblem(input) catch |err| return solveErrorStatus(err);
    const sizes = solve.query(problem) catch |err| return solveErrorStatus(err);
    const out: *SolveRequirements = @alignCast(raw_out);
    out.* = std.mem.zeroes(SolveRequirements);
    out.abi_version = 1;
    out.struct_size = @sizeOf(SolveRequirements);
    out.points = sizes.points;
    out.workspace_elements = sizes.workspace_elements;
    out.workspace_bytes = sizes.workspace_bytes;
    out.workspace_alignment = solve.Requirements.workspace_alignment;
    return c.ZYNUM_NATIVE_OK;
}

fn writeSolveReport(out: *SolveReport, result: solve.Report) void {
    out.* = std.mem.zeroes(SolveReport);
    out.abi_version = 1;
    out.struct_size = @sizeOf(SolveReport);
    out.phase = @backingInt(result.phase);
    out.stop_reason = @backingInt(result.stop_reason);
    if (result.rhs) |rhs| {
        out.valid_flags |= c.ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_RHS_VALID;
        out.rhs_mean = rhs.mean;
        out.rhs_l2 = rhs.norm_l2;
        out.projected_rhs_l2 = rhs.projected_norm_l2;
    }
    if (result.solver) |solver| {
        out.valid_flags |= c.ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_SOLVER_VALID;
        out.iterations = solver.iterations;
        out.solver_actions = solver.action_count;
        out.residual_restarts = solver.residual_restarts;
    }
    if (result.final) |final| {
        out.valid_flags |= c.ZYNUM_NATIVE_PERIODIC_NEG_LAPLACIAN_FINAL_VALID;
        out.solution_mean = final.solution_mean;
        out.residual_l2 = final.residual_l2;
    }
    out.native.abi_version = 1;
    out.native.struct_size = @sizeOf(Report);
    reportResult(&out.native, result.native);
}

const CheckedRange = struct {
    address: usize,
    bytes: usize,
    fn control(pointer: anytype) ?CheckedRange {
        const T = @typeInfo(@TypeOf(pointer)).pointer.child;
        const address = @intFromPtr(pointer);
        if (!validAddress(address, @sizeOf(T), @alignOf(T))) return null;
        return .{ .address = address, .bytes = @sizeOf(T) };
    }
    fn backing(pointer: anytype, count: u64) BufferError!CheckedRange {
        const address = if (pointer) |value| @intFromPtr(value) else 0;
        if (address == 0 and count != 0) return error.InvalidView;
        const bytes = gridDeclaredBytes(address, count) orelse return error.Overflow;
        if (bytes > std.math.maxInt(isize)) return error.Overflow;
        return .{ .address = address, .bytes = bytes };
    }
    fn intersects(self: CheckedRange, other: CheckedRange) bool {
        return overlaps(self.address, self.bytes, other.address, other.bytes);
    }
    fn fits(self: CheckedRange, count: usize) bool {
        return self.address != 0 and self.address % @alignOf(f64) == 0 and self.bytes / @sizeOf(f64) >= count;
    }
};

export fn zynum_native_periodic_neg_laplacian_solve_f64_v1(
    storage: ?*anyopaque,
    problem_arg: ?*align(1) const SolveProblem,
    rhs_arg: ?*align(1) const f64,
    rhs_count: u64,
    solution_arg: ?*align(1) f64,
    solution_count: u64,
    workspace_arg: ?*align(1) f64,
    workspace_count: u64,
    report_arg: ?*align(1) SolveReport,
) i32 {
    const ctx_raw: *align(1) Context = @ptrCast(storage orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT);
    const problem_raw = problem_arg orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const report_raw = report_arg orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const ctx_range = CheckedRange.control(ctx_raw) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const problem_range = CheckedRange.control(problem_raw) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const report_range = CheckedRange.control(report_raw) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const rhs = CheckedRange.backing(rhs_arg, rhs_count) catch |err| return bufferStatus(err);
    const solution = CheckedRange.backing(solution_arg, solution_count) catch |err| return bufferStatus(err);
    const workspace = CheckedRange.backing(workspace_arg, workspace_count) catch |err| return bufferStatus(err);
    const controls = [_]CheckedRange{ ctx_range, problem_range, report_range };
    const buffers = [_]CheckedRange{ rhs, solution, workspace };
    for (controls, 0..) |control, i| {
        for (controls[i + 1 ..]) |other| if (control.intersects(other)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
        for (buffers) |buffer| if (control.intersects(buffer)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    }
    if (rhs.intersects(solution) or rhs.intersects(workspace) or solution.intersects(workspace))
        return c.ZYNUM_NATIVE_VIEW_INVALID;

    const input: *const SolveProblem = @alignCast(problem_raw);
    // Only the fixed v1 range was checked. An unsupported declared size
    // cannot establish that the report is outside the caller's descriptor.
    if (input.struct_size != @sizeOf(SolveProblem)) return c.ZYNUM_NATIVE_INVALID_ARGUMENT;

    const report: *SolveReport = @alignCast(report_raw);
    var result: solve.Report = .{};
    defer writeSolveReport(report, result);
    const problem = solveProblem(input) catch |err| return solveErrorStatus(err);
    const sizes = solve.query(problem) catch |err| return solveErrorStatus(err);
    if (!rhs.fits(sizes.points) or !solution.fits(sizes.points) or !workspace.fits(sizes.workspace_elements))
        return c.ZYNUM_NATIVE_VIEW_INVALID;

    const b: [*]const f64 = @ptrFromInt(rhs.address);
    const x: [*]f64 = @ptrFromInt(solution.address);
    const w: [*]f64 = @ptrFromInt(workspace.address);
    const ctx: *Context = @alignCast(ctx_raw);
    solve.solve(ctx, problem, b[0..@intCast(rhs_count)], x[0..@intCast(solution_count)], w[0..@intCast(workspace_count)], &result) catch |err| return solveErrorStatus(err);
    return c.ZYNUM_NATIVE_OK;
}

const F64SumReport = c.zynum_native_f64_sum_summary_report_v1;

export fn zynum_native_f64_sum_summary_v1(values_arg: ?*align(1) const f64, backing_elements: u64, count: u64, scale: f64, limit: f64, out_arg: ?*align(1) F64SumReport) i32 {
    const raw_out = out_arg orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const out_range = CheckedRange.control(raw_out) orelse return c.ZYNUM_NATIVE_INVALID_ARGUMENT;
    const values_range = CheckedRange.backing(values_arg, backing_elements) catch |err| return switch (err) {
        error.InvalidView => c.ZYNUM_NATIVE_INVALID_ARGUMENT,
        error.Overflow => c.ZYNUM_NATIVE_OVERFLOW,
    };
    if (count == 0 or count > backing_elements or !values_range.fits(@intCast(count)) or out_range.intersects(values_range))
        return c.ZYNUM_NATIVE_INVALID_ARGUMENT;

    const values: [*]const f64 = @ptrFromInt(values_range.address);
    const result = native.f64_sum_summary.summarizeF64(values[0..@intCast(count)], scale, limit) catch |err| return switch (err) {
        error.OutputRange, error.AccumulatorRange => c.ZYNUM_NATIVE_OVERFLOW,
        error.InvalidLength, error.InvalidScale, error.InvalidLimit, error.NonFiniteInput => c.ZYNUM_NATIVE_INVALID_ARGUMENT,
    };
    const out: *F64SumReport = @alignCast(raw_out);
    out.* = std.mem.zeroes(F64SumReport);
    out.abi_version = 1;
    out.struct_size = @sizeOf(F64SumReport);
    out.count = result.count;
    out.sum_sign = result.sum_sign;
    if (result.exact_zero) out.flags |= c.ZYNUM_NATIVE_F64_SUM_EXACT_ZERO;
    if (result.quantization_zero_interior) out.flags |= c.ZYNUM_NATIVE_F64_SUM_QUANTIZATION_ZERO_INTERIOR;
    if (result.scaled_abs_sum_within_limit) out.flags |= c.ZYNUM_NATIVE_F64_SUM_SCALED_ABS_WITHIN_LIMIT;
    out.mean_lower = result.mean_lower;
    out.mean_upper = result.mean_upper;
    out.scaled_sum_lower = result.scaled_sum_lower;
    out.scaled_sum_upper = result.scaled_sum_upper;
    out.scaled_l1_lower = result.scaled_l1_lower;
    out.scaled_l1_upper = result.scaled_l1_upper;
    out.scaled_direction_half_width_lower = result.scaled_direction_half_width_lower;
    out.scaled_direction_half_width_upper = result.scaled_direction_half_width_upper;
    out.maximum_absolute = result.maximum_absolute;
    return c.ZYNUM_NATIVE_OK;
}
