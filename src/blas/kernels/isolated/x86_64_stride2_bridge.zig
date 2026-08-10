// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Main-graph typed facade for the isolated x86_64 stride-two object.

const builtin = @import("builtin");

const abi = @import("x86_64_stride2_abi.zig");
const task_runtime_host = @import("task_runtime_host.zig");
const types = @import("../../types.zig");

pub const BlasInt = types.BlasInt;
pub const ComplexF32 = types.ComplexF32;
pub const ComplexF64 = types.ComplexF64;

extern fn zynum_internal_x86_64_stride2_execute(request: *abi.Request) callconv(.c) u8;

comptime {
    _ = task_runtime_host;
}

fn Real(comptime T: type) type {
    if (T == f32 or T == ComplexF32) return f32;
    if (T == f64 or T == ComplexF64) return f64;
    @compileError("isolated stride-two bridge supports BLAS real and complex scalars");
}

fn scalarTag(comptime T: type) abi.Scalar {
    if (T == f32) return .f32;
    if (T == f64) return .f64;
    if (T == ComplexF32) return .complex_f32;
    if (T == ComplexF64) return .complex_f64;
    @compileError("isolated stride-two bridge supports BLAS real and complex scalars");
}

fn putReal(comptime T: type, args: *[8]u64, index: usize, value: T) void {
    if (T == f32) {
        args[index] = @as(u32, @bitCast(value));
    } else if (T == f64) {
        args[index] = @bitCast(value);
    } else {
        @compileError("putReal supports f32 and f64");
    }
}

fn getReal(comptime T: type, bits: u64) T {
    if (T == f32) return @bitCast(@as(u32, @truncate(bits)));
    if (T == f64) return @bitCast(bits);
    @compileError("getReal supports f32 and f64");
}

fn putScalar(comptime T: type, args: *[8]u64, index: usize, value: T) void {
    if (T == f32 or T == f64) {
        putReal(T, args, index, value);
    } else {
        putReal(Real(T), args, index, value.re);
        putReal(Real(T), args, index + 1, value.im);
    }
}

fn getScalar(comptime T: type, result: [2]u64) T {
    if (T == f32 or T == f64) return getReal(T, result[0]);
    return .{
        .re = getReal(Real(T), result[0]),
        .im = getReal(Real(T), result[1]),
    };
}

fn init(
    comptime T: type,
    operation: abi.Operation,
    n: usize,
    x: anytype,
    y: anytype,
) abi.Request {
    return abi.init(
        operation,
        scalarTag(T),
        n,
        @ptrCast(@constCast(x)),
        if (@TypeOf(y) == @TypeOf(null)) null else @ptrCast(@constCast(y)),
    );
}

fn execute(request: *abi.Request) bool {
    if (comptime builtin.cpu.arch != .x86_64) return false;
    return zynum_internal_x86_64_stride2_execute(request) != 0;
}

pub noinline fn parallelSwapStride2(comptime T: type, n: usize, x: [*]T, y: [*]T) bool {
    var request = init(T, .swap, n, x, y);
    return execute(&request);
}

pub noinline fn parallelRotStride2(comptime T: type, n: usize, x: [*]T, y: [*]T, c: Real(T), s: T) bool {
    var request = init(T, .rot, n, x, y);
    putReal(Real(T), &request.args, 0, c);
    putScalar(T, &request.args, 1, s);
    return execute(&request);
}

pub noinline fn parallelRotmStride2Real(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    var request = init(T, .rotm, n, x, y);
    inline for (.{ flag, h11, h21, h12, h22 }, 0..) |value, index| putReal(T, &request.args, index, value);
    return execute(&request);
}

pub noinline fn parallelScalStride2Real(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    var request = init(T, .scal, n, x, null);
    putReal(T, &request.args, 0, alpha);
    return execute(&request);
}

pub noinline fn parallelComplexScalStride2(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    var request = init(T, .scal, n, x, null);
    putScalar(T, &request.args, 0, alpha);
    return execute(&request);
}

pub noinline fn parallelComplexRealScalStride2(comptime T: type, n: usize, alpha: Real(T), x: [*]T) bool {
    var request = init(T, .rscal, n, x, null);
    putReal(Real(T), &request.args, 0, alpha);
    return execute(&request);
}

pub noinline fn parallelAxpyStride2Real(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    var request = init(T, .axpy, n, x, y);
    putReal(T, &request.args, 0, alpha);
    return execute(&request);
}

fn parallelComplexAxpyStride2(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    var request = init(T, .axpy, n, x, y);
    putScalar(T, &request.args, 0, alpha);
    return execute(&request);
}

pub noinline fn parallelComplexAxpyStride2C32(n: usize, alpha: ComplexF32, x: [*]const ComplexF32, y: [*]ComplexF32) bool {
    return parallelComplexAxpyStride2(ComplexF32, n, alpha, x, y);
}

pub noinline fn parallelComplexAxpyStride2C64(n: usize, alpha: ComplexF64, x: [*]const ComplexF64, y: [*]ComplexF64) bool {
    return parallelComplexAxpyStride2(ComplexF64, n, alpha, x, y);
}

pub noinline fn parallelAxpbyStride2Real(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    var request = init(T, .axpby, n, x, y);
    putReal(T, &request.args, 0, alpha);
    putReal(T, &request.args, 1, beta);
    return execute(&request);
}

pub noinline fn parallelComplexAxpbyStride2(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    var request = init(T, .axpby, n, x, y);
    putScalar(T, &request.args, 0, alpha);
    putScalar(T, &request.args, 2, beta);
    return execute(&request);
}

pub noinline fn parallelDotF32AccF64Stride2(n: usize, x: [*]const f32, y: [*]const f32) ?f64 {
    var request = init(f32, .dot_f32_acc_f64, n, x, y);
    if (!execute(&request)) return null;
    return getReal(f64, request.result[0]);
}

pub noinline fn parallelDotStride2Real(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    var request = init(T, .dot, n, x, y);
    if (!execute(&request)) return null;
    return getScalar(T, request.result);
}

pub noinline fn parallelDotStride2Complex(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    var request = init(T, .dot, n, x, y);
    if (conjx) request.flags |= abi.conjugate_x;
    if (!execute(&request)) return null;
    return getScalar(T, request.result);
}

pub noinline fn parallelAsumStride2Real(comptime T: type, n: usize, x: [*]const T) ?T {
    var request = init(T, .asum, n, x, null);
    if (!execute(&request)) return null;
    return getReal(T, request.result[0]);
}

pub noinline fn parallelAsumStride2Complex(comptime T: type, n: usize, x: [*]const T) ?Real(T) {
    var request = init(T, .asum, n, x, null);
    if (!execute(&request)) return null;
    return getReal(Real(T), request.result[0]);
}

pub noinline fn parallelNrm2Stride2Real(comptime T: type, n: usize, x: [*]const T) ?T {
    var request = init(T, .nrm2, n, x, null);
    if (!execute(&request)) return null;
    return getReal(T, request.result[0]);
}

pub noinline fn parallelNrm2Stride2Complex(comptime T: type, n: usize, x: [*]const T) ?Real(T) {
    var request = init(T, .nrm2, n, x, null);
    if (!execute(&request)) return null;
    return getReal(Real(T), request.result[0]);
}

pub noinline fn parallelIamaxStride2(comptime T: type, n: usize, x: [*]const T) ?BlasInt {
    var request = init(T, .iamax, n, x, null);
    if (!execute(&request)) return null;
    return @intCast(request.result_index);
}
