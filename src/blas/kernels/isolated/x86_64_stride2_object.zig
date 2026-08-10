// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Separately compiled x86_64 stride-two implementation object.

const abi = @import("x86_64_stride2_abi.zig");
const parallel = @import("../../core/vector/stride2_parallel.zig");
const types = @import("../../types.zig");

const ComplexF32 = types.ComplexF32;
const ComplexF64 = types.ComplexF64;

fn realArg(comptime T: type, request: *const abi.Request, index: usize) T {
    if (T == f32) return @bitCast(@as(u32, @truncate(request.args[index])));
    if (T == f64) return @bitCast(request.args[index]);
    @compileError("realArg supports f32 and f64");
}

fn scalarArg(comptime T: type, request: *const abi.Request, index: usize) T {
    if (T == f32 or T == f64) return realArg(T, request, index);
    const R = if (T == ComplexF32) f32 else f64;
    return .{ .re = realArg(R, request, index), .im = realArg(R, request, index + 1) };
}

fn setRealResult(comptime T: type, request: *abi.Request, value: T) void {
    if (T == f32) {
        request.result[0] = @as(u32, @bitCast(value));
    } else if (T == f64) {
        request.result[0] = @bitCast(value);
    } else {
        @compileError("setRealResult supports f32 and f64");
    }
}

fn setScalarResult(comptime T: type, request: *abi.Request, value: T) void {
    if (T == f32 or T == f64) return setRealResult(T, request, value);
    const R = if (T == ComplexF32) f32 else f64;
    setRealResult(R, request, value.re);
    if (R == f32) {
        request.result[1] = @as(u32, @bitCast(value.im));
    } else {
        request.result[1] = @bitCast(value.im);
    }
}

fn xPtr(comptime T: type, request: *const abi.Request) [*]T {
    return @ptrCast(@alignCast(request.x));
}

fn constXPtr(comptime T: type, request: *const abi.Request) [*]const T {
    return @ptrCast(@alignCast(request.x));
}

fn yPtr(comptime T: type, request: *const abi.Request) [*]T {
    return @ptrCast(@alignCast(request.y.?));
}

fn runScal(comptime T: type, request: *abi.Request) bool {
    if (T == f32 or T == f64) {
        return parallel.parallelScalStride2Real(T, request.n, realArg(T, request, 0), xPtr(T, request));
    }
    return parallel.parallelComplexScalStride2(T, request.n, scalarArg(T, request, 0), xPtr(T, request));
}

fn runRealScal(comptime T: type, request: *abi.Request) bool {
    const R = if (T == ComplexF32) f32 else f64;
    return parallel.parallelComplexRealScalStride2(T, request.n, realArg(R, request, 0), xPtr(T, request));
}

fn runSwap(comptime T: type, request: *abi.Request) bool {
    return parallel.parallelSwapStride2(T, request.n, xPtr(T, request), yPtr(T, request));
}

fn runAxpy(comptime T: type, request: *abi.Request) bool {
    if (T == f32 or T == f64) {
        return parallel.parallelAxpyStride2Real(T, request.n, realArg(T, request, 0), constXPtr(T, request), yPtr(T, request));
    }
    if (T == ComplexF32) {
        return parallel.parallelComplexAxpyStride2C32(request.n, scalarArg(T, request, 0), constXPtr(T, request), yPtr(T, request));
    }
    return parallel.parallelComplexAxpyStride2C64(request.n, scalarArg(T, request, 0), constXPtr(T, request), yPtr(T, request));
}

fn runAxpby(comptime T: type, request: *abi.Request) bool {
    if (T == f32 or T == f64) {
        return parallel.parallelAxpbyStride2Real(T, request.n, realArg(T, request, 0), constXPtr(T, request), realArg(T, request, 1), yPtr(T, request));
    }
    return parallel.parallelComplexAxpbyStride2(T, request.n, scalarArg(T, request, 0), constXPtr(T, request), scalarArg(T, request, 2), yPtr(T, request));
}

fn runDot(comptime T: type, request: *abi.Request) bool {
    const value = if (T == f32 or T == f64)
        parallel.parallelDotStride2Real(T, request.n, constXPtr(T, request), @ptrCast(@alignCast(request.y.?)))
    else
        parallel.parallelDotStride2Complex(T, request.n, constXPtr(T, request), @ptrCast(@alignCast(request.y.?)), request.flags & abi.conjugate_x != 0);
    const result = value orelse return false;
    setScalarResult(T, request, result);
    return true;
}

fn runAsum(comptime T: type, request: *abi.Request) bool {
    if (T == f32 or T == f64) {
        const result = parallel.parallelAsumStride2Real(T, request.n, constXPtr(T, request)) orelse return false;
        setRealResult(T, request, result);
    } else {
        const R = if (T == ComplexF32) f32 else f64;
        const result = parallel.parallelAsumStride2Complex(T, request.n, constXPtr(T, request)) orelse return false;
        setRealResult(R, request, result);
    }
    return true;
}

fn runNrm2(comptime T: type, request: *abi.Request) bool {
    if (T == f32 or T == f64) {
        const result = parallel.parallelNrm2Stride2Real(T, request.n, constXPtr(T, request)) orelse return false;
        setRealResult(T, request, result);
    } else {
        const R = if (T == ComplexF32) f32 else f64;
        const result = parallel.parallelNrm2Stride2Complex(T, request.n, constXPtr(T, request)) orelse return false;
        setRealResult(R, request, result);
    }
    return true;
}

fn runIamax(comptime T: type, request: *abi.Request) bool {
    request.result_index = parallel.parallelIamaxStride2(T, request.n, constXPtr(T, request)) orelse return false;
    return true;
}

fn runRot(comptime T: type, request: *abi.Request) bool {
    const R = if (T == f32 or T == ComplexF32) f32 else f64;
    return parallel.parallelRotStride2(T, request.n, xPtr(T, request), yPtr(T, request), realArg(R, request, 0), scalarArg(T, request, 1));
}

fn runRotm(comptime T: type, request: *abi.Request) bool {
    return parallel.parallelRotmStride2Real(
        T,
        request.n,
        xPtr(T, request),
        yPtr(T, request),
        realArg(T, request, 0),
        realArg(T, request, 1),
        realArg(T, request, 2),
        realArg(T, request, 3),
        realArg(T, request, 4),
    );
}

fn dispatchScalar(comptime T: type, operation: abi.Operation, request: *abi.Request) bool {
    return switch (operation) {
        .scal => runScal(T, request),
        .rscal => if (T == ComplexF32 or T == ComplexF64) runRealScal(T, request) else false,
        .swap => runSwap(T, request),
        .axpy => runAxpy(T, request),
        .axpby => runAxpby(T, request),
        .dot => runDot(T, request),
        .dot_f32_acc_f64 => if (T == f32) dot: {
            const result = parallel.parallelDotF32AccF64Stride2(request.n, constXPtr(f32, request), @ptrCast(@alignCast(request.y.?))) orelse break :dot false;
            setRealResult(f64, request, result);
            break :dot true;
        } else false,
        .asum => runAsum(T, request),
        .nrm2 => runNrm2(T, request),
        .iamax => runIamax(T, request),
        .rot => runRot(T, request),
        .rotm => if (T == f32 or T == f64) runRotm(T, request) else false,
    };
}

fn execute(request: *abi.Request) callconv(.c) u8 {
    const operation: abi.Operation = @enumFromInt(request.operation);
    const scalar: abi.Scalar = @enumFromInt(request.scalar);
    const handled = switch (scalar) {
        .f32 => dispatchScalar(f32, operation, request),
        .f64 => dispatchScalar(f64, operation, request),
        .complex_f32 => dispatchScalar(ComplexF32, operation, request),
        .complex_f64 => dispatchScalar(ComplexF64, operation, request),
    };
    return @intFromBool(handled);
}

comptime {
    @export(&execute, .{ .name = "zynum_internal_x86_64_stride2_execute", .visibility = .hidden });
}
