// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Native correctness probe for the selected production Level 2 width object.

const std = @import("std");
const builtin = @import("builtin");
const contract = @import("level2_width_artifact_probe_contract.zig");

const ComplexF32 = extern struct { re: f32, im: f32 };
const ComplexF64 = extern struct { re: f64, im: f64 };

const ProbeError = error{
    BuiltAsTest,
    UnexpectedTarget,
    CandidateDisabled,
    RequestRejected,
    RejectedRequestChangedOutput,
    UnexpectedResult,
    GuardChanged,
};

fn rawReal(comptime T: type, value: T) u64 {
    return switch (T) {
        f32 => @as(u32, @bitCast(value)),
        f64 => @bitCast(value),
        else => @compileError("rawReal expects f32 or f64"),
    };
}

fn execute(request: *contract.Request) ProbeError!void {
    if (contract.zynum_internal_x86_64_level2_width_execute(request) != 1) {
        return error.RequestRejected;
    }
}

fn expectRealClose(comptime T: type, actual: T, expected: T) ProbeError!void {
    const tolerance: T = if (T == f32) 0.0005 else 0.0000000001;
    const scale = @max(@as(T, 1), @abs(expected));
    if (@abs(actual - expected) > tolerance * scale) return error.UnexpectedResult;
}

fn expectF32Exact(actual: []const f32, expected: []const f32) ProbeError!void {
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(actual), std.mem.sliceAsBytes(expected))) {
        return error.UnexpectedResult;
    }
}

fn verifyMinimalRealWitnesses() ProbeError!void {
    {
        const matrix = [_]f32{ 1, -2, 3, -4, 5, -6, 7, -8 };
        const x = [_]f32{2};
        var output_storage = [_]f32{ 1001, 10, 20, 30, 40, 50, 60, 70, 80, -1003 };
        const expected = [_]f32{ 11, 18, 33, 36, 55, 54, 77, 72 };
        var request: contract.Request = .{
            .operation = @intFromEnum(contract.Operation.gemv_no_trans_unit),
            .scalar = @intFromEnum(contract.Scalar.f32),
            .flags = 0,
            .lda = 8,
            .m = 8,
            .n = 1,
            .alpha_re = rawReal(f32, 0.5),
            .alpha_im = 0,
            .beta_re = rawReal(f32, 1),
            .beta_im = 0,
            .input0 = @ptrCast(&matrix),
            .input1 = @ptrCast(&x),
            .output = @ptrCast(&output_storage[1]),
        };
        try execute(&request);
        try expectF32Exact(output_storage[1 .. output_storage.len - 1], &expected);
        if (output_storage[0] != 1001 or output_storage[output_storage.len - 1] != -1003) {
            return error.GuardChanged;
        }
    }

    {
        const x = [_]f32{ 1, -2, 3, -4, 5, -6, 7, -8 };
        const y = [_]f32{2};
        var output_storage = [_]f32{ 2001, 11, 12, 13, 14, 15, 16, 17, 18, -2003 };
        const expected = [_]f32{ 12, 10, 16, 10, 20, 10, 24, 10 };
        var request: contract.Request = .{
            .operation = @intFromEnum(contract.Operation.ger_real),
            .scalar = @intFromEnum(contract.Scalar.f32),
            .flags = 0,
            .lda = 8,
            .m = 8,
            .n = 1,
            .alpha_re = rawReal(f32, 0.5),
            .alpha_im = 0,
            .beta_re = 0,
            .beta_im = 0,
            .input0 = @ptrCast(&x),
            .input1 = @ptrCast(&y),
            .output = @ptrCast(&output_storage[1]),
        };
        try execute(&request);
        try expectF32Exact(output_storage[1 .. output_storage.len - 1], &expected);
        if (output_storage[0] != 2001 or output_storage[output_storage.len - 1] != -2003) {
            return error.GuardChanged;
        }
    }
}

fn complexAdd(comptime T: type, lhs: T, rhs: T) T {
    return .{ .re = lhs.re + rhs.re, .im = lhs.im + rhs.im };
}

fn complexMul(comptime T: type, lhs: T, rhs: T) T {
    return .{
        .re = lhs.re * rhs.re - lhs.im * rhs.im,
        .im = lhs.re * rhs.im + lhs.im * rhs.re,
    };
}

fn complexConj(comptime T: type, value: T) T {
    return .{ .re = value.re, .im = -value.im };
}

fn expectComplexClose(comptime T: type, actual: T, expected: T) ProbeError!void {
    try expectRealClose(@TypeOf(actual.re), actual.re, expected.re);
    try expectRealClose(@TypeOf(actual.im), actual.im, expected.im);
}

fn realValue(index: usize) f32 {
    const signed: i32 = @as(i32, @intCast(index % 17)) - 8;
    return @as(f32, @floatFromInt(signed)) / 9.0;
}

fn verifyRealGemv() ProbeError!void {
    const m = 17;
    const n = 9;
    const lda = 21;
    const alpha: f32 = -1.375;
    const beta: f32 = 0.625;
    const guard: f32 = 19_937.25;

    var matrix: [lda * n]f32 = undefined;
    for (&matrix, 0..) |*value, index| value.* = realValue(index + 3);
    var x_no_trans: [n]f32 = undefined;
    for (&x_no_trans, 0..) |*value, index| value.* = realValue(index * 5 + 1);
    var x_trans: [m]f32 = undefined;
    for (&x_trans, 0..) |*value, index| value.* = realValue(index * 7 + 2);

    const operations = [_]contract.Operation{
        .gemv_no_trans_unit,
        .gemv_no_trans_full,
        .gemv_trans_unit,
        .gemv_trans_full,
    };
    for (operations) |operation| {
        const transposed = operation == .gemv_trans_unit or operation == .gemv_trans_full;
        const full = operation == .gemv_no_trans_full or operation == .gemv_trans_full;
        const output_len: usize = if (transposed) n else m;
        var output_storage: [m + 2]f32 = undefined;
        output_storage[0] = guard;
        output_storage[output_len + 1] = guard;
        const output = output_storage[1 .. output_len + 1];
        var expected: [m]f32 = undefined;
        for (output, 0..) |*value, index| {
            value.* = realValue(index * 11 + 4);
            expected[index] = value.*;
        }
        if (transposed) {
            for (0..n) |col| {
                var sum: f32 = 0;
                for (0..m) |row| sum += matrix[row + col * lda] * x_trans[row];
                expected[col] = (if (full) beta * expected[col] else expected[col]) + alpha * sum;
            }
        } else {
            for (0..m) |row| {
                var sum: f32 = 0;
                for (0..n) |col| sum += matrix[row + col * lda] * x_no_trans[col];
                expected[row] = (if (full) beta * expected[row] else expected[row]) + alpha * sum;
            }
        }
        var request: contract.Request = .{
            .operation = @intFromEnum(operation),
            .scalar = @intFromEnum(contract.Scalar.f32),
            .flags = 0,
            .lda = lda,
            .m = m,
            .n = n,
            .alpha_re = rawReal(f32, alpha),
            .alpha_im = 0,
            .beta_re = rawReal(f32, beta),
            .beta_im = 0,
            .input0 = @ptrCast(&matrix),
            .input1 = @ptrCast(if (transposed) x_trans[0..].ptr else x_no_trans[0..].ptr),
            .output = @ptrCast(output.ptr),
        };
        try execute(&request);
        for (output, expected[0..output_len]) |actual, wanted| try expectRealClose(f32, actual, wanted);
        if (output_storage[0] != guard or output_storage[output_len + 1] != guard) return error.GuardChanged;
    }
}

fn verifyRealGer() ProbeError!void {
    const m = 19;
    const n = 7;
    const lda = 23;
    const alpha: f32 = 1.125;
    const guard: f32 = -18_271.5;
    var x: [m]f32 = undefined;
    var y: [n]f32 = undefined;
    for (&x, 0..) |*value, index| value.* = realValue(index * 3 + 2);
    for (&y, 0..) |*value, index| value.* = realValue(index * 7 + 5);
    var output_storage: [lda * n + 2]f32 = undefined;
    output_storage[0] = guard;
    output_storage[output_storage.len - 1] = guard;
    const output = output_storage[1 .. output_storage.len - 1];
    var expected: [lda * n]f32 = undefined;
    for (output, &expected, 0..) |*actual, *wanted, index| {
        actual.* = realValue(index * 11 + 6);
        wanted.* = actual.*;
    }
    for (0..n) |col| {
        for (0..m) |row| expected[row + col * lda] += alpha * x[row] * y[col];
    }
    var request: contract.Request = .{
        .operation = @intFromEnum(contract.Operation.ger_real),
        .scalar = @intFromEnum(contract.Scalar.f32),
        .flags = 0,
        .lda = lda,
        .m = m,
        .n = n,
        .alpha_re = rawReal(f32, alpha),
        .alpha_im = 0,
        .beta_re = 0,
        .beta_im = 0,
        .input0 = @ptrCast(&x),
        .input1 = @ptrCast(&y),
        .output = @ptrCast(output.ptr),
    };
    try execute(&request);
    for (output, expected) |actual, wanted| try expectRealClose(f32, actual, wanted);
    if (output_storage[0] != guard or output_storage[output_storage.len - 1] != guard) return error.GuardChanged;
}

fn complexValue(comptime T: type, index: usize) T {
    const R = @TypeOf(@as(T, undefined).re);
    const re: i32 = @as(i32, @intCast(index % 13)) - 6;
    const im: i32 = @as(i32, @intCast((index * 5 + 3) % 11)) - 5;
    return .{
        .re = @as(R, @floatFromInt(re)) / 11.0,
        .im = @as(R, @floatFromInt(im)) / 13.0,
    };
}

fn sameComplex(comptime T: type, lhs: T, rhs: T) bool {
    return lhs.re == rhs.re and lhs.im == rhs.im;
}

fn verifyComplex(comptime T: type, scalar: contract.Scalar) !void {
    const R = @TypeOf(@as(T, undefined).re);
    const m = 128;
    const n = 128;
    const lda = 131;
    const alpha: T = .{ .re = 0.75, .im = -0.375 };
    const guard: T = .{ .re = 17_003.25, .im = -23_009.5 };
    const allocator = std.heap.page_allocator;

    const matrix_storage = try allocator.alloc(T, lda * n + 2);
    defer allocator.free(matrix_storage);
    matrix_storage[0] = guard;
    matrix_storage[matrix_storage.len - 1] = guard;
    const matrix = matrix_storage[1 .. matrix_storage.len - 1];
    for (matrix, 0..) |*value, index| value.* = complexValue(T, index + 1);

    const x = try allocator.alloc(T, m);
    defer allocator.free(x);
    const y = try allocator.alloc(T, n);
    defer allocator.free(y);
    for (x, 0..) |*value, index| value.* = complexValue(T, index * 3 + 2);
    for (y, 0..) |*value, index| value.* = complexValue(T, index * 7 + 4);

    const output_storage = try allocator.alloc(T, @max(lda * n, @max(m, n)) + 2);
    defer allocator.free(output_storage);
    const expected = try allocator.alloc(T, @max(lda * n, @max(m, n)));
    defer allocator.free(expected);

    const gemv_operations = [_]contract.Operation{ .gemv_no_trans_complex, .gemv_trans_complex };
    for (gemv_operations) |operation| {
        const transposed = operation == .gemv_trans_complex;
        const output_len: usize = if (transposed) n else m;
        const flag_count: usize = if (transposed) 2 else 1;
        for (0..flag_count) |flag_index| {
            const flag: u8 = @intCast(flag_index);
            output_storage[0] = guard;
            output_storage[output_len + 1] = guard;
            const output = output_storage[1 .. output_len + 1];
            for (output, expected[0..output_len], 0..) |*actual, *wanted, index| {
                actual.* = complexValue(T, index * 11 + 8);
                wanted.* = actual.*;
            }
            if (transposed) {
                for (0..n) |col| {
                    var sum: T = .{ .re = 0, .im = 0 };
                    for (0..m) |row| {
                        const a = matrix[row + col * lda];
                        sum = complexAdd(T, sum, complexMul(T, if (flag != 0) complexConj(T, a) else a, x[row]));
                    }
                    expected[col] = complexAdd(T, expected[col], complexMul(T, alpha, sum));
                }
            } else {
                for (0..m) |row| {
                    var sum: T = .{ .re = 0, .im = 0 };
                    for (0..n) |col| sum = complexAdd(T, sum, complexMul(T, matrix[row + col * lda], y[col]));
                    expected[row] = complexAdd(T, expected[row], complexMul(T, alpha, sum));
                }
            }
            var request: contract.Request = .{
                .operation = @intFromEnum(operation),
                .scalar = @intFromEnum(scalar),
                .flags = flag,
                .lda = lda,
                .m = m,
                .n = n,
                .alpha_re = rawReal(R, alpha.re),
                .alpha_im = rawReal(R, alpha.im),
                .beta_re = 0,
                .beta_im = 0,
                .input0 = @ptrCast(matrix.ptr),
                .input1 = @ptrCast(if (transposed) x.ptr else y.ptr),
                .output = @ptrCast(output.ptr),
            };
            try execute(&request);
            for (output, expected[0..output_len]) |actual, wanted| try expectComplexClose(T, actual, wanted);
            if (!sameComplex(T, output_storage[0], guard) or
                !sameComplex(T, output_storage[output_len + 1], guard)) return error.GuardChanged;
        }
    }

    for ([_]u8{ 0, 1 }) |flag| {
        output_storage[0] = guard;
        output_storage[lda * n + 1] = guard;
        const output = output_storage[1 .. lda * n + 1];
        for (output, expected[0 .. lda * n], 0..) |*actual, *wanted, index| {
            actual.* = complexValue(T, index * 17 + 9);
            wanted.* = actual.*;
        }
        for (0..n) |col| {
            const coefficient = complexMul(T, alpha, if (flag != 0) complexConj(T, y[col]) else y[col]);
            for (0..m) |row| {
                const index = row + col * lda;
                expected[index] = complexAdd(T, expected[index], complexMul(T, x[row], coefficient));
            }
        }
        var request: contract.Request = .{
            .operation = @intFromEnum(contract.Operation.ger_complex),
            .scalar = @intFromEnum(scalar),
            .flags = flag,
            .lda = lda,
            .m = m,
            .n = n,
            .alpha_re = rawReal(R, alpha.re),
            .alpha_im = rawReal(R, alpha.im),
            .beta_re = 0,
            .beta_im = 0,
            .input0 = @ptrCast(x.ptr),
            .input1 = @ptrCast(y.ptr),
            .output = @ptrCast(output.ptr),
        };
        try execute(&request);
        for (output, expected[0 .. lda * n]) |actual, wanted| try expectComplexClose(T, actual, wanted);
        if (!sameComplex(T, output_storage[0], guard) or
            !sameComplex(T, output_storage[lda * n + 1], guard)) return error.GuardChanged;
    }

    if (!sameComplex(T, matrix_storage[0], guard) or
        !sameComplex(T, matrix_storage[matrix_storage.len - 1], guard)) return error.GuardChanged;
}

fn verifyF64RejectedWithoutMutation() ProbeError!void {
    var input0 = [_]f64{ 1, 2, 3, 4 };
    var input1 = [_]f64{ 5, 6, 7, 8 };
    var output = [_]f64{ 91, 92, 93, 94 };
    const before = output;
    var request: contract.Request = .{
        .operation = @intFromEnum(contract.Operation.gemv_no_trans_full),
        .scalar = @intFromEnum(contract.Scalar.f64),
        .flags = 0,
        .lda = 4,
        .m = 4,
        .n = 1,
        .alpha_re = rawReal(f64, 1.25),
        .alpha_im = 0,
        .beta_re = rawReal(f64, -0.5),
        .beta_im = 0,
        .input0 = @ptrCast(&input0),
        .input1 = @ptrCast(&input1),
        .output = @ptrCast(&output),
    };
    if (contract.zynum_internal_x86_64_level2_width_execute(&request) != 0) return error.UnexpectedResult;
    if (!std.mem.eql(f64, &output, &before)) return error.RejectedRequestChangedOutput;
}

pub fn main() !void {
    if (builtin.is_test) return error.BuiltAsTest;
    if (builtin.cpu.arch != .x86_64 or !builtin.cpu.features.isEnabled(
        @intFromEnum(std.Target.x86.Feature.avx512f),
    )) return error.UnexpectedTarget;
    if (contract.zynum_internal_x86_64_level2_width_enabled != 1) return error.CandidateDisabled;

    try verifyMinimalRealWitnesses();
    try verifyRealGemv();
    try verifyRealGer();
    try verifyComplex(ComplexF32, .complex_f32);
    try verifyComplex(ComplexF64, .complex_f64);
    try verifyF64RejectedWithoutMutation();
}
