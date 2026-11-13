// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// One C wire-to-admitted-call adapter; no mathematical loops or factor owner.
const execution = @import("zynum-execution");
const lapack_checked = @import("zynum-core").internal.lapack_checked;
const status = @import("zynum-cholesky-c-status");

pub fn Adapter(comptime WireBridge: type) type {
    return struct {
        const abi = WireBridge.Wire;
        const decode = @import("decode.zig").Decoder(WireBridge);

        fn result(value: anyerror!usize) abi.result {
            const info = value catch |err| return .{
                .status = if (err == error.InvalidHermitianDiagonal) status.INVALID_HERMITIAN_DIAGONAL else WireBridge.statusCode(err),
                .reserved = 0,
                .info = 0,
            };
            return .{ .status = if (info == 0) abi.OK else abi.NUMERICAL_FAILURE, .reserved = 0, .info = @intCast(info) };
        }
        fn factorInner(comptime T: type, operation: *const execution.context.Operation, a: abi.dense, flag: i32) !usize {
            const uplo = try decode.triangle(flag);
            return lapack_checked.choleskyFactorizeAdmitted(operation, T, try decode.matrix(T, a), uplo);
        }

        pub fn factor(comptime T: type, a: abi.dense, flag: i32) abi.result {
            var operation = execution.beginDefault();
            defer operation.end();
            return result(factorInner(T, &operation, a, flag));
        }

        fn solveInner(comptime T: type, operation: *const execution.context.Operation, a: abi.const_dense, b: abi.dense, flag: i32) !usize {
            const uplo = try decode.triangle(flag);
            // Both descriptor ranges are checked before public operations may read memory.
            const factor_matrix = try decode.constMatrix(T, a);
            const rhs = try decode.matrix(T, b);
            return lapack_checked.choleskySolveAdmitted(operation, T, factor_matrix, rhs, uplo);
        }

        pub fn solve(comptime T: type, a: abi.const_dense, b: abi.dense, flag: i32) abi.result {
            var operation = execution.beginDefault();
            defer operation.end();
            return result(solveInner(T, &operation, a, b, flag));
        }
    };
}
