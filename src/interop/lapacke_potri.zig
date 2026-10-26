// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Reference LAPACKE argument/NaN flow over the owner's Cholesky inverse.
//! Its widened numerical workspace borrows the existing scratch lifecycle.
//! Transpose OOM is -1011; intrinsic wide-workspace OOM is -1010. Returning
//! diagnostics are a private test contract, not a Fortran STOP/public ABI.
const common = @import("lapacke_common.zig");
pub const Level = common.Level;
pub const ErrorEvent = common.ErrorEvent;

pub fn ReferencePotriAdapter(comptime Core: type, comptime compile_nan_scan: bool) type {
    return AdapterEngine(Core, compile_nan_scan, .typed);
}
pub fn RawPotriAdapter(comptime Core: type, comptime compile_nan_scan: bool) type {
    return AdapterEngine(Core, compile_nan_scan, .raw);
}
fn AdapterEngine(comptime Core: type, comptime compile_nan_scan: bool, comptime kind: common.access.Kind) type {
    return struct {
        pub const Services = Core.Services;
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub const Dense = Core.Dense;
        pub const profile = "reference_3_12_1_frontend_own_wide_workspace_v1";
        const helpers = common.HelpersWithAccess(Core, kind);
        pub fn Arguments(comptime T: type, comptime I: type) type {
            helpers.scalar(T);
            helpers.integer(I);
            return struct { layout: c_int, uplo: u8, n: I, a: common.access.Operand(T, true, kind), lda: I };
        }
        pub fn Result(comptime I: type) type {
            helpers.integer(I);
            return union(enum) { lapacke_info: I, admission_fault: common.access.Fault(kind) };
        }
        fn hook(comptime T: type, services: Services, channel: common.HookKind, entry: common.Entry, argument: i32) void {
            helpers.hook(T, services, channel, entry, argument, .potri);
        }
        fn fortranArgument(uplo: u8, n: anytype, lda: @TypeOf(n)) i32 {
            if (helpers.triangle(uplo) == null) return 1;
            if (n < 0) return 2;
            if (lda < @max(@as(@TypeOf(n), 1), n)) return 4;
            return 0;
        }
        pub fn call(comptime T: type, comptime I: type, level: Level, borrowed: *const Services, args: Arguments(T, I)) Result(I) {
            const services = borrowed.*;
            return callCaptured(T, I, level, .unmanaged, services, services.allocator, args);
        }
        pub fn callManaged(comptime T: type, comptime I: type, level: Level, context: *Core.Execution.context.Context, borrowed: *const Services, args: Arguments(T, I)) Core.Execution.context.AdmissionError!Result(I) {
            var invocation = try common.ManagedInvocation(Core).begin(context, borrowed);
            defer invocation.operation.end();
            return callCaptured(T, I, level, .context_managed, invocation.services, &invocation.operation, args);
        }
        pub fn callAdmitted(comptime T: type, comptime I: type, level: Level, admitted: *const Core.Execution.context.Operation, borrowed: *const Services, args: Arguments(T, I)) error{CounterOverflow}!Result(I) {
            const services: Services = .{ .allocator = admitted.allocator(), .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
            return callCaptured(T, I, level, .context_managed, services, admitted, args);
        }
        fn callCaptured(comptime T: type, comptime I: type, level: Level, comptime lifetime: common.Lifetime, services: Services, source: common.Scratch(Core, T, lifetime).Source, args: Arguments(T, I)) common.EngineResult(lifetime, Result(I)) {
            const transpose_scratch = common.Scratch(Core, T, lifetime);
            const W = Core.CholeskyInverse.WorkValue(T);
            const wide_scratch = common.Scratch(Core, W, lifetime);
            if (args.layout != 101 and args.layout != 102) {
                hook(T, services, .lapacke, if (level == .high) .high else .work, -1);
                return .{ .lapacke_info = -1 };
            }
            if (comptime compile_nan_scan) {
                if (level == .high and services.nan_enabled(services.context)) {
                    const has_nan = helpers.scanTriangular(T, I, .{ .layout = args.layout, .uplo = args.uplo, .diag = 'N', .n = args.n, .lda = args.lda, .values = args.a }) catch |err| return .{ .admission_fault = err };
                    if (has_nan) return .{ .lapacke_info = -4 };
                }
            }
            const row_major = args.layout == 101;
            var transpose_lease: ?transpose_scratch.Lease = null;
            defer if (transpose_lease) |*lease| lease.deinit();
            var original: []T = &.{};
            var column: Dense.Matrix(T) = undefined;
            var n: usize = 0;
            var original_ld: usize = 0;
            if (row_major) {
                if (args.lda < args.n) {
                    hook(T, services, .lapacke, .work, -5);
                    return .{ .lapacke_info = -5 };
                }
                const temporary_ld = helpers.native(@max(@as(I, 1), args.n)) catch |err| return .{ .admission_fault = err };
                const count = Core.Storage.columnMajorExtent(temporary_ld, temporary_ld, temporary_ld) catch |err| return .{ .admission_fault = err };
                _ = Core.Storage.byteLength(count, @sizeOf(T)) catch |err| return .{ .admission_fault = err };
                transpose_lease = transpose_scratch.acquire(source, count) catch |err| {
                    if (comptime lifetime == .context_managed) switch (err) {
                        error.CounterOverflow => return error.CounterOverflow,
                        error.OutOfMemory => {},
                    } else switch (err) {
                        error.OutOfMemory => {},
                    }
                    hook(T, services, .lapacke, .work, -1011);
                    return .{ .lapacke_info = -1011 };
                };
                n = if (args.n > 0) temporary_ld else 0;
                const upper = helpers.triangle(args.uplo);
                if (n != 0 and upper != null) {
                    original_ld = helpers.native(args.lda) catch |err| return .{ .admission_fault = err };
                    const extent = helpers.triangularReadExtent(n, original_ld, 101, upper.?, false) catch |err| return .{ .admission_fault = err };
                    original = helpers.capacity(T, args.a, extent) catch |err| return .{ .admission_fault = err };
                    helpers.copyTriangularToColumn(T, original, transpose_scratch.values(&transpose_lease.?), n, original_ld, upper.?, false);
                }
                const argument = fortranArgument(args.uplo, args.n, @as(I, @intCast(temporary_ld)));
                if (argument != 0) {
                    hook(T, services, .fortran_returning_test, .fortran_potri, argument);
                    return .{ .lapacke_info = -@as(I, argument) - 1 };
                }
                column = .{ .values = transpose_scratch.values(&transpose_lease.?), .row_count = n, .column_count = n, .leading_dimension = temporary_ld };
            } else {
                const argument = fortranArgument(args.uplo, args.n, args.lda);
                if (argument != 0) {
                    hook(T, services, .fortran_returning_test, .fortran_potri, argument);
                    return .{ .lapacke_info = -@as(I, argument) - 1 };
                }
                n = helpers.native(args.n) catch |err| return .{ .admission_fault = err };
                if (n == 0) return .{ .lapacke_info = 0 };
                original_ld = helpers.native(args.lda) catch |err| return .{ .admission_fault = err };
                const extent = helpers.triangularReadExtent(n, original_ld, 102, helpers.triangle(args.uplo).?, false) catch |err| return .{ .admission_fault = err };
                original = helpers.capacity(T, args.a, extent) catch |err| return .{ .admission_fault = err };
                column = .{ .values = original, .row_count = n, .column_count = n, .leading_dimension = original_ld };
            }
            const triangle: Core.Triangular.Triangle = if (helpers.triangle(args.uplo).?) .upper else .lower;
            const info = Core.Triangular.firstZeroDiagonal(T, column.asConst(), .non_unit);
            var wide_lease: ?wide_scratch.Lease = null;
            defer if (wide_lease) |*lease| lease.deinit();
            if (info == 0) {
                const count = Core.CholeskyInverse.workspaceLength(n) catch return .{ .admission_fault = error.SizeOverflow };
                var workspace: []W = &.{};
                if (count != 0) {
                    _ = Core.Storage.byteLength(count, @sizeOf(W)) catch |err| return .{ .admission_fault = err };
                    wide_lease = wide_scratch.acquire(source, count) catch |err| {
                        if (comptime lifetime == .context_managed) switch (err) {
                            error.CounterOverflow => return error.CounterOverflow,
                            error.OutOfMemory => {},
                        } else switch (err) {
                            error.OutOfMemory => {},
                        }
                        if (transpose_lease) |*lease| lease.deinit();
                        transpose_lease = null;
                        hook(T, services, .lapacke, .work, -1010);
                        return .{ .lapacke_info = -1010 };
                    };
                    workspace = wide_scratch.values(&wide_lease.?);
                }
                Core.CholeskyInverse.invertNonsingular(T, column, triangle, workspace) catch unreachable;
            }
            // Publish before either resource's free callback. The opposite
            // triangle and padding bytes are never copied or overwritten.
            if (row_major and n != 0) {
                for (0..n) |row| for (0..n) |col| {
                    if (helpers.selected(row, col, triangle == .upper)) original[row * original_ld + col] = column.values[row + col * column.leading_dimension];
                };
            }
            return .{ .lapacke_info = @intCast(info) };
        }
    };
}
