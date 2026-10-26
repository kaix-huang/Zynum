// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Private Reference LAPACKE 3.12.1 TRTRS control-flow adapter. Returning
//! diagnostic handlers are a test profile, not Fortran STOP or a public ABI.
//! Existing Core owns all arithmetic, admission, allocator and scratch state.
const common = @import("lapacke_common.zig");
pub const Level = common.Level;
pub const ErrorEvent = common.ErrorEvent;
pub const AdmissionFault = common.AdmissionFault;

pub fn ReferenceTrtrsAdapter(comptime Core: type, comptime compile_nan_scan: bool) type {
    return AdapterEngine(Core, compile_nan_scan, .typed);
}
pub fn RawTrtrsAdapter(comptime Core: type, comptime compile_nan_scan: bool) type {
    return AdapterEngine(Core, compile_nan_scan, .raw);
}
fn AdapterEngine(comptime Core: type, comptime compile_nan_scan: bool, comptime kind: common.access.Kind) type {
    return struct {
        pub const Services = Core.Services;
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub const Dense = Core.Dense;
        pub const ScratchLease = Core.ScratchLease;
        pub const profile = "reference_3_12_1_control_flow_returning_handler_test";
        const helpers = common.HelpersWithAccess(Core, kind);
        pub fn Arguments(comptime T: type, comptime I: type) type {
            helpers.scalar(T);
            helpers.integer(I);
            return struct { layout: c_int, uplo: u8, trans: u8, diag: u8, n: I, nrhs: I, a: common.access.Operand(T, false, kind), lda: I, b: common.access.Operand(T, true, kind), ldb: I };
        }
        pub fn Result(comptime I: type) type {
            helpers.integer(I);
            return union(enum) { lapacke_info: I, admission_fault: common.access.Fault(kind) };
        }
        fn hook(comptime T: type, services: Services, channel: common.HookKind, entry: common.Entry, argument: i32) void {
            helpers.hook(T, services, channel, entry, argument, .trtrs);
        }
        fn operation(trans: u8) ?Core.Triangular.Operation {
            return switch (trans) {
                'N', 'n' => .normal,
                'T', 't' => .transposed,
                'C', 'c' => .adjoint,
                else => null,
            };
        }
        fn fortranArgument(comptime I: type, args: ArgumentsForValidation(I)) i32 {
            if (helpers.triangle(args.uplo) == null) return 1;
            if (operation(args.trans) == null) return 2;
            if (helpers.unitDiagonal(args.diag) == null) return 3;
            if (args.n < 0) return 4;
            if (args.nrhs < 0) return 5;
            if (args.lda < @max(@as(I, 1), args.n)) return 7;
            if (args.ldb < @max(@as(I, 1), args.n)) return 9;
            return 0;
        }
        fn ArgumentsForValidation(comptime I: type) type {
            return struct { uplo: u8, trans: u8, diag: u8, n: I, nrhs: I, lda: I, ldb: I };
        }
        fn solve(comptime T: type, comptime I: type, a: Dense.ConstMatrix(T), b: Dense.Matrix(T), upper: bool, trans: u8, unit: bool) I {
            return @intCast(Core.Triangular.solveNonsingular(T, a, b, if (upper) .upper else .lower, operation(trans).?, if (unit) .unit else .non_unit));
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
            const scratch = common.Scratch(Core, T, lifetime);
            if (args.layout != 101 and args.layout != 102) {
                hook(T, services, .lapacke, if (level == .high) .high else .work, -1);
                return .{ .lapacke_info = -1 };
            }
            if (comptime compile_nan_scan) {
                if (level == .high and services.nan_enabled(services.context)) {
                    const a_nan = helpers.scanTriangular(T, I, .{ .layout = args.layout, .uplo = args.uplo, .diag = args.diag, .n = args.n, .lda = args.lda, .values = args.a }) catch |err| return .{ .admission_fault = err };
                    if (a_nan) return .{ .lapacke_info = -7 };
                    const b_nan = helpers.scanGeneral(T, args.layout, args.n, args.nrhs, args.ldb, args.b) catch |err| return .{ .admission_fault = err };
                    if (b_nan) return .{ .lapacke_info = -9 };
                }
            }
            if (args.layout == 102) {
                const argument = fortranArgument(I, .{ .uplo = args.uplo, .trans = args.trans, .diag = args.diag, .n = args.n, .nrhs = args.nrhs, .lda = args.lda, .ldb = args.ldb });
                if (argument != 0) {
                    hook(T, services, .fortran_returning_test, .fortran_trtrs, argument);
                    return .{ .lapacke_info = -@as(I, argument) - 1 };
                }
                const unit = helpers.unitDiagonal(args.diag).?;
                // Only N=0 is an unconditional Fortran quick return. Unit A
                // with no RHS also has no numerical operand accesses.
                if (args.n == 0 or (unit and args.nrhs == 0)) return .{ .lapacke_info = 0 };
                const n = helpers.native(args.n) catch |err| return .{ .admission_fault = err };
                const nrhs = helpers.native(args.nrhs) catch |err| return .{ .admission_fault = err };
                const lda = helpers.native(args.lda) catch |err| return .{ .admission_fault = err };
                const ldb = helpers.native(args.ldb) catch |err| return .{ .admission_fault = err };
                const upper = helpers.triangle(args.uplo).?;
                const ae = helpers.triangularReadExtent(n, lda, 102, upper, unit) catch |err| return .{ .admission_fault = err };
                const be = Core.Storage.columnMajorExtent(n, nrhs, ldb) catch |err| return .{ .admission_fault = err };
                const a = helpers.constCapacity(T, args.a, ae) catch |err| return .{ .admission_fault = err };
                const b = helpers.capacity(T, args.b, be) catch |err| return .{ .admission_fault = err };
                helpers.disjoint(T, a, b) catch |err| return .{ .admission_fault = err };
                return .{ .lapacke_info = solve(T, I, .{ .values = a, .row_count = n, .column_count = n, .leading_dimension = lda }, .{ .values = b, .row_count = n, .column_count = nrhs, .leading_dimension = ldb }, upper, args.trans, unit) };
            }
            if (args.lda < args.n) {
                hook(T, services, .lapacke, .work, -8);
                return .{ .lapacke_info = -8 };
            }
            if (args.ldb < args.nrhs) {
                hook(T, services, .lapacke, .work, -10);
                return .{ .lapacke_info = -10 };
            }
            const temp_n = helpers.native(@max(@as(I, 1), args.n)) catch |err| return .{ .admission_fault = err };
            const acount = Core.Storage.columnMajorExtent(temp_n, temp_n, temp_n) catch |err| return .{ .admission_fault = err };
            _ = Core.Storage.byteLength(acount, @sizeOf(T)) catch |err| return .{ .admission_fault = err };
            var a_lease = scratch.acquire(source, acount) catch |err| {
                if (comptime lifetime == .context_managed) switch (err) {
                    error.CounterOverflow => return error.CounterOverflow,
                    error.OutOfMemory => {},
                } else switch (err) {
                    error.OutOfMemory => {},
                }
                hook(T, services, .lapacke, .work, -1011);
                return .{ .lapacke_info = -1011 };
            };
            var a_live = true;
            defer if (a_live) a_lease.deinit();
            const temp_rhs = helpers.native(@max(@as(I, 1), args.nrhs)) catch |err| return .{ .admission_fault = err };
            const bcount = Core.Storage.columnMajorExtent(temp_n, temp_rhs, temp_n) catch |err| return .{ .admission_fault = err };
            _ = Core.Storage.byteLength(bcount, @sizeOf(T)) catch |err| return .{ .admission_fault = err };
            var b_lease = scratch.acquire(source, bcount) catch |err| {
                if (comptime lifetime == .context_managed) switch (err) {
                    error.CounterOverflow => return error.CounterOverflow,
                    error.OutOfMemory => {},
                } else switch (err) {
                    error.OutOfMemory => {},
                }
                a_lease.deinit();
                a_live = false;
                hook(T, services, .lapacke, .work, -1011);
                return .{ .lapacke_info = -1011 };
            };
            defer b_lease.deinit();
            const n = if (args.n > 0) temp_n else 0;
            const nrhs = if (args.nrhs > 0) temp_rhs else 0;
            var a: []const T = &.{};
            var b: []T = &.{};
            var ldb: usize = 0;
            const upper = helpers.triangle(args.uplo);
            const unit = helpers.unitDiagonal(args.diag);
            if (n != 0 and upper != null and unit != null) {
                const lda = helpers.native(args.lda) catch |err| return .{ .admission_fault = err };
                const ae = helpers.triangularReadExtent(n, lda, 101, upper.?, unit.?) catch |err| return .{ .admission_fault = err };
                a = helpers.constCapacity(T, args.a, ae) catch |err| return .{ .admission_fault = err };
                helpers.copyTriangularToColumn(T, a, scratch.values(&a_lease), n, lda, upper.?, unit.?);
            }
            if (n != 0 and nrhs != 0) {
                ldb = helpers.native(args.ldb) catch |err| return .{ .admission_fault = err };
                const be = Core.Storage.columnMajorExtent(nrhs, n, ldb) catch |err| return .{ .admission_fault = err };
                b = helpers.capacity(T, args.b, be) catch |err| return .{ .admission_fault = err };
                helpers.transferGeneral(T, b, scratch.values(&b_lease), n, nrhs, ldb, true);
            }
            const argument = fortranArgument(I, .{ .uplo = args.uplo, .trans = args.trans, .diag = args.diag, .n = args.n, .nrhs = args.nrhs, .lda = @intCast(temp_n), .ldb = @intCast(temp_n) });
            var info: I = 0;
            if (argument != 0) {
                hook(T, services, .fortran_returning_test, .fortran_trtrs, argument);
                info = -@as(I, argument) - 1;
            } else if (n != 0) {
                helpers.disjoint(T, a, b) catch |err| return .{ .admission_fault = err };
                info = solve(T, I, .{ .values = scratch.values(&a_lease), .row_count = n, .column_count = n, .leading_dimension = temp_n }, .{ .values = scratch.values(&b_lease), .row_count = n, .column_count = nrhs, .leading_dimension = temp_n }, upper.?, args.trans, unit.?);
            }
            if (n != 0 and nrhs != 0) helpers.transferGeneral(T, b, scratch.values(&b_lease), n, nrhs, ldb, false);
            return .{ .lapacke_info = info };
        }
    };
}
