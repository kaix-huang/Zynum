// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Private typed Reference LAPACKE 3.12.1 POTRS state machine. Error hooks
//! deliberately return in this test profile; this is not Reference STOP or ABI.
//! Core owns all numerical and resource identities. No singleton or Context is
//! created here. Managed admission uses only the injected execution owner.
const common = @import("lapacke_common.zig");
pub const Level = common.Level;
pub const ErrorEvent = common.ErrorEvent;
pub const AdmissionFault = common.AdmissionFault;

/// Services is borrowed until return and captured before callbacks. call is
/// unmanaged; callManaged encloses checks, callbacks, copies and every free.
/// Typed input slices must be truthful/stable; rejected admission is not INFO.
pub fn ReferencePotrsAdapter(comptime Core: type, comptime compile_nan_scan: bool) type {
    return AdapterEngine(Core, compile_nan_scan, .typed);
}

/// Internal raw-operand specialization, not a C entry point. Caller-owned raw
/// storage must satisfy every stage-local access; raw faults are never INFO.
pub fn RawPotrsAdapter(comptime Core: type, comptime compile_nan_scan: bool) type {
    return AdapterEngine(Core, compile_nan_scan, .raw);
}

fn AdapterEngine(comptime Core: type, comptime compile_nan_scan: bool, comptime access_kind: common.access.Kind) type {
    return struct {
        pub const Services = Core.Services;
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub const Dense = Core.Dense;
        pub const ScratchLease = Core.ScratchLease;
        pub const profile = "reference_3_12_1_control_flow_returning_handler_test";
        const helpers = common.HelpersWithAccess(Core, access_kind);
        pub fn Arguments(comptime T: type, comptime I: type) type {
            helpers.integer(I);
            helpers.scalar(T);
            return struct { layout: c_int, uplo: u8, n: I, nrhs: I, a: common.access.Operand(T, false, access_kind), lda: I, b: common.access.Operand(T, true, access_kind), ldb: I };
        }
        pub fn Result(comptime I: type) type {
            helpers.integer(I);
            return union(enum) { lapacke_info: I, admission_fault: common.access.Fault(access_kind) };
        }
        fn hook(comptime T: type, services: Services, kind: common.HookKind, entry: common.Entry, argument: i32) void {
            helpers.hook(T, services, kind, entry, argument, .potrs);
        }
        fn fortranArgument(comptime I: type, uplo: u8, n: I, nrhs: I, lda: I, ldb: I) i32 {
            if (helpers.triangle(uplo) == null) return 1;
            if (n < 0) return 2;
            if (nrhs < 0) return 3;
            if (lda < @max(@as(I, 1), n)) return 5;
            if (ldb < @max(@as(I, 1), n)) return 7;
            return 0;
        }
        pub fn call(comptime T: type, comptime I: type, level: Level, borrowed: *const Services, args: Arguments(T, I)) Result(I) {
            const services = borrowed.*;
            return callCaptured(T, I, level, .unmanaged, services, services.allocator, args);
        }
        /// Admit before reading Services or user arrays. Only the Context's
        /// allocator is used; borrowed.allocator is ignored, never inspected.
        /// The synchronous operation spans callbacks, copyback and every free.
        pub fn callManaged(comptime T: type, comptime I: type, level: Level, context: *Core.Execution.context.Context, borrowed: *const Services, args: Arguments(T, I)) Core.Execution.context.AdmissionError!Result(I) {
            var invocation = try common.ManagedInvocation(Core).begin(context, borrowed);
            defer invocation.operation.end();
            return callCaptured(T, I, level, .context_managed, invocation.services, &invocation.operation, args);
        }
        /// Internal seam for an already admitted caller. The operation remains
        /// live through return; do not copy/end it concurrently. Rebuild allocator
        /// authority from it, ignoring borrowed.allocator even if undefined.
        /// Callback storage/code is borrowed synchronously through engine cleanup.
        pub fn callAdmitted(comptime T: type, comptime I: type, level: Level, operation: *const Core.Execution.context.Operation, borrowed: *const Services, args: Arguments(T, I)) error{CounterOverflow}!Result(I) {
            const services: Services = .{ .allocator = operation.allocator(), .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
            return callCaptured(T, I, level, .context_managed, services, operation, args);
        }
        fn callCaptured(comptime T: type, comptime I: type, level: Level, comptime lifetime: common.Lifetime, services: Services, source: common.Scratch(Core, T, lifetime).Source, args: Arguments(T, I)) common.EngineResult(lifetime, Result(I)) {
            const scratch = common.Scratch(Core, T, lifetime);
            if (args.layout != 101 and args.layout != 102) {
                hook(T, services, .lapacke, if (level == .high) .high else .work, -1);
                return .{ .lapacke_info = -1 };
            }
            if (comptime compile_nan_scan) {
                if (level == .high and services.nan_enabled(services.context)) {
                    const a_nan = helpers.scanTriangle(T, I, .{ .layout = args.layout, .uplo = args.uplo, .n = args.n, .lda = args.lda, .values = args.a }) catch |err| return .{ .admission_fault = err };
                    if (a_nan) return .{ .lapacke_info = -5 };
                    const b_nan = helpers.scanGeneral(T, args.layout, args.n, args.nrhs, args.ldb, args.b) catch |err| return .{ .admission_fault = err };
                    if (b_nan) return .{ .lapacke_info = -7 };
                }
            }
            if (args.layout == 102) {
                const argument = fortranArgument(I, args.uplo, args.n, args.nrhs, args.lda, args.ldb);
                if (argument != 0) {
                    hook(T, services, .fortran_returning_test, .fortran_potrs, argument);
                    return .{ .lapacke_info = -@as(I, argument) - 1 };
                }
                if (args.n == 0 or args.nrhs == 0) return .{ .lapacke_info = 0 };
                const n = helpers.native(args.n) catch |err| return .{ .admission_fault = err };
                const nrhs = helpers.native(args.nrhs) catch |err| return .{ .admission_fault = err };
                const lda = helpers.native(args.lda) catch |err| return .{ .admission_fault = err };
                const ldb = helpers.native(args.ldb) catch |err| return .{ .admission_fault = err };
                const ae = Core.Storage.columnMajorExtent(n, n, lda) catch |err| return .{ .admission_fault = err };
                const be = Core.Storage.columnMajorExtent(n, nrhs, ldb) catch |err| return .{ .admission_fault = err };
                const a = helpers.constCapacity(T, args.a, ae) catch |err| return .{ .admission_fault = err };
                const b = helpers.capacity(T, args.b, be) catch |err| return .{ .admission_fault = err };
                helpers.disjoint(T, a, b) catch |err| return .{ .admission_fault = err };
                Core.solvePotrs(T, .{ .values = a, .row_count = n, .column_count = n, .leading_dimension = lda }, .{ .values = b, .row_count = n, .column_count = nrhs, .leading_dimension = ldb }, helpers.triangle(args.uplo).?);
                return .{ .lapacke_info = 0 };
            }
            if (args.lda < args.n) {
                hook(T, services, .lapacke, .work, -6);
                return .{ .lapacke_info = -6 };
            }
            if (args.ldb < args.nrhs) {
                hook(T, services, .lapacke, .work, -8);
                return .{ .lapacke_info = -8 };
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
            // Reference makes this second acquisition after the first. Safe
            // extent admission follows that stage, and is never a vendor INFO.
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
            const upper = helpers.triangle(args.uplo);
            var a: []const T = &.{};
            var b: []T = &.{};
            var ldb: usize = 0;
            if (n != 0 and upper != null) {
                const lda = helpers.native(args.lda) catch |err| return .{ .admission_fault = err };
                const ae = Core.Storage.columnMajorExtent(n, n, lda) catch |err| return .{ .admission_fault = err };
                a = helpers.constCapacity(T, args.a, ae) catch |err| return .{ .admission_fault = err };
                helpers.transfer(T, a, scratch.values(&a_lease), n, lda, upper.?, true);
            }
            if (n != 0 and nrhs != 0) {
                ldb = helpers.native(args.ldb) catch |err| return .{ .admission_fault = err };
                const be = Core.Storage.columnMajorExtent(nrhs, n, ldb) catch |err| return .{ .admission_fault = err };
                b = helpers.capacity(T, args.b, be) catch |err| return .{ .admission_fault = err };
                helpers.transferGeneral(T, b, scratch.values(&b_lease), n, nrhs, ldb, true);
            }
            const argument = fortranArgument(I, args.uplo, args.n, args.nrhs, @intCast(temp_n), @intCast(temp_n));
            var info: I = 0;
            if (argument != 0) {
                hook(T, services, .fortran_returning_test, .fortran_potrs, argument);
                info = -@as(I, argument) - 1;
            } else if (n != 0 and nrhs != 0) {
                helpers.disjoint(T, a, b) catch |err| return .{ .admission_fault = err };
                Core.solvePotrs(T, .{ .values = scratch.values(&a_lease), .row_count = n, .column_count = n, .leading_dimension = temp_n }, .{ .values = scratch.values(&b_lease), .row_count = n, .column_count = nrhs, .leading_dimension = temp_n }, upper.?);
            }
            if (n != 0 and nrhs != 0) helpers.transferGeneral(T, b, scratch.values(&b_lease), n, nrhs, ldb, false);
            return .{ .lapacke_info = info };
        }
    };
}
