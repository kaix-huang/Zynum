// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! GETRI boundary control over the one numerical/resource owner. Public WORK
//! remains distinct from the intrinsic wide output workspace. High query,
//! public allocation and execution share one already admitted operation.
const common = @import("lapacke_common.zig");
const query_codec = @import("lapacke_gelsd_workspace.zig");
pub const Level = common.Level;
pub const ErrorEvent = common.ErrorEvent;
pub fn ReferenceGetriAdapter(comptime Core: type, comptime scan: bool) type {
    return AdapterEngine(Core, scan, .typed);
}
pub fn RawGetriAdapter(comptime Core: type, comptime scan: bool) type {
    return AdapterEngine(Core, scan, .raw);
}
fn AdapterEngine(comptime Core: type, comptime compile_nan_scan: bool, comptime kind: common.access.Kind) type {
    return struct {
        pub const Services = Core.Services;
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub const Dense = Core.Dense;
        pub const profile = "reference_3_12_1_frontend_own_getri_workspaces_v1";
        const H = common.HelpersWithAccess(Core, kind);
        const Fault = common.access.Fault(kind) || error{InvalidPivot};
        pub fn Arguments(comptime T: type, comptime I: type) type {
            H.scalar(T);
            H.integer(I);
            return struct {
                layout: c_int,
                n: I,
                a: common.access.Operand(T, true, kind),
                lda: I,
                pivots: common.access.Operand(I, false, kind),
                work: common.access.Operand(T, true, kind) = null,
                lwork: I = 0,
            };
        }
        pub fn Result(comptime I: type) type {
            H.integer(I);
            return union(enum) { lapacke_info: I, admission_fault: Fault };
        }
        fn hook(comptime T: type, services: Services, channel: common.HookKind, entry: common.Entry, argument: i32) void {
            H.hook(T, services, channel, entry, argument, .getri);
        }
        fn fortranArgument(comptime I: type, n: I, lda: I, lwork: I) i32 {
            if (n < 0) return 1;
            if (lda < @max(@as(I, 1), n)) return 3;
            if (lwork != -1 and lwork < @max(@as(I, 1), n)) return 6;
            return 0;
        }
        fn recommendation(comptime T: type, comptime I: type, n: I) common.access.Fault(kind)!T {
            const count = try H.native(@max(@as(I, 1), n));
            // Existing upward encoding codec, not a vendor block-size claim.
            const R = Core.Scalar.Real(T);
            const value = query_codec.queryLength(R, I, count) catch return error.SizeOverflow;
            return if (comptime Core.Scalar.isComplex(T)) .{ .re = value, .im = 0 } else value;
        }
        fn publicLength(comptime T: type, comptime I: type, value: T) common.access.Fault(kind)!usize {
            const integer: I = @intFromFloat(if (comptime Core.Scalar.isComplex(T)) value.re else value);
            return H.native(integer);
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
        pub fn callAdmitted(comptime T: type, comptime I: type, level: Level, operation: *const Core.Execution.context.Operation, borrowed: *const Services, args: Arguments(T, I)) error{CounterOverflow}!Result(I) {
            const services: Services = .{ .allocator = operation.allocator(), .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
            return callCaptured(T, I, level, .context_managed, services, operation, args);
        }
        fn callCaptured(comptime T: type, comptime I: type, level: Level, comptime lifetime: common.Lifetime, services: Services, source: common.Scratch(Core, T, lifetime).Source, args: Arguments(T, I)) common.EngineResult(lifetime, Result(I)) {
            if (args.layout != 101 and args.layout != 102) {
                hook(T, services, .lapacke, if (level == .high) .high else .work, -1);
                return .{ .lapacke_info = -1 };
            }
            if (comptime compile_nan_scan) {
                if (level == .high and services.nan_enabled(services.context)) {
                    const has_nan = H.scanGeneral(T, args.layout, args.n, args.n, args.lda, args.a) catch |err| return .{ .admission_fault = err };
                    if (has_nan) return .{ .lapacke_info = -3 };
                }
            }
            if (level == .work) return workCall(T, I, lifetime, services, source, args);
            var query: [1]T = undefined;
            var query_args = args;
            query_args.work = if (kind == .typed) &query else query[0..].ptr;
            query_args.lwork = -1;
            const queried = if (comptime lifetime == .context_managed) try workCall(T, I, lifetime, services, source, query_args) else workCall(T, I, lifetime, services, source, query_args);
            switch (queried) {
                .admission_fault => return queried,
                .lapacke_info => |info| if (info != 0) return queried,
            }
            const count = publicLength(T, I, query[0]) catch |err| return .{ .admission_fault = err };
            _ = Core.Storage.byteLength(count, @sizeOf(T)) catch |err| return .{ .admission_fault = err };
            const scratch = common.Scratch(Core, T, lifetime);
            var public_lease = scratch.acquire(source, count) catch |err| {
                if (comptime lifetime == .context_managed) switch (err) {
                    error.CounterOverflow => return error.CounterOverflow,
                    error.OutOfMemory => {},
                } else switch (err) {
                    error.OutOfMemory => {},
                }
                hook(T, services, .lapacke, .high, -1010);
                return .{ .lapacke_info = -1010 };
            };
            var public_live = true;
            defer if (public_live) public_lease.deinit();
            var execution_args = args;
            execution_args.work = if (kind == .typed) scratch.values(&public_lease) else scratch.values(&public_lease).ptr;
            execution_args.lwork = @intCast(count);
            const result = if (comptime lifetime == .context_managed) try workCall(T, I, lifetime, services, source, execution_args) else workCall(T, I, lifetime, services, source, execution_args);
            switch (result) {
                .lapacke_info => |info| if (info == -1010) {
                    public_lease.deinit();
                    public_live = false;
                    hook(T, services, .lapacke, .high, -1010);
                },
                .admission_fault => {},
            }
            return result;
        }
        fn publish(comptime T: type, row: bool, original: []T, column: Dense.Matrix(T), lda: usize) void {
            if (!row or column.row_count == 0) return;
            H.transferGeneral(T, original, column.values, column.row_count, column.column_count, lda, false);
        }
        fn workCall(comptime T: type, comptime I: type, comptime lifetime: common.Lifetime, services: Services, source: common.Scratch(Core, T, lifetime).Source, args: Arguments(T, I)) common.EngineResult(lifetime, Result(I)) {
            const scratch = common.Scratch(Core, T, lifetime);
            const W = Core.LuInverse.WorkValue(T);
            const wide_scratch = common.Scratch(Core, W, lifetime);
            const row = args.layout == 101;
            const query = args.lwork == -1;
            if (row and args.lda < args.n) {
                hook(T, services, .lapacke, .work, -4);
                return .{ .lapacke_info = -4 };
            }
            var transpose_lease: ?scratch.Lease = null;
            defer if (transpose_lease) |*lease| lease.deinit();
            var original: []T = &.{};
            var original_ld: usize = 0;
            var column: Dense.Matrix(T) = .{ .values = &.{}, .row_count = 0, .column_count = 0, .leading_dimension = 1 };
            const backend_lda = if (row) @max(@as(I, 1), args.n) else args.lda;
            if (row and !query) {
                const ld = H.native(backend_lda) catch |err| return .{ .admission_fault = err };
                const count = Core.Storage.columnMajorExtent(ld, ld, ld) catch |err| return .{ .admission_fault = err };
                _ = Core.Storage.byteLength(count, @sizeOf(T)) catch |err| return .{ .admission_fault = err };
                transpose_lease = scratch.acquire(source, count) catch |err| {
                    if (comptime lifetime == .context_managed) switch (err) {
                        error.CounterOverflow => return error.CounterOverflow,
                        error.OutOfMemory => {},
                    } else switch (err) {
                        error.OutOfMemory => {},
                    }
                    hook(T, services, .lapacke, .work, -1011);
                    return .{ .lapacke_info = -1011 };
                };
                const n = if (args.n > 0) ld else 0;
                if (n != 0) {
                    original_ld = H.native(args.lda) catch |err| return .{ .admission_fault = err };
                    const extent = Core.Storage.columnMajorExtent(n, n, original_ld) catch |err| return .{ .admission_fault = err };
                    original = H.capacity(T, args.a, extent) catch |err| return .{ .admission_fault = err };
                    H.copyGeneralToColumn(T, original, scratch.values(&transpose_lease.?), n, n, original_ld);
                }
                column = .{ .values = scratch.values(&transpose_lease.?), .row_count = n, .column_count = n, .leading_dimension = ld };
            }
            const encoded = recommendation(T, I, args.n) catch |err| return .{ .admission_fault = err };
            const work_one = H.capacity(T, args.work, 1) catch |err| return .{ .admission_fault = err };
            const invalid = fortranArgument(I, args.n, backend_lda, args.lwork);
            if (invalid != 0) {
                work_one[0] = encoded;
                hook(T, services, .fortran_returning_test, .fortran_getri, invalid);
                publish(T, row, original, column, original_ld);
                return .{ .lapacke_info = -@as(I, invalid) - 1 };
            }
            if (query or args.n == 0) {
                work_one[0] = encoded;
                return .{ .lapacke_info = 0 };
            }
            const n = H.native(args.n) catch |err| return .{ .admission_fault = err };
            const public_work = H.capacity(T, args.work, n) catch |err| return .{ .admission_fault = err };
            if (!row) {
                original_ld = H.native(args.lda) catch |err| return .{ .admission_fault = err };
                const extent = Core.Storage.columnMajorExtent(n, n, original_ld) catch |err| return .{ .admission_fault = err };
                original = H.capacity(T, args.a, extent) catch |err| return .{ .admission_fault = err };
                column = .{ .values = original, .row_count = n, .column_count = n, .leading_dimension = original_ld };
            }
            H.disjoint(T, original, public_work) catch |err| return .{ .admission_fault = err };
            const info = Core.Triangular.firstZeroDiagonal(T, column.asConst(), .non_unit);
            if (info != 0) {
                work_one[0] = encoded;
                publish(T, row, original, column, original_ld);
                return .{ .lapacke_info = @intCast(info) };
            }
            const pivots = H.constCapacity(I, args.pivots, n - 1) catch |err| return .{ .admission_fault = err };
            H.disjointMixed(T, original, I, pivots) catch |err| return .{ .admission_fault = err };
            H.disjointMixed(T, public_work, I, pivots) catch |err| return .{ .admission_fault = err };
            for (pivots, 0..) |value, step| {
                if (!Core.Pivot.validStep(I, .one, value, step, n)) return .{ .admission_fault = error.InvalidPivot };
            }
            work_one[0] = encoded;
            const count = Core.LuInverse.workspaceLength(n) catch return .{ .admission_fault = error.SizeOverflow };
            _ = Core.Storage.byteLength(count, @sizeOf(W)) catch |err| return .{ .admission_fault = err };
            var wide_lease: ?wide_scratch.Lease = null;
            defer if (wide_lease) |*lease| lease.deinit();
            var workspace: []W = &.{};
            if (count != 0) {
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
            Core.LuInverse.invertNonsingular(T, column, Core.Pivot.Reader(I, .one){ .values = pivots }, workspace) catch unreachable;
            publish(T, row, original, column, original_ld);
            return .{ .lapacke_info = 0 };
        }
    };
}
