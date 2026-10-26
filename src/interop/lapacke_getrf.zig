// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal typed/raw GETRF with Reference 3.12.1 LAPACKE stage order and the
//! existing explicitly returning error-event test profile. Not a C ABI export.
//! Numerical factorization and one-based pivot storage are borrowed from the
//! same Core; Services, scratch and operation admission retain their one owner.
const common = @import("lapacke_common.zig");
pub const Level = common.Level;

pub fn ReferenceGetrfAdapter(comptime Core: type, comptime compile_nan_scan: bool) type {
    return AdapterEngine(Core, compile_nan_scan, .typed);
}
pub fn RawGetrfAdapter(comptime Core: type, comptime compile_nan_scan: bool) type {
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
        const native = helpers.native;
        const capacity = helpers.capacity;
        const transfer = helpers.transferGeneral;
        const Fault = common.access.Fault(access_kind);

        pub fn Result(comptime I: type) type {
            helpers.integer(I);
            return union(enum) { lapacke_info: I, admission_fault: Fault };
        }
        pub fn Arguments(comptime T: type, comptime I: type) type {
            helpers.integer(I);
            helpers.scalar(T);
            return struct {
                layout: c_int,
                m: I,
                n: I,
                lda: I,
                values: common.access.Operand(T, true, access_kind),
                pivots: common.access.Operand(I, true, access_kind),
            };
        }
        fn hook(comptime T: type, services: Services, kind: common.HookKind, entry: common.Entry, argument: i32) void {
            helpers.hook(T, services, kind, entry, argument, .getrf);
        }
        fn fortranArgument(comptime I: type, m: I, n: I, ld: I) i32 {
            if (m < 0) return 1;
            if (n < 0) return 2;
            if (ld < @max(@as(I, 1), m)) return 4;
            return 0;
        }
        // A factorization writes pivots; previous contents are never read or
        // validated. The output prefix must be disjoint from caller A, including
        // row-major A whose numerical work happens in a separate scratch slice.
        fn pivotOutput(comptime T: type, comptime I: type, args: Arguments(T, I), original: []const T, rows: usize, columns: usize) Fault![]I {
            const steps = @min(rows, columns);
            const pivots = try capacity(I, args.pivots, steps);
            if (!Core.Pivot.canWriteRows(I, .one, rows, steps)) return error.IntegerNotRepresentable;
            try helpers.disjointMixed(T, original, I, pivots);
            return pivots;
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
        /// No second admission: the existing operation spans callbacks, matrix
        /// copyback and allocator free, then the caller ends it. Services callback
        /// storage/code and caller operands remain valid synchronously throughout.
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
                    const found = helpers.scanGeneral(T, args.layout, args.m, args.n, args.lda, args.values) catch |err| return .{ .admission_fault = err };
                    if (found) return .{ .lapacke_info = -4 };
                }
            }
            if (args.layout == 102) {
                const argument = fortranArgument(I, args.m, args.n, args.lda);
                if (argument != 0) {
                    hook(T, services, .fortran_returning_test, .fortran_getrf, argument);
                    return .{ .lapacke_info = -@as(I, argument) - 1 };
                }
                if (args.m == 0 or args.n == 0) return .{ .lapacke_info = 0 };
                const m = native(args.m) catch |err| return .{ .admission_fault = err };
                const n = native(args.n) catch |err| return .{ .admission_fault = err };
                const ld = native(args.lda) catch |err| return .{ .admission_fault = err };
                const extent = Core.Dense.extent(m, n, ld) catch return .{ .admission_fault = error.SizeOverflow };
                const values = capacity(T, args.values, extent) catch |err| return .{ .admission_fault = err };
                const pivots = pivotOutput(T, I, args, values, m, n) catch |err| return .{ .admission_fault = err };
                const info = Core.factorGetrf(T, I, .{ .values = values, .row_count = m, .column_count = n, .leading_dimension = ld }, pivots);
                return .{ .lapacke_info = @intCast(info) };
            }
            // Reference row work checks lda before m/n, then allocates even for
            // empty/invalid dimensions. This is not a generic early no-op path.
            if (args.lda < args.n) {
                hook(T, services, .lapacke, .work, -5);
                return .{ .lapacke_info = -5 };
            }
            const temp_m = native(@max(@as(I, 1), args.m)) catch |err| return .{ .admission_fault = err };
            const temp_n = native(@max(@as(I, 1), args.n)) catch |err| return .{ .admission_fault = err };
            const count = Core.Storage.columnMajorExtent(temp_m, temp_n, temp_m) catch |err| return .{ .admission_fault = err };
            _ = Core.Storage.byteLength(count, @sizeOf(T)) catch |err| return .{ .admission_fault = err };
            var lease = scratch.acquire(source, count) catch |err| {
                if (comptime lifetime == .context_managed) switch (err) {
                    error.CounterOverflow => return error.CounterOverflow,
                    error.OutOfMemory => {},
                } else switch (err) {
                    error.OutOfMemory => {},
                }
                hook(T, services, .lapacke, .work, -1011);
                return .{ .lapacke_info = -1011 };
            };
            defer lease.deinit();
            var original: []T = &.{};
            var m: usize = 0;
            var n: usize = 0;
            var ld: usize = 0;
            if (args.m > 0 and args.n > 0) {
                m = native(args.m) catch |err| return .{ .admission_fault = err };
                n = native(args.n) catch |err| return .{ .admission_fault = err };
                ld = native(args.lda) catch |err| return .{ .admission_fault = err };
                const extent = Core.Storage.columnMajorExtent(n, m, ld) catch |err| return .{ .admission_fault = err };
                original = capacity(T, args.values, extent) catch |err| return .{ .admission_fault = err };
                transfer(T, original, scratch.values(&lease), m, n, ld, true);
            }
            const argument = fortranArgument(I, args.m, args.n, @intCast(temp_m));
            var info: I = 0;
            if (argument != 0) {
                hook(T, services, .fortran_returning_test, .fortran_getrf, argument);
                info = -@as(I, argument) - 1;
            } else if (m != 0 and n != 0) {
                const pivots = pivotOutput(T, I, args, original, m, n) catch |err| return .{ .admission_fault = err };
                info = @intCast(Core.factorGetrf(T, I, .{ .values = scratch.values(&lease), .row_count = m, .column_count = n, .leading_dimension = temp_m }, pivots));
            }
            // Singular positive INFO still commits the entire active factor;
            // pivots were already written directly. Free precedes Operation.end.
            if (m != 0 and n != 0) transfer(T, original, scratch.values(&lease), m, n, ld, false);
            return .{ .lapacke_info = info };
        }
    };
}
