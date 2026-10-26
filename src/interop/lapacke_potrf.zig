// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Private typed POTRF interface prototype. No C exports or wire reinterpretation.
//! Reference LAPACKE 3.12.1 control order, with an explicitly returning Fortran
//! error-event test profile (the shipped Reference XERBLA instead STOPs).
//! Core and borrowed services come from one implementation owner. No numerical
//! modules, allocation owner, configuration singleton or runtime are imported.
const std = @import("std");
const common = @import("lapacke_common.zig");
pub const Level = common.Level;
pub const HookKind = common.HookKind;
pub const Entry = common.Entry;
pub const ErrorEvent = common.ErrorEvent;
pub const AdmissionFault = common.AdmissionFault;

/// Core supplies Dense, Storage, Scalar, ScratchLease, Services, OwnerIdentity,
/// and factorPotrf(T, Dense.Matrix(T), upper). Services is borrowed for this
/// synchronous call; callbacks/allocator state must remain alive and compatible.
/// Capture the service binding once, before callbacks. Typed slices must denote
/// truthful stable allocations. Neither capacity checks nor the standard raw
/// interface can prove that a caller has supplied valid addresses.
pub fn ReferencePotrfAdapter(comptime Core: type, comptime compile_nan_scan: bool) type {
    return AdapterEngine(Core, compile_nan_scan, .typed);
}

/// Internal raw-operand specialization, not a C entry point. Caller-owned raw
/// storage must satisfy every stage-local access; raw faults are never INFO.
pub fn RawPotrfAdapter(comptime Core: type, comptime compile_nan_scan: bool) type {
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
        const integer = helpers.integer;
        const scalar = helpers.scalar;
        const triangle = helpers.triangle;
        fn hook(comptime T: type, services: Services, kind: HookKind, entry: Entry, argument: i32) void {
            helpers.hook(T, services, kind, entry, argument, .potrf);
        }
        const native = helpers.native;
        const capacity = helpers.capacity;
        const scan = helpers.scanTriangle;
        const transfer = helpers.transfer;

        pub fn Result(comptime I: type) type {
            integer(I);
            return union(enum) { lapacke_info: I, admission_fault: common.access.Fault(access_kind) };
        }
        pub fn Arguments(comptime T: type, comptime I: type) type {
            integer(I);
            scalar(T);
            return struct { layout: c_int, uplo: u8, n: I, lda: I, values: common.access.Operand(T, true, access_kind) };
        }
        fn fortranArgument(comptime I: type, uplo: u8, n: I, ld: I) i32 {
            if (triangle(uplo) == null) return 1;
            if (n < 0) return 2;
            if (ld < @max(@as(I, 1), n)) return 4;
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
                    const found = scan(T, I, args) catch |err| return .{ .admission_fault = err };
                    if (found) return .{ .lapacke_info = -4 };
                }
            }
            if (args.layout == 102) {
                const argument = fortranArgument(I, args.uplo, args.n, args.lda);
                if (argument != 0) {
                    hook(T, services, .fortran_returning_test, .fortran_potrf, argument);
                    return .{ .lapacke_info = -@as(I, argument) - 1 };
                }
                if (args.n == 0) return .{ .lapacke_info = 0 };
                const n = native(args.n) catch |err| return .{ .admission_fault = err };
                const ld = native(args.lda) catch |err| return .{ .admission_fault = err };
                const extent = Core.Dense.extent(n, n, ld) catch return .{ .admission_fault = error.SizeOverflow };
                const values = capacity(T, args.values, extent) catch |err| return .{ .admission_fault = err };
                const info = Core.factorPotrf(T, .{ .values = values, .row_count = n, .column_count = n, .leading_dimension = ld }, triangle(args.uplo).?);
                return .{ .lapacke_info = @intCast(info) };
            }
            if (args.lda < args.n) {
                hook(T, services, .lapacke, .work, -5);
                return .{ .lapacke_info = -5 };
            }
            const temp_n = native(@max(@as(I, 1), args.n)) catch |err| return .{ .admission_fault = err };
            const count = Core.Storage.columnMajorExtent(temp_n, temp_n, temp_n) catch |err| return .{ .admission_fault = err };
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
            var n: usize = 0;
            var ld: usize = 0;
            const upper = triangle(args.uplo);
            if (args.n > 0 and upper != null) {
                n = native(args.n) catch |err| return .{ .admission_fault = err };
                ld = native(args.lda) catch |err| return .{ .admission_fault = err };
                const extent = Core.Storage.columnMajorExtent(n, n, ld) catch |err| return .{ .admission_fault = err };
                original = capacity(T, args.values, extent) catch |err| return .{ .admission_fault = err };
                transfer(T, original, scratch.values(&lease), n, ld, upper.?, true);
            }
            const argument = fortranArgument(I, args.uplo, args.n, @intCast(temp_n));
            var info: I = 0;
            if (argument != 0) {
                hook(T, services, .fortran_returning_test, .fortran_potrf, argument);
                info = -@as(I, argument) - 1;
            } else if (n != 0) {
                info = @intCast(Core.factorPotrf(T, .{ .values = scratch.values(&lease), .row_count = n, .column_count = n, .leading_dimension = temp_n }, upper.?));
            }
            if (n != 0) transfer(T, original, scratch.values(&lease), n, ld, upper.?, false);
            return .{ .lapacke_info = info };
        }
    };
}
