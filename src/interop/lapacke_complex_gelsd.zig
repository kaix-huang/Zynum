// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal complex GELSD high/work layout adapter over the original borrowed owner.
//! Explicit numerical faults are not yet a complete standard IEEE export policy.
const common = @import("lapacke_common.zig");
const query = @import("lapacke_complex_gelsd_query.zig");
const column = @import("lapacke_complex_gelsd_column.zig");
const workspace = @import("lapacke_complex_gelsd_workspace.zig");
const policy = @import("lapacke_complex_gelsd_policy.zig");
pub fn Adapter(comptime Core: type, comptime scan: bool, comptime kind: common.access.Kind) type {
    return struct {
        const C = column.ColumnWork(Core, kind);
        const H = common.HelpersWithAccess(Core, kind);
        const Q = query.Query(Core, kind);
        const TypedQ = query.Query(Core, .typed);
        pub const Services = Core.Services;
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub const Result = C.Result;
        pub fn Arguments(comptime T: type, comptime I: type) type {
            return struct { layout: c_int, column: C.Arguments(T, I) };
        }
        fn operand(comptime T: type, values: []T) common.access.Operand(T, true, kind) {
            return if (kind == .typed) values else @as([*]align(1) T, values.ptr);
        }
        fn fromQuery(comptime I: type, value: anytype) Result(I) {
            return switch (value) {
                .lapacke_info => |info| .{ .lapacke_info = info },
                .admission_fault => |err| .{ .admission_fault = err },
            };
        }
        fn hook(comptime T: type, services: Services, level: common.Level, info: i32) void {
            H.hook(T, services, .lapacke, if (level == .high) .high else .work, info, .gelsd);
        }
        pub fn call(comptime T: type, comptime I: type, level: common.Level, borrowed: *const Services, args: Arguments(T, I)) Result(I) {
            const services = borrowed.*;
            return captured(T, I, level, .unmanaged, services, services.allocator, args);
        }
        pub fn callManaged(comptime T: type, comptime I: type, level: common.Level, context: *Core.Execution.context.Context, borrowed: *const Services, args: Arguments(T, I)) Core.Execution.context.AdmissionError!Result(I) {
            var invocation = try common.ManagedInvocation(Core).begin(context, borrowed);
            defer invocation.operation.end();
            return captured(T, I, level, .context_managed, invocation.services, &invocation.operation, args);
        }
        pub fn callAdmitted(comptime T: type, comptime I: type, level: common.Level, operation: *const Core.Execution.context.Operation, borrowed: *const Services, args: Arguments(T, I)) error{CounterOverflow}!Result(I) {
            const services: Services = .{ .allocator = operation.allocator(), .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
            return captured(T, I, level, .context_managed, services, operation, args);
        }
        fn captured(comptime T: type, comptime I: type, level: common.Level, comptime lifetime: common.Lifetime, services: Services, source: common.Scratch(Core, T, lifetime).Source, args: Arguments(T, I)) common.EngineResult(lifetime, Result(I)) {
            if (args.layout != 101 and args.layout != 102) {
                hook(T, services, level, -1);
                return .{ .lapacke_info = -1 };
            }
            const a = args.column;
            const R = Core.Scalar.Real(T);
            if (comptime scan) {
                if (level == .high and services.nan_enabled(services.context)) {
                    if (H.scanGeneral(T, args.layout, a.m, a.n, a.lda, a.a) catch |err| return .{ .admission_fault = err }) return .{ .lapacke_info = -5 };
                    if (H.scanGeneral(T, args.layout, @max(a.m, a.n), a.nrhs, a.ldb, a.b) catch |err| return .{ .admission_fault = err }) return .{ .lapacke_info = -7 };
                    if (H.nan(R, a.rcond)) return .{ .lapacke_info = -10 };
                }
            }
            if (level == .work) return middle(T, I, lifetime, services, source, args);
            var work_query: [1]T = undefined;
            var integer_query: [1]I = undefined;
            var real_query: [1]R = undefined;
            const queried = TypedQ.call(T, I, &services, .{ .layout = args.layout, .m = a.m, .n = a.n, .nrhs = a.nrhs, .lda = a.lda, .ldb = a.ldb, .work = &work_query, .rwork = &real_query, .iwork = &integer_query });
            switch (queried) {
                // All three query destinations are distinct one-element arrays.
                .admission_fault => |err| return .{ .admission_fault = switch (err) {
                    error.BufferTooSmall, error.MissingBuffer, error.AliasingNotAllowed => unreachable,
                    error.SizeOverflow => error.SizeOverflow,
                    error.IntegerNotRepresentable => error.IntegerNotRepresentable,
                } },
                .lapacke_info => |info| if (info != 0) return .{ .lapacke_info = info },
            }
            const IntegerScratch = common.Scratch(Core, I, lifetime);
            const ScalarScratch = common.Scratch(Core, T, lifetime);
            const RealScratch = common.Scratch(Core, R, lifetime);
            var integers = IntegerScratch.acquire(source, @intCast(integer_query[0])) catch |err| {
                if (comptime lifetime == .context_managed) {
                    if (err == error.CounterOverflow) return error.CounterOverflow;
                }
                hook(T, services, .high, -1010);
                return .{ .lapacke_info = -1010 };
            };
            defer integers.deinit();
            var reals = RealScratch.acquire(source, @intFromFloat(real_query[0])) catch |err| {
                if (comptime lifetime == .context_managed) {
                    if (err == error.CounterOverflow) return error.CounterOverflow;
                }
                hook(T, services, .high, -1010);
                return .{ .lapacke_info = -1010 };
            };
            defer reals.deinit();
            const length: I = @intFromFloat(work_query[0].re);
            var work = ScalarScratch.acquire(source, @intCast(length)) catch |err| {
                if (comptime lifetime == .context_managed) {
                    if (err == error.CounterOverflow) return error.CounterOverflow;
                }
                hook(T, services, .high, -1010);
                return .{ .lapacke_info = -1010 };
            };
            defer work.deinit();
            var prepared = args;
            prepared.column.work = operand(T, ScalarScratch.values(&work));
            prepared.column.lwork = length;
            prepared.column.iwork = operand(I, IntegerScratch.values(&integers));
            prepared.column.rwork = operand(R, RealScratch.values(&reals));
            return middle(T, I, lifetime, services, source, prepared);
        }
        fn middle(comptime T: type, comptime I: type, comptime lifetime: common.Lifetime, services: Services, source: common.Scratch(Core, T, lifetime).Source, args: Arguments(T, I)) common.EngineResult(lifetime, Result(I)) {
            if (args.layout == 102) return C.callAdmitted(T, I, lifetime, &services, source, args.column, .{});
            const a = args.column;
            const R = Core.Scalar.Real(T);
            const scalar_info = policy.scalarInfo(I, 101, a.m, a.n, a.nrhs, a.lda, a.ldb);
            if (scalar_info != 0 or a.lwork == -1) return fromQuery(I, Q.call(T, I, &services, .{ .layout = 101, .m = a.m, .n = a.n, .nrhs = a.nrhs, .lda = a.lda, .ldb = a.ldb, .work = a.work, .rwork = a.rwork, .iwork = a.iwork }));
            const m = H.native(a.m) catch |e| return .{ .admission_fault = e };
            const n = H.native(a.n) catch |e| return .{ .admission_fault = e };
            const rhs = H.native(a.nrhs) catch |e| return .{ .admission_fault = e };
            const minimum = policy.minimumWork(I, m, n, rhs) catch return .{ .admission_fault = error.SizeOverflow };
            var normalized = a;
            normalized.lda = @max(@as(I, 1), a.m);
            normalized.ldb = @max(@as(I, 1), @max(a.m, a.n));
            // Empty extents and scalar/workspace faults never allocate or read
            // numerical operands, even when a legal ABI dimension is enormous.
            if (m == 0 or n == 0 or a.lwork < minimum) return C.callAdmitted(T, I, lifetime, &services, source, normalized, .{});
            const lda = H.native(a.lda) catch |e| return .{ .admission_fault = e };
            const ldb = H.native(a.ldb) catch |e| return .{ .admission_fault = e };
            const ar = Core.Storage.columnMajorExtent(n, m, lda) catch |e| return .{ .admission_fault = e };
            const br = Core.Storage.columnMajorExtent(rhs, @max(m, n), ldb) catch |e| return .{ .admission_fault = e };
            const original_a = H.capacity(T, a.a, ar) catch |e| return .{ .admission_fault = e };
            const original_b = H.capacity(T, a.b, br) catch |e| return .{ .admission_fault = e };
            const singular = H.capacity(R, a.singular, @min(m, n)) catch |e| return .{ .admission_fault = e };
            const rank = H.capacity(I, a.rank, 1) catch |e| return .{ .admission_fault = e };
            const work = H.capacity(T, a.work, @intCast(a.lwork)) catch |e| return .{ .admission_fault = e };
            const real = H.capacity(R, a.rwork, 1) catch |e| return .{ .admission_fault = e };
            const integer = H.capacity(I, a.iwork, 1) catch |e| return .{ .admission_fault = e };
            const spans = .{ original_a, original_b, singular, rank, work, real, integer };
            inline for (spans, 0..) |one, i| inline for (spans, 0..) |other, j| {
                if (comptime j > i) H.disjointMixed(@TypeOf(one[0]), one, @TypeOf(other[0]), other) catch |e| return .{ .admission_fault = e };
            };
            const acount = Core.Storage.byteLength(m, n) catch |e| return .{ .admission_fault = e };
            const bcount = Core.Storage.byteLength(@max(m, n), rhs) catch |e| return .{ .admission_fault = e };
            _ = Core.Storage.byteLength(acount, @sizeOf(T)) catch |e| return .{ .admission_fault = e };
            _ = Core.Storage.byteLength(bcount, @sizeOf(T)) catch |e| return .{ .admission_fault = e };
            const Scratch = common.Scratch(Core, T, lifetime);
            var at = Scratch.acquire(source, acount) catch |e| {
                if (comptime lifetime == .context_managed) {
                    if (e == error.CounterOverflow) return error.CounterOverflow;
                }
                hook(T, services, .work, -1011);
                return .{ .lapacke_info = -1011 };
            };
            defer at.deinit();
            var bt = Scratch.acquire(source, bcount) catch |e| {
                if (comptime lifetime == .context_managed) {
                    if (e == error.CounterOverflow) return error.CounterOverflow;
                }
                hook(T, services, .work, -1011);
                return .{ .lapacke_info = -1011 };
            };
            defer bt.deinit();
            H.transferGeneral(T, original_a, Scratch.values(&at), m, n, lda, true);
            H.transferGeneral(T, original_b, Scratch.values(&bt), @max(m, n), rhs, ldb, true);
            normalized.a = operand(T, Scratch.values(&at));
            normalized.b = operand(T, Scratch.values(&bt));
            const result = if (comptime lifetime == .context_managed) try C.callAdmitted(T, I, lifetime, &services, source, normalized, .{}) else C.callAdmitted(T, I, lifetime, &services, source, normalized, .{});
            if (result == .lapacke_info and result.lapacke_info == 0) H.transferGeneral(T, original_b, Scratch.values(&bt), @max(m, n), rhs, ldb, false);
            return result;
        }
    };
}
