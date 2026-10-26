// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal GELSD high/work adapter over one borrowed numerical owner.
//! Explicit numerical faults are not yet a complete standard IEEE export policy.
const common = @import("lapacke_common.zig");
const query = @import("lapacke_gelsd_query.zig");
const column = @import("lapacke_gelsd_column.zig");
const workspace = @import("lapacke_gelsd_workspace.zig");
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
            if (comptime scan) {
                if (level == .high and services.nan_enabled(services.context)) {
                    if (H.scanGeneral(T, args.layout, a.m, a.n, a.lda, a.a) catch |err| return .{ .admission_fault = err }) return .{ .lapacke_info = -5 };
                    if (H.scanGeneral(T, args.layout, @max(a.m, a.n), a.nrhs, a.ldb, a.b) catch |err| return .{ .admission_fault = err }) return .{ .lapacke_info = -7 };
                    if (H.nan(T, a.rcond)) return .{ .lapacke_info = -10 };
                }
            }
            if (level == .work) return middle(T, I, lifetime, services, source, args);
            var work_query: [1]T = undefined;
            var integer_query: [1]I = undefined;
            const queried = TypedQ.call(T, I, &services, .{ .layout = args.layout, .m = a.m, .n = a.n, .nrhs = a.nrhs, .lda = a.lda, .ldb = a.ldb, .work = &work_query, .iwork = &integer_query });
            switch (queried) {
                // Both query destinations are distinct local arrays of exactly one element.
                .admission_fault => |err| return .{ .admission_fault = switch (err) {
                    error.BufferTooSmall, error.MissingBuffer, error.AliasingNotAllowed => unreachable,
                    error.SizeOverflow => error.SizeOverflow,
                    error.IntegerNotRepresentable => error.IntegerNotRepresentable,
                } },
                .lapacke_info => |info| if (info != 0) return .{ .lapacke_info = info },
            }
            const IntegerScratch = common.Scratch(Core, I, lifetime);
            const ScalarScratch = common.Scratch(Core, T, lifetime);
            var integers = IntegerScratch.acquire(source, @intCast(integer_query[0])) catch |err| {
                if (comptime lifetime == .context_managed) {
                    if (err == error.CounterOverflow) return error.CounterOverflow;
                }
                hook(T, services, .high, -1010);
                return .{ .lapacke_info = -1010 };
            };
            defer integers.deinit();
            const length: I = @intFromFloat(work_query[0]);
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
            return middle(T, I, lifetime, services, source, prepared);
        }
        fn middle(comptime T: type, comptime I: type, comptime lifetime: common.Lifetime, services: Services, source: common.Scratch(Core, T, lifetime).Source, args: Arguments(T, I)) common.EngineResult(lifetime, Result(I)) {
            if (args.layout == 102) return C.call(T, I, &services, args.column, .{});
            const a = args.column;
            if (a.lda < a.n) {
                hook(T, services, .work, -6);
                return .{ .lapacke_info = -6 };
            }
            if (a.ldb < a.nrhs) {
                hook(T, services, .work, -8);
                return .{ .lapacke_info = -8 };
            }
            if (a.lwork == -1) return fromQuery(I, Q.call(T, I, &services, .{ .layout = 101, .m = a.m, .n = a.n, .nrhs = a.nrhs, .lda = a.lda, .ldb = a.ldb, .work = a.work, .iwork = a.iwork }));
            const Scratch = common.Scratch(Core, T, lifetime);
            const lda_t = @max(@as(I, 1), a.m);
            const ldb_t = @max(@as(I, 1), @max(a.m, a.n));
            const a_count = Core.Storage.byteLength(H.native(lda_t) catch |err| return .{ .admission_fault = err }, H.native(@max(@as(I, 1), a.n)) catch |err| return .{ .admission_fault = err }) catch |err| return .{ .admission_fault = err };
            const b_count = Core.Storage.byteLength(H.native(ldb_t) catch |err| return .{ .admission_fault = err }, H.native(@max(@as(I, 1), a.nrhs)) catch |err| return .{ .admission_fault = err }) catch |err| return .{ .admission_fault = err };
            _ = Core.Storage.byteLength(a_count, @sizeOf(T)) catch |err| return .{ .admission_fault = err };
            _ = Core.Storage.byteLength(b_count, @sizeOf(T)) catch |err| return .{ .admission_fault = err };
            var at = Scratch.acquire(source, a_count) catch |err| {
                if (comptime lifetime == .context_managed) {
                    if (err == error.CounterOverflow) return error.CounterOverflow;
                }
                hook(T, services, .work, -1011);
                return .{ .lapacke_info = -1011 };
            };
            defer at.deinit();
            var bt = Scratch.acquire(source, b_count) catch |err| {
                if (comptime lifetime == .context_managed) {
                    if (err == error.CounterOverflow) return error.CounterOverflow;
                }
                hook(T, services, .work, -1011);
                return .{ .lapacke_info = -1011 };
            };
            defer bt.deinit();
            const m = H.native(@max(@as(I, 0), a.m)) catch |err| return .{ .admission_fault = err };
            const n = H.native(@max(@as(I, 0), a.n)) catch |err| return .{ .admission_fault = err };
            const nrhs = H.native(@max(@as(I, 0), a.nrhs)) catch |err| return .{ .admission_fault = err };
            const lda = H.native(@max(@as(I, 0), a.lda)) catch |err| return .{ .admission_fault = err };
            const ldb = H.native(@max(@as(I, 0), a.ldb)) catch |err| return .{ .admission_fault = err };
            const original_a = H.capacity(T, a.a, Core.Storage.columnMajorExtent(n, m, lda) catch |err| return .{ .admission_fault = err }) catch |err| return .{ .admission_fault = err };
            const original_b = H.capacity(T, a.b, Core.Storage.columnMajorExtent(nrhs, @max(m, n), ldb) catch |err| return .{ .admission_fault = err }) catch |err| return .{ .admission_fault = err };
            H.disjoint(T, original_a, original_b) catch |err| return .{ .admission_fault = err };
            // Transposed operands no longer expose the original addresses to the
            // column engine. Admit original/output overlap before any copyback.
            if (a.m >= 0 and a.n >= 0 and a.nrhs >= 0) {
                const plan = workspace.Planner(Core).make(T, I, m, n, nrhs) catch return .{ .admission_fault = error.SizeOverflow };
                const numerical = a.lwork >= plan.work;
                const work_values = H.capacity(T, a.work, if (numerical) @intCast(plan.work) else 1) catch |err| return .{ .admission_fault = err };
                const integer_values = H.capacity(I, a.iwork, if (numerical and m != 0 and n != 0) @intCast(plan.iwork) else 1) catch |err| return .{ .admission_fault = err };
                const singular_values = H.capacity(T, a.singular, if (numerical) @min(m, n) else 0) catch |err| return .{ .admission_fault = err };
                const rank_values = H.capacity(I, a.rank, if (numerical) 1 else 0) catch |err| return .{ .admission_fault = err };
                for ([_][]T{ original_a, original_b }) |original| {
                    const original_bytes = Core.Storage.byteLength(original.len, @sizeOf(T)) catch |err| return .{ .admission_fault = err };
                    const original_range = Core.Storage.byteRange(@intFromPtr(original.ptr), original_bytes) catch |err| return .{ .admission_fault = err };
                    inline for (.{ work_values, integer_values, singular_values, rank_values }) |other| {
                        const bytes = Core.Storage.byteLength(other.len, @sizeOf(@TypeOf(other[0]))) catch |err| return .{ .admission_fault = err };
                        const other_range = Core.Storage.byteRange(@intFromPtr(other.ptr), bytes) catch |err| return .{ .admission_fault = err };
                        if (Core.Storage.overlaps(original_range, other_range)) return .{ .admission_fault = error.AliasingNotAllowed };
                    }
                }
            }
            H.transferGeneral(T, original_a, Scratch.values(&at), m, n, lda, true);
            H.transferGeneral(T, original_b, Scratch.values(&bt), @max(m, n), nrhs, ldb, true);
            var normalized = a;
            normalized.a = operand(T, Scratch.values(&at));
            normalized.b = operand(T, Scratch.values(&bt));
            normalized.lda = lda_t;
            normalized.ldb = ldb_t;
            const result = C.call(T, I, &services, normalized, .{});
            H.transferGeneral(T, original_a, Scratch.values(&at), m, n, lda, false);
            H.transferGeneral(T, original_b, Scratch.values(&bt), @max(m, n), nrhs, ldb, false);
            return result;
        }
    };
}
