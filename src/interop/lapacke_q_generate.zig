// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Internal ORGQR/UNGQR high/work staging; not a generated or installed C ABI.
//! Work uses the shared IEEE numerical policy; checked Zig calls retain theirs.
//! No positive arithmetic INFO or vendor payload/trajectory equivalence is claimed.
const common = @import("lapacke_common.zig");
pub fn Adapter(comptime Core: type, comptime kind: common.access.Kind) type {
    return struct {
        const std = @import("std");
        const h = common.HelpersWithAccess(Core, kind);
        const E = Core.Execution;
        const Owner = Core.OwnerIdentity;
        pub const Error = common.access.Fault(kind) || E.context.AdmissionError || error{};
        pub fn Arguments(comptime T: type) type {
            h.scalar(T);
            return struct {
                layout: c_int,
                m: i32,
                n: i32,
                k: i32,
                a: common.access.Operand(T, true, kind) = null,
                lda: i32,
                tau: common.access.Operand(T, false, kind) = null,
                work: common.access.Operand(T, true, kind) = null,
                lwork: i32 = 0,
            };
        }
        fn event(comptime T: type, services: *const Core.Services, entry: common.Entry, info: i32) void {
            h.hook(T, services.*, if (entry == .high or entry == .work) .lapacke else .fortran_returning_test, entry, info, if (Core.Scalar.isComplex(T)) .ungqr else .orgqr);
        }
        fn scalarCount(comptime T: type, count: usize) T {
            const R = if (comptime Core.Scalar.isComplex(T)) @TypeOf(@as(T, undefined).re) else T;
            const value = common.workspaceQueryValue(R, count);
            if (comptime Core.Scalar.isComplex(T)) return .{ .re = value, .im = 0 };
            return value;
        }
        fn validate(comptime T: type, services: *const Core.Services, a: Arguments(T), lda: i32) ?i32 {
            const arg: i32 = if (a.m < 0) 1 else if (a.n < 0 or a.n > a.m) 2 else if (a.k < 0 or a.k > a.n) 3 else if (lda < @max(1, a.m)) 5 else if (a.lwork != -1 and a.lwork < @max(1, a.n)) 8 else 0;
            if (arg == 0) return null;
            event(T, services, if (Core.Scalar.isComplex(T)) .fortran_ungqr else .fortran_orgqr, arg);
            return -arg - 1;
        }
        fn recommendation(a: anytype) usize {
            return @intCast(@max(1, a.n));
        }
        pub fn workAdmitted(comptime T: type, op: *const E.context.Operation, services: *const Core.Services, a: Arguments(T)) Error!i32 {
            _ = op.borrowedContext();
            if (a.layout != 101 and a.layout != 102) {
                event(T, services, .work, -1);
                return -1;
            }
            const row = a.layout == 101;
            if (row and a.lda < a.n) {
                event(T, services, .work, -6);
                return -6;
            }
            const lda = if (row) @max(1, a.m) else a.lda;
            if (a.lwork == -1) {
                if (validate(T, services, a, lda)) |info| return info;
                const work = try h.capacity(T, a.work, 1);
                work[0] = scalarCount(T, recommendation(a));
                return 0;
            }
            // Column work validates scalar arguments before any data access.
            if (!row) if (validate(T, services, a, lda)) |info| return info;
            var lease: ?E.context.ContextScratch(T) = null;
            defer if (lease) |*v| v.deinit();
            if (row) {
                const count = Core.Storage.columnMajorExtent(@intCast(@max(1, a.m)), @intCast(@max(1, a.n)), @intCast(lda)) catch return error.SizeOverflow;
                _ = Core.Storage.byteLength(count, @sizeOf(T)) catch return error.SizeOverflow;
                lease = E.context.ContextScratch(T).acquire(op, count) catch |err| switch (err) {
                    error.CounterOverflow => return error.CounterOverflow,
                    error.OutOfMemory => {
                        event(T, services, .work, -1011);
                        return -1011;
                    },
                };
            }
            const m: usize = @intCast(@max(0, a.m));
            const n: usize = @intCast(@max(0, a.n));
            const ld: usize = @intCast(@max(0, a.lda));
            const span = Core.Storage.columnMajorExtent(if (row) n else m, if (row) m else n, ld) catch return error.SizeOverflow;
            const values: []T = if (row) try h.capacity(T, a.a, span) else &.{};
            if (row) {
                const temp = lease.?.lease.values;
                for (0..m) |i| for (0..n) |j| {
                    temp[i + j * @as(usize, @intCast(lda))] = values[i * ld + j];
                };
                if (validate(T, services, a, lda)) |info| return info;
            }
            const work = try h.capacity(T, a.work, @intCast(a.lwork));
            if (m == 0 or n == 0) {
                work[0] = scalarCount(T, 1);
                return 0;
            }
            const tau = try h.constCapacity(T, a.tau, @intCast(a.k));
            const column = if (row) lease.?.lease.values else try h.capacity(T, a.a, span);
            try h.disjoint(T, tau, column);
            try h.disjoint(T, column, work);
            try h.disjoint(T, tau, work);
            if (row) {
                try h.disjoint(T, tau, values);
                try h.disjoint(T, values, work);
            }
            const info = Owner.internal.lapack_checked.qrGenerateQWithContractAdmitted(op, T, .{ .values = column, .row_count = @intCast(m), .column_count = @intCast(n), .leading_dimension = lda }, tau, @intCast(a.k), .lapack_ieee_v1) catch |err| switch (err) {
                error.AliasingNotAllowed => return error.AliasingNotAllowed,
                else => unreachable,
            };
            if (row) for (0..m) |i| for (0..n) |j| {
                values[i * ld + j] = column[i + j * @as(usize, @intCast(lda))];
            };
            std.debug.assert(info == 0);
            work[0] = scalarCount(T, recommendation(a));
            return 0;
        }
        pub fn highAdmitted(comptime scan: bool, comptime T: type, op: *const E.context.Operation, services: *const Core.Services, a: Arguments(T)) Error!i32 {
            _ = op.borrowedContext();
            if (a.layout != 101 and a.layout != 102) {
                event(T, services, .high, -1);
                return -1;
            }
            if (comptime scan) if (services.nan_enabled(services.context)) {
                if (try h.scanGeneral(T, a.layout, a.m, a.n, a.lda, a.a)) return -5;
                if (a.k > 0 and a.tau != null) {
                    const tau = try h.constCapacity(T, a.tau, @intCast(a.k));
                    for (tau) |value| if (h.nan(T, value)) return -7;
                }
            };
            var query: [1]T = undefined;
            var qa = a;
            qa.lwork = -1;
            qa.work = if (kind == .typed) &query else @ptrCast(&query);
            const info = try workAdmitted(T, op, services, qa);
            if (info != 0) return info;
            const count = recommendation(a);
            _ = Core.Storage.byteLength(count, @sizeOf(T)) catch return error.SizeOverflow;
            var lease = E.context.ContextScratch(T).acquire(op, count) catch |err| switch (err) {
                error.CounterOverflow => return error.CounterOverflow,
                error.OutOfMemory => {
                    event(T, services, .high, -1010);
                    return -1010;
                },
            };
            defer lease.deinit();
            var wa = a;
            wa.work = if (kind == .typed) lease.lease.values else @ptrCast(lease.lease.values.ptr);
            wa.lwork = @intCast(count);
            return workAdmitted(T, op, services, wa);
        }
        pub fn callManaged(comptime scan: bool, comptime T: type, level: common.Level, ctx: *E.context.Context, services: *const Core.Services, a: Arguments(T)) Error!i32 {
            var invocation = try common.ManagedInvocation(Core).begin(ctx, services);
            defer invocation.operation.end();
            return if (level == .high) highAdmitted(scan, T, &invocation.operation, &invocation.services, a) else workAdmitted(T, &invocation.operation, &invocation.services, a);
        }
    };
}
