// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Internal ORMQR/UNMQR IEEE high/work/query staging. Not installed C ABI.
//! All scratch and admission belong to the injected existing numerical owner.
const std = @import("std");
const common = @import("lapacke_common.zig");
pub fn Adapter(comptime Core: type, comptime kind: common.access.Kind) type {
    return struct {
        const h = common.HelpersWithAccess(Core, kind);
        const E = Core.Execution;
        const Owner = Core.OwnerIdentity;
        pub const Error = common.access.Fault(kind) || E.context.AdmissionError || error{};
        pub fn Arguments(comptime T: type) type {
            h.scalar(T);
            return struct {
                layout: c_int,
                side: u8,
                trans: u8,
                m: i32,
                n: i32,
                k: i32,
                a: common.access.Operand(T, false, kind) = null,
                lda: i32,
                tau: common.access.Operand(T, false, kind) = null,
                c: common.access.Operand(T, true, kind) = null,
                ldc: i32,
                work: common.access.Operand(T, true, kind) = null,
                lwork: i32 = 0,
            };
        }
        fn left(a: anytype) bool {
            return std.ascii.toUpper(a.side) == 'L';
        }
        fn order(a: anytype) i32 {
            return if (left(a)) a.m else a.n;
        }
        fn count(a: anytype) usize {
            return @intCast(@max(1, if (left(a)) a.n else a.m));
        }
        fn number(comptime T: type, n: usize) T {
            const R = if (comptime Core.Scalar.isComplex(T)) @TypeOf(@as(T, undefined).re) else T;
            const value = common.workspaceQueryValue(R, n);
            if (comptime Core.Scalar.isComplex(T)) return .{ .re = value, .im = 0 };
            return value;
        }
        fn notify(comptime T: type, services: *const Core.Services, entry: common.Entry, arg: i32) void {
            const complex = Core.Scalar.isComplex(T);
            h.hook(T, services.*, if (entry == .high or entry == .work) .lapacke else .fortran_returning_test, entry, arg, if (complex) .unmqr else .ormqr);
        }
        fn validate(comptime T: type, services: *const Core.Services, a: Arguments(T), lda: i32, ldc: i32) ?i32 {
            const side = std.ascii.toUpper(a.side);
            const trans = std.ascii.toUpper(a.trans);
            const allowed: u8 = if (Core.Scalar.isComplex(T)) 'C' else 'T';
            const arg: i32 = if (side != 'L' and side != 'R') 1 else if (trans != 'N' and trans != allowed) 2 else if (a.m < 0) 3 else if (a.n < 0) 4 else if (a.k < 0 or a.k > order(a)) 5 else if (lda < @max(1, order(a))) 7 else if (ldc < @max(1, a.m)) 10 else if (a.lwork != -1 and a.lwork < @max(1, if (left(a)) a.n else a.m)) 12 else 0;
            if (arg == 0) return null;
            notify(T, services, if (Core.Scalar.isComplex(T)) .fortran_unmqr else .fortran_ormqr, arg);
            return -arg - 1;
        }
        fn extent(rows: usize, cols: usize, ld: usize, row: bool) Error!usize {
            return Core.Storage.columnMajorExtent(if (row) cols else rows, if (row) rows else cols, ld) catch error.SizeOverflow;
        }
        fn scratch(comptime T: type, op: *const E.context.Operation, services: *const Core.Services, n: usize) Error!?E.context.ContextScratch(T) {
            _ = Core.Storage.byteLength(n, @sizeOf(T)) catch return error.SizeOverflow;
            return E.context.ContextScratch(T).acquire(op, n) catch |err| switch (err) {
                error.CounterOverflow => return error.CounterOverflow,
                error.OutOfMemory => {
                    notify(T, services, .work, -1011);
                    return null;
                },
            };
        }
        pub fn workAdmitted(comptime T: type, op: *const E.context.Operation, services: *const Core.Services, a: Arguments(T)) Error!i32 {
            _ = op.borrowedContext();
            if (a.layout != 101 and a.layout != 102) {
                notify(T, services, .work, -1);
                return -1;
            }
            const row = a.layout == 101;
            if (row and a.lda < a.k) {
                notify(T, services, .work, -8);
                return -8;
            }
            if (row and a.ldc < a.n) {
                notify(T, services, .work, -11);
                return -11;
            }
            const lda = if (row) @max(1, order(a)) else a.lda;
            const ldc = if (row) @max(1, a.m) else a.ldc;
            if (a.lwork == -1) {
                if (validate(T, services, a, lda, ldc)) |info| return info;
                const work = try h.capacity(T, a.work, 1);
                work[0] = number(T, count(a));
                return 0;
            }
            if (!row) if (validate(T, services, a, lda, ldc)) |info| return info;
            var af: ?E.context.ContextScratch(T) = null;
            defer if (af) |*v| v.deinit();
            var cf: ?E.context.ContextScratch(T) = null;
            defer if (cf) |*v| v.deinit();
            if (row) {
                af = try scratch(T, op, services, try extent(@intCast(lda), @intCast(@max(1, a.k)), @intCast(lda), false));
                if (af == null) return -1011;
                cf = try scratch(T, op, services, try extent(@intCast(ldc), @intCast(@max(1, a.n)), @intCast(ldc), false));
                if (cf == null) return -1011;
            }
            const m: usize = @intCast(@max(0, a.m));
            const n: usize = @intCast(@max(0, a.n));
            const k: usize = @intCast(@max(0, a.k));
            const nq: usize = @intCast(@max(0, order(a)));
            var original_c: []T = &.{};
            var original_a: []const T = &.{};
            if (row) {
                original_a = try h.constCapacity(T, a.a, try extent(nq, k, @intCast(@max(0, a.lda)), true));
                original_c = try h.capacity(T, a.c, try extent(m, n, @intCast(@max(0, a.ldc)), true));
                for (0..nq) |i| for (0..k) |j| {
                    af.?.lease.values[i + j * @as(usize, @intCast(lda))] = original_a[i * @as(usize, @intCast(a.lda)) + j];
                };
                for (0..m) |i| for (0..n) |j| {
                    cf.?.lease.values[i + j * @as(usize, @intCast(ldc))] = original_c[i * @as(usize, @intCast(a.ldc)) + j];
                };
                if (validate(T, services, a, lda, ldc)) |info| return info;
            }
            const work = try h.capacity(T, a.work, @intCast(a.lwork));
            if (m == 0 or n == 0 or k == 0) {
                work[0] = number(T, 1);
                return 0;
            }
            const factor: []const T = if (row) af.?.lease.values else try h.constCapacity(T, a.a, try extent(nq, k, @intCast(lda), false));
            const target: []T = if (row) cf.?.lease.values else try h.capacity(T, a.c, try extent(m, n, @intCast(ldc), false));
            const tau = try h.constCapacity(T, a.tau, k);
            try h.disjoint(T, factor, target);
            try h.disjointMixed(T, factor, T, tau);
            try h.disjoint(T, tau, target);
            try h.disjoint(T, factor, work);
            try h.disjoint(T, tau, work);
            try h.disjoint(T, target, work);
            if (row) {
                try h.disjoint(T, original_a, original_c);
                try h.disjoint(T, original_a, work);
                try h.disjoint(T, original_c, work);
                try h.disjoint(T, tau, original_c);
            }
            const trans = std.ascii.toUpper(a.trans);
            const operation: Owner.lapack.QOperation = if (trans == 'N') .normal else if (Core.Scalar.isComplex(T)) .conjugate_transposed else .transposed;
            const info = Owner.internal.lapack_checked.qrApplyQSideWithContractAdmitted(op, T, .{ .values = factor, .row_count = @intCast(nq), .column_count = @intCast(k), .leading_dimension = lda }, tau, .{ .values = target, .row_count = @intCast(m), .column_count = @intCast(n), .leading_dimension = ldc }, if (left(a)) .left else .right, operation, .lapack_ieee_v1) catch |err| switch (err) {
                error.AliasingNotAllowed => return error.AliasingNotAllowed,
                else => unreachable,
            };
            if (row) for (0..m) |i| for (0..n) |j| {
                original_c[i * @as(usize, @intCast(a.ldc)) + j] = target[i + j * @as(usize, @intCast(ldc))];
            };
            std.debug.assert(info == 0);
            work[0] = number(T, count(a));
            return 0;
        }
        pub fn highAdmitted(comptime scan: bool, comptime T: type, op: *const E.context.Operation, services: *const Core.Services, a: Arguments(T)) Error!i32 {
            _ = op.borrowedContext();
            if (a.layout != 101 and a.layout != 102) {
                notify(T, services, .high, -1);
                return -1;
            }
            if (comptime scan) if (services.nan_enabled(services.context)) {
                if (try h.scanGeneral(T, a.layout, order(a), a.k, a.lda, a.a)) return -7;
                if (try h.scanGeneral(T, a.layout, a.m, a.n, a.ldc, a.c)) return -10;
                if (a.k > 0 and a.tau != null) {
                    const tau = try h.constCapacity(T, a.tau, @intCast(a.k));
                    for (tau) |z| if (h.nan(T, z)) return -9;
                }
            };
            var query: [1]T = undefined;
            var qa = a;
            qa.work = if (kind == .typed) &query else @ptrCast(&query);
            qa.lwork = -1;
            const info = try workAdmitted(T, op, services, qa);
            if (info != 0) return info;
            var lease = E.context.ContextScratch(T).acquire(op, count(a)) catch |err| switch (err) {
                error.CounterOverflow => return error.CounterOverflow,
                error.OutOfMemory => {
                    notify(T, services, .high, -1010);
                    return -1010;
                },
            };
            defer lease.deinit();
            var wa = a;
            wa.work = if (kind == .typed) lease.lease.values else @ptrCast(lease.lease.values.ptr);
            wa.lwork = @intCast(count(a));
            return workAdmitted(T, op, services, wa);
        }
        pub fn callManaged(comptime scan: bool, comptime T: type, level: common.Level, ctx: *E.context.Context, services: *const Core.Services, a: Arguments(T)) Error!i32 {
            var invocation = try common.ManagedInvocation(Core).begin(ctx, services);
            defer invocation.operation.end();
            return if (level == .high) highAdmitted(scan, T, &invocation.operation, &invocation.services, a) else workAdmitted(T, &invocation.operation, &invocation.services, a);
        }
    };
}
