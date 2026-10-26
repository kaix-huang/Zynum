// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! LP64 GELS high/work staging over the configured full-rank numerical owner.
//! External WORK and native ContextScratch remain separate. Every copy, callback,
//! computation and free borrows the caller's one admitted execution operation.
const std = @import("std");
const common = @import("lapacke_common.zig");

pub fn Adapter(comptime Core: type) type {
    return Engine(Core, .typed);
}
pub fn RawAdapter(comptime Core: type) type {
    return Engine(Core, .raw);
}

fn Engine(comptime Core: type, comptime kind: common.access.Kind) type {
    return struct {
        const N = Core.internal.numerics;
        const S = N.Scalar;
        const G = N.FullRankLeastSquares;
        const context = N.Execution.context;
        pub const OwnerIdentity = Core;
        pub const Error = common.access.Fault(kind) || context.AdmissionError || error{ OutOfMemory, NumericalOutcomePendingABI };

        pub fn Arguments(comptime T: type) type {
            if (T != f32 and T != f64 and !S.isComplex(T)) @compileError("GELS scalar type");
            return struct {
                layout: c_int,
                trans: u8,
                m: i32,
                n: i32,
                nrhs: i32,
                a: common.access.Operand(T, true, kind) = null,
                lda: i32,
                b: common.access.Operand(T, true, kind) = null,
                ldb: i32,
                work: common.access.Operand(T, true, kind) = null,
                lwork: i32 = 0,
            };
        }
        fn capacity(comptime T: type, values: common.access.Operand(T, true, kind), count: usize) Error![]T {
            return common.access.capacity(N.Storage, kind, T, true, values, count);
        }
        fn operand(comptime T: type, values: []T) common.access.Operand(T, true, kind) {
            return if (kind == .typed) values else @ptrCast(values.ptr);
        }
        fn disjoint(comptime T: type, a: []const T, b: []const T) Error!void {
            const ar = try N.Storage.byteRange(@intFromPtr(a.ptr), try N.Storage.byteLength(a.len, @sizeOf(T)));
            const br = try N.Storage.byteRange(@intFromPtr(b.ptr), try N.Storage.byteLength(b.len, @sizeOf(T)));
            if (N.Storage.overlaps(ar, br)) return error.AliasingNotAllowed;
        }
        fn callbacks(borrowed: anytype) @TypeOf(.{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event }) {
            return .{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
        }
        fn event(comptime T: type, services: anytype, channel: common.HookKind, entry: common.Entry, info: i32) void {
            services.error_event(services.context, .{ .kind = channel, .entry = entry, .scalar = if (T == f32) 's' else if (T == f64) 'd' else if (T == S.ComplexF32) 'c' else 'z', .argument = info, .routine = .gels });
        }
        fn notify(comptime T: type, services: anytype, args: Arguments(T), info: i32) void {
            if (info >= 0) return;
            const row_local = args.layout == 101 and (args.lda < args.n or args.ldb < args.nrhs);
            if (info == -1 or info == -1010 or info == -1011 or row_local)
                event(T, services, .lapacke, .work, info)
            else
                event(T, services, .fortran_returning_test, .fortran_gels, -info - 1);
        }
        fn problem(comptime T: type, trans: u8) ?G.Operation {
            return switch (trans) {
                'N', 'n' => .normal,
                'T', 't' => if (S.isComplex(T)) null else .adjoint,
                'C', 'c' => if (S.isComplex(T)) .adjoint else null,
                else => null,
            };
        }
        fn controls(comptime T: type, args: Arguments(T)) i32 {
            if (problem(T, args.trans) == null) return -2;
            if (args.m < 0) return -3;
            if (args.n < 0) return -4;
            if (args.nrhs < 0) return -5;
            if (args.lda < @max(1, args.m)) return -7;
            if (args.ldb < @max(1, @max(args.m, args.n))) return -9;
            return 0;
        }
        /// Reference minimum; an implementation-selected recommendation need
        /// not reproduce another backend's block size. Widen before addition.
        pub fn minimumWork(m: i32, n: i32, nrhs: i32) Error!usize {
            if (m < 0 or n < 0 or nrhs < 0) return error.IntegerNotRepresentable;
            const k: usize = @intCast(@min(m, n));
            return @max(1, std.math.add(usize, k, @max(k, @as(usize, @intCast(nrhs)))) catch return error.SizeOverflow);
        }
        fn queryScalar(comptime T: type, count: usize) T {
            const value = common.workspaceQueryValue(S.Real(T), count);
            return if (comptime S.isComplex(T)) .{ .re = value, .im = 0 } else value;
        }
        fn matrix(comptime T: type, values: []T, m: usize, n: usize, ld: usize) Core.lapack.Matrix(T) {
            return .{ .values = values, .row_count = @intCast(m), .column_count = @intCast(n), .leading_dimension = @intCast(ld) };
        }
        fn columnWork(comptime T: type, operation: *const context.Operation, args: Arguments(T), original_a: []const T, original_b: []const T) Error!i32 {
            const invalid = controls(T, args);
            if (invalid != 0) return invalid;
            const minimum = try minimumWork(args.m, args.n, args.nrhs);
            const head = try capacity(T, args.work, 1);
            head[0] = queryScalar(T, minimum);
            if (args.lwork != -1 and (args.lwork < 0 or @as(usize, @intCast(args.lwork)) < minimum)) return -11;
            if (args.lwork == -1) return 0;
            const m: usize = @intCast(args.m);
            const n: usize = @intCast(args.n);
            const nrhs: usize = @intCast(args.nrhs);
            const rows = @max(m, n);
            const lda: usize = @intCast(args.lda);
            const ldb: usize = @intCast(args.ldb);
            const empty = m == 0 or n == 0 or nrhs == 0;
            // A column quick return never admits A. Only WORK(1) and the B
            // elements actually zeroed by the driver participate in admission.
            const a = try capacity(T, args.a, if (empty) 0 else try N.Storage.columnMajorExtent(m, n, lda));
            const b = try capacity(T, args.b, try N.Storage.columnMajorExtent(rows, nrhs, ldb));
            const work = try capacity(T, args.work, if (empty) 1 else @intCast(args.lwork));
            try disjoint(T, a, b);
            try disjoint(T, a, work);
            try disjoint(T, b, work);
            try disjoint(T, original_a, work);
            try disjoint(T, original_b, work);
            try disjoint(T, original_a, b);
            try disjoint(T, original_b, a);
            if (empty) {
                for (0..nrhs) |j| for (0..rows) |i| {
                    b[i + j * ldb] = S.zero(T);
                };
                return 0;
            }
            const required = G.plan(T, m, n, nrhs) catch return error.NumericalOutcomePendingABI;
            var native = context.ContextScratch(T).acquire(operation, required.total_elements) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            defer native.deinit();
            const result = G.solveWithContractAdmitted(operation, T, matrix(T, a, m, n, lda), matrix(T, b, rows, nrhs, ldb), problem(T, args.trans).?, native.lease.values, .lapack_ieee_v1) catch return error.NumericalOutcomePendingABI;
            if (result.failure) |failure| {
                if (failure != .singular) return error.NumericalOutcomePendingABI;
            }
            // Only a successful exit promises the final recommendation. The
            // remaining caller WORK slots are implementation-defined scratch.
            if (result.info == 0) head[0] = queryScalar(T, minimum);
            return std.math.cast(i32, result.info) orelse error.IntegerNotRepresentable;
        }
        pub fn workAdmitted(comptime T: type, operation: *const context.Operation, args: Arguments(T)) Error!i32 {
            _ = operation.borrowedContext();
            if (args.layout != 101 and args.layout != 102) return -1;
            if (args.layout == 102) return columnWork(T, operation, args, &.{}, &.{});
            if (args.lda < args.n) return -7;
            if (args.ldb < args.nrhs) return -9;
            var converted = args;
            converted.layout = 102;
            converted.lda = @max(1, args.m);
            converted.ldb = @max(1, @max(args.m, args.n));
            if (args.lwork == -1) return columnWork(T, operation, converted, &.{}, &.{});
            const ld_a: usize = @intCast(converted.lda);
            const ld_b: usize = @intCast(converted.ldb);
            const a_count = try N.Storage.byteLength(ld_a, @intCast(@max(1, args.n)));
            _ = try N.Storage.byteLength(a_count, @sizeOf(T));
            var row_a = context.ContextScratch(T).acquire(operation, a_count) catch |err| switch (err) {
                error.OutOfMemory => return -1011,
                else => return err,
            };
            defer row_a.deinit();
            const b_count = try N.Storage.byteLength(ld_b, @intCast(@max(1, args.nrhs)));
            _ = try N.Storage.byteLength(b_count, @sizeOf(T));
            var row_b = context.ContextScratch(T).acquire(operation, b_count) catch |err| switch (err) {
                error.OutOfMemory => return -1011,
                else => return err,
            };
            defer row_b.deinit();
            const m: usize = @intCast(@max(0, args.m));
            const n: usize = @intCast(@max(0, args.n));
            const nrhs: usize = @intCast(@max(0, args.nrhs));
            const rows: usize = @intCast(@max(0, @max(args.m, args.n)));
            var original_a: []T = &.{};
            var original_b: []T = &.{};
            var lda: usize = 0;
            var ldb: usize = 0;
            if (m != 0 and n != 0) {
                lda = std.math.cast(usize, args.lda) orelse return error.IntegerNotRepresentable;
                original_a = try capacity(T, args.a, try N.Storage.columnMajorExtent(n, m, lda));
                for (0..m) |i| for (0..n) |j| {
                    row_a.lease.values[i + j * ld_a] = original_a[i * lda + j];
                };
            }
            if (rows != 0 and nrhs != 0) {
                ldb = std.math.cast(usize, args.ldb) orelse return error.IntegerNotRepresentable;
                original_b = try capacity(T, args.b, try N.Storage.columnMajorExtent(nrhs, rows, ldb));
                for (0..rows) |i| for (0..nrhs) |j| {
                    row_b.lease.values[i + j * ld_b] = original_b[i * ldb + j];
                };
            }
            try disjoint(T, original_a, original_b);
            converted.a = operand(T, row_a.lease.values);
            converted.b = operand(T, row_b.lease.values);
            const info = try columnWork(T, operation, converted, original_a, original_b);
            for (0..m) |i| for (0..n) |j| {
                original_a[i * lda + j] = row_a.lease.values[i + j * ld_a];
            };
            for (0..rows) |i| for (0..nrhs) |j| {
                original_b[i * ldb + j] = row_b.lease.values[i + j * ld_b];
            };
            return info;
        }
        pub fn workWithDiagnosticsAdmitted(comptime T: type, operation: *const context.Operation, borrowed: anytype, args: Arguments(T)) Error!i32 {
            const services = callbacks(borrowed);
            const info = try workAdmitted(T, operation, args);
            notify(T, services, args, info);
            return info;
        }
        fn highCompute(comptime T: type, operation: *const context.Operation, services: anytype, args: Arguments(T)) Error!i32 {
            var query: [1]T = undefined;
            var call = args;
            call.work = operand(T, &query);
            call.lwork = -1;
            const info = try workWithDiagnosticsAdmitted(T, operation, services, call);
            if (info != 0) return info;
            const hint = if (comptime S.isComplex(T)) query[0].re else query[0];
            if (!std.math.isFinite(hint) or @as(f128, hint) > @as(f128, std.math.maxInt(i32))) return -1010;
            call.lwork = @intFromFloat(hint);
            var work = try context.ContextScratch(T).acquire(operation, @intCast(call.lwork));
            defer work.deinit();
            call.work = operand(T, work.lease.values);
            const result = try workAdmitted(T, operation, call);
            if (result != -1010) notify(T, services, call, result);
            return result;
        }
        pub fn highAdmitted(comptime compile_nan_scan: bool, comptime T: type, operation: *const context.Operation, borrowed: anytype, args: Arguments(T)) Error!i32 {
            _ = operation.borrowedContext();
            const services = callbacks(borrowed);
            if (args.layout != 101 and args.layout != 102) {
                event(T, services, .lapacke, .high, -1);
                return -1;
            }
            if (comptime compile_nan_scan) if (services.nan_enabled(services.context)) {
                const Provider = struct {
                    pub const Services = @TypeOf(services);
                    pub const Storage = N.Storage;
                    pub const Scalar = S;
                };
                const h = common.HelpersWithAccess(Provider, kind);
                if (try h.scanGeneral(T, args.layout, args.m, args.n, args.lda, @as(common.access.Operand(T, false, kind), args.a))) return -6;
                if (try h.scanGeneral(T, args.layout, @max(args.m, args.n), args.nrhs, args.ldb, @as(common.access.Operand(T, false, kind), args.b))) return -8;
            };
            const result = highCompute(T, operation, services, args) catch |err| switch (err) {
                error.OutOfMemory => @as(i32, -1010),
                else => return err,
            };
            if (result == -1010) event(T, services, .lapacke, .high, result);
            return result;
        }
    };
}
