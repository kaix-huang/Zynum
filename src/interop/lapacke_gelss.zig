// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! LP64 GELSS documented MKL boundary over the configured shared SVD owner.
//! External WORK/RWORK never define internal data or resource ownership.
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
        const G = N.SvdLeastSquares;
        const context = N.Execution.context;
        pub const OwnerIdentity = Core;
        pub const Error = common.access.Fault(kind) || context.AdmissionError || error{ OutOfMemory, NumericalOutcomePendingABI };
        pub fn Arguments(comptime T: type) type {
            if (T != f32 and T != f64 and !S.isComplex(T)) @compileError("GELSS scalar type");
            return struct {
                layout: c_int,
                m: i32,
                n: i32,
                nrhs: i32,
                a: common.access.Operand(T, true, kind) = null,
                lda: i32,
                b: common.access.Operand(T, true, kind) = null,
                ldb: i32,
                singular: common.access.Operand(S.Real(T), true, kind) = null,
                rcond: S.Real(T),
                rank: common.access.Operand(i32, true, kind) = null,
                work: common.access.Operand(T, true, kind) = null,
                lwork: i32 = 0,
                rwork: common.access.Operand(S.Real(T), true, kind) = null,
            };
        }
        fn capacity(comptime T: type, source: common.access.Operand(T, true, kind), length: usize) Error![]T {
            return common.access.capacity(N.Storage, kind, T, true, source, length);
        }
        fn operand(comptime T: type, values: []T) common.access.Operand(T, true, kind) {
            return if (kind == .typed) values else @ptrCast(values.ptr);
        }
        fn span(values: anytype) Error!N.Storage.ByteRange {
            return N.Storage.byteRange(@intFromPtr(values.ptr), try N.Storage.byteLength(values.len, @sizeOf(@TypeOf(values[0]))));
        }
        fn disjoint(a: anytype, b: anytype) Error!void {
            if (N.Storage.overlaps(try span(a), try span(b))) return error.AliasingNotAllowed;
        }
        fn add(a: usize, b: usize) Error!usize {
            return std.math.add(usize, a, b) catch error.SizeOverflow;
        }
        fn count(a: usize, b: usize) Error!usize {
            return N.Storage.byteLength(a, b);
        }
        fn callbacks(borrowed: anytype) @TypeOf(.{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event }) {
            return .{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
        }
        fn event(comptime T: type, services: anytype, channel: common.HookKind, entry: common.Entry, info: i32) void {
            services.error_event(services.context, .{ .kind = channel, .entry = entry, .scalar = if (T == f32) 's' else if (T == f64) 'd' else if (T == S.ComplexF32) 'c' else 'z', .argument = info, .routine = .gelss });
        }
        fn notify(comptime T: type, services: anytype, args: Arguments(T), info: i32) void {
            if (info >= 0) return;
            const row_local = args.layout == 101 and (args.lda < args.n or args.ldb < args.nrhs);
            if (info == -1 or info == -1010 or info == -1011 or row_local) event(T, services, .lapacke, .work, info) else event(T, services, .fortran_returning_test, .fortran_gelss, -info - 1);
        }
        fn controls(args: anytype) i32 {
            if (args.m < 0) return -2;
            if (args.n < 0) return -3;
            if (args.nrhs < 0) return -4;
            if (args.lda < @max(1, args.m)) return -6;
            if (args.ldb < @max(1, @max(args.m, args.n))) return -8;
            return 0;
        }
        /// Documented MKL sufficient minimum; recommendation is own-algorithm selected.
        pub fn minimumWork(comptime T: type, m_input: i32, n_input: i32, rhs_input: i32) Error!usize {
            if (m_input < 0 or n_input < 0 or rhs_input < 0) return error.IntegerNotRepresentable;
            const m: usize = @intCast(m_input);
            const n: usize = @intCast(n_input);
            const rhs: usize = @intCast(rhs_input);
            const k = @min(m, n);
            if (k == 0) return 1;
            return if (S.isComplex(T)) try add(try count(2, k), @max(m, n, rhs)) else try add(try count(3, k), @max(try count(2, k), m, n, rhs));
        }
        fn queryScalar(comptime T: type, length: usize) T {
            const value = common.workspaceQueryValue(S.Real(T), length);
            return if (comptime S.isComplex(T)) .{ .re = value, .im = 0 } else value;
        }
        fn magnitude(comptime T: type, value: T) G.Wide(T) {
            if (comptime S.isComplex(T)) {
                if (std.math.isNan(value.re) or std.math.isNan(value.im)) return std.math.nan(G.Wide(T));
                return std.math.hypot(@as(G.Wide(T), value.re), @as(G.Wide(T), value.im));
            } else return @abs(@as(G.Wide(T), value));
        }
        fn norm(comptime T: type, values: []const T, m: usize, n: usize, ld: usize) G.Wide(T) {
            var largest: G.Wide(T) = 0;
            for (0..n) |j| for (0..m) |i| {
                const value = magnitude(T, values[i + j * ld]);
                if (std.math.isNan(value)) return value;
                largest = @max(largest, value);
            };
            return largest;
        }
        fn scaleFactor(comptime T: type, value: G.Wide(T)) G.Wide(T) {
            const small = @as(G.Wide(T), std.math.floatMin(S.Real(T))) / @as(G.Wide(T), std.math.floatEps(S.Real(T)));
            const large = 1 / small;
            return if (value > 0 and value < small) small / value else if (value > large) large / value else 1;
        }
        fn scaled(comptime T: type, value: T, factor: G.Wide(T)) T {
            return if (comptime S.isComplex(T)) .{ .re = @floatCast(@as(G.Wide(T), value.re) * factor), .im = @floatCast(@as(G.Wide(T), value.im) * factor) } else @floatCast(@as(G.Wide(T), value) * factor);
        }
        fn conjugate(comptime T: type, value: T) T {
            return if (comptime S.isComplex(T)) .{ .re = value.re, .im = -value.im } else value;
        }
        fn matrix(comptime T: type, values: []T, m: usize, n: usize, ld: usize) N.Dense.Matrix(T) {
            return .{ .values = values, .row_count = m, .column_count = n, .leading_dimension = ld };
        }
        fn columnWork(comptime T: type, operation: *const context.Operation, args: Arguments(T), original_a: []const T, original_b: []const T, options: G.Options) Error!i32 {
            const invalid = controls(args);
            if (invalid != 0) return invalid;
            const minimum = try minimumWork(T, args.m, args.n, args.nrhs);
            // Documented MKL short-work behavior: do not publish a recommendation.
            if (args.lwork != -1 and (args.lwork < 0 or @as(usize, @intCast(args.lwork)) < minimum)) return -13;
            if (args.lwork == -1) {
                (try capacity(T, args.work, 1))[0] = queryScalar(T, minimum);
                return 0;
            }
            const m: usize = @intCast(args.m);
            const n: usize = @intCast(args.n);
            const rhs: usize = @intCast(args.nrhs);
            const k = @min(m, n);
            const rows = @max(m, n);
            const lda: usize = @intCast(args.lda);
            const ldb: usize = @intCast(args.ldb);
            const a = try capacity(T, args.a, if (k == 0) 0 else try N.Storage.columnMajorExtent(m, n, lda));
            const b = try capacity(T, args.b, if (k == 0) 0 else try N.Storage.columnMajorExtent(rows, rhs, ldb));
            const singular = try capacity(S.Real(T), args.singular, k);
            const rank = try capacity(i32, args.rank, 1);
            const work = try capacity(T, args.work, if (k == 0) 1 else @intCast(args.lwork));
            const operands = .{ a, b, singular, rank, work };
            inline for (operands, 0..) |one, i| inline for (operands, 0..) |other, j| {
                if (comptime j > i) try disjoint(one, other);
            };
            inline for (operands) |one| {
                try disjoint(original_a, one);
                try disjoint(original_b, one);
            }
            if (k == 0) {
                work[0] = queryScalar(T, minimum);
                rank[0] = 0;
                return 0;
            }
            const anorm = norm(T, a, m, n, lda);
            if (anorm == 0) {
                for (0..rhs) |j| @memset(b[j * ldb ..][0..rows], S.zero(T));
                @memset(singular, 0);
                rank[0] = 0;
                work[0] = queryScalar(T, minimum);
                return 0;
            }
            if (comptime S.isComplex(T)) {
                const real = try capacity(S.Real(T), args.rwork, try count(5, k));
                inline for (operands) |one| try disjoint(real, one);
                try disjoint(original_a, real);
                try disjoint(original_b, real);
            }
            work[0] = queryScalar(T, minimum);
            const acount = try count(m, n);
            const bcount = try count(m, rhs);
            const xcount = try count(n, rhs);
            const vcount = try count(n, k);
            const rcount = try count(m -| n, rhs);
            const total = try add(try add(acount, bcount), try add(try add(xcount, vcount), rcount));
            _ = try N.Storage.byteLength(total, @sizeOf(T));
            var scratch = context.ContextScratch(T).acquire(operation, total) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            defer scratch.deinit();
            const values = scratch.lease.values;
            const at = values[0..acount];
            const bt = values[acount..][0..bcount];
            const x = values[acount + bcount ..][0..xcount];
            const v = values[acount + bcount + xcount ..][0..vcount];
            const residual = values[acount + bcount + xcount + vcount ..][0..rcount];
            const afactor = scaleFactor(T, anorm);
            const bfactor = scaleFactor(T, norm(T, b, m, rhs, ldb));
            for (0..n) |j| for (0..m) |i| {
                at[i + j * m] = scaled(T, a[i + j * lda], afactor);
            };
            for (0..rhs) |j| for (0..m) |i| {
                bt[i + j * m] = scaled(T, b[i + j * ldb], bfactor);
            };
            const rcond: G.Wide(T) = if (args.rcond < 0) std.math.floatEps(S.Real(T)) else args.rcond;
            const result = G.solveAdmitted(operation, T, matrix(T, at, m, n, m).asConst(), matrix(T, bt, m, rhs, m).asConst(), .{ .solution = matrix(T, x, n, rhs, n), .singular = singular, .right_vectors = matrix(T, v, n, k, n), .residual = matrix(T, residual, m -| n, rhs, m -| n) }, .{ .relative = rcond, .absolute = std.math.floatMin(S.Real(T)) }, options) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                error.CounterOverflow => return error.CounterOverflow,
                else => return error.NumericalOutcomePendingABI,
            };
            if (result.status == .no_convergence) return std.math.cast(i32, result.unconverged) orelse error.IntegerNotRepresentable;
            if (result.status != .converged) return error.NumericalOutcomePendingABI;
            for (0..n) |j| for (0..k) |i| {
                a[i + j * lda] = conjugate(T, v[j + i * n]);
            };
            for (0..rhs) |j| {
                for (0..n) |i| b[i + j * ldb] = scaled(T, x[i + j * n], afactor / bfactor);
                if (m > n and result.rank == n) for (0..m - n) |i| {
                    b[n + i + j * ldb] = residual[i + j * (m - n)];
                };
            }
            for (singular) |*sigma| sigma.* = @floatCast(@as(G.Wide(T), sigma.*) / afactor);
            rank[0] = @intCast(result.rank);
            work[0] = queryScalar(T, minimum);
            return 0;
        }
        pub fn workAdmitted(comptime T: type, operation: *const context.Operation, args: Arguments(T)) Error!i32 {
            return workWithOptionsAdmitted(T, operation, args, .{ .numerical_policy = .ieee });
        }
        pub fn workWithOptionsAdmitted(comptime T: type, operation: *const context.Operation, args: Arguments(T), options: G.Options) Error!i32 {
            _ = operation.borrowedContext();
            if (args.layout != 101 and args.layout != 102) return -1;
            if (args.layout == 102) return columnWork(T, operation, args, &.{}, &.{}, options);
            if (args.lda < args.n) return -6;
            if (args.ldb < args.nrhs) return -8;
            var converted = args;
            converted.layout = 102;
            converted.lda = @max(1, args.m);
            converted.ldb = @max(1, @max(args.m, args.n));
            if (args.lwork == -1) return columnWork(T, operation, converted, &.{}, &.{}, options);
            const lda_t: usize = @intCast(converted.lda);
            const ldb_t: usize = @intCast(converted.ldb);
            const acount = try count(lda_t, @intCast(@max(1, args.n)));
            const bcount = try count(ldb_t, @intCast(@max(1, args.nrhs)));
            _ = try N.Storage.byteLength(acount, @sizeOf(T));
            _ = try N.Storage.byteLength(bcount, @sizeOf(T));
            var at = context.ContextScratch(T).acquire(operation, acount) catch |err| switch (err) {
                error.OutOfMemory => return -1011,
                else => return err,
            };
            defer at.deinit();
            var bt = context.ContextScratch(T).acquire(operation, bcount) catch |err| switch (err) {
                error.OutOfMemory => return -1011,
                else => return err,
            };
            defer bt.deinit();
            const m: usize = @intCast(@max(0, args.m));
            const n: usize = @intCast(@max(0, args.n));
            const rhs: usize = @intCast(@max(0, args.nrhs));
            const rows = @max(m, n);
            const lda: usize = @intCast(@max(0, args.lda));
            const ldb: usize = @intCast(@max(0, args.ldb));
            const original_a = try capacity(T, args.a, try N.Storage.columnMajorExtent(n, m, lda));
            const original_b = try capacity(T, args.b, try N.Storage.columnMajorExtent(rhs, rows, ldb));
            try disjoint(original_a, original_b);
            for (0..m) |i| for (0..n) |j| {
                at.lease.values[i + j * lda_t] = original_a[i * lda + j];
            };
            for (0..rows) |i| for (0..rhs) |j| {
                bt.lease.values[i + j * ldb_t] = original_b[i * ldb + j];
            };
            converted.a = operand(T, at.lease.values);
            converted.b = operand(T, bt.lease.values);
            const info = try columnWork(T, operation, converted, original_a, original_b, options);
            for (0..m) |i| for (0..n) |j| {
                original_a[i * lda + j] = at.lease.values[i + j * lda_t];
            };
            for (0..rows) |i| for (0..rhs) |j| {
                original_b[i * ldb + j] = bt.lease.values[i + j * ldb_t];
            };
            return info;
        }
        pub fn workWithDiagnosticsAdmitted(comptime T: type, operation: *const context.Operation, borrowed: anytype, args: Arguments(T)) Error!i32 {
            const info = try workAdmitted(T, operation, args);
            notify(T, callbacks(borrowed), args, info);
            return info;
        }
        fn highCompute(comptime T: type, operation: *const context.Operation, services: anytype, args: Arguments(T)) Error!i32 {
            const R = S.Real(T);
            var real: ?context.ContextScratch(R) = null;
            defer if (real) |*lease| lease.deinit();
            var call = args;
            if (comptime S.isComplex(T)) {
                const k: usize = @intCast(@max(0, @min(args.m, args.n)));
                real = try context.ContextScratch(R).acquire(operation, @max(1, try count(5, k)));
                call.rwork = operand(R, real.?.lease.values);
            }
            var query: [1]T = undefined;
            call.work = operand(T, &query);
            call.lwork = -1;
            const queried = try workWithDiagnosticsAdmitted(T, operation, services, call);
            if (queried != 0) return queried;
            const hint = if (comptime S.isComplex(T)) query[0].re else query[0];
            if (!std.math.isFinite(hint) or @as(f128, hint) > @as(f128, std.math.maxInt(i32))) return -1010;
            call.lwork = @intFromFloat(hint);
            var work = try context.ContextScratch(T).acquire(operation, @intCast(call.lwork));
            defer work.deinit();
            call.work = operand(T, work.lease.values);
            const info = try workAdmitted(T, operation, call);
            if (info != -1010) notify(T, services, call, info);
            return info;
        }
        pub fn highAdmitted(comptime scan: bool, comptime T: type, operation: *const context.Operation, borrowed: anytype, args: Arguments(T)) Error!i32 {
            _ = operation.borrowedContext();
            const services = callbacks(borrowed);
            if (args.layout != 101 and args.layout != 102) {
                event(T, services, .lapacke, .high, -1);
                return -1;
            }
            if (comptime scan) if (services.nan_enabled(services.context)) {
                const Provider = struct {
                    pub const Services = @TypeOf(services);
                    pub const Storage = N.Storage;
                    pub const Scalar = S;
                };
                const H = common.HelpersWithAccess(Provider, kind);
                if (try H.scanGeneral(T, args.layout, args.m, args.n, args.lda, @as(common.access.Operand(T, false, kind), args.a))) return -5;
                if (try H.scanGeneral(T, args.layout, @max(args.m, args.n), args.nrhs, args.ldb, @as(common.access.Operand(T, false, kind), args.b))) return -7;
                if (std.math.isNan(args.rcond)) return -10;
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
