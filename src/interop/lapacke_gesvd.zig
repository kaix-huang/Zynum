// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Four-scalar GESVD staging. Only the injected shared Core owns mathematics,
//! operation admission and scratch. No standard symbol or independent state.
const std = @import("std");
const common = @import("lapacke_common.zig");
pub const policy = @import("lapacke_gesvd_policy.zig");
pub fn Adapter(comptime Core: type, comptime kind: common.access.Kind) type {
    return struct {
        const H = common.HelpersWithAccess(Core, kind);
        const E = Core.Execution;
        const Fault = common.access.Fault(kind);
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub const NumericalFault = Core.Svd.Error || Core.BidiagonalSvd.Error || Core.ComplexBidiagonalSvd.Error || error{ NoConvergence, RangeLimit, NonFiniteResult };
        pub const Result = union(enum) { lapacke_info: i32, admission_fault: Fault, numerical_fault: NumericalFault };
        pub fn Arguments(comptime T: type) type {
            H.scalar(T);
            const R = Core.Scalar.Real(T);
            return struct {
                layout: c_int,
                jobu: u8,
                jobvt: u8,
                m: i32,
                n: i32,
                lda: i32,
                ldu: i32,
                ldvt: i32,
                a: common.access.Operand(T, true, kind) = null,
                singular: common.access.Operand(R, true, kind) = null,
                u: common.access.Operand(T, true, kind) = null,
                vt: common.access.Operand(T, true, kind) = null,
                work: common.access.Operand(T, true, kind) = null,
                lwork: i32 = 0,
                rwork: common.access.Operand(R, true, kind) = null,
                superb: common.access.Operand(R, true, kind) = null,
            };
        }
        fn implementation(comptime T: type) type {
            return if (comptime Core.Scalar.isComplex(T)) Core.ComplexBidiagonalSvd else Core.BidiagonalSvd;
        }
        fn Wide(comptime T: type) type {
            return if (comptime Core.Scalar.isComplex(T)) Core.ComplexBidiagonalSvd.Wide(T) else Core.BidiagonalSvd.WorkspaceScalar(T);
        }
        fn count(comptime T: type, n: usize) T {
            const R = Core.Scalar.Real(T);
            var r: R = @floatFromInt(n);
            if (@as(f128, r) < @as(f128, @floatFromInt(n))) r = std.math.nextAfter(R, r, std.math.inf(R));
            return if (comptime Core.Scalar.isComplex(T)) .{ .re = r, .im = 0 } else r;
        }
        fn hook(comptime T: type, services: *const Core.Services, entry: common.Entry, info: i32) void {
            H.hook(T, services.*, if (entry == .fortran_gesvd) .fortran_returning_test else .lapacke, entry, info, .gesvd);
        }
        fn shapeSpan(a: anytype, rows: usize, cols: usize, ld: i32) Fault!usize {
            if (rows == 0 or cols == 0) return 0;
            return Core.Storage.columnMajorExtent(if (a.layout == 101) cols else rows, if (a.layout == 101) rows else cols, try H.native(ld));
        }
        fn index(row: bool, i: usize, j: usize, ld: usize) usize {
            return if (row) i * ld + j else i + j * ld;
        }
        fn conjugate(comptime T: type, z: T) T {
            return if (comptime Core.Scalar.isComplex(T)) .{ .re = z.re, .im = -z.im } else z;
        }
        fn nativeValue(comptime T: type, value: implementation(T).WorkspaceScalar(T)) T {
            return if (comptime Core.Scalar.isComplex(T)) .{ .re = @floatCast(value.re), .im = @floatCast(value.im) } else @floatCast(value);
        }
        fn realValue(comptime T: type, value: implementation(T).WorkspaceScalar(T)) Core.Scalar.Real(T) {
            return @floatCast(if (comptime Core.Scalar.isComplex(T)) value.re else value);
        }
        fn memoryInfo(comptime T: type, services: *const Core.Services, entry: common.Entry, row: bool) Result {
            const code: i32 = if (entry == .high) -1010 else if (row) -1011 else -1010;
            hook(T, services, entry, code);
            return .{ .lapacke_info = code };
        }
        pub fn workAdmitted(comptime T: type, op: *const E.context.Operation, services: *const Core.Services, a: Arguments(T), options: Core.Svd.Options) E.context.AdmissionError!Result {
            _ = op.borrowedContext();
            const R = Core.Scalar.Real(T);
            const Impl = implementation(T);
            const row = a.layout == 101;
            if (a.layout != 101 and a.layout != 102) {
                hook(T, services, .work, -1);
                return .{ .lapacke_info = -1 };
            }
            const info = policy.scalarInfo(a);
            if (info != 0) {
                if (row and (info == -7 or info == -10 or info == -12)) hook(T, services, .work, info) else hook(T, services, .fortran_gesvd, -info - 1);
                return .{ .lapacke_info = info };
            }
            const m: usize = @intCast(a.m);
            const n: usize = @intCast(a.n);
            const k = @min(m, n);
            const uj = policy.job(a.jobu).?;
            const vj = policy.job(a.jobvt).?;
            const minimum = policy.minimumWork(Core.Scalar.isComplex(T), m, n, uj, vj) catch |err| return .{ .admission_fault = err };
            const minimum_i = std.math.cast(i32, minimum) orelse return .{ .admission_fault = error.SizeOverflow };
            const query = a.lwork == -1;
            const short = !query and a.lwork < minimum_i;
            const w = H.capacity(T, a.work, if (query or short) 1 else @intCast(a.lwork)) catch |err| return .{ .admission_fault = err };
            if (query or short) {
                w[0] = count(T, minimum);
                if (query) return .{ .lapacke_info = 0 };
                hook(T, services, .fortran_gesvd, 13);
                return .{ .lapacke_info = -14 };
            }
            if (k == 0) {
                w[0] = count(T, minimum);
                return .{ .lapacke_info = 0 };
            }
            const uc: usize = if (uj == .all) m else if (uj == .some) k else 0;
            const vr: usize = if (vj == .all) n else if (vj == .some) k else 0;
            const av = H.capacity(T, a.a, shapeSpan(a, m, n, a.lda) catch |err| return .{ .admission_fault = err }) catch |err| return .{ .admission_fault = err };
            const sv = H.capacity(R, a.singular, k) catch |err| return .{ .admission_fault = err };
            const uv = H.capacity(T, a.u, shapeSpan(a, m, uc, a.ldu) catch |err| return .{ .admission_fault = err }) catch |err| return .{ .admission_fault = err };
            const vv = H.capacity(T, a.vt, shapeSpan(a, vr, n, a.ldvt) catch |err| return .{ .admission_fault = err }) catch |err| return .{ .admission_fault = err };
            const rw = H.capacity(R, a.rwork, if (comptime Core.Scalar.isComplex(T)) std.math.mul(usize, 5, k) catch return .{ .admission_fault = error.SizeOverflow } else 0) catch |err| return .{ .admission_fault = err };
            const operands = .{ av, sv, uv, vv, w, rw };
            inline for (operands, 0..) |one, i| inline for (operands, 0..) |other, j| {
                if (comptime j > i) H.disjointMixed(@TypeOf(one[0]), one, @TypeOf(other[0]), other) catch |err| return .{ .admission_fault = err };
            };
            const acount = std.math.mul(usize, m, if (uj == .all and m > k) m else n) catch return .{ .admission_fault = error.SizeOverflow };
            const vcols = if (vj == .all) n else k;
            const vcount = std.math.mul(usize, n, vcols) catch return .{ .admission_fault = error.SizeOverflow };
            const complex_plan = if (comptime Core.Scalar.isComplex(T)) Impl.plan(T, m, n) catch return .{ .admission_fault = error.SizeOverflow } else {};
            const wc = if (comptime Core.Scalar.isComplex(T)) complex_plan.complex_elements else Impl.workspaceLength(T, m, n) catch return .{ .admission_fault = error.SizeOverflow };
            const tc: usize = if (comptime Core.Scalar.isComplex(T)) complex_plan.real_elements else 0;
            const max_steps = std.math.mul(usize, options.max_sweeps, k) catch return .{ .admission_fault = error.SizeOverflow };
            var aa = E.context.ContextScratch(T).acquire(op, acount) catch |err| {
                if (err == error.CounterOverflow) return error.CounterOverflow;
                return memoryInfo(T, services, .work, row);
            };
            defer aa.deinit();
            var vs = E.context.ContextScratch(T).acquire(op, vcount) catch |err| {
                if (err == error.CounterOverflow) return error.CounterOverflow;
                return memoryInfo(T, services, .work, row);
            };
            defer vs.deinit();
            var ss = E.context.ContextScratch(R).acquire(op, k) catch |err| {
                if (err == error.CounterOverflow) return error.CounterOverflow;
                return memoryInfo(T, services, .work, row);
            };
            defer ss.deinit();
            var ws = E.context.ContextScratch(Impl.WorkspaceScalar(T)).acquire(op, wc) catch |err| {
                if (err == error.CounterOverflow) return error.CounterOverflow;
                return memoryInfo(T, services, .work, row);
            };
            defer ws.deinit();
            var ts = E.context.ContextScratch(Wide(T)).acquire(op, tc) catch |err| {
                if (err == error.CounterOverflow) return error.CounterOverflow;
                return memoryInfo(T, services, .work, row);
            };
            defer ts.deinit();
            const adata = aa.lease.values;
            const vdata = vs.lease.values;
            const sdata = ss.lease.values;
            const lda: usize = @intCast(a.lda);
            for (0..n) |j| for (0..m) |i| {
                adata[i + j * m] = av[index(row, i, j, lda)];
            };
            w[0] = count(T, minimum);
            const numerical_options: Impl.Options = .{ .max_steps = max_steps, .relative_tolerance = options.relative_tolerance };
            const numerical = if (comptime Core.Scalar.isComplex(T))
                Impl.decompose(T, .{ .values = adata, .row_count = m, .column_count = n, .leading_dimension = m }, sdata, .{ .values = vdata, .row_count = n, .column_count = k, .leading_dimension = n }, ws.lease.values, ts.lease.values, numerical_options) catch |err| return .{ .numerical_fault = err }
            else
                Impl.decompose(T, .{ .values = adata, .row_count = m, .column_count = n, .leading_dimension = m }, sdata, .{ .values = vdata, .row_count = n, .column_count = k, .leading_dimension = n }, ws.lease.values, numerical_options) catch |err| return .{ .numerical_fault = err };
            var partial_state: ?Impl.Factors(T) = null;
            var partial_info: i32 = 0;
            if (numerical.status == .no_convergence) {
                var state = Impl.factors(T, m, n, ws.lease.values) catch |err| return .{ .numerical_fault = err };
                const remaining = Impl.preparePartialUpper(T, &state) catch |err| return .{ .numerical_fault = err };
                partial_info = std.math.cast(i32, remaining) orelse return .{ .admission_fault = error.SizeOverflow };
                const ucols: usize = if (uj == .all) m else if (uj == .none) 0 else k;
                const rcols: usize = if (vj == .all) n else if (vj == .none) 0 else k;
                Impl.checkPartialNative(T, state, ucols, rcols, options.relative_tolerance orelse 32 * @as(f64, std.math.floatEps(R))) catch |err| return .{ .numerical_fault = err };
                // Conversion stays in the existing owned staging leases. All
                // finite/range/rounded-unitarity checks precede public writes.
                for (0..k) |i| sdata[i] = realValue(T, state.b_tall.values[i + i * state.b_tall.leading_dimension]);
                for (0..ucols) |j| for (0..m) |i| {
                    adata[i + j * m] = nativeValue(T, state.left.values[i + j * state.left.leading_dimension]);
                };
                for (0..rcols) |j| for (0..n) |i| {
                    vdata[i + j * n] = nativeValue(T, state.right.values[i + j * state.right.leading_dimension]);
                };
                partial_state = state;
            } else if (numerical.status != .converged) return .{ .numerical_fault = switch (numerical.status) {
                .no_convergence => unreachable,
                .non_finite_result => error.NonFiniteResult,
                .range_limit => error.RangeLimit,
                .converged => unreachable,
            } };
            var basis_work: ?E.context.ContextScratch(Core.OrthonormalCompletion.WorkspaceScalar(T)) = null;
            defer if (basis_work) |*lease| lease.deinit();
            const udata = adata;
            const dimension = if (uj == .all and m > k) m else if (vj == .all and n > k) n else 0;
            if (partial_state == null and dimension > 0) {
                const bc = Core.OrthonormalCompletion.workspaceLength(T, dimension) catch return .{ .admission_fault = error.SizeOverflow };
                basis_work = E.context.ContextScratch(Core.OrthonormalCompletion.WorkspaceScalar(T)).acquire(op, bc) catch |err| {
                    if (err == error.CounterOverflow) return error.CounterOverflow;
                    return memoryInfo(T, services, .work, row);
                };
                const basis = if (uj == .all and m > k) udata else vdata;
                const completed = Core.OrthonormalCompletion.complete(T, .{ .values = basis, .row_count = dimension, .column_count = dimension, .leading_dimension = dimension }, k, basis_work.?.lease.values, .{ .relative_tolerance = options.relative_tolerance }) catch |err| return .{ .numerical_fault = err };
                if (completed.status != .converged) return .{ .numerical_fault = error.RangeLimit };
            }
            @memcpy(sv, sdata);
            if (partial_state) |state| for (0..k - 1) |i| {
                const edge = realValue(T, state.b_tall.values[i + (i + 1) * state.b_tall.leading_dimension]);
                if (comptime Core.Scalar.isComplex(T)) rw[i] = edge else w[i + 1] = edge;
            };
            if (uc > 0) {
                const ld: usize = @intCast(a.ldu);
                for (0..uc) |j| for (0..m) |i| {
                    uv[index(row, i, j, ld)] = udata[i + j * m];
                };
            }
            if (vr > 0) {
                const ld: usize = @intCast(a.ldvt);
                for (0..n) |j| for (0..vr) |i| {
                    vv[index(row, i, j, ld)] = conjugate(T, vdata[j + i * n]);
                };
            }
            if (uj == .overwrite) for (0..k) |j| {
                for (0..m) |i| av[index(row, i, j, lda)] = adata[i + j * m];
            };
            if (vj == .overwrite) for (0..n) |j| {
                for (0..k) |i| av[index(row, i, j, lda)] = conjugate(T, vdata[j + i * n]);
            };
            return .{ .lapacke_info = partial_info };
        }
        pub fn highAdmitted(comptime scan: bool, comptime T: type, op: *const E.context.Operation, services: *const Core.Services, a: Arguments(T), options: Core.Svd.Options) E.context.AdmissionError!Result {
            _ = op.borrowedContext();
            if (a.layout != 101 and a.layout != 102) {
                hook(T, services, .high, -1);
                return .{ .lapacke_info = -1 };
            }
            if (scan and services.nan_enabled(services.context)) if (H.scanGeneral(T, a.layout, a.m, a.n, a.lda, a.a) catch |err| return .{ .admission_fault = err }) return .{ .lapacke_info = -6 };
            var slot: [1]T = undefined;
            var qa = a;
            qa.lwork = -1;
            qa.work = if (kind == .typed) &slot else @ptrCast(&slot);
            const query = try workAdmitted(T, op, services, qa, options);
            if (query != .lapacke_info or query.lapacke_info != 0) return query;
            const m: usize = @intCast(a.m);
            const n: usize = @intCast(a.n);
            const k = @min(m, n);
            const count_w = policy.minimumWork(Core.Scalar.isComplex(T), m, n, policy.job(a.jobu).?, policy.job(a.jobvt).?) catch return .{ .admission_fault = error.SizeOverflow };
            const R = Core.Scalar.Real(T);
            const superb = H.capacity(R, a.superb, k -| 1) catch |err| return .{ .admission_fault = err };
            const av = H.capacity(T, a.a, if (k == 0) 0 else shapeSpan(a, m, n, a.lda) catch |err| return .{ .admission_fault = err }) catch |err| return .{ .admission_fault = err };
            const uc: usize = if (policy.job(a.jobu) == .all) m else if (policy.job(a.jobu) == .some) k else 0;
            const vr: usize = if (policy.job(a.jobvt) == .all) n else if (policy.job(a.jobvt) == .some) k else 0;
            const sv = H.capacity(R, a.singular, k) catch |err| return .{ .admission_fault = err };
            const uv = H.capacity(T, a.u, if (k == 0) 0 else shapeSpan(a, m, uc, a.ldu) catch |err| return .{ .admission_fault = err }) catch |err| return .{ .admission_fault = err };
            const vv = H.capacity(T, a.vt, if (k == 0) 0 else shapeSpan(a, vr, n, a.ldvt) catch |err| return .{ .admission_fault = err }) catch |err| return .{ .admission_fault = err };
            inline for (.{ av, sv, uv, vv }) |operand| H.disjointMixed(R, superb, @TypeOf(operand[0]), operand) catch |err| return .{ .admission_fault = err };
            var work = E.context.ContextScratch(T).acquire(op, count_w) catch |err| {
                if (err == error.CounterOverflow) return error.CounterOverflow;
                return memoryInfo(T, services, .high, a.layout == 101);
            };
            defer work.deinit();
            var real: ?E.context.ContextScratch(R) = null;
            defer if (real) |*lease| lease.deinit();
            if (comptime Core.Scalar.isComplex(T)) real = E.context.ContextScratch(R).acquire(op, std.math.mul(usize, 5, k) catch return .{ .admission_fault = error.SizeOverflow }) catch |err| {
                if (err == error.CounterOverflow) return error.CounterOverflow;
                return memoryInfo(T, services, .high, a.layout == 101);
            };
            var wa = a;
            wa.work = if (kind == .typed) work.lease.values else @ptrCast(work.lease.values.ptr);
            wa.lwork = @intCast(count_w);
            if (comptime Core.Scalar.isComplex(T)) wa.rwork = if (kind == .typed) real.?.lease.values else @ptrCast(real.?.lease.values.ptr);
            const result = try workAdmitted(T, op, services, wa, options);
            if (result == .lapacke_info and result.lapacke_info == 0) @memset(superb, 0);
            if (result == .lapacke_info and result.lapacke_info > 0) for (superb, 0..) |*edge, i| {
                edge.* = if (comptime Core.Scalar.isComplex(T)) real.?.lease.values[i] else work.lease.values[i + 1];
            };
            return result;
        }
        pub fn callManaged(comptime scan: bool, comptime T: type, level: common.Level, ctx: *E.context.Context, services: *const Core.Services, a: Arguments(T), options: Core.Svd.Options) E.context.AdmissionError!Result {
            var invocation = try common.ManagedInvocation(Core).begin(ctx, services);
            defer invocation.operation.end();
            return if (level == .high) highAdmitted(scan, T, &invocation.operation, &invocation.services, a, options) else workAdmitted(T, &invocation.operation, &invocation.services, a, options);
        }
    };
}
