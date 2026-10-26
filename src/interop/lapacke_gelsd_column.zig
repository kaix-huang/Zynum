// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal column-major GELSD work stage. Numerical faults remain explicit;
//! this is not a standard export until the complete IEEE/interface policy is wired.
const common = @import("lapacke_common.zig");
const policy = @import("lapacke_gelsd_policy.zig");
const workspace = @import("lapacke_gelsd_workspace.zig");
const query = @import("lapacke_gelsd_query.zig");
pub fn ColumnWork(comptime Core: type, comptime kind: common.access.Kind) type {
    return struct {
        const H = common.HelpersWithAccess(Core, kind);
        const Q = query.Query(Core, kind);
        const Planner = workspace.Planner(Core);
        const Fault = common.access.Fault(kind);
        pub const Services = Core.Services;
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub const NumericalFault = Core.RankLeastSquares.Error || error{ NonFiniteResult, RangeLimit };
        pub fn Arguments(comptime T: type, comptime I: type) type {
            return struct {
                m: I,
                n: I,
                nrhs: I,
                lda: I,
                ldb: I,
                a: common.access.Operand(T, true, kind),
                b: common.access.Operand(T, true, kind),
                singular: common.access.Operand(T, true, kind),
                rank: common.access.Operand(I, true, kind),
                rcond: T,
                work: common.access.Operand(T, true, kind),
                lwork: I,
                iwork: common.access.Operand(I, true, kind),
            };
        }
        pub fn Result(comptime I: type) type {
            return union(enum) { lapacke_info: I, admission_fault: Fault, numerical_fault: NumericalFault };
        }
        pub fn call(comptime T: type, comptime I: type, borrowed: *const Services, args: Arguments(T, I), options: Core.RankLeastSquares.Options) Result(I) {
            const services = borrowed.*;
            const queried = Q.call(T, I, &services, .{ .layout = 102, .m = args.m, .n = args.n, .nrhs = args.nrhs, .lda = args.lda, .ldb = args.ldb, .work = args.work, .iwork = args.iwork });
            switch (queried) {
                .admission_fault => |err| return .{ .admission_fault = err },
                .lapacke_info => |info| if (info != 0 or args.lwork == -1) return .{ .lapacke_info = info },
            }
            const m = H.native(args.m) catch |err| return .{ .admission_fault = err };
            const n = H.native(args.n) catch |err| return .{ .admission_fault = err };
            const nrhs = H.native(args.nrhs) catch |err| return .{ .admission_fault = err };
            const plan = Planner.make(T, I, m, n, nrhs) catch return .{ .admission_fault = error.SizeOverflow };
            if (args.lwork < plan.work) {
                H.hook(T, services, .fortran_returning_test, .fortran_gelsd, 12, .gelsd);
                return .{ .lapacke_info = -13 };
            }
            const rank = H.capacity(I, args.rank, 1) catch |err| return .{ .admission_fault = err };
            if (m == 0 or n == 0) {
                rank[0] = 0;
                return .{ .lapacke_info = 0 };
            }
            const lda = H.native(args.lda) catch |err| return .{ .admission_fault = err };
            const ldb = H.native(args.ldb) catch |err| return .{ .admission_fault = err };
            const acount = Core.Storage.columnMajorExtent(m, n, lda) catch |err| return .{ .admission_fault = err };
            const bcount = Core.Storage.columnMajorExtent(@max(m, n), nrhs, ldb) catch |err| return .{ .admission_fault = err };
            const a = H.capacity(T, args.a, acount) catch |err| return .{ .admission_fault = err };
            const b = H.capacity(T, args.b, bcount) catch |err| return .{ .admission_fault = err };
            const singular = H.capacity(T, args.singular, @min(m, n)) catch |err| return .{ .admission_fault = err };
            const work = H.capacity(T, args.work, @intCast(plan.work)) catch |err| return .{ .admission_fault = err };
            const iwork = H.capacity(I, args.iwork, @intCast(plan.iwork)) catch |err| return .{ .admission_fault = err };
            const ranges = .{ a, b, singular, work, rank, iwork };
            inline for (ranges, 0..) |one, index| {
                const bytes = Core.Storage.byteLength(one.len, @sizeOf(@TypeOf(one[0]))) catch |err| return .{ .admission_fault = err };
                const first = Core.Storage.byteRange(@intFromPtr(one.ptr), bytes) catch |err| return .{ .admission_fault = err };
                inline for (ranges, 0..) |other, other_index| {
                    if (comptime other_index <= index) continue;
                    const other_bytes = Core.Storage.byteLength(other.len, @sizeOf(@TypeOf(other[0]))) catch |err| return .{ .admission_fault = err };
                    const second = Core.Storage.byteRange(@intFromPtr(other.ptr), other_bytes) catch |err| return .{ .admission_fault = err };
                    if (Core.Storage.overlaps(first, second)) return .{ .admission_fault = error.AliasingNotAllowed };
                }
            }
            var all_zero = true;
            for (0..n) |j| for (0..m) |i| {
                all_zero = all_zero and a[i + j * lda] == 0;
            };
            if (all_zero) {
                for (0..nrhs) |j| @memset(b[j * ldb ..][0..@max(m, n)], 0);
                @memset(singular, 0);
                rank[0] = 0;
                return .{ .lapacke_info = 0 };
            }
            const buffers = Planner.split(T, I, plan, work) catch |err| return .{ .admission_fault = switch (err) {
                error.SizeOverflow => error.SizeOverflow,
                error.BufferTooSmall => unreachable,
            } };
            var numerical_options = options;
            numerical_options.output_policy = .ieee;
            numerical_options.rhs_policy = .ieee;
            numerical_options.numerical_policy = .ieee;
            numerical_options.cutoff_policy = .unordered;
            const result = Core.RankLeastSquares.solveWithResidual(T, .{ .values = a, .row_count = m, .column_count = n, .leading_dimension = lda }, .{ .values = b, .row_count = m, .column_count = nrhs, .leading_dimension = ldb }, .{ .values = buffers.solution, .row_count = n, .column_count = nrhs, .leading_dimension = n }, singular, .{ .values = buffers.residual, .row_count = m -| n, .column_count = nrhs, .leading_dimension = m -| n }, policy.effectiveRcond(T, @min(m, n), args.rcond), buffers.scalar, buffers.wide, numerical_options) catch |err| return .{ .numerical_fault = err };
            if (result.status == .no_convergence) return .{ .lapacke_info = @intCast(result.unconverged) };
            if (result.status != .converged) return .{ .numerical_fault = if (result.status == .range_limit) error.RangeLimit else error.NonFiniteResult };
            for (0..nrhs) |j| {
                @memcpy(b[j * ldb ..][0..n], buffers.solution[j * n ..][0..n]);
                if (m > n and result.rank == n) @memcpy(b[j * ldb + n ..][0 .. m - n], buffers.residual[j * (m - n) ..][0 .. m - n]);
            }
            rank[0] = @intCast(result.rank);
            return .{ .lapacke_info = 0 };
        }
        pub fn callManaged(comptime T: type, comptime I: type, context: *Core.Execution.context.Context, borrowed: *const Services, args: Arguments(T, I), options: Core.RankLeastSquares.Options) Core.Execution.context.AdmissionError!Result(I) {
            var invocation = try common.ManagedInvocation(Core).begin(context, borrowed);
            defer invocation.operation.end();
            return call(T, I, &invocation.services, args, options);
        }
    };
}
