// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Borrowed finite column-major complex GELSD engine. No standard C export.
//! External workspace/rcond policy is separate from the accepted finite Core.
const common = @import("lapacke_common.zig");
const std = @import("std");
const policy = @import("lapacke_complex_gelsd_policy.zig");
const workspace = @import("lapacke_complex_gelsd_workspace.zig");
pub fn ColumnWork(comptime Core: type, comptime kind: common.access.Kind) type {
    return struct {
        const H = common.HelpersWithAccess(Core, kind);
        const Rank = Core.ComplexRankLeastSquares;
        const Math = Core.SvdLeastSquares;
        const Planner = workspace.Planner(Core);
        const Fault = common.access.Fault(kind);
        pub const Services = Core.Services;
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub const NumericalFault = Rank.Error || Math.Error || error{ NonFiniteResult, RangeLimit };
        fn scaled(comptime C: type, value: C, factor: Math.Wide(C)) C {
            return .{ .re = @floatCast(@as(Math.Wide(C), value.re) * factor), .im = @floatCast(@as(Math.Wide(C), value.im) * factor) };
        }
        fn rightHandSideScale(comptime C: type, values: []const C, m: usize, rhs: usize, ld: usize) Math.Wide(C) {
            var largest: Math.Wide(C) = 0;
            for (0..rhs) |j| for (0..m) |i| {
                const value = values[i + j * ld];
                // Leave nonfinite classification and its priority to Math.
                if (!std.math.isFinite(value.re) or !std.math.isFinite(value.im)) return 1;
                largest = @max(largest, std.math.hypot(@as(Math.Wide(C), value.re), @as(Math.Wide(C), value.im)));
            };
            return policy.rightHandSideScale(Core.Scalar.Real(C), Math.Wide(C), largest);
        }
        pub fn Arguments(comptime C: type, comptime I: type) type {
            const R = Core.Scalar.Real(C);
            return struct {
                m: I,
                n: I,
                nrhs: I,
                lda: I,
                ldb: I,
                a: common.access.Operand(C, true, kind),
                b: common.access.Operand(C, true, kind),
                singular: common.access.Operand(R, true, kind),
                rank: common.access.Operand(I, true, kind),
                rcond: R,
                work: common.access.Operand(C, true, kind),
                lwork: I,
                rwork: common.access.Operand(R, true, kind),
                iwork: common.access.Operand(I, true, kind),
            };
        }
        pub fn Result(comptime I: type) type {
            return union(enum) { lapacke_info: I, admission_fault: Fault, numerical_fault: NumericalFault };
        }
        pub fn callAdmitted(comptime C: type, comptime I: type, comptime lifetime: common.Lifetime, borrowed: *const Services, source: common.Scratch(Core, C, lifetime).Source, args: Arguments(C, I), options: Rank.Options) common.EngineResult(lifetime, Result(I)) {
            const services = borrowed.*;
            const R = Core.Scalar.Real(C);
            const info = policy.scalarInfo(I, 102, args.m, args.n, args.nrhs, args.lda, args.ldb);
            if (info != 0) {
                H.hook(C, services, .fortran_returning_test, .fortran_gelsd, @intCast(-info - 1), .gelsd);
                return .{ .lapacke_info = info };
            }
            const m = H.native(args.m) catch |e| return .{ .admission_fault = e };
            const n = H.native(args.n) catch |e| return .{ .admission_fault = e };
            const rhs = H.native(args.nrhs) catch |e| return .{ .admission_fault = e };
            const plan = Planner.make(C, I, m, n, rhs) catch return .{ .admission_fault = error.SizeOverflow };
            const minimum = policy.minimumWork(I, m, n, rhs) catch return .{ .admission_fault = error.SizeOverflow };
            const work_count: usize = if (args.lwork == -1 or args.lwork < minimum) 1 else @intCast(args.lwork);
            const work = H.capacity(C, args.work, work_count) catch |e| return .{ .admission_fault = e };
            const real = H.capacity(R, args.rwork, 1) catch |e| return .{ .admission_fault = e };
            const integers = H.capacity(I, args.iwork, 1) catch |e| return .{ .admission_fault = e };
            H.disjointMixed(C, work, R, real) catch |e| return .{ .admission_fault = e };
            H.disjointMixed(C, work, I, integers) catch |e| return .{ .admission_fault = e };
            H.disjointMixed(R, real, I, integers) catch |e| return .{ .admission_fault = e };
            if (args.lwork == -1 or args.lwork < minimum) {
                work[0] = .{ .re = @floatFromInt(plan.work), .im = 0 };
                real[0] = 1;
                integers[0] = 1;
                if (args.lwork == -1) return .{ .lapacke_info = 0 };
                H.hook(C, services, .fortran_returning_test, .fortran_gelsd, 12, .gelsd);
                return .{ .lapacke_info = -13 };
            }
            const rank = H.capacity(I, args.rank, 1) catch |e| return .{ .admission_fault = e };
            const k = @min(m, n);
            const lda = H.native(args.lda) catch |e| return .{ .admission_fault = e };
            const ldb = H.native(args.ldb) catch |e| return .{ .admission_fault = e };
            const acount = if (k == 0) 0 else Core.Storage.columnMajorExtent(m, n, lda) catch |e| return .{ .admission_fault = e };
            const bcount = if (k == 0) 0 else Core.Storage.columnMajorExtent(@max(m, n), rhs, ldb) catch |e| return .{ .admission_fault = e };
            const a = H.capacity(C, args.a, if (k == 0) 0 else acount) catch |e| return .{ .admission_fault = e };
            const b = H.capacity(C, args.b, if (k == 0) 0 else bcount) catch |e| return .{ .admission_fault = e };
            const singular = H.capacity(R, args.singular, k) catch |e| return .{ .admission_fault = e };
            const operands = .{ a, b, singular, rank, work, real, integers };
            inline for (operands, 0..) |one, i| inline for (operands, 0..) |other, j| {
                if (comptime j > i) H.disjointMixed(@TypeOf(one[0]), one, @TypeOf(other[0]), other) catch |e| return .{ .admission_fault = e };
            };
            // Metadata and every numerical operand are disjoint before any write.
            work[0] = .{ .re = @floatFromInt(plan.work), .im = 0 };
            real[0] = 1;
            integers[0] = 1;
            if (k == 0) {
                rank[0] = 0;
                return .{ .lapacke_info = 0 };
            }
            var all_zero = true;
            for (0..n) |j| for (0..m) |i| {
                const z = a[i + j * lda];
                all_zero = all_zero and z.re == 0 and z.im == 0;
            };
            if (all_zero) {
                for (0..rhs) |j| @memset(b[j * ldb ..][0..@max(m, n)], .{ .re = 0, .im = 0 });
                @memset(singular, 0);
                rank[0] = 0;
                return .{ .lapacke_info = 0 };
            }
            // A valid standard minimum is accepted even when the own widened
            // algorithm needs more storage. Fallback uses the already-admitted
            // Context lease (or the supplied unmanaged test allocator).
            const Scratch = common.Scratch(Core, C, lifetime);
            var fallback: ?Scratch.Lease = null;
            defer if (fallback) |*lease| lease.deinit();
            const own_work = if (args.lwork >= plan.work) work[0..@intCast(plan.work)] else blk: {
                fallback = Scratch.acquire(source, @intCast(plan.work)) catch |e| {
                    if (comptime lifetime == .context_managed) {
                        if (e == error.CounterOverflow) return error.CounterOverflow;
                    }
                    H.hook(C, services, .lapacke, .work, -1010, .gelsd);
                    return .{ .lapacke_info = -1010 };
                };
                break :blk Scratch.values(&fallback.?);
            };
            const buffers = Planner.split(C, I, plan, own_work) catch return .{ .admission_fault = error.SizeOverflow };
            // Interface cutoff rules remain GELSD-specific. The common kernel
            // receives their finite result, not GELSS defaults or raw RCOND.
            const cutoff = policy.effectiveRcond(R, k, args.rcond);
            if (!std.math.isFinite(cutoff)) return .{ .numerical_fault = error.InvalidTolerance };
            const bfactor = rightHandSideScale(C, b, m, rhs, ldb);
            var input_b: Core.Dense.ConstMatrix(C) = .{ .values = b, .row_count = m, .column_count = rhs, .leading_dimension = ldb };
            if (bfactor != 1) {
                for (0..rhs) |j| for (0..m) |i| {
                    buffers.rhs[i + j * m] = scaled(C, b[i + j * ldb], bfactor);
                };
                input_b = .{ .values = buffers.rhs, .row_count = m, .column_count = rhs, .leading_dimension = m };
            }
            // Preserve the internal sweep-budget knob while counting actual
            // bidiagonal QR steps; zero still forces an exhaustion observation.
            const steps = std.math.mul(usize, options.max_sweeps, k) catch return .{ .numerical_fault = error.SizeOverflow };
            const result = Math.solveBorrowed(C, .{ .values = a, .row_count = m, .column_count = n, .leading_dimension = lda }, input_b, .{
                .solution = .{ .values = buffers.solution, .row_count = n, .column_count = rhs, .leading_dimension = n },
                .singular = singular,
                .right_vectors = .{ .values = buffers.right_vectors, .row_count = n, .column_count = k, .leading_dimension = n },
                .residual = .{ .values = buffers.residual, .row_count = m -| n, .column_count = rhs, .leading_dimension = m -| n },
            }, .{ .relative = cutoff }, .{ .max_steps = steps, .relative_tolerance = options.relative_tolerance, .numerical_policy = .finite }, .{ .scalar = buffers.scalar, .computational = buffers.wide, .real = buffers.real }) catch |e| return .{ .numerical_fault = e };
            // The shared bidiagonal state supplies the actual unfinished edge
            // count. Other numerical faults retain their separate projection.
            if (result.status == .no_convergence) return .{ .lapacke_info = @intCast(result.unconverged) };
            if (result.status != .converged) return .{ .numerical_fault = if (result.status == .range_limit) error.RangeLimit else error.NonFiniteResult };
            // Preserve the reviewed Reference nested NRHS=0 boundary without
            // injecting it into the native Core. A/S on this error are unspecified.
            if (rhs == 0) {
                H.hook(C, services, .fortran_returning_test, .fortran_lalsd, 4, .lalsd);
                return .{ .lapacke_info = -5 };
            }
            // Stage restored X in existing WORK before any B/rank publication.
            // Residual rows keep selected interface units; Math stays neutral.
            if (bfactor != 1) for (buffers.solution) |*value| {
                const restored = scaled(C, value.*, 1 / bfactor);
                if (!std.math.isFinite(restored.re) or !std.math.isFinite(restored.im)) return .{ .numerical_fault = error.NonFiniteResult };
                value.* = restored;
            };
            for (0..rhs) |j| {
                @memcpy(b[j * ldb ..][0..n], buffers.solution[j * n ..][0..n]);
                if (m > n and result.rank == n) @memcpy(b[j * ldb + n ..][0 .. m - n], buffers.residual[j * (m - n) ..][0 .. m - n]);
            }
            rank[0] = @intCast(result.rank);
            return .{ .lapacke_info = 0 };
        }
        pub fn call(comptime C: type, comptime I: type, borrowed: *const Services, args: Arguments(C, I), options: Rank.Options) Result(I) {
            return callAdmitted(C, I, .unmanaged, borrowed, borrowed.allocator, args, options);
        }
        pub fn callManaged(comptime C: type, comptime I: type, context: *Core.Execution.context.Context, borrowed: *const Services, args: Arguments(C, I), options: Rank.Options) Core.Execution.context.AdmissionError!Result(I) {
            var invocation = try common.ManagedInvocation(Core).begin(context, borrowed);
            defer invocation.operation.end();
            return callAdmitted(C, I, .context_managed, &invocation.services, &invocation.operation, args, options);
        }
    };
}
