// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Complex GELSD query representation over the existing borrowed Core.
//! Own backend recommendations are independent of vendor ILAENV optima.
const common = @import("lapacke_common.zig");
const policy = @import("lapacke_gelsd_policy.zig");
const workspace = @import("lapacke_complex_gelsd_workspace.zig");
pub fn Query(comptime Core: type, comptime kind: common.access.Kind) type {
    return struct {
        const H = common.HelpersWithAccess(Core, kind);
        const Planner = workspace.Planner(Core);
        const Fault = common.access.Fault(kind);
        pub const Services = Core.Services;
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub fn Arguments(comptime C: type, comptime I: type) type {
            H.integer(I);
            H.scalar(C);
            if (!Core.Scalar.isComplex(C)) @compileError("complex GELSD requires complex scalar");
            const R = Core.Scalar.Real(C);
            return struct {
                layout: c_int,
                m: I,
                n: I,
                nrhs: I,
                lda: I,
                ldb: I,
                a: common.access.Operand(C, true, kind) = null,
                b: common.access.Operand(C, true, kind) = null,
                singular: common.access.Operand(R, true, kind) = null,
                rank: common.access.Operand(I, true, kind) = null,
                rcond: R = -1,
                work: common.access.Operand(C, true, kind),
                rwork: common.access.Operand(R, true, kind),
                iwork: common.access.Operand(I, true, kind),
            };
        }
        pub fn Result(comptime I: type) type {
            return union(enum) { lapacke_info: I, admission_fault: Fault };
        }
        pub fn call(comptime C: type, comptime I: type, borrowed: *const Services, args: Arguments(C, I)) Result(I) {
            const services = borrowed.*;
            const info = policy.scalarInfo(I, args.layout, args.m, args.n, args.nrhs, args.lda, args.ldb);
            if (info != 0) {
                if (info == -1 or (args.layout == 101 and (info == -6 or info == -8))) {
                    H.hook(C, services, .lapacke, .work, @intCast(info), .gelsd);
                } else {
                    H.hook(C, services, .fortran_returning_test, .fortran_gelsd, @intCast(-info - 1), .gelsd);
                }
                return .{ .lapacke_info = info };
            }
            const m = H.native(args.m) catch |err| return .{ .admission_fault = err };
            const n = H.native(args.n) catch |err| return .{ .admission_fault = err };
            const rhs = H.native(args.nrhs) catch |err| return .{ .admission_fault = err };
            const plan = Planner.make(C, I, m, n, rhs) catch return .{ .admission_fault = error.SizeOverflow };
            const R = Core.Scalar.Real(C);
            const work = H.capacity(C, args.work, 1) catch |err| return .{ .admission_fault = err };
            const real = H.capacity(R, args.rwork, 1) catch |err| return .{ .admission_fault = err };
            const integers = H.capacity(I, args.iwork, 1) catch |err| return .{ .admission_fault = err };
            H.disjointMixed(C, work, R, real) catch |err| return .{ .admission_fault = err };
            H.disjointMixed(C, work, I, integers) catch |err| return .{ .admission_fault = err };
            H.disjointMixed(R, real, I, integers) catch |err| return .{ .admission_fault = err };
            // Query operands A/B/S/rank/rcond are never accessed. Validate every
            // output before writing any of the three independent metadata slots.
            work[0] = .{ .re = @floatFromInt(plan.work), .im = 0 };
            real[0] = @floatFromInt(plan.rwork);
            integers[0] = plan.iwork;
            return .{ .lapacke_info = 0 };
        }
        pub fn callManaged(comptime C: type, comptime I: type, context: *Core.Execution.context.Context, borrowed: *const Services, args: Arguments(C, I)) Core.Execution.context.AdmissionError!Result(I) {
            var invocation = try common.ManagedInvocation(Core).begin(context, borrowed);
            defer invocation.operation.end();
            return call(C, I, &invocation.services, args);
        }
    };
}
