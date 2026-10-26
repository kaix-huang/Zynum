// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! GELSD WORK-query stage only. No standard routine is exported here.
const common = @import("lapacke_common.zig");
const policy = @import("lapacke_gelsd_policy.zig");
const workspace = @import("lapacke_gelsd_workspace.zig");
pub fn Query(comptime Core: type, comptime kind: common.access.Kind) type {
    return struct {
        const H = common.HelpersWithAccess(Core, kind);
        const Planner = workspace.Planner(Core);
        const Fault = common.access.Fault(kind);
        pub const OwnerIdentity = Core.OwnerIdentity;
        pub const Services = Core.Services;
        pub fn Arguments(comptime T: type, comptime I: type) type {
            H.integer(I);
            if (T != f32 and T != f64) @compileError("GELSD supports f32/f64");
            return struct {
                layout: c_int,
                m: I,
                n: I,
                nrhs: I,
                lda: I,
                ldb: I,
                // Unused operands are retained to test the query nonread contract.
                a: common.access.Operand(T, true, kind) = null,
                b: common.access.Operand(T, true, kind) = null,
                singular: common.access.Operand(T, true, kind) = null,
                rank: common.access.Operand(I, true, kind) = null,
                rcond: T = -1,
                work: common.access.Operand(T, true, kind),
                iwork: common.access.Operand(I, true, kind),
            };
        }
        pub fn Result(comptime I: type) type {
            return union(enum) { lapacke_info: I, admission_fault: Fault };
        }
        pub fn call(comptime T: type, comptime I: type, borrowed: *const Services, args: Arguments(T, I)) Result(I) {
            const services = borrowed.*;
            const info = policy.scalarInfo(I, args.layout, args.m, args.n, args.nrhs, args.lda, args.ldb);
            if (info != 0) {
                if (info == -1 or (args.layout == 101 and (info == -6 or info == -8))) {
                    H.hook(T, services, .lapacke, .work, @intCast(info), .gelsd);
                } else {
                    H.hook(T, services, .fortran_returning_test, .fortran_gelsd, @intCast(-info - 1), .gelsd);
                }
                return .{ .lapacke_info = info };
            }
            const m = H.native(args.m) catch |err| return .{ .admission_fault = err };
            const n = H.native(args.n) catch |err| return .{ .admission_fault = err };
            const nrhs = H.native(args.nrhs) catch |err| return .{ .admission_fault = err };
            const plan = Planner.make(T, I, m, n, nrhs) catch return .{ .admission_fault = error.SizeOverflow };
            const work = H.capacity(T, args.work, 1) catch |err| return .{ .admission_fault = err };
            const integers = H.capacity(I, args.iwork, 1) catch |err| return .{ .admission_fault = err };
            const wr = Core.Storage.byteRange(@intFromPtr(work.ptr), @sizeOf(T)) catch |err| return .{ .admission_fault = err };
            const ir = Core.Storage.byteRange(@intFromPtr(integers.ptr), @sizeOf(I)) catch |err| return .{ .admission_fault = err };
            if (Core.Storage.overlaps(wr, ir)) return .{ .admission_fault = error.AliasingNotAllowed };
            work[0] = @floatFromInt(plan.work);
            integers[0] = plan.iwork;
            return .{ .lapacke_info = 0 };
        }
        pub fn callManaged(comptime T: type, comptime I: type, context: *Core.Execution.context.Context, borrowed: *const Services, args: Arguments(T, I)) Core.Execution.context.AdmissionError!Result(I) {
            var invocation = try common.ManagedInvocation(Core).begin(context, borrowed);
            defer invocation.operation.end();
            return call(T, I, &invocation.services, args);
        }
    };
}
