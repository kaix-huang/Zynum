// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Hermitian numerical requests borrow the existing execution identity.
//! No allocator policy, default context, thread pool or handle owner is added.
const std = @import("std");
pub const core = @import("core/hermitian_eigen.zig");
const dense = @import("../numerics/dense.zig");
const scalar = @import("../numerics/complex.zig").scalar;
pub fn Request(comptime C: type) type {
    return struct {
        a: dense.ConstMatrix(C),
        b: ?dense.ConstMatrix(C) = null,
        problem: core.Problem = .ax_lambda_bx,
        triangle: core.Triangle = .lower,
        eigenvalues: []scalar.Real(C),
        vectors: ?dense.Matrix(C) = null,
        options: core.Options = .{},
        contract: core.Contract = .checked_finite,
    };
}
fn apply(comptime C: type, request: Request(C), work: []core.WorkspaceScalar(C)) core.Error!core.Result {
    return switch (request.contract) {
        inline else => |contract| if (request.b) |b|
            core.solveGeneralizedWithContract(C, request.a, b, request.problem, request.triangle, request.eigenvalues, request.vectors, work, request.options, contract)
        else
            core.solveWithContract(C, request.a, request.triangle, request.eigenvalues, request.vectors, work, request.options, contract),
    };
}
pub fn Managed(comptime Execution: type) type {
    return struct {
        const context = Execution.context;
        pub const Error = core.Error || context.AdmissionError || std.mem.Allocator.Error;
        /// Does not end the caller's operation or acquire a resource pin.
        pub fn runBorrowedAdmitted(comptime C: type, operation: *const context.Operation, request: Request(C), work: []core.WorkspaceScalar(C)) Error!core.Result {
            _ = operation.borrowedContext();
            return apply(C, request, work);
        }
        /// One scratch lease pins this exact admitted operation's owner.
        /// Allocation precedes leaf validation; free precedes pin release.
        pub fn runScratchAdmitted(comptime C: type, operation: *const context.Operation, request: Request(C)) Error!core.Result {
            _ = operation.borrowedContext();
            const used = try core.workspaceLength(C, request.a.row_count);
            var work = try context.ContextScratch(core.WorkspaceScalar(C)).acquire(operation, used);
            defer work.deinit();
            return apply(C, request, work.lease.values);
        }
        pub fn runBorrowed(comptime C: type, owner: *context.Context, request: Request(C), work: []core.WorkspaceScalar(C)) Error!core.Result {
            var operation = try owner.begin();
            defer operation.end();
            return runBorrowedAdmitted(C, &operation, request, work);
        }
        pub fn runScratch(comptime C: type, owner: *context.Context, request: Request(C)) Error!core.Result {
            var operation = try owner.begin();
            defer operation.end();
            return runScratchAdmitted(C, &operation, request);
        }
    };
}
