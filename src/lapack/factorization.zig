// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const views = @import("../blas/api/views.zig");
const types = @import("../blas/types.zig");
const validation = @import("validation.zig");
const execution = @import("zynum-execution");

fn isComplex(comptime T: type) bool {
    return T == types.ComplexF32 or T == types.ComplexF64;
}

/// Checked real symmetric or complex Hermitian positive-definite factorization.
/// Complex selected diagonals must be exactly real; see module documentation.
pub fn choleskyFactorize(comptime T: type, a: views.Matrix(T), triangle: validation.Triangle) validation.Error!usize {
    var operation = execution.beginDefault();
    defer operation.end();
    return choleskyFactorizeAdmitted(&operation, T, a, triangle);
}

/// Internal checked dispatch borrowing a live operation; never acquires or ends it.
pub fn choleskyFactorizeAdmitted(operation: *const execution.context.Operation, comptime T: type, a: views.Matrix(T), triangle: validation.Triangle) validation.Error!usize {
    _ = operation.borrowedContext();
    if (comptime isComplex(T)) return @import("complex_cholesky.zig").factorize(T, a, triangle);
    return @import("cholesky.zig").factorize(T, a, triangle);
}

/// Solve from a successful selected Cholesky triangle; factors are read-only.
pub fn choleskySolve(comptime T: type, factor: views.ConstMatrix(T), rhs: views.Matrix(T), triangle: validation.Triangle) validation.Error!usize {
    var operation = execution.beginDefault();
    defer operation.end();
    return choleskySolveAdmitted(&operation, T, factor, rhs, triangle);
}

/// Internal checked dispatch borrowing a live operation; never acquires or ends it.
pub fn choleskySolveAdmitted(operation: *const execution.context.Operation, comptime T: type, factor: views.ConstMatrix(T), rhs: views.Matrix(T), triangle: validation.Triangle) validation.Error!usize {
    _ = operation.borrowedContext();
    if (comptime isComplex(T)) return @import("complex_cholesky.zig").solve(T, factor, rhs, triangle);
    return @import("cholesky.zig").solve(T, factor, rhs, triangle);
}

/// Rectangular partial-pivoted LU with zero-based row swaps and P*A=L*U.
pub fn luFactorize(comptime T: type, a: views.Matrix(T), pivots: []views.BlasInt) validation.Error!usize {
    var admission = execution.beginDefault();
    defer admission.end();
    if (comptime isComplex(T)) return @import("complex_lu.zig").factorize(T, a, pivots);
    return @import("lu.zig").factorize(T, a, pivots);
}

/// Solve normal A*X=B from successful square LU factors and matching pivots.
pub fn luSolve(comptime T: type, factor: views.ConstMatrix(T), pivots: []const views.BlasInt, rhs: views.Matrix(T)) validation.Error!usize {
    var admission = execution.beginDefault();
    defer admission.end();
    if (comptime isComplex(T)) return @import("complex_lu.zig").solve(T, factor, pivots, rhs);
    return @import("lu.zig").solve(T, factor, pivots, rhs);
}
