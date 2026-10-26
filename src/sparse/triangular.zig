// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Checked real/complex CSR triangular solve over borrowed caller storage.
//! No allocation, analyzed cache, external backend, or parallel workers.

const std = @import("std");
const execution = @import("zynum-execution");
const storage = @import("../numerics/storage.zig");
const views = @import("views.zig");
const scalar = @import("scalar.zig");
const arithmetic = @import("../numerics/complex.zig");

pub const Error = views.Error;
pub const Contract = enum { checked, sparse_ieee_v1 };

pub const Triangle = enum { lower, upper };
pub const Diagonal = enum { non_unit, unit };
pub const Failure = enum { singular_diagonal, non_finite_diagonal, non_finite_result };

pub const SolveResult = struct {
    /// Zero on success; otherwise a one-based logical equation/diagonal index.
    info: usize = 0,
    failure: ?Failure = null,
    /// Zero on success or diagonal preflight failure; otherwise the one-based RHS column.
    rhs_column: usize = 0,
    /// Completed RHS columns; the total RHS count on success. Later columns are unchanged on failure.
    completed_rhs: usize = 0,
};

/// Exact active workspace ELEMENT count of the scalar type T. A non-unit solve
/// needs n aggregate diagonals, except an empty system or zero RHS needs none.
/// This checks workspace byte-size arithmetic, not matrix shape or storage.
pub fn triangularSolveWorkspaceLength(comptime T: type, n: usize, rhs_count: usize, diagonal: Diagonal) Error!usize {
    if (comptime T != f32 and T != f64 and !scalar.isComplex(T)) @compileError("Sparse triangular scalar type must be real or complex f32/f64");
    const count = if (n == 0 or rhs_count == 0 or diagonal == .unit) 0 else n;
    _ = storage.byteLength(count, @sizeOf(T)) catch return error.SizeOverflow;
    return count;
}

const Range = storage.ByteRange;

fn finite(comptime T: type, value: T) bool {
    return if (comptime scalar.isComplex(T)) arithmetic.finite(T, value) else std.math.isFinite(value);
}

fn byteRange(comptime T: type, values: []const T) Error!Range {
    const start = @intFromPtr(values.ptr);
    const bytes = storage.byteLength(values.len, @sizeOf(T)) catch return error.SizeOverflow;
    return storage.byteRange(start, bytes);
}

fn disjoint(first: Range, second: Range) Error!void {
    if (storage.overlaps(first, second)) return error.AliasingNotAllowed;
}

fn requiredDenseLength(rhs: anytype) Error!usize {
    // rhs.check() must already have checked the leading dimension and capacity.
    return storage.columnMajorExtent(rhs.row_count, rhs.column_count, rhs.leading_dimension);
}

fn inStrictTriangle(row: usize, column: usize, triangle: Triangle) bool {
    return if (triangle == .lower) column < row else column > row;
}

fn isUsed(row: usize, column: usize, triangle: Triangle, diagonal: Diagonal) bool {
    return inStrictTriangle(row, column, triangle) or (row == column and diagonal == .non_unit);
}

fn numericFailure(index: usize, kind: Failure, rhs_column: usize, completed_rhs: usize) SolveResult {
    return .{ .info = index + 1, .failure = kind, .rhs_column = rhs_column, .completed_rhs = completed_rhs };
}

// Call only after all used scalar inputs are known finite. Duplicate diagonals
// accumulate from +0 in row storage order. Intermediate overflow is a failure,
// even if a later opposite-signed duplicate could cancel it mathematically.
fn aggregateDiagonal(comptime T: type, a: anytype, row: usize, comptime contract: Contract) struct { value: T, failure: ?Failure } {
    var sum: T = scalar.zero(T);
    var found = false;
    const begin: usize = @intCast(a.row_offsets[row]);
    const end: usize = @intCast(a.row_offsets[row + 1]);
    for (begin..end) |entry| {
        if (@as(usize, @intCast(a.column_indices[entry])) != row) continue;
        found = true;
        sum = scalar.add(T, sum, a.values[entry]);
        if (contract == .checked and !finite(T, sum)) return .{ .value = sum, .failure = .non_finite_diagonal };
    }
    return .{ .value = sum, .failure = if (!found or scalar.isZero(T, sum)) .singular_diagonal else null };
}

/// Solve op(A) * X = B, overwriting the column-major `rhs` with X. Required
/// arguments: matrix (ConstCsrMatrix), triangle, diagonal, rhs (Matrix), workspace
/// ([]Scalar). Matrix.operation selects normal/transposed/adjoint; real adjoint equals
/// transpose. The triangle names
/// the ORIGINAL stored matrix; transposition reverses the solve direction.
/// The matrix must be square and rhs.row_count must match it. All independent signed/unsigned 32/64-bit offset and ordinal combinations
/// and real/complex f32/f64 are supported. Complex adjoint conjugates coefficients
/// and aggregate diagonals.
///
/// Unsorted rows, duplicate coordinates and explicit zeros are allowed. Strict
/// triangle duplicates contribute individually in row storage order. Non-unit
/// diagonal duplicates sum from +0 in storage order. Unit diagonals are exactly
/// one, whether stored or absent; their numerical values and opposite-triangle
/// numerical values are never read. Their indices are still fully validated.
/// With zero RHS columns or n == 0, only structural validation runs, no numerical
/// values are read, no diagonal failures are reported, and workspace is unused.
///
/// Before ANY mutation: check complete CSR structure, dimensions, capacity,
/// size arithmetic and disjoint active spans; require every actually-used A
/// value and logical RHS value to be finite; then preflight all non-unit
/// diagonals. Argument errors leave RHS and workspace unchanged. A missing,
/// zero, or nonfinite aggregate diagonal returns one-based info with rhs_column
/// zero, likewise without mutation. Argument errors precede numerical status.
///
/// Execution overflow/nonfinite arithmetic returns info at the affected logical
/// coordinate, rhs_column identifying the current RHS. Earlier RHS columns are
/// fully solved; the current column may contain partial finite work; later RHS
/// columns are unchanged. Normal solves only store a row after completing it.
/// Transposed solves store a pivot then scatter individual finite updates; a
/// failed update itself is not stored. No rollback is attempted. Active
/// workspace contains aggregate diagonals after any execution starts.
///
/// RHS's bounding span (including inter-column padding) and active workspace
/// must be mutually disjoint and disjoint from every CSR buffer. Read-only CSR
/// buffers may overlap each other. Only logical RHS and active workspace are
/// accessed through those operands; unused capacity is excluded from alias
/// checks and can change if reused through another active operand. Caller owns
/// all buffers and must keep them valid and free of concurrent mutation.
///
/// Preflight/setup takes O(n + nnz + n*nrhs); serial execution takes
/// O((n + nnz)*nrhs). Checked validation is repeated for every call. There is no
/// workspace for unit/empty solves, hidden sorting, conversion, or allocation.
pub fn triangularSolveInPlace(arguments: anytype) Error!SolveResult {
    var operation = execution.beginDefault();
    defer operation.end();
    return triangularSolveInPlaceAdmitted(&operation, arguments);
}

pub fn triangularSolveInPlaceAdmitted(operation: *const execution.context.Operation, arguments: anytype) Error!SolveResult {
    return triangularSolveWithContractAdmitted(operation, arguments, .checked);
}

pub fn triangularSolveWithContractAdmitted(operation: *const execution.context.Operation, arguments: anytype, comptime contract: Contract) Error!SolveResult {
    _ = operation.borrowedContext();
    const a = arguments.matrix;
    const rhs = arguments.rhs;
    const T = @TypeOf(a).Scalar;
    const I = @TypeOf(a).Index;
    const O = if (@hasDecl(@TypeOf(a), "Offset")) @TypeOf(a).Offset else I;
    const triangle: Triangle = arguments.triangle;
    const diagonal: Diagonal = arguments.diagonal;
    const workspace: []T = arguments.workspace;
    comptime {
        if (@TypeOf(rhs).Scalar != T) @compileError("Sparse triangular solve requires matching scalar types");
    }
    try a.check();
    try rhs.check();
    if (a.row_count != a.column_count or rhs.row_count != a.row_count) return error.DimensionMismatch;
    const n = a.row_count;
    const required = try triangularSolveWorkspaceLength(T, n, rhs.column_count, diagonal);
    if (workspace.len < required) return error.WorkspaceTooSmall;
    const dense_length = try requiredDenseLength(rhs);
    const rhs_range = try byteRange(T, rhs.values[0..dense_length]);
    const work_range = try byteRange(T, workspace[0..required]);
    try disjoint(rhs_range, work_range);
    const input_ranges = [_]Range{
        try byteRange(T, a.values),
        try byteRange(O, a.row_offsets),
        try byteRange(I, a.column_indices),
    };
    for (input_ranges) |input_range| {
        try disjoint(rhs_range, input_range);
        try disjoint(work_range, input_range);
    }
    if (n == 0 or rhs.column_count == 0) return .{ .completed_rhs = rhs.column_count };

    if (comptime contract == .checked) {
        // Read only selected numerical entries: ignored NaNs/infinities stay unused.
        for (0..n) |row| {
            const begin: usize = @intCast(a.row_offsets[row]);
            const end: usize = @intCast(a.row_offsets[row + 1]);
            for (begin..end) |entry| {
                const column: usize = @intCast(a.column_indices[entry]);
                if (isUsed(row, column, triangle, diagonal) and !finite(T, a.values[entry])) return error.NonFiniteInput;
            }
        }
        for (0..rhs.column_count) |column| {
            for (rhs.values[column * rhs.leading_dimension ..][0..n]) |value| {
                if (!finite(T, value)) return error.NonFiniteInput;
            }
        }
    }
    if (diagonal == .non_unit) {
        // Dry-run every diagonal before even workspace can be changed.
        for (0..n) |row| {
            if (aggregateDiagonal(T, a, row, contract).failure) |failure| return numericFailure(row, failure, 0, 0);
        }
    }
    if (comptime scalar.isComplex(T)) {
        if (diagonal == .non_unit) for (0..n) |row| {
            workspace[row] = aggregateDiagonal(T, a, row, contract).value;
        };
        return @import("core/triangular_complex.zig").execute(T, SolveResult, a, rhs, triangle == .lower, diagonal == .unit, workspace, contract == .checked);
    }
    return executeValidated(T, a, rhs, triangle, diagonal, workspace, contract);
}

// Call only after this invocation's structural, storage, finite-input and
// diagonal preflight succeeds. Typed runtime parameters keep fixed empty-array
// caller specializations out of numerical indexing; no validation is cached.
fn executeValidated(comptime T: type, a: anytype, rhs: views.Matrix(T), triangle: Triangle, diagonal: Diagonal, workspace: []T, comptime contract: Contract) SolveResult {
    const n = a.row_count;
    if (diagonal == .non_unit) {
        for (0..n) |row| workspace[row] = aggregateDiagonal(T, a, row, contract).value;
    }
    const forward = if (a.operation == .normal) triangle == .lower else triangle == .upper;
    for (0..rhs.column_count) |rhs_column| {
        const x = rhs.values[rhs_column * rhs.leading_dimension ..][0..n];
        for (0..n) |step| {
            const row = if (forward) step else n - 1 - step;
            const begin: usize = @intCast(a.row_offsets[row]);
            const end: usize = @intCast(a.row_offsets[row + 1]);
            const divisor: T = if (diagonal == .unit) 1 else workspace[row];
            if (a.operation == .normal) {
                var residual = x[row];
                for (begin..end) |entry| {
                    const column: usize = @intCast(a.column_indices[entry]);
                    if (!inStrictTriangle(row, column, triangle)) continue;
                    const product = a.values[entry] * x[column];
                    if (contract == .checked and !std.math.isFinite(product)) return numericFailure(row, .non_finite_result, rhs_column + 1, rhs_column);
                    residual -= product;
                    if (contract == .checked and !std.math.isFinite(residual)) return numericFailure(row, .non_finite_result, rhs_column + 1, rhs_column);
                }
                const solution = residual / divisor;
                if (contract == .checked and !std.math.isFinite(solution)) return numericFailure(row, .non_finite_result, rhs_column + 1, rhs_column);
                x[row] = solution;
            } else {
                const solution = x[row] / divisor;
                if (contract == .checked and !std.math.isFinite(solution)) return numericFailure(row, .non_finite_result, rhs_column + 1, rhs_column);
                x[row] = solution;
                for (begin..end) |entry| {
                    const column: usize = @intCast(a.column_indices[entry]);
                    if (!inStrictTriangle(row, column, triangle)) continue;
                    const product = a.values[entry] * solution;
                    if (contract == .checked and !std.math.isFinite(product)) return numericFailure(column, .non_finite_result, rhs_column + 1, rhs_column);
                    const updated = x[column] - product;
                    if (contract == .checked and !std.math.isFinite(updated)) return numericFailure(column, .non_finite_result, rhs_column + 1, rhs_column);
                    x[column] = updated;
                }
            }
        }
    }
    return .{ .completed_rhs = rhs.column_count };
}
