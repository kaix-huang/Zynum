// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Full-spectrum real symmetric-definite eigensolve over one admitted operation.
const std = @import("std");
const execution = @import("zynum-execution");
const dense = @import("../numerics/dense.zig");
const storage = @import("../numerics/storage.zig");
const triangular = @import("../numerics/triangular_ieee.zig");
const views = @import("../blas/api/views.zig");
const factorization = @import("factorization.zig");
const symmetric_eigen = @import("symmetric_eigen.zig");

pub const ScratchExclusion = storage.ByteRange;
pub const scratchExclusion = storage.byteRange;
pub const Arguments = struct {
    h: dense.ConstMatrix(f64),
    s: dense.ConstMatrix(f64),
    eigenvalues: []f64,
    coefficients: dense.Matrix(f64),
    options: symmetric_eigen.Options = .{},
    /// Prevalidated spans of controls that must not be occupied by scratch.
    excluded_scratch: []const ScratchExclusion = &.{},
};

pub const Error = execution.ExecutionError || symmetric_eigen.Error || error{ InvalidView, NonSymmetricInput };
pub const Status = enum { converged, not_positive_definite, iteration_limit, nonfinite_arithmetic, accuracy_not_met };
pub const Result = struct {
    status: Status,
    /// One-based nonpositive pivot, or zero when not applicable.
    failing_pivot: usize = 0,
    sweeps: usize = 0,
};

fn matrixExtent(matrix: anytype) Error!usize {
    return matrix.check() catch |err| switch (err) {
        error.SizeOverflow => error.ExtentOverflow,
        error.InvalidLeadingDimension => error.InvalidLeadingDimension,
        error.BufferTooSmall => error.BufferTooSmall,
    };
}

fn byteRange(values: []const f64) Error!storage.ByteRange {
    const bytes = storage.byteLength(values.len, @sizeOf(f64)) catch return error.ExtentOverflow;
    if (bytes > std.math.maxInt(isize)) return error.ExtentOverflow;
    const address = @intFromPtr(values.ptr);
    if ((bytes != 0 and address == 0) or address % @alignOf(f64) != 0) return error.InvalidView;
    return storage.byteRange(address, bytes) catch error.ExtentOverflow;
}

fn checkSymmetric(matrix: dense.ConstMatrix(f64)) Error!void {
    const n = matrix.row_count;
    const ld = matrix.leading_dimension;
    for (0..n) |j| {
        if (!allFinite(matrix.values[j * ld ..][0..n])) return error.NonFiniteInput;
    }
    for (0..n) |j| for (j + 1..n) |i| {
        const lower: u64 = @bitCast(matrix.values[i + j * ld]);
        const upper: u64 = @bitCast(matrix.values[j + i * ld]);
        if (lower != upper) return error.NonSymmetricInput;
    };
}

fn allFinite(values: []const f64) bool {
    for (values) |value| if (!std.math.isFinite(value)) return false;
    return true;
}

fn workspaceLength(square: usize, n: usize) Error!usize {
    // Reduced H, factor of S, standard eigensolver work, staged C and lambda.
    const matrix_elements = std.math.mul(usize, 4, square) catch return error.ExtentOverflow;
    const elements = std.math.add(usize, matrix_elements, n) catch return error.ExtentOverflow;
    const rounded = std.math.add(usize, elements, elements % 2) catch return error.ExtentOverflow;
    const bytes = storage.byteLength(rounded, @sizeOf(f64)) catch return error.ExtentOverflow;
    if (bytes > std.math.maxInt(isize)) return error.ExtentOverflow;
    return elements;
}

fn view(values: []f64, dim: views.BlasInt) views.Matrix(f64) {
    return .{ .values = values, .row_count = dim, .column_count = dim, .leading_dimension = dim };
}

fn columnMajor(values: []f64, n: usize) dense.Matrix(f64) {
    return .{ .values = values, .row_count = n, .column_count = n, .leading_dimension = n };
}

fn pack(source: dense.ConstMatrix(f64), destination: []f64) void {
    const n = source.row_count;
    for (0..n) |j| @memcpy(destination[j * n ..][0..n], source.values[j * source.leading_dimension ..][0..n]);
}

const orthogonality_tolerance: f128 = 1e-10;
const residual_tolerance: f128 = 1e-12;

const NormBounds = struct {
    frobenius: f128,
    spectral_lower_bound: f128,
};

fn normBounds(matrix: dense.ConstMatrix(f64)) NormBounds {
    var frobenius_squared: f128 = 0;
    var largest_column_squared: f128 = 0;
    for (0..matrix.column_count) |j| {
        var column_squared: f128 = 0;
        const column = matrix.values[j * matrix.leading_dimension ..][0..matrix.row_count];
        for (column) |value| {
            const widened: f128 = value;
            column_squared += widened * widened;
        }
        frobenius_squared += column_squared;
        largest_column_squared = @max(largest_column_squared, column_squared);
    }
    return .{ .frobenius = @sqrt(frobenius_squared), .spectral_lower_bound = @sqrt(largest_column_squared) };
}

fn dot(first: []const f64, second: []const f64) f128 {
    var total: f128 = 0;
    for (first, second) |a, b| total += @as(f128, a) * @as(f128, b);
    return total;
}

/// Check the original eigenproblem before any output write. Frobenius
/// orthogonality upper-bounds the spectral defect. The largest column norm
/// lower-bounds each operator spectral norm, giving a conservative
/// normalized residual bound for each eigenpair.
fn accuracyMet(args: Arguments, values: []const f64, vectors: []const f64, s_times_vector: []f128) bool {
    const n = values.len;
    const h_norm = normBounds(args.h);
    const s_norm = normBounds(args.s);
    var residual_squared: f128 = 0;
    var vectors_squared: f128 = 0;
    var weighted_vectors_squared: f128 = 0;
    var orthogonality_squared: f128 = 0;
    for (values, 0..) |value, j| {
        const lambda: f128 = value;
        const vector = vectors[j * n ..][0..n];
        const vector_squared = dot(vector, vector);
        if (vector_squared == 0) return false;

        var column_residual_squared: f128 = 0;
        for (s_times_vector, 0..) |*sc, i| {
            // H/S are bit-symmetric, so a contiguous column also gives the row.
            const h_row = args.h.values[i * args.h.leading_dimension ..][0..n];
            const s_row = args.s.values[i * args.s.leading_dimension ..][0..n];
            sc.* = dot(s_row, vector);
            const residual = dot(h_row, vector) - lambda * sc.*;
            column_residual_squared += residual * residual;
        }
        const column_scale = (h_norm.spectral_lower_bound + @abs(lambda) * s_norm.spectral_lower_bound) * @sqrt(vector_squared);
        if (!(@sqrt(column_residual_squared) <= residual_tolerance * column_scale)) return false;
        residual_squared += column_residual_squared;
        vectors_squared += vector_squared;
        weighted_vectors_squared += lambda * lambda * vector_squared;

        for (0..n) |i| {
            const other = vectors[i * n ..][0..n];
            const target: f128 = if (i == j) 1 else 0;
            var inner_product: f128 = 0;
            for (other, s_times_vector) |c, sc| inner_product += @as(f128, c) * sc;
            const defect = inner_product - target;
            orthogonality_squared += defect * defect;
        }
    }
    const residual_scale = h_norm.frobenius * @sqrt(vectors_squared) + s_norm.frobenius * @sqrt(weighted_vectors_squared);
    return @sqrt(residual_squared) <= residual_tolerance * residual_scale and
        @sqrt(orthogonality_squared) <= orthogonality_tolerance;
}

fn solveStaged(
    operation: *const execution.Operation,
    args: Arguments,
    dim: views.BlasInt,
    square: usize,
    work: []f64,
    measured_s_times_vector: []f128,
) Error!Result {
    const n: usize = @intCast(dim);
    const reduced = work[0..square];
    const lower = work[square .. 2 * square];
    const standard_work = work[2 * square .. 3 * square];
    const vectors = work[3 * square .. 4 * square];
    const values = work[4 * square ..];
    pack(args.h, reduced);
    pack(args.s, lower);

    const factor = try factorization.realCholeskyFactorizeDetailedAdmitted(operation, f64, view(lower, dim), .lower);
    switch (factor.status) {
        .complete => {},
        .not_positive_definite => return .{ .status = .not_positive_definite, .failing_pivot = factor.failing_pivot },
        .nonfinite_arithmetic => return .{ .status = .nonfinite_arithmetic },
    }

    // Form L^-1 H L^-T. Transposing between left solves applies the second
    // solve from the right, using symmetry of the input. The resulting matrix
    // is symmetric up to rounding in the two triangular solves.
    const l = columnMajor(lower, n).asConst();
    const b = columnMajor(reduced, n);
    triangular.solve(f64, l, b, .lower, .normal, .non_unit);
    if (!allFinite(reduced)) return .{ .status = .nonfinite_arithmetic };
    for (0..n) |j| for (j + 1..n) |i| {
        std.mem.swap(f64, &reduced[i + j * n], &reduced[j + i * n]);
    };
    triangular.solve(f64, l, b, .lower, .normal, .non_unit);
    if (!allFinite(reduced)) return .{ .status = .nonfinite_arithmetic };

    for (0..n) |j| for (j + 1..n) |i| {
        const a = &reduced[i + j * n];
        const transposed = reduced[j + i * n];
        if (a.* != transposed) a.* = a.* * 0.5 + transposed * 0.5;
    };
    const standard = try symmetric_eigen.decomposeAdmitted(
        operation,
        f64,
        view(reduced, dim).asConst(),
        .lower,
        view(standard_work, dim),
        values,
        view(vectors, dim),
        args.options,
    );
    switch (standard.status) {
        .converged => {},
        .iteration_limit => return .{ .status = .iteration_limit, .sweeps = standard.sweeps },
        .nonfinite_iteration, .eigenvalue_out_of_range => return .{
            .status = .nonfinite_arithmetic,
            .sweeps = standard.sweeps,
        },
    }

    // Backtransform the orthonormal eigenvectors to the S metric.
    triangular.solve(f64, l, columnMajor(vectors, n), .lower, .transposed, .non_unit);
    if (!allFinite(vectors)) return .{ .status = .nonfinite_arithmetic, .sweeps = standard.sweeps };

    if (!accuracyMet(args, values, vectors, measured_s_times_vector))
        return .{ .status = .accuracy_not_met, .sweeps = standard.sweeps };

    @memcpy(args.eigenvalues[0..n], values);
    for (0..n) |j| {
        @memcpy(args.coefficients.values[j * args.coefficients.leading_dimension ..][0..n], vectors[j * n ..][0..n]);
    }
    return .{ .status = .converged, .sweeps = standard.sweeps };
}

/// Solve H C = S C diag(lambda) with ascending eigenvalues. H and S are full,
/// finite, bit-symmetric real matrices. Both input matrices may share storage.
/// The used writable spans (including inter-column padding) must be disjoint
/// from each other and both inputs. Unused output padding and tails survive.
/// One provider scratch lease stages the full result. Wide real accumulation
/// verifies backward residual (1e-12) and metric orthogonality (1e-10) in the
/// original problem, including extreme finite f64 scales. Numerical failure,
/// extent/argument errors, and allocation rejection preserve both outputs.
/// The caller owns admission; the solver returns with its scratch released.
pub fn solveAdmitted(
    operation: *const execution.Operation,
    args: Arguments,
    report: *execution.CallReport,
) Error!Result {
    _ = operation.borrowedContext();
    report.* = .{};
    const n = args.h.row_count;
    if (args.h.column_count != n or args.s.row_count != n or args.s.column_count != n or
        args.coefficients.row_count != n or args.coefficients.column_count != n)
        return error.DimensionMismatch;
    const dim = std.math.cast(views.BlasInt, n) orelse return error.ExtentOverflow;
    const square = std.math.mul(usize, n, n) catch return error.ExtentOverflow;
    const length = try workspaceLength(square, n);

    const h = try byteRange(args.h.values[0..try matrixExtent(args.h)]);
    const s = try byteRange(args.s.values[0..try matrixExtent(args.s)]);
    const c = try byteRange(args.coefficients.values[0..try matrixExtent(args.coefficients)]);
    if (args.eigenvalues.len < n) return error.BufferTooSmall;
    const values = try byteRange(args.eigenvalues[0..n]);
    if (storage.overlaps(h, c) or storage.overlaps(s, c) or
        storage.overlaps(h, values) or storage.overlaps(s, values) or storage.overlaps(c, values))
        return error.AliasingNotAllowed;
    const backing = [_]storage.ByteRange{
        try byteRange(args.h.values),
        try byteRange(args.s.values),
        try byteRange(args.coefficients.values),
        try byteRange(args.eigenvalues),
    };
    try checkSymmetric(args.h);
    try checkSymmetric(args.s);

    const result: Result = if (n == 0) .{ .status = .converged } else result: {
        // One aligned block for f64 staging. Reuse the standard eigensolver
        // workspace afterward for an f128 column of S*C.
        const words = length / 2 + length % 2;
        var scratch = try execution.acquireOperationScratch(operation, f128, words, report);
        defer scratch.release(report);
        const workspace = std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(scratch.values));
        const leased = try byteRange(workspace);
        for (backing) |range| {
            if (storage.overlaps(leased, range)) return error.InvalidProvider;
        }
        for (args.excluded_scratch) |range| {
            if (storage.overlaps(leased, range)) return error.InvalidProvider;
        }
        var scalar_product: [1]f128 = undefined;
        const s_times_vector: []f128 = if (n == 1) &scalar_product else scratch.values[square..][0..n];
        break :result try solveStaged(operation, args, dim, square, workspace[0..length], s_times_vector);
    };
    report.output = if (result.status == .converged) .complete else .unchanged;
    return result;
}
