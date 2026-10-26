// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Unpreconditioned CG for caller-supplied real SPD CSR systems.
//! Structural validation does not prove symmetry or positive definiteness.
const std = @import("std");
const execution = @import("zynum-execution");
const storage = @import("../numerics/storage.zig");
const views = @import("views.zig");

pub const Error = views.Error;
pub const InitialGuess = enum { zero, provided };
pub const Status = enum { converged, max_iterations, nonpositive_curvature, nonfinite };

pub fn Accumulator(comptime T: type) type {
    if (T == f32) return f64;
    if (T == f64) return f128;
    @compileError("CG scalar type must be f32 or f64");
}

pub fn Result(comptime T: type) type {
    return struct {
        status: Status,
        /// Number of fully committed x updates (initialization is not an iteration).
        iterations: usize,
        rhs_norm: Accumulator(T),
        tolerance: Accumulator(T),
        /// Recomputed b-A*x norm; null if its evaluation is not representable.
        true_residual_norm: ?Accumulator(T),
    };
}

/// Exact active T element count: residual, direction, matrix product and trial
/// vector. All count and byte-size arithmetic is checked, including n == 0.
pub fn conjugateGradientWorkspaceLength(comptime T: type, n: usize) Error!usize {
    _ = Accumulator(T);
    const count = std.math.mul(usize, n, 4) catch return error.SizeOverflow;
    _ = storage.byteLength(count, @sizeOf(T)) catch return error.SizeOverflow;
    return count;
}

const Range = storage.ByteRange;
fn byteRange(comptime T: type, slice: []const T) Error!Range {
    const start = @intFromPtr(slice.ptr);
    const bytes = storage.byteLength(slice.len, @sizeOf(T)) catch return error.SizeOverflow;
    return storage.byteRange(start, bytes);
}
fn disjoint(first: Range, second: Range) Error!void {
    if (storage.overlaps(first, second)) return error.AliasingNotAllowed;
}

fn Sum(comptime A: type) type {
    return struct {
        value: A = 0,
        correction: A = 0,
        fn add(self: *@This(), term: A) void {
            const next = self.value + term;
            self.correction += if (@abs(self.value) >= @abs(term)) (self.value - next) + term else (term - next) + self.value;
            self.value = next;
        }
        fn total(self: @This()) A {
            return self.value + self.correction;
        }
    };
}

fn dot(comptime T: type, x: []const T, y: []const T) Accumulator(T) {
    const A = Accumulator(T);
    var sum: Sum(A) = .{};
    for (x, y) |left, right| sum.add(@as(A, left) * @as(A, right));
    return sum.total();
}

fn fitsScalar(comptime T: type, value: Accumulator(T)) bool {
    return std.math.isFinite(value) and @abs(value) <= @as(Accumulator(T), std.math.floatMax(T));
}

fn matrixVector(comptime T: type, a: anytype, x: []const T, result: []T) bool {
    const A = Accumulator(T);
    for (0..a.row_count) |row| {
        var sum: Sum(A) = .{};
        const begin: usize = @intCast(a.row_offsets[row]);
        const end: usize = @intCast(a.row_offsets[row + 1]);
        for (begin..end) |entry| {
            const column: usize = @intCast(a.column_indices[entry]);
            sum.add(@as(A, a.values[entry]) * @as(A, x[column]));
        }
        const value = sum.total();
        if (!fitsScalar(T, value)) return false;
        result[row] = @floatCast(value);
    }
    return true;
}

// Recompute each residual directly from b and the current x, independently of
// the recurrence. The reported norm uses widened residuals BEFORE narrowing;
// underflow in the T workspace cannot by itself manufacture convergence.
fn trueResidual(comptime T: type, a: anytype, b: []const T, x: []const T, residual: []T) ?Accumulator(T) {
    const A = Accumulator(T);
    var squared_norm: Sum(A) = .{};
    for (0..a.row_count) |row| {
        var sum: Sum(A) = .{};
        sum.add(b[row]);
        const begin: usize = @intCast(a.row_offsets[row]);
        const end: usize = @intCast(a.row_offsets[row + 1]);
        for (begin..end) |entry| {
            const column: usize = @intCast(a.column_indices[entry]);
            sum.add(-(@as(A, a.values[entry]) * @as(A, x[column])));
        }
        const value = sum.total();
        if (!fitsScalar(T, value)) return null;
        squared_norm.add(value * value);
        residual[row] = @floatCast(value);
    }
    const norm = @sqrt(squared_norm.total());
    return if (std.math.isFinite(norm)) norm else null;
}

fn finish(comptime T: type, a: anytype, b: []const T, x: []const T, r: []T, status: Status, iterations: usize, rhs_norm: Accumulator(T), tolerance: Accumulator(T)) Result(T) {
    const norm = trueResidual(T, a, b, x, r);
    return .{
        .status = if (norm) |value| (if (value <= tolerance) .converged else status) else .nonfinite,
        .iterations = iterations,
        .rhs_norm = rhs_norm,
        .tolerance = tolerance,
        .true_residual_norm = norm,
    };
}

/// Solve a normal CSR system supplied as symmetric positive definite by caller.
/// Required fields: matrix, rhs ([]const T), solution ([]T), workspace ([]T),
/// initial_guess (.zero/.provided), relative_tolerance (T), absolute_tolerance
/// (T), max_iterations (usize). Tolerances must be finite and nonnegative.
/// Threshold is atol + rtol*||rhs||₂. A transposed view is explicitly rejected.
///
/// Full structure, shape, exact vector lengths, workspace capacity, byte-range
/// aliases, tolerances and finite inputs are checked BEFORE any mutation in all
/// modes. .zero ignores old solution values, then initializes x=0; .provided
/// requires them finite. All CSR values and RHS elements must be finite. Output
/// and active workspace are mutually disjoint and disjoint from all inputs.
/// Read-only inputs may overlap. Only exact active prefixes are accessed through
/// workspace; excluded tails may change if reused by another active operand.
/// Caller owns all buffers and must keep them valid and free of concurrent edits.
///
/// Each convergence claim uses a fresh widened b-A*x residual. A recurrence
/// residual that appears converged but fails verification restarts the search
/// from that true residual, within the same total iteration cap. No symmetry or
/// definiteness test is claimed. Nonpositive p'Ap reports numerical breakdown.
/// Argument errors leave all storage unchanged. Numerical termination preserves
/// the latest complete x iterate (or the initialized zero guess); workspace may
/// contain partial work. No rollback of previous iterations is performed.
///
/// Dot products, norms and CSR sums use compensated f64/f128 accumulation for
/// f32/f64 input; stored vectors remain T. Extreme-range products/updates that
/// cannot fit T return nonfinite; underflow, roundoff, stagnation and loss of
/// conjugacy are not eliminated. This is an unscaled, unpreconditioned serial
/// method with no hidden allocation/cache/backend or performance guarantee.
pub fn conjugateGradient(arguments: anytype) Error!Result(@TypeOf(arguments.matrix).Scalar) {
    var operation = execution.beginDefault();
    defer operation.end();
    const a = arguments.matrix;
    const T = @TypeOf(a).Scalar;
    const I = @TypeOf(a).Index;
    const A = Accumulator(T);
    const b: []const T = arguments.rhs;
    const x: []T = arguments.solution;
    const workspace: []T = arguments.workspace;
    const guess: InitialGuess = arguments.initial_guess;
    const relative: T = arguments.relative_tolerance;
    const absolute: T = arguments.absolute_tolerance;
    const max_iterations: usize = arguments.max_iterations;

    try a.check();
    if (a.operation != .normal) return error.UnsupportedTransform;
    const n = a.row_count;
    if (a.column_count != n or b.len != n or x.len != n) return error.DimensionMismatch;
    const required = try conjugateGradientWorkspaceLength(T, n);
    if (workspace.len < required) return error.WorkspaceTooSmall;
    if (!std.math.isFinite(relative) or !std.math.isFinite(absolute) or relative < 0 or absolute < 0) return error.InvalidTolerance;
    const x_range = try byteRange(T, x);
    const work_range = try byteRange(T, workspace[0..required]);
    try disjoint(x_range, work_range);
    const inputs = [_]Range{ try byteRange(T, b), try byteRange(T, a.values), try byteRange(I, a.row_offsets), try byteRange(I, a.column_indices) };
    for (inputs) |input| {
        try disjoint(x_range, input);
        try disjoint(work_range, input);
    }
    for (a.values) |value| if (!std.math.isFinite(value)) return error.NonFiniteInput;
    for (b) |value| if (!std.math.isFinite(value)) return error.NonFiniteInput;
    if (guess == .provided) {
        for (x) |value| if (!std.math.isFinite(value)) return error.NonFiniteInput;
    }
    const rhs_norm = @sqrt(dot(T, b, b));
    const tolerance = @as(A, absolute) + @as(A, relative) * rhs_norm;
    const r = workspace[0..n];
    const p = workspace[n..][0..n];
    const ap = workspace[2 * n ..][0..n];
    const trial = workspace[3 * n ..][0..n];
    if (guess == .zero) @memset(x, 0);
    const initial_norm = trueResidual(T, a, b, x, r) orelse return .{ .status = .nonfinite, .iterations = 0, .rhs_norm = rhs_norm, .tolerance = tolerance, .true_residual_norm = null };
    if (initial_norm <= tolerance) return .{ .status = .converged, .iterations = 0, .rhs_norm = rhs_norm, .tolerance = tolerance, .true_residual_norm = initial_norm };
    if (max_iterations == 0) return .{ .status = .max_iterations, .iterations = 0, .rhs_norm = rhs_norm, .tolerance = tolerance, .true_residual_norm = initial_norm };
    @memcpy(p, r);
    var rho = dot(T, r, r);
    var iterations: usize = 0;
    while (iterations < max_iterations) {
        if (!matrixVector(T, a, p, ap)) return finish(T, a, b, x, r, .nonfinite, iterations, rhs_norm, tolerance);
        const curvature = dot(T, p, ap);
        if (!std.math.isFinite(curvature)) return finish(T, a, b, x, r, .nonfinite, iterations, rhs_norm, tolerance);
        if (curvature <= 0) return finish(T, a, b, x, r, .nonpositive_curvature, iterations, rhs_norm, tolerance);
        const alpha = rho / curvature;
        if (!std.math.isFinite(alpha)) return finish(T, a, b, x, r, .nonfinite, iterations, rhs_norm, tolerance);
        // Both x and r updates must be representable before committing x.
        for (0..n) |i| {
            const next_x = @as(A, x[i]) + alpha * @as(A, p[i]);
            const next_r = @as(A, r[i]) - alpha * @as(A, ap[i]);
            if (!fitsScalar(T, next_x) or !fitsScalar(T, next_r)) return finish(T, a, b, x, r, .nonfinite, iterations, rhs_norm, tolerance);
            trial[i] = @floatCast(next_x);
        }
        for (0..n) |i| r[i] = @floatCast(@as(A, r[i]) - alpha * @as(A, ap[i]));
        @memcpy(x, trial);
        iterations += 1;
        const next_rho = dot(T, r, r);
        if (@sqrt(next_rho) <= tolerance or iterations == max_iterations) {
            const norm = trueResidual(T, a, b, x, r) orelse return .{ .status = .nonfinite, .iterations = iterations, .rhs_norm = rhs_norm, .tolerance = tolerance, .true_residual_norm = null };
            if (norm <= tolerance) return .{ .status = .converged, .iterations = iterations, .rhs_norm = rhs_norm, .tolerance = tolerance, .true_residual_norm = norm };
            if (iterations == max_iterations) return .{ .status = .max_iterations, .iterations = iterations, .rhs_norm = rhs_norm, .tolerance = tolerance, .true_residual_norm = norm };
            @memcpy(p, r);
            rho = dot(T, r, r);
            continue;
        }
        const beta = next_rho / rho;
        if (!std.math.isFinite(beta)) return finish(T, a, b, x, r, .nonfinite, iterations, rhs_norm, tolerance);
        for (0..n) |i| {
            const direction = @as(A, r[i]) + beta * @as(A, p[i]);
            if (!fitsScalar(T, direction)) return finish(T, a, b, x, r, .nonfinite, iterations, rhs_norm, tolerance);
            trial[i] = @floatCast(direction);
        }
        @memcpy(p, trial);
        rho = next_rho;
    }
    unreachable;
}
