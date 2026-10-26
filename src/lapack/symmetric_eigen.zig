// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Real symmetric eigendecomposition by row-cyclic Jacobi rotations.
//! Allocation-free reference foundation; no nonsymmetric or complex modes.
const execution = @import("zynum-execution");
const std = @import("std");
const storage = @import("../numerics/storage.zig");
const views = @import("../blas/api/views.zig");
const validation = @import("validation.zig");
pub const Matrix = views.Matrix;
pub const ConstMatrix = views.ConstMatrix;
pub const Triangle = validation.Triangle;
pub const Error = validation.Error;

pub const Options = struct { max_sweeps: usize = 64 };
pub const Status = enum { converged, iteration_limit, nonfinite_iteration, eigenvalue_out_of_range };
pub const Result = struct {
    status: Status,
    /// Number of complete row-cyclic sweeps. A failed partial sweep is excluded.
    sweeps: usize = 0,
    /// Workspace represents A * 2^(-scale_exponent), up to rotations/rounding.
    scale_exponent: i32 = 0,
    scaled_off_diagonal_max: f64 = 0,
    /// Zero except on numerical failure: one-based pivot row for
    /// nonfinite_iteration, or sorted eigenvalue index for out_of_range.
    failure_index: usize = 0,
};

fn matrixSpan(comptime T: type, value: anytype) Error!usize {
    if (T != f32 and T != f64) @compileError("symmetric eigen supports f32 and f64 only");
    if (value.row_count < 0 or value.column_count < 0) return error.InvalidLength;
    if (value.leading_dimension < 0) return error.InvalidLeadingDimension;
    if (value.row_count == 0 or value.column_count == 0) return 0;
    if (value.leading_dimension < value.row_count) return error.InvalidLeadingDimension;
    const rows: usize = @intCast(value.row_count);
    const columns: usize = @intCast(value.column_count);
    const ld: usize = @intCast(value.leading_dimension);
    const used = storage.columnMajorExtent(rows, columns, ld) catch return error.InvalidLength;
    if (value.values.len < used) return error.BufferTooSmall;
    return used;
}

fn disjoint(comptime T: type, first: []const T, second: []const T) Error!void {
    const first_size = storage.byteLength(first.len, @sizeOf(T)) catch return error.InvalidLength;
    const second_size = storage.byteLength(second.len, @sizeOf(T)) catch return error.InvalidLength;
    const first_start = @intFromPtr(first.ptr);
    const second_start = @intFromPtr(second.ptr);
    const first_range = storage.byteRange(first_start, first_size) catch return error.InvalidLength;
    const second_range = storage.byteRange(second_start, second_size) catch return error.InvalidLength;
    if (storage.overlaps(first_range, second_range)) return error.AliasingNotAllowed;
}

fn powerScale(comptime T: type, value: T, exponent: i32) T {
    return if (value == 0) value else std.math.ldexp(value, exponent);
}

fn offDiagonalMax(comptime T: type, work: Matrix(T), n: usize, ld: usize) f64 {
    var result: f64 = 0;
    for (0..n) |q| for (0..q) |p| {
        result = @max(result, @abs(@as(f64, work.values[p + q * ld])));
    };
    return result;
}

fn stoppingThreshold(comptime T: type, work: Matrix(T), n: usize, ld: usize, initial_max: f64) f64 {
    var scale = initial_max;
    for (0..n) |i| scale = @max(scale, @abs(@as(f64, work.values[i + i * ld])));
    return 4 * @as(f64, std.math.floatEps(T)) * scale;
}

fn rotate(comptime T: type, work: Matrix(T), vectors: Matrix(T), n: usize, ldw: usize, ldv: usize, p: usize, q: usize) bool {
    const app: f64 = work.values[p + p * ldw];
    const aqq: f64 = work.values[q + q * ldw];
    const b: f64 = work.values[p + q * ldw];
    if (b == 0) return true;
    const delta = 0.5 * (aqq - app);
    // Normalize the 2-by-2 rotation formula before its quotient. This avoids
    // forming (aqq-app)/(2*b), which can overflow for tiny nonzero b.
    const scale = @max(@abs(delta), @abs(b));
    const ds = delta / scale;
    const bs = b / scale;
    const denominator = @abs(ds) + std.math.hypot(ds, bs);
    const t = (if (delta < 0) -bs else bs) / denominator;
    const c = 1 / @sqrt(1 + t * t);
    const s = t * c;
    const new_app: T = @floatCast(app - t * b);
    const new_aqq: T = @floatCast(aqq + t * b);
    if (!std.math.isFinite(new_app) or !std.math.isFinite(new_aqq)) return false;
    for (0..n) |k| {
        if (k == p or k == q) continue;
        const x: f64 = work.values[k + p * ldw];
        const y: f64 = work.values[k + q * ldw];
        const new_p: T = @floatCast(c * x - s * y);
        const new_q: T = @floatCast(s * x + c * y);
        if (!std.math.isFinite(new_p) or !std.math.isFinite(new_q)) return false;
        work.values[k + p * ldw] = new_p;
        work.values[p + k * ldw] = new_p;
        work.values[k + q * ldw] = new_q;
        work.values[q + k * ldw] = new_q;
    }
    work.values[p + p * ldw] = new_app;
    work.values[q + q * ldw] = new_aqq;
    work.values[p + q * ldw] = 0;
    work.values[q + p * ldw] = 0;
    for (0..n) |k| {
        const x: f64 = vectors.values[k + p * ldv];
        const y: f64 = vectors.values[k + q * ldv];
        const new_p: T = @floatCast(c * x - s * y);
        const new_q: T = @floatCast(s * x + c * y);
        if (!std.math.isFinite(new_p) or !std.math.isFinite(new_q)) return false;
        vectors.values[k + p * ldv] = new_p;
        vectors.values[k + q * ldv] = new_q;
    }
    return true;
}

fn sortPairs(comptime T: type, work: Matrix(T), vectors: Matrix(T), n: usize, ldw: usize, ldv: usize) void {
    for (0..n) |p| {
        var smallest = p;
        for (p + 1..n) |q| if (work.values[q + q * ldw] < work.values[smallest + smallest * ldw]) {
            smallest = q;
        };
        if (smallest == p) continue;
        // Permute the complete symmetric workspace and matching vector columns.
        for (0..n) |k| std.mem.swap(T, &work.values[p + k * ldw], &work.values[smallest + k * ldw]);
        for (0..n) |k| {
            std.mem.swap(T, &work.values[k + p * ldw], &work.values[k + smallest * ldw]);
            std.mem.swap(T, &vectors.values[k + p * ldv], &vectors.values[k + smallest * ldv]);
        }
    }
}

/// Compute A = V*diag(w)*V^T for a real symmetric n-by-n matrix, reading only
/// the selected input triangle. A must have operation=normal and is read-only.
/// work and vectors are separate n-by-n column-major outputs; eigenvalues[0..n]
/// is the output prefix. All four used physical spans must be pairwise disjoint,
/// including intervening padding. Output contents need not initially be finite.
/// Dimensions, capacity, overlap, and every selected input entry are checked
/// before mutation in all build modes. Padding/trailing storage is not written
/// directly; unselected input entries may be NaN. n=0 is a validated no-op.
///
/// The selected triangle is mirrored into work, scaled by a power of two so
/// its largest magnitude lies in [0.5,1), and V starts as identity. Each sweep
/// visits (p,q) in row-cyclic order, with p<q. Rotation coefficients and scalar
/// expressions use f64; updated workspace/vector entries round to T. Stopping
/// requires max(abs(offdiag(work))) <= 4*eps(T)*max(initial_max,max(abs(diag))).
/// This is a normwise accuracy target, not relative accuracy for tiny or tightly
/// clustered eigenvalues. Scaling can lose very small entries in a matrix with
/// extreme dynamic range. Subnormal eigenvalue rescaling may round to zero.
///
/// On converged, eigenvalues ascend and vector columns match; signs and bases
/// within repeated eigenspaces are unspecified. With max_sweeps=0, diagonal
/// matrices can still succeed. On iteration_limit, work/V hold an unfinished
/// iterate and eigenvalues are unchanged; no eigenpairs are promised. Numerical
/// failures also preserve eigenvalues but may leave partial work/V. Eigenvalue
/// range failure follows convergence and sorting. Argument errors preserve all
/// outputs; padding is preserved on every return. No allocation or backend.
pub fn decompose(comptime T: type, input: ConstMatrix(T), triangle: Triangle, work: Matrix(T), eigenvalues: []T, vectors: Matrix(T), options: Options) Error!Result {
    var admission = execution.beginDefault();
    defer admission.end();
    return decomposeAdmitted(&admission, T, input, triangle, work, eigenvalues, vectors, options);
}

/// Borrow an existing operation from the caller's Context. No additional
/// admission, allocator, workspace owner, or executor is created here.
/// The operation must remain alive for this synchronous numerical call.
pub fn decomposeAdmitted(operation: *const execution.context.Operation, comptime T: type, input: ConstMatrix(T), triangle: Triangle, work: Matrix(T), eigenvalues: []T, vectors: Matrix(T), options: Options) Error!Result {
    _ = operation.borrowedContext();
    const input_span = try matrixSpan(T, input);
    const work_span = try matrixSpan(T, work);
    const vectors_span = try matrixSpan(T, vectors);
    if (input.operation != .normal) return error.InvalidMatrixOperation;
    const dim = input.row_count;
    if (input.column_count != dim or work.row_count != dim or work.column_count != dim or vectors.row_count != dim or vectors.column_count != dim) return error.DimensionMismatch;
    const n: usize = @intCast(dim);
    if (eigenvalues.len < n) return error.BufferTooSmall;
    const a = input.values[0..input_span];
    const w = work.values[0..work_span];
    const v = vectors.values[0..vectors_span];
    const e = eigenvalues[0..n];
    try disjoint(T, a, w);
    try disjoint(T, a, v);
    try disjoint(T, a, e);
    try disjoint(T, w, v);
    try disjoint(T, w, e);
    try disjoint(T, v, e);
    if (n == 0) return .{ .status = .converged };
    const lda: usize = @intCast(input.leading_dimension);
    const ldw: usize = @intCast(work.leading_dimension);
    const ldv: usize = @intCast(vectors.leading_dimension);
    var largest: T = 0;
    for (0..n) |j| {
        const start: usize = if (triangle == .lower) j else 0;
        const end: usize = if (triangle == .lower) n else j + 1;
        for (start..end) |i| {
            const value = input.values[i + j * lda];
            if (!std.math.isFinite(value)) return error.NonFiniteInput;
            largest = @max(largest, @abs(value));
        }
    }
    const exponent: i32 = if (largest == 0) 0 else std.math.frexp(largest).exponent;
    const initial_max: f64 = powerScale(T, largest, -exponent);
    for (0..n) |j| for (0..n) |i| {
        const source = if ((triangle == .lower and i >= j) or (triangle == .upper and i <= j)) i + j * lda else j + i * lda;
        work.values[i + j * ldw] = powerScale(T, input.values[source], -exponent);
        vectors.values[i + j * ldv] = if (i == j) 1 else 0;
    };
    var result: Result = .{ .status = .iteration_limit, .scale_exponent = exponent };
    while (true) {
        result.scaled_off_diagonal_max = offDiagonalMax(T, work, n, ldw);
        if (result.scaled_off_diagonal_max <= stoppingThreshold(T, work, n, ldw, initial_max)) break;
        if (result.sweeps == options.max_sweeps) return result;
        for (0..n) |p| for (p + 1..n) |q| {
            if (!rotate(T, work, vectors, n, ldw, ldv, p, q)) {
                result.status = .nonfinite_iteration;
                result.failure_index = p + 1;
                result.scaled_off_diagonal_max = offDiagonalMax(T, work, n, ldw);
                return result;
            }
        };
        result.sweeps += 1;
    }
    sortPairs(T, work, vectors, n, ldw, ldv);
    // Check every final rescaling before writing any eigenvalue.
    for (0..n) |i| if (!std.math.isFinite(powerScale(T, work.values[i + i * ldw], exponent))) {
        result.status = .eigenvalue_out_of_range;
        result.failure_index = i + 1;
        return result;
    };
    for (0..n) |i| eigenvalues[i] = powerScale(T, work.values[i + i * ldw], exponent);
    result.status = .converged;
    return result;
}
