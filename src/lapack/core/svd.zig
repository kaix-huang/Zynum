// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Real thin one-sided row-cyclic Jacobi SVD, m>=n.
//! Borrowed native-width operands; the caller owns admission and workspace.
//! Original unblocked foundation, not LAPACK's optimized DGESVJ algorithm.
const std = @import("std");
const storage = @import("../../numerics/storage.zig");
const dense = @import("../../numerics/dense.zig");
const validation = @import("../validation.zig");
pub const Error = validation.Error;
pub const Status = enum { converged, no_convergence, non_finite_result, range_limit };
pub const Result = struct {
    status: Status,
    sweeps: usize,
    max_correlation: f64,
    output_orthogonality: f64 = 0,
    deflated_columns: usize = 0,
};
pub const Options = struct { max_sweeps: usize = 64, relative_tolerance: ?f64 = null };

/// Widen both the precision and exponent range of the iterated A and V.
pub fn WorkspaceScalar(comptime T: type) type {
    if (T != f32 and T != f64) @compileError("SVD supports f32 and f64");
    return if (T == f32) f64 else f128;
}

/// Exact workspace length in WorkspaceScalar(T) elements: m*n+n*n+n.
/// Both element and byte arithmetic are overflow checked. Empty n returns zero.
pub fn workspaceLength(comptime T: type, m: usize, n: usize) Error!usize {
    const W = WorkspaceScalar(T);
    if (m < n) return error.DimensionMismatch;
    const rows: usize = @intCast(m);
    const columns: usize = @intCast(n);
    const a = std.math.mul(usize, rows, columns) catch return error.InvalidLength;
    const v = std.math.mul(usize, columns, columns) catch return error.InvalidLength;
    const av = std.math.add(usize, a, v) catch return error.InvalidLength;
    const used = std.math.add(usize, av, columns) catch return error.InvalidLength;
    _ = storage.byteLength(used, @sizeOf(W)) catch return error.InvalidLength;
    return used;
}
fn span(a: anytype) Error!usize {
    if (a.row_count == 0 or a.column_count == 0) return 0;
    if (a.leading_dimension < a.row_count) return error.InvalidLeadingDimension;
    const used = storage.columnMajorExtent(@intCast(a.row_count), @intCast(a.column_count), @intCast(a.leading_dimension)) catch return error.InvalidLength;
    if (a.values.len < used) return error.BufferTooSmall;
    return used;
}
fn disjoint(comptime A: type, a: []const A, comptime B: type, b: []const B) Error!void {
    const alen = storage.byteLength(a.len, @sizeOf(A)) catch return error.InvalidLength;
    const blen = storage.byteLength(b.len, @sizeOf(B)) catch return error.InvalidLength;
    const ap = @intFromPtr(a.ptr);
    const bp = @intFromPtr(b.ptr);
    const ar = storage.byteRange(ap, alen) catch return error.InvalidLength;
    const br = storage.byteRange(bp, blen) catch return error.InvalidLength;
    if (storage.overlaps(ar, br)) return error.AliasingNotAllowed;
}
fn Columns(comptime W: type) type {
    return struct {
        data: []W,
        rows: usize,
        count: usize,
        fn column(self: @This(), j: usize) []W {
            return self.data[j * self.rows ..][0..self.rows];
        }
    };
}
fn add(comptime W: type, sum: *W, compensation: *W, term: W) void {
    const adjusted = term - compensation.*;
    const next = sum.* + adjusted;
    compensation.* = (next - sum.*) - adjusted;
    sum.* = next;
}
fn norm(comptime W: type, x: []const W) W {
    var scale: W = 0;
    for (x) |v| scale = @max(scale, @abs(v));
    if (scale == 0) return 0;
    var sum: W = 0;
    var compensation: W = 0;
    for (x) |v| {
        const q = v / scale;
        add(W, &sum, &compensation, q * q);
    }
    return scale * @sqrt(sum);
}
fn correlation(comptime W: type, x: []const W, y: []const W, nx: W, ny: W) W {
    if (nx == 0 or ny == 0) return 0;
    var sum: W = 0;
    var compensation: W = 0;
    for (x, y) |a, b| add(W, &sum, &compensation, (a / nx) * (b / ny));
    return @max(-1, @min(1, sum));
}
fn rotate(comptime W: type, x: []W, y: []W, c: W, s: W) bool {
    for (x, y) |*a, *b| {
        const old_a = a.*;
        const old_b = b.*;
        const next_a = @mulAdd(W, c, old_a, -s * old_b);
        const next_b = @mulAdd(W, s, old_a, c * old_b);
        if (!std.math.isFinite(next_a) or !std.math.isFinite(next_b)) return false;
        a.* = next_a;
        b.* = next_b;
    }
    return true;
}
fn maxCorrelation(comptime W: type, a: Columns(W), norms: []W) ?f64 {
    for (0..a.count) |j| {
        norms[j] = norm(W, a.column(j));
        if (!std.math.isFinite(norms[j])) return null;
    }
    var largest: W = 0;
    for (0..a.count) |p| for (p + 1..a.count) |q| {
        largest = @max(largest, @abs(correlation(W, a.column(p), a.column(q), norms[p], norms[q])));
    };
    return metricToF64(W, largest);
}
fn metricToF64(comptime W: type, value: W) f64 {
    const rounded: f64 = @floatCast(value);
    return if (@as(W, rounded) < value) std.math.nextAfter(f64, rounded, std.math.inf(f64)) else rounded;
}
fn upward(comptime W: type, x: W) W {
    return std.math.nextAfter(W, x, std.math.inf(W));
}
fn zeroOutput(comptime T: type, length: WorkspaceScalar(T)) bool {
    return @as(T, @floatCast(length)) == 0;
}

// A set of individually unrepresentable column norms can jointly carry a
// representable singular value. Drop it only when a conservative upper bound
// on the entire discarded Frobenius norm, including earlier drops, rounds to
// zero in T. Each positive arithmetic step is rounded upward by one W ulp.
// The cumulative sum is conservative (triangle inequality), not a rank cutoff.
fn deflateTiny(comptime T: type, a: Columns(WorkspaceScalar(T)), norms: []WorkspaceScalar(T), discarded_bound: *WorkspaceScalar(T)) usize {
    const W = WorkspaceScalar(T);
    var scale: W = 0;
    for (0..a.count) |j| {
        norms[j] = norm(W, a.column(j));
        if (norms[j] != 0 and zeroOutput(T, norms[j])) {
            for (a.column(j)) |x| scale = @max(scale, @abs(x));
        }
    }
    if (scale == 0) return 0;
    var square_sum: W = 0;
    for (0..a.count) |j| if (norms[j] != 0 and zeroOutput(T, norms[j])) {
        for (a.column(j)) |x| if (x != 0) {
            const quotient = upward(W, @abs(x) / scale);
            const square = upward(W, quotient * quotient);
            square_sum = upward(W, square_sum + square);
        };
    };
    const bound = upward(W, scale * upward(W, @sqrt(square_sum)));
    const total_bound = upward(W, discarded_bound.* + bound);
    if (!std.math.isFinite(total_bound) or !zeroOutput(T, total_bound)) return 0;
    var removed: usize = 0;
    for (0..a.count) |j| if (norms[j] != 0 and zeroOutput(T, norms[j])) {
        @memset(a.column(j), 0);
        norms[j] = 0;
        removed += 1;
    };
    discarded_bound.* = total_bound;
    return removed;
}

fn roundedOrthogonality(comptime T: type, a: Columns(WorkspaceScalar(T))) f64 {
    const W = WorkspaceScalar(T);
    var largest: W = 0;
    for (0..a.count) |p| for (p..a.count) |q| {
        var dot: W = 0;
        var compensation: W = 0;
        for (a.column(p), a.column(q)) |x, y| {
            const rx: T = @floatCast(x);
            const ry: T = @floatCast(y);
            if (!std.math.isFinite(rx) or !std.math.isFinite(ry)) return std.math.inf(f64);
            add(W, &dot, &compensation, @as(W, rx) * @as(W, ry));
        }
        if (!std.math.isFinite(dot)) return std.math.inf(f64);
        largest = @max(largest, @abs(dot - @as(W, if (p == q) 1 else 0)));
    };
    return metricToF64(W, largest);
}

// On entry the nonzero columns satisfy the working correlation criterion.
// All finalization occurs in caller workspace until every range/orthogonality
// check passes. There are no failure branches after committing public outputs.
fn finish(comptime T: type, a: Columns(WorkspaceScalar(T)), right: Columns(WorkspaceScalar(T)), lengths: []WorkspaceScalar(T), tolerance: f64, result: *Result) bool {
    const W = WorkspaceScalar(T);
    for (0..a.count) |j| {
        lengths[j] = norm(W, a.column(j));
        if (!std.math.isFinite(@as(T, @floatCast(lengths[j])))) {
            result.status = .non_finite_result;
            return false;
        }
    }
    for (0..a.count) |j| {
        var largest = j;
        for (j + 1..a.count) |k| if (lengths[k] > lengths[largest]) {
            largest = k;
        };
        if (largest != j) {
            std.mem.swap(W, &lengths[j], &lengths[largest]);
            for (a.column(j), a.column(largest)) |*x, *y| std.mem.swap(W, x, y);
            for (right.column(j), right.column(largest)) |*x, *y| std.mem.swap(W, x, y);
        }
    }
    for (0..a.count) |j| {
        const column = a.column(j);
        if (!zeroOutput(T, lengths[j])) {
            for (column) |*x| x.* /= lengths[j];
            continue;
        }
        // Only zero-output singular directions are completed. This threshold
        // selects a usable basis vector; it is never applied to singular values.
        var found = false;
        for (0..a.rows) |basis| {
            @memset(column, 0);
            column[basis] = 1;
            for (0..2) |_| for (0..j) |previous| {
                const vector = a.column(previous);
                var dot: W = 0;
                var compensation: W = 0;
                for (column, vector) |x, y| add(W, &dot, &compensation, x * y);
                for (column, vector) |*x, y| x.* = @mulAdd(W, -dot, y, x.*);
            };
            const candidate_norm = norm(W, column);
            if (candidate_norm > 0.25 / @sqrt(@as(W, @floatFromInt(a.rows)))) {
                for (column) |*x| x.* /= candidate_norm;
                found = true;
                break;
            }
        }
        if (!found) {
            result.status = .range_limit;
            return false;
        }
    }
    result.output_orthogonality = @max(roundedOrthogonality(T, a), roundedOrthogonality(T, right));
    if (!std.math.isFinite(result.output_orthogonality) or result.output_orthogonality > tolerance) {
        result.status = .range_limit;
        return false;
    }
    return true;
}

/// Thin real SVD A=U*diag(singular)*V^T for m>=n. U overwrites A, and right is
/// n-by-n V. Singular values descend; vector signs and repeated-subspace bases
/// are unspecified. Supply workspaceLength(T,m,n) WorkspaceScalar(T) elements:
/// contiguous m-by-n iterated A, n-by-n iterated V, and n column norms. No
/// allocation/backend is used. Output padding/workspace suffix are not written
/// directly. Unused storage may alias a different active operand.
///
/// All dimensions, capacities, finite input, tolerance and mixed-type byte-span
/// aliases are checked before mutation, including fast builds. All four used
/// operand spans are pairwise disjoint; matrix spans include intervening padding.
/// Workspace old contents and right/singular old contents are not read. Empty
/// shapes accept nonnegative leading dimensions. A, right and singular remain
/// unchanged on ANY unsuccessful return; workspace may contain partial work.
///
/// Default tolerance is 32*eps(T). Each complete sweep is followed by a full
/// working column-correlation check. max_sweeps=0 permits already orthogonal
/// input. Widened storage prevents T-subnormal quantization during iteration;
/// scaled norms and compensated normalized dots limit range loss. Only a whole
/// set of tiny columns whose conservatively bounded cumulative discarded norm
/// rounds to zero in T may be deflated before convergence. No relative rank
/// threshold is used. Computed zero-output directions receive an orthonormal
/// completion. Roundoff can still turn algebraic zeros into small nonzero
/// singular values; this is not a rank estimator or an exact/relative-accuracy
/// guarantee for arbitrary ill-conditioned matrices.
///
/// Success also requires the prospective T-rounded U and V to satisfy
/// max(abs(Q^T*Q-I)) <= tolerance, reported as output_orthogonality. An
/// unattainable requested tolerance, stalled rotation or failed completion
/// returns range_limit; no_convergence reports an exhausted sweep budget.
/// max_correlation describes the last full working check (infinity for an
/// unrepresentable norm). Out-of-range singular values return non_finite_result.
/// Widened storage, including f128 arithmetic for f64, is an unmeasured reference
/// approach, not LAPACK DGESVJ equivalence or a performance claim.
pub fn decompose(comptime T: type, a: dense.Matrix(T), singular: []T, right: dense.Matrix(T), workspace: []WorkspaceScalar(T), options: Options) Error!Result {
    const W = WorkspaceScalar(T);
    const aspan = try span(a);
    const vspan = try span(right);
    const used = try workspaceLength(T, a.row_count, a.column_count);
    if (right.row_count != a.column_count or right.column_count != a.column_count) return error.DimensionMismatch;
    const m: usize = @intCast(a.row_count);
    const n: usize = @intCast(a.column_count);
    if (singular.len < n or workspace.len < used) return error.BufferTooSmall;
    const tolerance = options.relative_tolerance orelse 32 * @as(f64, std.math.floatEps(T));
    if (!std.math.isFinite(tolerance) or tolerance <= 0 or tolerance >= 1) return error.InvalidTolerance;
    try disjoint(T, a.values[0..aspan], T, singular[0..n]);
    try disjoint(T, a.values[0..aspan], T, right.values[0..vspan]);
    try disjoint(T, singular[0..n], T, right.values[0..vspan]);
    try disjoint(T, a.values[0..aspan], W, workspace[0..used]);
    try disjoint(T, singular[0..n], W, workspace[0..used]);
    try disjoint(T, right.values[0..vspan], W, workspace[0..used]);
    const lda: usize = @intCast(a.leading_dimension);
    const ldv: usize = @intCast(right.leading_dimension);
    for (0..n) |j| for (a.values[j * lda ..][0..m]) |entry| if (!std.math.isFinite(entry)) {
        return error.NonFiniteInput;
    };
    var result: Result = .{ .status = .no_convergence, .sweeps = 0, .max_correlation = 0 };
    if (n == 0) {
        result.status = .converged;
        return result;
    }
    const aw: Columns(W) = .{ .data = workspace[0 .. m * n], .rows = m, .count = n };
    const vw: Columns(W) = .{ .data = workspace[m * n ..][0 .. n * n], .rows = n, .count = n };
    const lengths = workspace[m * n + n * n ..][0..n];
    for (0..n) |j| {
        for (aw.column(j), a.values[j * lda ..][0..m]) |*x, y| x.* = y;
        for (vw.column(j), 0..) |*x, i| x.* = if (i == j) 1 else 0;
    }
    var discarded_bound: W = 0;
    while (true) {
        result.deflated_columns += deflateTiny(T, aw, lengths, &discarded_bound);
        result.max_correlation = maxCorrelation(W, aw, lengths) orelse {
            result.status = .non_finite_result;
            result.max_correlation = std.math.inf(f64);
            return result;
        };
        if (result.max_correlation <= tolerance) break;
        if (result.sweeps == options.max_sweeps) return result;
        for (0..n) |p| for (p + 1..n) |q| {
            const x = aw.column(p);
            const y = aw.column(q);
            const nx = norm(W, x);
            const ny = norm(W, y);
            if (!std.math.isFinite(nx) or !std.math.isFinite(ny)) {
                result.status = .non_finite_result;
                return result;
            }
            const rho = correlation(W, x, y, nx, ny);
            if (@abs(rho) <= tolerance) continue;
            const ratio = @min(nx, ny) / @max(nx, ny);
            const delta = if (nx > ny) ratio * ratio - 1 else 1 - ratio * ratio;
            const cross = rho * ratio;
            const denominator = delta + std.math.copysign(std.math.hypot(delta, 2 * cross), delta);
            const tangent = 2 * cross / denominator;
            if (tangent == 0 or !std.math.isFinite(tangent)) {
                result.status = .range_limit;
                return result;
            }
            const c = 1 / @sqrt(1 + tangent * tangent);
            const s = tangent * c;
            if (!rotate(W, x, y, c, s) or !rotate(W, vw.column(p), vw.column(q), c, s)) {
                result.status = .non_finite_result;
                return result;
            }
        };
        result.sweeps += 1;
    }
    if (!finish(T, aw, vw, lengths, tolerance, &result)) return result;
    for (0..n) |j| {
        singular[j] = @floatCast(lengths[j]);
        for (a.values[j * lda ..][0..m], aw.column(j)) |*x, y| x.* = @floatCast(y);
        for (right.values[j * ldv ..][0..n], vw.column(j)) |*x, y| x.* = @floatCast(y);
    }
    result.status = .converged;
    return result;
}

// Internal regression for the deflation decision, which cannot be reproduced
// by supplying T-valued input entries below T's own representable range.
test "tiny coupled columns are retained until their combined direction is unrepresentable" {
    inline for (.{ f32, f64 }) |T| {
        const W = WorkspaceScalar(T);
        const minimum: W = std.math.floatTrueMin(T);
        var data: [64]W = @splat(0);
        var lengths: [8]W = undefined;
        const a: Columns(W) = .{ .data = &data, .rows = 8, .count = 8 };
        for (0..8) |j| a.column(j)[0] = minimum / @sqrt(@as(W, 8));
        var bound: W = 0;
        try std.testing.expectEqual(@as(usize, 0), deflateTiny(T, a, &lengths, &bound));
        for (lengths) |length| try std.testing.expect(zeroOutput(T, length));
        // The combined rank-one singular value rounds to the nonzero minimum.
        try std.testing.expectEqual(std.math.floatTrueMin(T), @as(T, @floatCast(norm(W, &data))));
        for (&data) |*x| x.* /= 4;
        try std.testing.expectEqual(@as(usize, 8), deflateTiny(T, a, &lengths, &bound));
        try std.testing.expect(bound > 0 and zeroOutput(T, bound));
        // Deflation budgets accumulate: a second equally small set must not
        // cross the half-minimum rounding boundary and erase a combined value.
        for (0..8) |j| a.column(j)[0] = minimum / (4 * @sqrt(@as(W, 8)));
        try std.testing.expectEqual(@as(usize, 0), deflateTiny(T, a, &lengths, &bound));
    }
}

test "prospective output validation does not hide nonfinite casts behind a maximum reduction" {
    inline for (.{ f32, f64 }) |T| {
        const W = WorkspaceScalar(T);
        var data = [_]W{ 2 * @as(W, std.math.floatMax(T)), 1 };
        const a: Columns(W) = .{ .data = &data, .rows = 2, .count = 1 };
        try std.testing.expect(std.math.isInf(roundedOrthogonality(T, a)));
    }
}

/// Additional T-valued scratch for wide inputs: n*m+m*m. Tall inputs need none.
pub fn rectangularScratchLength(comptime T: type, m: usize, n: usize) Error!usize {
    _ = WorkspaceScalar(T);
    if (m >= n) return 0;
    const matrix = std.math.mul(usize, n, m) catch return error.InvalidLength;
    const right = std.math.mul(usize, m, m) catch return error.InvalidLength;
    const count = std.math.add(usize, matrix, right) catch return error.InvalidLength;
    _ = storage.byteLength(count, @sizeOf(T)) catch return error.InvalidLength;
    return count;
}

/// Rectangular thin SVD: first k=min(m,n) columns of A become U; right is n-by-k V.
/// For m<n, decompose A^T with the same core and swap the resulting factor roles.
/// Trailing A columns, matrix padding and scratch suffixes are unchanged.
/// Supply workspaceLength(T,max(m,n),k) widened elements and
/// rectangularScratchLength(T,m,n) T elements. All active storage is disjoint.
/// A/right/singular remain unchanged on failure; both scratch buffers may change.
pub fn decomposeRectangular(comptime T: type, a: dense.Matrix(T), singular: []T, right: dense.Matrix(T), workspace: []WorkspaceScalar(T), transpose_scratch: []T, options: Options) Error!Result {
    if (a.row_count >= a.column_count) return decompose(T, a, singular, right, workspace, options);
    const m = a.row_count;
    const n = a.column_count;
    const aspan = try span(a);
    const vspan = try span(right);
    if (right.row_count != n or right.column_count != m) return error.DimensionMismatch;
    const used = try workspaceLength(T, n, m);
    const temporary = try rectangularScratchLength(T, m, n);
    if (singular.len < m or workspace.len < used or transpose_scratch.len < temporary) return error.BufferTooSmall;
    const tolerance = options.relative_tolerance orelse 32 * @as(f64, std.math.floatEps(T));
    if (!std.math.isFinite(tolerance) or tolerance <= 0 or tolerance >= 1) return error.InvalidTolerance;
    const operands = [_][]T{ a.values[0..aspan], right.values[0..vspan], singular[0..m], transpose_scratch[0..temporary] };
    for (operands, 0..) |operand, i| {
        for (operands[i + 1 ..]) |other| try disjoint(T, operand, T, other);
        try disjoint(T, operand, WorkspaceScalar(T), workspace[0..used]);
    }
    if (m == 0) return .{ .status = .converged, .sweeps = 0, .max_correlation = 0 };
    for (0..n) |j| for (a.values[j * a.leading_dimension ..][0..m]) |entry| {
        if (!std.math.isFinite(entry)) return error.NonFiniteInput;
    };
    const at = transpose_scratch[0 .. n * m];
    const small_right = transpose_scratch[n * m ..][0 .. m * m];
    for (0..n) |j| for (0..m) |i| {
        at[j + i * n] = a.values[i + j * a.leading_dimension];
    };
    const result = try decompose(T, .{ .values = at, .row_count = n, .column_count = m, .leading_dimension = n }, singular, .{ .values = small_right, .row_count = m, .column_count = m, .leading_dimension = m }, workspace, options);
    if (result.status != .converged) return result;
    for (0..m) |j| {
        for (0..m) |i| a.values[i + j * a.leading_dimension] = small_right[i + j * m];
        for (0..n) |i| right.values[i + j * right.leading_dimension] = at[i + j * n];
    }
    return result;
}
