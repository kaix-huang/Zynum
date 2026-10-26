// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Finite complex rank-revealing minimum-norm least squares, all rectangular shapes.
//! Borrowed native descriptors and scratch: no context, allocation or backend.
const std = @import("std");
const cx = @import("../../numerics/complex.zig");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
const svd = @import("complex_svd.zig");
const reconstruct = @import("complex_minimum_norm.zig");
pub const Real = cx.scalar.Real;
pub const Matrix = dense.Matrix;
pub const ConstMatrix = dense.ConstMatrix;
pub const WorkspaceScalar = cx.WideComplex;
pub const Error = svd.Error || reconstruct.Error;
pub const Options = svd.Options;
pub const Result = struct { status: svd.Status, rank: usize, sweeps: usize };
pub const Plan = struct {
    scalar_elements: usize,
    wide_elements: usize,
    a_elements: usize,
    v_elements: usize,
    singular_elements: usize,
    transpose_elements: usize,
    residual_elements: usize,
    complement_elements: usize,
};
fn mul(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch error.SizeOverflow;
}
fn plus(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch error.SizeOverflow;
}
pub fn plan(comptime C: type, m: usize, n: usize, rhs: usize) Error!Plan {
    return planWithResidual(C, m, n, rhs, false);
}
pub fn planWithResidual(comptime C: type, m: usize, n: usize, rhs: usize, include_residual: bool) Error!Plan {
    comptime cx.expectComplex(C);
    const k = @min(m, n);
    const a = try mul(m, n);
    const v = try mul(n, k);
    const trans = try svd.rectangularScratchLength(C, m, n);
    const rows = if (include_residual and rhs != 0) m -| n else 0;
    const residual = try mul(rows, rhs);
    const total = try plus(try plus(a, v), try plus(trans, residual));
    _ = try storage.byteLength(total, @sizeOf(C));
    const complement = if (n != 0) try mul(m, rows) else 0;
    const complete_work = try plus(complement, if (complement != 0) m else 0);
    const algorithm = @max(try svd.workspaceLength(C, @max(m, n), k), @max(try reconstruct.workspaceLength(C, n, rhs), complete_work));
    // The first k WideComplex slots are a real Wide(C) singular-value arena.
    // This internal scratch reinterpretation does not depend on component offsets.
    const wide = try plus(k, algorithm);
    _ = try storage.byteLength(wide, @sizeOf(WorkspaceScalar(C)));
    return .{ .scalar_elements = total, .wide_elements = wide, .a_elements = a, .v_elements = v, .singular_elements = k, .transpose_elements = trans, .residual_elements = residual, .complement_elements = complement };
}
fn range(comptime T: type, values: []const T) Error!storage.ByteRange {
    return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(T)));
}
fn add(comptime W: type, sum: *W, correction: *W, term: W) void {
    const adjusted = term - correction.*;
    const next = sum.* + adjusted;
    correction.* = (next - sum.*) - adjusted;
    sum.* = next;
}
fn dot(comptime C: type, a: []const WorkspaceScalar(C), b: []const WorkspaceScalar(C)) WorkspaceScalar(C) {
    const W = cx.Wide(C);
    var sum: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
    var correction: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
    for (a, b) |x, y| {
        add(W, &sum.re, &correction.re, x.re * y.re + x.im * y.im);
        add(W, &sum.im, &correction.im, x.re * y.im - x.im * y.re);
    }
    return sum;
}
fn norm(comptime C: type, values: []const WorkspaceScalar(C)) cx.Wide(C) {
    const W = cx.Wide(C);
    var scale: W = 0;
    for (values) |x| scale = @max(scale, @max(@abs(x.re), @abs(x.im)));
    if (scale == 0) return 0;
    var sum: W = 0;
    var correction: W = 0;
    for (values) |x| {
        const r = x.re / scale;
        const i = x.im / scale;
        add(W, &sum, &correction, r * r + i * i);
    }
    return scale * @sqrt(sum);
}
fn subtract(comptime C: type, values: []WorkspaceScalar(C), basis: []const WorkspaceScalar(C), coefficient: WorkspaceScalar(C)) void {
    for (values, basis) |*x, q| {
        x.re -= q.re * coefficient.re - q.im * coefficient.im;
        x.im -= q.re * coefficient.im + q.im * coefficient.re;
    }
}

// Form an orthonormal complement using largest projected coordinate directions.
// Reorthogonalize twice, then verify the complete prospective rounded basis.
// Scratch only; no public output changes if the requested tolerance is unattainable.
fn complete(comptime C: type, u: dense.ConstMatrix(C), complement: []WorkspaceScalar(C), candidate: []WorkspaceScalar(C), tolerance: f64) bool {
    const W = cx.Wide(C);
    const m = u.row_count;
    const n = u.column_count;
    const d = m - n;
    for (0..d) |j| {
        var largest: W = 0;
        for (0..m) |axis| {
            @memset(candidate, .{ .re = 0, .im = 0 });
            candidate[axis].re = 1;
            for (0..2) |_| {
                for (0..n) |q| {
                    var z: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
                    var corr: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
                    for (candidate, 0..) |x, r| {
                        const v = cx.widen(C, u.values[r + q * u.leading_dimension]);
                        add(W, &z.re, &corr.re, v.re * x.re + v.im * x.im);
                        add(W, &z.im, &corr.im, v.re * x.im - v.im * x.re);
                    }
                    for (candidate, 0..) |*x, r| {
                        const v = cx.widen(C, u.values[r + q * u.leading_dimension]);
                        x.re -= v.re * z.re - v.im * z.im;
                        x.im -= v.re * z.im + v.im * z.re;
                    }
                }
                for (0..j) |q| {
                    const basis = complement[q * m ..][0..m];
                    subtract(C, candidate, basis, dot(C, basis, candidate));
                }
            }
            const length = norm(C, candidate);
            if (!std.math.isFinite(length)) return false;
            if (length > largest) {
                largest = length;
                @memcpy(complement[j * m ..][0..m], candidate);
            }
        }
        if (largest == 0) return false;
        for (complement[j * m ..][0..m]) |*x| {
            x.re /= largest;
            x.im /= largest;
        }
    }
    for (0..d) |p| {
        for (0..n) |q| {
            var z: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
            var corr: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
            for (0..m) |r| {
                const a = cx.widen(C, u.values[r + q * u.leading_dimension]);
                const b = cx.widen(C, cx.narrow(C, complement[r + p * m]) orelse return false);
                add(W, &z.re, &corr.re, a.re * b.re + a.im * b.im);
                add(W, &z.im, &corr.im, a.re * b.im - a.im * b.re);
            }
            if (!cx.finite(WorkspaceScalar(C), z) or std.math.hypot(z.re, z.im) > @as(W, tolerance)) return false;
        }
        for (p..d) |q| {
            var z: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
            var corr: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
            for (0..m) |r| {
                const a = cx.widen(C, cx.narrow(C, complement[r + p * m]) orelse return false);
                const b = cx.widen(C, cx.narrow(C, complement[r + q * m]) orelse return false);
                add(W, &z.re, &corr.re, a.re * b.re + a.im * b.im);
                add(W, &z.im, &corr.im, a.re * b.im - a.im * b.re);
            }
            z.re -= if (p == q) @as(W, 1) else 0;
            if (!cx.finite(WorkspaceScalar(C), z) or std.math.hypot(z.re, z.im) > @as(W, tolerance)) return false;
        }
    }
    return true;
}
pub fn solve(comptime C: type, a: dense.ConstMatrix(C), b: dense.ConstMatrix(C), x: dense.Matrix(C), singular: []Real(C), rcond: Real(C), scalar_scratch: []C, wide_scratch: []WorkspaceScalar(C), options: Options) Error!Result {
    return solveImpl(C, a, b, x, singular, null, rcond, scalar_scratch, wide_scratch, options);
}
pub fn solveWithResidual(comptime C: type, a: dense.ConstMatrix(C), b: dense.ConstMatrix(C), x: dense.Matrix(C), singular: []Real(C), residual: dense.Matrix(C), rcond: Real(C), scalar_scratch: []C, wide_scratch: []WorkspaceScalar(C), options: Options) Error!Result {
    return solveImpl(C, a, b, x, singular, residual, rcond, scalar_scratch, wide_scratch, options);
}
/// A/B are immutable; X/S/residual commit together only on success. For a tall
/// full-column-rank system residual is an (m-n)-by-r array of complex orthogonal
/// coordinates whose squared magnitudes sum to ||B-A*X||^2 per RHS. It remains
/// untouched on rank deficiency. All used writable/read-only spans are checked
/// for byte overlap before any work; padding and unused suffixes are preserved.
fn solveImpl(comptime C: type, a: dense.ConstMatrix(C), b: dense.ConstMatrix(C), x: dense.Matrix(C), singular: []Real(C), residual: ?dense.Matrix(C), rcond: Real(C), scalar_scratch: []C, wide_scratch: []WorkspaceScalar(C), options: Options) Error!Result {
    comptime cx.expectComplex(C);
    const W = cx.Wide(C);
    const Z = WorkspaceScalar(C);
    const aspan = try a.check();
    const bspan = try b.check();
    const xspan = try x.check();
    const m = a.row_count;
    const n = a.column_count;
    if (b.row_count != m or x.row_count != n or b.column_count != x.column_count) return error.DimensionMismatch;
    if (!std.math.isFinite(rcond)) return error.InvalidTolerance;
    const rs = if (residual) |r| try r.check() else 0;
    if (residual) |r| {
        if (r.row_count != m -| n or r.column_count != b.column_count) return error.DimensionMismatch;
    }
    const needed = try planWithResidual(C, m, n, b.column_count, residual != null);
    const k = needed.singular_elements;
    if (singular.len < k or scalar_scratch.len < needed.scalar_elements or wide_scratch.len < needed.wide_elements) return error.BufferTooSmall;
    const writes = [_]storage.ByteRange{ try range(C, x.values[0..xspan]), try range(Real(C), singular[0..k]), try range(C, if (residual) |r| r.values[0..rs] else &.{}), try range(C, scalar_scratch[0..needed.scalar_elements]), try range(Z, wide_scratch[0..needed.wide_elements]) };
    const reads = [_]storage.ByteRange{ try range(C, a.values[0..aspan]), try range(C, b.values[0..bspan]) };
    for (writes, 0..) |out, i| {
        for (writes[i + 1 ..]) |other| if (storage.overlaps(out, other)) return error.AliasingNotAllowed;
        for (reads) |in| if (storage.overlaps(out, in)) return error.AliasingNotAllowed;
    }
    // Validate the RHS even for zero retained singular directions or zero columns.
    if (m != 0) for (0..b.column_count) |j| for (b.values[j * b.leading_dimension ..][0..m]) |entry| if (!cx.finite(C, entry)) return error.NonFiniteInput;
    const fa = scalar_scratch[0..needed.a_elements];
    const fv = scalar_scratch[needed.a_elements..][0..needed.v_elements];
    const ts = scalar_scratch[needed.a_elements + needed.v_elements ..][0..needed.transpose_elements];
    const rv = scalar_scratch[needed.a_elements + needed.v_elements + needed.transpose_elements ..][0..needed.residual_elements];
    const values = if (k == 0) @as([]W, &.{}) else @as([*]W, @ptrCast(wide_scratch.ptr))[0..k];
    const work = wide_scratch[k..needed.wide_elements];
    if (m != 0) for (0..n) |j| @memcpy(fa[j * m ..][0..m], a.values[j * a.leading_dimension ..][0..m]);
    const factor: dense.Matrix(C) = .{ .values = fa, .row_count = m, .column_count = n, .leading_dimension = m };
    const right: dense.Matrix(C) = .{ .values = fv, .row_count = n, .column_count = k, .leading_dimension = n };
    const decomposed = try svd.decomposeRectangularWithSingularType(C, W, factor, values, right, work, ts, options);
    if (decomposed.status != .converged) return .{ .status = decomposed.status, .rank = 0, .sweeps = decomposed.sweeps };
    for (values) |s| if (!std.math.isFinite(@as(Real(C), @floatCast(s)))) return .{ .status = .non_finite_result, .rank = 0, .sweeps = decomposed.sweeps };
    const rank = reconstruct.rankFromWideSingular(C, W, values, rcond);
    const emit = residual != null and b.column_count != 0 and m > n and rank == n;
    const u: dense.ConstMatrix(C) = .{ .values = fa, .row_count = m, .column_count = k, .leading_dimension = m };
    if (emit) {
        const d = m - n;
        if (n == 0) {
            for (0..b.column_count) |j| @memcpy(rv[j * m ..][0..m], b.values[j * b.leading_dimension ..][0..m]);
        } else {
            const complement = work[0..needed.complement_elements];
            const candidate = work[needed.complement_elements..][0..m];
            const tolerance = options.relative_tolerance orelse 32 * @as(f64, std.math.floatEps(Real(C)));
            if (!complete(C, u, complement, candidate, tolerance)) return .{ .status = .range_limit, .rank = rank, .sweeps = decomposed.sweeps };
            for (0..b.column_count) |j| for (0..d) |i| {
                var z: Z = .{ .re = 0, .im = 0 };
                var corr: Z = .{ .re = 0, .im = 0 };
                for (0..m) |r| {
                    const q = complement[r + i * m];
                    const in = cx.widen(C, b.values[r + j * b.leading_dimension]);
                    add(W, &z.re, &corr.re, q.re * in.re + q.im * in.im);
                    add(W, &z.im, &corr.im, q.re * in.im - q.im * in.re);
                }
                rv[i + j * d] = cx.narrow(C, z) orelse return .{ .status = .non_finite_result, .rank = rank, .sweeps = decomposed.sweeps };
            };
        }
    }
    const solved = try reconstruct.solveWithSingularType(C, W, u, values, right.asConst(), b, x, rcond, work);
    if (solved.status != .success) return .{ .status = .non_finite_result, .rank = solved.rank, .sweeps = decomposed.sweeps };
    for (values, singular[0..k]) |s, *out| out.* = @floatCast(s);
    if (emit) {
        const r = residual.?;
        for (0..r.column_count) |j| @memcpy(r.values[j * r.leading_dimension ..][0..r.row_count], rv[j * r.row_count ..][0..r.row_count]);
    }
    return .{ .status = .converged, .rank = solved.rank, .sweeps = decomposed.sweeps };
}
