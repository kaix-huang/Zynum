// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Transactional completion of a borrowed orthonormal/unitary column basis.
//! No admission, allocation or resource owner. Known columns stay bitwise fixed;
//! candidate columns are rounded and checked in caller scratch before commit.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
const scalar = @import("../../blas/core/shared/scalar.zig");
const real_svd = @import("svd.zig");
const complex_svd = @import("complex_svd.zig");
pub const Error = real_svd.Error;
pub const Options = struct { relative_tolerance: ?f64 = null };
pub const Result = struct { status: real_svd.Status, output_orthogonality: f64 };

pub fn WorkspaceScalar(comptime T: type) type {
    return if (comptime scalar.isComplex(T)) complex_svd.WorkspaceScalar(T) else real_svd.WorkspaceScalar(T);
}
fn Wide(comptime T: type) type {
    return if (comptime scalar.isComplex(T)) complex_svd.Wide(T) else WorkspaceScalar(T);
}
pub fn workspaceLength(comptime T: type, dimension: usize) Error!usize {
    const square = std.math.mul(usize, dimension, dimension) catch return error.InvalidLength;
    const used = std.math.add(usize, square, dimension) catch return error.InvalidLength;
    _ = storage.byteLength(used, @sizeOf(WorkspaceScalar(T))) catch return error.InvalidLength;
    return used;
}
fn value(comptime T: type, re: Wide(T), im: Wide(T)) WorkspaceScalar(T) {
    return if (comptime scalar.isComplex(T)) .{ .re = re, .im = im } else re;
}
fn real(comptime T: type, z: WorkspaceScalar(T)) Wide(T) {
    return if (comptime scalar.isComplex(T)) z.re else z;
}
fn imag(comptime T: type, z: WorkspaceScalar(T)) Wide(T) {
    return if (comptime scalar.isComplex(T)) z.im else 0;
}
fn widen(comptime T: type, z: T) WorkspaceScalar(T) {
    return if (comptime scalar.isComplex(T)) .{ .re = z.re, .im = z.im } else z;
}
fn narrow(comptime T: type, z: WorkspaceScalar(T)) T {
    return if (comptime scalar.isComplex(T)) .{ .re = @floatCast(z.re), .im = @floatCast(z.im) } else @floatCast(z);
}
fn finite(comptime T: type, z: T) bool {
    return if (comptime scalar.isComplex(T)) std.math.isFinite(z.re) and std.math.isFinite(z.im) else std.math.isFinite(z);
}
fn add(comptime W: type, sum: *W, compensation: *W, term: W) void {
    const adjusted = term - compensation.*;
    const next = sum.* + adjusted;
    compensation.* = (next - sum.*) - adjusted;
    sum.* = next;
}
fn dot(comptime T: type, x: []const WorkspaceScalar(T), y: []const WorkspaceScalar(T)) WorkspaceScalar(T) {
    const W = Wide(T);
    var re: W = 0;
    var im: W = 0;
    var cr: W = 0;
    var ci: W = 0;
    for (x, y) |a, b| {
        add(W, &re, &cr, real(T, a) * real(T, b));
        if (comptime scalar.isComplex(T)) {
            add(W, &re, &cr, imag(T, a) * imag(T, b));
            add(W, &im, &ci, real(T, a) * imag(T, b));
            add(W, &im, &ci, -imag(T, a) * real(T, b));
        }
    }
    return value(T, re, im);
}
fn norm(comptime T: type, x: []const WorkspaceScalar(T)) Wide(T) {
    const W = Wide(T);
    var scale: W = 0;
    for (x) |z| scale = @max(scale, @max(@abs(real(T, z)), @abs(imag(T, z))));
    if (scale == 0) return 0;
    var sum: W = 0;
    var compensation: W = 0;
    for (x) |z| {
        const re = real(T, z) / scale;
        add(W, &sum, &compensation, re * re);
        if (comptime scalar.isComplex(T)) {
            const im = imag(T, z) / scale;
            add(W, &sum, &compensation, im * im);
        }
    }
    return scale * @sqrt(sum);
}
fn projectSeed(comptime T: type, columns: []WorkspaceScalar(T), d: usize, count: usize, seed: usize, candidate: []WorkspaceScalar(T)) void {
    const W = Wide(T);
    @memset(candidate, value(T, 0, 0));
    candidate[seed] = value(T, 1, 0);
    // Two modified Gram-Schmidt passes; each Hermitian projection is summed
    // with compensation in the existing widened SVD workspace scalar.
    for (0..2) |_| for (0..count) |j| {
        const q = columns[j * d ..][0..d];
        const projection = dot(T, q, candidate);
        const pr = real(T, projection);
        const pi = imag(T, projection);
        for (candidate, q) |*x, y| {
            const xr = real(T, x.*);
            const xi = imag(T, x.*);
            const yr = real(T, y);
            const yi = imag(T, y);
            x.* = value(T, @mulAdd(W, -pr, yr, @mulAdd(W, pi, yi, xr)), @mulAdd(W, -pr, yi, @mulAdd(W, -pi, yr, xi)));
        }
    };
}
fn orthogonality(comptime T: type, columns: []const WorkspaceScalar(T), d: usize, count: usize) f64 {
    const W = Wide(T);
    var largest: W = 0;
    for (0..count) |p| for (p..count) |q| {
        const product = dot(T, columns[p * d ..][0..d], columns[q * d ..][0..d]);
        const re = real(T, product) - @as(W, if (p == q) 1 else 0);
        const im = imag(T, product);
        const error_size = if (comptime scalar.isComplex(T)) std.math.hypot(re, im) else @abs(re);
        if (!std.math.isFinite(error_size)) return std.math.inf(f64);
        largest = @max(largest, error_size);
    };
    const rounded: f64 = @floatCast(largest);
    return if (@as(W, rounded) < largest) std.math.nextAfter(f64, rounded, std.math.inf(f64)) else rounded;
}
fn span(comptime T: type, basis: dense.Matrix(T)) Error!usize {
    if (basis.row_count == 0) return 0;
    if (basis.leading_dimension < basis.row_count) return error.InvalidLeadingDimension;
    const used = storage.columnMajorExtent(basis.row_count, basis.column_count, basis.leading_dimension) catch return error.InvalidLength;
    if (basis.values.len < used) return error.BufferTooSmall;
    return used;
}
fn range(comptime T: type, values: []const T) Error!storage.ByteRange {
    const bytes = storage.byteLength(values.len, @sizeOf(T)) catch return error.InvalidLength;
    return storage.byteRange(@intFromPtr(values.ptr), bytes) catch return error.InvalidLength;
}

/// Complete a d-by-d column-major basis. Active basis/workspace byte spans,
/// including intervening basis padding, must be disjoint. Unused capacity is
/// outside those operands and may alias another active view. Only known columns
/// are read; a failure does not change the physical basis operand.
/// Numerical rejection returns range_limit without changing basis; scratch may
/// change on error, but workspace beyond workspaceLength is never written.
pub fn complete(comptime T: type, basis: dense.Matrix(T), known_columns: usize, workspace: []WorkspaceScalar(T), options: Options) Error!Result {
    const d = basis.row_count;
    if (basis.column_count != d or known_columns > d) return error.DimensionMismatch;
    const tolerance = options.relative_tolerance orelse 32 * @as(f64, std.math.floatEps(scalar.Real(T)));
    if (!std.math.isFinite(tolerance) or tolerance <= 0 or tolerance >= 1) return error.InvalidTolerance;
    const basis_used = try span(T, basis);
    const used = try workspaceLength(T, d);
    if (workspace.len < used) return error.BufferTooSmall;
    const br = try range(T, basis.values[0..basis_used]);
    const wr = try range(WorkspaceScalar(T), workspace[0..used]);
    if (storage.overlaps(br, wr)) return error.AliasingNotAllowed;
    const columns = workspace[0 .. used - d];
    const candidate = workspace[used - d .. used];
    for (0..known_columns) |j| for (0..d) |i| {
        const x = basis.values[j * basis.leading_dimension + i];
        if (!finite(T, x)) return error.NonFiniteInput;
        columns[j * d + i] = widen(T, x);
    };
    const known_error = orthogonality(T, columns, d, known_columns);
    if (known_error > tolerance) return .{ .status = .range_limit, .output_orthogonality = known_error };
    for (known_columns..d) |j| {
        var best_norm: Wide(T) = 0;
        const next = columns[j * d ..][0..d];
        for (0..d) |seed| {
            projectSeed(T, columns, d, j, seed, candidate);
            const length = norm(T, candidate);
            if (std.math.isFinite(length) and length > best_norm) {
                best_norm = length;
                @memcpy(next, candidate);
            }
        }
        if (best_norm == 0) return .{ .status = .range_limit, .output_orthogonality = std.math.inf(f64) };
        for (next) |*x| {
            x.* = value(T, real(T, x.*) / best_norm, imag(T, x.*) / best_norm);
            // Check exactly the final stored-T vector, then reuse that rounded
            // vector in later projections. There is no unchecked final cast.
            const rounded = narrow(T, x.*);
            if (!finite(T, rounded)) return .{ .status = .range_limit, .output_orthogonality = std.math.inf(f64) };
            x.* = widen(T, rounded);
        }
    }
    const output_error = orthogonality(T, columns, d, d);
    if (!std.math.isFinite(output_error) or output_error > tolerance) return .{ .status = .range_limit, .output_orthogonality = output_error };
    for (known_columns..d) |j| {
        for (0..d) |i| basis.values[j * basis.leading_dimension + i] = narrow(T, columns[j * d + i]);
    }
    return .{ .status = .converged, .output_orthogonality = output_error };
}
