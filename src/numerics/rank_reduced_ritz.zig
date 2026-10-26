// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Rayleigh-Ritz on the numerical range of a positive-semidefinite Gram matrix.
//! Given projected H,S, diagonalize S, retain its positive numerical range and
//! solve W*H*W with W*S*W=I. This is a range-restricted Ritz problem, not a
//! general singular-pencil eigensolver. All matrices are small subspace order;
//! input/output storage may overlap, publication is staged, tails are untouched.
//! Uses the existing Hermitian core and original operation's ContextScratch.
const std = @import("std");
const S = @import("../blas/core/shared/scalar.zig");
const D = @import("dense.zig");
const Store = @import("storage.zig");
const Cx = @import("complex.zig");
const H = @import("../lapack/core/hermitian_eigen.zig");
const E = @import("zynum-execution");
pub const Status = enum { converged, gram_iteration_limit, reduced_iteration_limit, indefinite_metric, range_limit };
pub const Result = struct { status: Status, rank: usize = 0, gram_sweeps: usize = 0, reduced_sweeps: usize = 0 };
pub const Workspace = struct { elements: usize, real_elements: usize, wide_elements: usize };
pub fn Request(comptime T: type) type {
    return struct {
        hamiltonian: D.ConstMatrix(T),
        overlap: D.ConstMatrix(T),
        eigenvalues: []S.Real(T),
        coefficients: D.Matrix(T),
        triangle: H.Triangle = .lower,
        rank_relative_tolerance: ?S.Real(T) = null,
        rank_absolute_tolerance: S.Real(T) = 0,
        eigen_options: H.Options = .{},
    };
}
fn supported(comptime T: type) void {
    if (T != S.ComplexF32 and T != S.ComplexF64) @compileError("complex Ritz precision required");
}
pub fn workspaceLength(comptime T: type, order: usize) !Workspace {
    supported(T);
    const square = std.math.mul(usize, order, order) catch return error.SizeOverflow;
    const elements = std.math.mul(usize, square, 4) catch return error.SizeOverflow;
    const reals = std.math.mul(usize, order, 2) catch return error.SizeOverflow;
    const wides = try H.workspaceLength(T, order);
    _ = try Store.byteLength(elements, @sizeOf(T));
    _ = try Store.byteLength(reals, @sizeOf(S.Real(T)));
    return .{ .elements = elements, .real_elements = reals, .wide_elements = wides };
}
fn range(values: anytype) !Store.ByteRange {
    return Store.byteRange(@intFromPtr(values.ptr), try Store.byteLength(values.len, @sizeOf(@typeInfo(@TypeOf(values)).pointer.child)));
}
fn separate(a: anytype, b: anytype) !void {
    if (Store.overlaps(try range(a), try range(b))) return error.AliasingNotAllowed;
}
fn get(comptime T: type, operand: D.ConstMatrix(T), triangle: H.Triangle, row: usize, column: usize) T {
    if (row == column) return .{ .re = operand.values[row + column * operand.leading_dimension].re, .im = 0 };
    const stored = if (triangle == .lower) row > column else row < column;
    const value = operand.values[if (stored) row + column * operand.leading_dimension else column + row * operand.leading_dimension];
    return if (stored) value else .{ .re = value.re, .im = -value.im };
}
fn options(comptime T: type, q: Request(T)) !void {
    supported(T);
    const n = q.hamiltonian.row_count;
    _ = try q.hamiltonian.check();
    _ = try q.overlap.check();
    const out = q.coefficients.values[0..try q.coefficients.check()];
    if (q.hamiltonian.column_count != n or q.overlap.row_count != n or q.overlap.column_count != n or q.coefficients.row_count != n or q.coefficients.column_count != n or q.eigenvalues.len < n) return error.DimensionMismatch;
    if (q.rank_relative_tolerance) |t| if (!std.math.isFinite(t) or t < 0 or t >= 1) return error.InvalidOptions;
    if (!std.math.isFinite(q.rank_absolute_tolerance) or q.rank_absolute_tolerance < 0) return error.InvalidOptions;
    if (q.eigen_options.relative_tolerance) |t| if (!std.math.isFinite(t) or t < 0) return error.InvalidOptions;
    try separate(out, q.eigenvalues[0..n]);
    for (0..n) |column| for (0..n) |row| {
        if (!Cx.finite(T, get(T, q.hamiltonian, q.triangle, row, column)) or !Cx.finite(T, get(T, q.overlap, q.triangle, row, column))) return error.NonFiniteInput;
    };
}
fn preflight(comptime T: type, q: Request(T), values: []T, real: []S.Real(T), wide: []H.WorkspaceScalar(T)) !Workspace {
    try options(T, q);
    const length = try workspaceLength(T, q.hamiltonian.row_count);
    if (values.len < length.elements or real.len < length.real_elements or wide.len < length.wide_elements) return error.WorkspaceTooSmall;
    const work = values[0..length.elements];
    const roots = real[0..length.real_elements];
    const extra = wide[0..length.wide_elements];
    try separate(work, roots);
    try separate(work, extra);
    try separate(roots, extra);
    for ([_][]const T{ q.hamiltonian.values[0..try q.hamiltonian.check()], q.overlap.values[0..try q.overlap.check()], q.coefficients.values[0..try q.coefficients.check()] }) |v| {
        try separate(v, work);
        try separate(v, roots);
        try separate(v, extra);
    }
    try separate(q.eigenvalues[0..q.hamiltonian.row_count], work);
    try separate(q.eigenvalues[0..q.hamiltonian.row_count], roots);
    try separate(q.eigenvalues[0..q.hamiltonian.row_count], extra);
    return length;
}
fn matrix(comptime T: type, values: []T, n: usize, order: usize) D.Matrix(T) {
    return .{ .values = values, .row_count = order, .column_count = order, .leading_dimension = n };
}
fn addProduct(comptime T: type, total: *Cx.WideComplex(T), a: Cx.WideComplex(T), b: Cx.WideComplex(T), conjugate_first: bool) void {
    const imaginary = if (conjugate_first) -a.im else a.im;
    total.re += a.re * b.re - imaginary * b.im;
    total.im += a.re * b.im + imaginary * b.re;
}
fn narrowed(comptime T: type, value: Cx.WideComplex(T)) !T {
    return Cx.narrow(T, value) orelse error.NonFiniteResult;
}
pub fn runAdmitted(comptime T: type, operation: *const E.context.Operation, q: Request(T), values: []T, real: []S.Real(T), wide: []H.WorkspaceScalar(T)) !Result {
    _ = operation.borrowedContext();
    const length = try preflight(T, q, values, real, wide);
    const n = q.hamiltonian.row_count;
    if (n == 0) return .{ .status = .converged };
    const square = n * n;
    const whitener = values[0..square];
    const reduced = values[square .. 2 * square];
    const rotations = values[2 * square .. 3 * square];
    const coefficients = values[3 * square .. length.elements];
    const gram_values = real[0..n];
    const ritz_values = real[n..length.real_elements];
    const extra = wide[0..length.wide_elements];
    const gram = try H.solveWithContract(T, q.overlap, q.triangle, gram_values, matrix(T, whitener, n, n), extra, q.eigen_options, .checked_finite);
    if (gram.status != .converged) return .{ .status = if (gram.status == .iteration_limit) .gram_iteration_limit else .range_limit, .gram_sweeps = gram.sweeps };
    const W = Cx.Wide(T);
    const relative = q.rank_relative_tolerance orelse @min(@as(S.Real(T), 0.5), std.math.floatEps(S.Real(T)) * @as(S.Real(T), @floatFromInt(n)) * 16);
    const threshold: W = @max(@as(W, q.rank_absolute_tolerance), @as(W, relative) * @max(@as(W, 0), gram_values[n - 1]));
    if (@as(W, gram_values[0]) < -threshold) return .{ .status = .indefinite_metric, .gram_sweeps = gram.sweeps };
    var first: usize = 0;
    while (first < n and @as(W, gram_values[first]) <= threshold) : (first += 1) {}
    const rank = n - first;
    if (rank == 0) return .{ .status = .converged, .rank = 0, .gram_sweeps = gram.sweeps };
    for (0..rank) |column| {
        const scale: W = 1 / @sqrt(@as(W, gram_values[first + column]));
        for (0..n) |row| {
            const v = Cx.widen(T, whitener[row + (first + column) * n]);
            whitener[row + column * n] = try narrowed(T, .{ .re = v.re * scale, .im = v.im * scale });
        }
    }
    // H*W remains wide, so unrepresentable intermediate products can cancel
    // before the final representable reduced matrix is narrowed.
    const applied = extra[0 .. n * rank];
    for (0..rank) |column| for (0..n) |row| {
        var total: Cx.WideComplex(T) = .{ .re = 0, .im = 0 };
        for (0..n) |k| addProduct(T, &total, Cx.widen(T, get(T, q.hamiltonian, q.triangle, row, k)), Cx.widen(T, whitener[k + column * n]), false);
        applied[row + column * n] = total;
    };
    for (0..rank) |column| for (0..rank) |row| {
        var total: Cx.WideComplex(T) = .{ .re = 0, .im = 0 };
        for (0..n) |k| addProduct(T, &total, Cx.widen(T, whitener[k + row * n]), applied[k + column * n], true);
        reduced[row + column * n] = try narrowed(T, total);
    };
    const solved = try H.solveWithContract(T, matrix(T, reduced, n, rank).asConst(), .lower, ritz_values[0..rank], matrix(T, rotations, n, rank), extra, q.eigen_options, .checked_finite);
    if (solved.status != .converged) return .{ .status = if (solved.status == .iteration_limit) .reduced_iteration_limit else .range_limit, .rank = rank, .gram_sweeps = gram.sweeps, .reduced_sweeps = solved.sweeps };
    for (0..rank) |column| for (0..n) |row| {
        var total: Cx.WideComplex(T) = .{ .re = 0, .im = 0 };
        for (0..rank) |k| addProduct(T, &total, Cx.widen(T, whitener[row + k * n]), Cx.widen(T, rotations[k + column * n]), false);
        coefficients[row + column * n] = try narrowed(T, total);
    };
    @memcpy(q.eigenvalues[0..rank], ritz_values[0..rank]);
    for (0..rank) |column| for (0..n) |row| {
        q.coefficients.values[row + column * q.coefficients.leading_dimension] = coefficients[row + column * n];
    };
    return .{ .status = .converged, .rank = rank, .gram_sweeps = gram.sweeps, .reduced_sweeps = solved.sweeps };
}
pub fn Managed(comptime Execution: type) type {
    return struct {
        const C = Execution.context;
        pub fn runBorrowedAdmitted(comptime T: type, operation: *const C.Operation, q: Request(T), values: []T, real: []S.Real(T), wide: []H.WorkspaceScalar(T)) !Result {
            return runAdmitted(T, operation, q, values, real, wide);
        }
        pub fn runScratchAdmitted(comptime T: type, operation: *const C.Operation, q: Request(T)) !Result {
            _ = operation.borrowedContext();
            try options(T, q);
            const length = try workspaceLength(T, q.hamiltonian.row_count);
            var values = try C.ContextScratch(T).acquire(operation, length.elements);
            defer values.deinit();
            var real = try C.ContextScratch(S.Real(T)).acquire(operation, length.real_elements);
            defer real.deinit();
            var wide = try C.ContextScratch(H.WorkspaceScalar(T)).acquire(operation, length.wide_elements);
            defer wide.deinit();
            return runAdmitted(T, operation, q, values.lease.values, real.lease.values, wide.lease.values);
        }
        pub fn runBorrowed(comptime T: type, owner: *C.Context, q: Request(T), values: []T, real: []S.Real(T), wide: []H.WorkspaceScalar(T)) !Result {
            var operation = try owner.begin();
            defer operation.end();
            return runAdmitted(T, &operation, q, values, real, wide);
        }
        pub fn runScratch(comptime T: type, owner: *C.Context, q: Request(T)) !Result {
            var operation = try owner.begin();
            defer operation.end();
            return runScratchAdmitted(T, &operation, q);
        }
    };
}
