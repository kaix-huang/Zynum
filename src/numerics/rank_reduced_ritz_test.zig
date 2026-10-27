// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const F = @import("rank_reduced_ritz.zig");
const H = @import("../lapack/core/hermitian_eigen.zig");
const S = @import("../blas/core/shared/scalar.zig");
const D = @import("dense.zig");
const E = @import("zynum-execution");
fn z(comptime T: type, re: S.Real(T), im: S.Real(T)) T {
    return .{ .re = re, .im = im };
}
fn close(owner: *E.context.Context) !void {
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    _ = owner.requestClose();
    try std.testing.expectEqual(E.context.CloseResult.closed, fixtureProgressClose(owner));
}
fn matrix(comptime T: type, values: []T, n: usize, ld: usize) D.Matrix(T) {
    return .{ .values = values, .row_count = n, .column_count = n, .leading_dimension = ld };
}
fn projected(comptime T: type, h: []T, s: []T, ld: usize, triangle: H.Triangle) void {
    // B=[e1,e2,e1+i*e2], S=B*B, H=B*[[2,i],[-i,2]]*B.
    const hs = [_]T{ z(T, 2, 0), z(T, 0, -1), z(T, 1, 0), z(T, 0, 1), z(T, 2, 0), z(T, 0, -1), z(T, 1, 0), z(T, 0, 1), z(T, 2, 0) };
    const ss = [_]T{ z(T, 1, 0), z(T, 0, 0), z(T, 1, 0), z(T, 0, 0), z(T, 1, 0), z(T, 0, -1), z(T, 1, 0), z(T, 0, 1), z(T, 2, 0) };
    for (0..3) |col| for (0..3) |row| {
        if ((triangle == .lower and row >= col) or (triangle == .upper and row <= col)) {
            h[row + col * ld] = hs[row + col * 3];
            s[row + col * ld] = ss[row + col * 3];
            if (row == col) {
                h[row + col * ld].im = std.math.nan(S.Real(T));
                s[row + col * ld].im = std.math.nan(S.Real(T));
            }
        }
    };
}
fn checkPhysical(comptime T: type, values: []const S.Real(T), coefficients: []const T, ld: usize) !void {
    const tol: S.Real(T) = if (S.Real(T) == f32) 7e-5 else 4e-12;
    for (0..2) |col| {
        try std.testing.expectApproxEqAbs(@as(S.Real(T), @floatFromInt(1 + 2 * col)), values[col], tol);
        const c = coefficients[col * ld ..][0..3];
        const x = S.add(T, c[0], c[2]);
        const y = S.add(T, c[1], S.mul(T, z(T, 0, 1), c[2]));
        const ax = S.add(T, S.mul(T, z(T, 2, 0), x), S.mul(T, z(T, 0, 1), y));
        const ay = S.add(T, S.mul(T, z(T, 0, -1), x), S.mul(T, z(T, 2, 0), y));
        const rx = S.sub(T, ax, S.mul(T, z(T, values[col], 0), x));
        const ry = S.sub(T, ay, S.mul(T, z(T, values[col], 0), y));
        try std.testing.expectApproxEqAbs(@as(S.Real(T), 0), @sqrt(rx.re * rx.re + rx.im * rx.im + ry.re * ry.re + ry.im * ry.im), tol);
        try std.testing.expectApproxEqAbs(@as(S.Real(T), 1), x.re * x.re + x.im * x.im + y.re * y.re + y.im * y.im, tol);
        const off_diagonal = S.mul(T, x, z(T, y.re, -y.im));
        try std.testing.expectApproxEqAbs(@as(S.Real(T), 0), off_diagonal.re, tol);
        try std.testing.expectApproxEqAbs(@as(S.Real(T), if (col == 0) -0.5 else 0.5), off_diagonal.im, tol);
    }
}
test "Rank reduced Ritz two precisions redundant complex basis recovers independent physical projectors" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| for ([_]H.Triangle{ .lower, .upper }) |triangle| {
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer close(owner) catch unreachable;
        var h: [17]T = @splat(z(T, std.math.nan(S.Real(T)), std.math.nan(S.Real(T))));
        var s = h;
        projected(T, &h, &s, 5, triangle);
        const h_before = h;
        const s_before = s;
        var output: [17]T = @splat(z(T, 77, -33));
        var values: [5]S.Real(T) = @splat(91);
        const result = try F.Managed(E).runScratch(T, owner, .{ .hamiltonian = matrix(T, &h, 3, 5).asConst(), .overlap = matrix(T, &s, 3, 5).asConst(), .eigenvalues = &values, .coefficients = matrix(T, &output, 3, 4), .triangle = triangle });
        try std.testing.expectEqual(F.Status.converged, result.status);
        try std.testing.expectEqual(@as(usize, 2), result.rank);
        try checkPhysical(T, &values, &output, 4);
        for (output, 0..) |v, i| if (i >= 8 or i == 3 or i == 7) {
            try std.testing.expectEqualDeep(z(T, 77, -33), v);
        };
        for (values[2..]) |v| try std.testing.expectEqual(@as(S.Real(T), 91), v);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&h_before), std.mem.asBytes(&h));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&s_before), std.mem.asBytes(&s));
    };
}
test "Rank reduced Ritz input output overlap stages the entire projected problem" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var h: [11]T = @splat(z(T, 77, -33));
    var s = h;
    projected(T, &h, &s, 3, .lower);
    const third_column = h[6..11].*;
    var values: [3]f64 = @splat(91);
    const result = try F.Managed(E).runScratch(T, owner, .{ .hamiltonian = matrix(T, &h, 3, 3).asConst(), .overlap = matrix(T, &s, 3, 3).asConst(), .coefficients = matrix(T, &h, 3, 3), .eigenvalues = &values });
    try std.testing.expectEqual(F.Status.converged, result.status);
    try std.testing.expectEqual(@as(usize, 2), result.rank);
    try checkPhysical(T, &values, &h, 3);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&third_column), std.mem.sliceAsBytes(h[6..11]));
}
test "Rank reduced Ritz zero rank indefinite metric and controlled eigensolver stops preserve all output" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var h = [_]T{ z(T, 2, 0), z(T, 1, 0), z(T, 1, 0), z(T, 3, 0) };
    var s: [4]T = @splat(z(T, 0, 0));
    var output: [5]T = @splat(z(T, 77, -33));
    var values = [_]f64{ 91, 92, 93 };
    const before_output = output;
    const before_values = values;
    var q = F.Request(T){ .hamiltonian = matrix(T, &h, 2, 2).asConst(), .overlap = matrix(T, &s, 2, 2).asConst(), .coefficients = matrix(T, &output, 2, 2), .eigenvalues = &values };
    const empty = try F.Managed(E).runScratch(T, owner, q);
    try std.testing.expectEqual(F.Status.converged, empty.status);
    try std.testing.expectEqual(@as(usize, 0), empty.rank);
    s = .{ z(T, -1, 0), z(T, 0, 0), z(T, 0, 0), z(T, 1, 0) };
    try std.testing.expectEqual(F.Status.indefinite_metric, (try F.Managed(E).runScratch(T, owner, q)).status);
    s = .{ z(T, 2, 0), z(T, 0.5, 0), z(T, 0.5, 0), z(T, 2, 0) };
    q.eigen_options.max_sweeps = 0;
    try std.testing.expectEqual(F.Status.gram_iteration_limit, (try F.Managed(E).runScratch(T, owner, q)).status);
    s = .{ z(T, 1, 0), z(T, 0, 0), z(T, 0, 0), z(T, 1, 0) };
    try std.testing.expectEqual(F.Status.reduced_iteration_limit, (try F.Managed(E).runScratch(T, owner, q)).status);
    try std.testing.expectEqualDeep(before_output, output);
    try std.testing.expectEqualDeep(before_values, values);
    q.eigen_options.max_sweeps = 96;
    try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runScratch(T, owner, q)).status);
}
test "Rank reduced Ritz tiny metric scales and final range failure remain transactional" {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer close(owner) catch unreachable;
        const tiny: R = if (R == f32) 1e-30 else 1e-300;
        var h = [_]T{z(T, tiny * 3, 0)};
        const s = [_]T{z(T, tiny, 0)};
        var output = [_]T{ z(T, 77, -33), z(T, 78, -34) };
        var values = [_]R{ 91, 92 };
        const q = F.Request(T){ .hamiltonian = matrix(T, &h, 1, 1).asConst(), .overlap = .{ .values = &s, .row_count = 1, .column_count = 1, .leading_dimension = 1 }, .coefficients = matrix(T, &output, 1, 1), .eigenvalues = &values };
        try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runScratch(T, owner, q)).status);
        const tol: R = if (R == f32) 4e-6 else 1e-13;
        try std.testing.expectApproxEqAbs(@as(R, 3), values[0], tol);
        try std.testing.expectApproxEqAbs(@as(R, 1), output[0].re * @sqrt(tiny), tol);
        output[0] = z(T, 77, -33);
        values[0] = 91;
        h[0] = z(T, std.math.floatMax(R), 0);
        try std.testing.expectError(error.NonFiniteResult, F.Managed(E).runScratch(T, owner, q));
        try std.testing.expectEqualDeep(z(T, 77, -33), output[0]);
        try std.testing.expectEqual(@as(R, 91), values[0]);
        try std.testing.expectEqualDeep(z(T, 78, -34), output[1]);
        try std.testing.expectEqual(@as(R, 92), values[1]);
    }
}
test "Rank reduced Ritz borrowed capacity byte aliases options and overflow fail before publication" {
    const T = S.ComplexF64;
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    defer close(owner) catch unreachable;
    var h = [_]T{z(T, 2, 0)};
    var s = [_]T{z(T, 1, 0)};
    var output = [_]T{ z(T, 77, -33), z(T, 78, -34) };
    var values = [_]f64{ 91, 92 };
    var q = F.Request(T){ .hamiltonian = matrix(T, &h, 1, 1).asConst(), .overlap = matrix(T, &s, 1, 1).asConst(), .coefficients = matrix(T, &output, 1, 1), .eigenvalues = &values };
    var work: [5]T = @splat(z(T, 66, -22));
    var real: [3]f64 = @splat(55);
    var wide: [5]H.WorkspaceScalar(T) = @splat(.{ .re = 44, .im = -11 });
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(T, owner, q, work[0..3], &real, &wide));
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(T, owner, q, &work, real[0..1], &wide));
    try std.testing.expectError(error.WorkspaceTooSmall, F.Managed(E).runBorrowed(T, owner, q, &work, &real, wide[0..3]));
    q.rank_relative_tolerance = 1;
    try std.testing.expectError(error.InvalidOptions, F.Managed(E).runBorrowed(T, owner, q, &work, &real, &wide));
    q.rank_relative_tolerance = null;
    const raw: [*]f64 = @ptrCast(&output);
    q.eigenvalues = raw[0..2];
    try std.testing.expectError(error.AliasingNotAllowed, F.Managed(E).runBorrowed(T, owner, q, &work, &real, &wide));
    q.eigenvalues = &values;
    h[0].re = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteInput, F.Managed(E).runBorrowed(T, owner, q, &work, &real, &wide));
    h[0] = z(T, 2, 0);
    try std.testing.expectError(error.SizeOverflow, F.workspaceLength(T, std.math.maxInt(usize)));
    try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runBorrowed(T, owner, q, &work, &real, &wide)).status);
    try std.testing.expectEqualDeep(z(T, 66, -22), work[4]);
    try std.testing.expectEqual(@as(f64, 55), real[2]);
    try std.testing.expectEqualDeep(@as(H.WorkspaceScalar(T), .{ .re = 44, .im = -11 }), wide[4]);
}
test "Rank reduced Ritz allocation rollback and closing captured operation retain common ownership" {
    const T = S.ComplexF64;
    var h = [_]T{z(T, 2, 0)};
    var s = [_]T{z(T, 1, 0)};
    var output = [_]T{z(T, 77, -33)};
    var values = [_]f64{91};
    const q = F.Request(T){ .hamiltonian = matrix(T, &h, 1, 1).asConst(), .overlap = matrix(T, &s, 1, 1).asConst(), .coefficients = matrix(T, &output, 1, 1), .eigenvalues = &values };
    for (0..3) |budget| {
        var allocations = E.test_support.Allocations{ .budget = budget };
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(allocations.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer close(owner) catch unreachable;
        try std.testing.expectError(error.OutOfMemory, F.Managed(E).runScratch(T, owner, q));
        try std.testing.expectEqualDeep(z(T, 77, -33), output[0]);
        try std.testing.expectEqual(@as(f64, 91), values[0]);
        const stats = allocations.stats();
        try std.testing.expectEqual(budget, stats.accepted);
        try std.testing.expectEqual(stats.accepted, stats.freed);
    }
    var reference_owner: FixtureReferenceContext = undefined;
    reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const owner = reference_owner.borrowedContext();
    var operation = try owner.begin();
    _ = owner.requestClose();
    try std.testing.expectEqual(E.context.CloseResult.busy, fixtureProgressClose(owner));
    try std.testing.expectEqual(F.Status.converged, (try F.Managed(E).runScratchAdmitted(T, &operation, q)).status);
    try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    operation.end();
    try std.testing.expectEqual(E.context.CloseResult.closed, fixtureProgressClose(owner));
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
