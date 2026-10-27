// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const N = Core.internal.numerics;
const S = N.Scalar;
const E = N.Execution;
const C = E.context;
const B = @import("interop/operator_binding.zig").Binding(Core, @import("checked/implementation.zig"));
const W = B.Wire;
fn value(comptime T: type, re: S.Real(T)) T {
    return if (comptime S.isComplex(T)) .{ .re = re, .im = 0 } else re;
}
fn Foreign(comptime T: type) type {
    return struct {
        owner: *C.Context,
        calls: usize = 0,
        fail_at: usize = 0,
        close_at: usize = 0,
        fn apply(raw: ?*anyopaque, _: *const B.Token, input: W.const_buffer, output: W.buffer, _: W.buffer) callconv(.c) i32 {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            const snapshot = self.owner.snapshot();
            if (snapshot.active_operations != 1 or snapshot.resource_pins != 2) return 901;
            if (self.close_at == self.calls) _ = self.owner.requestClose();
            const x: [*]const T = @ptrCast(@alignCast(input.data.?));
            const y: [*]T = @ptrCast(@alignCast(output.data.?));
            y[0] = S.add(T, S.mul(T, value(T, 2), x[0]), x[1]);
            y[1] = S.add(T, S.add(T, x[0], S.mul(T, value(T, 3), x[1])), S.mul(T, value(T, 0.5), x[2]));
            y[2] = S.add(T, S.mul(T, value(T, 0.5), x[1]), S.mul(T, value(T, 4), x[2]));
            return if (self.calls == self.fail_at) -731 else 0;
        }
        fn action(self: *@This()) B.Action {
            return .{ .context = self, .rows = 3, .columns = 3, .workspace_elements = 0, .apply = apply };
        }
    };
}
fn initial(comptime T: type) [3]T {
    return .{ value(T, 1), value(T, 0), value(T, 0) };
}
fn sentinel(comptime T: type) B.ChainResult(T) {
    return .{ .status = 71, .reserved = 72, .order = 73, .action_count = 74, .residual_norm = 75 };
}
pub fn success() !void {
    try checkRationalFilterBoundary();
    try checkBiCGStabBoundary();
    try checkSubspaceProjectionBoundary();
    try checkSeriesSetBoundary();
    try checkNormalizedFilterBoundary();
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer {
            _ = owner.requestClose();
            std.debug.assert(fixtureProgressClose(owner) == .closed);
        }
        var op = try owner.begin();
        defer op.end();
        var foreign = Foreign(T){ .owner = owner };
        const x = initial(T);
        var alpha: [6]R = @splat(91);
        var beta: [5]R = @splat(92);
        var result = sentinel(T);
        const r = B.lanczosChainAdmitted(T, &op, foreign.action(), null, .{ .data = &x, .length = 3 }, 3, 5, 0, 0, .{ .data = &alpha, .length = 6 }, .{ .data = &beta, .length = 5 }, &result);
        try std.testing.expectEqual(W.OK, r.status);
        try std.testing.expectEqual(@as(usize, 32), @sizeOf(B.ChainResult(T)));
        try std.testing.expectEqual(@as(usize, 24), @offsetOf(B.ChainResult(T), "residual_norm"));
        try std.testing.expectEqual(@as(i32, 1), result.status);
        try std.testing.expectEqual(@as(i32, 0), result.reserved);
        try std.testing.expectEqual(@as(i64, 3), result.order);
        try std.testing.expectEqual(@as(i64, 3), result.action_count);
        try std.testing.expectEqual(@as(R, 0), result.residual_norm);
        try std.testing.expectEqualSlices(R, &.{ 2, 3, 4 }, alpha[0..3]);
        try std.testing.expectEqualSlices(R, &.{ 1, 0.5 }, beta[0..2]);
        for (alpha[3..]) |v| try std.testing.expectEqual(@as(R, 91), v);
        for (beta[2..]) |v| try std.testing.expectEqual(@as(R, 92), v);
        try std.testing.expectEqualDeep(initial(T), x);
        try std.testing.expectEqual(@as(usize, 3), foreign.calls);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
}
pub fn boundaries() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer {
            _ = owner.requestClose();
            std.debug.assert(fixtureProgressClose(owner) == .closed);
        }
        var op = try owner.begin();
        defer op.end();
        var foreign = Foreign(T){ .owner = owner };
        var x = initial(T);
        var alpha: [8]R = @splat(91);
        var beta: [6]R = @splat(92);
        var result = sentinel(T);
        const saved = result;
        var bytes: [96]u8 align(16) = @splat(0x5a);
        for (0..17) |case| {
            var action = foreign.action();
            var xb = W.const_buffer{ .data = &x, .length = 3 };
            var ab = W.buffer{ .data = &alpha, .length = 8 };
            var bb = W.buffer{ .data = &beta, .length = 6 };
            var target: ?*anyopaque = &result;
            var reduction: ?*const anyopaque = null;
            var n: i64 = 3;
            var depth: i64 = 5;
            var tol: R = 0;
            var herm: R = 0;
            const end_ptr: *anyopaque = @ptrFromInt(std.math.maxInt(usize) - 7);
            switch (case) {
                0 => target = null,
                1 => target = &bytes[1],
                2 => target = end_ptr,
                3 => reduction = end_ptr,
                4 => ab.data = &x,
                5 => bb.data = &alpha,
                6 => target = &alpha,
                7 => ab.length = 4,
                8 => bb.length = 3,
                9 => action.columns = 2,
                10 => n = 0,
                11 => depth = 0,
                12 => tol = -1,
                13 => herm = std.math.nan(R),
                14 => xb.length = 2,
                15 => ab.data = end_ptr,
                16 => action.apply = null,
                else => unreachable,
            }
            const r = B.lanczosChainAdmitted(T, &op, action, reduction, xb, n, depth, tol, herm, ab, bb, target);
            try std.testing.expect(r.status != W.OK);
            try std.testing.expectEqual(@as(usize, 0), foreign.calls);
            try std.testing.expectEqualDeep(saved, result);
            try std.testing.expectEqualDeep(initial(T), x);
            for (alpha) |v| try std.testing.expectEqual(@as(R, 91), v);
            for (beta) |v| try std.testing.expectEqual(@as(R, 92), v);
            for (bytes) |v| try std.testing.expectEqual(@as(u8, 0x5a), v);
            try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        }
    }
}
pub fn failureAndClose() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        var op = try owner.begin();
        var foreign = Foreign(T){ .owner = owner, .fail_at = 2 };
        const x = initial(T);
        var alpha: [5]R = @splat(91);
        var beta: [4]R = @splat(92);
        var result = sentinel(T);
        const saved = result;
        var r = B.lanczosChainAdmitted(T, &op, foreign.action(), null, .{ .data = &x, .length = 3 }, 3, 5, 0, 0, .{ .data = &alpha, .length = 5 }, .{ .data = &beta, .length = 4 }, &result);
        try std.testing.expectEqual(B.ACTION_FAILED, r.status);
        try std.testing.expectEqual(@as(i64, -731), r.info);
        try std.testing.expectEqualDeep(saved, result);
        for (alpha) |v| try std.testing.expectEqual(@as(R, 91), v);
        for (beta) |v| try std.testing.expectEqual(@as(R, 92), v);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        foreign.calls = 0;
        foreign.fail_at = 0;
        foreign.close_at = 1;
        r = B.lanczosChainAdmitted(T, &op, foreign.action(), null, .{ .data = &x, .length = 3 }, 3, 5, 0, 0, .{ .data = &alpha, .length = 5 }, .{ .data = &beta, .length = 4 }, &result);
        try std.testing.expectEqual(W.OK, r.status);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        try std.testing.expectEqual(C.CloseResult.busy, fixtureProgressClose(owner));
        op.end();
        try std.testing.expectEqual(C.CloseResult.closed, fixtureProgressClose(owner));
    }
}

fn FilterForeign(comptime T: type) type {
    return struct {
        owner: *C.Context,
        calls: usize = 0,
        fail_at: usize = 0,
        remote_commit: bool = false,
        final_filter_arithmetic: bool = false,
        close_at: usize = 0,
        fn action(self: *@This()) B.Action {
            return .{ .context = self, .rows = 2, .columns = 2, .workspace_elements = 0, .apply = apply };
        }
        fn apply(raw: ?*anyopaque, _: *const B.Token, input: W.const_buffer, output: W.buffer, _: W.buffer) callconv(.c) i32 {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            const snapshot = self.owner.snapshot();
            if (snapshot.active_operations != 1 or snapshot.resource_pins == 0) return 901;
            if (self.close_at == self.calls) _ = self.owner.requestClose();
            const x: [*]const T = @ptrCast(@alignCast(input.data.?));
            const y: [*]T = @ptrCast(@alignCast(output.data.?));
            const off: T = if (comptime S.isComplex(T)) .{ .re = 0, .im = 1 } else 1;
            y[0] = S.add(T, S.mul(T, value(T, 2), x[0]), S.mul(T, off, x[1]));
            y[1] = S.add(T, S.mul(T, S.conj(T, off), x[0]), S.mul(T, value(T, 2), x[1]));
            return if (self.calls == self.fail_at) -731 else 0;
        }
        fn agree(raw: ?*anyopaque, _: ?*const B.Token, point: B.Point, local: W.result) callconv(.c) W.result {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (local.status != W.OK) return local;
            if (point.stage == .local_arithmetic and point.index == 2) self.final_filter_arithmetic = true;
            if (point.stage == .commit and self.remote_commit and self.final_filter_arithmetic) return .{ .status = W.NONFINITE_INPUT, .reserved = 0, .info = 77 };
            return local;
        }
    };
}
pub fn checkNormalizedFilterBoundary() !void {
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(B.ConstPanel));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(B.Panel));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(B.Panel, "layout"));
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer {
            _ = owner.requestClose();
            std.debug.assert(fixtureProgressClose(owner) == .closed);
        }
        var operation = try owner.begin();
        defer operation.end();
        var foreign = FilterForeign(T){ .owner = owner };
        var data = [_]T{ value(T, 1), value(T, 0), value(T, 91), value(T, 0), value(T, 1), value(T, 92), value(T, 93) };
        const original = data;
        const x: B.ConstPanel = .{ .dense = .{ .storage = .{ .data = &data, .length = data.len }, .rows = 2, .cols = 2, .ld = 3 }, .layout = 101 };
        const y: B.Panel = .{ .dense = .{ .storage = .{ .data = &data, .length = data.len }, .rows = 2, .cols = 2, .ld = 3 }, .layout = 101 };
        const agreement: B.Agreement = .{ .context = &foreign, .agree = FilterForeign(T).agree };
        const result = B.normalizedFilterPanelAgreedAdmitted(T, &operation, foreign.action(), agreement, x, 2, 1, 2, 4, y);
        try std.testing.expectEqual(W.OK, result.status);
        try std.testing.expectEqual(@as(usize, 4), foreign.calls);
        // Eigenvalues 1 and 3 have exact filter factors 1 and -1/7.
        const off: T = if (comptime S.isComplex(T)) .{ .re = 0, .im = -@as(R, 4) / 7 } else -@as(R, 4) / 7;
        const expected = [_]T{ value(T, @as(R, 3) / 7), off, S.conj(T, off), value(T, @as(R, 3) / 7) };
        for ([_]usize{ 0, 1, 3, 4 }, expected) |i, v| {
            const tol: R = if (R == f32) 2e-6 else 2e-14;
            if (comptime S.isComplex(T)) {
                try std.testing.expectApproxEqAbs(v.re, data[i].re, tol);
                try std.testing.expectApproxEqAbs(v.im, data[i].im, tol);
            } else try std.testing.expectApproxEqAbs(v, data[i], tol);
        }
        for ([_]usize{ 2, 5, 6 }) |i| try std.testing.expectEqualDeep(original[i], data[i]);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        data = original;
        foreign.calls = 0;
        foreign.fail_at = 2;
        var failed = B.normalizedFilterPanelAgreedAdmitted(T, &operation, foreign.action(), agreement, x, 2, 1, 2, 4, y);
        try std.testing.expectEqual(B.ACTION_FAILED, failed.status);
        try std.testing.expectEqual(@as(i64, -731), failed.info);
        try std.testing.expectEqualDeep(original, data);
        try std.testing.expectEqual(@as(usize, 2), foreign.calls);
        foreign.calls = 0;
        foreign.fail_at = 0;
        foreign.remote_commit = true;
        foreign.final_filter_arithmetic = false;
        failed = B.normalizedFilterPanelAgreedAdmitted(T, &operation, foreign.action(), agreement, x, 2, 1, 2, 4, y);
        try std.testing.expectEqual(W.NONFINITE_INPUT, failed.status);
        try std.testing.expectEqual(@as(i64, 77), failed.info);
        try std.testing.expectEqualDeep(original, data);
        foreign.remote_commit = false;
        foreign.calls = 0;
        var bad_x = x;
        bad_x.reserved = 1;
        failed = B.normalizedFilterPanelAgreedAdmitted(T, &operation, foreign.action(), agreement, bad_x, 2, 1, 2, 4, y);
        try std.testing.expectEqual(W.INVALID_FLAG, failed.status);
        failed = B.normalizedFilterPanelAdmitted(T, &operation, foreign.action(), x, -1, 1, 2, 4, y);
        try std.testing.expectEqual(B.INVALID_OPTIONS, failed.status);
        try std.testing.expectEqual(@as(usize, 0), foreign.calls);
        try std.testing.expectEqualDeep(original, data);
        // Input row-major, output column-major, degree-zero bounds unused.
        var out: [7]T = @splat(value(T, 71));
        var output = y;
        output.dense.storage.data = &out;
        output.layout = 102;
        failed = B.normalizedFilterPanelAdmitted(T, &operation, foreign.action(), x, 0, std.math.nan(R), std.math.nan(R), std.math.nan(R), output);
        try std.testing.expectEqual(W.OK, failed.status);
        try std.testing.expectEqualDeep(value(T, 1), out[0]);
        try std.testing.expectEqualDeep(value(T, 1), out[4]);
        for ([_]usize{ 2, 5, 6 }) |i| try std.testing.expectEqualDeep(value(T, 71), out[i]);
        // The canonical phase reply is shared by the pre-existing algorithms.
        // These local preflight failures must keep their public status in Fast.
        const invalid_vector = [_]T{value(T, 1)};
        const valid_vector = [_]T{ value(T, 1), value(T, 0) };
        var untouched = [_]T{ value(T, 81), value(T, 82) };
        const source: W.const_buffer = .{ .data = &valid_vector, .length = 2 };
        const target: W.buffer = .{ .data = &untouched, .length = 2 };
        failed = B.chebyshevStepAgreedAdmitted(T, &operation, foreign.action(), agreement, .{ .data = &invalid_vector, .length = 1 }, source, target);
        try std.testing.expectEqual(W.DIMENSION_MISMATCH, failed.status);
        failed = B.chebyshevAgreedAdmitted(T, &operation, foreign.action(), agreement, source, .{ .data = null, .length = 0 }, target);
        try std.testing.expectEqual(B.INVALID_OPTIONS, failed.status);
        var ritz: B.RitzResult(T) = .{ .status = 91, .has_previous_ritz = 0, .order = 92, .action_count = 93, .minimum = 94, .maximum = 95, .min_change = 96, .max_change = 97, .residual_norm = 98 };
        const saved_ritz = ritz;
        failed = B.lanczosAgreedAdmitted(T, &operation, foreign.action(), agreement, null, source, 2, 0, 0, 0, &ritz);
        try std.testing.expectEqual(B.INVALID_OPTIONS, failed.status);
        try std.testing.expectEqualDeep(saved_ritz, ritz);
        var alpha = [_]R{ 91, 92 };
        var beta = [_]R{93};
        var chain = sentinel(T);
        const saved_chain = chain;
        failed = B.lanczosChainAgreedAdmitted(T, &operation, foreign.action(), agreement, null, source, 2, 0, 0, 0, .{ .data = &alpha, .length = 2 }, .{ .data = &beta, .length = 1 }, &chain);
        try std.testing.expectEqual(B.INVALID_OPTIONS, failed.status);
        try std.testing.expectEqualDeep(saved_chain, chain);
        try std.testing.expectEqualDeep([_]T{ value(T, 81), value(T, 82) }, untouched);
        try std.testing.expectEqualSlices(R, &.{ 91, 92 }, &alpha);
        try std.testing.expectEqualSlices(R, &.{93}, &beta);
        try std.testing.expectEqual(@as(usize, 0), foreign.calls);
        const vector = [_]T{ value(T, 1), value(T, 0) };
        var filtered = [_]T{ value(T, 81), value(T, 82) };
        failed = B.normalizedFilterAgreedAdmitted(T, &operation, foreign.action(), agreement, .{ .data = &vector, .length = 2 }, 1, 1, 2, 4, .{ .data = &filtered, .length = 2 });
        try std.testing.expectEqual(W.OK, failed.status);
        foreign.close_at = foreign.calls + 1;
        failed = B.normalizedFilterAdmitted(T, &operation, foreign.action(), .{ .data = &vector, .length = 2 }, 2, 1, 2, 4, .{ .data = &filtered, .length = 2 });
        try std.testing.expectEqual(W.OK, failed.status);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
}

fn seriesSetValue(comptime T: type, real: S.Real(T), imaginary: S.Real(T)) T {
    return if (comptime S.isComplex(T)) .{ .re = real, .im = imaginary } else real;
}
fn seriesSetIndex(shape: N.Dense.PanelShape, row: usize, column: usize) usize {
    return if (shape.layout == .column_major) row + column * shape.leading_dimension else column + row * shape.leading_dimension;
}
fn seriesSetEigenPolynomial(comptime T: type, coefficients: N.Dense.ConstPanel(T), column: usize, count: usize, lambda: f64) T {
    var real: f64 = 0;
    var imaginary: f64 = 0;
    for (0..count) |order| {
        const factor = if (lambda == 1) 1 else std.math.cosh(@as(f64, @floatFromInt(order)) * std.math.acosh(lambda));
        const c = coefficients.column(column).at(order);
        real += factor * (if (comptime S.isComplex(T)) c.re else c);
        if (comptime S.isComplex(T)) imaginary += factor * c.im;
    }
    return seriesSetValue(T, @floatCast(real), @floatCast(imaginary));
}
fn SeriesSetAgreement(comptime T: type) type {
    return struct {
        fn agree(raw: ?*anyopaque, _: ?*const B.Token, point: B.Point, local: W.result) callconv(.c) W.result {
            const foreign: *FilterForeign(T) = @ptrCast(@alignCast(raw.?));
            if (local.status != W.OK) return local;
            if (point.stage == .local_arithmetic and point.index == 4) foreign.final_filter_arithmetic = true;
            if (point.stage == .commit and foreign.remote_commit and foreign.final_filter_arithmetic) return .{ .status = W.NONFINITE_INPUT, .reserved = 0, .info = 77 };
            return local;
        }
    };
}
pub fn checkSeriesSetBoundary() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| inline for (.{ N.Dense.Layout.row_major, N.Dense.Layout.column_major }) |layout| {
        const R = S.Real(T);
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer {
            _ = owner.requestClose();
            std.debug.assert(fixtureProgressClose(owner) == .closed);
        }
        var operation = try owner.begin();
        defer operation.end();
        var foreign = FilterForeign(T){ .owner = owner };
        var coefficients: [25]T = @splat(seriesSetValue(T, std.math.nan(R), 7));
        const shape: N.Dense.PanelShape = .{ .row_count = 5, .column_count = 3, .leading_dimension = if (layout == .row_major) 5 else 7, .layout = layout };
        var counts = [_]i64{ 1, 3, 5 };
        for (counts, 0..) |count, column| for (0..@intCast(count)) |order| {
            coefficients[seriesSetIndex(shape, order, column)] = seriesSetValue(T, 1 / @as(R, @floatFromInt(order + column + 1)), if (order % 2 == 0) 0.02 else -0.01);
        };
        const table: N.Dense.ConstPanel(T) = .{ .values = &coefficients, .shape = shape };
        const wire: B.ConstPanel = .{ .dense = .{ .storage = .{ .data = &coefficients, .length = coefficients.len }, .rows = 5, .cols = 3, .ld = @intCast(shape.leading_dimension) }, .layout = if (layout == .row_major) 101 else 102 };
        const starting_vector = [_]T{ value(T, 1), value(T, 0) };
        var output: [9]T = @splat(value(T, 91));
        const output_shape: N.Dense.PanelShape = .{ .row_count = 2, .column_count = 3, .leading_dimension = if (layout == .row_major) 3 else 5, .layout = if (layout == .row_major) .column_major else .row_major };
        var destination: B.Panel = .{ .dense = .{ .storage = .{ .data = &output, .length = output.len }, .rows = 2, .cols = 3, .ld = @intCast(output_shape.leading_dimension) }, .layout = if (output_shape.layout == .row_major) 101 else 102 };
        const agreement: B.Agreement = .{ .context = &foreign, .agree = SeriesSetAgreement(T).agree };
        const x: W.const_buffer = .{ .data = &starting_vector, .length = starting_vector.len };
        const lengths: W.const_buffer = .{ .data = &counts, .length = counts.len };
        const result = if (layout == .row_major) B.chebyshevSeriesSetAdmitted(T, &operation, foreign.action(), x, wire, lengths, destination) else B.chebyshevSeriesSetAgreedAdmitted(T, &operation, foreign.action(), agreement, x, wire, lengths, destination);
        try std.testing.expectEqual(W.OK, result.status);
        try std.testing.expectEqual(@as(usize, 4), foreign.calls);
        const k = seriesSetValue(T, if (S.isComplex(T)) 0 else 1, if (S.isComplex(T)) 1 else 0);
        for (counts, 0..) |count, column| {
            const low = seriesSetEigenPolynomial(T, table, column, @intCast(count), 1);
            const high = seriesSetEigenPolynomial(T, table, column, @intCast(count), 3);
            const mean = S.mul(T, value(T, 0.5), S.add(T, low, high));
            const difference = S.mul(T, value(T, 0.5), S.sub(T, high, low));
            const expected = [_]T{ mean, S.mul(T, difference, S.conj(T, k)) };
            for (expected, 0..) |v, row| {
                const actual = output[seriesSetIndex(output_shape, row, column)];
                const tolerance: R = if (R == f32) 1e-4 else 2e-12;
                if (comptime S.isComplex(T)) {
                    try std.testing.expectApproxEqAbs(v.re, actual.re, tolerance * @max(1, @abs(v.re)));
                    try std.testing.expectApproxEqAbs(v.im, actual.im, tolerance * @max(1, @abs(v.im)));
                } else try std.testing.expectApproxEqAbs(v, actual, tolerance * @max(1, @abs(v)));
            }
        }
        for (0..output.len) |i| {
            var logical = false;
            for (0..3) |column| for (0..2) |row| {
                if (i == seriesSetIndex(output_shape, row, column)) logical = true;
            };
            if (!logical) try std.testing.expectEqualDeep(value(T, 91), output[i]);
        }
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        @memset(&output, value(T, 91));
        const before = output;
        foreign.calls = 0;
        foreign.fail_at = 2;
        var failed = B.chebyshevSeriesSetAgreedAdmitted(T, &operation, foreign.action(), agreement, x, wire, lengths, destination);
        try std.testing.expectEqual(B.ACTION_FAILED, failed.status);
        try std.testing.expectEqual(@as(i64, -731), failed.info);
        try std.testing.expectEqualDeep(before, output);
        foreign.calls = 0;
        foreign.fail_at = 0;
        foreign.remote_commit = true;
        foreign.final_filter_arithmetic = false;
        failed = B.chebyshevSeriesSetAgreedAdmitted(T, &operation, foreign.action(), agreement, x, wire, lengths, destination);
        try std.testing.expectEqual(W.NONFINITE_INPUT, failed.status);
        try std.testing.expectEqual(@as(i64, 77), failed.info);
        try std.testing.expectEqual(@as(usize, 4), foreign.calls);
        try std.testing.expectEqualDeep(before, output);
        foreign.remote_commit = false;
        foreign.calls = 0;
        counts[1] = 0;
        failed = B.chebyshevSeriesSetAgreedAdmitted(T, &operation, foreign.action(), agreement, x, wire, lengths, destination);
        try std.testing.expectEqual(B.INVALID_OPTIONS, failed.status);
        counts[1] = -1;
        failed = B.chebyshevSeriesSetAdmitted(T, &operation, foreign.action(), x, wire, lengths, destination);
        try std.testing.expectEqual(B.INVALID_OPTIONS, failed.status);
        counts[1] = 6;
        failed = B.chebyshevSeriesSetAdmitted(T, &operation, foreign.action(), x, wire, lengths, destination);
        try std.testing.expectEqual(B.INVALID_OPTIONS, failed.status);
        counts[1] = 3;
        var invalid = wire;
        invalid.reserved = 1;
        failed = B.chebyshevSeriesSetAgreedAdmitted(T, &operation, foreign.action(), agreement, x, invalid, lengths, destination);
        try std.testing.expectEqual(W.INVALID_FLAG, failed.status);
        try std.testing.expectEqual(@as(usize, 0), foreign.calls);
        try std.testing.expectEqualDeep(before, output);
        var overlap: [18]u64 = @splat(0);
        overlap[0] = 1;
        overlap[1] = 3;
        overlap[2] = 5;
        destination.dense.storage.data = &overlap;
        failed = B.chebyshevSeriesSetAdmitted(T, &operation, foreign.action(), x, wire, .{ .data = &overlap, .length = 3 }, destination);
        try std.testing.expectEqual(W.ALIASING_NOT_ALLOWED, failed.status);
        try std.testing.expectEqual(@as(usize, 0), foreign.calls);
    };
    for ([_]usize{ 0, 1 }) |budget| {
        var allocations = E.test_support.Allocations{ .budget = budget };
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(allocations.allocator(), E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer {
            _ = owner.requestClose();
            std.debug.assert(fixtureProgressClose(owner) == .closed);
        }
        var operation = try owner.begin();
        defer operation.end();
        var foreign = FilterForeign(f64){ .owner = owner };
        const input = [_]f64{ 1, 0 };
        const coefficients = [_]f64{ 1, 2, 3, 4 };
        const counts = [_]i64{ 2, 2 };
        var output = [_]f64{ 91, 92, 93, 94 };
        const failed = B.chebyshevSeriesSetAdmitted(f64, &operation, foreign.action(), .{ .data = &input, .length = 2 }, .{ .dense = .{ .storage = .{ .data = &coefficients, .length = 4 }, .rows = 2, .cols = 2, .ld = 2 }, .layout = 102 }, .{ .data = &counts, .length = 2 }, .{ .dense = .{ .storage = .{ .data = &output, .length = 4 }, .rows = 2, .cols = 2, .ld = 2 }, .layout = 102 });
        try std.testing.expectEqual(W.INTERNAL_ERROR, failed.status);
        try std.testing.expectEqual(@as(usize, 0), foreign.calls);
        try std.testing.expectEqualSlices(f64, &.{ 91, 92, 93, 94 }, &output);
        try std.testing.expectEqual(allocations.stats().accepted, allocations.stats().freed);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
        try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    }
}

fn ProjectionForeign(comptime T: type) type {
    return struct {
        owner: *C.Context,
        calls: usize = 0,
        fail: bool = false,
        nan: bool = false,
        remote: bool = false,
        weight_mode: u64 = 99,
        fill_empty: bool = false,
        fn reduce(raw: ?*anyopaque, token: *const B.Token, values: W.buffer) callconv(.c) i32 {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (@intFromPtr(token) == 0 or self.owner.snapshot().active_operations != 1 or self.owner.snapshot().resource_pins == 0 or values.length != 8) return 990;
            self.calls += 1;
            const data: [*]T = @ptrCast(@alignCast(values.data.?));
            if (self.fill_empty) for (0..8) |i| {
                data[i] = value(T, if (i % 4 == 0 or i % 4 == 3) (if (i < 4) 2 else 3) else 0);
            };
            if (self.nan) data[7] = value(T, std.math.nan(S.Real(T)));
            return if (self.fail) -741 else 0;
        }
        fn agree(raw: ?*anyopaque, _: ?*const B.Token, point: B.Point, local: W.result) callconv(.c) W.result {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (local.status != W.OK) return local;
            if (point.stage == .panel_backend) self.weight_mode = point.control;
            if (self.remote and point.stage == .commit) return .{ .status = B.PHASE_MISMATCH, .reserved = 0, .info = 74 };
            return local;
        }
    };
}
pub fn checkSubspaceProjectionBoundary() !void {
    @setEvalBranchQuota(20000);
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| inline for (.{ N.Dense.Layout.row_major, N.Dense.Layout.column_major }) |yl| inline for (.{ N.Dense.Layout.row_major, N.Dense.Layout.column_major }) |zl| inline for (.{ N.Dense.Layout.row_major, N.Dense.Layout.column_major }) |hl| inline for (.{ N.Dense.Layout.row_major, N.Dense.Layout.column_major }) |sl| {
        const R = S.Real(T);
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer {
            _ = owner.requestClose();
            std.debug.assert(fixtureProgressClose(owner) == .closed);
        }
        var op = try owner.begin();
        defer op.end();
        var foreign = ProjectionForeign(T){ .owner = owner };
        const reducer: B.BlockReduction = .{ .context = &foreign, .reduce = ProjectionForeign(T).reduce };
        const agreement: B.Agreement = .{ .context = &foreign, .agree = ProjectionForeign(T).agree };
        var y: [13]T = @splat(value(T, std.math.nan(R)));
        var z = y;
        const ys: N.Dense.PanelShape = .{ .row_count = 3, .column_count = 2, .leading_dimension = 4, .layout = yl };
        const zs: N.Dense.PanelShape = .{ .row_count = 3, .column_count = 2, .leading_dimension = 4, .layout = zl };
        const hs: N.Dense.PanelShape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = hl };
        const ss: N.Dense.PanelShape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = sl };
        const first = [_]R{ 1, 2, -1 };
        const second = [_]R{ 0, 1, 2 };
        const lambda = [_]R{ -2, 3, 4 };
        for (0..3) |row| for (0..2) |column| {
            const source = if (column == 0) value(T, first[row]) else seriesSetValue(T, if (S.isComplex(T)) 0 else second[row], if (S.isComplex(T)) second[row] else 0);
            y[seriesSetIndex(ys, row, column)] = source;
            z[seriesSetIndex(zs, row, column)] = S.mul(T, value(T, lambda[row]), source);
        };
        const y_before = std.mem.asBytes(&y).*;
        const z_before = std.mem.asBytes(&z).*;
        var h: [8]T = @splat(value(T, 91));
        var s: [8]T = @splat(value(T, 92));
        const yw: B.ConstPanel = .{ .dense = .{ .storage = .{ .data = &y, .length = y.len }, .rows = 3, .cols = 2, .ld = 4 }, .layout = if (yl == .row_major) 101 else 102 };
        var zw: B.ConstPanel = .{ .dense = .{ .storage = .{ .data = &z, .length = z.len }, .rows = 3, .cols = 2, .ld = 4 }, .layout = if (zl == .row_major) 101 else 102 };
        const hw: B.Panel = .{ .dense = .{ .storage = .{ .data = &h, .length = h.len }, .rows = 2, .cols = 2, .ld = 3 }, .layout = if (hl == .row_major) 101 else 102 };
        const sw: B.Panel = .{ .dense = .{ .storage = .{ .data = &s, .length = s.len }, .rows = 2, .cols = 2, .ld = 3 }, .layout = if (sl == .row_major) 101 else 102 };
        var weights = [_]R{ 1, 2, 0.5 };
        const ww: W.const_buffer = .{ .data = &weights, .length = 3 };
        var result = B.subspaceProjectAgreedAdmitted(T, &op, reducer, agreement, yw, zw, 1, ww, hw, sw);
        try std.testing.expectEqual(W.OK, result.status);
        try std.testing.expectEqual(@as(usize, 1), foreign.calls);
        try std.testing.expectEqual(@as(u64, 1), foreign.weight_mode);
        for (0..2) |column| for (0..2) |row| {
            const hr: R = if (row == column) (if (row == 0) 24 else 14) else 8;
            const sr: R = if (row == column) (if (row == 0) 9.5 else 4) else 3;
            const imaginary = S.isComplex(T) and row != column;
            const sign: R = if (row == 0) 1 else -1;
            try std.testing.expectEqualDeep(seriesSetValue(T, if (imaginary) 0 else hr, if (imaginary) sign * hr else 0), h[seriesSetIndex(hs, row, column)]);
            try std.testing.expectEqualDeep(seriesSetValue(T, if (imaginary) 0 else sr, if (imaginary) sign * sr else 0), s[seriesSetIndex(ss, row, column)]);
        };
        for (0..8) |i| {
            var hi = false;
            var si = false;
            for (0..2) |column| for (0..2) |row| {
                if (i == seriesSetIndex(hs, row, column)) hi = true;
                if (i == seriesSetIndex(ss, row, column)) si = true;
            };
            if (!hi) try std.testing.expectEqualDeep(value(T, 91), h[i]);
            if (!si) try std.testing.expectEqualDeep(value(T, 92), s[i]);
        }
        try std.testing.expectEqualDeep(y_before, std.mem.asBytes(&y).*);
        try std.testing.expectEqualDeep(z_before, std.mem.asBytes(&z).*);
        @memset(&h, value(T, 91));
        @memset(&s, value(T, 92));
        const hb = h;
        const sb = s;
        foreign.fail = true;
        result = B.subspaceProjectAgreedAdmitted(T, &op, reducer, agreement, yw, zw, 1, ww, hw, sw);
        try std.testing.expectEqual(B.REDUCTION_FAILED, result.status);
        try std.testing.expectEqual(@as(i64, -741), result.info);
        foreign.fail = false;
        foreign.nan = true;
        result = B.subspaceProjectAdmitted(T, &op, reducer, yw, zw, 1, ww, hw, sw);
        try std.testing.expectEqual(B.NONFINITE_RESULT, result.status);
        foreign.nan = false;
        foreign.remote = true;
        result = B.subspaceProjectAgreedAdmitted(T, &op, reducer, agreement, yw, zw, 1, ww, hw, sw);
        try std.testing.expectEqual(B.PHASE_MISMATCH, result.status);
        try std.testing.expectEqual(@as(i64, 74), result.info);
        foreign.remote = false;
        const calls = foreign.calls;
        weights[1] = -1;
        result = B.subspaceProjectAgreedAdmitted(T, &op, reducer, agreement, yw, zw, 1, ww, hw, sw);
        try std.testing.expectEqual(B.INVALID_OPTIONS, result.status);
        weights[1] = 2;
        result = B.subspaceProjectAdmitted(T, &op, reducer, yw, zw, 2, ww, hw, sw);
        try std.testing.expectEqual(B.INVALID_OPTIONS, result.status);
        zw.reserved = 1;
        result = B.subspaceProjectAgreedAdmitted(T, &op, reducer, agreement, yw, zw, 1, ww, hw, sw);
        try std.testing.expectEqual(W.INVALID_FLAG, result.status);
        zw.reserved = 0;
        var alias = sw;
        alias.dense.storage = hw.dense.storage;
        result = B.subspaceProjectAdmitted(T, &op, reducer, yw, zw, 1, ww, hw, alias);
        try std.testing.expectEqual(W.ALIASING_NOT_ALLOWED, result.status);
        try std.testing.expectEqual(calls, foreign.calls);
        try std.testing.expectEqualDeep(hb, h);
        try std.testing.expectEqualDeep(sb, s);
        result = B.subspaceProjectAdmitted(T, &op, .{ .context = null, .reduce = null }, yw, zw, 0, .{ .data = null, .length = -1 }, hw, sw);
        try std.testing.expectEqual(W.OK, result.status);
        @memset(&h, value(T, 91));
        @memset(&s, value(T, 92));
        foreign.fill_empty = true;
        var empty = yw;
        empty.dense.rows = 0;
        empty.dense.storage = .{ .data = null, .length = 0 };
        result = B.subspaceProjectAgreedAdmitted(T, &op, reducer, agreement, empty, empty, 1, .{ .data = null, .length = 0 }, hw, sw);
        try std.testing.expectEqual(W.OK, result.status);
        try std.testing.expectEqual(@as(u64, 1), foreign.weight_mode);
        for (0..2) |column| for (0..2) |row| {
            try std.testing.expectEqualDeep(value(T, if (row == column) 2 else 0), h[seriesSetIndex(hs, row, column)]);
            try std.testing.expectEqualDeep(value(T, if (row == column) 3 else 0), s[seriesSetIndex(ss, row, column)]);
        };
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    };
}

fn BiCGForeign(comptime T: type) type {
    return struct {
        owner: *C.Context,
        matrix: [9]T,
        calls: usize = 0,
        fail_at: usize = 0,
        reduction_fail: bool = false,
        remote_commit: bool = false,
        remote_rational_commit: bool = false,
        fn action(self: *@This()) B.Action {
            return .{ .context = self, .rows = 3, .columns = 3, .workspace_elements = 0, .apply = apply };
        }
        fn apply(raw: ?*anyopaque, token: *const B.Token, input: W.const_buffer, output: W.buffer, _: W.buffer) callconv(.c) i32 {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (@intFromPtr(token) == 0 or input.length != 3 or output.length != 3 or self.owner.snapshot().active_operations != 1) return 971;
            self.calls += 1;
            const x: [*]const T = @ptrCast(@alignCast(input.data.?));
            const y: [*]T = @ptrCast(@alignCast(output.data.?));
            for (0..3) |row| {
                y[row] = S.zero(T);
                for (0..3) |column| y[row] = S.add(T, y[row], S.mul(T, self.matrix[row * 3 + column], x[column]));
            }
            return if (self.calls == self.fail_at) -761 else 0;
        }
        fn inner(raw: ?*anyopaque, token: *const B.Token, x: W.const_buffer, y: W.const_buffer, out: *T) callconv(.c) i32 {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (@intFromPtr(token) == 0 or self.reduction_fail) return -762;
            const xp: [*]const T = @ptrCast(@alignCast(x.data.?));
            const yp: [*]const T = @ptrCast(@alignCast(y.data.?));
            out.* = N.Reductions.localInner(T, xp[0..@intCast(x.length)], yp[0..@intCast(y.length)]) catch return -763;
            return 0;
        }
        fn norm(raw: ?*anyopaque, token: *const B.Token, x: W.const_buffer, out: *S.Real(T)) callconv(.c) i32 {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (@intFromPtr(token) == 0 or self.reduction_fail) return -762;
            const xp: [*]const T = @ptrCast(@alignCast(x.data.?));
            out.* = N.Reductions.localNorm(T, xp[0..@intCast(x.length)]) catch return -763;
            return 0;
        }
        fn agree(raw: ?*anyopaque, _: ?*const B.Token, point: B.Point, local: W.result) callconv(.c) W.result {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (local.status != W.OK) return local;
            if (self.remote_rational_commit and point.stage == .rational_commit) return .{ .status = W.NONFINITE_INPUT, .reserved = 0, .info = 92 };
            if (self.remote_commit and point.stage == .commit) return .{ .status = W.NONFINITE_INPUT, .reserved = 0, .info = 91 };
            return local;
        }
    };
}
pub fn checkBiCGStabBoundary() !void {
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const Real = S.Real(T);
        const tolerance: Real = if (Real == f32) 3e-5 else 4e-12;
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer {
            _ = owner.requestClose();
            std.debug.assert(fixtureProgressClose(owner) == .closed);
        }
        var operation = try owner.begin();
        defer operation.end();
        var foreign = BiCGForeign(T){ .owner = owner, .matrix = .{ seriesSetValue(T, 4, -0.5), seriesSetValue(T, 1, 0.2), seriesSetValue(T, -0.25, 0), seriesSetValue(T, -2, -0.1), seriesSetValue(T, 5, 0.7), seriesSetValue(T, 1, 0), seriesSetValue(T, 0.5, 0), seriesSetValue(T, -1, 0.1), seriesSetValue(T, 3, -0.3) } };
        var inverse = BiCGForeign(T){ .owner = owner, .matrix = .{ value(T, 1), value(T, 0), value(T, 0), value(T, 0), value(T, 1), value(T, 0), value(T, 0), value(T, 0), value(T, 1) } };
        var inverse_action = inverse.action();
        const reduction: B.Reduction(T) = .{ .context = &foreign, .inner = BiCGForeign(T).inner, .norm = BiCGForeign(T).norm };
        const agreement: B.Agreement = .{ .context = &foreign, .agree = BiCGForeign(T).agree };
        const exact = [_]T{ seriesSetValue(T, 1, 0.5), seriesSetValue(T, -2, 0.25), seriesSetValue(T, 0.25, -0.75) };
        var b: [3]T = undefined;
        for (&b, 0..) |*out, row| {
            out.* = S.zero(T);
            for (exact, 0..) |x, column| out.* = S.add(T, out.*, S.mul(T, foreign.matrix[row * 3 + column], x));
        }
        const b_before = b;
        // The result-alias case must reach overlap validation for four-byte scalars.
        var output: [4]T align(@alignOf(B.LinearResult(T))) = @splat(value(T, 77));
        var result: B.LinearResult(T) = .{ .status = 41, .reserved = 42, .iterations = 43, .action_count = 44, .preconditioner_count = 45, .residual_restarts = 46, .residual_norm = 47 };
        const original = result;
        const rhs: W.const_buffer = .{ .data = &b, .length = 3 };
        const target: W.buffer = .{ .data = &output, .length = 4 };
        for ([_]bool{ false, true }) |agreed| {
            @memset(&output, value(T, 0));
            output[3] = value(T, 77);
            foreign.calls = 0;
            inverse.calls = 0;
            const status = if (agreed) B.bicgstabAgreedAdmitted(T, &operation, foreign.action(), &inverse_action, &reduction, agreement, rhs, 1, .{ .data = &output, .length = 3 }, 3, 32, if (Real == f32) 2e-6 else 1e-14, 0, value(T, 1), value(T, 0), target, &result) else B.bicgstabAdmitted(T, &operation, foreign.action(), null, null, rhs, 0, .{ .data = null, .length = -1 }, 3, 32, if (Real == f32) 2e-6 else 1e-14, 0, value(T, 1), value(T, 0), target, &result);
            try std.testing.expectEqual(W.OK, status.status);
            try std.testing.expectEqual(@as(i32, 0), result.status);
            try std.testing.expectEqual(@as(i64, @intCast(foreign.calls)), result.action_count);
            if (agreed) try std.testing.expectEqual(@as(i64, @intCast(inverse.calls)), result.preconditioner_count);
            for (exact, output[0..3]) |want, actual| {
                if (comptime S.isComplex(T)) {
                    try std.testing.expectApproxEqAbs(want.re, actual.re, tolerance);
                    try std.testing.expectApproxEqAbs(want.im, actual.im, tolerance);
                } else try std.testing.expectApproxEqAbs(want, actual, tolerance);
            }
            try std.testing.expectEqualDeep(value(T, 77), output[3]);
            try std.testing.expectEqualDeep(b_before, b);
        }
        // Exercise the public affine coefficients with a complex shift.
        const scale = value(T, -1);
        const shift = seriesSetValue(T, 6, 0.75);
        var shifted_rhs: [3]T = undefined;
        for (&shifted_rhs, b, exact) |*out, ax, x| out.* = S.add(T, S.mul(T, scale, ax), S.mul(T, shift, x));
        const shifted = B.bicgstabAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, .{ .data = &shifted_rhs, .length = 3 }, 0, .{ .data = null, .length = 0 }, 3, 32, if (Real == f32) 2e-6 else 1e-14, 0, scale, shift, target, &result);
        try std.testing.expectEqual(W.OK, shifted.status);
        try std.testing.expectEqual(@as(i32, 0), result.status);
        for (exact, output[0..3]) |want, actual| {
            if (comptime S.isComplex(T)) {
                try std.testing.expectApproxEqAbs(want.re, actual.re, tolerance);
                try std.testing.expectApproxEqAbs(want.im, actual.im, tolerance);
            } else try std.testing.expectApproxEqAbs(want, actual, tolerance);
        }
        @memset(&output, value(T, 77));
        result = original;
        const before = output;
        foreign.calls = 0;
        foreign.fail_at = 2;
        var failed = B.bicgstabAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, rhs, 0, .{ .data = null, .length = 0 }, 3, 32, 1e-6, 0, value(T, 1), value(T, 0), target, &result);
        try std.testing.expectEqual(B.ACTION_FAILED, failed.status);
        try std.testing.expectEqual(@as(i64, -761), failed.info);
        try std.testing.expectEqualDeep(before, output);
        try std.testing.expectEqualDeep(original, result);
        foreign.fail_at = 0;
        foreign.calls = 0;
        foreign.reduction_fail = true;
        failed = B.bicgstabAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, rhs, 0, .{ .data = null, .length = 0 }, 3, 32, 1e-6, 0, value(T, 1), value(T, 0), target, &result);
        try std.testing.expectEqual(B.REDUCTION_FAILED, failed.status);
        try std.testing.expectEqual(@as(i64, -762), failed.info);
        foreign.reduction_fail = false;
        inverse.calls = 0;
        inverse.fail_at = 1;
        failed = B.bicgstabAgreedAdmitted(T, &operation, foreign.action(), &inverse_action, &reduction, agreement, rhs, 0, .{ .data = null, .length = 0 }, 3, 32, 1e-6, 0, value(T, 1), value(T, 0), target, &result);
        try std.testing.expectEqual(B.ACTION_FAILED, failed.status);
        try std.testing.expectEqual(@as(i64, -761), failed.info);
        inverse.fail_at = 0;
        foreign.remote_commit = true;
        failed = B.bicgstabAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, rhs, 0, .{ .data = null, .length = 0 }, 3, 32, 1e-6, 0, value(T, 1), value(T, 0), target, &result);
        try std.testing.expectEqual(W.NONFINITE_INPUT, failed.status);
        try std.testing.expectEqual(@as(i64, 91), failed.info);
        foreign.remote_commit = false;
        const calls = foreign.calls;
        failed = B.bicgstabAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, rhs, 2, .{ .data = null, .length = 0 }, 3, 32, 1e-6, 0, value(T, 1), value(T, 0), target, &result);
        try std.testing.expectEqual(B.INVALID_OPTIONS, failed.status);
        failed = B.bicgstabAdmitted(T, &operation, foreign.action(), null, null, rhs, 0, .{ .data = null, .length = 0 }, 3, std.math.maxInt(i64), 1e-6, 0, value(T, 1), value(T, 0), target, &result);
        try std.testing.expectEqual(W.SIZE_OVERFLOW, failed.status);
        failed = B.bicgstabAdmitted(T, &operation, foreign.action(), null, null, rhs, 0, .{ .data = null, .length = 0 }, 3, 32, 1e-6, 0, value(T, std.math.nan(Real)), value(T, 0), target, &result);
        try std.testing.expectEqual(W.NONFINITE_INPUT, failed.status);
        failed = B.bicgstabAdmitted(T, &operation, foreign.action(), null, null, rhs, 0, .{ .data = null, .length = 0 }, 3, 32, 1e-6, 0, value(T, 1), value(T, std.math.inf(Real)), target, &result);
        try std.testing.expectEqual(W.NONFINITE_INPUT, failed.status);
        failed = B.bicgstabAdmitted(T, &operation, foreign.action(), null, null, rhs, 0, .{ .data = null, .length = 0 }, 3, 32, 1e-6, 0, value(T, 1), value(T, 0), target, null);
        try std.testing.expectEqual(W.NULL_POINTER, failed.status);
        failed = B.bicgstabAdmitted(T, &operation, foreign.action(), null, null, rhs, 0, .{ .data = null, .length = 0 }, 3, 32, 1e-6, 0, value(T, 1), value(T, 0), target, &output);
        try std.testing.expectEqual(W.ALIASING_NOT_ALLOWED, failed.status);
        try std.testing.expectEqual(calls, foreign.calls);
        try std.testing.expectEqualDeep(before, output);
        try std.testing.expectEqualDeep(original, result);
        const stopped = B.bicgstabAdmitted(T, &operation, foreign.action(), null, null, rhs, 0, .{ .data = null, .length = -1 }, 3, 0, 0, 0, value(T, 1), value(T, 0), target, &result);
        try std.testing.expectEqual(W.OK, stopped.status);
        try std.testing.expectEqual(@as(i32, 1), result.status);
        try std.testing.expectEqual(@as(i64, 0), result.iterations);
        try std.testing.expectEqual(@as(i64, 0), result.action_count);
        for (output[0..3]) |v| try std.testing.expectEqualDeep(value(T, 0), v);
        const retried = B.bicgstabAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, rhs, 0, .{ .data = null, .length = 0 }, 3, 32, 1e-6, 0, value(T, 1), value(T, 0), target, &result);
        try std.testing.expectEqual(W.OK, retried.status);
        try std.testing.expectEqual(@as(i32, 0), result.status);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
}

pub fn checkRationalFilterBoundary() !void {
    inline for (.{ S.ComplexF32, S.ComplexF64 }) |T| {
        const R = S.Real(T);
        const tolerance: R = if (R == f32) 4e-5 else 3e-12;
        try std.testing.expectEqual(@as(usize, 72), @sizeOf(B.RationalResult(T)));
        try std.testing.expectEqual(@as(usize, 64), @offsetOf(B.RationalResult(T), "max_residual_norm"));
        var reference_owner: FixtureReferenceContext = undefined;
        reference_owner.initInPlace(std.testing.allocator, E.test_support.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const owner = reference_owner.borrowedContext();
        defer {
            _ = owner.requestClose();
            std.debug.assert(fixtureProgressClose(owner) == .closed);
        }
        var operation = try owner.begin();
        defer operation.end();
        var foreign = BiCGForeign(T){ .owner = owner, .matrix = @splat(value(T, 0)) };
        var inverse = BiCGForeign(T){ .owner = owner, .matrix = .{ value(T, 1), value(T, 0), value(T, 0), value(T, 0), value(T, 1), value(T, 0), value(T, 0), value(T, 0), value(T, 1) } };
        const inverse_action = inverse.action();
        const reduction: B.Reduction(T) = .{ .context = &foreign, .inner = BiCGForeign(T).inner, .norm = BiCGForeign(T).norm };
        const agreement: B.Agreement = .{ .context = &foreign, .agree = BiCGForeign(T).agree };
        const input = [_]T{ seriesSetValue(T, 2, -1), seriesSetValue(T, -1, 1), seriesSetValue(T, 0.5, 0.25), seriesSetValue(T, 3, 0.5), seriesSetValue(T, -2, 1), seriesSetValue(T, 1, -0.5) };
        // The summary-alias case must reach overlap validation even for ComplexF32.
        var output: [8]T align(@alignOf(B.RationalResult(T))) = @splat(seriesSetValue(T, 77, -33));
        const before = output;
        var nodes = [_]T{ value(T, 1), value(T, 2) };
        const weights = [_]T{ value(T, 1), value(T, 1) };
        const nb: W.const_buffer = .{ .data = &nodes, .length = 2 };
        const wb: W.const_buffer = .{ .data = &weights, .length = 2 };
        var x: B.ConstPanel = .{ .dense = .{ .storage = .{ .data = &input, .length = 6 }, .rows = 3, .cols = 2, .ld = 3 }, .layout = 102 };
        var y: B.Panel = .{ .dense = .{ .storage = .{ .data = &output, .length = 8 }, .rows = 3, .cols = 2, .ld = 3 }, .layout = 102 };
        var summary: B.RationalResult(T) = .{ .status = 41, .linear_status = 42, .completed_systems = 43, .stopped_node = 44, .stopped_column = 45, .iterations = 46, .action_count = 47, .preconditioner_count = 48, .residual_restarts = 49, .max_residual_norm = 50 };
        const original = summary;
        for ([_]bool{ false, true }) |agreed| {
            foreign.calls = 0;
            inverse.calls = 0;
            if (agreed) {
                x.layout = 101;
                x.dense.ld = 2;
                y.layout = 101;
                y.dense.ld = 2;
            }
            const status = if (agreed) B.rationalFilterAgreedAdmitted(T, &operation, foreign.action(), &inverse_action, &reduction, agreement, x, nb, wb, 3, 8, if (R == f32) 1e-6 else 1e-14, 0, y, &summary) else B.rationalFilterAdmitted(T, &operation, foreign.action(), null, null, x, nb, wb, 3, 8, if (R == f32) 1e-6 else 1e-14, 0, y, &summary);
            try std.testing.expectEqual(W.OK, status.status);
            try std.testing.expectEqual(@as(i32, 0), summary.status);
            try std.testing.expectEqual(@as(i32, -1), summary.linear_status);
            try std.testing.expectEqual(@as(i64, -1), summary.stopped_node);
            try std.testing.expectEqual(@as(i64, -1), summary.stopped_column);
            try std.testing.expectEqual(@as(i64, 4), summary.completed_systems);
            try std.testing.expectEqual(@as(i64, @intCast(foreign.calls)), summary.action_count);
            if (agreed) try std.testing.expectEqual(@as(i64, @intCast(inverse.calls)), summary.preconditioner_count);
            for (input, output[0..6]) |b, actual| {
                const expected = S.mul(T, value(T, 1.5), b);
                try std.testing.expectApproxEqAbs(expected.re, actual.re, tolerance);
                try std.testing.expectApproxEqAbs(expected.im, actual.im, tolerance);
            }
            try std.testing.expectEqualDeep(before[6..8].*, output[6..8].*);
        }
        output = before;
        summary = original;
        foreign.calls = 0;
        foreign.fail_at = 5;
        var failed = B.rationalFilterAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, x, nb, wb, 3, 8, 1e-6, 0, y, &summary);
        try std.testing.expectEqual(B.ACTION_FAILED, failed.status);
        try std.testing.expectEqual(@as(i64, -761), failed.info);
        try std.testing.expectEqualDeep(before, output);
        try std.testing.expectEqualDeep(original, summary);
        foreign.fail_at = 0;
        foreign.reduction_fail = true;
        failed = B.rationalFilterAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, x, nb, wb, 3, 8, 1e-6, 0, y, &summary);
        try std.testing.expectEqual(B.REDUCTION_FAILED, failed.status);
        try std.testing.expectEqual(@as(i64, -762), failed.info);
        foreign.reduction_fail = false;
        inverse.calls = 0;
        inverse.fail_at = 2;
        failed = B.rationalFilterAgreedAdmitted(T, &operation, foreign.action(), &inverse_action, &reduction, agreement, x, nb, wb, 3, 8, 1e-6, 0, y, &summary);
        try std.testing.expectEqual(B.ACTION_FAILED, failed.status);
        inverse.fail_at = 0;
        foreign.remote_rational_commit = true;
        failed = B.rationalFilterAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, x, nb, wb, 3, 8, 1e-6, 0, y, &summary);
        try std.testing.expectEqual(W.NONFINITE_INPUT, failed.status);
        try std.testing.expectEqual(@as(i64, 92), failed.info);
        try std.testing.expectEqualDeep(before, output);
        try std.testing.expectEqualDeep(original, summary);
        foreign.remote_rational_commit = false;
        const calls = foreign.calls;
        failed = B.rationalFilterAdmitted(T, &operation, foreign.action(), null, null, x, nb, wb, 3, @divTrunc(std.math.maxInt(i64) - 1, 3), 1e-6, 0, y, &summary);
        try std.testing.expectEqual(W.SIZE_OVERFLOW, failed.status);
        failed = B.rationalFilterAdmitted(T, &operation, foreign.action(), null, null, x, nb, wb, 3, 8, 1e-6, 0, y, null);
        try std.testing.expectEqual(W.NULL_POINTER, failed.status);
        failed = B.rationalFilterAdmitted(T, &operation, foreign.action(), null, null, x, nb, wb, 3, 8, 1e-6, 0, y, &output);
        try std.testing.expectEqual(W.ALIASING_NOT_ALLOWED, failed.status);
        failed = B.rationalFilterAdmitted(T, &operation, foreign.action(), null, null, x, nb, wb, 3, 8, 1e-6, 0, y, @ptrFromInt(@intFromPtr(&summary) + 1));
        try std.testing.expectEqual(W.MISALIGNED_POINTER, failed.status);
        y.reserved = 1;
        failed = B.rationalFilterAdmitted(T, &operation, foreign.action(), null, null, x, nb, wb, 3, 8, 1e-6, 0, y, &summary);
        try std.testing.expectEqual(W.INVALID_FLAG, failed.status);
        y.reserved = 0;
        const saved_nodes = nodes;
        nodes[0] = seriesSetValue(T, std.math.nan(R), 0);
        failed = B.rationalFilterAdmitted(T, &operation, foreign.action(), null, null, x, nb, wb, 3, 8, 1e-6, 0, y, &summary);
        try std.testing.expectEqual(W.NONFINITE_INPUT, failed.status);
        nodes = saved_nodes;
        try std.testing.expectEqual(calls, foreign.calls);
        try std.testing.expectEqualDeep(before, output);
        try std.testing.expectEqualDeep(original, summary);
        nodes[1] = value(T, 0);
        const stopped = B.rationalFilterAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, x, nb, wb, 3, 8, 1e-6, 0, y, &summary);
        try std.testing.expectEqual(W.OK, stopped.status);
        try std.testing.expectEqual(@as(i32, 1), summary.status);
        try std.testing.expectEqual(@as(i32, 3), summary.linear_status);
        try std.testing.expectEqual(@as(i64, 2), summary.completed_systems);
        try std.testing.expectEqual(@as(i64, 1), summary.stopped_node);
        try std.testing.expectEqual(@as(i64, 0), summary.stopped_column);
        try std.testing.expectEqualDeep(before, output);
        nodes = saved_nodes;
        const retried = B.rationalFilterAgreedAdmitted(T, &operation, foreign.action(), null, &reduction, agreement, x, nb, wb, 3, 8, 1e-6, 0, y, &summary);
        try std.testing.expectEqual(W.OK, retried.status);
        try std.testing.expectEqual(@as(i32, 0), summary.status);
        try std.testing.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    }
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
