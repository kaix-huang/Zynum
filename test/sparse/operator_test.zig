// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const CsrBackend = core.internal.sparse_checked.operator.CsrBackend;
const Operator = core.internal.numerics.Operator;
const core = @import("zynum-core");
const execution = @import("zynum-execution");
const Z = core.sparse.ComplexF64;
const Callback = struct {
    fail: bool = false,
    fail_after_write: bool = false,
    fail_on_call: usize = 0,
    calls: *usize,
    fn apply(raw: *const anyopaque, operation: *const execution.context.Operation, x: []const Z, y: []Z, work: []Z) !void {
        const self: *const Callback = @ptrCast(@alignCast(raw));
        try std.testing.expectEqual(@as(usize, 1), operation.borrowedContext().snapshot().active_operations);
        self.calls.* += 1;
        if (self.fail) return error.BackendFailure;
        // Independent literal dense action: [1+i, 2; 0, 3-i].
        work[0] = .{ .re = x[0].re - x[0].im + 2 * x[1].re, .im = x[0].re + x[0].im + 2 * x[1].im };
        y[0] = work[0];
        y[1] = .{ .re = 3 * x[1].re + x[1].im, .im = 3 * x[1].im - x[1].re };
        if (self.fail_after_write or (self.fail_on_call != 0 and self.calls.* == self.fail_on_call)) return error.BackendFailure;
    }
};
test "CSR and matrix-free operators borrow one admission through close" {
    var ctx = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
    var operation = try ctx.begin();
    const offsets = [_]i64{ 0, 2, 3 };
    const indices = [_]i32{ 0, 1, 1 };
    const values = [_]Z{ .{ .re = 1, .im = 1 }, .{ .re = 2, .im = 0 }, .{ .re = 3, .im = -1 } };
    const matrix = try core.sparse.constCsrMatrixWithIndexTypes(Z, i64, i32, &offsets, &indices, &values, .{ .row_count = 2, .column_count = 2 });
    const csr = CsrBackend(@TypeOf(matrix)).borrow(&matrix);
    var calls: usize = 0;
    const backend = Callback{ .calls = &calls };
    const callback = Operator(Z){ .context = &backend, .rows = 2, .columns = 2, .workspace_elements = 1, .apply_fn = Callback.apply };
    const x = [_]Z{ .{ .re = 2, .im = 1 }, .{ .re = 1, .im = -2 } };
    var first: [2]Z = undefined;
    var second: [2]Z = undefined;
    var work = [_]Z{ .{ .re = 91, .im = 92 }, .{ .re = 93, .im = 94 } };
    try std.testing.expectEqual(execution.context.RequestResult.pending, ctx.requestClose());
    try csr.applyAdmitted(&operation, &x, &first, &.{});
    try callback.applyAdmitted(&operation, &x, &second, &work);
    try std.testing.expectEqualSlices(Z, &first, &second);
    try std.testing.expectEqualSlices(Z, &.{ .{ .re = 3, .im = -1 }, .{ .re = 1, .im = -7 } }, &second);
    try std.testing.expectEqual(Z{ .re = 93, .im = 94 }, work[1]);
    try std.testing.expectEqual(@as(usize, 1), calls);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    // Same scaled action through the fused CSR and matrix-free fallback.
    const alpha = Z{ .re = 2, .im = 1 };
    const beta = Z{ .re = -1, .im = 0 };
    first = .{ .{ .re = 4, .im = 5 }, .{ .re = 6, .im = 7 } };
    second = first;
    var scaled_work = [_]Z{ .{ .re = 90, .im = 91 }, .{ .re = 92, .im = 93 }, .{ .re = 94, .im = 95 }, .{ .re = 96, .im = 97 } };
    try std.testing.expectEqual(@as(usize, 0), try csr.scaledWorkspaceElements(alpha, beta));
    try std.testing.expectEqual(@as(usize, 3), try callback.scaledWorkspaceElements(alpha, beta));
    try csr.applyScaledAdmitted(&operation, alpha, &x, beta, &first, &.{});
    try callback.applyScaledAdmitted(&operation, alpha, &x, beta, &second, &scaled_work);
    try std.testing.expectEqualSlices(Z, &.{ .{ .re = 3, .im = -4 }, .{ .re = 3, .im = -20 } }, &first);
    try std.testing.expectEqualSlices(Z, &first, &second);
    try std.testing.expectEqual(Z{ .re = 96, .im = 97 }, scaled_work[3]);
    second = .{ .{ .re = std.math.nan(f64), .im = std.math.nan(f64) }, .{ .re = std.math.nan(f64), .im = std.math.nan(f64) } };
    try callback.applyScaledAdmitted(&operation, .{ .re = 2, .im = 0 }, &x, .{ .re = 0, .im = 0 }, &second, &scaled_work);
    try std.testing.expectEqualSlices(Z, &.{ .{ .re = 6, .im = -2 }, .{ .re = 2, .im = -14 } }, &second);
    const calls_before_zero = calls;
    second = .{ .{ .re = std.math.nan(f64), .im = std.math.nan(f64) }, .{ .re = std.math.nan(f64), .im = std.math.nan(f64) } };
    try callback.applyScaledAdmitted(&operation, .{ .re = 0, .im = 0 }, &x, .{ .re = 0, .im = 0 }, &second, &.{});
    try std.testing.expectEqualSlices(Z, &.{ .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 } }, &second);
    try std.testing.expectEqual(calls_before_zero, calls);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
    try panelNumerics(&operation, csr, callback, &calls);
    operation.end();
    _ = ctx.requestClose();
    try std.testing.expectEqual(execution.context.CloseResult.closed, ctx.tryProgressClose());
}
test "operator preflight rejects before callback and propagates backend error" {
    var ctx = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
    var operation = try ctx.begin();
    var calls: usize = 0;
    const backend = Callback{ .calls = &calls, .fail = true };
    var callback = Operator(Z){ .context = &backend, .rows = 2, .columns = 2, .workspace_elements = 1, .apply_fn = Callback.apply };
    var x: [2]Z = @splat(.{ .re = 1, .im = 2 });
    var y: [2]Z = @splat(.{ .re = 91, .im = 92 });
    var work: [1]Z = undefined;
    try std.testing.expectError(error.DimensionMismatch, callback.applyAdmitted(&operation, x[0..1], &y, &work));
    try std.testing.expectError(error.WorkspaceTooSmall, callback.applyAdmitted(&operation, &x, &y, &.{}));
    try std.testing.expectError(error.AliasingNotAllowed, callback.applyAdmitted(&operation, &x, &x, &work));
    try std.testing.expectError(error.AliasingNotAllowed, callback.applyAdmitted(&operation, &x, &y, x[0..1]));
    try std.testing.expectError(error.AliasingNotAllowed, callback.applyAdmitted(&operation, &x, &y, y[0..1]));
    callback.workspace_elements = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, callback.applyAdmitted(&operation, &x, &y, &work));
    try std.testing.expectError(error.SizeOverflow, callback.scaledWorkspaceElements(.{ .re = 2, .im = 0 }, .{ .re = 1, .im = 0 }));
    callback.workspace_elements = 1;
    try std.testing.expectEqual(@as(usize, 0), calls);
    try std.testing.expectError(error.BackendFailure, callback.applyAdmitted(&operation, &x, &y, &work));
    var fallback_work: [4]Z = undefined;
    const alpha = Z{ .re = 2, .im = 0 };
    const beta = Z{ .re = 1, .im = 0 };
    try std.testing.expectError(error.WorkspaceTooSmall, callback.applyScaledAdmitted(&operation, alpha, &x, beta, &y, fallback_work[0..2]));
    callback.workspace_elements = 0;
    try std.testing.expectError(error.AliasingNotAllowed, callback.applyScaledAdmitted(&operation, alpha, &x, beta, &y, &x));
    callback.workspace_elements = 1;
    try std.testing.expectError(error.BackendFailure, callback.applyScaledAdmitted(&operation, alpha, &x, beta, &y, &fallback_work));
    const late_backend = Callback{ .calls = &calls, .fail_after_write = true };
    callback.context = &late_backend;
    try std.testing.expectError(error.BackendFailure, callback.applyScaledAdmitted(&operation, alpha, &x, beta, &y, &fallback_work));
    try std.testing.expectEqual(@as(usize, 3), calls);
    try std.testing.expectEqualSlices(Z, &.{ .{ .re = 91, .im = 92 }, .{ .re = 91, .im = 92 } }, &y);
    try panelFailures(&operation, callback, &calls);
    operation.end();
    _ = ctx.requestClose();
    try std.testing.expectEqual(execution.context.CloseResult.closed, ctx.tryProgressClose());
}

test "CSR operator revalidates borrowed metadata and preserves output on rejection" {
    var operation = execution.beginDefault();
    defer operation.end();
    var offsets = [_]i64{ 0, 1, 2 };
    var indices = [_]i32{ 0, 1 };
    var values = [_]f64{ 2, 3 };
    var matrix = try core.sparse.constCsrMatrixWithIndexTypes(f64, i64, i32, &offsets, &indices, &values, .{ .row_count = 2, .column_count = 2 });
    const action = CsrBackend(@TypeOf(matrix)).borrow(&matrix);
    const x = [_]f64{ 5, 7 };
    var y = [_]f64{ 91, 92 };
    offsets[1] = 99;
    try std.testing.expectError(error.InvalidRowOffsets, action.applyAdmitted(&operation, &x, &y, &.{}));
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &y);
    try std.testing.expectError(error.InvalidRowOffsets, action.applyScaledAdmitted(&operation, 0, &x, 0, &y, &.{}));
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &y);
    const invalid_input: Dense.ConstPanel(f64) = .{ .values = &x, .shape = .{ .row_count = 2, .column_count = 1, .leading_dimension = 2, .layout = .column_major } };
    const invalid_output: Dense.Panel(f64) = .{ .values = &y, .shape = invalid_input.shape };
    try std.testing.expectError(error.InvalidRowOffsets, action.applyPanelAdmitted(&operation, 0, invalid_input, 0, invalid_output, &.{}));
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &y);
    offsets[1] = 1;
    indices[1] = 2;
    try std.testing.expectError(error.InvalidColumnIndex, action.applyAdmitted(&operation, &x, &y, &.{}));
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &y);
    indices[1] = 1;
    values[1] = 4;
    try action.applyAdmitted(&operation, &x, &y, &.{});
    try std.testing.expectEqualSlices(f64, &.{ 10, 28 }, &y);
    // Matrix-storage alias is checked by the existing CSR provider.
    try std.testing.expectError(error.AliasingNotAllowed, action.applyAdmitted(&operation, &x, &values, &.{}));
    try std.testing.expectEqualSlices(f64, &.{ 2, 4 }, &values);
    matrix.column_count = 3;
    try std.testing.expectError(error.DimensionMismatch, action.applyAdmitted(&operation, &x, &y, &.{}));
    try std.testing.expectEqualSlices(f64, &.{ 10, 28 }, &y);
}

test "CSR operator handles rectangular transpose adjoint and empty products" {
    var operation = execution.beginDefault();
    defer operation.end();
    const offsets = [_]u64{ 0, 2 };
    const indices = [_]u32{ 0, 1 };
    const values = [_]Z{ .{ .re = 1, .im = 2 }, .{ .re = 3, .im = -1 } };
    const x = [_]Z{.{ .re = 2, .im = 1 }};
    for ([_]core.sparse.MatrixTransform{ .transposed, .adjoint }) |transform| {
        const matrix = try core.sparse.constCsrMatrixWithIndexTypes(Z, u64, u32, &offsets, &indices, &values, .{ .row_count = 1, .column_count = 2, .operation = transform });
        const action = CsrBackend(@TypeOf(matrix)).borrow(&matrix);
        try std.testing.expectEqual(@as(usize, 2), action.rows);
        try std.testing.expectEqual(@as(usize, 1), action.columns);
        var y: [2]Z = undefined;
        try action.applyAdmitted(&operation, &x, &y, &.{});
        const expected = if (transform == .transposed) [_]Z{ .{ .re = 0, .im = 5 }, .{ .re = 7, .im = 1 } } else [_]Z{ .{ .re = 4, .im = -3 }, .{ .re = 5, .im = 5 } };
        try std.testing.expectEqualSlices(Z, &expected, &y);
        const panel_x = [_]Z{ x[0], .{ .re = 4, .im = 2 } };
        const input: Dense.ConstPanel(Z) = .{ .values = &panel_x, .shape = .{ .row_count = 1, .column_count = 2, .leading_dimension = 2, .layout = .row_major } };
        var panel_y: [6]Z = @splat(.{ .re = std.math.nan(f64), .im = std.math.nan(f64) });
        panel_y[2] = .{ .re = 91, .im = 92 };
        panel_y[5] = panel_y[2];
        const output: Dense.Panel(Z) = .{ .values = &panel_y, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = .row_major } };
        try action.applyPanelAdmitted(&operation, .{ .re = 1, .im = 0 }, input, .{ .re = 0, .im = 0 }, output, &.{});
        for (expected, 0..) |value, row| {
            try std.testing.expectEqual(value, output.column(0).at(row).*);
            try std.testing.expectEqual(Z{ .re = 2 * value.re, .im = 2 * value.im }, output.column(1).at(row).*);
        }
        try std.testing.expectEqual(Z{ .re = 91, .im = 92 }, panel_y[2]);
        try std.testing.expectEqual(panel_y[2], panel_y[5]);
    }
    const empty_offsets = [_]i64{ 0, 0, 0 };
    const empty_indices = [_]i32{};
    const empty_values = [_]f64{};
    const empty = try core.sparse.constCsrMatrixWithIndexTypes(f64, i64, i32, &empty_offsets, &empty_indices, &empty_values, .{ .row_count = 2, .column_count = 0 });
    const empty_action = CsrBackend(@TypeOf(empty)).borrow(&empty);
    var y = [_]f64{ std.math.nan(f64), 91 };
    try empty_action.applyAdmitted(&operation, &.{}, &y, &.{});
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, &y);
    const zero_offsets = [_]i64{0};
    const zero = try core.sparse.constCsrMatrixWithIndexTypes(f64, i64, i32, &zero_offsets, &empty_indices, &empty_values, .{ .row_count = 0, .column_count = 2 });
    try CsrBackend(@TypeOf(zero)).borrow(&zero).applyAdmitted(&operation, &.{ 3, 4 }, &.{}, &.{});
    const zero_input: Dense.ConstPanel(f64) = .{ .values = &.{ 3, 4, 5, 6 }, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 2, .layout = .row_major } };
    const zero_output: Dense.Panel(f64) = .{ .values = &.{}, .shape = .{ .row_count = 0, .column_count = 2, .leading_dimension = std.math.maxInt(usize), .layout = .column_major } };
    try CsrBackend(@TypeOf(zero)).borrow(&zero).applyPanelAdmitted(&operation, 1, zero_input, 0, zero_output, &.{});
    const empty_input: Dense.ConstPanel(f64) = .{ .values = &.{}, .shape = .{ .row_count = 0, .column_count = 1, .leading_dimension = std.math.maxInt(usize), .layout = .column_major } };
    const empty_output: Dense.Panel(f64) = .{ .values = &y, .shape = .{ .row_count = 2, .column_count = 1, .leading_dimension = 2, .layout = .column_major } };
    y = .{ std.math.nan(f64), std.math.nan(f64) };
    try empty_action.applyPanelAdmitted(&operation, 1, empty_input, 0, empty_output, &.{});
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, &y);
}

const Dense = core.internal.numerics.Dense;
fn panelNumerics(operation: *const execution.context.Operation, csr: Operator(Z), callback: Operator(Z), calls: *usize) !void {
    const sentinel = Z{ .re = 91, .im = 92 };
    const alpha = Z{ .re = 2, .im = 1 };
    const beta = Z{ .re = -1, .im = 0 };
    const layouts = [_]Dense.Layout{ .row_major, .column_major };
    for (layouts) |input_layout| for (layouts) |output_layout| {
        var x: [7]Z = @splat(sentinel);
        const input_mutable: Dense.Panel(Z) = .{ .values = &x, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = input_layout } };
        input_mutable.column(0).at(0).* = .{ .re = 2, .im = 1 };
        input_mutable.column(0).at(1).* = .{ .re = 1, .im = -2 };
        input_mutable.column(1).at(0).* = .{ .re = 4, .im = 2 };
        input_mutable.column(1).at(1).* = .{ .re = 2, .im = -4 };
        const saved_x = x;
        var first: [7]Z = @splat(sentinel);
        var second: [7]Z = @splat(sentinel);
        const output: Dense.Panel(Z) = .{ .values = &first, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = output_layout } };
        const fallback: Dense.Panel(Z) = .{ .values = &second, .shape = output.shape };
        for (0..2) |column| {
            output.column(column).at(0).* = .{ .re = 4, .im = 5 };
            output.column(column).at(1).* = .{ .re = 6, .im = 7 };
        }
        second = first;
        var work: [8]Z = @splat(sentinel);
        try std.testing.expectEqual(@as(usize, 0), try csr.panelWorkspaceElements(alpha, 2));
        try std.testing.expectEqual(@as(usize, 7), try callback.panelWorkspaceElements(alpha, 2));
        try csr.applyPanelAdmitted(operation, alpha, input_mutable.asConst(), beta, output, &.{});
        try callback.applyPanelAdmitted(operation, alpha, input_mutable.asConst(), beta, fallback, &work);
        try std.testing.expectEqualSlices(Z, &first, &second);
        try std.testing.expectEqualSlices(Z, &saved_x, &x);
        try std.testing.expectEqual(Z{ .re = 3, .im = -4 }, output.column(0).at(0).*);
        try std.testing.expectEqual(Z{ .re = 3, .im = -20 }, output.column(0).at(1).*);
        try std.testing.expectEqual(Z{ .re = 10, .im = -3 }, output.column(1).at(0).*);
        try std.testing.expectEqual(Z{ .re = 12, .im = -33 }, output.column(1).at(1).*);
        try std.testing.expectEqual(sentinel, work[7]);
        for ([_]usize{ 2, 5, 6 }) |padding| try std.testing.expectEqual(sentinel, first[padding]);
        for (0..2) |column| for (0..2) |row| {
            fallback.column(column).at(row).* = .{ .re = std.math.nan(f64), .im = std.math.nan(f64) };
        };
        try callback.applyPanelAdmitted(operation, .{ .re = 2, .im = 0 }, input_mutable.asConst(), .{ .re = 0, .im = 0 }, fallback, &work);
        try std.testing.expectEqual(Z{ .re = 6, .im = -2 }, fallback.column(0).at(0).*);
        try std.testing.expectEqual(Z{ .re = 4, .im = -28 }, fallback.column(1).at(1).*);
        const saved_calls = calls.*;
        try callback.applyPanelAdmitted(operation, .{ .re = 0, .im = 0 }, input_mutable.asConst(), .{ .re = 0, .im = 0 }, fallback, &.{});
        try std.testing.expectEqual(saved_calls, calls.*);
        for (0..2) |column| for (0..2) |row| try std.testing.expectEqual(Z{ .re = 0, .im = 0 }, fallback.column(column).at(row).*);
        for ([_]usize{ 2, 5, 6 }) |padding| try std.testing.expectEqual(sentinel, second[padding]);
    };
    const empty_input: Dense.ConstPanel(Z) = .{ .values = &.{}, .shape = .{ .row_count = 2, .column_count = 0, .leading_dimension = 0, .layout = .column_major } };
    const empty_output: Dense.Panel(Z) = .{ .values = &.{}, .shape = empty_input.shape };
    const saved_calls = calls.*;
    try callback.applyPanelAdmitted(operation, alpha, empty_input, beta, empty_output, &.{});
    try csr.applyPanelAdmitted(operation, alpha, empty_input, beta, empty_output, &.{});
    try std.testing.expectEqual(saved_calls, calls.*);
    try std.testing.expectEqual(@as(usize, 1), operation.borrowedContext().snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 0), operation.borrowedContext().snapshot().resource_pins);
}
fn panelFailures(operation: *const execution.context.Operation, original: Operator(Z), calls: *usize) !void {
    var callback = original;
    var x: [7]Z = @splat(.{ .re = 1, .im = 2 });
    var y: [7]Z = @splat(.{ .re = 91, .im = 92 });
    const saved_y = y;
    const input: Dense.ConstPanel(Z) = .{ .values = &x, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = .row_major } };
    const output: Dense.Panel(Z) = .{ .values = &y, .shape = input.shape };
    var work: [8]Z = undefined;
    const alpha = Z{ .re = 2, .im = 0 };
    const beta = Z{ .re = 1, .im = 0 };
    const saved_calls = calls.*;
    try std.testing.expectError(error.WorkspaceTooSmall, callback.applyPanelAdmitted(operation, alpha, input, beta, output, work[0..6]));
    try std.testing.expectError(error.AliasingNotAllowed, callback.applyPanelAdmitted(operation, alpha, input, beta, output, &x));
    try std.testing.expectError(error.AliasingNotAllowed, callback.applyPanelAdmitted(operation, alpha, input, beta, .{ .values = &x, .shape = input.shape }, &work));
    var invalid = input;
    invalid.shape.leading_dimension = 1;
    try std.testing.expectError(error.InvalidLeadingDimension, callback.applyPanelAdmitted(operation, alpha, invalid, beta, output, &work));
    invalid = input;
    invalid.shape.row_count = 1;
    try std.testing.expectError(error.DimensionMismatch, callback.applyPanelAdmitted(operation, alpha, invalid, beta, output, &work));
    callback.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, callback.panelWorkspaceElements(alpha, 2));
    callback = original;
    try std.testing.expectEqual(saved_calls, calls.*);
    const later = Callback{ .calls = calls, .fail_on_call = saved_calls + 2 };
    callback.context = &later;
    try std.testing.expectError(error.BackendFailure, callback.applyPanelAdmitted(operation, alpha, input, beta, output, &work));
    try std.testing.expectEqual(saved_calls + 2, calls.*);
    try std.testing.expectEqualSlices(Z, &saved_y, &y);
}

test "unified inverse triangular operator mixed complex N T H scalar panel and closing" {
    const S = core.sparse;
    const D = core.internal.numerics.Dense;
    inline for (.{ f32, f64, S.ComplexF32, S.ComplexF64 }) |T| {
        const complex = T == S.ComplexF32 or T == S.ComplexF64;
        const V = struct {
            fn value(re: f64, im: f64) T {
                return if (comptime complex) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
            }
        };
        const values = [_]T{ V.value(2, 0), V.value(1, 0), V.value(3, 0) };
        const offsets = [_]i64{ 0, 1, 3 };
        const columns = [_]i32{ 0, 0, 1 };
        for ([_]S.MatrixTransform{ .normal, .transposed, .adjoint }) |transform| {
            const Matrix = S.ConstCsrMatrixWithIndexTypes(T, i64, i32);
            const Backend = core.internal.sparse_checked.operator.TriangularInverseBackend(Matrix);
            var state: Backend = .{ .matrix = .{ .values = &values, .row_offsets = &offsets, .column_indices = &columns, .row_count = 2, .column_count = 2, .operation = transform }, .triangle = .lower, .diagonal = .non_unit };
            const inverse = try state.borrow();
            try std.testing.expectEqual(@as(usize, 4), inverse.workspace_elements);
            const x: [2]T = if (transform == .normal) .{ V.value(4, 0), V.value(11, 0) } else .{ V.value(7, 0), V.value(9, 0) };
            var y = [_]T{ V.value(99, 0), V.value(99, 0) };
            var workspace: [4]T = undefined;
            var ctx = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
            var op = try ctx.begin();
            _ = ctx.requestClose();
            try inverse.applyAdmitted(&op, &x, &y, &workspace);
            if (comptime complex) {
                try std.testing.expectApproxEqAbs(@as(@TypeOf(y[0].re), 2), y[0].re, 1e-5);
                try std.testing.expectApproxEqAbs(@as(@TypeOf(y[1].re), 3), y[1].re, 1e-5);
            } else {
                try std.testing.expectApproxEqAbs(@as(T, 2), y[0], 1e-5);
                try std.testing.expectApproxEqAbs(@as(T, 3), y[1], 1e-5);
            }
            var px: [6]T = @splat(V.value(77, 0));
            var py: [6]T = @splat(V.value(88, 0));
            for (0..2) |j| for (0..2) |i| {
                px[i + j * 3] = x[i];
            };
            var panel_work: [10]T = undefined;
            try std.testing.expectEqual(@as(usize, 10), try inverse.panelWorkspaceElements(V.value(1, 0), 2));
            try inverse.applyPanelAdmitted(&op, V.value(1, 0), D.ConstPanel(T){ .values = &px, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = .column_major } }, V.value(0, 0), D.Panel(T){ .values = &py, .shape = .{ .row_count = 2, .column_count = 2, .leading_dimension = 3, .layout = .column_major } }, &panel_work);
            try std.testing.expectEqual(V.value(88, 0), py[2]);
            try std.testing.expectEqual(V.value(88, 0), py[5]);
            try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().resource_pins);
            op.end();
            try std.testing.expectEqual(execution.context.CloseResult.closed, ctx.tryProgressClose());
        }
    }
}

test "inverse triangular operator failure leaves output and protects borrowed matrix aliases" {
    const S = core.sparse;
    var values = [_]f64{ 2, 1, 0 };
    const offsets = [_]i64{ 0, 1, 3 };
    const columns = [_]i32{ 0, 0, 1 };
    const Matrix = S.ConstCsrMatrixWithIndexTypes(f64, i64, i32);
    const Backend = core.internal.sparse_checked.operator.TriangularInverseBackend(Matrix);
    var state: Backend = .{ .matrix = .{ .values = &values, .row_offsets = &offsets, .column_indices = &columns, .row_count = 2, .column_count = 2 }, .triangle = .lower, .diagonal = .non_unit };
    const inverse = try state.borrow();
    var ctx = execution.context.Context.init(std.testing.allocator, execution.test_support.policy, .internal_closable);
    var op = try ctx.begin();
    const x = [_]f64{ 4, 11 };
    var y = [_]f64{ 99, 99 };
    var workspace: [4]f64 = undefined;
    try std.testing.expectError(error.TriangularSolveFailed, inverse.applyAdmitted(&op, &x, &y, &workspace));
    try std.testing.expectEqualSlices(f64, &.{ 99, 99 }, &y);
    const before = values;
    try std.testing.expectError(error.AliasingNotAllowed, inverse.applyAdmitted(&op, &x, values[0..2], &workspace));
    try std.testing.expectEqualSlices(f64, &before, &values);
    op.end();
    _ = ctx.requestClose();
    try std.testing.expectEqual(execution.context.CloseResult.closed, ctx.tryProgressClose());
}
