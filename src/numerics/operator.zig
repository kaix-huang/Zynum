// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal synchronous overwrite/scaled actions. No ABI, allocation or lifetime owner.
//! Context and callback code remain alive through return; concurrent use requires
//! backend support. A callback error may leave output/workspace partially changed.
const std = @import("std");
const scalar = @import("../blas/core/shared/scalar.zig");
const execution = @import("zynum-execution");
const storage = @import("storage.zig");
const dense = @import("dense.zig");
const synchronization = @import("synchronization.zig");

pub fn Operator(comptime T: type) type {
    if (T != f32 and T != f64 and T != scalar.ComplexF32 and T != scalar.ComplexF64)
        @compileError("unsupported operator scalar");
    return struct {
        const Self = @This();
        context: *const anyopaque,
        rows: usize,
        columns: usize,
        workspace_elements: usize = 0,
        synchronization: ?synchronization.Synchronization = null,
        apply_fn: *const fn (*const anyopaque, *const execution.context.Operation, []const T, []T, []T) anyerror!void,

        /// Optional fused scale/accumulate action over the same admission.
        apply_scaled_fn: ?*const fn (*const anyopaque, *const execution.context.Operation, T, []const T, T, []T, []T) anyerror!void = null,

        /// Optional fused multi-column action using the common dense panels.
        apply_panel_fn: ?*const fn (*const anyopaque, *const execution.context.Operation, T, dense.ConstPanel(T), T, dense.Panel(T), []T) anyerror!void = null,

        /// Matrix-free fallback packs one input column and keeps every product
        /// until all callbacks succeed. No final output changes on callback error.
        pub fn panelWorkspaceElements(self: Self, alpha: T, right_columns: usize) storage.Error!usize {
            if (self.apply_panel_fn != null) return self.workspace_elements;
            if (scalar.isZero(T, alpha) or right_columns == 0 or self.rows == 0) return 0;
            const products = std.math.mul(usize, self.rows, right_columns) catch return error.SizeOverflow;
            const prefix = std.math.add(usize, self.workspace_elements, self.columns) catch return error.SizeOverflow;
            return std.math.add(usize, prefix, products) catch error.SizeOverflow;
        }

        /// Synchronous alpha*A*X + beta*Y over row/column-major panels. Geometry,
        /// bounding-prefix aliases and workspace capacity are checked before
        /// any callback or mutation. Padding, storage tails and scratch tails
        /// are preserved. The fused backend retains its own error semantics.
        pub fn applyPanelAdmitted(self: Self, operation: *const execution.context.Operation, alpha: T, input: dense.ConstPanel(T), beta: T, output: dense.Panel(T), workspace: []T) !void {
            _ = operation.borrowedContext();
            try synchronization.enter(T, self.synchronization, operation, .operator_panel);
            const prepared = try synchronization.resolve(self.synchronization, operation, .{ .stage = .preflight }, self.panelPreflight(alpha, input, output, workspace));
            const scratch = workspace[0..prepared];
            try synchronization.check(self.synchronization, operation, .{ .stage = .panel_columns, .control = @intCast(input.shape.column_count) });
            try synchronization.check(self.synchronization, operation, .{ .stage = .panel_backend, .control = @intFromBool(self.apply_panel_fn != null) });
            try synchronization.scales(T, self.synchronization, operation, alpha, beta);
            if (self.apply_panel_fn) |apply_panel| return synchronization.resolve(self.synchronization, operation, .{ .stage = .action_complete }, apply_panel(self.context, operation, alpha, input, beta, output, scratch));
            if (output.shape.column_count == 0 or (self.rows == 0 and self.synchronization == null)) return;
            if (scalar.isZero(T, alpha)) {
                try synchronization.check(self.synchronization, operation, .{ .stage = .commit });
                if (!scalar.isOne(T, beta)) for (0..output.shape.column_count) |column| {
                    const destination = output.column(column);
                    for (0..self.rows) |row| destination.at(row).* = if (scalar.isZero(T, beta)) scalar.zero(T) else scalar.mul(T, beta, destination.at(row).*);
                };
                return;
            }
            const packed_input = scratch[self.workspace_elements..][0..self.columns];
            const products = scratch[self.workspace_elements + self.columns ..];
            for (0..input.shape.column_count) |column| {
                const source = input.column(column);
                for (packed_input, 0..) |*value, row| value.* = source.at(row);
                try synchronization.resolve(self.synchronization, operation, .{ .stage = .action_complete, .index = @intCast(column) }, self.apply_fn(self.context, operation, packed_input, products[column * self.rows ..][0..self.rows], scratch[0..self.workspace_elements]));
            }
            try synchronization.check(self.synchronization, operation, .{ .stage = .commit });
            for (0..output.shape.column_count) |column| {
                const destination = output.column(column);
                for (products[column * self.rows ..][0..self.rows], 0..) |value, row| {
                    const out = destination.at(row);
                    const scaled = scalar.mul(T, alpha, value);
                    out.* = if (scalar.isZero(T, beta)) scaled else scalar.add(T, scaled, scalar.mul(T, beta, out.*));
                }
            }
        }

        fn panelPreflight(self: Self, alpha: T, input: dense.ConstPanel(T), output: dense.Panel(T), workspace: []T) !usize {
            const input_length = try input.check();
            const output_length = try output.check();
            if (input.shape.row_count != self.columns or output.shape.row_count != self.rows or input.shape.column_count != output.shape.column_count) return error.DimensionMismatch;
            const required = try self.panelWorkspaceElements(alpha, input.shape.column_count);
            _ = try storage.byteLength(required, @sizeOf(T));
            if (workspace.len < required) return error.WorkspaceTooSmall;
            const scratch = workspace[0..required];
            const x = try range(input.values[0..input_length]);
            const y = try range(output.values[0..output_length]);
            const work = try range(scratch);
            if (storage.overlaps(x, y) or storage.overlaps(x, work) or storage.overlaps(y, work)) return error.AliasingNotAllowed;
            return required;
        }

        fn range(values: []const T) storage.Error!storage.ByteRange {
            return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(T)));
        }

        /// Borrow an existing admission without beginning, ending or retaining it.
        /// Preflight errors do not call the backend or modify any buffers.
        /// Only the required workspace prefix is borrowed; its tail is untouched.
        fn preflight(self: Self, input: []const T, output: []T, workspace: []T, required: usize) ![]T {
            if (input.len != self.columns or output.len != self.rows) return error.DimensionMismatch;
            _ = try storage.byteLength(required, @sizeOf(T));
            if (workspace.len < required) return error.WorkspaceTooSmall;
            const scratch = workspace[0..required];
            const x = try range(input);
            const y = try range(output);
            const work = try range(scratch);
            if (storage.overlaps(x, y) or storage.overlaps(x, work) or storage.overlaps(y, work)) return error.AliasingNotAllowed;
            return scratch;
        }

        /// Required scalar workspace for the selected action. CSR's fused
        /// backend uses no extra workspace. A generic callback needs a rows-sized
        /// result prefix only when it cannot directly overwrite the final output.
        pub fn scaledWorkspaceElements(self: Self, alpha: T, beta: T) storage.Error!usize {
            if (self.apply_scaled_fn != null or (scalar.isOne(T, alpha) and scalar.isZero(T, beta))) return self.workspace_elements;
            if (scalar.isZero(T, alpha)) return 0;
            return std.math.add(usize, self.workspace_elements, self.rows) catch error.SizeOverflow;
        }

        /// Uses the existing admission and explicit caller workspace. With no
        /// fused backend, scratch-backed callback failure preserves final output.
        /// Direct overwrite retains apply_fn output semantics on failure.
        /// alpha=0 never invokes an unfused callback; beta=0 never reads old y.
        pub fn applyScaledAdmitted(self: Self, operation: *const execution.context.Operation, alpha: T, input: []const T, beta: T, output: []T, workspace: []T) !void {
            _ = operation.borrowedContext();
            try synchronization.enter(T, self.synchronization, operation, .operator_scaled);
            const scratch = try synchronization.resolve(self.synchronization, operation, .{ .stage = .preflight }, self.scaledPreflight(alpha, input, beta, output, workspace));
            try synchronization.check(self.synchronization, operation, .{ .stage = .panel_backend, .control = @intFromBool(self.apply_scaled_fn != null) });
            try synchronization.scales(T, self.synchronization, operation, alpha, beta);
            if (self.apply_scaled_fn) |apply_scaled| {
                return synchronization.resolve(self.synchronization, operation, .{ .stage = .action_complete }, apply_scaled(self.context, operation, alpha, input, beta, output, scratch));
            }
            if (scalar.isZero(T, alpha)) {
                try synchronization.check(self.synchronization, operation, .{ .stage = .commit });
                if (scalar.isZero(T, beta)) @memset(output, scalar.zero(T)) else if (!scalar.isOne(T, beta)) for (output) |*value| {
                    value.* = scalar.mul(T, beta, value.*);
                };
                return;
            }
            if (scalar.isOne(T, alpha) and scalar.isZero(T, beta)) return synchronization.resolve(self.synchronization, operation, .{ .stage = .action_complete }, self.apply_fn(self.context, operation, input, output, scratch));
            const product = scratch[self.workspace_elements..];
            try synchronization.resolve(self.synchronization, operation, .{ .stage = .action_complete }, self.apply_fn(self.context, operation, input, product, scratch[0..self.workspace_elements]));
            try synchronization.check(self.synchronization, operation, .{ .stage = .commit });
            for (product, output) |value, *out| {
                const scaled = scalar.mul(T, alpha, value);
                out.* = if (scalar.isZero(T, beta)) scaled else scalar.add(T, scaled, scalar.mul(T, beta, out.*));
            }
        }

        fn scaledPreflight(self: Self, alpha: T, input: []const T, beta: T, output: []T, workspace: []T) ![]T {
            return self.preflight(input, output, workspace, try self.scaledWorkspaceElements(alpha, beta));
        }

        pub fn applyAdmitted(self: Self, operation: *const execution.context.Operation, input: []const T, output: []T, workspace: []T) !void {
            _ = operation.borrowedContext();
            try synchronization.enter(T, self.synchronization, operation, .operator_apply);
            const scratch = try synchronization.resolve(self.synchronization, operation, .{ .stage = .preflight }, self.preflight(input, output, workspace, self.workspace_elements));
            try synchronization.resolve(self.synchronization, operation, .{ .stage = .action_complete }, self.apply_fn(self.context, operation, input, output, scratch));
        }
    };
}
