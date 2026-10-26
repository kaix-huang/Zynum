// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const operator = @import("../numerics/operator.zig");
const execution = @import("zynum-execution");
const operations = @import("operations.zig");
const dense = @import("../numerics/dense.zig");

/// Borrow a native compressed operand and interpretation, including independent
/// index widths, CSR/CSC, zero/one bases and Hermitian half storage. No conversion,
/// control allocation, validation cache or new Context is introduced. Every
/// callback revalidates metadata, shape and output aliases before shared kernels.
pub fn PatternBackend(comptime T: type, comptime Offset: type, comptime Index: type) type {
    const pattern = @import("pattern.zig");
    const product = @import("core/product.zig");
    const storage = @import("../numerics/storage.zig");
    return struct {
        const Self = @This();
        operand: pattern.Operand(T, Offset, Index),
        semantics: product.MatrixSemantics = .general,

        pub fn borrow(self: *const Self) operator.Operator(T) {
            return .{ .context = self, .rows = self.operand.effectiveRowCount(), .columns = self.operand.effectiveColumnCount(), .apply_fn = apply, .apply_scaled_fn = applyScaled, .apply_panel_fn = applyPanel };
        }
        fn range(values: anytype) !storage.ByteRange {
            return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(@typeInfo(@TypeOf(values)).pointer.child)));
        }
        fn separate(a: storage.ByteRange, values: anytype) !void {
            if (storage.overlaps(a, try range(values))) return error.AliasingNotAllowed;
        }
        fn preflight(self: *const Self, input: []const T, output: []T) !void {
            _ = try self.operand.check();
            if (!product.acceptsShape(self.semantics, self.operand.pattern.row_count, self.operand.pattern.column_count)) return error.DimensionMismatch;
            const out = try range(output);
            try separate(out, input);
            try separate(out, self.operand.values);
            try separate(out, self.operand.pattern.ordinals);
            switch (self.operand.pattern.ranges) {
                .contiguous => |values| try separate(out, values),
                .split => |values| {
                    try separate(out, values.starts);
                    try separate(out, values.ends);
                },
            }
        }
        fn apply(raw: *const anyopaque, op: *const execution.context.Operation, x: []const T, y: []T, work: []T) !void {
            return applyScaled(raw, op, @import("scalar.zig").one(T), x, @import("scalar.zig").zero(T), y, work);
        }
        fn applyScaled(raw: *const anyopaque, op: *const execution.context.Operation, alpha: T, x: []const T, beta: T, y: []T, _: []T) !void {
            _ = op.borrowedContext();
            const self: *const Self = @ptrCast(@alignCast(raw));
            if (x.len != self.operand.effectiveColumnCount() or y.len != self.operand.effectiveRowCount()) return error.DimensionMismatch;
            try self.preflight(x, y);
            product.executeStructuredValidated(T, Offset, Index, self.operand, self.semantics, x, y, alpha, beta);
        }
        fn applyPanel(raw: *const anyopaque, op: *const execution.context.Operation, alpha: T, input: dense.ConstPanel(T), beta: T, output: dense.Panel(T), _: []T) !void {
            _ = op.borrowedContext();
            const self: *const Self = @ptrCast(@alignCast(raw));
            const x = input.values[0..try input.check()];
            const y = output.values[0..try output.check()];
            if (input.shape.row_count != self.operand.effectiveColumnCount() or output.shape.row_count != self.operand.effectiveRowCount() or input.shape.column_count != output.shape.column_count) return error.DimensionMismatch;
            try self.preflight(x, y);
            product.executePanelValidated(T, Offset, Index, self.operand, self.semantics, input, output, alpha, beta);
        }
    };
}

/// Borrow a stable CSR view object and its buffers. Recreate this adapter after
/// changing shape/transform; each action revalidates the current CSR metadata.
/// The generic operator adds no work buffer and uses the shared Sparse kernel.
pub fn CsrBackend(comptime Matrix: type) type {
    const T = Matrix.Scalar;
    return struct {
        pub fn borrow(matrix: *const Matrix) operator.Operator(T) {
            return .{ .context = matrix, .rows = matrix.effectiveRowCount(), .columns = matrix.effectiveColumnCount(), .apply_fn = apply, .apply_scaled_fn = applyScaled, .apply_panel_fn = applyPanel };
        }
        fn applyPanel(raw: *const anyopaque, operation: *const execution.context.Operation, alpha: T, input: dense.ConstPanel(T), beta: T, output: dense.Panel(T), _: []T) !void {
            const matrix: *const Matrix = @ptrCast(@alignCast(raw));
            try operations.matrixMultiplyAdmitted(operation, .{ .left_matrix = matrix.*, .right_matrix = input, .result_matrix = output, .product_scale = alpha, .result_scale = beta });
        }
        fn applyScaled(raw: *const anyopaque, operation: *const execution.context.Operation, alpha: T, input: []const T, beta: T, output: []T, _: []T) !void {
            const matrix: *const Matrix = @ptrCast(@alignCast(raw));
            try operations.matrixVectorMultiplyAdmitted(operation, .{ .matrix = matrix.*, .input_vector = input, .result_vector = output, .product_scale = alpha, .result_scale = beta });
        }
        fn apply(raw: *const anyopaque, operation: *const execution.context.Operation, input: []const T, output: []T, _: []T) !void {
            const matrix: *const Matrix = @ptrCast(@alignCast(raw));
            try operations.matrixVectorMultiplyAdmitted(operation, .{ .matrix = matrix.*, .input_vector = input, .result_vector = output });
        }
    };
}

/// Inverse-triangular action for the unified operator (including preconditioners).
/// The state and all CSR arrays are borrowed; only caller workspace is used.
/// Output commits after a successful solve. No Context, handle or allocator is created.
pub fn TriangularInverseBackend(comptime Matrix: type) type {
    const std = @import("std");
    const storage = @import("../numerics/storage.zig");
    const triangular = @import("triangular.zig");
    const views = @import("views.zig");
    const T = Matrix.Scalar;
    return struct {
        const Self = @This();
        matrix: Matrix,
        triangle: triangular.Triangle,
        diagonal: triangular.Diagonal,

        pub fn workspaceElements(self: *const Self) !usize {
            try self.matrix.check();
            if (self.matrix.row_count != self.matrix.column_count) return error.DimensionMismatch;
            const work = try triangular.triangularSolveWorkspaceLength(T, self.matrix.row_count, 1, self.diagonal);
            const count = std.math.add(usize, self.matrix.row_count, work) catch return error.SizeOverflow;
            _ = try storage.byteLength(count, @sizeOf(T));
            return count;
        }
        pub fn borrow(self: *const Self) !operator.Operator(T) {
            return .{ .context = self, .rows = self.matrix.row_count, .columns = self.matrix.column_count, .workspace_elements = try self.workspaceElements(), .apply_fn = apply };
        }
        fn range(comptime V: type, values: []const V) !storage.ByteRange {
            return storage.byteRange(@intFromPtr(values.ptr), try storage.byteLength(values.len, @sizeOf(V)));
        }
        fn apply(raw: *const anyopaque, op: *const execution.context.Operation, input: []const T, output: []T, workspace: []T) !void {
            const self: *const Self = @ptrCast(@alignCast(raw));
            const count = try self.workspaceElements();
            const n = self.matrix.row_count;
            if (input.len != n or output.len != n) return error.DimensionMismatch;
            if (workspace.len < count) return error.WorkspaceTooSmall;
            const scratch = workspace[0..count];
            const work_range = try range(T, scratch);
            const out_range = try range(T, output);
            const O = if (@hasDecl(Matrix, "Offset")) Matrix.Offset else Matrix.Index;
            const matrix_ranges = [_]storage.ByteRange{ try range(T, self.matrix.values), try range(O, self.matrix.row_offsets), try range(Matrix.Index, self.matrix.column_indices) };
            for (matrix_ranges) |r| if (storage.overlaps(r, work_range) or storage.overlaps(r, out_range)) return error.AliasingNotAllowed;
            @memcpy(scratch[0..n], input);
            const result = try triangular.triangularSolveInPlaceAdmitted(op, .{ .matrix = self.matrix, .triangle = self.triangle, .diagonal = self.diagonal, .rhs = views.Matrix(T){ .values = scratch[0..n], .row_count = n, .column_count = 1, .leading_dimension = n }, .workspace = scratch[n..] });
            if (result.info != 0) return error.TriangularSolveFailed;
            @memcpy(output, scratch[0..n]);
        }
    };
}
