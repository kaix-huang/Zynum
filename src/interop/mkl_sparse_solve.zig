// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Internal checked handle solve staging. No installed ABI/status or vendor IEEE claim.
//! Uses the existing handle Context, operation and ContextScratch; output commits only on success.
const std = @import("std");
pub fn Adapter(comptime Provider: type, comptime Handles: type, comptime Typed: type) type {
    const Core = Provider.OwnerIdentity;
    const S = Core.sparse;
    const E = Provider.Execution;
    const P = Provider.Pattern;
    comptime {
        if (Handles.OwnerIdentity != Core or Typed.Handle != Handles.Handle) @compileError("solve must share handle and Core owner");
    }
    return struct {
        pub const Contract = Core.internal.sparse_checked.TriangularSolveContract;
        pub const Triangle = S.Triangle;
        pub const Diagonal = S.Diagonal;
        pub const Result = S.TriangularSolveResult;
        pub const Error = Typed.Error || P.Error || Provider.Dense.Error || S.Error || E.context.AdmissionError || error{ OutOfMemory, NonFiniteScaling };
        fn finite(comptime T: type, v: T) bool {
            return if (comptime Provider.Scalar.isComplex(T)) std.math.isFinite(v.re) and std.math.isFinite(v.im) else std.math.isFinite(v);
        }
        fn offset(shape: Provider.Dense.PanelShape, row: usize, col: usize) usize {
            return if (shape.layout == .row_major) row * shape.leading_dimension + col else col * shape.leading_dimension + row;
        }
        pub fn Prepared(comptime T: type, comptime I: type) type {
            return struct { diagonal_only: bool = false, exported: Handles.BorrowedExport(T, I), transform: P.Operation, triangle: Triangle, diagonal: Diagonal, input_shape: Provider.Dense.PanelShape, output_shape: Provider.Dense.PanelShape, input_extent: usize, output_extent: usize };
        }
        /// Same-call borrow, not a reusable validation certificate. This neutral
        /// typed stage accepts both panel layouts; external profile restrictions
        /// and matrix-descriptor decoding belong to the later interface layer.
        pub fn prepareAdmitted(comptime T: type, comptime I: type, op: *const Handles.Operation, handle: Handles.Handle, transform: P.Operation, triangle: Triangle, diagonal: Diagonal, layout: Provider.Dense.Layout, columns: I, ldx: I, ldy: I) Error!Prepared(T, I) {
            if (try Handles.borrowedContext(handle) != op.borrowedContext()) return error.ContextMismatch;
            if (columns < 0) return error.InvalidDimension;
            if (ldx < 0 or ldy < 0) return error.InvalidLeadingDimension;
            const exported = try Typed.matrixAdmitted(T, I, op, handle);
            const pattern = exported.view.structure;
            if (pattern.row_count != pattern.column_count) return error.DimensionMismatch;
            const xs: Provider.Dense.PanelShape = .{ .row_count = pattern.row_count, .column_count = std.math.cast(usize, columns) orelse return error.SizeOverflow, .leading_dimension = std.math.cast(usize, ldx) orelse return error.SizeOverflow, .layout = layout };
            const ys: Provider.Dense.PanelShape = .{ .row_count = pattern.row_count, .column_count = xs.column_count, .leading_dimension = std.math.cast(usize, ldy) orelse return error.SizeOverflow, .layout = layout };
            return .{ .exported = exported, .transform = transform, .triangle = triangle, .diagonal = diagonal, .input_shape = xs, .output_shape = ys, .input_extent = try xs.extent(), .output_extent = try ys.extent() };
        }
        pub fn executeAdmitted(comptime T: type, comptime I: type, op: *const Handles.Operation, handle: Handles.Handle, prepared: Prepared(T, I), alpha: T, input: []const T, output: []T) Error!Result {
            return executeWithContractAdmitted(T, I, op, handle, prepared, alpha, input, output, .checked);
        }
        pub fn executeWithContractAdmitted(comptime T: type, comptime I: type, op: *const Handles.Operation, handle: Handles.Handle, prepared: Prepared(T, I), alpha: T, input: []const T, output: []T, comptime contract: Contract) Error!Result {
            if (try Handles.borrowedContext(handle) != op.borrowedContext()) return error.ContextMismatch;
            if (input.len < prepared.input_extent or output.len < prepared.output_extent) return error.BufferTooSmall;
            const x = input[0..prepared.input_extent];
            const y = output[0..prepared.output_extent];
            const destination = try Typed.Access.range(T, y);
            try Handles.rejectMetadataOverlap(handle, destination);
            try Typed.Access.rejectDataOverlap(T, I, destination, prepared.exported.mutable);
            if (contract == .checked and !finite(T, alpha)) return error.NonFiniteInput;
            const pattern = prepared.exported.view.structure;
            const n = pattern.row_count;
            const nrhs = prepared.input_shape.column_count;
            if (n == 0 or nrhs == 0) return .{ .completed_rhs = nrhs };
            if (contract == .sparse_ieee_v1 and Provider.Scalar.isZero(T, alpha)) {
                for (0..nrhs) |j| for (0..n) |i| {
                    y[offset(prepared.output_shape, i, j)] = Provider.Scalar.zero(T);
                };
                return .{ .completed_rhs = nrhs };
            }
            const stats = try pattern.check();
            const count = std.math.mul(usize, n, nrhs) catch return error.SizeOverflow;
            const offset_count = std.math.add(usize, n, 1) catch return error.SizeOverflow;
            const work_count = try S.triangularSolveWorkspaceLength(T, n, nrhs, prepared.diagonal);
            _ = try Provider.Storage.byteLength(count, @sizeOf(T));
            _ = try Provider.Storage.byteLength(offset_count, @sizeOf(u64));
            _ = try Provider.Storage.byteLength(stats.referenced_entries, @sizeOf(I));
            _ = try Provider.Storage.byteLength(stats.referenced_entries, @sizeOf(T));
            _ = try Provider.Storage.byteLength(work_count, @sizeOf(T));
            var offsets = try E.context.ContextScratch(u64).acquire(op, offset_count);
            defer offsets.deinit();
            var ordinals = try E.context.ContextScratch(I).acquire(op, stats.referenced_entries);
            defer ordinals.deinit();
            var values = try E.context.ContextScratch(T).acquire(op, stats.referenced_entries);
            defer values.deinit();
            var rhs = try E.context.ContextScratch(T).acquire(op, count);
            defer rhs.deinit();
            var work = try E.context.ContextScratch(T).acquire(op, work_count);
            defer work.deinit();
            var cursor: usize = 0;
            offsets.lease.values[0] = 0;
            // Normalize base, split/gapped intervals and CSR/CSC into one same-call
            // CSR representation. No source ownership, ordering or cache is changed.
            for (0..n) |row| {
                for (0..pattern.majorCount()) |major| {
                    if (pattern.major_axis == .row and major != row) continue;
                    const range = try pattern.checkedMajorRange(major);
                    for (range.begin..range.end) |entry| {
                        const minor = try pattern.checkedOrdinalAt(entry);
                        if (pattern.major_axis == .column and minor != row) continue;
                        const column = if (pattern.major_axis == .row) minor else major;
                        if (prepared.diagonal_only and column != row) continue;
                        ordinals.lease.values[cursor] = std.math.cast(I, column) orelse return error.SizeOverflow;
                        values.lease.values[cursor] = prepared.exported.view.values[entry];
                        cursor += 1;
                    }
                }
                offsets.lease.values[row + 1] = std.math.cast(u64, cursor) orelse return error.SizeOverflow;
            }
            std.debug.assert(if (prepared.diagonal_only) cursor <= stats.referenced_entries else cursor == stats.referenced_entries);
            for (0..nrhs) |j| for (0..n) |i| {
                const source = x[offset(prepared.input_shape, i, j)];
                if (contract == .checked and !finite(T, source)) return error.NonFiniteInput;
                const scaled = if (contract == .sparse_ieee_v1 and Provider.Scalar.isOne(T, alpha)) source else Provider.Scalar.multiply(T, alpha, source);
                if (contract == .checked and !finite(T, scaled)) return error.NonFiniteScaling;
                rhs.lease.values[i + j * n] = scaled;
            };
            const a: S.ConstCsrMatrixWithIndexTypes(T, u64, I) = .{ .values = values.lease.values[0..cursor], .row_offsets = offsets.lease.values, .column_indices = ordinals.lease.values[0..cursor], .row_count = n, .column_count = n, .operation = switch (prepared.transform) {
                .normal => .normal,
                .transposed => .transposed,
                .adjoint => .adjoint,
            } };
            const result = try Core.internal.sparse_checked.triangularSolveWithContractAdmitted(op, .{ .matrix = a, .triangle = prepared.triangle, .diagonal = prepared.diagonal, .rhs = S.Matrix(T){ .values = rhs.lease.values, .row_count = n, .column_count = nrhs, .leading_dimension = n }, .workspace = work.lease.values }, contract);
            if (result.info != 0) return result;
            for (0..nrhs) |j| for (0..n) |i| {
                y[offset(prepared.output_shape, i, j)] = rhs.lease.values[i + j * n];
            };
            return result;
        }
        pub fn panel(comptime T: type, comptime I: type, handle: ?Handles.Handle, transform: P.Operation, triangle: Triangle, diagonal: Diagonal, layout: Provider.Dense.Layout, alpha: T, input: []const T, columns: I, ldx: I, output: []T, ldy: I) Error!Result {
            const h = handle orelse return error.NullHandle;
            const ctx = try Handles.borrowedContext(h);
            var op = try ctx.begin();
            defer op.end();
            const prepared = try prepareAdmitted(T, I, &op, h, transform, triangle, diagonal, layout, columns, ldx, ldy);
            return executeAdmitted(T, I, &op, h, prepared, alpha, input, output);
        }
        pub fn panelWithContract(comptime T: type, comptime I: type, handle: ?Handles.Handle, transform: P.Operation, triangle: Triangle, diagonal: Diagonal, layout: Provider.Dense.Layout, alpha: T, input: []const T, columns: I, ldx: I, output: []T, ldy: I, comptime contract: Contract) Error!Result {
            const h = handle orelse return error.NullHandle;
            const ctx = try Handles.borrowedContext(h);
            var op = try ctx.begin();
            defer op.end();
            const prepared = try prepareAdmitted(T, I, &op, h, transform, triangle, diagonal, layout, columns, ldx, ldy);
            return executeWithContractAdmitted(T, I, &op, h, prepared, alpha, input, output, contract);
        }
    };
}
