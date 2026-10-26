// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Raw checked solve protocol. Not an installed ABI or numeric vendor status.
//! Descriptor/leading-dimension domains awaiting review remain explicit faults.
const std = @import("std");
pub fn Adapter(comptime Provider: type, comptime Handles: type, comptime Typed: type, comptime Solve: type, comptime Raw: type) type {
    comptime {
        if (Handles.OwnerIdentity != Provider.OwnerIdentity or Typed.Handle != Handles.Handle or Raw.Handle != Handles.Handle) @compileError("raw solve requires the existing handle/Core owner");
    }
    return struct {
        pub const Error = Solve.Error || Typed.MatrixMultiplyError || Raw.Error || error{ NullInput, InvalidAlignment, InvalidOperation, InvalidDescriptor, InvalidLayout, UnreviewedSolveDescriptor };
        pub const Descriptor = Raw.MatrixDescriptor;
        const Selection = struct { triangle: Solve.Triangle, diagonal: Solve.Diagonal, diagonal_only: bool };
        fn selection(d: Descriptor) Error!Selection {
            switch (d.matrix_type) {
                23, 24 => {},
                20, 21, 22, 25, 26 => return error.UnreviewedSolveDescriptor,
                else => return error.InvalidDescriptor,
            }
            const triangle: Solve.Triangle = if (d.matrix_type == 24) .lower else switch (d.mode) {
                40 => .lower,
                41 => .upper,
                42 => return error.UnreviewedSolveDescriptor,
                else => return error.InvalidDescriptor,
            };
            const diagonal: Solve.Diagonal = switch (d.diag) {
                50 => .non_unit,
                51 => .unit,
                else => return error.InvalidDescriptor,
            };
            return .{ .triangle = triangle, .diagonal = diagonal, .diagonal_only = d.matrix_type == 24 };
        }
        fn transform(bits: u32) Error!Provider.Pattern.Operation {
            return switch (bits) {
                10 => .normal,
                11 => .transposed,
                12 => .adjoint,
                else => error.InvalidOperation,
            };
        }
        fn prefix(comptime T: type, comptime writable: bool, raw: if (writable) Handles.RawPointer(T) else Raw.ConstPointer(T), count: usize) Error!(if (writable) []T else []const T) {
            if (count == 0) return &.{};
            const ptr = raw orelse return if (writable) error.NullOutput else error.NullInput;
            if (@intFromPtr(ptr) % @alignOf(T) != 0) return error.InvalidAlignment;
            _ = try Provider.Storage.byteRange(@intFromPtr(ptr), try Provider.Storage.byteLength(count, @sizeOf(T)));
            const aligned: if (writable) [*]T else [*]const T = @alignCast(ptr);
            return aligned[0..count];
        }
        pub fn panelAdmitted(comptime T: type, comptime I: type, op: *const Handles.Operation, h: Handles.Handle, operation_bits: u32, descriptor: Descriptor, layout_bits: u32, alpha: T, x: Raw.ConstPointer(T), columns: I, ldx: I, y: Handles.RawPointer(T), ldy: I) Error!Solve.Result {
            if (try Handles.borrowedContext(h) != op.borrowedContext()) return error.ContextMismatch;
            const operation = try transform(operation_bits);
            const selected = try selection(descriptor);
            const layout: Provider.Dense.Layout = switch (layout_bits) {
                101 => .row_major,
                102 => .column_major,
                else => return error.InvalidLayout,
            };
            var prepared = try Solve.prepareAdmitted(T, I, op, h, operation, selected.triangle, selected.diagonal, layout, columns, ldx, ldy);
            prepared.diagonal_only = selected.diagonal_only;
            const pattern = prepared.exported.view.structure;
            const semantics: Typed.MatrixSemantics = if (selected.diagonal_only) .{ .diagonal = if (selected.diagonal == .unit) .unit else .stored } else .{ .triangular = .{ .triangle = if (selected.triangle == .lower) .lower else .upper, .diagonal = if (selected.diagonal == .unit) .unit else .stored } };
            try Typed.checkMmConfiguration(pattern.index_base, pattern.major_axis, operation, semantics, layout);
            const input = try prefix(T, false, x, prepared.input_extent);
            const output = try prefix(T, true, y, prepared.output_extent);
            return Solve.executeAdmitted(T, I, op, h, prepared, alpha, input, output);
        }
        pub fn panelAdmittedWithContract(comptime T: type, comptime I: type, op: *const Handles.Operation, h: Handles.Handle, operation_bits: u32, descriptor: Descriptor, layout_bits: u32, alpha: T, x: Raw.ConstPointer(T), columns: I, ldx: I, y: Handles.RawPointer(T), ldy: I, comptime contract: Solve.Contract) Error!Solve.Result {
            if (try Handles.borrowedContext(h) != op.borrowedContext()) return error.ContextMismatch;
            const operation = try transform(operation_bits);
            const selected = try selection(descriptor);
            const layout: Provider.Dense.Layout = switch (layout_bits) {
                101 => .row_major,
                102 => .column_major,
                else => return error.InvalidLayout,
            };
            var prepared = try Solve.prepareAdmitted(T, I, op, h, operation, selected.triangle, selected.diagonal, layout, columns, ldx, ldy);
            prepared.diagonal_only = selected.diagonal_only;
            const pattern = prepared.exported.view.structure;
            const semantics: Typed.MatrixSemantics = if (selected.diagonal_only) .{ .diagonal = if (selected.diagonal == .unit) .unit else .stored } else .{ .triangular = .{ .triangle = if (selected.triangle == .lower) .lower else .upper, .diagonal = if (selected.diagonal == .unit) .unit else .stored } };
            try Typed.checkMmConfiguration(pattern.index_base, pattern.major_axis, operation, semantics, layout);
            const input = try prefix(T, false, x, prepared.input_extent);
            const output = try prefix(T, true, y, prepared.output_extent);
            return Solve.executeWithContractAdmitted(T, I, op, h, prepared, alpha, input, output, contract);
        }
        pub fn panel(comptime T: type, comptime I: type, handle: ?Handles.Handle, operation_bits: u32, descriptor: Descriptor, layout_bits: u32, alpha: T, x: Raw.ConstPointer(T), columns: I, ldx: I, y: Handles.RawPointer(T), ldy: I) Error!Solve.Result {
            const h = handle orelse return error.NullHandle;
            const ctx = try Handles.borrowedContext(h);
            var op = try ctx.begin();
            defer op.end();
            return panelAdmitted(T, I, &op, h, operation_bits, descriptor, layout_bits, alpha, x, columns, ldx, y, ldy);
        }
        pub fn panelWithContract(comptime T: type, comptime I: type, handle: ?Handles.Handle, operation_bits: u32, descriptor: Descriptor, layout_bits: u32, alpha: T, x: Raw.ConstPointer(T), columns: I, ldx: I, y: Handles.RawPointer(T), ldy: I, comptime contract: Solve.Contract) Error!Solve.Result {
            const h = handle orelse return error.NullHandle;
            const ctx = try Handles.borrowedContext(h);
            var op = try ctx.begin();
            defer op.end();
            return panelAdmittedWithContract(T, I, &op, h, operation_bits, descriptor, layout_bits, alpha, x, columns, ldx, y, ldy, contract);
        }
        pub fn vectorAdmitted(comptime T: type, comptime I: type, op: *const Handles.Operation, h: Handles.Handle, operation_bits: u32, descriptor: Descriptor, alpha: T, x: Raw.ConstPointer(T), y: Handles.RawPointer(T)) Error!Solve.Result {
            if (try Handles.borrowedContext(h) != op.borrowedContext()) return error.ContextMismatch;
            const operation = try transform(operation_bits);
            const selected = try selection(descriptor);
            const borrowed = try Typed.matrixAdmitted(T, I, op, h);
            const n = borrowed.view.structure.row_count;
            const ld = std.math.cast(I, n) orelse return error.SizeOverflow;
            // Vector API has no dense-layout argument or MM base/layout filter.
            var prepared = try Solve.prepareAdmitted(T, I, op, h, operation, selected.triangle, selected.diagonal, .column_major, 1, ld, ld);
            prepared.diagonal_only = selected.diagonal_only;
            const input = try prefix(T, false, x, prepared.input_extent);
            const output = try prefix(T, true, y, prepared.output_extent);
            return Solve.executeAdmitted(T, I, op, h, prepared, alpha, input, output);
        }
        pub fn vectorAdmittedWithContract(comptime T: type, comptime I: type, op: *const Handles.Operation, h: Handles.Handle, operation_bits: u32, descriptor: Descriptor, alpha: T, x: Raw.ConstPointer(T), y: Handles.RawPointer(T), comptime contract: Solve.Contract) Error!Solve.Result {
            if (try Handles.borrowedContext(h) != op.borrowedContext()) return error.ContextMismatch;
            const operation = try transform(operation_bits);
            const selected = try selection(descriptor);
            const borrowed = try Typed.matrixAdmitted(T, I, op, h);
            const n = borrowed.view.structure.row_count;
            const ld = std.math.cast(I, n) orelse return error.SizeOverflow;
            // Vector API has no dense-layout argument or MM base/layout filter.
            var prepared = try Solve.prepareAdmitted(T, I, op, h, operation, selected.triangle, selected.diagonal, .column_major, 1, ld, ld);
            prepared.diagonal_only = selected.diagonal_only;
            const input = try prefix(T, false, x, prepared.input_extent);
            const output = try prefix(T, true, y, prepared.output_extent);
            return Solve.executeWithContractAdmitted(T, I, op, h, prepared, alpha, input, output, contract);
        }
        pub fn vector(comptime T: type, comptime I: type, handle: ?Handles.Handle, operation_bits: u32, descriptor: Descriptor, alpha: T, x: Raw.ConstPointer(T), y: Handles.RawPointer(T)) Error!Solve.Result {
            const h = handle orelse return error.NullHandle;
            const ctx = try Handles.borrowedContext(h);
            var op = try ctx.begin();
            defer op.end();
            return vectorAdmitted(T, I, &op, h, operation_bits, descriptor, alpha, x, y);
        }
        pub fn vectorWithContract(comptime T: type, comptime I: type, handle: ?Handles.Handle, operation_bits: u32, descriptor: Descriptor, alpha: T, x: Raw.ConstPointer(T), y: Handles.RawPointer(T), comptime contract: Solve.Contract) Error!Solve.Result {
            const h = handle orelse return error.NullHandle;
            const ctx = try Handles.borrowedContext(h);
            var op = try ctx.begin();
            defer op.end();
            return vectorAdmittedWithContract(T, I, &op, h, operation_bits, descriptor, alpha, x, y, contract);
        }
    };
}
