// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal raw-pointer lifecycle, not C exports, a vendor status mapping or ABI.
//! Nonzero input prefixes must be alive, stable and actually accessible; their
//! allocation capacity cannot be discovered from C-like pointers. Null ends
//! selects major+1 offsets; nonnull ends must provide an explicit major prefix.
//! Arbitrary nonnull dummy-end discrimination remains outside this narrow domain.
const std = @import("std");

pub fn RawAdapter(comptime Provider: type, comptime Handles: type, comptime Typed: type) type {
    const Access = Typed.Access;
    const Storage = Provider.Storage;
    const Execution = Provider.Execution.context;
    comptime {
        if (Handles.OwnerIdentity != Provider.OwnerIdentity or Access.OwnerIdentity != Provider.OwnerIdentity or
            Typed.Handle != Handles.Handle or Handles.Context != Execution.Context or Handles.Operation != Execution.Operation)
            @compileError("raw Sparse entry must borrow the original typed/Core/Context owner");
    }
    return struct {
        pub const Handle = Handles.Handle;
        pub const Context = Execution.Context;
        pub const Operation = Execution.Operation;
        pub const Error = Typed.Error || error{ NullInput, InvalidAlignment };
        pub const Axis = Typed.Axis;
        pub const Pointer = Handles.RawPointer;

        // Semantic raw bits, not an extern C struct or a vendor status mapping.
        // General ignores mode/diag; diagonal ignores mode. Blocks/FULL are deferred.
        pub const MatrixDescriptor = struct { matrix_type: u32, mode: u32, diag: u32 };
        pub const MultiplyError = Error || Typed.MultiplyError || error{ InvalidOperation, InvalidDescriptor, UnsupportedDescriptor };
        pub fn ConstPointer(comptime T: type) type {
            return ?[*]align(1) const T;
        }

        // Exact required prefix, never inferred allocation capacity. Constness
        // follows the original capability, with no cast-away or fabricated
        // pointer. Even an unaligned nonnull empty prefix is never align-cast.
        fn vectorPrefix(comptime T: type, comptime writable: bool, raw: if (writable) Pointer(T) else ConstPointer(T), count: usize) MultiplyError!(if (writable) []T else []const T) {
            if (count == 0) return &.{};
            const pointer = raw orelse return if (writable) error.NullOutput else error.NullInput;
            if (@intFromPtr(pointer) % @alignOf(T) != 0) return error.InvalidAlignment;
            _ = try Storage.byteRange(@intFromPtr(pointer), try Storage.byteLength(count, @sizeOf(T)));
            const aligned: if (writable) [*]T else [*]const T = @alignCast(pointer);
            return aligned[0..count];
        }

        // This ordering is the adapter contract, not measured vendor priority.
        fn matrixSemantics(descriptor: MatrixDescriptor) MultiplyError!Typed.MatrixSemantics {
            switch (descriptor.matrix_type) {
                20 => return .general,
                21, 22, 23, 24 => {},
                25, 26 => return error.UnsupportedDescriptor,
                else => return error.InvalidDescriptor,
            }
            const triangle: Provider.Product.StoredTriangle = if (descriptor.matrix_type == 24) .lower else switch (descriptor.mode) {
                40 => .lower,
                41 => .upper,
                42 => return error.UnsupportedDescriptor,
                else => return error.InvalidDescriptor,
            };
            const diagonal: Provider.Product.Diagonal = switch (descriptor.diag) {
                50 => .stored,
                51 => .unit,
                else => return error.InvalidDescriptor,
            };
            return switch (descriptor.matrix_type) {
                21 => .{ .symmetric = .{ .triangle = triangle, .diagonal = diagonal } },
                22 => .{ .hermitian = .{ .triangle = triangle, .diagonal = diagonal } },
                23 => .{ .triangular = .{ .triangle = triangle, .diagonal = diagonal } },
                24 => .{ .diagonal = diagonal },
                else => unreachable,
            };
        }

        /// CSR/CSC general or square structured view; no vendor priority claim. Source
        /// admission precedes enum/descriptor and all borrowed-array checks.
        /// Required nonempty vectors remain required when alpha is zero. All
        /// metadata, capacity/byte/alias checks complete before Core writes y.
        pub fn matrixVectorMultiply(comptime T: type, comptime I: type, handle: ?Handle, operation_bits: u32, descriptor: MatrixDescriptor, alpha: T, input: ConstPointer(T), beta: T, output: Pointer(T)) MultiplyError!void {
            const h = handle orelse return error.NullHandle;
            const context = try Handles.borrowedContext(h);
            var operation = try context.begin();
            defer operation.end();
            const transform: Provider.Pattern.Operation = switch (operation_bits) {
                10 => .normal,
                11 => .transposed,
                12 => .adjoint,
                else => return error.InvalidOperation,
            };
            const semantics = try matrixSemantics(descriptor);
            const exported = try Typed.matrixAdmitted(T, I, &operation, h);
            const operand: Provider.Pattern.Operand(T, I, I) = .{ .pattern = exported.view.structure, .values = exported.view.values, .operation = transform };
            try Typed.checkProductShape(semantics, operand.pattern.row_count, operand.pattern.column_count);
            const x = try vectorPrefix(T, false, input, operand.effectiveColumnCount());
            const y = try vectorPrefix(T, true, output, operand.effectiveRowCount());
            try Typed.structuredProductAdmitted(T, I, &operation, h, exported, transform, semantics, alpha, x, beta, y);
        }
        // Keep lifecycle and MV errors/bodies unchanged. This union describes
        // MM admission/geometry faults; no numeric vendor status is returned.
        pub const MatrixMultiplyError = MultiplyError || Typed.MatrixMultiplyError || error{InvalidLayout};

        /// Single-layout CSR/CSC MM. Ordering is a chosen adapter contract, not
        /// measured MKL priority. Both panel geometries precede pointer prefixes;
        /// zero alpha never bypasses required metadata/capacity/alias checks.
        pub fn mm(comptime T: type, comptime I: type, handle: ?Handle, operation_bits: u32, descriptor: MatrixDescriptor, layout_bits: u32, alpha: T, input: ConstPointer(T), columns: I, ldb: I, beta: T, output: Pointer(T), ldc: I) MatrixMultiplyError!void {
            const h = handle orelse return error.NullHandle;
            const context = try Handles.borrowedContext(h);
            var operation = try context.begin();
            defer operation.end();
            const transform: Provider.Pattern.Operation = switch (operation_bits) {
                10 => .normal,
                11 => .transposed,
                12 => .adjoint,
                else => return error.InvalidOperation,
            };
            const semantics = try matrixSemantics(descriptor);
            const layout: Provider.Dense.Layout = switch (layout_bits) {
                101 => .row_major,
                102 => .column_major,
                else => return error.InvalidLayout,
            };
            const prepared = try Typed.prepareMmAdmitted(T, I, &operation, h, transform, semantics, layout, columns, ldb, ldc);
            const b = try vectorPrefix(T, false, input, prepared.input_extent);
            const c = try vectorPrefix(T, true, output, prepared.output_extent);
            try Typed.executeMmAdmitted(T, I, &operation, h, prepared, alpha, b, beta, c);
        }

        pub fn Slot(comptime T: type) type {
            return ?*align(1) T;
        }
        pub fn Input(comptime T: type, comptime I: type) type {
            comptime Handles.expectTypes(T, I);
            return struct {
                // Semantic raw bits, not a declaration of sparse_index_base_t.
                base: u32,
                rows: I,
                columns: I,
                axis: Axis,
                starts: Pointer(I),
                ends: Pointer(I),
                ordinals: Pointer(I),
                values: Pointer(T),
            };
        }
        pub fn Output(comptime T: type, comptime I: type) type {
            comptime Handles.expectTypes(T, I);
            return struct {
                base: Slot(u32),
                rows: Slot(I),
                columns: Slot(I),
                starts: Slot(Pointer(I)),
                ends: Slot(Pointer(I)),
                ordinals: Slot(Pointer(I)),
                values: Slot(Pointer(T)),
            };
        }
        fn slot(comptime T: type, raw: Slot(T)) Error!*T {
            const pointer = raw orelse return error.NullOutput;
            if (@intFromPtr(pointer) % @alignOf(T) != 0) return error.InvalidAlignment;
            _ = try Storage.byteRange(@intFromPtr(pointer), @sizeOf(T));
            return @alignCast(pointer);
        }
        /// The count is the exact required prefix, never an invented capacity.
        /// Empty pointer values are preserved separately, never aligned or read.
        fn prefix(comptime T: type, raw: Pointer(T), count: usize) Error![]T {
            if (count == 0) return &.{};
            const pointer = raw orelse return error.NullInput;
            if (@intFromPtr(pointer) % @alignOf(T) != 0) return error.InvalidAlignment;
            _ = try Storage.byteRange(@intFromPtr(pointer), try Storage.byteLength(count, @sizeOf(T)));
            const aligned: [*]T = @alignCast(pointer);
            return aligned[0..count];
        }
        fn emptySource(comptime T: type, raw: Pointer(T), count: usize) Handles.EmptySource(T) {
            return if (count == 0) .{ .empty_original = raw } else .not_captured;
        }

        /// One admission precedes scalar/pointer validation and covers actual
        /// destination commit and both metadata allocations/rollback. Never read
        /// the old output value. The explicit Context is the existing owner type.
        pub fn createBorrowed(comptime T: type, comptime I: type, context: *Context, input: Input(T, I), output: Slot(?Handle)) Error!void {
            var operation = try context.begin();
            defer operation.end();
            const out = try slot(?Handle, output);
            const base: i32 = switch (input.base) {
                0 => 0,
                1 => 1,
                else => return error.InvalidBase,
            };
            const geometry = try Access.shape(I, base, input.rows, input.columns, input.axis);
            const null_end = input.ends == null;
            var starts: []I = undefined;
            var ends: []I = undefined;
            if (null_end and geometry.major != 0) {
                const count = std.math.add(usize, geometry.major, 1) catch return error.SizeOverflow;
                const offsets = try prefix(I, input.starts, count);
                starts = offsets[0..geometry.major];
                ends = offsets[1..count];
            } else {
                starts = try prefix(I, input.starts, geometry.major);
                ends = try prefix(I, input.ends, geometry.major);
            }
            const ranges = try Access.captureRanges(I, geometry, starts, ends);
            const ordinals = try prefix(I, input.ordinals, ranges.extent);
            const values = try prefix(T, input.values, ranges.extent);
            const data = try Access.captureData(T, I, ranges, ordinals, values);
            const provenance: Handles.Provenance(T, I) = .{
                .encoding = if (null_end) .c_null_end else .c_split,
                .starts = emptySource(I, input.starts, geometry.major),
                .ends = if (null_end) .not_captured else emptySource(I, input.ends, geometry.major),
                .ordinals = emptySource(I, input.ordinals, ranges.extent),
                .values = emptySource(T, input.values, ranges.extent),
            };
            const handle = try Typed.borrowAdmitted(T, I, &operation, geometry, input.axis, data, try Access.outputRange(?Handle, out), provenance);
            out.* = handle;
        }

        /// Seven slots are independent. No caller output is touched until every
        /// type/format/structure/provenance/alias/cast check has succeeded. Plain
        /// final stores are not an atomic transaction or a concurrency protocol.
        pub fn exportBorrowed(comptime T: type, comptime I: type, handle: ?Handle, axis: Axis, output: Output(T, I)) Error!void {
            const h = handle orelse return error.NullHandle;
            const context = try Handles.borrowedContext(h);
            var operation = try context.begin();
            defer operation.end();
            const base_out = try slot(u32, output.base);
            const rows_out = try slot(I, output.rows);
            const columns_out = try slot(I, output.columns);
            const starts_out = try slot(Pointer(I), output.starts);
            const ends_out = try slot(Pointer(I), output.ends);
            const ordinals_out = try slot(Pointer(I), output.ordinals);
            const values_out = try slot(Pointer(T), output.values);
            const exported = try Typed.exportAdmitted(T, I, &operation, h, axis);
            const spans = [_]Storage.ByteRange{
                try Access.outputRange(u32, base_out),          try Access.outputRange(I, rows_out),          try Access.outputRange(I, columns_out),
                try Access.outputRange(Pointer(I), starts_out), try Access.outputRange(Pointer(I), ends_out), try Access.outputRange(Pointer(I), ordinals_out),
                try Access.outputRange(Pointer(T), values_out),
            };
            for (spans, 0..) |span, index| {
                for (spans[index + 1 ..]) |other| if (Storage.overlaps(span, other)) return error.BufferOverlap;
                try Handles.rejectMetadataOverlap(h, span);
                try Access.rejectDataOverlap(T, I, span, exported.mutable);
            }
            const rows = std.math.cast(I, exported.view.structure.row_count) orelse return error.SizeOverflow;
            const columns = std.math.cast(I, exported.view.structure.column_count) orelse return error.SizeOverflow;
            const pointers = try Handles.rawPointers(T, I, exported);
            base_out.* = if (exported.view.structure.index_base == .zero) 0 else 1;
            rows_out.* = rows;
            columns_out.* = columns;
            starts_out.* = pointers.starts;
            ends_out.* = pointers.ends;
            ordinals_out.* = pointers.ordinals;
            values_out.* = pointers.values;
        }

        /// Original body and captured-allocator/pin release order are shared.
        /// Closing must never require a fresh admission merely to destroy.
        pub const destroy = Handles.destroy;
    };
}
