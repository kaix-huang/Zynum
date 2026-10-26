// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal typed lifecycle slice motivated by oneMKL; no C ABI/status mapping.
//! Truthful mutable slices make capacity/alias faults explicit and separate from
//! admission and allocation faults. They cannot prove raw-pointer validity.
//! Supported representation is split starts/ends (including offsets/offsets+1),
//! monotone nonoverlapping intervals with physical gaps and stored duplicates.
//! Broader internal Pattern intervals and documented null/dummy-end forms are
//! outside this slice, not declarations that MKL defines them as invalid.
//! Outputs commit once on success. Typed record output is not vendor wire ABI.

const std = @import("std");

pub fn TypedAdapter(comptime Provider: type, comptime Handles: type) type {
    const P = Provider.Pattern;
    const Storage = Provider.Storage;
    const Execution = Provider.Execution.context;
    comptime {
        if (Handles.OwnerIdentity != Provider.OwnerIdentity or Handles.Context != Execution.Context or Handles.Operation != Execution.Operation)
            @compileError("Sparse adapter must use the binding's single shell/Core/Context owner");
    }
    return struct {
        pub const Handle = Handles.Handle;
        pub const Access = @import("mkl_sparse_access.zig").Access(Provider, Handles);
        pub const Error = Access.Error;
        pub const Axis = P.MajorAxis;

        // Separate from the unchanged lifecycle error set.
        pub const MultiplyError = Error || error{DimensionMismatch};
        pub const MatrixSemantics = Provider.Product.MatrixSemantics;

        /// Same-call matrix borrow for general multiplication, with no cached
        /// validation. WrongFormat alone selects CSC; common export checks that
        /// format before reading arrays, so successful metadata validation runs
        /// once. Scalar/index/Context/geometry errors never select another path.
        pub fn matrixAdmitted(comptime T: type, comptime I: type, operation: *const Handles.Operation, handle: Handle) Error!Handles.BorrowedExport(T, I) {
            const row = exportAdmitted(T, I, operation, handle, .row) catch |err| {
                if (err != error.WrongFormat) return err;
                return exportAdmitted(T, I, operation, handle, .column);
            };
            return row;
        }

        /// Internal admitted seam. `exported` must come from matrixAdmitted for
        /// this handle and this still-live Operation; all borrowed storage stays
        /// stable through return. It is not a reusable validation certificate.
        /// Shared by typed/raw entries: no allocation, retain, pin or backend.
        pub fn productAdmitted(comptime T: type, comptime I: type, operation: *const Handles.Operation, handle: Handle, exported: Handles.BorrowedExport(T, I), transform: P.Operation, alpha: T, input: []const T, beta: T, output: []T) MultiplyError!void {
            return structuredProductAdmitted(T, I, operation, handle, exported, transform, .general, alpha, input, beta, output);
        }

        /// Narrow supported shape, not a claim about vendor rectangular behavior.
        /// Raw callers check this before constructing nonempty vector prefixes.
        pub fn checkProductShape(semantics: MatrixSemantics, rows: usize, columns: usize) Error!void {
            if (!Provider.Product.acceptsShape(semantics, rows, columns)) return error.UnsupportedDomain;
        }

        /// Shared general/structured preflight; no new numerical loop or owner.
        pub fn structuredProductAdmitted(comptime T: type, comptime I: type, operation: *const Handles.Operation, handle: Handle, exported: Handles.BorrowedExport(T, I), transform: P.Operation, semantics: MatrixSemantics, alpha: T, input: []const T, beta: T, output: []T) MultiplyError!void {
            if (try Handles.borrowedContext(handle) != operation.borrowedContext()) return error.ContextMismatch;
            const operand: P.Operand(T, I, I) = .{ .pattern = exported.view.structure, .values = exported.view.values, .operation = transform };
            try checkProductShape(semantics, operand.pattern.row_count, operand.pattern.column_count);
            if (input.len != operand.effectiveColumnCount() or output.len != operand.effectiveRowCount()) return error.DimensionMismatch;
            const destination = try Access.range(T, output);
            if (Storage.overlaps(destination, try Access.range(T, input))) return error.BufferOverlap;
            try Handles.rejectMetadataOverlap(handle, destination);
            try Access.rejectDataOverlap(T, I, destination, exported.mutable);
            Provider.Product.executeStructuredValidated(T, I, I, operand, semantics, input, output, alpha, beta);
        }

        /// General CSR/CSC only. Every call validates structure, exact vector
        /// lengths and output aliases before mutation, also when alpha/beta is
        /// zero. The unchanged Core implements N/T/H and unused-value semantics.
        /// The live handle supplies Context; no current-default lookup occurs.
        pub fn matrixVectorMultiply(comptime T: type, comptime I: type, handle: ?Handle, transform: P.Operation, alpha: T, input: []const T, beta: T, output: []T) MultiplyError!void {
            const h = handle orelse return error.NullHandle;
            const context = try Handles.borrowedContext(h);
            var operation = try context.begin();
            defer operation.end();
            const exported = try matrixAdmitted(T, I, &operation, h);
            try productAdmitted(T, I, &operation, h, exported, transform, alpha, input, beta, output);
        }

        /// Square LOWER/UPPER symmetric, Hermitian, triangular or diagonal view.
        /// Selection uses original coordinates; N/T/H is applied by the same Core.
        /// Metadata and all aliases are still checked for zero numerical scales.
        pub fn structuredMatrixVectorMultiply(comptime T: type, comptime I: type, handle: ?Handle, transform: P.Operation, semantics: MatrixSemantics, alpha: T, input: []const T, beta: T, output: []T) MultiplyError!void {
            const h = handle orelse return error.NullHandle;
            const context = try Handles.borrowedContext(h);
            var operation = try context.begin();
            defer operation.end();
            const exported = try matrixAdmitted(T, I, &operation, h);
            try structuredProductAdmitted(T, I, &operation, h, exported, transform, semantics, alpha, input, beta, output);
        }

        // MM is a separate interface contract: one shared dense layout and the
        // documented base/layout/format/operation filter. Core panels stay wider.
        pub const MatrixMultiplyError = Error || Provider.Dense.Error || error{UnsupportedConfiguration};

        /// Interface policy only. Never restrict the neutral Product/Panel types.
        pub fn checkMmConfiguration(base: P.IndexBase, axis: P.MajorAxis, transform: P.Operation, semantics: MatrixSemantics, layout: Provider.Dense.Layout) MatrixMultiplyError!void {
            if ((base == .zero and layout == .row_major) or (base == .one and layout == .column_major)) return;
            const csr_general_normal = switch (semantics) {
                .general => axis == .row and transform == .normal,
                else => false,
            };
            if (!csr_general_normal) return error.UnsupportedConfiguration;
        }

        /// Borrowed same-call preparation, not ownership or a reusable validation
        /// certificate. Keep the original handle/Operation/arrays stable until the
        /// matching executeMmAdmitted returns; do not cache or fabricate this value.
        pub fn PreparedMm(comptime T: type, comptime I: type) type {
            comptime Handles.expectTypes(T, I);
            return struct {
                exported: Handles.BorrowedExport(T, I),
                transform: P.Operation,
                semantics: MatrixSemantics,
                input_shape: Provider.Dense.PanelShape,
                output_shape: Provider.Dense.PanelShape,
                input_extent: usize,
                output_extent: usize,
            };
        }

        /// Shared typed/raw scalar, matrix and geometry preflight. No dense
        /// pointer is reached here. Sparse structure is checked once, through the
        /// existing matrixAdmitted path (WrongFormat alone selects CSC).
        pub fn prepareMmAdmitted(comptime T: type, comptime I: type, operation: *const Handles.Operation, handle: Handle, transform: P.Operation, semantics: MatrixSemantics, layout: Provider.Dense.Layout, columns: I, ldb: I, ldc: I) MatrixMultiplyError!PreparedMm(T, I) {
            if (try Handles.borrowedContext(handle) != operation.borrowedContext()) return error.ContextMismatch;
            if (columns < 0) return error.InvalidDimension;
            const rhs_count = std.math.cast(usize, columns) orelse return error.SizeOverflow;
            if (ldb < 0) return error.InvalidLeadingDimension;
            const input_stride = std.math.cast(usize, ldb) orelse return error.SizeOverflow;
            if (ldc < 0) return error.InvalidLeadingDimension;
            const output_stride = std.math.cast(usize, ldc) orelse return error.SizeOverflow;
            const exported = try matrixAdmitted(T, I, operation, handle);
            const operand: P.Operand(T, I, I) = .{ .pattern = exported.view.structure, .values = exported.view.values, .operation = transform };
            try checkProductShape(semantics, operand.pattern.row_count, operand.pattern.column_count);
            try checkMmConfiguration(operand.pattern.index_base, operand.pattern.major_axis, transform, semantics, layout);
            const input_shape: Provider.Dense.PanelShape = .{ .row_count = operand.effectiveColumnCount(), .column_count = rhs_count, .leading_dimension = input_stride, .layout = layout };
            const output_shape: Provider.Dense.PanelShape = .{ .row_count = operand.effectiveRowCount(), .column_count = rhs_count, .leading_dimension = output_stride, .layout = layout };
            const input_extent = try input_shape.extent();
            const output_extent = try output_shape.extent();
            return .{ .exported = exported, .transform = transform, .semantics = semantics, .input_shape = input_shape, .output_shape = output_shape, .input_extent = input_extent, .output_extent = output_extent };
        }

        /// Same-call seam: `prepared` belongs to this exact live handle/Operation.
        /// Typed slices carry real capacities; raw has already resolved the exact
        /// prefixes. Full bounding spans include internal padding. No C mutation
        /// occurs until every capacity/address/alias check has passed.
        pub fn executeMmAdmitted(comptime T: type, comptime I: type, operation: *const Handles.Operation, handle: Handle, prepared: PreparedMm(T, I), alpha: T, input: []const T, beta: T, output: []T) MatrixMultiplyError!void {
            if (try Handles.borrowedContext(handle) != operation.borrowedContext()) return error.ContextMismatch;
            if (input.len < prepared.input_extent) return error.BufferTooSmall;
            if (output.len < prepared.output_extent) return error.BufferTooSmall;
            const b = input[0..prepared.input_extent];
            const c = output[0..prepared.output_extent];
            const destination = try Access.range(T, c);
            if (Storage.overlaps(destination, try Access.range(T, b))) return error.BufferOverlap;
            try Handles.rejectMetadataOverlap(handle, destination);
            try Access.rejectDataOverlap(T, I, destination, prepared.exported.mutable);
            const operand: P.Operand(T, I, I) = .{ .pattern = prepared.exported.view.structure, .values = prepared.exported.view.values, .operation = prepared.transform };
            Provider.Product.executePanelValidated(T, I, I, operand, prepared.semantics, .{ .values = b, .shape = prepared.input_shape }, .{ .values = c, .shape = prepared.output_shape }, alpha, beta);
        }

        /// MKL-shaped semantic MM, not a C ABI. Typed/raw apply the same single
        /// layout/configuration policy. Even zero scales/RHS retain all checks.
        /// Source Context admits once; no allocator lookup, retain or new pin.
        pub fn mm(comptime T: type, comptime I: type, handle: ?Handle, transform: P.Operation, semantics: MatrixSemantics, layout: Provider.Dense.Layout, alpha: T, input: []const T, columns: I, ldb: I, beta: T, output: []T, ldc: I) MatrixMultiplyError!void {
            const h = handle orelse return error.NullHandle;
            const context = try Handles.borrowedContext(h);
            var operation = try context.begin();
            defer operation.end();
            const prepared = try prepareMmAdmitted(T, I, &operation, h, transform, semantics, layout, columns, ldb, ldc);
            try executeMmAdmitted(T, I, &operation, h, prepared, alpha, input, beta, output);
        }

        pub fn Input(comptime T: type, comptime I: type) type {
            comptime Handles.expectTypes(T, I);
            return struct {
                base: i32,
                rows: I,
                columns: I,
                axis: Axis,
                starts: []I,
                ends: []I,
                ordinals: []I,
                values: []T,
            };
        }
        pub fn Export(comptime T: type, comptime I: type) type {
            comptime Handles.expectTypes(T, I);
            return struct { base: i32, rows: I, columns: I, starts: []I, ends: []I, ordinals: []I, values: []T };
        }

        /// One admission covers validation, both metadata allocations, output
        /// commit and every cleanup. Never read the old output handle value.
        /// Only active prefixes are retained/checked; unused capacity is excluded
        /// from alias checks and may be accessed through other caller operands.
        pub fn createBorrowed(comptime T: type, comptime I: type, context: *Execution.Context, input: Input(T, I), output: ?*?Handle) Error!void {
            var operation = try context.begin();
            defer operation.end();
            const out = output orelse return error.NullOutput;
            const geometry = try Access.shape(I, input.base, input.rows, input.columns, input.axis);
            const ranges = try Access.captureRanges(I, geometry, input.starts, input.ends);
            const data = try Access.captureData(T, I, ranges, input.ordinals, input.values);
            const handle = try borrowAdmitted(T, I, &operation, geometry, input.axis, data, try Access.outputRange(?Handle, out), .{});
            out.* = handle;
        }

        /// Internal seam: caller retains one same-owner Operation through the
        /// final output commit. Data are stable checked prefixes, not cached proof.
        /// The actual destination range is checked before common ordinal checks
        /// and before either metadata allocation, as in the typed entry.
        pub fn borrowAdmitted(comptime T: type, comptime I: type, operation: *const Handles.Operation, geometry: Access.Shape, axis: Axis, data: Handles.MutableData(T, I), destination: Storage.ByteRange, provenance: Handles.Provenance(T, I)) Error!Handle {
            try Access.rejectDataOverlap(T, I, destination, data);
            return Handles.borrowWithProvenance(T, I, operation, geometry.rows, geometry.columns, axis, geometry.base, data, provenance);
        }

        /// Shared same-Context export/revalidation, with no output mutation and
        /// no dimension cast until each frontend has checked its destinations.
        pub fn exportAdmitted(comptime T: type, comptime I: type, operation: *const Handles.Operation, handle: Handle, axis: Axis) Error!Handles.BorrowedExport(T, I) {
            const exported = try Handles.exportView(T, I, handle, operation, axis);
            _ = try Access.backingExtent(I, exported.mutable.starts, exported.mutable.ends, exported.view.structure.index_base);
            return exported;
        }

        /// No allocation/retain/conversion. Recheck common borrowed metadata on
        /// every call, then enforce the interface's narrower interval domain.
        /// The returned mutable slices retain only their original C-like input
        /// capability; no readonly or library-owned control can enter this path.
        pub fn exportBorrowed(comptime T: type, comptime I: type, handle: ?Handle, axis: Axis, output: ?*Export(T, I)) Error!void {
            const h = handle orelse return error.NullHandle;
            const context = try Handles.borrowedContext(h);
            var operation = try context.begin();
            defer operation.end();
            const out = output orelse return error.NullOutput;
            const exported = try exportAdmitted(T, I, &operation, h, axis);
            const structure = exported.view.structure;
            const data = exported.mutable;
            const output_span = try Access.outputRange(Export(T, I), out);
            try Handles.rejectMetadataOverlap(h, output_span);
            try Access.rejectDataOverlap(T, I, output_span, data);
            const rows = std.math.cast(I, structure.row_count) orelse return error.SizeOverflow;
            const columns = std.math.cast(I, structure.column_count) orelse return error.SizeOverflow;
            out.* = .{ .base = if (structure.index_base == .zero) 0 else 1, .rows = rows, .columns = columns, .starts = data.starts, .ends = data.ends, .ordinals = data.ordinals, .values = data.values };
        }

        /// Destruction must remain available while Context is closing; never
        /// admit work or inspect borrowed arrays on this path.
        pub const destroy = Handles.destroy;
    };
}
