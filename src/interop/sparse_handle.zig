// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Interface-owned typed shell, not Intel's private sparse_matrix layout.
//! One Family is instantiated by the binding for every scalar/index entry.
//! Its common control owns the only resource pin; the shell owns one reference.
//! All handle/control/array access is externally serialized. A handle must be
//! live and originate here: no arbitrary-pointer, stale-pointer or double-free
//! detection is promised. Same-handle allocator-callback reentry is forbidden,
//! including during free before it returns. Copying a handle does not retain it.
//! Context/allocator/code backing must outlive every API stack, not only its pin.

const std = @import("std");
const builtin = @import("builtin");

pub fn Family(comptime Provider: type) type {
    const P = Provider.Pattern;
    const S = Provider.Scalar;
    const Object = Provider.Object;
    const Storage = Provider.Storage;
    const Execution = Provider.Execution.context;
    return struct {
        pub const OwnerIdentity = Provider.OwnerIdentity;
        pub const Context = Execution.Context;
        pub const Operation = Execution.Operation;
        pub const Handle = *opaque {};
        pub const Error = Object.Error || error{ CounterOverflow, ContextMismatch, NullHandle, InvalidHandle, WrongScalar, WrongIndexWidth, BufferOverlap };

        pub fn Control(comptime T: type, comptime I: type) type {
            comptime expectTypes(T, I);
            return Object.ManagedMatrixControl(Execution, T, I, I);
        }

        /// This capability comes only from actually mutable borrowed inputs.
        /// There is deliberately no constructor from a readonly common Export.
        pub fn MutableData(comptime T: type, comptime I: type) type {
            comptime expectTypes(T, I);
            return struct { starts: []I, ends: []I, ordinals: []I, values: []T };
        }
        pub fn BorrowedExport(comptime T: type, comptime I: type) type {
            return struct { view: Control(T, I).Export, mutable: MutableData(T, I), provenance: Provenance(T, I) };
        }
        fn Record(comptime T: type, comptime I: type) type {
            return struct { control: *Control(T, I), mutable: MutableData(T, I), provenance: Provenance(T, I) };
        }
        /// Interface-only pointer-value carriers. Alignment one is intentional:
        /// an empty input's original value is never dereferenced or strengthened.
        pub fn RawPointer(comptime T: type) type {
            return ?[*]align(1) T;
        }
        pub fn EmptySource(comptime T: type) type {
            return union(enum) { not_captured, empty_original: RawPointer(T) };
        }
        pub const Encoding = enum { typed_split, c_split, c_null_end };
        pub fn Provenance(comptime T: type, comptime I: type) type {
            comptime expectTypes(T, I);
            return struct {
                encoding: Encoding = .typed_split,
                starts: EmptySource(I) = .not_captured,
                ends: EmptySource(I) = .not_captured,
                ordinals: EmptySource(I) = .not_captured,
                values: EmptySource(T) = .not_captured,
            };
        }
        pub fn RawPointers(comptime T: type, comptime I: type) type {
            comptime expectTypes(T, I);
            return struct { starts: RawPointer(I), ends: RawPointer(I), ordinals: RawPointer(I), values: RawPointer(T) };
        }
        fn captured(source: anytype) bool {
            return switch (source) {
                .not_captured => false,
                .empty_original => true,
            };
        }
        fn checkEmptySource(source: anytype, length: usize) Error!void {
            if (captured(source) != (length == 0)) return error.InvalidHandle;
        }
        /// InvalidHandle here means an inconsistent internal provenance record,
        /// not a proof that arbitrary raw addresses are safe to inspect.
        fn checkProvenance(comptime T: type, comptime I: type, rows: usize, columns: usize, axis: P.MajorAxis, data: MutableData(T, I), source: Provenance(T, I)) Error!void {
            if (source.encoding == .typed_split) {
                if (captured(source.starts) or captured(source.ends) or captured(source.ordinals) or captured(source.values)) return error.InvalidHandle;
                return; // Preserve common Pattern's original typed error order.
            }
            const major = if (axis == .row) rows else columns;
            if (data.starts.len != major or data.ends.len != major or data.ordinals.len != data.values.len) return error.InvalidHandle;
            if (major == 0 and data.ordinals.len != 0) return error.InvalidHandle;
            try checkEmptySource(source.starts, data.starts.len);
            if (source.encoding == .c_null_end) {
                if (captured(source.ends)) return error.InvalidHandle;
                if (major != 0) {
                    const shifted = std.math.add(usize, @intFromPtr(data.starts.ptr), @sizeOf(I)) catch return error.InvalidHandle;
                    if (@intFromPtr(data.ends.ptr) != shifted) return error.InvalidHandle;
                }
            } else {
                try checkEmptySource(source.ends, data.ends.len);
            }
            try checkEmptySource(source.ordinals, data.ordinals.len);
            try checkEmptySource(source.values, data.values.len);
        }
        fn restorePointer(comptime T: type, data: []T, source: EmptySource(T)) RawPointer(T) {
            return switch (source) {
                .not_captured => data.ptr,
                .empty_original => |original| original,
            };
        }
        /// Only a same-call export may use these values. No empty token becomes
        /// an accessible span, no integer is converted to a pointer, and no extra
        /// reference, resource pin or array allocation is created.
        pub fn rawPointers(comptime T: type, comptime I: type, exported: BorrowedExport(T, I)) Error!RawPointers(T, I) {
            const shape = exported.view.structure;
            const data = exported.mutable;
            const source = exported.provenance;
            try checkProvenance(T, I, shape.row_count, shape.column_count, shape.major_axis, data, source);
            return .{
                .starts = restorePointer(I, data.starts, source.starts),
                .ends = if (source.encoding == .c_null_end) null else restorePointer(I, data.ends, source.ends),
                .ordinals = restorePointer(I, data.ordinals, source.ordinals),
                .values = restorePointer(T, data.values, source.values),
            };
        }

        // Private semantic tags, never wire enum values or vendor discriminators.
        const Tag = enum { s32, s64, d32, d64, c32, c64, z32, z64 };
        const Payload = union(Tag) {
            s32: Record(f32, i32),
            s64: Record(f32, i64),
            d32: Record(f64, i32),
            d64: Record(f64, i64),
            c32: Record(S.ComplexF32, i32),
            c64: Record(S.ComplexF32, i64),
            z32: Record(S.ComplexF64, i32),
            z64: Record(S.ComplexF64, i64),
        };
        const marker: u64 = 0x5a594e5350415253;
        const Shell = struct {
            family_marker: u64 = marker,
            schema: u32 = 1,
            phase: enum { live, destroying } = .live,
            context: *Context,
            allocator: std.mem.Allocator,
            payload: Payload,
        };

        pub fn expectTypes(comptime T: type, comptime I: type) void {
            if (T != f32 and T != f64 and T != S.ComplexF32 and T != S.ComplexF64)
                @compileError("typed Sparse shell supports only the Core's four real/complex scalar types");
            if (I != i32 and I != i64) @compileError("typed Sparse shell requires signed i32 or i64 indices");
        }
        fn tag(comptime T: type, comptime I: type) Tag {
            comptime expectTypes(T, I);
            if (T == f32) return if (I == i32) .s32 else .s64;
            if (T == f64) return if (I == i32) .d32 else .d64;
            if (T == S.ComplexF32) return if (I == i32) .c32 else .c64;
            return if (I == i32) .z32 else .z64;
        }
        fn scalarTag(value: Tag) u8 {
            return switch (value) {
                .s32, .s64 => 0,
                .d32, .d64 => 1,
                .c32, .c64 => 2,
                .z32, .z64 => 3,
            };
        }
        fn live(handle: Handle) Error!*Shell {
            // Alignment/magic checks only inspect an already legally readable
            // object. They cannot turn an invalid external pointer into safety.
            if (@intFromPtr(handle) % @alignOf(Shell) != 0) return error.InvalidHandle;
            const self: *Shell = @ptrCast(@alignCast(handle));
            if (self.family_marker != marker or self.schema != 1 or self.phase != .live) return error.InvalidHandle;
            return self;
        }
        fn record(comptime T: type, comptime I: type, self: *Shell) Error!*Record(T, I) {
            const actual = std.meta.activeTag(self.payload);
            const requested = comptime tag(T, I);
            if (scalarTag(actual) != scalarTag(requested)) return error.WrongScalar;
            if (actual != requested) return error.WrongIndexWidth;
            return &@field(self.payload, @tagName(requested));
        }

        /// Internal construction seam. Engine supplies checked active prefixes;
        /// common borrow still performs its full structural check before pinning.
        /// The shape is made here from mutable provenance, never independently
        /// supplied readonly pointers which could grant an accidental cast-away.
        pub fn borrow(comptime T: type, comptime I: type, operation: *const Operation, rows: usize, columns: usize, axis: P.MajorAxis, base: P.IndexBase, data: MutableData(T, I)) Error!Handle {
            return borrowWithProvenance(T, I, operation, rows, columns, axis, base, data, .{});
        }

        /// Shared metadata construction. Raw frontends can attach only explicit
        /// mutable input provenance; typed calls keep the original split record.
        pub fn borrowWithProvenance(comptime T: type, comptime I: type, operation: *const Operation, rows: usize, columns: usize, axis: P.MajorAxis, base: P.IndexBase, data: MutableData(T, I), provenance: Provenance(T, I)) Error!Handle {
            try checkProvenance(T, I, rows, columns, axis, data, provenance);
            const structure: P.Pattern(I, I) = .{ .row_count = rows, .column_count = columns, .major_axis = axis, .index_base = base, .ranges = .{ .split = .{ .starts = data.starts, .ends = data.ends } }, .ordinals = data.ordinals };
            const control = try Control(T, I).borrow(operation, structure, data.values);
            errdefer control.release();
            // Reuse the control's captured allocator, never a second config or
            // a second allocator lookup after the first allocation's callback.
            const allocator = control.allocator;
            const self = try allocator.create(Shell);
            self.* = .{ .context = operation.borrowedContext(), .allocator = allocator, .payload = @unionInit(Payload, @tagName(tag(T, I)), .{ .control = control, .mutable = data, .provenance = provenance }) };
            return @ptrCast(self);
        }

        pub fn borrowedContext(handle: Handle) Error!*Context {
            return (try live(handle)).context;
        }
        pub fn exportView(comptime T: type, comptime I: type, handle: Handle, operation: *const Operation, axis: P.MajorAxis) Error!BorrowedExport(T, I) {
            const self = try live(handle);
            if (self.context != operation.borrowedContext()) return error.ContextMismatch;
            const entry = try record(T, I, self);
            const view = try entry.control.exportView(operation, axis);
            return .{ .view = view, .mutable = entry.mutable, .provenance = entry.provenance };
        }
        /// Reject output writes into the shell or common control itself. No
        /// mutable pointer to these records is exported by the production API.
        pub fn rejectMetadataOverlap(handle: Handle, output: Storage.ByteRange) Error!void {
            const self = try live(handle);
            const shell_range = try Storage.byteRange(@intFromPtr(self), @sizeOf(Shell));
            if (Storage.overlaps(output, shell_range)) return error.BufferOverlap;
            switch (self.payload) {
                inline else => |entry| {
                    const control_range = try Storage.byteRange(@intFromPtr(entry.control), @sizeOf(@TypeOf(entry.control.*)));
                    if (Storage.overlaps(output, control_range)) return error.BufferOverlap;
                },
            }
        }

        /// No begin, no borrowed metadata reads. The still-held control reference
        /// keeps its one pin alive throughout shell free and until common final
        /// release finishes. Do not read self or Context after release returns.
        pub fn destroy(handle: ?Handle) Error!void {
            const self = try live(handle orelse return error.NullHandle);
            const allocator = self.allocator;
            const payload = self.payload;
            self.phase = .destroying;
            allocator.destroy(self);
            switch (payload) {
                inline else => |entry| entry.control.release(),
            }
        }

        pub const testing = if (builtin.is_test) struct {
            /// Real explicit reference for existing worker-cache lifecycle tests;
            /// not a public retain API or a readonly-to-mutable shell constructor.
            pub fn retainControl(comptime T: type, comptime I: type, handle: Handle) Error!*Control(T, I) {
                return (try record(T, I, try live(handle))).control.retain();
            }
        } else struct {};
    };
}
