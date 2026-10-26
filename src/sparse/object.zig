// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Internal synchronous CSR/CSC ownership prototype, not an MKL handle/API.
//! Each allocation belongs to one captured allocator; its state/code must outlive
//! every retained reference. No allocator, pool, cache or Context is created here.
//! Calls, retain/release and exported reads must be externally serialized.
//! Managed data calls borrow one live same-Context Operation; retain/release
//! need no new admission. Each control owns one pin regardless of reference count.
//! A live control must not be copied, moved, or have its fields changed directly.
//! Retain is explicit; copying a pointer does not retain. Last release invalidates
//! the pointer. No stale/illegal pointer or double-release detection is promised.
//!
//! Borrowed arrays stay caller-owned and stable during each data operation.
//! Such operations revalidate, including after a prior successful export; raw
//! exports have no retained reference or mutation authority. They remain valid
//! only while both the control and its underlying storage remain alive.
//! Owned exports are read-only, with stable addresses until last release. There
//! are no mutation methods, vendor status mappings or persistent analysis here.

const std = @import("std");
const pattern = @import("pattern.zig");

pub const Ownership = enum { borrowed, owned_snapshot };
pub const Error = pattern.Error || std.mem.Allocator.Error || error{ WrongFormat, ReferenceCountOverflow };
pub const Capabilities = struct { export_same_format: bool = true, deep_snapshot: bool = true, update_values: bool = false };

/// Original allocator-only entry. It does not participate in Context close.
pub fn MatrixControl(comptime T: type, comptime O: type, comptime I: type) type {
    return Control(void, T, O, I);
}

/// Explicit managed entry, injected by the unique implementation owner. Execution
/// supplies the already-owned Operation and ResourcePin types; this file does not
/// import/create a Context, default, pool, allocator owner or named Module.
/// Data calls require an admitted operation from the object's same Context.
/// References remain non-atomic and externally serialized; a pin is not ownership
/// of borrowed arrays, exported slices, allocator backing state or callback code.
pub fn ManagedMatrixControl(comptime Execution: type, comptime T: type, comptime O: type, comptime I: type) type {
    return Control(Execution, T, O, I);
}

fn Control(comptime Execution: type, comptime T: type, comptime O: type, comptime I: type) type {
    const managed = Execution != void;
    const Source = if (managed) *const Execution.Operation else std.mem.Allocator;
    const Pin = if (managed) Execution.ResourcePin else void;
    const ControlError = if (managed) Error || error{ CounterOverflow, ContextMismatch } else Error;
    const Operand = pattern.Operand(T, O, I); // Also enforces supported scalar/index types.
    return struct {
        const Self = @This();
        const OwnedRanges = union(enum) {
            none,
            contiguous: []O,
            split: struct { starts: []O, ends: []O },
        };
        const PatternRecord = struct {
            descriptor: pattern.Pattern(O, I),
            ranges_allocation: OwnedRanges = .none,
            ordinals_allocation: ?[]I = null,
        };
        const ValuesRecord = struct { data: []const T, allocation: ?[]T = null };

        // Internal fields are not a public mutation API. Typed allocation slices
        // retain exact sizes/alignment; descriptor subspans are never freed.
        allocator: std.mem.Allocator,
        resource_pin: Pin,
        references: usize = 1,
        ownership: Ownership,
        pattern_record: PatternRecord,
        values_record: ValuesRecord,

        pub const Export = struct {
            structure: pattern.Pattern(O, I),
            values: []const T,
            statistics: pattern.Statistics,
            ownership: Ownership,
            // Null for borrowed data: no version certifies external bytes.
            // Owned snapshots are immutable; these are object-local revisions,
            // not globally unique cache identities or external mutation detectors.
            pattern_revision: ?u64,
            values_revision: ?u64,
            capabilities: Capabilities = .{},
        };

        /// Allocates the control only. Validation precedes allocation and reads
        /// no numeric values. Operation N/T/H belongs to a call, not this object.
        pub fn borrow(source: Source, structure: pattern.Pattern(O, I), values: []const T) ControlError!*Self {
            _ = try (Operand{ .pattern = structure, .values = values }).check();
            const allocator = allocatorFrom(source);
            var pin = try acquirePin(source);
            errdefer releasePin(&pin);
            const self = try allocator.create(Self);
            self.* = .{ .allocator = allocator, .resource_pin = pin, .ownership = .borrowed, .pattern_record = .{ .descriptor = structure }, .values_record = .{ .data = values } };
            return self;
        }

        /// Deep snapshot preserves format, base, widths, capacities, ranges,
        /// physical entry positions, duplicates and stored order. Only active
        /// ordinals/values are read. Inactive gaps and tails become all-zero bits
        /// (not necessarily valid one-based ordinals, but never active entries).
        /// Empty split ranges may end beyond values.len; they read/copy no values.
        /// This gap policy is a common prototype rule, not MKL copy conformance.
        /// All temporary allocations unwind on failure; source remains untouched.
        pub fn snapshot(source: Source, structure: pattern.Pattern(O, I), values: []const T) ControlError!*Self {
            return snapshotUsing(source, allocatorFrom(source), structure, values);
        }

        fn snapshotUsing(source: Source, allocator: std.mem.Allocator, structure: pattern.Pattern(O, I), values: []const T) ControlError!*Self {
            _ = try (Operand{ .pattern = structure, .values = values }).check();
            try checkAllocationBytes(O, switch (structure.ranges) {
                .contiguous => |v| v.len,
                .split => |v| v.starts.len,
            });
            try checkAllocationBytes(I, structure.ordinals.len);
            try checkAllocationBytes(T, values.len);
            var pin = try acquirePin(source);
            // This outer errdefer runs after storage/control cleanup below.
            errdefer releasePin(&pin);
            const self = try allocator.create(Self);
            self.* = .{ .allocator = allocator, .resource_pin = pin, .ownership = .owned_snapshot, .pattern_record = .{ .descriptor = structure }, .values_record = .{ .data = values } };
            errdefer {
                self.freeStorage();
                allocator.destroy(self);
            }
            switch (structure.ranges) {
                .contiguous => |input| {
                    const offsets = try allocator.dupe(O, input);
                    self.pattern_record.ranges_allocation = .{ .contiguous = offsets };
                    self.pattern_record.descriptor.ranges = .{ .contiguous = offsets };
                },
                .split => |input| {
                    const starts = try allocator.dupe(O, input.starts);
                    // Own this first allocation immediately, even if ends fails.
                    self.pattern_record.ranges_allocation = .{ .contiguous = starts };
                    const ends = try allocator.dupe(O, input.ends);
                    self.pattern_record.ranges_allocation = .{ .split = .{ .starts = starts, .ends = ends } };
                    self.pattern_record.descriptor.ranges = .{ .split = .{ .starts = starts, .ends = ends } };
                },
            }
            const ordinals = try allocator.alloc(I, structure.ordinals.len);
            self.pattern_record.ordinals_allocation = ordinals;
            self.pattern_record.descriptor.ordinals = ordinals;
            @memset(std.mem.sliceAsBytes(ordinals), 0);
            const copied_values = try allocator.alloc(T, values.len);
            self.values_record = .{ .data = copied_values, .allocation = copied_values };
            @memset(std.mem.sliceAsBytes(copied_values), 0);
            for (0..structure.majorCount()) |major| {
                // Geometry was validated above and input must stay stable. This
                // remains fallible for bounds rather than manufacturing unchecked
                // physical pointers. Any error still unwinds unpublished storage.
                const range = try structure.checkedMajorRange(major);
                if (range.begin == range.end) continue;
                @memcpy(ordinals[range.begin..range.end], structure.ordinals[range.begin..range.end]);
                @memcpy(std.mem.sliceAsBytes(copied_values[range.begin..range.end]), std.mem.sliceAsBytes(values[range.begin..range.end]));
            }
            return self;
        }

        /// Copy into an independent object using this object's captured allocator.
        /// Revalidates borrowed metadata; does not retain the source object.
        pub const deepCopy = if (managed) deepCopyManaged else deepCopyUnmanaged;
        fn deepCopyUnmanaged(self: *const Self) Error!*Self {
            return snapshot(self.allocator, self.pattern_record.descriptor, self.values_record.data);
        }
        /// Same Context only. A fresh control receives its own pin, but uses the
        /// source's captured allocator; no retain or implicit ownership transfer.
        fn deepCopyManaged(self: *const Self, operation: Source) ControlError!*Self {
            try self.requireContext(operation);
            return snapshotUsing(operation, self.allocator, self.pattern_record.descriptor, self.values_record.data);
        }

        /// Nonallocating, same-format borrowed export. No conversion, transform,
        /// finite scan or retain. Successful validation is valid for this stable
        /// call only. Wrong format is the first error for this new internal API.
        pub const exportView = if (managed) exportViewManaged else exportViewUnmanaged;
        fn exportViewManaged(self: *const Self, operation: Source, major_axis: pattern.MajorAxis) ControlError!Export {
            try self.requireContext(operation);
            return self.exportViewUnmanaged(major_axis);
        }
        fn exportViewUnmanaged(self: *const Self, major_axis: pattern.MajorAxis) Error!Export {
            if (major_axis != self.pattern_record.descriptor.major_axis) return error.WrongFormat;
            const stats = try (Operand{ .pattern = self.pattern_record.descriptor, .values = self.values_record.data }).check();
            const revision: ?u64 = if (self.ownership == .owned_snapshot) 1 else null;
            return .{ .structure = self.pattern_record.descriptor, .values = self.values_record.data, .statistics = stats, .ownership = self.ownership, .pattern_revision = revision, .values_revision = revision };
        }

        /// Only explicit references count. No structural reads: a caller can
        /// retain/release a control even if its external buffers became invalid.
        pub fn retain(self: *Self) Error!*Self {
            self.references = std.math.add(usize, self.references, 1) catch return error.ReferenceCountOverflow;
            return self;
        }

        /// Consumes exactly one live reference. Last release frees owned values,
        /// ordinals, range arrays, then the control with their original allocator.
        /// Borrowed inputs are never read or freed. Managed pin release is after the
        /// final free returns; last release never initiates drain or shutdown.
        pub fn release(self: *Self) void {
            std.debug.assert(self.references != 0);
            self.references -= 1;
            if (self.references != 0) return;
            const allocator = self.allocator;
            // Move the one pin into the destruction frame before freeing self.
            // No access to self is permitted after destroy, including callbacks.
            var pin = self.resource_pin;
            self.freeStorage();
            allocator.destroy(self);
            releasePin(&pin);
        }

        fn allocatorFrom(source: Source) std.mem.Allocator {
            return if (managed) source.allocator() else source;
        }
        fn acquirePin(source: Source) ControlError!Pin {
            if (managed) return source.pinResource();
            return {};
        }
        fn releasePin(pin: *Pin) void {
            if (managed) pin.releaseAfterResourceFree();
        }
        fn requireContext(self: *const Self, operation: Source) ControlError!void {
            if (self.resource_pin.context != operation.borrowedContext()) return error.ContextMismatch;
        }

        fn freeStorage(self: *Self) void {
            if (self.values_record.allocation) |allocation| self.allocator.free(allocation);
            if (self.pattern_record.ordinals_allocation) |allocation| self.allocator.free(allocation);
            switch (self.pattern_record.ranges_allocation) {
                .none => {},
                .contiguous => |allocation| self.allocator.free(allocation),
                .split => |allocations| {
                    self.allocator.free(allocations.ends);
                    self.allocator.free(allocations.starts);
                },
            }
        }

        fn checkAllocationBytes(comptime E: type, count: usize) Error!void {
            _ = std.math.mul(usize, @sizeOf(E), count) catch return error.SizeOverflow;
        }
    };
}
