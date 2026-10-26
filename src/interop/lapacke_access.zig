// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Closed, call-local operand representations for internal LAPACKE adapters.
//! This is not a C ABI or an allocation owner. Raw operands carry no capacity:
//! their caller guarantees truthful, stable, accessible storage for every span
//! actually requested. Checks establish arithmetic/alignment, not allocation
//! provenance, address validity, lifetime, or freedom from concurrent mutation.
//! No slice is constructed until the existing engine reaches its access stage.

pub const Kind = enum { typed, raw };
pub const TypedFault = error{ IntegerNotRepresentable, SizeOverflow, MissingBuffer, BufferTooSmall, AliasingNotAllowed };
pub const RawFault = error{ IntegerNotRepresentable, SizeOverflow, MissingBuffer, MisalignedPointer, AliasingNotAllowed };

pub fn Fault(comptime kind: Kind) type {
    return if (kind == .typed) TypedFault else RawFault;
}
pub fn Operand(comptime T: type, comptime writable: bool, comptime kind: Kind) type {
    return if (kind == .typed)
        ?(if (writable) []T else []const T)
    else
        ?(if (writable) [*]align(1) T else [*]align(1) const T);
}

pub fn capacity(comptime Storage: type, comptime kind: Kind, comptime T: type, comptime writable: bool, values: Operand(T, writable, kind), count: usize) Fault(kind)!(if (writable) []T else []const T) {
    if (comptime kind == .typed) {
        // Preserve the original typed capacityImpl order, including the empty
        // slice's original pointer when one was supplied.
        if (count == 0) return if (values) |v| v[0..0] else &.{};
        const bytes = try Storage.byteLength(count, @sizeOf(T));
        const v = values orelse return error.MissingBuffer;
        if (v.len < count) return error.BufferTooSmall;
        _ = try Storage.byteRange(@intFromPtr(v.ptr), bytes);
        return v[0..count];
    } else {
        // Raw-only order: zero count, byte size, null, alignment, endpoint.
        // The optional operand itself retains NULL for scan's nonread branch.
        if (count == 0) return &.{};
        const bytes = try Storage.byteLength(count, @sizeOf(T));
        const v = values orelse return error.MissingBuffer;
        const address = @intFromPtr(v);
        if (address % @alignOf(T) != 0) return error.MisalignedPointer;
        _ = try Storage.byteRange(address, bytes);
        const pointer: if (writable) [*]T else [*]const T = @alignCast(v);
        return pointer[0..count];
    }
}
