// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const views = @import("views.zig");

const ByteRange = struct {
    start: usize,
    end: usize,

    fn isEmpty(self: ByteRange) bool {
        return self.start == self.end;
    }
};

fn byteRange(ptr: anytype, len: usize, comptime T: type) ByteRange {
    const start = @intFromPtr(ptr);
    return .{ .start = start, .end = start + len * @sizeOf(T) };
}

fn rangesOverlap(first: ByteRange, second: ByteRange) bool {
    if (first.isEmpty() or second.isEmpty()) return false;
    return first.start < second.end and second.start < first.end;
}

pub fn vectorsExactlyMatch(first: anytype, second: anytype) bool {
    return @intFromPtr(first.values.ptr) == @intFromPtr(second.values.ptr) and
        first.length == second.length and
        first.stride == second.stride;
}

pub fn vectorRange(comptime T: type, vector: anytype) views.Error!ByteRange {
    const len = try views.requiredVectorStorageLength(vector.length, vector.stride);
    return byteRange(vector.values.ptr, len, T);
}

pub fn matrixRange(comptime T: type, matrix: anytype) views.Error!ByteRange {
    const len = try views.requiredMatrixStorageLength(matrix.row_count, matrix.column_count, matrix.leading_dimension);
    return byteRange(matrix.values.ptr, len, T);
}

pub fn vectorsOverlap(comptime T: type, first: anytype, second: anytype) views.Error!bool {
    return rangesOverlap(try vectorRange(T, first), try vectorRange(T, second));
}

pub fn vectorMatrixOverlap(comptime T: type, vector: anytype, matrix: anytype) views.Error!bool {
    return rangesOverlap(try vectorRange(T, vector), try matrixRange(T, matrix));
}

pub fn matricesOverlap(comptime T: type, first: anytype, second: anytype) views.Error!bool {
    return rangesOverlap(try matrixRange(T, first), try matrixRange(T, second));
}

pub fn ensureNoVectorOverlap(comptime T: type, first: anytype, second: anytype) views.Error!void {
    if (!views.runtime_checks_enabled) return;
    if (try vectorsOverlap(T, first, second)) return error.AliasingNotAllowed;
}

pub fn ensureNoVectorMatrixOverlap(comptime T: type, vector: anytype, matrix: anytype) views.Error!void {
    if (!views.runtime_checks_enabled) return;
    if (try vectorMatrixOverlap(T, vector, matrix)) return error.AliasingNotAllowed;
}

pub fn ensureNoMatrixOverlap(comptime T: type, first: anytype, second: anytype) views.Error!void {
    if (!views.runtime_checks_enabled) return;
    if (try matricesOverlap(T, first, second)) return error.AliasingNotAllowed;
}

pub fn ensureNoPartialVectorOverlap(comptime T: type, first: anytype, second: anytype) views.Error!void {
    if (!views.runtime_checks_enabled) return;
    if (vectorsExactlyMatch(first, second)) return;
    if (try vectorsOverlap(T, first, second)) return error.AliasingNotAllowed;
}
