// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Pure SYEVD/HEEVD boundary policy. Numerical work, admission, scratch storage
//! and callbacks remain with their existing owners. This module exports no C ABI.
const std = @import("std");
const query_codec = @import("lapacke_gelsd_workspace.zig");

pub const Family = enum { real, complex };
pub const Error = error{ InvalidDimension, IntegerNotRepresentable, InvalidOutcome };

fn integer(comptime I: type) void {
    if (I != i32 and I != i64) @compileError("EVD uses signed LP64/ILP64 ABI integers");
}

pub fn Workspace(comptime I: type) type {
    integer(I);
    return struct { work: I, rwork: I = 1, iwork: I };
}

pub fn minimumWorkspace(comptime I: type, size: I, vectors: bool, family: Family) Error!Workspace(I) {
    if (size < 0) return error.InvalidDimension;
    if (size <= 1) return .{ .work = 1, .rwork = 1, .iwork = 1 };
    // The widest supported positive ABI integer fits these expressions in u128.
    const n: u128 = @intCast(size);
    const work: u128 = switch (family) {
        .real => if (vectors) 1 + 6 * n + 2 * n * n else 2 * n + 1,
        .complex => if (vectors) n * n + 2 * n else n + 1,
    };
    const rwork: u128 = if (family == .complex) (if (vectors) 1 + 5 * n + 2 * n * n else n) else 1;
    const iwork: u128 = if (vectors) 3 + 5 * n else 1;
    return .{
        .work = std.math.cast(I, work) orelse return error.IntegerNotRepresentable,
        .rwork = std.math.cast(I, rwork) orelse return error.IntegerNotRepresentable,
        .iwork = std.math.cast(I, iwork) orelse return error.IntegerNotRepresentable,
    };
}

pub fn isQuery(comptime I: type, family: Family, lengths: Workspace(I)) bool {
    return lengths.work == -1 or lengths.iwork == -1 or (family == .complex and lengths.rwork == -1);
}

/// LAPACKE positions include matrix_layout; a query suppresses all workspace
/// length errors, while the adapter must still validate the preceding scalars.
pub fn workspaceInfo(comptime I: type, family: Family, lengths: Workspace(I), required: Workspace(I)) I {
    if (isQuery(I, family, lengths)) return 0;
    if (lengths.work < required.work) return -9;
    if (family == .complex and lengths.rwork < required.rwork) return -11;
    if (lengths.iwork < required.iwork) return if (family == .complex) -13 else -11;
    return 0;
}

/// Reuse the existing upward, ABI-bounded query codec instead of duplicating it.
pub fn queryValue(comptime T: type, comptime I: type, count: I) Error!T {
    integer(I);
    if (count < 1) return error.InvalidDimension;
    const native = std.math.cast(usize, count) orelse return error.IntegerNotRepresentable;
    return query_codec.queryLength(T, I, native) catch error.IntegerNotRepresentable;
}

/// JOBZ=N reports remaining edges. JOBZ=V reports the actual failed submatrix
/// using one-based inclusive endpoints encoded as first*(N+1)+last.
pub fn iterationInfo(comptime I: type, size: I, vectors: bool, unconverged: usize, block: anytype) Error!I {
    integer(I);
    if (size < 0) return error.InvalidDimension;
    if (unconverged == 0) return 0;
    const n = std.math.cast(usize, size) orelse return error.IntegerNotRepresentable;
    if (unconverged >= n) return error.InvalidOutcome;
    if (!vectors) return std.math.cast(I, unconverged) orelse error.IntegerNotRepresentable;
    const failed = block orelse return error.InvalidOutcome;
    if (failed.first >= failed.limit or failed.limit > n or failed.limit - failed.first < 2) return error.InvalidOutcome;
    const encoded = (@as(u128, failed.first) + 1) * (@as(u128, n) + 1) + failed.limit;
    return std.math.cast(I, encoded) orelse error.IntegerNotRepresentable;
}
