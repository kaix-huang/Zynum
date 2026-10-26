// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Raw FEAST LP64 boundary over the package's actual Core module. Generated
//! declarations forward here. Truthful, live caller pointers remain required.
const std = @import("std");
pub const implementation = @import("zynum-core");
const n = implementation.internal.numerics;
pub const Provider = struct {
    pub const OwnerIdentity = implementation;
    pub const Scalar = n.Scalar;
    pub const Dense = n.Dense;
    pub const Storage = n.Storage;
    pub const Execution = n.Execution;
    pub const Contour = n.ContourEigensolver;
    pub const SparseOperator = implementation.internal.sparse_checked.operator;
};
pub const Driver = @import("feast.zig").Adapter(Provider);
const Parameters = @import("feast_parameters.zig");
const T = n.Scalar.ComplexF64;
pub fn initialize(fpm: [*c]i32) void {
    if (fpm == null) return;
    Parameters.initialize(fpm[0..64]) catch unreachable;
}
pub fn hcsrev(uplo: [*c]u8, dimension: [*c]i32, values: [*c]f64, rows: [*c]i32, columns: [*c]i32, fpm: [*c]i32, epsout: [*c]f64, loop: [*c]i32, lower: [*c]f64, upper: [*c]f64, capacity: [*c]i32, eigenvalues: [*c]f64, vectors: [*c]f64, found: [*c]i32, residuals: [*c]f64, info: [*c]i32) void {
    if (info == null) return;
    info.* = Parameters.unsupported;
    if (uplo == null) {
        info.* = -101;
        return;
    }
    const triangle = std.ascii.toUpper(uplo.*);
    if (triangle != 'F' and triangle != 'L' and triangle != 'U') {
        info.* = -101;
        return;
    }
    if (dimension == null or dimension.* <= 0) {
        info.* = -102;
        return;
    }
    if (capacity == null or capacity.* <= 1 or capacity.* > dimension.*) {
        info.* = 201;
        return;
    }
    if (lower == null or upper == null or !std.math.isFinite(lower.*) or !std.math.isFinite(upper.*) or !(lower.* < upper.*)) {
        info.* = 200;
        return;
    }
    if (rows == null) {
        info.* = -104;
        return;
    }
    if (fpm == null or epsout == null or loop == null or found == null or eigenvalues == null or vectors == null or residuals == null) return;
    const order: usize = @intCast(dimension.*);
    const m: usize = @intCast(capacity.*);
    if (rows[order] < 1) {
        info.* = -104;
        return;
    }
    const entries: usize = @intCast(rows[order] - 1);
    if (entries != 0 and values == null) {
        info.* = -103;
        return;
    }
    if (entries != 0 and columns == null) {
        info.* = -105;
        return;
    }
    const size = std.math.mul(usize, order, m) catch return;
    _ = n.Storage.byteLength(size, @sizeOf(T)) catch return;
    _ = n.Storage.byteLength(entries, @sizeOf(T)) catch return;
    const matrix: []const T = if (entries == 0) &.{} else @as([*]const T, @ptrCast(values))[0..entries];
    const ordinals: []const i32 = if (entries == 0) &.{} else columns[0..entries];
    var operation = n.Execution.beginDefault();
    defer operation.end();
    const result = Driver.solveAdmitted(&operation, .{ .uplo = uplo.*, .dimension = dimension.*, .capacity = capacity.*, .matrix = matrix, .rows = rows[0 .. order + 1], .columns = ordinals, .fpm = fpm[0..64], .lower = lower.*, .upper = upper.*, .eigenvalues = eigenvalues[0..m], .vectors = @as([*]T, @ptrCast(vectors))[0..size], .residuals = residuals[0..m] }) catch |err| {
        info.* = switch (err) {
            error.OutOfMemory => -1,
            else => Parameters.unsupported,
        };
        return;
    };
    info.* = result.info;
    if (result.found) |count| found.* = @intCast(count);
    if (result.loop) |count| loop.* = @intCast(count);
    if (result.epsout) |error_trace| epsout.* = error_trace;
    if (result.native) |native| if (result.info >= 0 and native.published != 0) {
        capacity.* = @intCast(native.published);
    };
}
