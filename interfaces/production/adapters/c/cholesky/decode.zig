// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// C descriptor/flag decoding only. Numeric preflight and arithmetic stay in Core.
const lapack = @import("zynum-core").lapack;

// WireBridge is provided by artifact composition, never discovered at runtime.
// Reuse its sole ABI types, dense span preflight and pointer-to-slice rules.
pub fn Decoder(comptime WireBridge: type) type {
    return struct {
        const abi = WireBridge.Wire;
        pub fn triangle(flag: i32) !lapack.Triangle {
            return switch (flag) {
                abi.LOWER => .lower,
                abi.UPPER => .upper,
                else => error.InvalidFlag,
            };
        }

        pub fn matrix(comptime T: type, a: abi.dense) !lapack.Matrix(T) {
            try WireBridge.densePreflight(a);
            return .{ .values = try WireBridge.writeBuffer(T, a.storage), .row_count = @intCast(a.rows), .column_count = @intCast(a.cols), .leading_dimension = @intCast(a.ld) };
        }

        pub fn constMatrix(comptime T: type, a: abi.const_dense) !lapack.ConstMatrix(T) {
            try WireBridge.densePreflight(a);
            return .{ .values = try WireBridge.readBuffer(T, a.storage), .row_count = @intCast(a.rows), .column_count = @intCast(a.cols), .leading_dimension = @intCast(a.ld) };
        }
    };
}
