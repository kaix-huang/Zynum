// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const blas = @import("zynum-blas");

test "BLAS package module imports directly" {
    var values = [_]f32{ 1, 2, 3, 4 };
    const vector = try blas.constVector(f32, &values, .{});
    try std.testing.expectEqual(@as(blas.BlasInt, 4), vector.length);
}

test "BLAS package vector operations use shared-prefix semantics" {
    var source_values = [_]f32{ 1, 2, 3, 100 };
    var destination_values = [_]f32{ 10, 20 };
    const source = try blas.constVector(f32, &source_values, .{});
    const destination = try blas.vector(f32, &destination_values, .{});

    try blas.addScaledVector(.{
        .source_vector = source,
        .destination_vector = destination,
        .scale = 3,
    });

    try std.testing.expectEqualSlices(f32, &.{ 13, 26 }, &destination_values);
}

test "BLAS package Into vector operations require equal lengths" {
    var input_values = [_]f32{ 1, 2, 3 };
    var result_values = [_]f32{ 0, 0 };
    const input = try blas.constVector(f32, &input_values, .{});
    const result = try blas.vector(f32, &result_values, .{});

    try std.testing.expectError(error.DimensionMismatch, blas.scaleVectorInto(.{
        .input_vector = input,
        .result_vector = result,
        .scale = 2,
    }));
}
