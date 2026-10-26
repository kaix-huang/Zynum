// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const formats = @import("zynum-sparse");

test "independent coordinate-order reference for randomized CSC and COO conversion" {
    var generator = std.Random.DefaultPrng.init(0x9c7124af307be810);
    const random = generator.random();
    for (0..500) |_| {
        const rows = random.intRangeAtMost(usize, 1, 16);
        const columns = random.intRangeAtMost(usize, 1, 16);
        const nnz = random.intRangeAtMost(usize, 0, 64);
        var input_rows: [64]i64 = undefined;
        var input_columns: [64]i64 = undefined;
        var input_values: [64]f64 = undefined;
        for (0..nnz) |entry| {
            input_rows[entry] = @intCast(random.intRangeLessThan(usize, 0, rows));
            input_columns[entry] = @intCast(random.intRangeLessThan(usize, 0, columns));
            input_values[entry] = @floatFromInt(entry + 1);
        }
        var expected_offsets: [17]i64 = undefined;
        var expected_columns: [64]i64 = undefined;
        var expected_values: [64]f64 = undefined;
        var cursor: usize = 0;
        expected_offsets[0] = 0;
        // Exhaustively visit coordinates, then retain original entry order.
        // No radix sort, row histogram, or cursor-scattering algorithm is used.
        for (0..rows) |row| {
            for (0..columns) |column| {
                for (0..nnz) |entry| {
                    if (input_rows[entry] == row and input_columns[entry] == column) {
                        expected_columns[cursor] = @intCast(column);
                        expected_values[cursor] = input_values[entry];
                        cursor += 1;
                    }
                }
            }
            expected_offsets[row + 1] = @intCast(cursor);
        }
        try std.testing.expectEqual(nnz, cursor);
        const coo = try formats.constCooMatrix(f64, i64, input_rows[0..nnz], input_columns[0..nnz], input_values[0..nnz], .{ .row_count = rows, .column_count = columns });
        var output_offsets: [17]i64 = undefined;
        var output_columns: [64]i64 = undefined;
        var output_values: [64]f64 = undefined;
        var scratch: [400]usize = undefined;
        const coo_result = try formats.cooToCsr(.{ .matrix = coo, .row_offsets = &output_offsets, .column_indices = &output_columns, .values = &output_values, .workspace = &scratch });
        try std.testing.expectEqualSlices(i64, expected_offsets[0 .. rows + 1], coo_result.row_offsets);
        try std.testing.expectEqualSlices(i64, expected_columns[0..nnz], coo_result.column_indices);
        try std.testing.expectEqualSlices(f64, expected_values[0..nnz], coo_result.values);
        try coo_result.check();
        var csc_offsets: [17]i64 = undefined;
        var csc_rows: [64]i64 = undefined;
        var csc_values: [64]f64 = undefined;
        cursor = 0;
        csc_offsets[0] = 0;
        for (0..columns) |column| {
            for (0..nnz) |entry| {
                if (input_columns[entry] == column) {
                    csc_rows[cursor] = input_rows[entry];
                    csc_values[cursor] = input_values[entry];
                    cursor += 1;
                }
            }
            csc_offsets[column + 1] = @intCast(cursor);
        }
        const csc = try formats.constCscMatrix(f64, i64, csc_offsets[0 .. columns + 1], csc_rows[0..nnz], csc_values[0..nnz], .{ .row_count = rows, .column_count = columns });
        const csc_result = try formats.cscToCsr(.{ .matrix = csc, .row_offsets = &output_offsets, .column_indices = &output_columns, .values = &output_values, .workspace = &scratch });
        try std.testing.expectEqualSlices(i64, expected_offsets[0 .. rows + 1], csc_result.row_offsets);
        try std.testing.expectEqualSlices(i64, expected_columns[0..nnz], csc_result.column_indices);
        try std.testing.expectEqualSlices(f64, expected_values[0..nnz], csc_result.values);
        try csc_result.check();
    }
}
