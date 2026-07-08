// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const zynum = @import("zynum");

const blas = zynum.blas;

pub fn main(init: std.process.Init) !void {
    // Zynum matrix views use BLAS/Fortran column-major storage by default.
    const a_values = [_]f64{
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0,
    };
    const b_values = [_]f64{
        7.0, 9.0,  11.0,
        8.0, 10.0, 12.0,
    };
    var c_values = [_]f64{0.0} ** 4;

    const a = try blas.constMatrix(f64, a_values[0..], .{
        .row_count = 2,
        .column_count = 3,
    });
    const b = try blas.constMatrix(f64, b_values[0..], .{
        .row_count = 3,
        .column_count = 2,
    });
    const c = try blas.matrix(f64, c_values[0..], .{
        .row_count = 2,
        .column_count = 2,
    });

    try blas.matrixMultiply(.{
        .left_matrix = a,
        .right_matrix = b,
        .result_matrix = c,
    });

    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout_writer.interface.print("C = A x B\n", .{});
    try printColumnMajorMatrix(&stdout_writer.interface, 2, 2, c_values[0..]);
    try stdout_writer.flush();
}

fn printColumnMajorMatrix(writer: *std.Io.Writer, rows: usize, cols: usize, values: []const f64) !void {
    for (0..rows) |row| {
        for (0..cols) |col| {
            try writer.print("{d:>8.1}", .{values[row + col * rows]});
        }
        try writer.print("\n", .{});
    }
}
