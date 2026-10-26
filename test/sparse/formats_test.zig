// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const sparse = @import("zynum-sparse");
const formats = sparse;

fn expectResult(comptime T: type, comptime I: type, result: sparse.ConstCsrMatrix(T, I)) !void {
    try result.check();
    try std.testing.expectEqual(@as(usize, 3), result.row_count);
    try std.testing.expectEqual(@as(usize, 4), result.column_count);
    try std.testing.expectEqualSlices(I, &.{ 0, 2, 4, 7 }, result.row_offsets);
    try std.testing.expectEqualSlices(I, &.{ 0, 2, 2, 3, 0, 0, 3 }, result.column_indices);
    try std.testing.expectEqualSlices(T, &.{ 1, -2, 3, 1, 2, -0.5, 4 }, result.values);
    var output = [_]T{ 0, 0, 0 };
    try sparse.matrixVectorMultiply(.{ .matrix = result, .input_vector = &[_]T{ 2, 3, 5, 7 }, .result_vector = &output });
    // Independent dense multiplication including additive duplicate (2,0).
    try std.testing.expectEqualSlices(T, &.{ -8, 22, 31 }, &output);
}

test "CSC converts unsorted repeated rows to sorted CSR for all supported types" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |I| {
            const source = try formats.constCscMatrix(T, I, &.{ 0, 3, 3, 5, 7 }, &.{ 2, 0, 2, 1, 0, 2, 1 }, &.{ 2, 1, -0.5, 3, -2, 4, 1 }, .{ .row_count = 3, .column_count = 4 });
            const required = try formats.cscToCsrRequirements(I, 3, 7);
            try std.testing.expectEqualDeep(formats.ConversionRequirements{ .row_offsets = 4, .column_indices = 7, .values = 7, .workspace = 3 }, required);
            var offsets: [5]I = @splat(99);
            var indices: [8]I = @splat(99);
            var values: [8]T = @splat(99);
            var workspace: [4]usize = @splat(99);
            const result = try formats.cscToCsr(.{ .matrix = source, .row_offsets = &offsets, .column_indices = &indices, .values = &values, .workspace = &workspace });
            try expectResult(T, I, result);
            try std.testing.expectEqual(@as(I, 99), offsets[4]);
            try std.testing.expectEqual(@as(I, 99), indices[7]);
            try std.testing.expectEqual(@as(T, 99), values[7]);
            try std.testing.expectEqual(@as(usize, 99), workspace[3]);
        }
    }
}

test "COO stable radix conversion preserves duplicates for all supported types" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ i32, u32, i64, u64 }) |I| {
            const source = try formats.constCooMatrix(T, I, &.{ 2, 0, 1, 2, 0, 2, 1 }, &.{ 3, 2, 3, 0, 0, 0, 2 }, &.{ 4, -2, 1, 2, 1, -0.5, 3 }, .{ .row_count = 3, .column_count = 4 });
            const required = try formats.cooToCsrRequirements(I, 3, 7);
            try std.testing.expectEqual(@as(usize, 273), required.workspace);
            var offsets: [5]I = @splat(99);
            var indices: [8]I = @splat(99);
            var values: [8]T = @splat(99);
            var workspace: [274]usize = @splat(99);
            const result = try formats.cooToCsr(.{ .matrix = source, .row_offsets = &offsets, .column_indices = &indices, .values = &values, .workspace = &workspace });
            try expectResult(T, I, result);
            try std.testing.expectEqual(@as(I, 99), offsets[4]);
            try std.testing.expectEqual(@as(I, 99), indices[7]);
            try std.testing.expectEqual(@as(T, 99), values[7]);
            try std.testing.expectEqual(@as(usize, 99), workspace[273]);
        }
    }
}

test "CSC CSR transpose reinterpretation roundtrip preserves sorted duplicate storage" {
    const original = try formats.constCooMatrix(f64, i64, &.{ 2, 0, 1, 2, 0, 2, 1 }, &.{ 3, 2, 3, 0, 0, 0, 2 }, &.{ 4, -2, 1, 2, 1, -0.5, 3 }, .{ .row_count = 3, .column_count = 4 });
    var original_offsets: [4]i64 = undefined;
    var original_indices: [7]i64 = undefined;
    var original_values: [7]f64 = undefined;
    var original_workspace: [273]usize = undefined;
    const csr = try formats.cooToCsr(.{ .matrix = original, .row_offsets = &original_offsets, .column_indices = &original_indices, .values = &original_values, .workspace = &original_workspace });

    const transposed_csc = try formats.constCscMatrix(f64, i64, csr.row_offsets, csr.column_indices, csr.values, .{ .row_count = 4, .column_count = 3 });
    var transpose_offsets: [5]i64 = undefined;
    var transpose_indices: [7]i64 = undefined;
    var transpose_values: [7]f64 = undefined;
    var workspace: [4]usize = undefined;
    const transposed_csr = try formats.cscToCsr(.{ .matrix = transposed_csc, .row_offsets = &transpose_offsets, .column_indices = &transpose_indices, .values = &transpose_values, .workspace = &workspace });
    try transposed_csr.check();
    const back_csc = try formats.constCscMatrix(f64, i64, transposed_csr.row_offsets, transposed_csr.column_indices, transposed_csr.values, .{ .row_count = 3, .column_count = 4 });
    var back_offsets: [4]i64 = undefined;
    var back_indices: [7]i64 = undefined;
    var back_values: [7]f64 = undefined;
    const back = try formats.cscToCsr(.{ .matrix = back_csc, .row_offsets = &back_offsets, .column_indices = &back_indices, .values = &back_values, .workspace = &workspace });
    try expectResult(f64, i64, back);
    try std.testing.expectEqualSlices(i64, csr.row_offsets, back.row_offsets);
    try std.testing.expectEqualSlices(i64, csr.column_indices, back.column_indices);
    try std.testing.expectEqualSlices(f64, csr.values, back.values);
}

test "COO radix passes cover high 32 and 64 bit coordinates without column-sized storage" {
    inline for (.{ i32, u32, i64, u64 }) |I| {
        const limit = @min(@as(u64, std.math.maxInt(I)), @as(u64, std.math.maxInt(usize)) - 1);
        const high: I = @intCast(limit / 2 + 17);
        const dimension: usize = @as(usize, @intCast(high)) + 1;
        const source = try formats.constCooMatrix(f64, I, &.{ 0, 0, 0, 0 }, &.{ high, 3, 256, high }, &.{ 1, 2, 3, 4 }, .{ .row_count = 1, .column_count = dimension });
        var offsets: [2]I = undefined;
        var indices: [4]I = undefined;
        var values: [4]f64 = undefined;
        var workspace: [265]usize = undefined;
        const result = try formats.cooToCsr(.{ .matrix = source, .row_offsets = &offsets, .column_indices = &indices, .values = &values, .workspace = &workspace });
        try std.testing.expectEqualSlices(I, &.{ 0, 4 }, result.row_offsets);
        try std.testing.expectEqualSlices(I, &.{ 3, 256, high, high }, result.column_indices);
        try std.testing.expectEqualSlices(f64, &.{ 2, 3, 1, 4 }, result.values);
    }
}

test "conversion copies NaN payloads signed zero and repeated values without arithmetic" {
    const raw = [_]u32{ 0x7fc00001, 0x80000000, 0x7fc00007, 0x00000000, 0x7f800000 };
    const values = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(&raw));
    const source = try formats.constCooMatrix(f32, u32, &.{ 0, 0, 0, 0, 0 }, &.{ 1, 0, 1, 0, 1 }, values, .{ .row_count = 1, .column_count = 2 });
    var offsets: [2]u32 = undefined;
    var indices: [5]u32 = undefined;
    var output: [5]f32 = undefined;
    var workspace: [267]usize = undefined;
    const result = try formats.cooToCsr(.{ .matrix = source, .row_offsets = &offsets, .column_indices = &indices, .values = &output, .workspace = &workspace });
    const output_bits = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(result.values));
    try std.testing.expectEqualSlices(u32, &.{ 0x80000000, 0x00000000, 0x7fc00001, 0x7fc00007, 0x7f800000 }, output_bits);
}

test "zero dimensions and zero nnz require only row offsets and no workspace" {
    const coo = try formats.constCooMatrix(f64, u64, &.{}, &.{}, &.{}, .{ .row_count = 0, .column_count = 9 });
    var one_offset = [_]u64{99};
    var empty_indices: [0]u64 = .{};
    var empty_values: [0]f64 = .{};
    var empty_workspace: [0]usize = .{};
    const zero_rows = try formats.cooToCsr(.{ .matrix = coo, .row_offsets = &one_offset, .column_indices = &empty_indices, .values = &empty_values, .workspace = &empty_workspace });
    try zero_rows.check();
    try std.testing.expectEqual(@as(u64, 0), one_offset[0]);
    const csc = try formats.constCscMatrix(f64, u64, &.{0}, &.{}, &.{}, .{ .row_count = 3, .column_count = 0 });
    var offsets = [_]u64{ 99, 99, 99, 99 };
    const zero_columns = try formats.cscToCsr(.{ .matrix = csc, .row_offsets = &offsets, .column_indices = &empty_indices, .values = &empty_values, .workspace = &empty_workspace });
    try zero_columns.check();
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 0, 0 }, &offsets);
    try std.testing.expectEqual(@as(usize, 0), (try formats.cscToCsrRequirements(u64, 3, 0)).workspace);
    try std.testing.expectEqual(@as(usize, 0), (try formats.cooToCsrRequirements(u64, 3, 0)).workspace);
    try std.testing.expectError(error.InvalidRowIndex, formats.constCooMatrix(f64, i32, &.{0}, &.{0}, &.{1}, .{ .row_count = 0, .column_count = 1 }));
    try std.testing.expectError(error.InvalidColumnIndex, formats.constCooMatrix(f64, i32, &.{0}, &.{0}, &.{1}, .{ .row_count = 1, .column_count = 0 }));
}

test "borrowed CSC and COO reject malformed structural input" {
    const dimensions = .{ .row_count = 2, .column_count = 2 };
    try std.testing.expectError(error.InvalidLength, formats.constCscMatrix(f64, i32, &.{ 0, 1 }, &.{0}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidLength, formats.constCscMatrix(f64, i32, &.{ 0, 0, 1 }, &.{}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidColumnOffsets, formats.constCscMatrix(f64, i32, &.{ 1, 1, 1 }, &.{0}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidColumnOffsets, formats.constCscMatrix(f64, i32, &.{ 0, 1, 0 }, &.{0}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidColumnOffsets, formats.constCscMatrix(f64, i32, &.{ 0, -1, 1 }, &.{0}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidColumnOffsets, formats.constCscMatrix(f64, i32, &.{ 0, 0, 0 }, &.{0}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidColumnOffsets, formats.constCscMatrix(f64, i32, &.{ 0, 2, 2 }, &.{0}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidRowIndex, formats.constCscMatrix(f64, i32, &.{ 0, 0, 1 }, &.{-1}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidRowIndex, formats.constCscMatrix(f64, i32, &.{ 0, 0, 1 }, &.{2}, &.{1}, dimensions));
    try std.testing.expectError(error.SizeOverflow, formats.constCscMatrix(f64, u64, &.{}, &.{}, &.{}, .{ .row_count = 0, .column_count = std.math.maxInt(usize) }));
    try std.testing.expectError(error.InvalidLength, formats.constCooMatrix(f64, i32, &.{0}, &.{}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidLength, formats.constCooMatrix(f64, i32, &.{}, &.{0}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidRowIndex, formats.constCooMatrix(f64, i32, &.{-1}, &.{0}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidRowIndex, formats.constCooMatrix(f64, u64, &.{std.math.maxInt(u64)}, &.{0}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidColumnIndex, formats.constCooMatrix(f64, i32, &.{0}, &.{-1}, &.{1}, dimensions));
    try std.testing.expectError(error.InvalidColumnIndex, formats.constCooMatrix(f64, u64, &.{0}, &.{std.math.maxInt(u64)}, &.{1}, dimensions));
}

test "requirements detect row count terminal offset and workspace overflow" {
    const max = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, formats.cscToCsrRequirements(u64, max, 0));
    try std.testing.expectError(error.SizeOverflow, formats.cooToCsrRequirements(u64, max - 1, 1));
    try std.testing.expectError(error.SizeOverflow, formats.cooToCsrRequirements(u64, 1, max / 2 + 1));
    try std.testing.expectError(error.SizeOverflow, formats.cooToCsrRequirements(u64, 1, max / 2));
    try std.testing.expectError(error.SizeOverflow, formats.cscToCsrRequirements(i32, 1, @as(usize, std.math.maxInt(i32)) + 1));
    if (comptime @bitSizeOf(usize) > 32) {
        try std.testing.expectError(error.SizeOverflow, formats.cooToCsrRequirements(u32, 1, @as(usize, std.math.maxInt(u32)) + 1));
    }
}

fn expectSentinels(offsets: []const u64, indices: []const u64, values: []const f64, workspace: []const usize) !void {
    for (offsets) |value| try std.testing.expectEqual(@as(u64, 99), value);
    for (indices) |value| try std.testing.expectEqual(@as(u64, 99), value);
    for (values) |value| try std.testing.expectEqual(@as(f64, 99), value);
    for (workspace) |value| try std.testing.expectEqual(@as(usize, 99), value);
}

test "COO conversion rejects each insufficient capacity before mutating any buffer" {
    const source = try formats.constCooMatrix(f64, u64, &.{ 1, 0 }, &.{ 0, 1 }, &.{ 2, 3 }, .{ .row_count = 2, .column_count = 2 });
    var offsets: [3]u64 = @splat(99);
    var indices: [2]u64 = @splat(99);
    var values: [2]f64 = @splat(99);
    var workspace: [262]usize = @splat(99);
    var args = .{ .matrix = source, .row_offsets = @as([]u64, &offsets), .column_indices = @as([]u64, &indices), .values = @as([]f64, &values), .workspace = @as([]usize, &workspace) };
    args.row_offsets = offsets[0..2];
    try std.testing.expectError(error.BufferTooSmall, formats.cooToCsr(args));
    args.row_offsets = &offsets;
    args.column_indices = indices[0..1];
    try std.testing.expectError(error.BufferTooSmall, formats.cooToCsr(args));
    args.column_indices = &indices;
    args.values = values[0..1];
    try std.testing.expectError(error.BufferTooSmall, formats.cooToCsr(args));
    args.values = &values;
    args.workspace = workspace[0..261];
    try std.testing.expectError(error.WorkspaceTooSmall, formats.cooToCsr(args));
    try expectSentinels(&offsets, &indices, &values, &workspace);
}

test "CSC conversion rejects short output or workspace before mutation" {
    const source = try formats.constCscMatrix(f64, u64, &.{ 0, 1, 2 }, &.{ 1, 0 }, &.{ 2, 3 }, .{ .row_count = 2, .column_count = 2 });
    var offsets: [3]u64 = @splat(99);
    var indices: [2]u64 = @splat(99);
    var values: [2]f64 = @splat(99);
    var workspace: [2]usize = @splat(99);
    try std.testing.expectError(error.BufferTooSmall, formats.cscToCsr(.{ .matrix = source, .row_offsets = offsets[0..2], .column_indices = &indices, .values = &values, .workspace = &workspace }));
    try std.testing.expectError(error.WorkspaceTooSmall, formats.cscToCsr(.{ .matrix = source, .row_offsets = &offsets, .column_indices = &indices, .values = &values, .workspace = workspace[0..1] }));
    try expectSentinels(&offsets, &indices, &values, &workspace);
}

test "both conversions revalidate mutated source and preserve all outputs on failure" {
    var rows = [_]u64{ 1, 0 };
    var columns = [_]u64{ 0, 1 };
    const coo = try formats.constCooMatrix(f64, u64, &rows, &columns, &.{ 2, 3 }, .{ .row_count = 2, .column_count = 2 });
    var source_offsets = [_]u64{ 0, 1, 2 };
    const csc = try formats.constCscMatrix(f64, u64, &source_offsets, &rows, &.{ 2, 3 }, .{ .row_count = 2, .column_count = 2 });
    var offsets: [3]u64 = @splat(99);
    var indices: [2]u64 = @splat(99);
    var values: [2]f64 = @splat(99);
    var workspace: [262]usize = @splat(99);
    rows[1] = 2;
    try std.testing.expectError(error.InvalidRowIndex, formats.cooToCsr(.{ .matrix = coo, .row_offsets = &offsets, .column_indices = &indices, .values = &values, .workspace = &workspace }));
    try std.testing.expectError(error.InvalidRowIndex, formats.cscToCsr(.{ .matrix = csc, .row_offsets = &offsets, .column_indices = &indices, .values = &values, .workspace = &workspace }));
    rows[1] = 0;
    columns[1] = 2;
    try std.testing.expectError(error.InvalidColumnIndex, formats.cooToCsr(.{ .matrix = coo, .row_offsets = &offsets, .column_indices = &indices, .values = &values, .workspace = &workspace }));
    source_offsets[2] = 1;
    try std.testing.expectError(error.InvalidColumnOffsets, formats.cscToCsr(.{ .matrix = csc, .row_offsets = &offsets, .column_indices = &indices, .values = &values, .workspace = &workspace }));
    var huge = coo;
    huge.column_indices = &.{ 0, 1 };
    huge.row_count = std.math.maxInt(usize);
    try std.testing.expectError(error.SizeOverflow, formats.cooToCsr(.{ .matrix = huge, .row_offsets = &offsets, .column_indices = &indices, .values = &values, .workspace = &workspace }));
    try expectSentinels(&offsets, &indices, &values, &workspace);
}

test "conversion rejects output and workspace aliasing including index-value overlap" {
    var source_rows = [_]u64{ 1, 0 };
    var source_columns = [_]u64{ 0, 1 };
    var source_values = [_]f64{ 2, 3 };
    const source = try formats.constCooMatrix(f64, u64, &source_rows, &source_columns, &source_values, .{ .row_count = 2, .column_count = 2 });
    var offsets: [3]u64 = @splat(99);
    var indices: [2]u64 = @splat(99);
    var values: [2]f64 = @splat(99);
    var workspace: [262]usize = @splat(99);
    const args = .{ .matrix = source, .row_offsets = @as([]u64, &offsets), .column_indices = @as([]u64, &indices), .values = @as([]f64, &values), .workspace = @as([]usize, &workspace) };
    var bad = args;
    bad.column_indices = offsets[1..3];
    try std.testing.expectError(error.AliasingNotAllowed, formats.cooToCsr(bad));
    bad = args;
    bad.column_indices = &source_columns;
    try std.testing.expectError(error.AliasingNotAllowed, formats.cooToCsr(bad));
    bad = args;
    bad.column_indices = &source_rows;
    try std.testing.expectError(error.AliasingNotAllowed, formats.cooToCsr(bad));
    bad = args;
    bad.values = &source_values;
    try std.testing.expectError(error.AliasingNotAllowed, formats.cooToCsr(bad));
    bad = args;
    bad.column_indices = std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(&values));
    try std.testing.expectError(error.AliasingNotAllowed, formats.cooToCsr(bad));
    bad = args;
    const offset_words = 3 * @sizeOf(u64) / @sizeOf(usize);
    bad.row_offsets = std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(workspace[1 .. 1 + offset_words]));
    try std.testing.expectError(error.AliasingNotAllowed, formats.cooToCsr(bad));
    try expectSentinels(&offsets, &indices, &values, &workspace);
    try std.testing.expectEqualSlices(u64, &.{ 1, 0 }, &source_rows);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1 }, &source_columns);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3 }, &source_values);
}

test "workspace overlapping source metadata fails before source mutation" {
    var workspace: [262]usize = @splat(99);
    const metadata_words = 2 * @sizeOf(u64) / @sizeOf(usize);
    const source_rows = std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(workspace[0..metadata_words]));
    source_rows[0] = 1;
    source_rows[1] = 0;
    const snapshot = workspace;
    const source = try formats.constCooMatrix(f64, u64, source_rows, &.{ 0, 1 }, &.{ 2, 3 }, .{ .row_count = 2, .column_count = 2 });
    var offsets: [3]u64 = @splat(99);
    var indices: [2]u64 = @splat(99);
    var values: [2]f64 = @splat(99);
    try std.testing.expectError(error.AliasingNotAllowed, formats.cooToCsr(.{ .matrix = source, .row_offsets = &offsets, .column_indices = &indices, .values = &values, .workspace = &workspace }));
    try std.testing.expectEqualSlices(usize, &snapshot, &workspace);
    try expectSentinels(&offsets, &indices, &values, &.{});
}

test "CSC output alias with source offsets and values is rejected" {
    var source_offsets = [_]u64{ 0, 1, 2 };
    var source_values = [_]f64{ 2, 3 };
    const source = try formats.constCscMatrix(f64, u64, &source_offsets, &.{ 1, 0 }, &source_values, .{ .row_count = 2, .column_count = 2 });
    var offsets: [3]u64 = @splat(99);
    var indices: [2]u64 = @splat(99);
    var values: [2]f64 = @splat(99);
    var workspace: [2]usize = @splat(99);
    try std.testing.expectError(error.AliasingNotAllowed, formats.cscToCsr(.{ .matrix = source, .row_offsets = &source_offsets, .column_indices = &indices, .values = &values, .workspace = &workspace }));
    try std.testing.expectError(error.AliasingNotAllowed, formats.cscToCsr(.{ .matrix = source, .row_offsets = &offsets, .column_indices = &indices, .values = &source_values, .workspace = &workspace }));
    try expectSentinels(&offsets, &indices, &values, &workspace);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 2 }, &source_offsets);
    try std.testing.expectEqualSlices(f64, &.{ 2, 3 }, &source_values);
}

test "disjoint active prefixes within larger shared backing storage are allowed" {
    var index_backing: [12]u64 = @splat(99);
    index_backing[0] = 0;
    index_backing[1] = 1;
    index_backing[2] = 2;
    index_backing[3] = 1;
    index_backing[4] = 0;
    var numeric_backing = [_]f64{ 2, 3, 99, 99, 99 };
    const source = try formats.constCscMatrix(f64, u64, index_backing[0..3], index_backing[3..5], numeric_backing[0..2], .{ .row_count = 2, .column_count = 2 });
    var workspace: [2]usize = undefined;
    const result = try formats.cscToCsr(.{ .matrix = source, .row_offsets = index_backing[5..], .column_indices = index_backing[8..], .values = numeric_backing[2..], .workspace = &workspace });
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 2 }, result.row_offsets);
    try std.testing.expectEqualSlices(u64, &.{ 1, 0 }, result.column_indices);
    try std.testing.expectEqualSlices(f64, &.{ 3, 2 }, result.values);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 2, 1, 0 }, index_backing[0..5]);
    try std.testing.expectEqualSlices(u64, &.{ 99, 99 }, index_backing[10..]);
    try std.testing.expectEqual(@as(f64, 99), numeric_backing[4]);
}
