// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");
const zynum = @import("zynum");
const blas = @import("zynum-blas");
const lapack = @import("zynum-lapack");
const sparse = @import("zynum-sparse");

test "top-level package exposes Zynum BLAS namespace" {
    var values = [_]f64{ 1, 2, 3 };
    const vector = try zynum.blas.constVector(f64, &values, .{});
    try std.testing.expectEqual(@as(zynum.blas.BlasInt, 3), vector.length);
}

test "modern typed gemm API" {
    var a = [_]f64{
        1, 2,
        3, 4,
        5, 6,
    };
    var b = [_]f64{
        7,  8,  9,
        10, 11, 12,
    };
    var c = [_]f64{ 0, 0, 0, 0 };
    const left_matrix = try zynum.constMatrix(f64, &a, .{ .row_count = 2, .column_count = 3 });
    const right_matrix = try zynum.constMatrix(f64, &b, .{ .row_count = 3, .column_count = 2 });
    const result_matrix = try zynum.matrix(f64, &c, .{ .row_count = 2, .column_count = 2 });
    try zynum.matrixMultiply(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
    });
    try std.testing.expectApproxEqAbs(@as(f64, 76), c[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 100), c[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 103), c[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 136), c[3], 1e-12);
}

test "modern API rejects structural shape errors in all build modes" {
    var short = [_]f64{ 1, 2, 3 };
    try std.testing.expectError(error.InvalidLength, zynum.constVector(f64, &short, .{ .length = -1 }));
    try std.testing.expectError(error.InvalidStride, zynum.constVector(f64, &short, .{ .length = 3, .stride = 0 }));

    var matrix_data = [_]f64{ 1, 2, 3, 4 };
    try std.testing.expectError(error.InvalidLeadingDimension, zynum.constMatrix(f64, &matrix_data, .{
        .row_count = 2,
        .column_count = 2,
        .leading_dimension = 1,
    }));

    var a_data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var c_data = [_]f64{ 0, 0, 0, 0 };
    const left_matrix = try zynum.constMatrix(f64, &a_data, .{ .row_count = 2, .column_count = 3 });
    const right_matrix = try zynum.constMatrix(f64, &b_data, .{ .row_count = 2, .column_count = 3 });
    const result_matrix = try zynum.matrix(f64, &c_data, .{ .row_count = 2, .column_count = 2 });
    try std.testing.expectError(error.DimensionMismatch, zynum.matrixMultiply(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
    }));

    var scale_input = [_]f64{ 1, 2, 3 };
    var scale_result = [_]f64{ 0, 0 };
    const input_vector = try zynum.constVector(f64, &scale_input, .{});
    const result_vector = try zynum.vector(f64, &scale_result, .{});
    try std.testing.expectError(error.DimensionMismatch, zynum.scaleVectorInto(.{
        .input_vector = input_vector,
        .result_vector = result_vector,
        .scale = 2,
    }));
}

test "modern checked API rejects short backing storage" {
    if (builtin.mode == .fast) return error.SkipZigTest;

    var short = [_]f64{ 1, 2, 3 };
    try std.testing.expectError(error.BufferTooSmall, zynum.constVector(f64, &short, .{ .length = 3, .stride = 2 }));
}

test "modern vector Into operations write out of place" {
    var input = [_]f64{ 1, 2, 3 };
    var result = [_]f64{ 0, 0, 0 };
    const input_vector = try zynum.constVector(f64, &input, .{});
    const result_vector = try zynum.vector(f64, &result, .{});

    try zynum.scaleVectorInto(.{
        .input_vector = input_vector,
        .result_vector = result_vector,
        .scale = 4,
    });

    try std.testing.expectEqualSlices(f64, &.{ 4, 8, 12 }, &result);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3 }, &input);
}

test "modern matrix vector multiply rejects aliasing without workspace" {
    if (builtin.mode == .fast) return error.SkipZigTest;

    var matrix_data = [_]f64{ 1, 2, 3, 4 };
    var vector_data = [_]f64{ 20, 10 };
    const matrix_view = try zynum.constMatrix(f64, &matrix_data, .{ .row_count = 2, .column_count = 2 });
    const input_vector = try zynum.constVector(f64, &vector_data, .{});
    const result_vector = try zynum.vector(f64, &vector_data, .{});

    try std.testing.expectError(error.AliasingNotAllowed, zynum.matrixVectorMultiply(.{
        .matrix = matrix_view,
        .input_vector = input_vector,
        .result_vector = result_vector,
    }));
}

test "modern matrix vector multiply workspace supports in-place result" {
    var matrix_data = [_]f64{ 1, 2, 3, 4 };
    var vector_data = [_]f64{ 20, 10 };
    var workspace_data = [_]f64{ 0, 0 };
    const matrix_view = try zynum.constMatrix(f64, &matrix_data, .{ .row_count = 2, .column_count = 2 });
    const input_vector = try zynum.constVector(f64, &vector_data, .{});
    const result_vector = try zynum.vector(f64, &vector_data, .{});

    try std.testing.expectEqual(@as(usize, 2), try zynum.matrixVectorMultiplyWorkspaceLength(.{ .matrix = matrix_view }));
    try zynum.matrixVectorMultiplyWithWorkspace(.{
        .matrix = matrix_view,
        .input_vector = input_vector,
        .result_vector = result_vector,
        .workspace = &workspace_data,
    });

    try std.testing.expectApproxEqAbs(@as(f64, 50), vector_data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 80), vector_data[1], 1e-12);
}

test "modern matrix multiply rejects aliasing without workspace" {
    if (builtin.mode == .fast) return error.SkipZigTest;

    var left_data = [_]f64{ 1, 2, 3, 4 };
    var right_data = [_]f64{ 5, 6, 7, 8 };
    const left_matrix = try zynum.constMatrix(f64, &left_data, .{ .row_count = 2, .column_count = 2 });
    const right_matrix = try zynum.constMatrix(f64, &right_data, .{ .row_count = 2, .column_count = 2 });
    const result_matrix = try zynum.matrix(f64, &left_data, .{ .row_count = 2, .column_count = 2 });

    try std.testing.expectError(error.AliasingNotAllowed, zynum.matrixMultiply(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
    }));
}

test "modern matrix multiply workspace supports in-place result" {
    var left_data = [_]f64{ 1, 2, 3, 4 };
    var right_data = [_]f64{ 5, 6, 7, 8 };
    var workspace_data = [_]f64{ 0, 0, 0, 0 };
    const left_matrix = try zynum.constMatrix(f64, &left_data, .{ .row_count = 2, .column_count = 2 });
    const right_matrix = try zynum.constMatrix(f64, &right_data, .{ .row_count = 2, .column_count = 2 });
    const result_matrix = try zynum.matrix(f64, &left_data, .{ .row_count = 2, .column_count = 2 });

    try std.testing.expectEqual(@as(usize, 4), try zynum.matrixMultiplyWorkspaceLength(.{ .result_matrix = result_matrix }));
    try zynum.matrixMultiplyWithWorkspace(.{
        .left_matrix = left_matrix,
        .right_matrix = right_matrix,
        .result_matrix = result_matrix,
        .workspace = &workspace_data,
    });

    try std.testing.expectApproxEqAbs(@as(f64, 23), left_data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 34), left_data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 31), left_data[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 46), left_data[3], 1e-12);
}

fn expectDeclarationAlias(comptime Facade: type, comptime Owner: type, comptime name: []const u8) !void {
    try std.testing.expect(@hasDecl(Facade, name));
    try std.testing.expect(@TypeOf(@field(Facade, name)) == @TypeOf(@field(Owner, name)));
    if (comptime @TypeOf(@field(Owner, name)) == type) {
        try std.testing.expect(@field(Facade, name) == @field(Owner, name));
    }
}

fn expectFacadeAliases(comptime Facade: type, comptime Owner: type) !void {
    const facade_decls = @typeInfo(Facade).@"struct".decl_names;
    const owner_decls = @typeInfo(Owner).@"struct".decl_names;
    try std.testing.expectEqual(owner_decls.len, facade_decls.len);
    inline for (owner_decls) |name| try expectDeclarationAlias(Facade, Owner, name);
    inline for (facade_decls) |name| try std.testing.expect(@hasDecl(Owner, name));
}

test "simultaneous public facades preserve complete declaration and type aliases" {
    @setEvalBranchQuota(20000);
    try expectFacadeAliases(blas, zynum.blas);
    try expectFacadeAliases(lapack, zynum.lapack);
    try expectFacadeAliases(sparse, zynum.sparse);
    // The top facade adds three numerical namespaces to the complete BLAS
    // surface. No implementation-owner import is needed by this consumer.
    const top_decls = @typeInfo(zynum).@"struct".decl_names;
    const blas_decls = @typeInfo(blas).@"struct".decl_names;
    try std.testing.expectEqual(blas_decls.len + 3, top_decls.len);
    inline for (blas_decls) |name| try expectDeclarationAlias(zynum, blas, name);
    inline for (.{ f32, f64, zynum.ComplexF32, zynum.ComplexF64 }) |T| {
        try std.testing.expect(zynum.Vector(T) == blas.Vector(T));
        try std.testing.expect(zynum.ConstVector(T) == blas.ConstVector(T));
        try std.testing.expect(zynum.Matrix(T) == blas.Matrix(T));
        try std.testing.expect(zynum.ConstMatrix(T) == blas.ConstMatrix(T));
        try std.testing.expect(blas.Matrix(T) == lapack.Matrix(T));
        try std.testing.expect(blas.ConstMatrix(T) == lapack.ConstMatrix(T));
        try std.testing.expect(sparse.Matrix(T) == zynum.sparse.Matrix(T));
        try std.testing.expect(sparse.ConstMatrix(T) == zynum.sparse.ConstMatrix(T));
        inline for (.{ i32, u32, i64, u64 }) |I| {
            try std.testing.expect(sparse.ConstCsrMatrix(T, I) == zynum.sparse.ConstCsrMatrix(T, I));
            try std.testing.expect(sparse.ConstCscMatrix(T, I) == zynum.sparse.ConstCscMatrix(T, I));
            try std.testing.expect(sparse.ConstCooMatrix(T, I) == zynum.sparse.ConstCooMatrix(T, I));
        }
    }
    try std.testing.expect(zynum.ComplexF32 == blas.ComplexF32 and blas.ComplexF32 == lapack.ComplexF32 and lapack.ComplexF32 == sparse.ComplexF32);
    try std.testing.expect(zynum.ComplexF64 == blas.ComplexF64 and blas.ComplexF64 == lapack.ComplexF64 and lapack.ComplexF64 == sparse.ComplexF64);
    // Sparse dense geometry intentionally differs from BLAS/LAPACK. A shared
    // owner does not equate distinct storage and view contracts.
    try std.testing.expect(sparse.Matrix(f64) != blas.Matrix(f64));
}

test "simultaneous public facades share and restore runtime settings bidirectionally" {
    try std.testing.expect(zynum.runtime == blas.runtime);
    try std.testing.expect(zynum.runtime == zynum.blas.runtime);
    try std.testing.expect(!@hasDecl(lapack, "runtime"));
    try std.testing.expect(!@hasDecl(sparse, "runtime"));
    const before = zynum.runtime.maxThreadsOverride();
    {
        defer zynum.runtime.setMaxThreads(before);
        zynum.runtime.setMaxThreads(2);
        try std.testing.expectEqual(@as(usize, 2), blas.runtime.maxThreadsOverride());
        try std.testing.expectEqual(@as(usize, 2), zynum.blas.runtime.maxThreadsOverride());
        blas.runtime.setMaxThreads(7);
        try std.testing.expectEqual(@as(usize, 7), zynum.runtime.maxThreadsOverride());
        try std.testing.expectEqual(@as(usize, 7), zynum.blas.runtime.maxThreadsOverride());
        zynum.blas.runtime.setMaxThreads(0);
        try std.testing.expectEqual(@as(usize, 0), zynum.runtime.maxThreadsOverride());
        try std.testing.expectEqual(@as(usize, 0), blas.runtime.maxThreadsOverride());
    }
    try std.testing.expectEqual(before, zynum.runtime.maxThreadsOverride());
    try std.testing.expectEqual(before, blas.runtime.maxThreadsOverride());
}

test "simultaneous public facades interchange matching views in BLAS LAPACK and Sparse calls" {
    var x = [_]f64{ 1, 2 };
    var y = [_]f64{ 3, 4 };
    const input = try zynum.constVector(f64, &x, .{});
    const output = try blas.vector(f64, &y, .{});
    try blas.addScaledVector(.{ .source_vector = input, .destination_vector = output, .scale = 2 });
    try std.testing.expectEqualSlices(f64, &.{ 5, 8 }, &y);
    var a = [_]f64{ 4, 0, 0, 9 };
    var rhs = [_]f64{ 8, 27 };
    const factor = try blas.matrix(f64, &a, .{ .row_count = 2, .column_count = 2 });
    const b = try zynum.matrix(f64, &rhs, .{ .row_count = 2, .column_count = 1 });
    try std.testing.expectEqual(@as(usize, 0), try lapack.choleskyFactorize(f64, factor, .lower));
    try std.testing.expectEqual(@as(usize, 0), try zynum.lapack.choleskySolve(f64, factor.asConst(), b, .lower));
    try std.testing.expectEqualSlices(f64, &.{ 2, 3 }, &rhs);
    const C = zynum.ComplexF64;
    const values = [_]C{ .{ .re = 2, .im = 1 }, .{ .re = 3, .im = -1 } };
    const matrix = try sparse.constCsrMatrix(C, u64, &.{ 0, 1, 2 }, &.{ 0, 1 }, &values, .{ .row_count = 2, .column_count = 2 });
    var product: [2]C = undefined;
    try zynum.sparse.matrixVectorMultiply(.{ .matrix = matrix, .input_vector = &[_]lapack.ComplexF64{ .{ .re = 1, .im = 0 }, .{ .re = 2, .im = 0 } }, .result_vector = &product });
    try std.testing.expectEqualDeep([_]C{ .{ .re = 2, .im = 1 }, .{ .re = 6, .im = -2 } }, product);
}

test "simultaneous public facades execute the selected vector kernel" {
    const tier = blas.runtime.selectedKernelTier();
    try std.testing.expect(tier.len != 0);
    try std.testing.expectEqualStrings(tier, zynum.runtime.selectedKernelTier());
    var x: [257]f64 = undefined;
    var y: [257]f64 = undefined;
    var expected: f64 = 0;
    for (0..257) |k| {
        x[k] = @as(f64, @floatFromInt(k % 7)) - 3;
        y[k] = @as(f64, @floatFromInt(k % 5)) - 2;
        expected += x[k] * y[k];
    }
    const xv = try blas.constVector(f64, &x, .{});
    const yv = try zynum.constVector(f64, &y, .{});
    try std.testing.expectEqual(expected, try blas.dotProduct(.{ .left_vector = xv, .right_vector = yv }));
}

test "checked GEMV workspace preserves validation precedence strided aliasing and beta zero storage" {
    var a = [_]f64{ 2, 1, 3, 101, 101, 4, 2, 1, 101, 101, 1, 5, 2, 101, 101, 103 };
    const a_before = a;
    const matrix_view = try zynum.constMatrix(f64, &a, .{ .row_count = 3, .column_count = 3, .leading_dimension = 5 });
    var shared = [_]f64{ 2, 107, 3, 107, 5, 109 };
    var workspace = [_]f64{ 113, 0, 0, 0, 127 };
    const input_view = try zynum.constVector(f64, &shared, .{ .length = 3, .stride = -2 });
    const result_view = try zynum.vector(f64, &shared, .{ .length = 3, .stride = -2 });
    var invalid_input = input_view;
    invalid_input.stride = 0;
    const before_rejection = shared;
    try std.testing.expectError(error.InvalidStride, zynum.matrixVectorMultiplyWithWorkspace(.{
        .matrix = matrix_view,
        .input_vector = invalid_input,
        .result_vector = result_view,
        .workspace = workspace[1..1],
    }));
    try std.testing.expectError(error.WorkspaceTooSmall, zynum.matrixVectorMultiplyWithWorkspace(.{
        .matrix = matrix_view,
        .input_vector = input_view,
        .result_vector = result_view,
        .workspace = workspace[1..3],
    }));
    try std.testing.expectEqualSlices(f64, &before_rejection, &shared);
    try std.testing.expectEqualSlices(f64, &.{ 113, 0, 0, 0, 127 }, &workspace);
    for ([_]f64{ -3, 0 }) |beta| {
        shared = before_rejection;
        @memset(workspace[1..4], std.math.nan(f64));
        var expected = before_rejection;
        for (0..3) |i| {
            var sum: i64 = 0;
            for (0..3) |j| sum += @as(i64, @intFromFloat(a_before[i + 5 * j])) * @as(i64, @intFromFloat(before_rejection[4 - 2 * j]));
            expected[4 - 2 * i] = 2 * @as(f64, @floatFromInt(sum)) + beta * before_rejection[4 - 2 * i];
        }
        try zynum.matrixVectorMultiplyWithWorkspace(.{
            .matrix = matrix_view,
            .input_vector = input_view,
            .result_vector = result_view,
            .workspace = workspace[1..4],
            .product_scale = @as(f64, 2),
            .result_scale = beta,
        });
        try std.testing.expectEqualSlices(f64, &expected, &shared);
        for (0..3) |i| try std.testing.expectEqual(expected[4 - 2 * i], workspace[1 + i]);
        try std.testing.expectEqual(@as(f64, 113), workspace[0]);
        try std.testing.expectEqual(@as(f64, 127), workspace[4]);
        try std.testing.expectEqualSlices(f64, &a_before, &a);
    }
    const before_empty = shared;
    const workspace_before_empty = workspace;
    const empty_matrix = try zynum.constMatrix(f64, &.{}, .{ .row_count = 0, .column_count = 3 });
    const empty_result = try zynum.vector(f64, shared[0..0], .{});
    try zynum.matrixVectorMultiplyWithWorkspace(.{
        .matrix = empty_matrix,
        .input_vector = input_view,
        .result_vector = empty_result,
        .workspace = workspace[1..1],
        .result_scale = @as(f64, -3),
    });
    try std.testing.expectEqualSlices(f64, &before_empty, &shared);
    try std.testing.expectEqualSlices(f64, &workspace_before_empty, &workspace);
}

fn level1ModernValue(comptime T: type, re: f64, im: f64) T {
    if (T == f32 or T == f64) return @floatCast(re);
    return .{ .re = @floatCast(re), .im = @floatCast(im) };
}

fn level1ModernInto(comptime T: type, comptime kind: enum { scale, add, combine }, stride: blas.BlasInt) !void {
    const complex = T == blas.ComplexF32 or T == blas.ComplexF64;
    var x: [7]T = @splat(level1ModernValue(T, 97, 43));
    var y: [7]T = @splat(level1ModernValue(T, 101, 47));
    var output: [7]T = @splat(level1ModernValue(T, 103, 53));
    for (0..3) |i| {
        x[1 + 2 * i] = level1ModernValue(T, @floatFromInt(i + 1), 1);
        y[1 + 2 * i] = level1ModernValue(T, @floatFromInt(i + 4), -2);
    }
    const x_before = x;
    const y_before = y;
    var expected = output;
    for (0..3) |i| {
        const a: f64 = @floatFromInt(i + 1);
        const b: f64 = @floatFromInt(i + 4);
        expected[1 + 2 * i] = switch (kind) {
            .scale => level1ModernValue(T, 2 * b, if (complex) -4 else 0),
            .add => level1ModernValue(T, b + 2 * a, 0),
            .combine => level1ModernValue(T, 2 * a - 3 * b, if (complex) 8 else 0),
        };
    }
    const source = try blas.constVector(T, x[1..6], .{ .length = 3, .stride = stride });
    const input = try blas.constVector(T, y[1..6], .{ .length = 3, .stride = stride });
    const result = try blas.vector(T, output[1..6], .{ .length = 3, .stride = stride });
    const two = level1ModernValue(T, 2, 0);
    const minus_three = level1ModernValue(T, -3, 0);
    switch (kind) {
        .scale => try zynum.scaleVectorInto(.{ .input_vector = input, .result_vector = result, .scale = two }),
        .add => try zynum.addScaledVectorInto(.{ .source_vector = source, .input_vector = input, .result_vector = result, .scale = two }),
        .combine => try zynum.combineVectorsInto(.{ .source_vector = source, .input_vector = input, .result_vector = result, .source_scale = two, .input_scale = minus_three }),
    }
    try std.testing.expectEqualDeep(expected, output);
    try std.testing.expectEqualDeep(x_before, x);
    try std.testing.expectEqualDeep(y_before, y);
    // The exact-alias branch must use the existing borrowed validation helper,
    // not call the public entry point a second time.
    const in_place = try blas.vector(T, y[1..6], .{ .length = 3, .stride = stride });
    switch (kind) {
        .scale => try zynum.scaleVectorInto(.{ .input_vector = input, .result_vector = in_place, .scale = two }),
        .add => try zynum.addScaledVectorInto(.{ .source_vector = source, .input_vector = input, .result_vector = in_place, .scale = two }),
        .combine => try zynum.combineVectorsInto(.{ .source_vector = source, .input_vector = input, .result_vector = in_place, .source_scale = two, .input_scale = minus_three }),
    }
    for (0..3) |i| try std.testing.expectEqualDeep(expected[1 + 2 * i], y[1 + 2 * i]);
    for ([_]usize{ 0, 2, 4, 6 }) |i| try std.testing.expectEqualDeep(y_before[i], y[i]);
    try std.testing.expectEqualDeep(x_before, x);
}

test "checked Level 1 Into preserves four scalar strided storage and exact alias composition" {
    inline for (.{ f32, f64, blas.ComplexF32, blas.ComplexF64 }) |T| {
        inline for (.{ .scale, .add, .combine }) |kind| {
            try level1ModernInto(T, kind, 2);
            try level1ModernInto(T, kind, -2);
        }
    }
}

test "checked Level 1 preserves validation order empty and partial alias rejection" {
    var data = [_]f64{ 2, 3, 5, 7, 11, 13 };
    const before = data;
    const valid = try blas.vector(f64, &data, .{ .length = 3 });
    const input = try blas.constVector(f64, &data, .{ .length = 3 });
    var invalid = valid;
    invalid.length = -1;
    invalid.stride = 0;
    // required vector extent checks length before stride, in every build mode.
    try std.testing.expectError(error.InvalidLength, zynum.scaleVector(.{ .target_vector = invalid, .scale = @as(f64, 2) }));
    try std.testing.expectError(error.InvalidLength, zynum.scaleVectorInto(.{ .input_vector = input, .result_vector = invalid, .scale = @as(f64, 2) }));
    const empty = try blas.vector(f64, data[0..0], .{});
    const empty_input = try blas.constVector(f64, data[0..0], .{});
    try zynum.copyVector(.{ .source_vector = empty_input, .destination_vector = empty });
    try zynum.swapVectors(.{ .first_vector = empty, .second_vector = empty });
    try zynum.scaleVectorInto(.{ .input_vector = empty_input, .result_vector = empty, .scale = @as(f64, 2) });
    try std.testing.expectEqual(@as(f64, 0), try zynum.dotProduct(.{ .left_vector = empty_input, .right_vector = empty_input }));
    try std.testing.expectEqual(@as(f64, 0), try zynum.euclideanNorm(.{ .input_vector = empty_input }));
    if (builtin.mode != .fast) {
        const overlap = try blas.vector(f64, data[1..4], .{});
        try std.testing.expectError(error.AliasingNotAllowed, zynum.copyVector(.{ .source_vector = input, .destination_vector = overlap }));
        try std.testing.expectError(error.AliasingNotAllowed, zynum.addScaledVector(.{ .source_vector = input, .destination_vector = overlap, .scale = @as(f64, 2) }));
        try std.testing.expectError(error.AliasingNotAllowed, zynum.combineVectorsInto(.{ .source_vector = input, .input_vector = input, .result_vector = overlap, .source_scale = @as(f64, 2), .input_scale = @as(f64, -3) }));
    }
    try std.testing.expectEqualDeep(before, data);
}
