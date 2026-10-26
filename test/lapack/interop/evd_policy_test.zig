const std = @import("std");
const Core = @import("zynum-core");
const N = Core.internal.numerics;
const Tri = N.TridiagonalEigen;
const H = N.Hermitian;
const P = @import("numerical-owner").evd_protocol.policy;
const E = std.testing;

test "EVD workspace tables for empty singleton and scalar families" {
    inline for (.{ i32, i64 }) |I| {
        inline for (.{ P.Family.real, P.Family.complex }) |family| inline for (.{ false, true }) |vectors| {
            for ([_]I{ 0, 1 }) |n| {
                const w = try P.minimumWorkspace(I, n, vectors, family);
                try E.expectEqual(@as(I, 1), w.work);
                try E.expectEqual(@as(I, 1), w.rwork);
                try E.expectEqual(@as(I, 1), w.iwork);
            }
        };
        const rows = [_]struct { n: I, vectors: bool, family: P.Family, work: I, real: I, integer: I }{
            .{ .n = 2, .vectors = false, .family = .real, .work = 5, .real = 1, .integer = 1 },
            .{ .n = 2, .vectors = true, .family = .real, .work = 21, .real = 1, .integer = 13 },
            .{ .n = 2, .vectors = false, .family = .complex, .work = 3, .real = 2, .integer = 1 },
            .{ .n = 2, .vectors = true, .family = .complex, .work = 8, .real = 19, .integer = 13 },
            .{ .n = 7, .vectors = false, .family = .real, .work = 15, .real = 1, .integer = 1 },
            .{ .n = 7, .vectors = true, .family = .real, .work = 141, .real = 1, .integer = 38 },
            .{ .n = 7, .vectors = false, .family = .complex, .work = 8, .real = 7, .integer = 1 },
            .{ .n = 7, .vectors = true, .family = .complex, .work = 63, .real = 134, .integer = 38 },
        };
        for (rows) |row| {
            const w = try P.minimumWorkspace(I, row.n, row.vectors, row.family);
            try E.expectEqual(row.work, w.work);
            try E.expectEqual(row.real, w.rwork);
            try E.expectEqual(row.integer, w.iwork);
        }
    }
}

test "EVD any query slot suppresses all workspace length errors in positional order" {
    inline for (.{ i32, i64 }) |I| inline for (.{ P.Family.real, P.Family.complex }) |family| {
        const required = try P.minimumWorkspace(I, 2, true, family);
        const values = [_]I{ -2, -1, 0, 1 };
        for (values) |work| for (values) |real| for (values) |integer| {
            const lengths: P.Workspace(I) = .{ .work = work, .rwork = real, .iwork = integer };
            const query = work == -1 or integer == -1 or (family == .complex and real == -1);
            try E.expectEqual(query, P.isQuery(I, family, lengths));
            try E.expectEqual(@as(I, if (query) 0 else -9), P.workspaceInfo(I, family, lengths, required));
        };
        try E.expectEqual(@as(I, 0), P.workspaceInfo(I, family, required, required));
        var lengths = required;
        lengths.work -= 1;
        lengths.rwork -= 1;
        lengths.iwork -= 1;
        try E.expectEqual(@as(I, -9), P.workspaceInfo(I, family, lengths, required));
        lengths.work = required.work;
        try E.expectEqual(@as(I, -11), P.workspaceInfo(I, family, lengths, required));
        lengths.rwork = required.rwork;
        try E.expectEqual(@as(I, if (family == .complex) -13 else -11), P.workspaceInfo(I, family, lengths, required));
    };
}

test "EVD query scalars round upward and reject unencodable ABI lengths" {
    const values = [_]i32{ 16777215, 16777216, 16777217, 16777218, 16777219, 33554433 };
    for (values) |count| {
        const single = try P.queryValue(f32, i32, count);
        const double = try P.queryValue(f64, i32, count);
        try E.expect(@as(f64, single) >= @as(f64, @floatFromInt(count)));
        try E.expectEqual(@as(f64, @floatFromInt(count)), double);
    }
    try E.expectEqual(@as(f32, 16777218), try P.queryValue(f32, i32, 16777217));
    try E.expectEqual(@as(f32, 33554436), try P.queryValue(f32, i32, 33554433));
    try E.expectError(error.IntegerNotRepresentable, P.queryValue(f32, i32, std.math.maxInt(i32)));
    try E.expectEqual(@as(f64, 2147483647), try P.queryValue(f64, i32, std.math.maxInt(i32)));
    try E.expectEqual(@as(f64, 9007199254740994), try P.queryValue(f64, i64, 9007199254740993));
    try E.expectError(error.IntegerNotRepresentable, P.queryValue(f64, i64, std.math.maxInt(i64)));
    try E.expectError(error.InvalidDimension, P.queryValue(f64, i32, 0));
}

test "EVD workspace arithmetic covers LP64 ILP64 quadratic limits without wrapping" {
    inline for (.{ P.Family.real, P.Family.complex }) |family| {
        const small = try P.minimumWorkspace(i32, 32766, true, family);
        try E.expect(small.work > 0 and small.rwork > 0 and small.iwork > 0);
        try E.expectError(error.IntegerNotRepresentable, P.minimumWorkspace(i32, 32767, true, family));
        const wide = try P.minimumWorkspace(i64, 2147483646, true, family);
        try E.expect(wide.work > 0 and wide.rwork > 0 and wide.iwork > 0);
        try E.expectError(error.IntegerNotRepresentable, P.minimumWorkspace(i64, 2147483647, true, family));
        try E.expectError(error.IntegerNotRepresentable, P.minimumWorkspace(i64, std.math.maxInt(i64), true, family));
        try E.expectError(error.InvalidDimension, P.minimumWorkspace(i32, -1, false, family));
    }
    try E.expectEqual(@as(i32, 2147418109), (try P.minimumWorkspace(i32, 32766, true, .real)).work);
    try E.expectEqual(@as(i32, 2147385343), (try P.minimumWorkspace(i32, 32766, true, .complex)).rwork);
    try E.expectEqual(std.math.maxInt(i32), (try P.minimumWorkspace(i32, 1073741823, false, .real)).work);
    try E.expectEqual(std.math.maxInt(i64), (try P.minimumWorkspace(i64, 4611686018427387903, false, .real)).work);
}

test "EVD actual tridiagonal failure identifies a disconnected interior block" {
    inline for (.{ f32, f64 }) |T| inline for (.{ false, true }) |vectors| {
        var d = [_]T{ 11, 13, 4, -1, 5, 17, 19 };
        var e = [_]T{ 0, 0, 2, 3, 0, 0, 97 };
        const od = d;
        const oe = e;
        var z: [49]T = @splat(73);
        const matrix: N.Dense.Matrix(T) = .{ .values = &z, .row_count = 7, .column_count = 7, .leading_dimension = 7 };
        const result = Tri.solve(T, &d, &e, if (vectors) matrix else null, .{ .max_iterations = 0 });
        try E.expectEqual(Tri.Status.iteration_limit, result.status);
        try E.expectEqual(@as(usize, 0), result.iterations);
        try E.expectEqual(@as(usize, 2), result.unconverged);
        const block = result.failed_block.?;
        try E.expectEqual(@as(usize, 2), block.first);
        try E.expectEqual(@as(usize, 5), block.limit);
        try E.expectEqual(@as(i32, 2), try P.iterationInfo(i32, 7, false, result.unconverged, result.failed_block));
        try E.expectEqual(@as(i32, 29), try P.iterationInfo(i32, 7, true, result.unconverged, result.failed_block));
        try E.expectEqualSlices(T, &od, &d);
        try E.expectEqualSlices(T, &oe, &e);
    };
}

test "EVD failed block survives the existing complex Hermitian numerical owner" {
    inline for (.{ N.Scalar.ComplexF32, N.Scalar.ComplexF64 }) |C| {
        const R = N.Scalar.Real(C);
        var a: [49]C = @splat(.{ .re = 0, .im = 0 });
        const diagonal = [_]R{ 11, 13, 4, -1, 5, 17, 19 };
        for (diagonal, 0..) |x, i| a[i + i * 7] = .{ .re = x, .im = 0 };
        a[3 + 2 * 7] = .{ .re = 2, .im = 0 };
        a[4 + 3 * 7] = .{ .re = 3, .im = 0 };
        const original = a;
        var values: [7]R = undefined;
        var vectors: [49]C = undefined;
        var work: [196]H.core.WorkspaceScalar(C) = undefined;
        const result = try H.core.solveWithContract(C, .{ .values = &a, .row_count = 7, .column_count = 7, .leading_dimension = 7 }, .lower, &values, .{ .values = &vectors, .row_count = 7, .column_count = 7, .leading_dimension = 7 }, &work, .{ .max_sweeps = 0 }, .heev_ieee_v1);
        try E.expectEqual(H.core.Status.iteration_limit, result.status);
        try E.expectEqual(@as(usize, 2), result.unconverged);
        try E.expectEqual(@as(usize, 2), result.failed_block.?.first);
        try E.expectEqual(@as(usize, 5), result.failed_block.?.limit);
        try E.expectEqual(@as(i32, 29), try P.iterationInfo(i32, 7, true, result.unconverged, result.failed_block));
        try E.expectEqualSlices(C, &original, &a);
    }
}

test "EVD positive INFO encoder rejects absent invalid or unrepresentable blocks" {
    const Block = Tri.UnconvergedBlock;
    const absent: ?Block = null;
    try E.expectEqual(@as(i32, 0), try P.iterationInfo(i32, 0, true, 0, absent));
    try E.expectEqual(@as(i32, 2), try P.iterationInfo(i32, 7, false, 2, absent));
    try E.expectError(error.InvalidOutcome, P.iterationInfo(i32, 7, true, 2, absent));
    for ([_]Block{ .{ .first = 5, .limit = 3 }, .{ .first = 2, .limit = 3 }, .{ .first = 0, .limit = 8 } }) |bad| {
        try E.expectError(error.InvalidOutcome, P.iterationInfo(i32, 7, true, 2, @as(?Block, bad)));
    }
    try E.expectError(error.InvalidOutcome, P.iterationInfo(i32, 7, false, 7, absent));
    try E.expectError(error.InvalidDimension, P.iterationInfo(i32, -1, false, 0, absent));
    const last = std.math.maxInt(i64);
    const large: Block = .{ .first = last - 2, .limit = last };
    try E.expectError(error.IntegerNotRepresentable, P.iterationInfo(i64, last, true, 1, @as(?Block, large)));
}

test "EVD converged existing solvers leave failure metadata absent" {
    inline for (.{ f32, f64 }) |T| {
        var d = [_]T{ 2, 2 };
        var e = [_]T{1};
        const result = Tri.solve(T, &d, &e, null, .{});
        try E.expectEqual(Tri.Status.converged, result.status);
        try E.expect(result.failed_block == null);
        try E.expectApproxEqAbs(@as(T, 1), d[0], 40 * std.math.floatEps(T));
        try E.expectApproxEqAbs(@as(T, 3), d[1], 40 * std.math.floatEps(T));
    }
    inline for (.{ N.Scalar.ComplexF32, N.Scalar.ComplexF64 }) |C| {
        const R = N.Scalar.Real(C);
        const a = [_]C{ .{ .re = 2, .im = 0 }, .{ .re = 1, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = 2, .im = 0 } };
        var values: [2]R = undefined;
        var work: [16]H.core.WorkspaceScalar(C) = undefined;
        const result = try H.core.solveWithContract(C, .{ .values = &a, .row_count = 2, .column_count = 2, .leading_dimension = 2 }, .lower, &values, null, &work, .{}, .checked_finite);
        try E.expectEqual(H.core.Status.converged, result.status);
        try E.expect(result.failed_block == null);
        try E.expectApproxEqAbs(@as(R, 1), values[0], 40 * std.math.floatEps(R));
        try E.expectApproxEqAbs(@as(R, 3), values[1], 40 * std.math.floatEps(R));
    }
}
