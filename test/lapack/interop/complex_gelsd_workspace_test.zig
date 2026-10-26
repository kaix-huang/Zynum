// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const owner = @import("numerical-owner");
const workspace = owner.complex_gelsd_workspace;
const Core = owner.Provider;
const Rank = Core.SvdLeastSquares;
const P = workspace.Planner(Core);
const C32 = Core.Scalar.ComplexF32;
const C64 = Core.Scalar.ComplexF64;
fn marked(bytes: []const u8) !void {
    for (bytes) |byte| try std.testing.expectEqual(@as(u8, 0xa5), byte);
}
fn checkLayout(comptime C: type, comptime I: type, m: usize, n: usize, rhs: usize) !void {
    const W = Rank.WorkspaceScalar(C);
    const plan = try P.make(C, I, m, n, rhs);
    const count: usize = @intCast(plan.work);
    const byte_count = count * @sizeOf(C);
    const memory = try std.testing.allocator.alignedAlloc(u8, .of(W), byte_count + 6 * @alignOf(W));
    defer std.testing.allocator.free(memory);
    // Offsetting complex[] by whole elements never changes its wider-alignment
    // residue: sizeof(C)==alignof(W). Start at both native-aligned byte residues.
    var residue: usize = 0;
    while (residue < @alignOf(W)) : (residue += @alignOf(C)) {
        @memset(memory, 0xa5);
        const start = 2 * @alignOf(W) + residue;
        const pointer: [*]C = @ptrCast(@alignCast(memory.ptr + start));
        const work = pointer[0 .. count + 2];
        const before = try std.testing.allocator.dupe(u8, memory);
        defer std.testing.allocator.free(before);
        try std.testing.expectError(error.BufferTooSmall, P.split(C, I, plan, work[0 .. count - 1]));
        try std.testing.expectEqualSlices(u8, before, memory);
        const buffers = try P.split(C, I, plan, work[0..count]);
        try std.testing.expectEqualSlices(u8, before, memory);
        try std.testing.expectEqual(plan.scalar, buffers.scalar.len);
        try std.testing.expectEqual(plan.solution, buffers.solution.len);
        try std.testing.expectEqual(plan.residual, buffers.residual.len);
        try std.testing.expectEqual(plan.right_vectors, buffers.right_vectors.len);
        try std.testing.expectEqual(plan.rhs, buffers.rhs.len);
        try std.testing.expectEqual(plan.wide, buffers.wide.len);
        try std.testing.expectEqual(plan.real, buffers.real.len);
        const base = @intFromPtr(work.ptr);
        const scalar_start = @intFromPtr(buffers.scalar.ptr);
        const solution_start = @intFromPtr(buffers.solution.ptr);
        const residual_start = @intFromPtr(buffers.residual.ptr);
        const vectors_start = @intFromPtr(buffers.right_vectors.ptr);
        const rhs_start = @intFromPtr(buffers.rhs.ptr);
        const wide_start = @intFromPtr(buffers.wide.ptr);
        const real_start = @intFromPtr(buffers.real.ptr);
        const native_end = rhs_start + buffers.rhs.len * @sizeOf(C);
        const wide_end = wide_start + buffers.wide.len * @sizeOf(W);
        const real_end = real_start + buffers.real.len * @sizeOf(Rank.Wide(C));
        try std.testing.expectEqual(base + @sizeOf(C), scalar_start);
        try std.testing.expectEqual(scalar_start + buffers.scalar.len * @sizeOf(C), solution_start);
        try std.testing.expectEqual(solution_start + buffers.solution.len * @sizeOf(C), residual_start);
        try std.testing.expectEqual(residual_start + buffers.residual.len * @sizeOf(C), vectors_start);
        try std.testing.expectEqual(vectors_start + buffers.right_vectors.len * @sizeOf(C), rhs_start);
        try std.testing.expectEqual(@as(usize, 0), wide_start % @alignOf(W));
        try std.testing.expect(wide_start >= native_end);
        try std.testing.expect(wide_start - native_end < @alignOf(W));
        try std.testing.expectEqual(wide_end, real_start);
        try std.testing.expectEqual(@as(usize, 0), real_start % @alignOf(Rank.Wide(C)));
        try std.testing.expect(real_end <= base + byte_count);
        // Actual caller scratch use must leave metadata, byte padding, over-query
        // reserve and every prefix/suffix byte unchanged at both base residues.
        for (buffers.scalar) |*value| value.* = .{ .re = 1, .im = -2 };
        for (buffers.solution) |*value| value.* = .{ .re = 3, .im = -4 };
        for (buffers.residual) |*value| value.* = .{ .re = 5, .im = -6 };
        for (buffers.right_vectors) |*value| value.* = .{ .re = 9, .im = -10 };
        for (buffers.rhs) |*value| value.* = .{ .re = 13, .im = -14 };
        for (buffers.wide) |*value| value.* = .{ .re = 7, .im = -8 };
        for (buffers.real) |*value| value.* = 11;
        try marked(memory[0 .. start + @sizeOf(C)]);
        try marked(memory[start + native_end - base .. start + wide_start - base]);
        try marked(memory[start + real_end - base ..]);
        // Oversized caller slices do not expand the layout beyond Plan.work.
        const oversized = try P.split(C, I, plan, work);
        try std.testing.expectEqual(wide_start, @intFromPtr(oversized.wide.ptr));
        try std.testing.expectEqual(real_start, @intFromPtr(oversized.real.ptr));
    }
}
test "complex GELSD own recommendation matches accepted native rank plan and full complex alignment reserve" {
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| {
        for ([_][3]usize{ .{ 3, 2, 2 }, .{ 2, 3, 2 }, .{ 2, 2, 0 }, .{ 7, 1, 3 } }) |shape| {
            const native = try Rank.plan(C, shape[0], shape[1], shape[2]);
            const plan = try P.make(C, I, shape[0], shape[1], shape[2]);
            try std.testing.expectEqual(native.scalar_elements, plan.scalar);
            try std.testing.expectEqual(native.computational_elements, plan.wide);
            try std.testing.expectEqual(native.real_elements, plan.real);
            try std.testing.expectEqual(shape[1] * shape[2], plan.solution);
            try std.testing.expectEqual((shape[0] -| shape[1]) * shape[2], plan.residual);
            try std.testing.expectEqual(shape[1] * @min(shape[0], shape[1]), plan.right_vectors);
            try std.testing.expectEqual(shape[0] * shape[2], plan.rhs);
            const exact_count = 1 + plan.scalar + plan.solution + plan.residual + plan.right_vectors + plan.rhs + 1 + 2 * plan.wide + plan.real;
            const rounded = try workspace.queryLength(Core.Scalar.Real(C), I, exact_count);
            try std.testing.expectEqual(@as(I, @intFromFloat(rounded)), plan.work);
            try std.testing.expectEqual(@as(I, 1), plan.rwork);
            try std.testing.expectEqual(@as(I, 1), plan.iwork);
        }
    };
}
test "complex GELSD layout is bounded at both native aligned base residues for C32" {
    inline for (.{ i32, i64 }) |I| try checkLayout(C32, I, 3, 2, 2);
}
test "complex GELSD layout is bounded at both native aligned base residues for C64" {
    inline for (.{ i32, i64 }) |I| try checkLayout(C64, I, 3, 2, 2);
}
test "complex GELSD wide and zero rhs layouts preserve metadata byte padding and suffix" {
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| {
        try checkLayout(C, I, 2, 3, 2);
        try checkLayout(C, I, 3, 2, 0);
    };
}
test "complex GELSD empty dimensions use only one own query element even with huge other dimensions" {
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| {
        const bound: usize = std.math.maxInt(I);
        for ([_][3]usize{ .{ 0, bound, bound }, .{ bound, 0, bound }, .{ 0, 0, 0 } }) |shape| {
            const plan = try P.make(C, I, shape[0], shape[1], shape[2]);
            try std.testing.expectEqual(@as(I, 1), plan.work);
            try std.testing.expectEqual(@as(I, 1), plan.rwork);
            try std.testing.expectEqual(@as(I, 1), plan.iwork);
            try std.testing.expectEqual(@as(usize, 0), plan.scalar + plan.solution + plan.residual + plan.right_vectors + plan.rhs + plan.wide + plan.real);
            var work = [_]C{ .{ .re = 91, .im = -17 }, .{ .re = 92, .im = -18 } };
            const buffers = try P.split(C, I, plan, &work);
            try std.testing.expectEqual(@as(usize, 0), buffers.scalar.len + buffers.solution.len + buffers.residual.len + buffers.right_vectors.len + buffers.rhs.len + buffers.wide.len + buffers.real.len);
            try std.testing.expectEqual(@as(Core.Scalar.Real(C), 91), work[0].re);
            try std.testing.expectEqual(@as(Core.Scalar.Real(C), -17), work[0].im);
            try std.testing.expectEqual(@as(Core.Scalar.Real(C), 92), work[1].re);
            try std.testing.expectError(error.BufferTooSmall, P.split(C, I, plan, work[0..0]));
        }
    };
}
test "complex GELSD floating query rounds upward and respects signed width conversion boundaries" {
    try std.testing.expectEqual(@as(f32, 16777218), try workspace.queryLength(f32, i32, 16777217));
    try std.testing.expectEqual(@as(f64, 9007199254740994), try workspace.queryLength(f64, i64, 9007199254740993));
    inline for (.{ f32, f64 }) |R| {
        try std.testing.expectEqual(@as(R, 0), try workspace.queryLength(R, i32, 0));
        try std.testing.expectEqual(@as(R, 1), try workspace.queryLength(R, i64, 1));
        try std.testing.expectError(error.SizeOverflow, workspace.queryLength(R, i32, @as(usize, std.math.maxInt(i32)) + 1));
        try std.testing.expectError(error.SizeOverflow, workspace.queryLength(R, i64, @as(usize, std.math.maxInt(i64)) + 1));
    }
    try std.testing.expectError(error.SizeOverflow, workspace.queryLength(f32, i32, std.math.maxInt(i32)));
    try std.testing.expectError(error.SizeOverflow, workspace.queryLength(f64, i64, std.math.maxInt(i64)));
    try std.testing.expectEqual(@as(f64, 2147483647), try workspace.queryLength(f64, i32, std.math.maxInt(i32)));
}
test "complex GELSD dimensions and native Core byte counts are checked before caller memory access" {
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| {
        const bound: usize = std.math.maxInt(I);
        try std.testing.expectError(error.SizeOverflow, P.make(C, I, bound + 1, 0, 0));
        try std.testing.expectError(error.SizeOverflow, P.make(C, I, 0, bound + 1, 0));
        try std.testing.expectError(error.SizeOverflow, P.make(C, I, 0, 0, bound + 1));
        try std.testing.expectError(error.SizeOverflow, P.make(C, I, bound, bound, 1));
        try std.testing.expectError(error.SizeOverflow, P.make(C, I, 1, bound, bound));
    };
}
// Isolate count arithmetic at large query boundaries without allocating giant
// matrices. Numerical and real native planning tests above use the actual Core.
const CountCore = struct {
    pub const Scalar = Core.Scalar;
    pub const SvdLeastSquares = struct {
        pub const Wide = Rank.Wide;
        pub const WorkspaceScalar = Rank.WorkspaceScalar;
        pub fn plan(comptime C: type, m: usize, n: usize, rhs: usize) !struct { scalar_elements: usize, computational_elements: usize, real_elements: usize } {
            _ = C;
            _ = n;
            _ = rhs;
            return .{ .scalar_elements = m, .computational_elements = 0, .real_elements = 0 };
        }
    };
};
test "complex GELSD make keeps upward rounded recommendation distinct from exact used count" {
    const CountP = workspace.Planner(CountCore);
    const f32_plan = try CountP.make(C32, i32, 16777216, 1, 0);
    try std.testing.expectEqual(@as(i32, 16777218), f32_plan.work);
    try std.testing.expectEqual(@as(usize, 16777216), f32_plan.scalar);
    const f64_plan = try CountP.make(C64, i64, 9007199254740992, 1, 0);
    try std.testing.expectEqual(@as(i64, 9007199254740994), f64_plan.work);
    try std.testing.expectError(error.SizeOverflow, CountP.make(C32, i32, std.math.maxInt(i32) - 1, 1, 0));
    inline for (.{ C32, C64 }) |C| try std.testing.expectError(error.SizeOverflow, CountP.make(C, i64, std.math.maxInt(usize) / @sizeOf(C), 1, 0));
}
test "complex GELSD split rejects forged count and widened endpoint overflow without writing" {
    inline for (.{ C32, C64 }) |C| inline for (.{ i32, i64 }) |I| {
        var work: [4]C = @splat(.{ .re = 93, .im = -19 });
        const before = work;
        var plan = try P.make(C, I, 0, 0, 0);
        plan.work = 0;
        try std.testing.expectError(error.BufferTooSmall, P.split(C, I, plan, &work));
        plan.work = -1;
        try std.testing.expectError(error.BufferTooSmall, P.split(C, I, plan, &work));
        plan.work = 4;
        plan.scalar = std.math.maxInt(usize);
        try std.testing.expectError(error.SizeOverflow, P.split(C, I, plan, &work));
        plan.scalar = 0;
        plan.right_vectors = std.math.maxInt(usize);
        try std.testing.expectError(error.SizeOverflow, P.split(C, I, plan, &work));
        plan.right_vectors = 0;
        plan.rhs = std.math.maxInt(usize);
        try std.testing.expectError(error.SizeOverflow, P.split(C, I, plan, &work));
        plan.rhs = 0;
        plan.wide = std.math.maxInt(usize);
        try std.testing.expectError(error.SizeOverflow, P.split(C, I, plan, &work));
        plan.wide = 2;
        try std.testing.expectError(error.BufferTooSmall, P.split(C, I, plan, &work));
        plan.wide = 0;
        plan.real = std.math.maxInt(usize);
        try std.testing.expectError(error.SizeOverflow, P.split(C, I, plan, &work));
        plan.real = 4;
        try std.testing.expectError(error.BufferTooSmall, P.split(C, I, plan, &work));
        try std.testing.expectEqualSlices(C, &before, &work);
    };
}
test "complex GELSD split checks pointer byte endpoint before forming any slice" {
    inline for (.{ C32, C64 }) |C| {
        const plan = try P.make(C, i64, 0, 0, 0);
        const address = std.math.maxInt(usize) & ~@as(usize, @alignOf(C) - 1);
        const pointer: [*]C = @ptrFromInt(address);
        // No dereference: endpoint overflow must return before constructing views.
        try std.testing.expectError(error.SizeOverflow, P.split(C, i64, plan, pointer[0..1]));
    }
}
