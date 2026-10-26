// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Caller-WORK layout for the shared GELSD core. Widened arithmetic is housed
//! inside WORK with explicit alignment padding; no hidden allocator/owner.
const std = @import("std");
pub const Error = error{ SizeOverflow, BufferTooSmall };
fn add(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch error.SizeOverflow;
}
fn mul(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch error.SizeOverflow;
}

/// Round upward so a floating WORK(1) query never understates a required length.
/// The rounded value must itself still convert to the selected signed ABI width.
pub fn queryLength(comptime T: type, comptime I: type, count: usize) Error!T {
    if (T != f32 and T != f64) @compileError("GELSD query requires real scalar");
    if (I != i32 and I != i64) @compileError("GELSD query requires signed ABI integer");
    if (count > std.math.maxInt(I)) return error.SizeOverflow;
    var value: T = @floatFromInt(count);
    if (@as(f128, value) < @as(f128, @floatFromInt(count))) value = std.math.nextAfter(T, value, std.math.inf(T));
    if (@as(f128, value) > @as(f128, @floatFromInt(std.math.maxInt(I)))) return error.SizeOverflow;
    return value;
}
pub fn Planner(comptime Core: type) type {
    return struct {
        pub fn Plan(comptime I: type) type {
            return struct { work: I, iwork: I, scalar: usize, solution: usize, residual: usize, wide: usize };
        }
        pub fn Buffers(comptime T: type) type {
            return struct { scalar: []T, solution: []T, residual: []T, wide: []Core.RankLeastSquares.WorkspaceScalar(T) };
        }
        pub fn make(comptime T: type, comptime I: type, m: usize, n: usize, nrhs: usize) Error!Plan(I) {
            const W = Core.RankLeastSquares.WorkspaceScalar(T);
            if (m > std.math.maxInt(I) or n > std.math.maxInt(I) or nrhs > std.math.maxInt(I)) return error.SizeOverflow;
            if (m == 0 or n == 0) return .{ .work = 1, .iwork = 1, .scalar = 0, .solution = 0, .residual = 0, .wide = 0 };
            const native = Core.RankLeastSquares.planWithResidual(T, m, n, nrhs, true) catch return error.SizeOverflow;
            const solution = try mul(n, nrhs);
            const residual = try mul(m -| n, nrhs);
            var count = try add(1, native.scalar_elements);
            count = try add(count, solution);
            count = try add(count, residual);
            if (native.wide_elements != 0) {
                count = try add(count, @alignOf(W) / @sizeOf(T) - 1);
                count = try add(count, try mul(native.wide_elements, @sizeOf(W) / @sizeOf(T)));
            }
            const query = try queryLength(T, I, count);
            const work: I = @intFromFloat(query);
            _ = try mul(@intCast(work), @sizeOf(T));
            const k = @min(m, n);
            // Reference's INT(log2(k/26))+1 truncates negative fractions toward zero.
            var levels: usize = if (k > 13) 1 else 0;
            var quotient = k / 26;
            while (quotient >= 2) : (quotient /= 2) levels += 1;
            const integers = @max(@as(usize, 1), try add(try mul(try mul(3, k), levels), try mul(11, k)));
            if (integers > std.math.maxInt(I)) return error.SizeOverflow;
            _ = try mul(integers, @sizeOf(I));
            return .{ .work = work, .iwork = @intCast(integers), .scalar = native.scalar_elements, .solution = solution, .residual = residual, .wide = native.wide_elements };
        }
        pub fn split(comptime T: type, comptime I: type, plan: Plan(I), work: []T) Error!Buffers(T) {
            const W = Core.RankLeastSquares.WorkspaceScalar(T);
            if (plan.work < 1 or work.len < @as(usize, @intCast(plan.work))) return error.BufferTooSmall;
            const scalar_end = try add(1, plan.scalar);
            const solution_end = try add(scalar_end, plan.solution);
            const residual_end = try add(solution_end, plan.residual);
            if (residual_end > @as(usize, @intCast(plan.work))) return error.BufferTooSmall;
            var wide: []W = &.{};
            if (plan.wide != 0) {
                const base = @intFromPtr(work.ptr);
                const address = try add(base, try mul(residual_end, @sizeOf(T)));
                const aligned = (try add(address, @alignOf(W) - 1)) & ~@as(usize, @alignOf(W) - 1);
                const end = try add(aligned, try mul(plan.wide, @sizeOf(W)));
                const limit = try add(base, try mul(@intCast(plan.work), @sizeOf(T)));
                if (end > limit) return error.BufferTooSmall;
                const pointer: [*]W = @ptrFromInt(aligned);
                wide = pointer[0..plan.wide];
            }
            return .{ .scalar = work[1..scalar_end], .solution = work[scalar_end..solution_end], .residual = work[solution_end..residual_end], .wide = wide };
        }
    };
}
