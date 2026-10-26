// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Own complex GELSD recommended caller-WORK layout, not a vendor minimum.
//! WORK[0] is reserved metadata; shared SVD scratch, solution, residual, V and B
//! follow. One native complex element covers the worst native-aligned base
//! before the widened complex and real arenas. No allocator or owner lives here.
//! RWORK/IWORK each require only their own query slot. Vendor metadata/minimum
//! policies and fallback for a shorter valid vendor WORK belong to the adapter.
const std = @import("std");
pub const Error = error{ SizeOverflow, BufferTooSmall };
fn add(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch error.SizeOverflow;
}
fn mul(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch error.SizeOverflow;
}
fn integer(comptime I: type) void {
    if (I != i32 and I != i64) @compileError("complex GELSD query requires i32 or i64");
}
/// Round upward, and reject a floating query that cannot itself be converted to
/// the selected signed ABI width, even when the exact count fitted that width.
pub fn queryLength(comptime R: type, comptime I: type, count: usize) Error!R {
    if (R != f32 and R != f64) @compileError("complex GELSD query requires real scalar");
    comptime integer(I);
    if (count > std.math.maxInt(I)) return error.SizeOverflow;
    var value: R = @floatFromInt(count);
    if (@as(f128, value) < @as(f128, @floatFromInt(count))) value = std.math.nextAfter(R, value, std.math.inf(R));
    if (@as(f128, value) > @as(f128, @floatFromInt(std.math.maxInt(I)))) return error.SizeOverflow;
    return value;
}
pub fn Planner(comptime Core: type) type {
    return struct {
        const Rank = Core.SvdLeastSquares;
        pub fn Plan(comptime I: type) type {
            comptime integer(I);
            return struct { work: I, rwork: I, iwork: I, scalar: usize, solution: usize, residual: usize, right_vectors: usize, rhs: usize, wide: usize, real: usize };
        }
        pub fn Buffers(comptime C: type) type {
            return struct { scalar: []C, solution: []C, residual: []C, right_vectors: []C, rhs: []C, wide: []Rank.WorkspaceScalar(C), real: []Rank.Wide(C) };
        }
        fn layoutTypes(comptime C: type) void {
            const W = Rank.WorkspaceScalar(C);
            if (@sizeOf(W) % @sizeOf(C) != 0 or @alignOf(W) > @sizeOf(C))
                @compileError("complex GELSD layout needs an integral pair-size ratio and one-complex alignment reserve");
            if (@sizeOf(Rank.Wide(C)) % @sizeOf(C) != 0 or @alignOf(Rank.Wide(C)) > @alignOf(W))
                @compileError("complex GELSD real arena must fit the widened pair alignment and native element count");
        }
        pub fn make(comptime C: type, comptime I: type, m: usize, n: usize, nrhs: usize) Error!Plan(I) {
            comptime integer(I);
            comptime layoutTypes(C);
            const W = Rank.WorkspaceScalar(C);
            if (m > std.math.maxInt(I) or n > std.math.maxInt(I) or nrhs > std.math.maxInt(I)) return error.SizeOverflow;
            if (m == 0 or n == 0) return .{ .work = 1, .rwork = 1, .iwork = 1, .scalar = 0, .solution = 0, .residual = 0, .right_vectors = 0, .rhs = 0, .wide = 0, .real = 0 };
            const native = Rank.plan(C, m, n, nrhs) catch return error.SizeOverflow;
            const solution = try mul(n, nrhs);
            const residual = try mul(m -| n, nrhs);
            const right_vectors = try mul(n, @min(m, n));
            const rhs = try mul(m, nrhs);
            var count = try add(1, native.scalar_elements);
            count = try add(count, solution);
            count = try add(count, residual);
            count = try add(count, right_vectors);
            count = try add(count, rhs);
            if (native.computational_elements != 0 or native.real_elements != 0) {
                // sizeof(C) equals the pair alignment here, but alignof(C) is
                // only half that value. No alignof(W)/sizeof(C)-1 formula fits.
                count = try add(count, 1);
                count = try add(count, try mul(native.computational_elements, @sizeOf(W) / @sizeOf(C)));
                count = try add(count, try mul(native.real_elements, @sizeOf(Rank.Wide(C)) / @sizeOf(C)));
            }
            const query = try queryLength(Core.Scalar.Real(C), I, count);
            const work: I = @intFromFloat(query);
            const work_count = std.math.cast(usize, work) orelse return error.SizeOverflow;
            _ = try mul(work_count, @sizeOf(C));
            return .{ .work = work, .rwork = 1, .iwork = 1, .scalar = native.scalar_elements, .solution = solution, .residual = residual, .right_vectors = right_vectors, .rhs = rhs, .wide = native.computational_elements, .real = native.real_elements };
        }
        /// A returned widened slice is wholly inside the recommended prefix.
        /// This only validates the own layout; a vendor-minimum short buffer may
        /// require the adapter's existing Context scratch fallback instead.
        /// Splitting never reads/writes WORK[0], alignment padding or suffix.
        pub fn split(comptime C: type, comptime I: type, plan: Plan(I), work: []C) Error!Buffers(C) {
            comptime layoutTypes(C);
            const W = Rank.WorkspaceScalar(C);
            if (plan.work < 1) return error.BufferTooSmall;
            const count = std.math.cast(usize, plan.work) orelse return error.SizeOverflow;
            if (work.len < count) return error.BufferTooSmall;
            const base = @intFromPtr(work.ptr);
            const limit = try add(base, try mul(count, @sizeOf(C)));
            const scalar_end = try add(1, plan.scalar);
            const solution_end = try add(scalar_end, plan.solution);
            const residual_end = try add(solution_end, plan.residual);
            const vectors_end = try add(residual_end, plan.right_vectors);
            const rhs_end = try add(vectors_end, plan.rhs);
            if (rhs_end > count) return error.BufferTooSmall;
            var wide: []W = &.{};
            var real: []Rank.Wide(C) = &.{};
            if (plan.wide != 0 or plan.real != 0) {
                const address = try add(base, try mul(rhs_end, @sizeOf(C)));
                const aligned = (try add(address, @alignOf(W) - 1)) & ~@as(usize, @alignOf(W) - 1);
                const wide_end = try add(aligned, try mul(plan.wide, @sizeOf(W)));
                const real_end = try add(wide_end, try mul(plan.real, @sizeOf(Rank.Wide(C))));
                if (real_end > limit) return error.BufferTooSmall;
                const pointer: [*]W = @ptrFromInt(aligned);
                wide = pointer[0..plan.wide];
                const real_pointer: [*]Rank.Wide(C) = @ptrFromInt(wide_end);
                real = real_pointer[0..plan.real];
            }
            return .{ .scalar = work[1..scalar_end], .solution = work[scalar_end..solution_end], .residual = work[solution_end..residual_end], .right_vectors = work[residual_end..vectors_end], .rhs = work[vectors_end..rhs_end], .wide = wide, .real = real };
        }
    };
}
