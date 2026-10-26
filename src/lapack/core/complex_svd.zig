// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Borrowed thin complex one-sided Jacobi SVD; no admission/allocation/backend.
//! Iteration uses compensated normalized Hermitian dots and unitary rotations,
//! never a formed Gram matrix. Outputs commit only after all final checks.
const std = @import("std");
const dense = @import("../../numerics/dense.zig");
const storage = @import("../../numerics/storage.zig");
const math = @import("../../numerics/complex.zig");
const real_svd = @import("svd.zig");
pub const Error = real_svd.Error;
pub const Status = real_svd.Status;
pub const Result = real_svd.Result;
pub const Options = real_svd.Options;
pub const Real = math.scalar.Real;
pub const Wide = math.Wide;
pub const WorkspaceScalar = math.WideComplex;
fn singularType(comptime C: type, comptime S: type) void {
    comptime math.expectComplex(C);
    if (S != Real(C) and S != Wide(C)) @compileError("complex SVD singular type must be its real or widened real scalar");
}
pub fn workspaceLength(comptime C: type, m: usize, n: usize) Error!usize {
    comptime math.expectComplex(C);
    if (m < n) return error.DimensionMismatch;
    const a = std.math.mul(usize, m, n) catch return error.InvalidLength;
    const v = std.math.mul(usize, n, n) catch return error.InvalidLength;
    const total = std.math.add(usize, std.math.add(usize, a, v) catch return error.InvalidLength, n) catch return error.InvalidLength;
    _ = storage.byteLength(total, @sizeOf(WorkspaceScalar(C))) catch return error.InvalidLength;
    return total;
}
pub fn rectangularScratchLength(comptime C: type, m: usize, n: usize) Error!usize {
    comptime math.expectComplex(C);
    if (m >= n) return 0;
    const a = std.math.mul(usize, n, m) catch return error.InvalidLength;
    const v = std.math.mul(usize, m, m) catch return error.InvalidLength;
    const total = std.math.add(usize, a, v) catch return error.InvalidLength;
    _ = storage.byteLength(total, @sizeOf(C)) catch return error.InvalidLength;
    return total;
}
fn span(a: anytype) Error!usize {
    if (a.row_count == 0 or a.column_count == 0) return 0;
    if (a.leading_dimension < a.row_count) return error.InvalidLeadingDimension;
    const used = storage.columnMajorExtent(a.row_count, a.column_count, a.leading_dimension) catch return error.InvalidLength;
    if (a.values.len < used) return error.BufferTooSmall;
    return used;
}
fn disjoint(comptime A: type, a: []const A, comptime B: type, b: []const B) Error!void {
    const asize = storage.byteLength(a.len, @sizeOf(A)) catch return error.InvalidLength;
    const bsize = storage.byteLength(b.len, @sizeOf(B)) catch return error.InvalidLength;
    const ar = storage.byteRange(@intFromPtr(a.ptr), asize) catch return error.InvalidLength;
    const br = storage.byteRange(@intFromPtr(b.ptr), bsize) catch return error.InvalidLength;
    if (storage.overlaps(ar, br)) return error.AliasingNotAllowed;
}
fn finiteInput(comptime C: type, a: dense.Matrix(C)) Error!void {
    if (a.row_count == 0 or a.column_count == 0) return;
    for (0..a.column_count) |j| for (a.values[j * a.leading_dimension ..][0..a.row_count]) |z| {
        if (!math.finite(C, z)) return error.NonFiniteInput;
    };
}
fn toleranceFor(comptime C: type, options: Options) Error!f64 {
    const tolerance = options.relative_tolerance orelse 32 * @as(f64, std.math.floatEps(Real(C)));
    if (!std.math.isFinite(tolerance) or tolerance <= 0 or tolerance >= 1) return error.InvalidTolerance;
    return tolerance;
}
fn Columns(comptime C: type) type {
    return struct {
        data: []WorkspaceScalar(C),
        rows: usize,
        count: usize,
        fn column(self: @This(), j: usize) []WorkspaceScalar(C) {
            return self.data[j * self.rows ..][0..self.rows];
        }
    };
}
fn compensated(comptime W: type, sum: *W, compensation: *W, term: W) void {
    const adjusted = term - compensation.*;
    const next = sum.* + adjusted;
    compensation.* = (next - sum.*) - adjusted;
    sum.* = next;
}
fn magnitude(comptime C: type, z: WorkspaceScalar(C)) Wide(C) {
    return std.math.hypot(z.re, z.im);
}
fn norm(comptime C: type, x: []const WorkspaceScalar(C)) Wide(C) {
    const W = Wide(C);
    var scale: W = 0;
    for (x) |z| scale = @max(scale, @max(@abs(z.re), @abs(z.im)));
    if (scale == 0) return 0;
    var sum: W = 0;
    var compensation: W = 0;
    for (x) |z| {
        const re = z.re / scale;
        const im = z.im / scale;
        compensated(W, &sum, &compensation, re * re);
        compensated(W, &sum, &compensation, im * im);
    }
    return scale * @sqrt(sum);
}
fn hermitianDot(comptime C: type, x: []const WorkspaceScalar(C), y: []const WorkspaceScalar(C), nx: Wide(C), ny: Wide(C)) WorkspaceScalar(C) {
    const W = Wide(C);
    var dot: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
    if (nx == 0 or ny == 0) return dot;
    var cr: W = 0;
    var ci: W = 0;
    for (x, y) |a, b| {
        const ar = a.re / nx;
        const ai = a.im / nx;
        const br = b.re / ny;
        const bi = b.im / ny;
        compensated(W, &dot.re, &cr, ar * br);
        compensated(W, &dot.re, &cr, ai * bi);
        compensated(W, &dot.im, &ci, ar * bi);
        compensated(W, &dot.im, &ci, -ai * br);
    }
    return dot;
}
fn metric(comptime W: type, x: W) f64 {
    const rounded: f64 = @floatCast(x);
    return if (@as(W, rounded) < x) std.math.nextAfter(f64, rounded, std.math.inf(f64)) else rounded;
}
fn maxCorrelation(comptime C: type, a: Columns(C), lengths: []WorkspaceScalar(C)) ?f64 {
    const W = Wide(C);
    for (0..a.count) |j| {
        lengths[j] = .{ .re = norm(C, a.column(j)), .im = 0 };
        if (!std.math.isFinite(lengths[j].re)) return null;
    }
    var largest: W = 0;
    for (0..a.count) |p| for (p + 1..a.count) |q| {
        largest = @max(largest, magnitude(C, hermitianDot(C, a.column(p), a.column(q), lengths[p].re, lengths[q].re)));
    };
    return metric(W, largest);
}
fn multiply(comptime C: type, a: WorkspaceScalar(C), b: WorkspaceScalar(C)) WorkspaceScalar(C) {
    const W = Wide(C);
    return .{ .re = @mulAdd(W, a.re, b.re, -a.im * b.im), .im = @mulAdd(W, a.re, b.im, a.im * b.re) };
}
// J=[c,s*phase;-s*conj(phase),c]. Apply the same unitary J to A and V.
fn rotate(comptime C: type, x: []WorkspaceScalar(C), y: []WorkspaceScalar(C), c: Wide(C), s: Wide(C), phase: WorkspaceScalar(C)) bool {
    const W = Wide(C);
    const conjugate: WorkspaceScalar(C) = .{ .re = phase.re, .im = -phase.im };
    for (x, y) |*a, *b| {
        const old_a = a.*;
        const old_b = b.*;
        const py = multiply(C, conjugate, old_b);
        const px = multiply(C, phase, old_a);
        const next_a: WorkspaceScalar(C) = .{ .re = @mulAdd(W, c, old_a.re, -s * py.re), .im = @mulAdd(W, c, old_a.im, -s * py.im) };
        const next_b: WorkspaceScalar(C) = .{ .re = @mulAdd(W, c, old_b.re, s * px.re), .im = @mulAdd(W, c, old_b.im, s * px.im) };
        if (!std.math.isFinite(next_a.re) or !std.math.isFinite(next_a.im) or !std.math.isFinite(next_b.re) or !std.math.isFinite(next_b.im)) return false;
        a.* = next_a;
        b.* = next_b;
    }
    return true;
}
// A source certificate is exact, not a small singular-value criterion. Wide
// elimination only proposes coefficients. Rounded native coefficients are
// accepted exclusively when an error-free expansion proves each original
// component of A*q is exactly zero. Native products are exact in Wide:
// 2*24 <= 53 and 2*53 <= 113, including the full native exponent range.
fn expansionSlot(comptime C: type, scratch: []WorkspaceScalar(C), i: usize) *Wide(C) {
    return if (i % 2 == 0) &scratch[i / 2].re else &scratch[i / 2].im;
}
fn expansionAdd(comptime C: type, scratch: []WorkspaceScalar(C), count: *usize, term: Wide(C)) bool {
    @setFloatMode(.strict);
    var q = term;
    var used: usize = 0;
    for (0..count.*) |i| {
        const x = expansionSlot(C, scratch, i).*;
        const sum = q + x;
        if (!std.math.isFinite(sum)) return false;
        const bv = sum - q;
        const av = sum - bv;
        const error_part = (q - av) + (x - bv);
        if (error_part != 0) {
            expansionSlot(C, scratch, used).* = error_part;
            used += 1;
        }
        q = sum;
    }
    if (q != 0) {
        if (used == scratch.len * 2) return false;
        expansionSlot(C, scratch, used).* = q;
        used += 1;
    }
    count.* = used;
    return true;
}
fn sourceRelation(comptime C: type, source: dense.Matrix(C), j: usize, indices: []const WorkspaceScalar(C), coefficients: []const WorkspaceScalar(C), target_scale: WorkspaceScalar(C), scratch: []WorkspaceScalar(C)) bool {
    @setFloatMode(.strict);
    const W = Wide(C);
    for (0..source.row_count) |row| {
        for (0..2) |component| {
            var count: usize = 0;
            const target = source.values[row + j * source.leading_dimension];
            const target1 = if (component == 0) -@as(W, target.re) * target_scale.re else -@as(W, target.re) * target_scale.im;
            const target2 = if (component == 0) @as(W, target.im) * target_scale.im else -@as(W, target.im) * target_scale.re;
            if (!expansionAdd(C, scratch, &count, target1) or !expansionAdd(C, scratch, &count, target2)) return false;
            for (coefficients, indices[0..coefficients.len]) |coefficient, index| {
                const column: usize = @intFromFloat(index.re);
                const x = source.values[row + column * source.leading_dimension];
                // Both coefficient components have already been narrowed to C.
                const t1 = if (component == 0) @as(W, x.re) * coefficient.re else @as(W, x.re) * coefficient.im;
                const t2 = if (component == 0) -@as(W, x.im) * coefficient.im else @as(W, x.im) * coefficient.re;
                if (!expansionAdd(C, scratch, &count, t1) or !expansionAdd(C, scratch, &count, t2)) return false;
            }
            if (count != 0) return false;
        }
    }
    return true;
}
fn divideWide(comptime C: type, a: WorkspaceScalar(C), b: WorkspaceScalar(C)) WorkspaceScalar(C) {
    const scale = @max(@abs(b.re), @abs(b.im));
    const br = b.re / scale;
    const bi = b.im / scale;
    const ar = a.re / scale;
    const ai = a.im / scale;
    const denominator = br * br + bi * bi;
    return .{ .re = (ar * br + ai * bi) / denominator, .im = (ai * br - ar * bi) / denominator };
}
fn proposeRelation(comptime C: type, source: dense.Matrix(C), j: usize, indices: []const WorkspaceScalar(C), rank: usize, work: Columns(C), coefficients: []WorkspaceScalar(C)) bool {
    const m = source.row_count;
    for (0..rank + 1) |column| {
        const index: usize = if (column == rank) j else @intFromFloat(indices[column].re);
        for (work.column(column), source.values[index * source.leading_dimension ..][0..m]) |*out, input| out.* = math.widen(C, input);
    }
    var determinant: WorkspaceScalar(C) = .{ .re = 1, .im = 0 };
    for (0..rank) |pivot| {
        var best = pivot;
        for (pivot + 1..m) |row| if (magnitude(C, work.column(pivot)[row]) > magnitude(C, work.column(pivot)[best])) {
            best = row;
        };
        if (magnitude(C, work.column(pivot)[best]) == 0) return false;
        for (0..rank + 1) |column| std.mem.swap(WorkspaceScalar(C), &work.column(column)[pivot], &work.column(column)[best]);
        determinant = multiply(C, determinant, work.column(pivot)[pivot]);
        for (pivot + 1..m) |row| {
            const factor = divideWide(C, work.column(pivot)[row], work.column(pivot)[pivot]);
            for (pivot + 1..rank + 1) |column| {
                const product = multiply(C, factor, work.column(column)[pivot]);
                work.column(column)[row].re -= product.re;
                work.column(column)[row].im -= product.im;
            }
            work.column(pivot)[row] = .{ .re = 0, .im = 0 };
        }
    }
    var reverse = rank;
    while (reverse != 0) {
        reverse -= 1;
        var value = work.column(rank)[reverse];
        for (reverse + 1..rank) |column| {
            const product = multiply(C, work.column(column)[reverse], coefficients[column]);
            value.re -= product.re;
            value.im -= product.im;
        }
        coefficients[reverse] = divideWide(C, value, work.column(reverse)[reverse]);
    }
    for (coefficients[0..rank]) |coefficient| if (!std.math.isFinite(coefficient.re) or !std.math.isFinite(coefficient.im)) return false;
    // A rounded determinant may propose a homogeneous denominator. It is
    // never evidence of dependence; the native coefficients must still prove
    // the exact original component equations below. Overflow just skips it.
    coefficients[rank] = determinant;
    return true;
}
// Homogeneous certificates also permit exactly representable scaled rational
// relations (for example [3*x,x]). Scaling is only a proposal: every native
// coefficient and the native target scale still undergo the exact source test.
fn certificateScale(comptime C: type, source: dense.Matrix(C), j: usize, indices: []const WorkspaceScalar(C), rank: usize, coefficients: []const WorkspaceScalar(C), candidate: []WorkspaceScalar(C), scratch: []WorkspaceScalar(C)) ?WorkspaceScalar(C) {
    for (0..rank + 3) |attempt| {
        var scale: WorkspaceScalar(C) = .{ .re = 1, .im = 0 };
        if (attempt == 1) {
            const native = math.narrow(C, coefficients[rank]) orelse continue;
            scale = math.widen(C, native);
            if (scale.re == 0 and scale.im == 0) continue;
        } else if (attempt >= 2) {
            const column: usize = if (attempt == rank + 2) j else @intFromFloat(indices[attempt - 2].re);
            scale = .{ .re = 0, .im = 0 };
            for (source.values[column * source.leading_dimension ..][0..source.row_count]) |value| {
                const wide = math.widen(C, value);
                if (@max(@abs(wide.re), @abs(wide.im)) > @max(@abs(scale.re), @abs(scale.im))) scale = wide;
            }
            if (scale.re == 0 and scale.im == 0) continue;
        }
        var representable = true;
        for (coefficients[0..rank], candidate[0..rank]) |coefficient, *out| {
            const native = math.narrow(C, multiply(C, coefficient, scale)) orelse {
                representable = false;
                break;
            };
            out.* = math.widen(C, native);
        }
        if (representable and sourceRelation(C, source, j, indices, candidate[0..rank], scale, scratch)) return scale;
    }
    return null;
}
fn orthonormalizeColumn(comptime C: type, v: Columns(C), j: usize) bool {
    const W = Wide(C);
    const x = v.column(j);
    // Scale before MGS so a legitimate large coefficient cannot overflow dots.
    const initial = norm(C, x);
    if (initial == 0 or !std.math.isFinite(initial)) return false;
    for (x) |*value| {
        value.re /= initial;
        value.im /= initial;
    }
    for (0..2) |_| for (0..j) |previous| {
        const y = v.column(previous);
        const dot = hermitianDot(C, y, x, 1, 1);
        for (x, y) |*value, basis| {
            const product = multiply(C, basis, dot);
            value.re -= product.re;
            value.im -= product.im;
        }
    };
    const length = norm(C, x);
    if (length <= 0.25 / @sqrt(@as(W, @floatFromInt(v.rows)))) return false;
    for (x) |*value| {
        value.re /= length;
        value.im /= length;
    }
    return true;
}
fn originalColumns(comptime C: type, source: dense.Matrix(C), work: Columns(C), v: Columns(C)) void {
    for (0..work.count) |j| {
        for (work.column(j), source.values[j * source.leading_dimension ..][0..work.rows]) |*x, y| x.* = math.widen(C, y);
        for (v.column(j), 0..) |*x, i| x.* = .{ .re = if (i == j) 1 else 0, .im = 0 };
    }
}
fn certifiedNullspace(comptime C: type, source: dense.Matrix(C), work: Columns(C), v: Columns(C), coefficients: []WorkspaceScalar(C)) bool {
    const W = Wide(C);
    const indices = v.column(v.count - 1);
    var rank: usize = 0;
    var nullity: usize = 0;
    for (0..v.count) |j| {
        const proposed = proposeRelation(C, source, j, indices, rank, work, coefficients);
        const certified_scale = if (proposed) certificateScale(C, source, j, indices, rank, coefficients, v.column(nullity), work.data) else null;
        if (certified_scale) |scale| {
            const vector = v.column(nullity);
            @memcpy(coefficients[0..rank], vector[0..rank]);
            // Metadata can overlap the final all-zero kernel vector only when
            // rank is zero, so no selected indices are then needed.
            @memset(vector, .{ .re = 0, .im = 0 });
            for (0..rank) |p| vector[@intFromFloat(indices[p].re)] = coefficients[p];
            vector[j] = .{ .re = -scale.re, .im = -scale.im };
            nullity += 1;
        } else {
            indices[rank] = .{ .re = @floatFromInt(j), .im = 0 };
            rank += 1;
        }
    }
    if (nullity == 0) {
        originalColumns(C, source, work, v);
        return false;
    }
    for (0..nullity) |j| if (!orthonormalizeColumn(C, v, j)) {
        originalColumns(C, source, work, v);
        return false;
    };
    for (nullity..v.count) |j| {
        var found = false;
        for (0..v.rows) |basis| {
            @memset(v.column(j), .{ .re = 0, .im = 0 });
            v.column(j)[basis].re = 1;
            if (orthonormalizeColumn(C, v, j)) {
                found = true;
                break;
            }
        }
        if (!found) {
            originalColumns(C, source, work, v);
            return false;
        }
    }
    for (0..work.count) |j| {
        if (j < nullity) {
            @memset(work.column(j), .{ .re = 0, .im = 0 });
            continue;
        }
        for (work.column(j), 0..) |*output, row| {
            var value: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
            var cr: W = 0;
            var ci: W = 0;
            for (v.column(j), 0..) |coefficient, column| {
                const product = multiply(C, math.widen(C, source.values[row + column * source.leading_dimension]), coefficient);
                compensated(W, &value.re, &cr, product.re);
                compensated(W, &value.im, &ci, product.im);
            }
            output.* = value;
        }
    }
    return true;
}

fn upward(comptime W: type, value: W) W {
    return std.math.nextAfter(W, value, std.math.inf(W));
}
fn zeroOutput(comptime S: type, value: anytype) bool {
    return @as(S, @floatCast(value)) == 0;
}
// Only an entire set with cumulatively bounded discarded Frobenius norm that
// rounds to zero in S may disappear. This is never a relative rank criterion.
fn deflateTiny(comptime C: type, comptime S: type, a: Columns(C), lengths: []WorkspaceScalar(C), discarded: *Wide(C)) usize {
    const W = Wide(C);
    var scale: W = 0;
    for (0..a.count) |j| {
        lengths[j] = .{ .re = norm(C, a.column(j)), .im = 0 };
        if (lengths[j].re != 0 and zeroOutput(S, lengths[j].re)) for (a.column(j)) |z| {
            scale = @max(scale, @max(@abs(z.re), @abs(z.im)));
        };
    }
    if (scale == 0) return 0;
    var squares: W = 0;
    for (0..a.count) |j| if (lengths[j].re != 0 and zeroOutput(S, lengths[j].re)) {
        for (a.column(j)) |z| for ([_]W{ z.re, z.im }) |value| if (value != 0) {
            const q = upward(W, @abs(value) / scale);
            squares = upward(W, squares + upward(W, q * q));
        };
    };
    const bound = upward(W, scale * upward(W, @sqrt(squares)));
    const total = upward(W, discarded.* + bound);
    if (!std.math.isFinite(total) or !zeroOutput(S, total)) return 0;
    var removed: usize = 0;
    for (0..a.count) |j| if (lengths[j].re != 0 and zeroOutput(S, lengths[j].re)) {
        @memset(a.column(j), .{ .re = 0, .im = 0 });
        lengths[j] = .{ .re = 0, .im = 0 };
        removed += 1;
    };
    discarded.* = total;
    return removed;
}
fn roundedOrthogonality(comptime C: type, a: Columns(C)) f64 {
    const W = Wide(C);
    var largest: W = 0;
    for (0..a.count) |p| for (p..a.count) |q| {
        var dot: WorkspaceScalar(C) = .{ .re = 0, .im = 0 };
        var cr: W = 0;
        var ci: W = 0;
        for (a.column(p), a.column(q)) |x, y| {
            const rx = math.narrow(C, x) orelse return std.math.inf(f64);
            const ry = math.narrow(C, y) orelse return std.math.inf(f64);
            compensated(W, &dot.re, &cr, @as(W, rx.re) * @as(W, ry.re));
            compensated(W, &dot.re, &cr, @as(W, rx.im) * @as(W, ry.im));
            compensated(W, &dot.im, &ci, @as(W, rx.re) * @as(W, ry.im));
            compensated(W, &dot.im, &ci, -@as(W, rx.im) * @as(W, ry.re));
        }
        dot.re -= @as(W, if (p == q) 1 else 0);
        if (!std.math.isFinite(dot.re) or !std.math.isFinite(dot.im)) return std.math.inf(f64);
        largest = @max(largest, magnitude(C, dot));
    };
    return metric(W, largest);
}
fn finish(comptime C: type, comptime S: type, a: Columns(C), v: Columns(C), lengths: []WorkspaceScalar(C), tolerance: f64, result: *Result) bool {
    const W = Wide(C);
    for (0..a.count) |j| {
        lengths[j] = .{ .re = norm(C, a.column(j)), .im = 0 };
        if (!std.math.isFinite(@as(S, @floatCast(lengths[j].re)))) {
            result.status = .non_finite_result;
            return false;
        }
        // A rejected cumulative discard cannot be bypassed at finalization.
        if (lengths[j].re != 0 and zeroOutput(S, lengths[j].re)) {
            result.status = .range_limit;
            return false;
        }
    }
    for (0..a.count) |j| {
        var largest = j;
        for (j + 1..a.count) |k| if (lengths[k].re > lengths[largest].re) {
            largest = k;
        };
        if (largest != j) {
            std.mem.swap(WorkspaceScalar(C), &lengths[j], &lengths[largest]);
            for (a.column(j), a.column(largest)) |*x, *y| std.mem.swap(WorkspaceScalar(C), x, y);
            for (v.column(j), v.column(largest)) |*x, *y| std.mem.swap(WorkspaceScalar(C), x, y);
        }
    }
    for (0..a.count) |j| {
        const column = a.column(j);
        if (lengths[j].re != 0) {
            for (column) |*z| {
                z.re /= lengths[j].re;
                z.im /= lengths[j].re;
            }
            continue;
        }
        var found = false;
        for (0..a.rows) |basis| {
            @memset(column, .{ .re = 0, .im = 0 });
            column[basis].re = 1;
            for (0..2) |_| for (0..j) |previous| {
                const vector = a.column(previous);
                const dot = hermitianDot(C, vector, column, 1, 1);
                for (column, vector) |*z, x| {
                    const product = multiply(C, x, dot);
                    z.re -= product.re;
                    z.im -= product.im;
                }
            };
            const length = norm(C, column);
            if (length > 0.25 / @sqrt(@as(W, @floatFromInt(a.rows)))) {
                for (column) |*z| {
                    z.re /= length;
                    z.im /= length;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            result.status = .range_limit;
            return false;
        }
    }
    result.output_orthogonality = @max(roundedOrthogonality(C, a), roundedOrthogonality(C, v));
    if (!std.math.isFinite(result.output_orthogonality) or result.output_orthogonality > tolerance) {
        result.status = .range_limit;
        return false;
    }
    return true;
}
pub fn decompose(comptime C: type, a: dense.Matrix(C), singular: []Real(C), right: dense.Matrix(C), workspace: []WorkspaceScalar(C), options: Options) Error!Result {
    return decomposeWithSingularType(C, Real(C), a, singular, right, workspace, options);
}
/// Tall thin A=U*diag(S)*V^H. Workspace is m*n+n*n+n widened complex values.
/// A/S/V, their padding and suffixes stay unchanged on every unsuccessful exit.
/// Workspace old contents are ignored; only its used prefix is touched.
pub fn decomposeWithSingularType(comptime C: type, comptime S: type, a: dense.Matrix(C), singular: []S, right: dense.Matrix(C), workspace: []WorkspaceScalar(C), options: Options) Error!Result {
    comptime singularType(C, S);
    const W = Wide(C);
    const aspan = try span(a);
    const vspan = try span(right);
    const used = try workspaceLength(C, a.row_count, a.column_count);
    const m = a.row_count;
    const n = a.column_count;
    if (right.row_count != n or right.column_count != n) return error.DimensionMismatch;
    if (singular.len < n or workspace.len < used) return error.BufferTooSmall;
    const tolerance = try toleranceFor(C, options);
    try disjoint(C, a.values[0..aspan], S, singular[0..n]);
    try disjoint(C, a.values[0..aspan], C, right.values[0..vspan]);
    try disjoint(C, right.values[0..vspan], S, singular[0..n]);
    try disjoint(C, a.values[0..aspan], WorkspaceScalar(C), workspace[0..used]);
    try disjoint(C, right.values[0..vspan], WorkspaceScalar(C), workspace[0..used]);
    try disjoint(S, singular[0..n], WorkspaceScalar(C), workspace[0..used]);
    try finiteInput(C, a);
    var result: Result = .{ .status = .no_convergence, .sweeps = 0, .max_correlation = 0 };
    if (n == 0) {
        result.status = .converged;
        return result;
    }
    const aw: Columns(C) = .{ .data = workspace[0 .. m * n], .rows = m, .count = n };
    const vw: Columns(C) = .{ .data = workspace[m * n ..][0 .. n * n], .rows = n, .count = n };
    const lengths = workspace[m * n + n * n ..][0..n];
    originalColumns(C, a, aw, vw);
    if (options.max_sweeps != 0 and certifiedNullspace(C, a, aw, vw, lengths)) result.sweeps = 1;
    var discarded: W = 0;
    while (true) {
        result.deflated_columns += deflateTiny(C, S, aw, lengths, &discarded);
        result.max_correlation = maxCorrelation(C, aw, lengths) orelse {
            result.status = .non_finite_result;
            result.max_correlation = std.math.inf(f64);
            return result;
        };
        if (result.max_correlation <= tolerance) break;
        if (result.sweeps == options.max_sweeps) return result;
        for (0..n) |p| for (p + 1..n) |q| {
            const x = aw.column(p);
            const y = aw.column(q);
            const nx = norm(C, x);
            const ny = norm(C, y);
            if (!std.math.isFinite(nx) or !std.math.isFinite(ny)) {
                result.status = .non_finite_result;
                return result;
            }
            const rho = hermitianDot(C, x, y, nx, ny);
            const absolute = magnitude(C, rho);
            if (absolute <= tolerance) continue;
            const ratio = @min(nx, ny) / @max(nx, ny);
            const delta = if (nx > ny) ratio * ratio - 1 else 1 - ratio * ratio;
            const cross = absolute * ratio;
            const denominator = delta + std.math.copysign(std.math.hypot(delta, 2 * cross), delta);
            const tangent = 2 * cross / denominator;
            if (tangent == 0 or !std.math.isFinite(tangent)) {
                result.status = .range_limit;
                return result;
            }
            const c: W = 1 / @sqrt(1 + tangent * tangent);
            const s = tangent * c;
            const phase: WorkspaceScalar(C) = .{ .re = rho.re / absolute, .im = rho.im / absolute };
            if (!rotate(C, x, y, c, s, phase) or !rotate(C, vw.column(p), vw.column(q), c, s, phase)) {
                result.status = .non_finite_result;
                return result;
            }
        };
        result.sweeps += 1;
    }
    if (!finish(C, S, aw, vw, lengths, tolerance, &result)) return result;
    for (0..n) |j| {
        singular[j] = @floatCast(lengths[j].re);
        for (a.values[j * a.leading_dimension ..][0..m], aw.column(j)) |*x, y| x.* = math.narrow(C, y).?;
        for (right.values[j * right.leading_dimension ..][0..n], vw.column(j)) |*x, y| x.* = math.narrow(C, y).?;
    }
    result.status = .converged;
    return result;
}
pub fn decomposeRectangular(comptime C: type, a: dense.Matrix(C), singular: []Real(C), right: dense.Matrix(C), workspace: []WorkspaceScalar(C), transpose_scratch: []C, options: Options) Error!Result {
    return decomposeRectangularWithSingularType(C, Real(C), a, singular, right, workspace, transpose_scratch, options);
}
/// Wide A uses a private conjugate transpose and swaps its thin factor roles.
/// Only the first min(m,n) columns of public A become U; remaining columns stay.
pub fn decomposeRectangularWithSingularType(comptime C: type, comptime S: type, a: dense.Matrix(C), singular: []S, right: dense.Matrix(C), workspace: []WorkspaceScalar(C), transpose_scratch: []C, options: Options) Error!Result {
    comptime singularType(C, S);
    if (a.row_count >= a.column_count) return decomposeWithSingularType(C, S, a, singular, right, workspace, options);
    const m = a.row_count;
    const n = a.column_count;
    const aspan = try span(a);
    const vspan = try span(right);
    if (right.row_count != n or right.column_count != m) return error.DimensionMismatch;
    const used = try workspaceLength(C, n, m);
    const temporary = try rectangularScratchLength(C, m, n);
    if (singular.len < m or workspace.len < used or transpose_scratch.len < temporary) return error.BufferTooSmall;
    _ = try toleranceFor(C, options);
    const operands = [_][]C{ a.values[0..aspan], right.values[0..vspan], transpose_scratch[0..temporary] };
    for (operands, 0..) |operand, i| {
        for (operands[i + 1 ..]) |other| try disjoint(C, operand, C, other);
        try disjoint(C, operand, S, singular[0..m]);
        try disjoint(C, operand, WorkspaceScalar(C), workspace[0..used]);
    }
    try disjoint(S, singular[0..m], WorkspaceScalar(C), workspace[0..used]);
    try finiteInput(C, a);
    if (m == 0) return .{ .status = .converged, .sweeps = 0, .max_correlation = 0 };
    const at = transpose_scratch[0 .. n * m];
    const small_right = transpose_scratch[n * m ..][0 .. m * m];
    for (0..n) |j| for (0..m) |i| {
        const value = a.values[i + j * a.leading_dimension];
        at[j + i * n] = .{ .re = value.re, .im = -value.im };
    };
    const result = try decomposeWithSingularType(C, S, .{ .values = at, .row_count = n, .column_count = m, .leading_dimension = n }, singular, .{ .values = small_right, .row_count = m, .column_count = m, .leading_dimension = m }, workspace, options);
    if (result.status != .converged) return result;
    for (0..m) |j| {
        for (0..m) |i| a.values[i + j * a.leading_dimension] = small_right[i + j * m];
        for (0..n) |i| right.values[i + j * right.leading_dimension] = at[i + j * n];
    }
    return result;
}
