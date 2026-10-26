// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const B = @import("numerical-owner");
const A = B.GvdAdapter;
const Raw = B.RawGvdAdapter;
const N = Core.internal.numerics;
const S = N.Scalar;
const P = B.evd_protocol.policy;
const Ctx = N.Execution.context;
const F = N.Execution.test_support;
const E = std.testing;
const types = .{ f32, f64, S.ComplexF32, S.ComplexF64 };
fn value(comptime T: type, re: f64, im: f64) T {
    return if (comptime S.isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
const Z = struct { re: f64, im: f64 = 0 };
fn read(comptime T: type, x: T) Z {
    return if (comptime S.isComplex(T)) .{ .re = x.re, .im = x.im } else .{ .re = x };
}
fn add(a: Z, b: Z) Z {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn mul(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn conj(a: Z) Z {
    return .{ .re = a.re, .im = -a.im };
}
fn scale(a: Z, b: f64) Z {
    return .{ .re = a.re * b, .im = a.im * b };
}
fn near(a: Z, b: Z, eps: f64) !void {
    try E.expectApproxEqAbs(a.re, b.re, eps);
    try E.expectApproxEqAbs(a.im, b.im, eps);
}
fn at(layout: c_int, i: usize, j: usize, ld: usize) usize {
    return if (layout == 102) i + j * ld else i * ld + j;
}
fn operand(comptime raw: bool, comptime T: type, values: []T) B.common.access.Operand(T, true, if (raw) .raw else .typed) {
    return if (raw) @ptrCast(values.ptr) else values;
}
fn finish(owner: *Ctx.Context, alloc: *F.Allocations) !void {
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    _ = owner.requestClose();
    try E.expectEqual(Ctx.CloseResult.closed, owner.tryProgressClose());
    try E.expectEqual(alloc.stats().accepted, alloc.stats().freed);
}
const Recorder = struct {
    owner: *Ctx.Context,
    enabled: bool = true,
    scans: usize = 0,
    count: usize = 0,
    events: [16]B.common.ErrorEvent = undefined,
    fn nan(ptr: *anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        self.scans += 1;
        return self.enabled;
    }
    fn event(ptr: *anyopaque, e: B.common.ErrorEvent) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        if (self.owner.snapshot().active_operations != 1) @panic("GVD lost its shared operation");
        self.events[self.count] = e;
        self.count += 1;
    }
    fn services(self: *@This()) B.Provider.Services {
        return .{ .allocator = undefined, .context = self, .nan_enabled = nan, .error_event = event };
    }
};

test "GVD any-slot query borrows one closing owner and reads no matrices or eigenvalues" {
    comptime {
        if (A.OwnerIdentity != Core or Raw.OwnerIdentity != Core) @compileError("GVD owner mismatch");
    }
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        const Engine = if (raw) Raw else A;
        const R = S.Real(T);
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        _ = owner.requestClose();
        for (0..if (S.isComplex(T)) @as(usize, 3) else 2) |slot| {
            var work = [_]T{value(T, 77, 33)};
            var real = [_]R{77};
            var integers = [_]i32{77};
            const args: Engine.Arguments(T) = .{ .layout = layout, .itype = 3, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .ldb = 2, .work = operand(raw, T, &work), .rwork = operand(raw, R, &real), .iwork = operand(raw, i32, &integers), .lwork = if (slot == 0) -1 else -2, .liwork = if (slot == 1) -1 else -2, .lrwork = if (slot == 2) -1 else -2 };
            try E.expectEqual(@as(i32, 0), try Engine.workAdmitted(T, &op, args));
            try near(.{ .re = if (S.isComplex(T)) 8 else 21 }, read(T, work[0]), 0);
            try E.expectEqual(@as(i32, 13), integers[0]);
            try E.expectEqual(@as(R, if (S.isComplex(T)) 19 else 77), real[0]);
        }
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        op.end();
        try finish(&owner, &alloc);
    };
}

fn numeric(comptime T: type, comptime raw: bool, high: bool, layout: c_int, upper: bool, wants: bool, itype: i32) !void {
    const Engine = if (raw) Raw else A;
    const R = S.Real(T);
    const eps: f64 = std.math.floatEps(R) * 4096;
    const ia: f64 = if (S.isComplex(T)) 0.5 else 0;
    const ib: f64 = if (S.isComplex(T)) 0.25 else 0;
    const am = [_]Z{ .{ .re = 4 }, .{ .re = 1, .im = -ia }, .{ .re = 1, .im = ia }, .{ .re = 3 } };
    const bm = [_]Z{ .{ .re = 3 }, .{ .re = 0.5, .im = -ib }, .{ .re = 0.5, .im = ib }, .{ .re = 2 } };
    var alloc: F.Allocations = .{ .budget = 256 };
    var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
    var op = try owner.begin();
    var rec: Recorder = .{ .owner = &owner };
    var a: [10]T = @splat(value(T, -997, 0));
    var b: [12]T = @splat(value(T, -993, 0));
    for (0..2) |j| for (0..2) |i| {
        const selected = if (upper) i <= j else i >= j;
        a[at(layout, i, j, 4)] = if (selected) value(T, am[i + j * 2].re, am[i + j * 2].im) else value(T, std.math.nan(f64), 0);
        b[at(layout, i, j, 5)] = if (selected) value(T, bm[i + j * 2].re, bm[i + j * 2].im) else value(T, std.math.nan(f64), 0);
    };
    const original_a = a;
    const original_b = b;
    var values = [_]R{ 77, 77, 77 };
    var work: [64]T = undefined;
    var real: [64]R = undefined;
    var integers: [64]i32 = undefined;
    const required = try P.minimumWorkspace(i32, 2, wants, if (S.isComplex(T)) .complex else .real);
    const args: Engine.Arguments(T) = .{ .layout = layout, .itype = itype, .jobz = if (wants) 'V' else 'N', .uplo = if (upper) 'U' else 'L', .n = 2, .lda = 4, .ldb = 5, .a = operand(raw, T, &a), .b = operand(raw, T, &b), .w = operand(raw, R, &values), .work = if (high) null else operand(raw, T, &work), .rwork = if (high) null else operand(raw, R, &real), .iwork = if (high) null else operand(raw, i32, &integers), .lwork = required.work, .lrwork = required.rwork, .liwork = required.iwork };
    const info = if (high) try Engine.highAdmitted(true, T, &op, &rec.services(), args) else try Engine.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args);
    try E.expectEqual(@as(i32, 0), info);
    const det_a = 12 - 1 - ia * ia;
    const det_b = 6 - 0.25 - ib * ib;
    const trace = if (itype == 1) 16 - 2 * ia * ib else 19 + 2 * ia * ib;
    const determinant = det_a * det_b;
    const divisor = if (itype == 1) det_b else 1;
    const disc = @sqrt(trace * trace - 4 * determinant);
    const expected = [_]f64{ (trace - disc) / (2 * divisor), (trace + disc) / (2 * divisor) };
    for (expected, 0..) |lambda, j| try E.expectApproxEqAbs(lambda, @as(f64, values[j]), eps * @max(1, @abs(lambda)));
    try E.expectEqual(@as(R, 77), values[2]);
    for (a, 0..) |_, k| {
        var used = false;
        for (0..2) |j| for (0..2) |i| {
            if (at(layout, i, j, 4) == k) used = true;
        };
        if (!used) try E.expectEqualSlices(u8, std.mem.asBytes(&original_a[k]), std.mem.asBytes(&a[k]));
    }
    for (b, 0..) |_, k| {
        var used = false;
        for (0..2) |j| for (0..2) |i| {
            if (at(layout, i, j, 5) == k) used = true;
        };
        if (!used) try E.expectEqualSlices(u8, std.mem.asBytes(&original_b[k]), std.mem.asBytes(&b[k]));
    }
    for (0..2) |j| for (0..2) |i| {
        var reconstructed: Z = .{ .re = 0 };
        for (0..2) |k| {
            const left = if (k <= i) read(T, b[at(layout, if (upper) k else i, if (upper) i else k, 5)]) else Z{ .re = 0 };
            const right = if (k <= j) read(T, b[at(layout, if (upper) k else j, if (upper) j else k, 5)]) else Z{ .re = 0 };
            reconstructed = add(reconstructed, if (upper) mul(conj(left), right) else mul(left, conj(right)));
        }
        try near(bm[i + j * 2], reconstructed, eps * 16);
        if (if (upper) i > j else i < j) {
            const index = at(layout, i, j, 5);
            try E.expectEqualSlices(u8, std.mem.asBytes(&original_b[index]), std.mem.asBytes(&b[index]));
        }
    };
    if (wants) {
        for (0..2) |j| {
            var ax: [2]Z = @splat(.{ .re = 0 });
            var bx: [2]Z = @splat(.{ .re = 0 });
            for (0..2) |i| for (0..2) |k| {
                const x = read(T, a[at(layout, k, j, 4)]);
                ax[i] = add(ax[i], mul(am[i + k * 2], x));
                bx[i] = add(bx[i], mul(bm[i + k * 2], x));
            };
            for (0..2) |i| {
                var left = ax[i];
                var right = scale(bx[i], values[j]);
                if (itype != 1) {
                    left = .{ .re = 0 };
                    for (0..2) |k| left = add(left, mul(if (itype == 2) am[i + k * 2] else bm[i + k * 2], if (itype == 2) bx[k] else ax[k]));
                    right = scale(read(T, a[at(layout, i, j, 4)]), values[j]);
                }
                try near(left, right, eps * 128);
            }
        }
        const metric = if (itype == 3) [_]Z{ scale(bm[3], 1 / det_b), scale(bm[1], -1 / det_b), scale(bm[2], -1 / det_b), scale(bm[0], 1 / det_b) } else bm;
        for (0..2) |j| for (0..2) |k| {
            var gram: Z = .{ .re = 0 };
            for (0..2) |i| for (0..2) |r| {
                gram = add(gram, mul(conj(read(T, a[at(layout, i, j, 4)])), mul(metric[i + r * 2], read(T, a[at(layout, r, k, 4)]))));
            };
            try near(.{ .re = if (j == k) 1 else 0 }, gram, eps * 16);
        };
    }
    try E.expectEqual(@as(usize, 0), rec.count);
    try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
    op.end();
    try finish(&owner, &alloc);
}

test "GVD all scalar layouts triangles problem types and entries satisfy independent eigen and metric checks" {
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]bool{ false, true }) |high| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for ([_]bool{ false, true }) |wants| for ([_]i32{ 1, 2, 3 }) |itype| try numeric(T, raw, high, layout, upper, wants, itype);
}

test "GVD failed B publishes the shared-kernel factor prefix before unused column-major A and W" {
    inline for (types) |T| for ([_]c_int{ 101, 102 }) |layout| {
        const R = S.Real(T);
        var alloc: F.Allocations = .{ .budget = 64 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var a: [4]T = @splat(value(T, 77, 0));
        var b = [_]T{ value(T, 4, 0), value(T, if (layout == 102) 2 else 99, 0), value(T, if (layout == 102) 99 else 2, 0), value(T, 1, 0) };
        var work = [_]T{value(T, 77, 0)};
        var real = [_]R{77};
        var integers = [_]i32{77};
        const required = try P.minimumWorkspace(i32, 2, true, if (S.isComplex(T)) .complex else .real);
        const args: A.Arguments(T) = .{ .layout = layout, .itype = 1, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .ldb = 2, .a = if (layout == 102) null else &a, .b = &b, .work = &work, .rwork = &real, .iwork = &integers, .lwork = required.work, .lrwork = required.rwork, .liwork = required.iwork };
        try E.expectEqual(@as(i32, 4), try A.workAdmitted(T, &op, args));
        try near(.{ .re = 2 }, read(T, b[0]), 0);
        try near(.{ .re = 1 }, read(T, b[at(layout, 1, 0, 2)]), 0);
        // GVD does not specify B after its factorization fails. Preserve the
        // existing kernel's explicit unblocked contract: committed prefix,
        // unchanged rejected pivot, and unchanged opposite triangle.
        try near(.{ .re = 1 }, read(T, b[3]), 0);
        try near(.{ .re = 99 }, read(T, b[at(layout, 0, 1, 2)]), 0);
        for (a) |x| try near(.{ .re = 77 }, read(T, x), 0);
        op.end();
        try finish(&owner, &alloc);
    };
}

test "GVD scalar and row leading dimension precedence retains public argument positions" {
    var alloc: F.Allocations = .{ .budget = 0 };
    var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
    var op = try owner.begin();
    var args: A.Arguments(f64) = .{ .layout = 99, .itype = 0, .jobz = 'X', .uplo = 'X', .n = -1, .lda = -2, .ldb = -2, .lwork = -1 };
    try E.expectEqual(@as(i32, -1), try A.workAdmitted(f64, &op, args));
    args.layout = 101;
    try E.expectEqual(@as(i32, -7), try A.workAdmitted(f64, &op, args));
    args.lda = 0;
    try E.expectEqual(@as(i32, -9), try A.workAdmitted(f64, &op, args));
    args.ldb = 0;
    try E.expectEqual(@as(i32, -2), try A.workAdmitted(f64, &op, args));
    args.itype = 1;
    try E.expectEqual(@as(i32, -3), try A.workAdmitted(f64, &op, args));
    args.jobz = 'V';
    try E.expectEqual(@as(i32, -4), try A.workAdmitted(f64, &op, args));
    args.uplo = 'L';
    try E.expectEqual(@as(i32, -5), try A.workAdmitted(f64, &op, args));
    args.n = 0;
    args.layout = 102;
    try E.expectEqual(@as(i32, -7), try A.workAdmitted(f64, &op, args));
    args.lda = 1;
    try E.expectEqual(@as(i32, -9), try A.workAdmitted(f64, &op, args));
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    op.end();
    try finish(&owner, &alloc);
}

test "GVD high allocation failures close all shared leases and emit one diagnostic" {
    inline for (types) |T| for ([_]c_int{ 101, 102 }) |layout| {
        const R = S.Real(T);
        var failures: usize = 0;
        var successes: usize = 0;
        for (0..11) |budget| {
            var alloc: F.Allocations = .{ .budget = budget };
            var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
            var op = try owner.begin();
            var rec: Recorder = .{ .owner = &owner };
            var a = [_]T{ value(T, 4, 0), value(T, 1, 0), value(T, 1, 0), value(T, 3, 0) };
            var b = [_]T{ value(T, 3, 0), value(T, 0.5, 0), value(T, 0.5, 0), value(T, 2, 0) };
            var values = [_]R{ 77, 77 };
            const info = try A.highAdmitted(true, T, &op, &rec.services(), .{ .layout = layout, .itype = 2, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .ldb = 2, .a = &a, .b = &b, .w = &values });
            if (info == 0) {
                successes += 1;
                try E.expectEqual(@as(usize, 0), rec.count);
            } else {
                failures += 1;
                try E.expect(info == -1010 or info == -1011);
                try E.expectEqual(@as(usize, 1), rec.count);
                try E.expectEqual(info, rec.events[0].argument);
                try E.expectEqual(if (info == -1010) B.common.Entry.high else .work, rec.events[0].entry);
            }
            op.end();
            try finish(&owner, &alloc);
        }
        try E.expect(failures != 0 and successes != 0);
    };
}

test "GVD workspace errors retain real and complex public positions before unused W" {
    inline for (types) |T| for ([_]c_int{ 101, 102 }) |layout| {
        const R = S.Real(T);
        var alloc: F.Allocations = .{ .budget = 32 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var a: [4]T = @splat(value(T, 77, 0));
        var b: [4]T = @splat(value(T, 77, 0));
        var work = [_]T{value(T, 77, 0)};
        var real = [_]R{77};
        var integers = [_]i32{77};
        const required = try P.minimumWorkspace(i32, 2, true, if (S.isComplex(T)) .complex else .real);
        var args: A.Arguments(T) = .{ .layout = layout, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .ldb = 2, .a = &a, .b = &b, .work = &work, .rwork = &real, .iwork = &integers, .lwork = required.work - 1, .lrwork = required.rwork - 1, .liwork = required.iwork - 1 };
        try E.expectEqual(@as(i32, -12), try A.workAdmitted(T, &op, args));
        args.lwork = required.work;
        try E.expectEqual(@as(i32, -14), try A.workAdmitted(T, &op, args));
        if (S.isComplex(T)) {
            args.lrwork = required.rwork;
            try E.expectEqual(@as(i32, -16), try A.workAdmitted(T, &op, args));
        }
        for (a) |x| try near(.{ .re = 77 }, read(T, x), 0);
        for (b) |x| try near(.{ .re = 77 }, read(T, x), 0);
        op.end();
        try finish(&owner, &alloc);
    };
}

test "GVD query heads reject missing overlapping and misaligned destinations before writes" {
    var alloc: F.Allocations = .{ .budget = 0 };
    var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
    var op = try owner.begin();
    var work = [_]f64{77};
    var integers = [_]i32{77};
    var args: A.Arguments(f64) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .ldb = 2, .work = &work, .lwork = -1 };
    try E.expectError(error.MissingBuffer, A.workAdmitted(f64, &op, args));
    try E.expectEqual(@as(f64, 77), work[0]);
    args.iwork = @as([*]i32, @ptrCast(&work))[0..1];
    try E.expectError(error.AliasingNotAllowed, A.workAdmitted(f64, &op, args));
    try E.expectEqual(@as(f64, 77), work[0]);
    var bytes: [16]u8 align(8) = @splat(77);
    const raw: Raw.Arguments(f64) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .ldb = 2, .work = @ptrCast(&bytes[1]), .iwork = @ptrCast(&integers), .lwork = -1 };
    try E.expectError(error.MisalignedPointer, Raw.workAdmitted(f64, &op, raw));
    for (bytes) |x| try E.expectEqual(@as(u8, 77), x);
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    op.end();
    try finish(&owner, &alloc);
}

test "GVD high NaN checks identify both selected matrices and preserve compile-time scan control" {
    inline for (types) |T| {
        var alloc: F.Allocations = .{ .budget = 32 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var rec: Recorder = .{ .owner = &owner };
        var a = [_]T{value(T, std.math.nan(f64), 0)};
        var b = [_]T{value(T, 4, 0)};
        var args: A.Arguments(T) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 1, .lda = 1, .ldb = 1, .a = &a, .b = &b };
        try E.expectEqual(@as(i32, -6), try A.highAdmitted(true, T, &op, &rec.services(), args));
        a[0] = value(T, 3, 0);
        b[0] = if (S.isComplex(T)) value(T, 4, std.math.nan(f64)) else value(T, std.math.nan(f64), 0);
        try E.expectEqual(@as(i32, -8), try A.highAdmitted(true, T, &op, &rec.services(), args));
        args.n = 0;
        args.a = null;
        args.b = null;
        const scans = rec.scans;
        try E.expectEqual(@as(i32, 0), try A.highAdmitted(false, T, &op, &rec.services(), args));
        try E.expectEqual(scans, rec.scans);
        rec.enabled = false;
        try E.expectEqual(@as(i32, 0), try A.highAdmitted(true, T, &op, &rec.services(), args));
        try E.expectEqual(@as(usize, 0), rec.count);
        op.end();
        try finish(&owner, &alloc);
    }
}

test "GVD failed iteration keeps reduced vectors and reports the actual EVD block without B offset" {
    inline for (types) |T| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |wants| for ([_]i32{ 1, 2, 3 }) |itype| {
        const R = S.Real(T);
        var alloc: F.Allocations = .{ .budget = 32 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var rec: Recorder = .{ .owner = &owner };
        var a: [49]T = @splat(value(T, 0, 0));
        var b: [49]T = @splat(value(T, 0, 0));
        const diagonal = [_]f64{ 11, 13, 4, -1, 5, 17, 19 };
        for (diagonal, 0..) |x, i| {
            a[at(layout, i, i, 7)] = value(T, x, 0);
            b[at(layout, i, i, 7)] = value(T, 4, 0);
        }
        a[at(layout, 3, 2, 7)] = value(T, 2, 0);
        a[at(layout, 4, 3, 7)] = value(T, 3, 0);
        var values: [7]R = undefined;
        var work: [256]T = undefined;
        var real: [256]R = undefined;
        var integers: [256]i32 = undefined;
        const required = try P.minimumWorkspace(i32, 7, wants, if (S.isComplex(T)) .complex else .real);
        const args: A.Arguments(T) = .{ .layout = layout, .itype = itype, .jobz = if (wants) 'V' else 'N', .uplo = 'L', .n = 7, .lda = 7, .ldb = 7, .a = &a, .b = &b, .w = &values, .work = &work, .rwork = &real, .iwork = &integers, .lwork = required.work, .lrwork = required.rwork, .liwork = required.iwork, .native_options = .{ .max_sweeps = 0 } };
        try E.expectEqual(@as(i32, if (wants) 29 else 2), try A.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        if (wants) for (0..7) |j| {
            var norm: f64 = 0;
            for (0..7) |i| {
                const x = read(T, a[at(layout, i, j, 7)]);
                norm += x.re * x.re + x.im * x.im;
            }
            try E.expectApproxEqAbs(@as(f64, 1), norm, std.math.floatEps(R) * 4096);
        };
        try E.expectEqual(@as(usize, 0), rec.count);
        op.end();
        try finish(&owner, &alloc);
    };
}

test "GVD zero singleton and negative row return paths retain their actual access spans" {
    inline for (types) |T| for ([_]c_int{ 101, 102 }) |layout| for ([_]i32{ 1, 2, 3 }) |itype| {
        const R = S.Real(T);
        var alloc: F.Allocations = .{ .budget = 32 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var work = [_]T{value(T, 77, 0)};
        var real = [_]R{77};
        var integers = [_]i32{77};
        var args: A.Arguments(T) = .{ .layout = layout, .itype = itype, .jobz = 'V', .uplo = 'L', .n = 0, .lda = if (layout == 101) 0 else 1, .ldb = if (layout == 101) 0 else 1, .work = &work, .rwork = &real, .iwork = &integers, .lwork = 1, .lrwork = 1, .liwork = 1 };
        try E.expectEqual(@as(i32, 0), try A.workAdmitted(T, &op, args));
        var a = [_]T{ value(T, 3, std.math.nan(f64)), value(T, 77, 0) };
        var b = [_]T{ value(T, 4, 0), value(T, 77, 0) };
        var values = [_]R{ 77, 77 };
        args.n = 1;
        args.lda = 1;
        args.ldb = 1;
        args.a = &a;
        args.b = &b;
        args.w = &values;
        args.lwork = std.math.maxInt(i32);
        args.lrwork = std.math.maxInt(i32);
        args.liwork = std.math.maxInt(i32);
        try E.expectEqual(@as(i32, 0), try A.workAdmitted(T, &op, args));
        try E.expectApproxEqAbs(@as(R, if (itype == 1) 0.75 else 12), values[0], std.math.floatEps(R) * 64);
        try near(.{ .re = if (itype == 3) 2 else 0.5 }, read(T, a[0]), std.math.floatEps(R) * 64);
        try near(.{ .re = 2 }, read(T, b[0]), 0);
        try E.expectEqual(@as(R, 77), values[1]);
        try near(.{ .re = 77 }, read(T, a[1]), 0);
        try near(.{ .re = 77 }, read(T, b[1]), 0);
        if (layout == 101) {
            args.n = -2;
            args.lda = -1;
            args.ldb = -1;
            args.a = null;
            args.b = null;
            args.w = null;
            try E.expectEqual(@as(i32, -5), try A.workAdmitted(T, &op, args));
        }
        op.end();
        try finish(&owner, &alloc);
    };
}

test "GVD managed bridge captures existing callbacks and rejects a closed owner" {
    inline for (types) |T| {
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var rec: Recorder = .{ .owner = &owner };
        const callback: B.shared_services.Callback = .{ .userdata = &rec, .function = Recorder.event };
        const args: Raw.Arguments(T) = .{ .layout = 99, .jobz = 'V', .uplo = 'L', .n = 0, .lda = 1, .ldb = 1 };
        try E.expectEqual(@as(i32, -1), try B.Shared.gvdRawManagedWithDefaults(true, T, .high, &owner, args, .{ .lapacke = callback, .fortran_returning_test = callback }));
        try E.expectEqual(@as(usize, 1), rec.count);
        try E.expectEqual(if (S.isComplex(T)) B.common.Routine.hegvd else .sygvd, rec.events[0].routine);
        try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try finish(&owner, &alloc);
        try E.expectError(error.Closing, B.Shared.gvdRawManagedWithDefaults(true, T, .high, &owner, args, .{}));
    }
}
