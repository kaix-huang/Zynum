// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
const Core = @import("zynum-core");
const B = @import("numerical-owner");
const A = B.EvdAdapter;
const Raw = B.RawEvdAdapter;
const N = Core.internal.numerics;
const S = N.Scalar;
const P = B.evd_protocol.policy;
const Ex = N.Execution;
const Ctx = Ex.context;
const F = Ex.test_support;
const E = std.testing;
const types = .{ f32, f64, S.ComplexF32, S.ComplexF64 };
fn value(comptime T: type, real: f64, imaginary: f64) T {
    return if (comptime S.isComplex(T)) .{ .re = @floatCast(real), .im = @floatCast(imaginary) } else @floatCast(real);
}
fn part(comptime T: type, x: T) f64 {
    return if (comptime S.isComplex(T)) x.re else x;
}
fn imag(comptime T: type, x: T) f64 {
    return if (comptime S.isComplex(T)) x.im else 0;
}
fn eqBytes(comptime T: type, a: []const T, b: []const T) !void {
    try E.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
fn at(layout: c_int, i: usize, j: usize, ld: usize) usize {
    return if (layout == 102) i + j * ld else i * ld + j;
}
fn finish(owner: *Ctx.Context, alloc: *F.Allocations) !void {
    try E.expectEqual(@as(usize, 0), owner.snapshot().resource_pins);
    try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
    _ = owner.requestClose();
    try E.expectEqual(Ctx.CloseResult.closed, owner.tryProgressClose());
    try E.expectEqual(alloc.stats().accepted, alloc.stats().freed);
}
fn argument(comptime raw: bool, comptime T: type, slice: []T) B.common.access.Operand(T, true, if (raw) .raw else .typed) {
    return if (raw) @ptrCast(slice.ptr) else slice;
}
const Recorder = struct {
    owner: *Ctx.Context,
    enabled: bool = true,
    nan_calls: usize = 0,
    count: usize = 0,
    events: [16]B.common.ErrorEvent = undefined,
    fn nan(ptr: *anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        self.nan_calls += 1;
        return self.enabled;
    }
    fn event(ptr: *anyopaque, e: B.common.ErrorEvent) void {
        const self: *@This() = @ptrCast(@alignCast(ptr));
        if (self.owner.snapshot().active_operations != 1) @panic("EVD lost its one admitted operation");
        self.events[self.count] = e;
        self.count += 1;
    }
    fn services(self: *@This()) B.Provider.Services {
        return .{ .allocator = undefined, .context = self, .nan_enabled = nan, .error_event = event };
    }
};

test "EVD frontends query every workspace slot without allocation after close" {
    comptime {
        if (A.OwnerIdentity != Core or Raw.OwnerIdentity != Core) @compileError("EVD owner mismatch");
    }
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| {
        const Engine = if (raw) Raw else A;
        const R = S.Real(T);
        const complex = S.isComplex(T);
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        _ = owner.requestClose();
        for (0..if (complex) @as(usize, 3) else 2) |slot| {
            var work = [_]T{value(T, 999, 33)};
            var real = [_]R{999};
            var integers = [_]i32{999};
            const args: Engine.Arguments(T) = .{ .layout = layout, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .work = argument(raw, T, &work), .rwork = argument(raw, R, &real), .iwork = argument(raw, i32, &integers), .lwork = if (slot == 0) -1 else -2, .liwork = if (slot == 1) -1 else -2, .lrwork = if (slot == 2) -1 else -2 };
            try E.expectEqual(@as(i32, 0), try Engine.workAdmitted(T, &op, args));
            try E.expectEqual(@as(f64, if (complex) 8 else 21), part(T, work[0]));
            try E.expectEqual(@as(f64, 0), imag(T, work[0]));
            try E.expectEqual(@as(i32, 13), integers[0]);
            try E.expectEqual(@as(R, if (complex) 19 else 999), real[0]);
        }
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try E.expectEqual(@as(usize, 1), owner.snapshot().active_operations);
        op.end();
        try finish(&owner, &alloc);
    };
}

test "EVD scalar row precedence and query nonreads preserve workspace heads" {
    inline for (types) |T| {
        const R = S.Real(T);
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var work = [_]T{value(T, 999, 33)};
        var real = [_]R{999};
        var integers = [_]i32{999};
        var args: A.Arguments(T) = .{ .layout = 99, .jobz = 'X', .uplo = 'X', .n = -1, .lda = -2, .work = &work, .rwork = &real, .iwork = &integers, .lwork = -1, .liwork = -1, .lrwork = -1 };
        try E.expectEqual(@as(i32, -1), try A.workAdmitted(T, &op, args));
        args.layout = 101;
        try E.expectEqual(@as(i32, -6), try A.workAdmitted(T, &op, args));
        args.lda = 1;
        try E.expectEqual(@as(i32, -2), try A.workAdmitted(T, &op, args));
        args.jobz = 'V';
        try E.expectEqual(@as(i32, -3), try A.workAdmitted(T, &op, args));
        args.uplo = 'L';
        try E.expectEqual(@as(i32, -4), try A.workAdmitted(T, &op, args));
        try E.expectEqual(@as(f64, 999), part(T, work[0]));
        try E.expectEqual(@as(R, 999), real[0]);
        try E.expectEqual(@as(i32, 999), integers[0]);
        args.n = 0;
        args.lda = 0;
        args.layout = 102;
        try E.expectEqual(@as(i32, -6), try A.workAdmitted(T, &op, args));
        args.layout = 101;
        try E.expectEqual(@as(i32, 0), try A.workAdmitted(T, &op, args));
        args.n = 2;
        args.lda = 2;
        args.jobz = 'X';
        args.uplo = 'X';
        args.lwork = 0;
        args.liwork = 0;
        args.lrwork = 0;
        try E.expectEqual(@as(i32, -1011), try A.workAdmitted(T, &op, args));
        args.layout = 102;
        try E.expectEqual(@as(i32, -2), try A.workAdmitted(T, &op, args));
        op.end();
        try finish(&owner, &alloc);
    }
}

test "EVD workspace errors retain public positions before unused outputs" {
    inline for (types) |T| for ([_]c_int{ 101, 102 }) |layout| {
        const R = S.Real(T);
        const family: P.Family = if (S.isComplex(T)) .complex else .real;
        const required = try P.minimumWorkspace(i32, 2, true, family);
        var alloc: F.Allocations = .{ .budget = 256 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var matrix = [_]T{ value(T, 4, 0), value(T, 1, 0), value(T, 1, 0), value(T, 3, 0) };
        const original = matrix;
        var work = [_]T{value(T, 999, 33)};
        var real = [_]R{999};
        var integers = [_]i32{999};
        var args: A.Arguments(T) = .{ .layout = layout, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .a = &matrix, .work = &work, .rwork = &real, .iwork = &integers, .lwork = required.work - 1, .lrwork = required.rwork - 1, .liwork = required.iwork - 1 };
        try E.expectEqual(@as(i32, -9), try A.workAdmitted(T, &op, args));
        args.lwork = required.work;
        try E.expectEqual(@as(i32, -11), try A.workAdmitted(T, &op, args));
        args.lrwork = required.rwork;
        try E.expectEqual(@as(i32, if (S.isComplex(T)) -13 else -11), try A.workAdmitted(T, &op, args));
        if (layout == 102) try eqBytes(T, &original, &matrix) else {
            for ([_]usize{ 0, 2, 3 }) |i| try eqBytes(T, original[i .. i + 1], matrix[i .. i + 1]);
        }
        op.end();
        try finish(&owner, &alloc);
    };
}

fn numeric(comptime T: type, comptime raw: bool, high: bool, layout: c_int, upper: bool, wants: bool) !void {
    const Engine = if (raw) Raw else A;
    const R = S.Real(T);
    const complex = S.isComplex(T);
    const eps: f64 = std.math.floatEps(R) * 400;
    var alloc: F.Allocations = .{ .budget = 256 };
    var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
    var op = try owner.begin();
    var recorder: Recorder = .{ .owner = &owner };
    var matrix: [10]T = @splat(value(T, -997, 0));
    const im: f64 = if (complex) 0.5 else 0;
    matrix[at(layout, 0, 0, 4)] = value(T, 4, 0);
    matrix[at(layout, 1, 1, 4)] = value(T, 3, 0);
    matrix[at(layout, 0, 1, 4)] = if (upper) value(T, 1, im) else value(T, std.math.nan(f64), 0);
    matrix[at(layout, 1, 0, 4)] = if (upper) value(T, std.math.nan(f64), 0) else value(T, 1, -im);
    var values = [_]R{ 77, 77, 77 };
    var work: [64]T = @splat(value(T, 73, 0));
    var real: [64]R = @splat(73);
    var integers: [64]i32 = @splat(73);
    const required = try P.minimumWorkspace(i32, 2, wants, if (complex) .complex else .real);
    const args: Engine.Arguments(T) = .{ .layout = layout, .jobz = if (wants) 'V' else 'N', .uplo = if (upper) 'U' else 'L', .n = 2, .lda = 4, .a = argument(raw, T, &matrix), .w = argument(raw, R, &values), .work = if (high) null else argument(raw, T, &work), .rwork = if (high) null else argument(raw, R, &real), .iwork = if (high) null else argument(raw, i32, &integers), .lwork = required.work, .lrwork = required.rwork, .liwork = required.iwork };
    const info = if (high) try Engine.highAdmitted(true, T, &op, &recorder.services(), args) else try Engine.workWithDiagnosticsAdmitted(T, &op, &recorder.services(), args);
    try E.expectEqual(@as(i32, 0), info);
    try E.expectEqual(@as(usize, 0), recorder.count);
    try E.expectEqual(@as(usize, if (high) 1 else 0), recorder.nan_calls);
    const radius = @sqrt(1.0 + 4.0 * (1.0 + im * im));
    try E.expectApproxEqAbs((7.0 - radius) / 2, @as(f64, values[0]), eps);
    try E.expectApproxEqAbs((7.0 + radius) / 2, @as(f64, values[1]), eps);
    try E.expectEqual(@as(R, 77), values[2]);
    if (wants) for (0..2) |j| {
        const x = matrix[at(layout, 0, j, 4)];
        const y = matrix[at(layout, 1, j, 4)];
        const xr = part(T, x);
        const xi = imag(T, x);
        const yr = part(T, y);
        const yi = imag(T, y);
        const lambda: f64 = values[j];
        try E.expectApproxEqAbs(lambda * xr, 4 * xr + yr - im * yi, eps);
        try E.expectApproxEqAbs(lambda * xi, 4 * xi + yi + im * yr, eps);
        try E.expectApproxEqAbs(lambda * yr, xr + im * xi + 3 * yr, eps);
        try E.expectApproxEqAbs(lambda * yi, xi - im * xr + 3 * yi, eps);
        try E.expectApproxEqAbs(@as(f64, 1), xr * xr + xi * xi + yr * yr + yi * yi, eps);
        const k = 1 - j;
        const otherx = matrix[at(layout, 0, k, 4)];
        const othery = matrix[at(layout, 1, k, 4)];
        try E.expectApproxEqAbs(@as(f64, 0), xr * part(T, otherx) + xi * imag(T, otherx) + yr * part(T, othery) + yi * imag(T, othery), eps);
        try E.expectApproxEqAbs(@as(f64, 0), xr * imag(T, otherx) - xi * part(T, otherx) + yr * imag(T, othery) - yi * part(T, othery), eps);
    };
    for ([_]usize{ 2, 3, 6, 7, 8, 9 }) |i| try E.expectEqual(@as(f64, -997), part(T, matrix[i]));
    if (!high) {
        try E.expectEqual(@as(f64, 73), part(T, work[@intCast(required.work)]));
        try E.expectEqual(@as(i32, 73), integers[@intCast(required.iwork)]);
        if (complex) try E.expectEqual(@as(R, 73), real[@intCast(required.rwork)]);
    }
    op.end();
    try finish(&owner, &alloc);
}
test "EVD all scalar typed raw high work layouts and triangles satisfy analytic residuals" {
    inline for (types) |T| inline for (.{ false, true }) |raw| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| for ([_]bool{ false, true }) |wants| for ([_]bool{ false, true }) |high| try numeric(T, raw, high, layout, upper, wants);
}

test "EVD high NaN scan order and diagnostic channels retain routine identity" {
    inline for (types) |T| {
        const R = S.Real(T);
        var alloc: F.Allocations = .{ .budget = 256 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var rec: Recorder = .{ .owner = &owner };
        var matrix = [_]T{value(T, std.math.nan(f64), 0)};
        var values = [_]R{73};
        var work = [_]T{value(T, 73, 0)};
        var real = [_]R{73};
        var integers = [_]i32{73};
        var args: A.Arguments(T) = .{ .layout = 102, .jobz = 'X', .uplo = 'L', .n = 1, .lda = 1, .a = &matrix, .w = &values, .work = &work, .rwork = &real, .iwork = &integers, .lwork = 1, .lrwork = 1, .liwork = 1 };
        try E.expectEqual(@as(i32, -5), try A.highAdmitted(true, T, &op, &rec.services(), args));
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try E.expectEqual(@as(usize, 0), rec.count);
        try E.expectEqual(@as(i32, -2), try A.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        try E.expectEqual(@as(usize, 1), rec.count);
        try E.expectEqual(@as(i32, 1), rec.events[0].argument);
        try E.expectEqual(if (S.isComplex(T)) B.common.Routine.heevd else .syevd, rec.events[0].routine);
        const message = try B.Diagnostics(i32).translate(rec.events[0]);
        var name: [16]u8 = undefined;
        const expected = try std.fmt.bufPrint(&name, "{c}{s}", .{ if (T == f32) @as(u8, 'S') else if (T == f64) @as(u8, 'D') else if (T == S.ComplexF32) @as(u8, 'C') else @as(u8, 'Z'), if (S.isComplex(T)) "HEEVD" else "SYEVD" });
        try E.expectEqualStrings(expected, message.name());
        rec.enabled = false;
        args.jobz = 'V';
        try E.expectEqual(@as(i32, 0), try A.highAdmitted(true, T, &op, &rec.services(), args));
        try E.expect(std.math.isNan(values[0]));
        try E.expectEqual(@as(f64, 1), part(T, matrix[0]));
        matrix[0] = value(T, std.math.nan(f64), 0);
        const scans = rec.nan_calls;
        try E.expectEqual(@as(i32, 0), try A.highAdmitted(false, T, &op, &rec.services(), args));
        try E.expectEqual(scans, rec.nan_calls);
        op.end();
        try finish(&owner, &alloc);
    }
}

test "EVD high allocation failures release all scratch and report exactly one error" {
    inline for (types) |T| for ([_]c_int{ 101, 102 }) |layout| {
        const R = S.Real(T);
        var failures: usize = 0;
        var successes: usize = 0;
        for (0..8) |budget| {
            var alloc: F.Allocations = .{ .budget = budget };
            var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
            var op = try owner.begin();
            var rec: Recorder = .{ .owner = &owner };
            var matrix = [_]T{ value(T, 4, 0), value(T, 1, 0), value(T, 1, 0), value(T, 3, 0) };
            const original = matrix;
            var values = [_]R{ 73, 73 };
            const info = try A.highAdmitted(true, T, &op, &rec.services(), .{ .layout = layout, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .a = &matrix, .w = &values });
            if (info == 0) {
                successes += 1;
                try E.expectEqual(@as(usize, 0), rec.count);
            } else {
                failures += 1;
                try eqBytes(T, &original, &matrix);
                try E.expectEqualSlices(R, &.{ 73, 73 }, &values);
                try E.expect(info == -1010 or info == -1011);
                try E.expectEqual(@as(usize, 1), rec.count);
                try E.expectEqual(info, rec.events[0].argument);
                try E.expectEqual(if (info == -1010) B.common.Entry.high else .work, rec.events[0].entry);
            }
            if (budget == 0) try E.expectEqual(@as(i32, -1010), info);
            op.end();
            try finish(&owner, &alloc);
        }
        try E.expect(failures != 0 and successes != 0);
    };
}

test "EVD zero singleton and oversized claimed lengths touch only actual quick-return spans" {
    inline for (types) |T| {
        const R = S.Real(T);
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var work = [_]T{value(T, 73, 0)};
        var real = [_]R{73};
        var integers = [_]i32{73};
        var values = [_]R{ 73, 73 };
        var matrix = [_]T{ value(T, std.math.inf(f64), std.math.nan(f64)), value(T, 73, 0) };
        var args: A.Arguments(T) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 0, .lda = 1, .work = &work, .rwork = &real, .iwork = &integers, .lwork = 1, .lrwork = 1, .liwork = 1 };
        try E.expectEqual(@as(i32, 0), try A.workAdmitted(T, &op, args));
        args.n = 1;
        args.a = &matrix;
        args.w = &values;
        args.lwork = std.math.maxInt(i32);
        args.lrwork = std.math.maxInt(i32);
        args.liwork = std.math.maxInt(i32);
        try E.expectEqual(@as(i32, 0), try A.workAdmitted(T, &op, args));
        try E.expect(std.math.isInf(values[0]));
        try E.expectEqual(@as(R, 73), values[1]);
        try E.expectEqual(@as(f64, 1), part(T, matrix[0]));
        try E.expectEqual(@as(f64, 0), imag(T, matrix[0]));
        try E.expectEqual(@as(f64, 73), part(T, matrix[1]));
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        op.end();
        try finish(&owner, &alloc);
    }
}

test "EVD query destinations reject missing misaligned and overlapping outputs before writes" {
    var alloc: F.Allocations = .{ .budget = 0 };
    var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
    var op = try owner.begin();
    var work = [_]f64{73};
    var integers = [_]i32{73};
    var args: A.Arguments(f64) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .work = &work, .lwork = -1 };
    try E.expectError(error.MissingBuffer, A.workAdmitted(f64, &op, args));
    try E.expectEqual(@as(f64, 73), work[0]);
    args.iwork = @as([*]i32, @ptrCast(&work))[0..1];
    try E.expectError(error.AliasingNotAllowed, A.workAdmitted(f64, &op, args));
    try E.expectEqual(@as(f64, 73), work[0]);
    var bytes: [16]u8 align(8) = @splat(73);
    const raw: Raw.Arguments(f64) = .{ .layout = 102, .jobz = 'V', .uplo = 'L', .n = 2, .lda = 2, .work = @ptrCast(&bytes[1]), .iwork = @ptrCast(&integers), .lwork = -1 };
    try E.expectError(error.MisalignedPointer, Raw.workAdmitted(f64, &op, raw));
    for (bytes) |x| try E.expectEqual(@as(u8, 73), x);
    try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
    op.end();
    try finish(&owner, &alloc);
}

test "EVD frontend nonconvergence encodes the actual failed submatrix" {
    inline for (types) |T| for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |wants| {
        const R = S.Real(T);
        var alloc: F.Allocations = .{ .budget = 256 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var op = try owner.begin();
        var rec: Recorder = .{ .owner = &owner };
        var matrix: [49]T = @splat(value(T, 0, 0));
        const diagonal = [_]f64{ 11, 13, 4, -1, 5, 17, 19 };
        for (diagonal, 0..) |x, i| matrix[at(layout, i, i, 7)] = value(T, x, 0);
        matrix[at(layout, 3, 2, 7)] = value(T, 2, 0);
        matrix[at(layout, 4, 3, 7)] = value(T, 3, 0);
        var values: [7]R = undefined;
        var work: [256]T = undefined;
        var real: [256]R = undefined;
        var integers: [256]i32 = undefined;
        const required = try P.minimumWorkspace(i32, 7, wants, if (S.isComplex(T)) .complex else .real);
        const args: A.Arguments(T) = .{ .layout = layout, .jobz = if (wants) 'V' else 'N', .uplo = 'L', .n = 7, .lda = 7, .a = &matrix, .w = &values, .work = &work, .rwork = &real, .iwork = &integers, .lwork = required.work, .lrwork = required.rwork, .liwork = required.iwork, .native_options = .{ .max_sweeps = 0 } };
        try E.expectEqual(@as(i32, if (wants) 29 else 2), try A.workWithDiagnosticsAdmitted(T, &op, &rec.services(), args));
        try E.expectEqual(@as(usize, 0), rec.count);
        op.end();
        try finish(&owner, &alloc);
    };
}

test "EVD managed raw bridge captures the existing callbacks and closes its one operation" {
    inline for (types) |T| {
        var alloc: F.Allocations = .{ .budget = 0 };
        var owner = Ctx.Context.init(alloc.allocator(), F.policy, .internal_closable);
        var rec: Recorder = .{ .owner = &owner };
        const callback: B.shared_services.Callback = .{ .userdata = &rec, .function = Recorder.event };
        const args: Raw.Arguments(T) = .{ .layout = 99, .jobz = 'V', .uplo = 'L', .n = 0, .lda = 1 };
        try E.expectEqual(@as(i32, -1), try B.Shared.evdRawManagedWithDefaults(true, T, .high, &owner, args, .{ .lapacke = callback, .fortran_returning_test = callback }));
        try E.expectEqual(@as(usize, 1), rec.count);
        try E.expectEqual(@as(usize, 0), owner.snapshot().active_operations);
        try E.expectEqual(@as(usize, 0), alloc.stats().attempts);
        try finish(&owner, &alloc);
        try E.expectError(error.Closing, B.Shared.evdRawManagedWithDefaults(true, T, .high, &owner, args, .{}));
    }
}
