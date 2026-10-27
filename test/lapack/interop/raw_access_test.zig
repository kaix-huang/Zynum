const std = @import("std");
const owner = @import("numerical-owner");
const h = @import("managed_support.zig");
const s = h.s;
const Core = owner.Provider;
const C = Core.Execution.context;
const F = owner.Adapter;
const S = owner.SolveAdapter;
const RF = owner.RawAdapter;
const RS = owner.RawSolveAdapter;
const access = owner.common.access;
const cfg = owner.shared_services;

fn fault(expected: anyerror, result: anytype) !void {
    switch (result) {
        .admission_fault => |err| try std.testing.expectEqual(expected, @as(anyerror, err)),
        .lapacke_info => return error.ExpectedSeparateAccessFault,
    }
}
fn sameRig(a: *h.Rig, b: *h.Rig) !void {
    try std.testing.expectEqual(a.reads, b.reads);
    try std.testing.expectEqual(a.hooks, b.hooks);
    try std.testing.expectEqual(a.x.attempts, b.x.attempts);
    try std.testing.expectEqual(a.x.frees, b.x.frees);
    try std.testing.expectEqual(a.x.live, b.x.live);
    try std.testing.expectEqual(@as(usize, 0), a.y.attempts + b.y.attempts + a.z.attempts + b.z.attempts);
    try std.testing.expectEqual(a.count, b.count);
    for (a.events[0..a.count], b.events[0..b.count]) |x, y| try std.testing.expectEqual(x, y);
    try std.testing.expectEqual(a.context.snapshot(), b.context.snapshot());
}
fn bind(r: *h.Rig) void {
    const cb: cfg.Callback = .{ .userdata = r, .function = h.Rig.hookForTest };
    _ = cfg.setCallbacks(.{ .lapacke = cb, .fortran_returning_test = cb });
}
fn close(r: *h.Rig) void {
    r.done() catch @panic("raw access fixture cleanup");
}
fn expectedX(comptime T: type, row: usize, rhs: usize) T {
    return if (rhs == 0)
        (if (row == 0) s.value(T, 1, 0) else s.value(T, 2, -1))
    else
        (if (row == 0) s.value(T, -2, 1) else s.value(T, 3, 0.5));
}
fn independentRhs(comptime T: type, uplo: u8, row: usize, rhs: usize) T {
    // Independent f64 complex arithmetic on the original Hermitian matrix,
    // not a second solve or a call back into the numerical implementation.
    const off_im: f64 = if (Core.Scalar.isComplex(T)) 2 else 0;
    var re: f64 = 0;
    var im: f64 = 0;
    for (0..2) |k| {
        const ar: f64 = if (row == k) (if (row == 0) 4 else 10) else 2;
        const ai: f64 = if (row == k) 0 else if ((uplo == 'U') == (row == 0)) off_im else -off_im;
        const x = expectedX(T, k, rhs);
        re += ar * s.re(T, x) - ai * s.im(T, x);
        im += ar * s.im(T, x) + ai * s.re(T, x);
    }
    return s.value(T, re, im);
}

test "stage access preserves typed field and error types with distinct raw storage faults" {
    try std.testing.expect(owner.common.AdmissionFault == error{ IntegerNotRepresentable, SizeOverflow, MissingBuffer, BufferTooSmall, AliasingNotAllowed });
    inline for (.{ f32, f64, Core.Scalar.ComplexF32, Core.Scalar.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        try std.testing.expect(@FieldType(F.Arguments(T, I), "values") == ?[]T);
        try std.testing.expect(@FieldType(S.Arguments(T, I), "a") == ?[]const T);
        try std.testing.expect(@FieldType(S.Arguments(T, I), "b") == ?[]T);
        try std.testing.expect(@FieldType(F.Result(I), "admission_fault") == owner.common.AdmissionFault);
        try std.testing.expect(@FieldType(S.Result(I), "lapacke_info") == I);
        try std.testing.expect(@FieldType(RF.Arguments(T, I), "values") == ?[*]align(1) T);
        try std.testing.expect(@FieldType(RS.Arguments(T, I), "a") == ?[*]align(1) const T);
        try std.testing.expect(@FieldType(RS.Result(I), "admission_fault") == access.RawFault);
        try std.testing.expect(RF.OwnerIdentity == F.OwnerIdentity and RS.OwnerIdentity == S.OwnerIdentity);
        try std.testing.expect(RF.Dense == Core.Dense and RS.Services == Core.Services);
    };
    var data = [_]f64{7};
    const typed_empty = try owner.common.Helpers(Core).capacity(f64, &data, 0);
    try std.testing.expect(typed_empty.ptr == data[0..].ptr and typed_empty.len == 0);
    try std.testing.expectError(error.BufferTooSmall, owner.common.Helpers(Core).capacity(f64, data[0..0], 1));
    try std.testing.expectError(error.SizeOverflow, owner.common.Helpers(Core).capacity(f64, null, std.math.maxInt(usize)));
}

test "stage raw and typed real complex factor solve match padded arrays events and known solutions" {
    inline for (.{ f32, f64, Core.Scalar.ComplexF32, Core.Scalar.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 'L', 'U' }) |uplo| for ([_]owner.common.Level{ .high, .work }) |level| {
            var typed: h.Rig = .{};
            typed.init();
            defer close(&typed);
            var raw: h.Rig = .{};
            raw.init();
            defer close(&raw);
            var a: [6]T = @splat(s.value(T, 71, -3));
            a[s.pos(layout, 0, 0, 3)] = s.value(T, 4, 0);
            a[s.pos(layout, 1, 1, 3)] = s.value(T, 10, 0);
            const selected = s.pos(layout, if (uplo == 'L') 1 else 0, if (uplo == 'L') 0 else 1, 3);
            a[selected] = s.value(T, 2, 2);
            const before = a;
            var a_raw = a;
            try h.info(I, try F.callManaged(T, I, level, typed.context, &typed.services, .{ .layout = layout, .uplo = uplo, .n = 2, .lda = 3, .values = &a }), 0);
            try h.info(I, try RF.callManaged(T, I, level, raw.context, &raw.services, .{ .layout = layout, .uplo = uplo, .n = 2, .lda = 3, .values = a_raw[0..].ptr }), 0);
            try s.bytes(&a, &a_raw);
            try sameRig(&typed, &raw);
            try std.testing.expectEqual(s.value(T, 2, 0), a[0]);
            for (0..a.len) |i| if (i != 0 and i != 4 and i != selected) try std.testing.expectEqual(before[i], a[i]);
            const factor = a;
            const factor_raw = a_raw;
            var b: [6]T = @splat(s.value(T, 83, -7));
            for (0..2) |rhs| for (0..2) |row| {
                b[s.pos(layout, row, rhs, 3)] = independentRhs(T, uplo, row, rhs);
            };
            const b_before = b;
            var b_raw = b;
            try h.info(I, try S.callManaged(T, I, level, typed.context, &typed.services, .{ .layout = layout, .uplo = uplo, .n = 2, .nrhs = 2, .lda = 3, .ldb = 3, .a = &factor, .b = &b }), 0);
            try h.info(I, try RS.callManaged(T, I, level, raw.context, &raw.services, .{ .layout = layout, .uplo = uplo, .n = 2, .nrhs = 2, .lda = 3, .ldb = 3, .a = factor_raw[0..].ptr, .b = b_raw[0..].ptr }), 0);
            try s.bytes(&b, &b_raw);
            try s.bytes(&a, &factor);
            try s.bytes(&a_raw, &factor_raw);
            try sameRig(&typed, &raw);
            for (0..2) |rhs| for (0..2) |row| {
                const found = b[s.pos(layout, row, rhs, 3)];
                const want = expectedX(T, row, rhs);
                const tolerance: f64 = if (T == f32 or T == Core.Scalar.ComplexF32) 2e-6 else 2e-14;
                try std.testing.expectApproxEqAbs(s.re(T, want), s.re(T, found), tolerance);
                try std.testing.expectApproxEqAbs(s.im(T, want), s.im(T, found), tolerance);
            };
            try std.testing.expectEqual(b_before[2], b[2]);
            try std.testing.expectEqual(b_before[5], b[5]);
        };
    };
}

test "stage raw invalid layout precedes pointer conversion getter and extent" {
    var r: h.Rig = .{};
    r.init();
    defer close(&r);
    var bytes: [16]u8 align(8) = @splat(0);
    const misaligned: [*]align(1) f64 = @ptrCast(bytes[1..].ptr);
    try h.info(i64, try RF.callManaged(f64, i64, .high, r.context, &r.services, .{ .layout = -1, .uplo = '?', .n = std.math.maxInt(i64), .lda = std.math.maxInt(i64), .values = misaligned }), -1);
    try h.info(i64, try RS.callManaged(f64, i64, .high, r.context, &r.services, .{ .layout = -1, .uplo = '?', .n = std.math.maxInt(i64), .nrhs = std.math.maxInt(i64), .lda = 0, .ldb = 0, .a = misaligned, .b = null }), -1);
    try std.testing.expectEqual(@as(usize, 0), r.reads);
    try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 2), r.hooks);
    for (bytes) |value| try std.testing.expectEqual(@as(u8, 0), value);
}

test "stage raw truncated high scans and A then B NaN precede later argument faults" {
    var typed: h.Rig = .{};
    typed.init();
    defer close(&typed);
    var raw: h.Rig = .{};
    raw.init();
    defer close(&raw);
    const a = [_]f64{ std.math.nan(f64), 2 };
    var b = [_]f64{ std.math.nan(f64), 3 };
    // n=2, lda=1 is illegal for the work call but the high scan visits A[0].
    var copy = a;
    try h.info(i64, try F.callManaged(f64, i64, .high, typed.context, &typed.services, .{ .layout = 102, .uplo = 'L', .n = 2, .lda = 1, .values = &copy }), -4);
    try h.info(i64, try RF.callManaged(f64, i64, .high, raw.context, &raw.services, .{ .layout = 102, .uplo = 'L', .n = 2, .lda = 1, .values = copy[0..].ptr }), -4);
    try h.info(i64, try S.callManaged(f64, i64, .high, typed.context, &typed.services, .{ .layout = 102, .uplo = 'L', .n = 2, .nrhs = 1, .lda = 1, .ldb = 1, .a = &a, .b = &b }), -5);
    try h.info(i64, try RS.callManaged(f64, i64, .high, raw.context, &raw.services, .{ .layout = 102, .uplo = 'L', .n = 2, .nrhs = 1, .lda = 1, .ldb = 1, .a = a[0..].ptr, .b = b[0..].ptr }), -5);
    try h.info(i64, try S.callManaged(f64, i64, .high, typed.context, &typed.services, .{ .layout = 102, .uplo = '?', .n = 2, .nrhs = 1, .lda = 1, .ldb = 1, .a = null, .b = &b }), -7);
    try h.info(i64, try RS.callManaged(f64, i64, .high, raw.context, &raw.services, .{ .layout = 102, .uplo = '?', .n = 2, .nrhs = 1, .lda = 1, .ldb = 1, .a = null, .b = b[0..].ptr }), -7);
    try sameRig(&typed, &raw);
    try std.testing.expectEqual(@as(usize, 0), raw.hooks);
    try std.testing.expectEqual(@as(usize, 0), raw.x.attempts);
}

test "stage raw null nonreads and column empty checks preserve original priority" {
    var r: h.Rig = .{};
    r.init();
    defer close(&r);
    try h.info(i64, try RF.callManaged(f64, i64, .high, r.context, &r.services, .{ .layout = 102, .uplo = '?', .n = -1, .lda = -1, .values = null }), -2);
    try h.info(i64, try RF.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = 102, .uplo = 'L', .n = 0, .lda = 0, .values = null }), -5);
    try h.info(i64, try RF.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = 102, .uplo = 'L', .n = 0, .lda = 1, .values = null }), 0);
    try fault(error.MissingBuffer, try RF.callManaged(f64, i64, .high, r.context, &r.services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = null }));
    const nan_a = [_]f64{std.math.nan(f64)};
    try h.info(i64, try RS.callManaged(f64, i64, .high, r.context, &r.services, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 0, .lda = 1, .ldb = 1, .a = nan_a[0..].ptr, .b = null }), -5);
    try h.info(i64, try RS.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 0, .lda = 1, .ldb = 1, .a = null, .b = null }), 0);
    try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 3), r.reads);
}

test "stage raw row empty calls allocate and zero RHS still accesses factor" {
    var r: h.Rig = .{};
    r.init();
    defer close(&r);
    try h.info(i64, try RF.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = 101, .uplo = 'L', .n = 0, .lda = 0, .values = null }), 0);
    try h.info(i64, try RS.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = 101, .uplo = 'L', .n = 0, .nrhs = 0, .lda = 0, .ldb = 0, .a = null, .b = null }), 0);
    try std.testing.expectEqual(@as(usize, 3), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 3), r.x.frees);
    try fault(error.MissingBuffer, try RS.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = 101, .uplo = 'L', .n = 1, .nrhs = 0, .lda = 1, .ldb = 0, .a = null, .b = null }));
    try std.testing.expectEqual(@as(usize, 5), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 5), r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.hooks);
}

test "stage raw actual row allocation failures precede pointer and uplo checks" {
    for ([_]bool{ false, true }) |solve| {
        var r: h.Rig = .{};
        r.init();
        defer close(&r);
        r.x.fail_at = if (solve) 1 else 0;
        if (solve) {
            try h.info(i64, try RS.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = 101, .uplo = '?', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = null, .b = null }), -1011);
        } else {
            try h.info(i64, try RF.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = 101, .uplo = '?', .n = 1, .lda = 1, .values = null }), -1011);
        }
        try std.testing.expectEqual(@as(usize, if (solve) 2 else 1), r.x.attempts);
        try std.testing.expectEqual(@as(usize, if (solve) 1 else 0), r.x.frees);
        try std.testing.expectEqual(@as(usize, 1), r.hooks);
        var seen_free = false;
        for (r.events[0..r.count]) |event| {
            if (event.tag == .free) seen_free = true;
            if (event.tag == .hook) {
                try std.testing.expectEqual(@as(i32, -1011), event.detail);
                try std.testing.expectEqual(@as(usize, 0), event.pins);
                try std.testing.expectEqual(solve, seen_free);
            }
        }
    }
}

test "stage raw second temporary extent fault follows first allocation and cleanup" {
    var typed: h.Rig = .{};
    typed.init();
    defer close(&typed);
    var raw: h.Rig = .{};
    raw.init();
    defer close(&raw);
    const huge = std.math.maxInt(i64);
    const expected: anyerror = if (@bitSizeOf(usize) == 64) error.SizeOverflow else error.IntegerNotRepresentable;
    try fault(expected, try S.callManaged(f64, i64, .work, typed.context, &typed.services, .{ .layout = 101, .uplo = '?', .n = 1, .nrhs = huge, .lda = 1, .ldb = huge, .a = null, .b = null }));
    try fault(expected, try RS.callManaged(f64, i64, .work, raw.context, &raw.services, .{ .layout = 101, .uplo = '?', .n = 1, .nrhs = huge, .lda = 1, .ldb = huge, .a = null, .b = null }));
    try sameRig(&typed, &raw);
    try std.testing.expectEqual(@as(usize, 1), raw.x.attempts);
    try std.testing.expectEqual(@as(usize, 1), raw.x.frees);
    try std.testing.expectEqual(@as(usize, 0), raw.hooks);
}

test "stage raw invalid uplo still copies RHS back before reverse release" {
    var typed: h.Rig = .{};
    typed.init();
    defer close(&typed);
    var raw: h.Rig = .{};
    raw.init();
    defer close(&raw);
    var b = [_]f64{ 10, 71, 20 };
    var b_raw = b;
    typed.b = &b;
    raw.b = &b_raw;
    typed.expect_copyback = true;
    raw.expect_copyback = true;
    try h.info(i64, try S.callManaged(f64, i64, .work, typed.context, &typed.services, .{ .layout = 101, .uplo = '?', .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = null, .b = &b }), -2);
    try h.info(i64, try RS.callManaged(f64, i64, .work, raw.context, &raw.services, .{ .layout = 101, .uplo = '?', .n = 2, .nrhs = 1, .lda = 2, .ldb = 2, .a = null, .b = b_raw[0..].ptr }), -2);
    try s.bytes(&b, &b_raw);
    try std.testing.expectEqualSlices(f64, &.{ 10, -900, 20 }, &b);
    try sameRig(&typed, &raw);
    try std.testing.expectEqual(@as(usize, 2), raw.x.frees);
}

test "stage raw alignment size and endpoint faults never construct an invalid nonempty span" {
    const H = owner.common.HelpersWithAccess(Core, .raw);
    var bytes: [16]u8 align(8) = @splat(0);
    const misaligned: [*]align(1) f64 = @ptrCast(bytes[1..].ptr);
    try std.testing.expectEqual(@as(usize, 0), (try H.capacity(f64, misaligned, 0)).len);
    try std.testing.expectError(error.SizeOverflow, H.capacity(f64, null, std.math.maxInt(usize)));
    try std.testing.expectError(error.MissingBuffer, H.capacity(f64, null, 1));
    try std.testing.expectError(error.MisalignedPointer, H.capacity(f64, misaligned, 1));
    // Synthetic address is used solely in a guaranteed checked-endpoint reject;
    // no memory read/write or claimed valid allocation is involved.
    const wrapping: [*]align(1) const f64 = @ptrFromInt(std.math.maxInt(usize) - 7);
    try std.testing.expectError(error.SizeOverflow, H.constCapacity(f64, wrapping, 1));
    // Both checks would fail; alignment must win without a dereference.
    const misaligned_wrapping: [*]align(1) const f64 = @ptrFromInt(std.math.maxInt(usize) - 6);
    try std.testing.expectError(error.MisalignedPointer, H.constCapacity(f64, misaligned_wrapping, 1));
    var data = [_]f64{7};
    const span = try H.capacity(f64, data[0..].ptr, 1);
    try std.testing.expect(span.ptr == data[0..].ptr and span.len == 1);
    span[0] = 9;
    try std.testing.expectEqual(@as(f64, 9), data[0]);
}

test "stage raw factor RHS overlap includes intervening padding and excludes trailing suffix" {
    for ([_]c_int{ 101, 102 }) |layout| {
        var r: h.Rig = .{};
        r.init();
        defer close(&r);
        var a = [_]f64{ 2, 0, 71, 0, 3, 12, 18 };
        const before = a;
        const overlap_b: [*]align(1) f64 = a[2..].ptr;
        try fault(error.AliasingNotAllowed, try RS.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = layout, .uplo = 'L', .n = 2, .nrhs = 1, .lda = 3, .ldb = if (layout == 101) 1 else 2, .a = a[0..].ptr, .b = overlap_b }));
        try s.bytes(&before, &a);
        try h.info(i64, try RS.callManaged(f64, i64, .work, r.context, &r.services, .{ .layout = layout, .uplo = 'L', .n = 2, .nrhs = 1, .lda = 3, .ldb = if (layout == 101) 1 else 2, .a = a[0..].ptr, .b = a[5..].ptr }), 0);
        try std.testing.expectEqualSlices(f64, before[0..5], a[0..5]);
        try std.testing.expectEqual(@as(f64, 3), a[5]);
        try std.testing.expectEqual(@as(f64, 2), a[6]);
    }
}

test "stage raw shared routes capture once preserve owner and bypass getter for work or no scan" {
    defer cfg.testing.reset();
    inline for (.{ true, false }) |scan| for ([_]owner.common.Level{ .high, .work }) |level| {
        cfg.testing.reset();
        cfg.setNancheck(0);
        var r: h.Rig = .{};
        r.init();
        defer close(&r);
        var a = [_]f64{4};
        var b = [_]f64{12};
        bind(&r);
        try h.info(i64, try owner.Shared.factorRawManaged(scan, f64, i64, level, r.context, .{ .layout = 101, .uplo = 'L', .n = 1, .lda = 1, .values = a[0..].ptr }), 0);
        try h.info(i64, try owner.Shared.solveRawManaged(scan, f64, i64, level, r.context, .{ .layout = 101, .uplo = 'U', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = a[0..].ptr, .b = b[0..].ptr }), 0);
        try std.testing.expectEqual(@as(f64, 3), b[0]);
        try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().captures);
        try std.testing.expectEqual(@as(usize, if (scan and level == .high) 2 else 0), cfg.testing.snapshot().getter_calls);
        try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().environment_reads);
        try std.testing.expectEqual(@as(usize, 1), r.max_active);
        try std.testing.expectEqual(@as(usize, 3), r.x.attempts);
        try std.testing.expectEqual(@as(usize, 3), r.x.frees);
    };
    cfg.testing.reset();
    cfg.setNancheck(0);
    const ctx = Core.Execution.getContext();
    const before = ctx.snapshot();
    var a = [_]f64{4};
    var b = [_]f64{12};
    try h.info(i32, try owner.Shared.factorRawDefault(true, f64, i32, .high, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = a[0..].ptr }), 0);
    try h.info(i64, try owner.Shared.solveRawDefault(true, f64, i64, .high, .{ .layout = 102, .uplo = 'L', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = a[0..].ptr, .b = b[0..].ptr }), 0);
    try std.testing.expectEqual(@as(f64, 3), b[0]);
    try std.testing.expectEqual(before, ctx.snapshot());
    try std.testing.expectEqual(@as(usize, 2), cfg.testing.snapshot().captures);
}

test "stage raw admission refuses before shared capture and borrowed services or pointers" {
    defer cfg.testing.reset();
    for ([_]bool{ false, true }) |overflow| {
        cfg.testing.reset();
        var r: h.Rig = .{};
        r.init();
        defer close(&r);
        if (overflow) C.Context.testing.configure(r.context, 0, null) else _ = r.context.requestClose();
        const expected = if (overflow) error.CounterOverflow else error.Closing;
        var services: Core.Services = undefined;
        try std.testing.expectError(expected, RF.callManaged(f64, i64, .high, r.context, &services, .{ .layout = -1, .uplo = '?', .n = 1, .lda = 1, .values = null }));
        try std.testing.expectError(expected, owner.Shared.solveRawManaged(true, f64, i64, .high, r.context, .{ .layout = -1, .uplo = '?', .n = 1, .nrhs = 1, .lda = 1, .ldb = 1, .a = null, .b = null }));
        try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().captures);
        try std.testing.expectEqual(@as(usize, 0), cfg.testing.snapshot().getter_calls);
        try std.testing.expectEqual(@as(usize, 0), r.x.attempts);
        try std.testing.expectEqual(@as(usize, 0), r.hooks);
    }
}

test "stage raw capture ignores rebound allocator and keeps cleanup inside the admitted call" {
    var r: h.Rig = .{};
    r.init();
    defer close(&r);
    r.rebind = true;
    var a = [_]f64{4};
    try h.info(i64, try RF.callManaged(f64, i64, .high, r.context, &r.services, .{ .layout = 101, .uplo = '?', .n = 1, .lda = 1, .values = a[0..].ptr }), -2);
    try std.testing.expectEqual(@as(usize, 1), r.hooks);
    r.services.allocator = undefined;
    r.services.error_event = h.Rig.hookForTest;
    {
        var operation = try r.context.begin();
        defer operation.end();
        try h.info(i64, try RF.callAdmitted(f64, i64, .work, &operation, &r.services, .{ .layout = 101, .uplo = 'L', .n = 1, .lda = 1, .values = a[0..].ptr }), 0);
        try std.testing.expectEqual(@as(usize, 1), r.context.snapshot().active_operations);
        try std.testing.expectEqual(@as(usize, 0), r.context.snapshot().resource_pins);
    }
    try std.testing.expectEqual(@as(f64, 2), a[0]);
    try std.testing.expectEqual(@as(usize, 2), r.x.attempts);
    try std.testing.expectEqual(@as(usize, 2), r.x.frees);
    try std.testing.expectEqual(@as(usize, 0), r.y.attempts + r.z.attempts);
    try std.testing.expectEqual(@as(usize, 1), r.max_active);
}
