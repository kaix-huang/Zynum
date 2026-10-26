const std = @import("std");
// Exactly one numerical owner import; no numerical file is imported by this root.
const owner = @import("numerical-owner");
const Core = owner.Core;
const A = owner.Adapter;
const Event = owner.protocol.ErrorEvent;
const C32 = owner.implementation.ComplexF32;
const C64 = owner.implementation.ComplexF64;

const Context = struct {
    enabled: bool = true,
    reads: usize = 0,
    hooks: [8]Event = undefined,
    hook_count: usize = 0,
    allocator: ?*std.testing.FailingAllocator = null,
    live_at_hook: usize = 0,
    services_to_change: ?*Core.Services = null,
    replacement: ?std.mem.Allocator = null,
    fn get(raw: *anyopaque) bool {
        const self: *Context = @ptrCast(@alignCast(raw));
        self.reads += 1;
        if (self.services_to_change) |binding| {
            binding.allocator = self.replacement.?;
            binding.error_event = unexpectedHook;
        }
        return self.enabled;
    }
    fn event(raw: *anyopaque, e: Event) void {
        const self: *Context = @ptrCast(@alignCast(raw));
        self.hooks[self.hook_count] = e;
        self.hook_count += 1;
        if (self.allocator) |allocator| self.live_at_hook = allocator.allocations - allocator.deallocations;
    }
    fn unexpectedHook(_: *anyopaque, _: Event) void {
        @panic("adapter re-read a changed service binding");
    }
    fn services(self: *Context, allocator: std.mem.Allocator) Core.Services {
        return .{ .allocator = allocator, .context = self, .nan_enabled = get, .error_event = event };
    }
};
fn expectInfo(comptime I: type, expected: I, actual: A.Result(I)) !void {
    switch (actual) {
        .lapacke_info => |info| try std.testing.expectEqual(expected, info),
        .admission_fault => |fault| {
            std.debug.print("unexpected admission: {s}\n", .{@errorName(fault)});
            return error.TestExpectedEqual;
        },
    }
}
fn expectFault(comptime I: type, expected: owner.protocol.AdmissionFault, actual: A.Result(I)) !void {
    switch (actual) {
        .admission_fault => |fault| try std.testing.expectEqual(expected, fault),
        .lapacke_info => return error.TestExpectedEqual,
    }
}
fn val(comptime T: type, re: f64, im: f64) T {
    if (comptime Core.Scalar.isComplex(T)) return .{ .re = @floatCast(re), .im = @floatCast(im) };
    return @floatCast(re);
}
fn real(comptime T: type, value: T) f64 {
    return @floatCast(if (comptime Core.Scalar.isComplex(T)) value.re else value);
}
fn imag(comptime T: type, value: T) f64 {
    return if (comptime Core.Scalar.isComplex(T)) @floatCast(value.im) else 0;
}
fn pos(layout: c_int, row: usize, column: usize, ld: usize) usize {
    return if (layout == 101) row * ld + column else row + column * ld;
}
fn bytes(comptime T: type, expected: []const T, actual: []const T) !void {
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual));
}

test "adapter aliases one owner scalar dense lease and service identity and calls checked owner" {
    try std.testing.expect(A.OwnerIdentity == Core.OwnerIdentity);
    try std.testing.expect(A.Services == Core.Services and owner.NoScanAdapter.Services == Core.Services);
    inline for (.{ f32, f64, C32, C64 }) |T| {
        try std.testing.expect(A.Dense.Matrix(T) == Core.Dense.Matrix(T));
        try std.testing.expect(A.ScratchLease(T) == Core.Resources.ScratchLease(T));
    }
    try std.testing.expect(C32 == Core.Scalar.ComplexF32 and C64 == Core.Scalar.ComplexF64);
    try std.testing.expect(C64 == owner.implementation.lapack.ComplexF64);
    var ctx = Context{};
    var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const services = ctx.services(allocator.allocator());
    var a = [_]f64{4};
    try expectInfo(i32, 0, A.call(f64, i32, .work, &services, .{ .layout = 101, .uplo = 'L', .n = 1, .lda = 1, .values = &a }));
    try std.testing.expectEqual(@as(f64, 2), a[0]);
    var z = [_]C64{.{ .re = 4, .im = 0 }};
    const checked = owner.implementation.lapack;
    try std.testing.expectEqual(@as(usize, 0), try checked.choleskyFactorize(C64, try checked.matrix(C64, &z, .{ .row_count = 1, .column_count = 1, .leading_dimension = 1 }), .lower));
    try std.testing.expectEqual(@as(f64, 2), z[0].re);
    try std.testing.expectEqual(@as(usize, 1), allocator.allocations);
    try std.testing.expectEqual(@as(usize, 1), allocator.deallocations);
}

test "sdcz integer widths levels layouts triangles reconstruct independently and preserve poison padding" {
    // Literal A=L*L^H from exact rational fixture, independent of factor loops.
    const real_a = [3][3]f64{ .{ 4, 2, -2 }, .{ 2, 10, 0.5 }, .{ -2, 0.5, 5.25 } };
    const complex_re = [3][3]f64{ .{ 4, 2, -2 }, .{ 2, 11, 2.5 }, .{ -2, 2.5, 9.5 } };
    const complex_im = [3][3]f64{ .{ 0, -2, -4 }, .{ 2, 0, -1.5 }, .{ 4, 1.5, 0 } };
    inline for (.{ f32, f64, C32, C64 }) |T| inline for (.{ i32, i64 }) |I| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]bool{ false, true }) |upper| {
            inline for (.{ owner.protocol.Level.high, owner.protocol.Level.work }) |level| {
                var matrix: [15]T = @splat(val(T, 91, -73));
                for (0..3) |r| for (0..3) |c| {
                    const selected = if (upper) r <= c else r >= c;
                    matrix[pos(layout, r, c, 5)] = if (selected)
                        val(T, if (Core.Scalar.isComplex(T)) complex_re[r][c] else real_a[r][c], if (Core.Scalar.isComplex(T)) complex_im[r][c] else 0)
                    else
                        val(T, std.math.nan(f64), std.math.nan(f64));
                };
                const before = matrix;
                var ctx = Context{};
                var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
                const services = ctx.services(allocator.allocator());
                try expectInfo(I, 0, A.call(T, I, level, &services, .{ .layout = layout, .uplo = if (upper) 'u' else 'l', .n = 3, .lda = 5, .values = &matrix }));
                try std.testing.expectEqual(@as(usize, if (level == .high) 1 else 0), ctx.reads);
                try std.testing.expectEqual(@as(usize, if (layout == 101) 1 else 0), allocator.allocations);
                try std.testing.expectEqual(allocator.allocations, allocator.deallocations);
                // Reconstruct with a separate dense product in f64, retrieving
                // logical L from the selected stored triangle. No factor algorithm.
                for (0..3) |r| for (0..3) |c| {
                    var rr: f64 = 0;
                    var ii: f64 = 0;
                    for (0..3) |k| {
                        var xr: f64 = 0;
                        var xi: f64 = 0;
                        var yr: f64 = 0;
                        var yi: f64 = 0;
                        if (k <= r) {
                            const x = matrix[pos(layout, if (upper) k else r, if (upper) r else k, 5)];
                            xr = real(T, x);
                            xi = imag(T, x) * (if (upper) @as(f64, -1) else 1);
                        }
                        if (k <= c) {
                            const y = matrix[pos(layout, if (upper) k else c, if (upper) c else k, 5)];
                            yr = real(T, y);
                            yi = imag(T, y) * (if (upper) @as(f64, -1) else 1);
                        }
                        rr += xr * yr + xi * yi;
                        ii += xi * yr - xr * yi;
                    }
                    try std.testing.expectApproxEqAbs(if (Core.Scalar.isComplex(T)) complex_re[r][c] else real_a[r][c], rr, 0.00002);
                    try std.testing.expectApproxEqAbs(if (Core.Scalar.isComplex(T)) complex_im[r][c] else @as(f64, 0), ii, 0.00002);
                };
                for (0..15) |p| {
                    const r = if (layout == 101) p / 5 else p % 5;
                    const c = if (layout == 101) p % 5 else p / 5;
                    if (r >= 3 or c >= 3 or (if (upper) r > c else r < c)) try bytes(T, before[p..][0..1], matrix[p..][0..1]);
                }
            }
        };
    };
}

test "layout and ordered column scalar errors have distinct returning Fortran events" {
    inline for (.{ i32, i64 }) |I| {
        const Case = struct { layout: c_int, uplo: u8, n: I, ld: I, info: I, arg: i32, kind: owner.protocol.HookKind };
        const cases = [_]Case{
            .{ .layout = 0, .uplo = '?', .n = -1, .ld = 0, .info = -1, .arg = -1, .kind = .lapacke },
            .{ .layout = 102, .uplo = '?', .n = -1, .ld = 0, .info = -2, .arg = 1, .kind = .fortran_returning_test },
            .{ .layout = 102, .uplo = 'L', .n = -1, .ld = 0, .info = -3, .arg = 2, .kind = .fortran_returning_test },
            .{ .layout = 102, .uplo = 'L', .n = 0, .ld = 0, .info = -5, .arg = 4, .kind = .fortran_returning_test },
        };
        for (cases) |case| {
            var ctx = Context{};
            const services = ctx.services(std.testing.failing_allocator);
            try expectInfo(I, case.info, A.call(f64, I, .high, &services, .{ .layout = case.layout, .uplo = case.uplo, .n = case.n, .lda = case.ld, .values = null }));
            try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
            try std.testing.expectEqual(case.kind, ctx.hooks[0].kind);
            try std.testing.expectEqual(case.arg, ctx.hooks[0].argument);
            try std.testing.expectEqual(@as(usize, if (case.layout == 0) 0 else 1), ctx.reads);
        }
    }
}

test "row preliminary leading dimension and allocation precede second stage parameter errors" {
    inline for (.{ i32, i64 }) |I| {
        var ctx = Context{};
        var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
        ctx.allocator = &allocator;
        const services = ctx.services(allocator.allocator());
        try expectInfo(I, -5, A.call(f64, I, .work, &services, .{ .layout = 101, .uplo = '?', .n = 2, .lda = 1, .values = null }));
        try std.testing.expectEqual(@as(usize, 0), allocator.allocations);
        try std.testing.expectEqual(owner.protocol.Entry.work, ctx.hooks[0].entry);
        ctx.hook_count = 0;
        try expectInfo(I, -2, A.call(f64, I, .work, &services, .{ .layout = 101, .uplo = '?', .n = -1, .lda = 0, .values = null }));
        try std.testing.expectEqual(@as(usize, 1), allocator.allocations);
        try std.testing.expectEqual(@as(usize, 1), allocator.deallocations);
        try std.testing.expectEqual(@as(usize, 1), ctx.live_at_hook);
        try std.testing.expectEqual(@as(i32, 1), ctx.hooks[0].argument);
        ctx.hook_count = 0;
        try expectInfo(I, -3, A.call(f64, I, .work, &services, .{ .layout = 101, .uplo = 'U', .n = -1, .lda = 0, .values = null }));
        try std.testing.expectEqual(@as(i32, 2), ctx.hooks[0].argument);
        try std.testing.expectEqual(allocator.allocations, allocator.deallocations);
    }
}

test "row empty calls allocate one owner scalar and allocation failure masks later errors" {
    inline for (.{ f32, f64, C32, C64 }) |T| inline for (.{ i32, i64 }) |I| {
        var ctx = Context{};
        var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
        const services = ctx.services(allocator.allocator());
        try expectInfo(I, 0, A.call(T, I, .work, &services, .{ .layout = 101, .uplo = 'U', .n = 0, .lda = 0, .values = null }));
        try std.testing.expectEqual(@sizeOf(T), allocator.allocated_bytes);
        try std.testing.expectEqual(allocator.allocated_bytes, allocator.freed_bytes);
        try std.testing.expectEqual(@as(usize, 1), allocator.allocations);
        try expectInfo(I, 0, A.call(T, I, .work, &services, .{ .layout = 102, .uplo = 'U', .n = 0, .lda = std.math.maxInt(I), .values = null }));
        try std.testing.expectEqual(@as(usize, 1), allocator.allocations);
        var failure = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
        ctx.allocator = &failure;
        const failing = ctx.services(failure.allocator());
        try expectInfo(I, -1011, A.call(T, I, .work, &failing, .{ .layout = 101, .uplo = '?', .n = -1, .lda = 0, .values = null }));
        try std.testing.expect(failure.has_induced_failure);
        try std.testing.expectEqual(@as(usize, 0), failure.deallocations);
        try std.testing.expectEqual(@as(usize, 0), ctx.live_at_hook);
        try std.testing.expectEqual(@as(i32, -1011), ctx.hooks[0].argument);
    };
}

test "high NaN scans precede later lda error and include only helper clipped spans" {
    inline for (.{ f32, f64, C32, C64 }) |T| {
        for ([_]c_int{ 101, 102 }) |layout| for ([_]u8{ 'U', 'L' }) |uplo| {
            var values = [_]T{ val(T, std.math.nan(f64), 0), val(T, 7, 0), val(T, 8, 0), val(T, 9, 0) };
            const before = values;
            var ctx = Context{};
            const services = ctx.services(std.testing.failing_allocator);
            try expectInfo(i32, -4, A.call(T, i32, .high, &services, .{ .layout = layout, .uplo = uplo, .n = 3, .lda = 1, .values = &values }));
            try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
            try bytes(T, &before, &values);
            try expectInfo(i32, -5, A.call(T, i32, .work, &services, .{ .layout = layout, .uplo = uplo, .n = 3, .lda = 1, .values = &values }));
        };
    }
}

test "NaN getter state shared across scalars while work and disabled build bypass reads" {
    var ctx = Context{};
    const services = ctx.services(std.testing.failing_allocator);
    inline for (.{ f32, f64, C32, C64 }) |T| {
        var x = [_]T{val(T, std.math.nan(f64), 0)};
        ctx.enabled = true;
        try expectInfo(i64, -4, A.call(T, i64, .high, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = &x }));
        ctx.enabled = false;
        try expectInfo(i64, 1, A.call(T, i64, .high, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = &x }));
        const reads = ctx.reads;
        try expectInfo(i64, 1, A.call(T, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = &x }));
        const disabled = owner.NoScanAdapter.call(T, i64, .high, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = &x });
        try std.testing.expectEqual(@as(i64, 1), disabled.lapacke_info);
        try std.testing.expectEqual(reads, ctx.reads);
    }
    try std.testing.expectEqual(@as(usize, 8), ctx.reads);
    try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
}

test "complex diagonal imaginary NaN differs between high and work without checked policy leakage" {
    inline for (.{ C32, C64 }) |T| for ([_]c_int{ 101, 102 }) |layout| {
        var ctx = Context{};
        const services = ctx.services(std.testing.allocator);
        var x = [_]T{val(T, 4, std.math.nan(f64))};
        const before = x;
        try expectInfo(i32, -4, A.call(T, i32, .high, &services, .{ .layout = layout, .uplo = 'L', .n = 1, .lda = 1, .values = &x }));
        try bytes(T, &before, &x);
        try expectInfo(i32, 0, A.call(T, i32, .work, &services, .{ .layout = layout, .uplo = 'L', .n = 1, .lda = 1, .values = &x }));
        try bytes(T, &.{val(T, 2, 0)}, &x);
        x[0] = val(T, 4, std.math.inf(f64));
        try expectInfo(i32, 0, A.call(T, i32, .high, &services, .{ .layout = layout, .uplo = 'L', .n = 1, .lda = 1, .values = &x }));
        try bytes(T, &.{val(T, 2, 0)}, &x);
        x[0] = val(T, 4, 1);
        const lapack = owner.implementation.lapack;
        try std.testing.expectError(error.InvalidHermitianDiagonal, lapack.choleskyFactorize(T, try lapack.matrix(T, &x, .{ .row_count = 1, .column_count = 1, .leading_dimension = 1 }), .lower));
        try bytes(T, &.{val(T, 4, 1)}, &x);
    };
}

test "numerical INFO partial row copyback equals shared core result without recursive tail claim" {
    inline for (.{ f32, f64, C32, C64 }) |T| for ([_]bool{ false, true }) |upper| {
        var row: [8]T = @splat(val(T, 99, -81));
        var col = row;
        for (0..2) |r| for (0..2) |c| {
            const x = val(T, if (r == 0 and c == 0) 4 else if (r == 1 and c == 1) 0 else 2, 0);
            row[pos(101, r, c, 4)] = x;
            col[pos(102, r, c, 4)] = x;
        };
        const before = row;
        var ctx = Context{};
        const services = ctx.services(std.testing.allocator);
        try expectInfo(i64, 2, A.call(T, i64, .work, &services, .{ .layout = 101, .uplo = if (upper) 'U' else 'L', .n = 2, .lda = 4, .values = &row }));
        const info = Core.factorPotrf(T, .{ .values = &col, .row_count = 2, .column_count = 2, .leading_dimension = 4 }, upper);
        try std.testing.expectEqual(@as(usize, 2), info);
        for (0..2) |r| for (0..2) |c| {
            const p = pos(101, r, c, 4);
            if (if (upper) r <= c else r >= c) try bytes(T, col[pos(102, r, c, 4)..][0..1], row[p..][0..1]) else try bytes(T, before[p..][0..1], row[p..][0..1]);
        };
        try std.testing.expectEqual(@as(f64, 2), real(T, row[0]));
        try std.testing.expectEqual(@as(f64, 1), real(T, row[if (upper) 1 else 4]));
        try std.testing.expectEqual(@as(f64, 0), real(T, row[5]));
        try bytes(T, before[2..4], row[2..4]);
        try bytes(T, before[6..8], row[6..8]);
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
    };
}

test "IEEE pivot rules and signed zero rejected storage remain distinct from argument errors" {
    inline for (.{ f32, f64, C32, C64 }) |T| {
        for ([_]f64{ -0.0, 0, -1, -std.math.inf(f64), std.math.nan(f64) }) |pivot| {
            var x = [_]T{val(T, pivot, -0.0)};
            const before = x;
            var ctx = Context{};
            const services = ctx.services(std.testing.allocator);
            try expectInfo(i32, 1, A.call(T, i32, .work, &services, .{ .layout = 101, .uplo = 'U', .n = 1, .lda = 1, .values = &x }));
            try bytes(T, &before, &x);
            try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
        }
        var x = [_]T{val(T, std.math.inf(f64), -0.0)};
        var ctx = Context{};
        const services = ctx.services(std.testing.failing_allocator);
        try expectInfo(i32, 0, A.call(T, i32, .high, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = &x }));
        try bytes(T, &.{val(T, std.math.inf(f64), 0)}, &x);
    }
}

test "admission faults are separate from INFO with cleanup and no partial caller writes" {
    inline for (.{ i32, i64 }) |I| {
        var ctx = Context{};
        var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
        const services = ctx.services(allocator.allocator());
        var x = [_]f64{4};
        try expectFault(I, error.BufferTooSmall, A.call(f64, I, .work, &services, .{ .layout = 101, .uplo = 'L', .n = 2, .lda = 2, .values = &x }));
        try std.testing.expectEqual(@as(f64, 4), x[0]);
        try std.testing.expectEqual(@as(usize, 1), allocator.allocations);
        try std.testing.expectEqual(allocator.allocations, allocator.deallocations);
        try expectFault(I, error.MissingBuffer, A.call(f64, I, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = 1, .values = null }));
        try expectFault(I, if (@bitSizeOf(usize) < @bitSizeOf(I)) error.IntegerNotRepresentable else error.SizeOverflow, A.call(f64, I, .work, &services, .{ .layout = 101, .uplo = 'L', .n = std.math.maxInt(I), .lda = std.math.maxInt(I), .values = null }));
        try std.testing.expectEqual(@as(usize, 0), ctx.hook_count);
    }
}

test "service binding and lease allocator captured before NaN getter can change caller variables" {
    var good = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var bad = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var ctx = Context{ .allocator = &good, .replacement = bad.allocator() };
    var services = ctx.services(good.allocator());
    ctx.services_to_change = &services;
    try expectInfo(i64, -2, A.call(f64, i64, .high, &services, .{ .layout = 101, .uplo = '?', .n = 1, .lda = 1, .values = null }));
    try std.testing.expectEqual(@as(usize, 1), ctx.reads);
    try std.testing.expectEqual(@as(usize, 1), ctx.hook_count);
    try std.testing.expectEqual(@as(usize, 1), good.allocations);
    try std.testing.expectEqual(@as(usize, 1), good.deallocations);
    try std.testing.expectEqual(@as(usize, 1), ctx.live_at_hook);
    try std.testing.expect(!bad.has_induced_failure);
    try std.testing.expectEqual(bad.allocator().ptr, services.allocator.ptr);
}

test "native width admission includes real one element huge stride and target32 i64 boundaries" {
    var ctx = Context{};
    const services = ctx.services(std.testing.failing_allocator);
    var x = [_]f64{4};
    const wide: i64 = @as(i64, std.math.maxInt(i32)) + 1;
    try expectInfo(i64, 0, A.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = wide, .values = &x }));
    try std.testing.expectEqual(@as(f64, 2), x[0]);
    if (@bitSizeOf(usize) < 64) {
        const too_wide: i64 = @as(i64, std.math.maxInt(u32)) + 1;
        try expectFault(i64, error.IntegerNotRepresentable, A.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 1, .lda = too_wide, .values = &x }));
        try expectFault(i64, error.IntegerNotRepresentable, A.call(f64, i64, .work, &services, .{ .layout = 101, .uplo = 'L', .n = too_wide, .lda = too_wide, .values = null }));
    }
    try expectInfo(i64, 0, A.call(f64, i64, .work, &services, .{ .layout = 102, .uplo = 'L', .n = 0, .lda = std.math.maxInt(i64), .values = null }));
}
