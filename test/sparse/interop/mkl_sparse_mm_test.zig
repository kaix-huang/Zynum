// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

// Source-only candidate; no compilation or execution is claimed.
const std = @import("std");
const binding = @import("sparse-interface-owner");
const Core = @import("zynum-core");
const execution = @import("zynum-execution");
const resources = @import("zynum-numerics-resources");
const fixture = execution.test_support;
const Context = execution.context.Context;
const P = Core.internal.numerics.SparsePattern;
const Dense = Core.internal.numerics.Dense;
const Product = Core.internal.numerics.SparseProduct;
const Typed = binding.Adapter;
const Raw = binding.Raw;
const Handles = binding.Handles;
const helpers = @import("handle_test.zig");
const equal = std.testing.expectEqual;
const expect = std.testing.expect;
const expectError = std.testing.expectError;
const general: Raw.MatrixDescriptor = .{ .matrix_type = 20, .mode = 0xfedcba98, .diag = 0x87654321 };

fn isComplex(comptime T: type) bool {
    return T == Core.ComplexF32 or T == Core.ComplexF64;
}
fn number(comptime T: type, re: f64, im: f64) T {
    return if (comptime isComplex(T)) .{ .re = @floatCast(re), .im = @floatCast(im) } else @floatCast(re);
}
fn release(h: ?Handles.Handle) void {
    Typed.destroy(h) catch @panic("MM valid handle destruction failed");
}
fn opBits(op: P.Operation) u32 {
    return switch (op) {
        .normal => 10,
        .transposed => 11,
        .adjoint => 12,
    };
}
fn layoutBits(layout: Dense.Layout) u32 {
    return if (layout == .row_major) 101 else 102;
}
fn at(layout: Dense.Layout, ld: usize, row: usize, column: usize) usize {
    return if (layout == .row_major) row * ld + column else column * ld + row;
}
fn used(layout: Dense.Layout, rows: usize, columns: usize, ld: usize) usize {
    if (rows == 0 or columns == 0) return 0;
    return if (layout == .row_major) (rows - 1) * ld + columns else (columns - 1) * ld + rows;
}
const Z = struct { re: f64, im: f64 };
fn z(comptime T: type, value: T) Z {
    return if (comptime isComplex(T)) .{ .re = value.re, .im = value.im } else .{ .re = value, .im = 0 };
}
fn add(a: Z, b: Z) Z {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}
fn mul(a: Z, b: Z) Z {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}
fn near(comptime T: type, wanted: Z, actual: T) !void {
    const got = z(T, actual);
    const tolerance: f64 = if (T == f32 or T == Core.ComplexF32) 0.0002 else 0.00000000002;
    try std.testing.expectApproxEqAbs(wanted.re, got.re, tolerance);
    try std.testing.expectApproxEqAbs(wanted.im, got.im, tolerance);
}
const Entry = struct { row: usize, column: usize, re: f64, im: f64 };
fn Matrix(comptime T: type, comptime I: type) type {
    return struct {
        starts: [3]I = @splat(0),
        ends: [3]I = @splat(0),
        ordinals: [32]I = @splat(-99),
        values: [32]T = @splat(number(T, std.math.nan(f64), std.math.nan(f64))),
        rows: usize,
        columns: usize,
        axis: P.MajorAxis,
        base: u32,
        fn init(rows: usize, columns: usize, axis: P.MajorAxis, base: u32, entries: []const Entry) @This() {
            var self: @This() = .{ .rows = rows, .columns = columns, .axis = axis, .base = base };
            var cursor: usize = 0;
            for (0..(if (axis == .row) rows else columns)) |major| {
                cursor += 1; // real inaccessible-by-mathematics gap, with poisoned data
                self.starts[major] = @intCast(cursor + base);
                for (entries) |entry| {
                    if ((if (axis == .row) entry.row else entry.column) != major) continue;
                    self.ordinals[cursor] = @intCast((if (axis == .row) entry.column else entry.row) + base);
                    self.values[cursor] = number(T, entry.re, entry.im);
                    cursor += 1;
                }
                self.ends[major] = @intCast(cursor + base);
            }
            return self;
        }
        fn create(self: *@This(), context: *Context) !Handles.Handle {
            var h: ?Handles.Handle = null;
            const major = if (self.axis == .row) self.rows else self.columns;
            try Typed.createBorrowed(T, I, context, .{ .base = @intCast(self.base), .rows = @intCast(self.rows), .columns = @intCast(self.columns), .axis = self.axis, .starts = self.starts[0..major], .ends = self.ends[0..major], .ordinals = &self.ordinals, .values = &self.values }, &h);
            return h.?;
        }
    };
}

// Independent reference consumes an explicitly supplied row-major mathematical
// matrix. It never traverses a Pattern, expands CSR/CSC or calls Product/MV.
fn checkNumerical(comptime T: type, comptime I: type, h: Handles.Handle, rows: usize, columns: usize, dense: []const Z, semantics: Typed.MatrixSemantics, descriptor: Raw.MatrixDescriptor, operation: P.Operation, layout: Dense.Layout, rhs: usize) !void {
    const m = if (operation == .normal) rows else columns;
    const k = if (operation == .normal) columns else rows;
    const ldb = (if (layout == .row_major) rhs else k) + 2;
    const ldc = (if (layout == .row_major) rhs else m) + 3;
    var b: [128]T = @splat(number(T, std.math.nan(f64), std.math.nan(f64)));
    var before: [128]T = @splat(number(T, 71, -53));
    var logical: [128]bool = @splat(false);
    for (0..k) |row| for (0..rhs) |column| {
        b[1 + at(layout, ldb, row, column)] = number(T, @as(f64, @floatFromInt(row + column)) + 1, @as(f64, @floatFromInt(column)) - 1);
    };
    const b_before = b;
    for (0..m) |row| for (0..rhs) |column| {
        const index = 1 + at(layout, ldc, row, column);
        before[index] = number(T, @as(f64, @floatFromInt(row)) - 2, @as(f64, @floatFromInt(column)) + 1);
        logical[index] = true;
    };
    var typed = before;
    var raw = before;
    const alpha = number(T, 2, -1);
    const beta = number(T, -1, 0.5);
    try Typed.mm(T, I, h, operation, semantics, layout, alpha, b[1..][0..used(layout, k, rhs, ldb)], @intCast(rhs), @intCast(ldb), beta, typed[1..][0..used(layout, m, rhs, ldc)], @intCast(ldc));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&b_before), std.mem.asBytes(&b));
    try Raw.mm(T, I, h, opBits(operation), descriptor, layoutBits(layout), alpha, b[1..].ptr, @intCast(rhs), @intCast(ldb), beta, raw[1..].ptr, @intCast(ldc));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&b_before), std.mem.asBytes(&b));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&typed), std.mem.asBytes(&raw));
    for (0..m) |row| for (0..rhs) |column| {
        var sum: Z = .{ .re = 0, .im = 0 };
        for (0..k) |inner| {
            var a = dense[if (operation == .normal) row * columns + inner else inner * columns + row];
            if (!isComplex(T)) a.im = 0;
            if (operation == .adjoint) a.im = -a.im;
            sum = add(sum, mul(a, z(T, b_before[1 + at(layout, ldb, inner, column)])));
        }
        const index = 1 + at(layout, ldc, row, column);
        try near(T, add(mul(z(T, alpha), sum), mul(z(T, beta), z(T, before[index]))), typed[index]);
    };
    for (logical, 0..) |active, index| if (!active) try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before[index]), std.mem.asBytes(&typed[index]));
}

test "Sparse MM shares the real Core panel scalar Context and handle owner" {
    comptime {
        if (binding.Provider.Dense != Dense or binding.Provider.Product != Product or binding.Provider.OwnerIdentity != Core or
            binding.Provider.Execution != execution or binding.Provider.Resources != resources or
            Handles.Context != Context or Handles.Operation != execution.context.Operation or Typed.Handle != Raw.Handle)
            @compileError("MM changed the common implementation owner");
        if (Raw.ConstPointer(Core.ComplexF64) != ?[*]align(1) const Core.ComplexF64) @compileError("MM weakened readonly provenance");
    }
    try expect(binding.Provider.Execution.getContext() == execution.getContext());
}

test "Sparse MM rectangular four scalar two width products match independent dense arithmetic and full padding" {
    @setEvalBranchQuota(20000);
    fixture.setThreads(1);
    const entries = [_]Entry{ .{ .row = 0, .column = 2, .re = 1, .im = 1 }, .{ .row = 0, .column = 0, .re = 1, .im = 2 }, .{ .row = 0, .column = 2, .re = 2, .im = -2 }, .{ .row = 1, .column = 1, .re = 4, .im = 2 }, .{ .row = 1, .column = 0, .re = -2, .im = 1 } };
    const dense = [_]Z{ .{ .re = 1, .im = 2 }, .{ .re = 0, .im = 0 }, .{ .re = 3, .im = -1 }, .{ .re = -2, .im = 1 }, .{ .re = 4, .im = 2 }, .{ .re = 0, .im = 0 } };
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        var reference_context: FixtureReferenceContext = undefined;
        reference_context.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const context = reference_context.borrowedContext();
        for ([_]P.MajorAxis{ .row, .column }) |axis| for ([_]u32{ 0, 1 }) |base| {
            var matrix = Matrix(T, I).init(2, 3, axis, base, &entries);
            const h = try matrix.create(context);
            defer release(h);
            const original = matrix.values;
            for ([_]Dense.Layout{ .row_major, .column_major }) |layout| for ([_]P.Operation{ .normal, .transposed, .adjoint }) |operation| {
                const broad = (base == 0 and layout == .row_major) or (base == 1 and layout == .column_major);
                if (!broad and (axis != .row or operation != .normal)) continue;
                for ([_]usize{ 1, 2, 3, 8, 9 }) |rhs| try checkNumerical(T, I, h, 2, 3, &dense, .general, general, operation, layout, rhs);
            };
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&original), std.mem.asBytes(&matrix.values));
        };
        try helpers.close(context);
    };
}

test "Sparse MM typed raw share the exact four configuration table cells including empty RHS" {
    fixture.setThreads(1);
    const semantics = [_]Typed.MatrixSemantics{ .general, .{ .symmetric = .{ .triangle = .lower, .diagonal = .stored } }, .{ .hermitian = .{ .triangle = .upper, .diagonal = .unit } }, .{ .triangular = .{ .triangle = .lower, .diagonal = .stored } }, .{ .diagonal = .unit } };
    const descriptors = [_]Raw.MatrixDescriptor{ general, .{ .matrix_type = 21, .mode = 40, .diag = 50 }, .{ .matrix_type = 22, .mode = 41, .diag = 51 }, .{ .matrix_type = 23, .mode = 40, .diag = 50 }, .{ .matrix_type = 24, .mode = 99, .diag = 51 } };
    var reference_context: FixtureReferenceContext = undefined;
    reference_context.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const context = reference_context.borrowedContext();
    // Independently tabulated: base0 row broad, base0 column narrow,
    // base1 row narrow, base1 column broad.
    const broad_cells = [2][2]bool{ .{ true, false }, .{ false, true } };
    for ([_]u32{ 0, 1 }) |base| for ([_]P.MajorAxis{ .row, .column }) |axis| {
        var matrix = Matrix(f64, i32).init(2, 2, axis, base, &.{});
        const h = try matrix.create(context);
        defer release(h);
        for ([_]Dense.Layout{ .row_major, .column_major }, 0..) |layout, li| for ([_]P.Operation{ .normal, .transposed, .adjoint }) |operation| {
            for (semantics, descriptors, 0..) |semantic, descriptor, si| for ([_]i32{ 0, 3 }) |rhs| {
                var out = [_]f64{ 17, 19, 23, 29, 31, 37, 41, 43 };
                const before = out;
                const allowed = broad_cells[base][li] or (axis == .row and si == 0 and operation == .normal);
                if (allowed) {
                    try Typed.mm(f64, i32, h, operation, semantic, layout, 0, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, rhs, 3, 1, &out, 3);
                    try Raw.mm(f64, i32, h, opBits(operation), descriptor, layoutBits(layout), 0, &[_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 }, rhs, 3, 1, &out, 3);
                } else {
                    try expectError(error.UnsupportedConfiguration, Typed.mm(f64, i32, h, operation, semantic, layout, 0, &.{}, rhs, 3, 0, &out, 3));
                    try expectError(error.UnsupportedConfiguration, Raw.mm(f64, i32, h, opBits(operation), descriptor, layoutBits(layout), 0, null, rhs, 3, 0, &out, 3));
                }
                try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&out));
            };
        };
    };
    try helpers.close(context);
}

test "Sparse MM square structured panels match independent semantic matrices for both halves and transforms" {
    @setEvalBranchQuota(20000);
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        var reference_context: FixtureReferenceContext = undefined;
        reference_context.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const context = reference_context.borrowedContext();
        for ([_]P.MajorAxis{ .row, .column }) |axis| for ([_]u32{ 0, 1 }) |base| for ([_]bool{ false, true }) |upper| {
            const layout: Dense.Layout = if (base == 0) .row_major else .column_major;
            for ([_]u32{ 21, 22, 23, 24 }) |kind| for ([_]bool{ false, true }) |unit| {
                const diagonal: Product.Diagonal = if (unit) .unit else .stored;
                const half: Product.Half = .{ .triangle = if (upper) .upper else .lower, .diagonal = diagonal };
                const semantic: Typed.MatrixSemantics = switch (kind) {
                    21 => .{ .symmetric = half },
                    22 => .{ .hermitian = half },
                    23 => .{ .triangular = half },
                    24 => .{ .diagonal = diagonal },
                    else => unreachable,
                };
                const poison = std.math.nan(f64);
                const dr: f64 = if (unit) poison else 2;
                const di: f64 = if (unit or kind == 22) poison else 1;
                const entries = [_]Entry{ .{ .row = if (upper) 0 else 1, .column = if (upper) 1 else 0, .re = if (kind == 24) poison else 1, .im = if (kind == 24) poison else 2 }, .{ .row = 0, .column = 0, .re = dr, .im = di }, .{ .row = 1, .column = 1, .re = if (unit) poison else 3, .im = if (unit or kind == 22) poison else -1 }, .{ .row = if (upper) 1 else 0, .column = if (upper) 0 else 1, .re = poison, .im = poison } };
                var matrix = Matrix(T, I).init(2, 2, axis, base, &entries);
                const h = try matrix.create(context);
                defer release(h);
                var dense = [_]Z{ .{ .re = if (unit) 1 else 2, .im = if (unit or kind == 22) 0 else 1 }, .{ .re = 0, .im = 0 }, .{ .re = 0, .im = 0 }, .{ .re = if (unit) 1 else 3, .im = if (unit or kind == 22) 0 else -1 } };
                if (kind != 24) {
                    dense[if (upper) 1 else 2] = .{ .re = 1, .im = 2 };
                    if (kind == 21 or kind == 22) dense[if (upper) 2 else 1] = .{ .re = 1, .im = if (kind == 22) -2 else 2 };
                }
                const descriptor: Raw.MatrixDescriptor = .{ .matrix_type = kind, .mode = if (upper) 41 else 40, .diag = if (unit) 51 else 50 };
                for ([_]P.Operation{ .normal, .transposed, .adjoint }) |operation| try checkNumerical(T, I, h, 2, 2, &dense, semantic, descriptor, operation, layout, 3);
            };
        };
        try helpers.close(context);
    };
}

test "Sparse MM zero scales and empty unit storage preserve padding while metadata remains mandatory" {
    @setEvalBranchQuota(12000);
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        var reference_context: FixtureReferenceContext = undefined;
        reference_context.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const context = reference_context.borrowedContext();
        var starts = [_]I{ 0, 0 };
        var ends = [_]I{ 0, 0 };
        var h: ?Handles.Handle = null;
        try Typed.createBorrowed(T, I, context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .column, .starts = &starts, .ends = &ends, .ordinals = &.{}, .values = &.{} }, &h);
        {
            defer release(h);
            const x = [_]T{ number(T, 1, 2), number(T, 3, 4), number(T, 5, 6), number(T, 91, 92), number(T, 7, 8), number(T, 9, 10), number(T, 11, 12) };
            var out: [9]T = @splat(number(T, 71, 73));
            const sentinel = out[7..9].*;
            const unit: Typed.MatrixSemantics = .{ .diagonal = .unit };
            try Typed.mm(T, I, h, .adjoint, unit, .row_major, number(T, 1, 0), &x, 3, 4, number(T, 0, 0), &out, 4);
            for ([_]usize{ 0, 1, 2, 4, 5, 6 }) |i| try near(T, z(T, x[i]), out[i]);
            try near(T, z(T, number(T, 71, 73)), out[3]);
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&sentinel), std.mem.asBytes(out[7..9]));
            const nan = number(T, std.math.nan(f64), std.math.nan(f64));
            const poison = [_]T{ nan, nan, nan, nan, nan, nan, nan };
            @memset(out[0..7], nan);
            const old_gap = std.mem.asBytes(&out[3]).*;
            try Raw.mm(T, I, h, 10, general, 101, number(T, 0, -0.0), &poison, 3, 4, number(T, 0, 0), &out, 4);
            const zero = number(T, 0, 0);
            for ([_]usize{ 0, 1, 2, 4, 5, 6 }) |i| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&zero), std.mem.asBytes(&out[i]));
            try std.testing.expectEqualSlices(u8, &old_gap, std.mem.asBytes(&out[3]));
            const before = std.mem.asBytes(&out).*;
            try expectError(error.NullInput, Raw.mm(T, I, h, 10, general, 101, number(T, 0, 0), null, 3, 4, number(T, 0, 0), &out, 4));
            ends[0] = 1;
            try expectError(error.InvalidOffsets, Raw.mm(T, I, h, 10, general, 101, number(T, 0, 0), null, 3, 4, number(T, 0, 0), &out, 4));
            ends[0] = 0;
            try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&out));
        }
        // Nonempty numerical storage must also be unread at alpha0. Then
        // finite alpha with NaN old C distinguishes beta0's no-read contract.
        var poisoned = Matrix(T, I).init(2, 2, .row, 0, &.{.{ .row = 0, .column = 0, .re = std.math.nan(f64), .im = std.math.inf(f64) }});
        const ph = try poisoned.create(context);
        {
            defer release(ph);
            const nan = number(T, std.math.nan(f64), std.math.nan(f64));
            const poison_input: [6]T = @splat(nan);
            var out: [8]T = @splat(nan);
            try Raw.mm(T, I, ph, 10, general, 101, number(T, 0, 0), &poison_input, 3, 3, number(T, 0, 0), &out, 4);
            const zero = number(T, 0, 0);
            for ([_]usize{ 0, 1, 2, 4, 5, 6 }) |i| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&zero), std.mem.asBytes(&out[i]));
            poisoned.values[1] = number(T, 2, 1);
            const finite_input = [_]T{ number(T, 1, 2), number(T, 3, 4), number(T, 5, 6), number(T, 7, 8), number(T, 9, 10), number(T, 11, 12) };
            @memset(&out, nan);
            try Typed.mm(T, I, ph, .normal, .general, .row_major, number(T, 1, 0), &finite_input, 3, 3, number(T, 0, 0), &out, 4);
            for (0..3) |i| try near(T, mul(z(T, number(T, 2, 1)), z(T, finite_input[i])), out[i]);
            for (out[4..7]) |value| try near(T, .{ .re = 0, .im = 0 }, value);
            const before = std.mem.asBytes(&out).*;
            poisoned.ordinals[1] = 2;
            try expectError(error.InvalidOrdinal, Raw.mm(T, I, ph, 10, .{ .matrix_type = 24, .mode = 999, .diag = 51 }, 101, number(T, 0, 0), null, 3, 3, number(T, 0, 0), &out, 4));
            poisoned.ordinals[1] = 0;
            try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&out));
        }
        try helpers.close(context);
    };
}

test "Sparse MM duplicate cancellation retains stored order across scalar block and odd tail" {
    @setEvalBranchQuota(10000);
    fixture.setThreads(1);
    inline for (.{ f32, f64, Core.ComplexF32, Core.ComplexF64 }) |T| {
        var reference_context: FixtureReferenceContext = undefined;
        reference_context.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const context = reference_context.borrowedContext();
        for ([_]P.MajorAxis{ .row, .column }) |axis| {
            var starts = [_]i64{0};
            var ends = [_]i64{3};
            var ordinals = [_]i64{ 0, 0, 0 };
            const huge: f64 = if (T == f32 or T == Core.ComplexF32) 0x1p30 else 0x1p60;
            var values = [_]T{ number(T, huge, 0), number(T, -huge, 0), number(T, 1, 0) };
            var h: ?Handles.Handle = null;
            try Typed.createBorrowed(T, i64, context, .{ .base = 0, .rows = 1, .columns = 1, .axis = axis, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values }, &h);
            defer release(h);
            const input = [_]T{ number(T, 1, 0), number(T, 1, 0), number(T, 1, 0) };
            for ([_]P.Operation{ .normal, .transposed, .adjoint }) |op| {
                var out: [4]T = @splat(number(T, 19, 23));
                try Raw.mm(T, i64, h, opBits(op), general, 101, number(T, 1, 0), &input, 3, 3, number(T, 0, 0), &out, 3);
                for (out[0..3]) |value| try near(T, .{ .re = 1, .im = 0 }, value);
                try near(T, z(T, number(T, 19, 23)), out[3]);
            }
        }
        try helpers.close(context);
    }
}

test "Sparse MM typed capacities signed parameters descriptors and metadata failures preserve complete output" {
    fixture.setThreads(1);
    var reference_context: FixtureReferenceContext = undefined;
    reference_context.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const context = reference_context.borrowedContext();
    var matrix = Matrix(f64, i32).init(2, 3, .row, 0, &.{ .{ .row = 0, .column = 0, .re = 2, .im = 0 }, .{ .row = 1, .column = 2, .re = 3, .im = 0 } });
    const h = try matrix.create(context);
    {
        defer release(h);
        var out = [_]f64{ 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61 };
        const before = std.mem.asBytes(&out).*;
        const input = [_]f64{ 1, 2, 3, 99, 4, 5, 6, 98, 7, 8, 9 };
        try expectError(error.BufferTooSmall, Typed.mm(f64, i32, h, .normal, .general, .row_major, 0, input[0..10], 3, 4, 0, out[0..1], 4));
        try expectError(error.BufferTooSmall, Typed.mm(f64, i32, h, .normal, .general, .row_major, 0, &input, 3, 4, 0, out[0..6], 4));
        try expectError(error.InvalidLeadingDimension, Raw.mm(f64, i32, h, 10, general, 101, 0, null, 3, 2, 0, null, 4));
        try expectError(error.InvalidLeadingDimension, Raw.mm(f64, i32, h, 10, general, 101, 0, null, 3, 4, 0, null, 2));
        try expectError(error.InvalidDimension, Raw.mm(f64, i32, h, 10, general, 101, 0, null, -1, -1, 0, null, -1));
        try expectError(error.InvalidLeadingDimension, Raw.mm(f64, i32, h, 10, general, 101, 0, null, 0, -1, 0, null, -1));
        try expectError(error.InvalidLeadingDimension, Raw.mm(f64, i32, h, 10, general, 101, 0, null, 0, 0, 0, null, -1));
        try expectError(error.NullHandle, Raw.mm(f64, i32, null, 99, general, 999, 0, null, -1, -1, 0, &out, -1));
        try expectError(error.InvalidOperation, Raw.mm(f64, i32, h, 99, general, 999, 0, null, -1, -1, 0, &out, -1));
        try expectError(error.InvalidDescriptor, Raw.mm(f64, i32, h, 10, .{ .matrix_type = 99, .mode = 99, .diag = 99 }, 999, 0, null, -1, -1, 0, &out, -1));
        try expectError(error.InvalidLayout, Raw.mm(f64, i32, h, 10, general, 999, 0, null, -1, -1, 0, &out, -1));
        try expectError(error.UnsupportedDescriptor, Raw.mm(f64, i32, h, 10, .{ .matrix_type = 21, .mode = 42, .diag = 99 }, 101, 0, null, 0, 0, 0, &out, 0));
        try expectError(error.UnsupportedDescriptor, Raw.mm(f64, i32, h, 10, .{ .matrix_type = 25, .mode = 99, .diag = 99 }, 101, 0, null, 0, 0, 0, &out, 0));
        try expectError(error.InvalidDescriptor, Raw.mm(f64, i32, h, 10, .{ .matrix_type = 23, .mode = 99, .diag = 50 }, 101, 0, null, 0, 0, 0, &out, 0));
        try expectError(error.InvalidDescriptor, Raw.mm(f64, i32, h, 10, .{ .matrix_type = 24, .mode = 99, .diag = 99 }, 101, 0, null, 0, 0, 0, &out, 0));
        try expectError(error.UnsupportedDomain, Raw.mm(f64, i32, h, 10, .{ .matrix_type = 21, .mode = 40, .diag = 51 }, 102, 0, null, 0, 0, 0, &out, 0));
        try expectError(error.WrongScalar, Raw.mm(f32, i32, h, 10, general, 101, 0, null, 0, 0, 0, @ptrCast(&out), 0));
        try expectError(error.WrongIndexWidth, Raw.mm(f64, i64, h, 10, general, 101, 0, null, 0, 0, 0, &out, 0));
        const saved = matrix.ordinals[1];
        matrix.ordinals[1] = 3;
        try expectError(error.InvalidDimension, Raw.mm(f64, i32, h, 10, general, 101, 0, null, -1, 0, 0, &out, 0));
        try expectError(error.InvalidOrdinal, Raw.mm(f64, i32, h, 11, general, 102, 0, null, 0, 0, 0, &out, 0));
        matrix.ordinals[1] = saved;
        try expectError(error.UnsupportedConfiguration, Raw.mm(f64, i32, h, 11, general, 102, 0, null, 0, 0, 0, &out, 0));
        try expectError(error.NullInput, Raw.mm(f64, i32, h, 10, general, 101, 0, null, 3, 4, 0, null, 4));
        try expectError(error.NullOutput, Raw.mm(f64, i32, h, 10, general, 101, 0, &input, 3, 4, 0, null, 4));
        try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&out));
        try Raw.mm(f64, i32, h, 10, general, 101, 1, &input, 3, 4, 0, &out, 4);
        try std.testing.expectEqualSlices(f64, &.{ 2, 4, 6, 29, 21, 24, 27, 43, 47, 53, 59, 61 }, &out);
    }
    try helpers.close(context);
}

test "Sparse MM empty prefixes zero K and zero M retain geometry and pointer arrival rules" {
    fixture.setThreads(1);
    var reference_context: FixtureReferenceContext = undefined;
    reference_context.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const context = reference_context.borrowedContext();
    var bytes: [32]u8 align(16) = @splat(0xab);
    const unaligned: Raw.Pointer(f64) = @ptrCast(bytes[1..].ptr);
    for ([_]P.MajorAxis{ .row, .column }) |axis| {
        var matrix = Matrix(f64, i64).init(2, 0, axis, 0, &.{});
        const h = try matrix.create(context);
        defer release(h);
        var out = [_]f64{ 1, 2, 3, 71, 4, 5, 6, 73 };
        try Raw.mm(f64, i64, h, 10, general, 101, std.math.inf(f64), unaligned, 3, std.math.maxInt(i64), -2, &out, 4);
        try std.testing.expectEqualSlices(f64, &.{ -2, -4, -6, 71, -8, -10, -12, 73 }, &out);
        try Raw.mm(f64, i64, h, 10, general, 101, 1, null, 0, 0, 1, unaligned, 0);
        try Raw.mm(f64, i64, h, 11, general, 101, 1, &[_]f64{ 1, 2, 3, 4, 5, 6 }, 3, 3, 0, unaligned, 0);
        try expectError(error.NullInput, Raw.mm(f64, i64, h, 11, general, 101, 0, null, 3, 3, 0, unaligned, 0));
        try expectError(error.InvalidAlignment, Raw.mm(f64, i64, h, 10, general, 101, 0, null, 3, 0, 0, unaligned, 3));
    }
    try helpers.close(context);
}

test "Sparse MM wide stride and checked byte overflow precede raw pointee access" {
    fixture.setThreads(1);
    var reference_context: FixtureReferenceContext = undefined;
    reference_context.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const context = reference_context.borrowedContext();
    var matrix = Matrix(f64, i64).init(1, 1, .row, 0, &.{.{ .row = 0, .column = 0, .re = 2, .im = 0 }});
    const h = try matrix.create(context);
    {
        defer release(h);
        var out = [_]f64{ 17, 19 };
        const input: f64 = 3;
        const wide: i64 = @as(i64, std.math.maxInt(i32)) + 1;
        if (@sizeOf(usize) > 4) {
            try Raw.mm(f64, i64, h, 10, general, 101, 1, @ptrCast(&input), 1, wide, 0, &out, wide);
            try std.testing.expectEqualSlices(f64, &.{ 6, 19 }, &out);
            // A truthful one-element pointer is enough: size arithmetic must
            // fail before forming or reading the enormous required prefix.
            try expectError(error.SizeOverflow, Raw.mm(f64, i64, h, 10, general, 102, 0, @ptrCast(&input), 2, std.math.maxInt(i64), 0, &out, 1));
        } else {
            try expectError(error.SizeOverflow, Raw.mm(f64, i64, h, 10, general, 101, 0, @ptrCast(&input), 1, @as(i64, std.math.maxInt(u32)) + 1, 0, &out, 1));
            try std.testing.expectEqualSlices(f64, &.{ 17, 19 }, &out);
        }
        const saved = out;
        const misaligned: [16]u8 align(16) = @splat(0xcd);
        try expectError(error.InvalidAlignment, Raw.mm(f64, i64, h, 10, general, 101, 0, @ptrCast(misaligned[1..].ptr), 1, 1, 0, &out, 1));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&saved), std.mem.asBytes(&out));
    }
    // An actual pointer with an arithmetically valid byte length whose end
    // would wrap must fail before a huge slice is formed; no fictitious pointer.
    var wide_handle: ?Handles.Handle = null;
    const huge_k: i64 = @intCast(std.math.maxInt(usize) / @sizeOf(f64));
    try Typed.createBorrowed(f64, i64, context, .{ .base = 0, .rows = 0, .columns = huge_k, .axis = .row, .starts = &.{}, .ends = &.{}, .ordinals = &.{}, .values = &.{} }, &wide_handle);
    {
        defer release(wide_handle);
        const live: f64 = 3;
        var canary = [_]f64{ 17, 19 };
        try expectError(error.SizeOverflow, Raw.mm(f64, i64, wide_handle, 10, general, 101, 0, @ptrCast(&live), 1, 1, 0, &canary, 0));
        try std.testing.expectEqualSlices(f64, &.{ 17, 19 }, &canary);
    }
    try helpers.close(context);
}

test "Sparse MM rejects partial dense padding sparse metadata and owner aliases before any write" {
    fixture.setThreads(1);
    var reference_context: FixtureReferenceContext = undefined;
    reference_context.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const context = reference_context.borrowedContext();
    var starts: [8]i64 align(@alignOf(f64)) = .{ 0, 1, 13, 17, 19, 23, 29, 31 };
    var ends: [8]i64 align(@alignOf(f64)) = .{ 1, 2, 13, 17, 19, 23, 29, 31 };
    var ordinals: [8]i64 align(@alignOf(f64)) = .{ 0, 1, 13, 17, 19, 23, 29, 31 };
    var values = [_]f64{ 2, 3, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 };
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i64, context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .row, .starts = starts[0..2], .ends = ends[0..2], .ordinals = ordinals[0..2], .values = &values }, &h);
    {
        defer release(h);
        var shared = [_]f64{ 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 };
        const before = std.mem.asBytes(&shared).*;
        try expectError(error.BufferOverlap, Typed.mm(f64, i64, h, .normal, .general, .row_major, 0, shared[0..6], 3, 3, 0, shared[1..8], 4));
        try expectError(error.BufferOverlap, Raw.mm(f64, i64, h, 10, general, 101, 0, shared[1..].ptr, 3, 3, 0, &shared, 4));
        // B touches only C's row padding: C logical offsets0,3, B offsets1,2.
        try expectError(error.BufferOverlap, Typed.mm(f64, i64, h, .normal, .general, .row_major, 0, shared[1..3], 1, 1, 0, shared[0..4], 3));
        try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&shared));
        const input = [_]f64{ 1, 2, 3, 4, 5, 6 };
        const starts_before = starts;
        const ends_before = ends;
        const ordinals_before = ordinals;
        const values_before = values;
        for ([_]Raw.Pointer(f64){ @ptrCast(&starts), @ptrCast(&ends), @ptrCast(&ordinals), &values, @ptrCast(h.?) }) |destination| {
            try expectError(error.BufferOverlap, Raw.mm(f64, i64, h, 10, .{ .matrix_type = 24, .mode = 99, .diag = 51 }, 101, 0, &input, 3, 3, 0, destination, 4));
        }
        const control = try Handles.testing.retainControl(f64, i64, h.?);
        try expectError(error.BufferOverlap, Raw.mm(f64, i64, h, 10, general, 101, 0, &input, 3, 3, 0, @ptrCast(control), 4));
        try equal(@as(usize, 2), control.references);
        control.release();
        try std.testing.expectEqualSlices(i64, &starts_before, &starts);
        try std.testing.expectEqualSlices(i64, &ends_before, &ends);
        try std.testing.expectEqualSlices(i64, &ordinals_before, &ordinals);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&values_before), std.mem.asBytes(&values));
        // Exact adjacency and unused array tails are not false positives.
        try Typed.mm(f64, i64, h, .normal, .general, .row_major, 1, shared[0..6], 3, 3, 0, shared[6..12], 3);
        try std.testing.expectEqualSlices(f64, &.{ 22, 26, 34, 57, 69, 87 }, shared[6..12]);
        try Typed.mm(f64, i64, h, .normal, .general, .row_major, 1, &input, 3, 3, 0, values[2..8], 3);
        try std.testing.expectEqualSlices(f64, &.{ 2, 3, 2, 4, 6, 12, 15, 18, 41, 43, 47, 53 }, &values);
        // The dense input may share sparse read-only values if C is disjoint.
        var result = [_]f64{ 59, 61 };
        try Typed.mm(f64, i64, h, .normal, .general, .row_major, 1, values[0..2], 1, 1, 0, &result, 1);
        try std.testing.expectEqualSlices(f64, &.{ 4, 9 }, &result);
    }
    var gapped = Matrix(f64, i64).init(1, 1, .row, 0, &.{.{ .row = 0, .column = 0, .re = 2, .im = 0 }});
    const gap_handle = try gapped.create(context);
    {
        defer release(gap_handle);
        const before = std.mem.asBytes(&gapped.values).*;
        try expectError(error.BufferOverlap, Raw.mm(f64, i64, gap_handle, 10, general, 101, 0, &[_]f64{3}, 1, 1, 0, &gapped.values, 1));
        try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&gapped.values));
    }
    try helpers.close(context);
}

test "Sparse MM source admission covers every odd RHS store through closing without allocations" {
    const Observed = struct {
        reference_context: FixtureReferenceContext = undefined,
        context: *Context = undefined,
        enabled: bool = false,
        admitted: usize = 0,
        finishing: usize = 0,
        output: [23]f64 = @splat(71),
        fn observe(context: *Context, event: execution.context.TestEvent) void {
            const self: *@This() = @fieldParentPtr("reference_context", @as(*FixtureReferenceContext, @fieldParentPtr("context", context)));
            if (!self.enabled) return;
            const state = context.snapshot();
            if (state.active_operations != 1 or state.resource_pins != 1) @panic("MM lost original operation/control pin");
            switch (event) {
                .admitted => {
                    self.admitted += 1;
                    for (self.output) |value| if (value != 71) @panic("MM output changed before admission");
                    _ = context.requestClose();
                },
                .finishing => {
                    self.finishing += 1;
                    for (self.output, 0..) |value, i| {
                        const wanted: f64 = if (i >= 1 and i <= 9) 6 else if (i >= 12 and i <= 20) 20 else 71;
                        if (value != wanted) @panic("MM finished before every logical column/tail or changed padding");
                    }
                },
            }
            if (fixtureProgressClose(context) != .busy) @panic("MM operation did not cover closing");
        }
    };
    fixture.setThreads(1);
    for ([_]bool{ false, true }) |raw| {
        var spy: helpers.Spy = .{};
        var observation: Observed = .{};
        observation.reference_context.initInPlace(spy.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        observation.context = observation.reference_context.borrowedContext();
        spy.ctx = observation.context;
        Context.testing.configure(observation.context, 1, Observed.observe);
        var starts = [_]i32{ 0, 1 };
        var ends = [_]i32{ 1, 2 };
        var ordinals = [_]i32{ 0, 1 };
        var values = [_]f64{ 2, 4 };
        var h: ?Handles.Handle = null;
        try Typed.createBorrowed(f64, i32, observation.context, .{ .base = 0, .rows = 2, .columns = 2, .axis = .column, .starts = &starts, .ends = &ends, .ordinals = &ordinals, .values = &values }, &h);
        try expect(try Handles.borrowedContext(h.?) != execution.getContext());
        const attempts = spy.attempts;
        const accepted = spy.accepted;
        const freed = spy.freed;
        var input: [18]f64 = undefined;
        @memset(input[0..9], 3);
        @memset(input[9..18], 5);
        observation.enabled = true;
        if (raw) {
            try Raw.mm(f64, i32, h, 10, general, 101, 1, &input, 9, 9, 0, observation.output[1..].ptr, 11);
        } else {
            try Typed.mm(f64, i32, h, .normal, .general, .row_major, 1, &input, 9, 9, 0, observation.output[1..], 11);
        }
        try equal(@as(usize, 1), observation.admitted);
        try equal(@as(usize, 1), observation.finishing);
        try equal(attempts, spy.attempts);
        try equal(accepted, spy.accepted);
        try equal(freed, spy.freed);
        try equal(@as(usize, 0), observation.context.snapshot().active_operations);
        try equal(execution.context.CloseResult.waiting_resources, fixtureProgressClose(observation.context));
        const saved = observation.output;
        try expectError(error.Closing, Raw.mm(f64, i32, h, 99, general, 999, 0, null, -1, -1, 0, &observation.output, -1));
        try expectError(error.Closing, Typed.mm(f64, i32, h, .normal, .general, .row_major, 0, &.{}, -1, -1, 0, &observation.output, -1));
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&saved), std.mem.asBytes(&observation.output));
        try Typed.destroy(h);
        try spy.balanced();
        try helpers.close(observation.context);
    }
}

test "Sparse MM admission overflow wrong Context seams and retained control preserve one resource lifecycle" {
    fixture.setThreads(1);
    var spy: helpers.Spy = .{};
    var reference_context: FixtureReferenceContext = undefined;
    reference_context.initInPlace(spy.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const context = reference_context.borrowedContext();
    spy.ctx = context;
    Context.testing.configure(context, 1, null);
    var matrix = Matrix(f64, i32).init(1, 1, .row, 0, &.{.{ .row = 0, .column = 0, .re = 2, .im = 0 }});
    const h = try matrix.create(context);
    const control = try Handles.testing.retainControl(f64, i32, h);
    try equal(@as(usize, 1), context.snapshot().resource_pins);
    var operation = try context.begin();
    const attempts = spy.attempts;
    var out = [_]f64{ 17, 19, 23 };
    const before = std.mem.asBytes(&out).*;
    try expectError(error.CounterOverflow, Raw.mm(f64, i32, h, 99, general, 999, 0, null, -1, -1, 0, &out, -1));
    try expectError(error.CounterOverflow, Typed.mm(f64, i32, h, .normal, .general, .row_major, 0, &.{}, -1, -1, 0, &out, -1));
    const prepared = try Typed.prepareMmAdmitted(f64, i32, &operation, h, .normal, .general, .row_major, 3, 3, 3);
    var reference_other_context: FixtureReferenceContext = undefined;
    reference_other_context.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const other_context = reference_other_context.borrowedContext();
    var other = try other_context.begin();
    try expectError(error.ContextMismatch, Typed.prepareMmAdmitted(f64, i32, &other, h, .normal, .general, .row_major, -1, -1, -1));
    try expectError(error.ContextMismatch, Typed.executeMmAdmitted(f64, i32, &other, h, prepared, 0, &.{}, 0, &out));
    other.end();
    try helpers.close(other_context);
    try std.testing.expectEqualSlices(u8, &before, std.mem.asBytes(&out));
    try Typed.executeMmAdmitted(f64, i32, &operation, h, prepared, 1, &.{ 1, 2, 3 }, 0, &out);
    try std.testing.expectEqualSlices(f64, &.{ 2, 4, 6 }, &out);
    operation.end();
    try equal(attempts, spy.attempts);
    _ = context.requestClose();
    try Typed.destroy(h);
    try equal(@as(usize, 1), control.references);
    try equal(@as(usize, 1), context.snapshot().resource_pins);
    try equal(execution.context.CloseResult.waiting_resources, fixtureProgressClose(context));
    control.release();
    try spy.balanced();
    try helpers.close(context);
}

test "checked shared handle solve CSR CSC bases both layouts complex alpha and padding" {
    inline for (.{ f32, f64, Core.sparse.ComplexF32, Core.sparse.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        const complex = comptime isComplex(T);
        for ([_]P.MajorAxis{ .row, .column }) |axis| for ([_]I{ 0, 1 }) |base| for ([_]Dense.Layout{ .row_major, .column_major }) |layout| for ([_]P.Operation{ .normal, .transposed, .adjoint }) |op| {
            var reference_ctx: FixtureReferenceContext = undefined;
            reference_ctx.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const ctx = reference_ctx.borrowedContext();
            var starts = [_]I{ base, 3 + base };
            var ends = [_]I{ 1 + base, 4 + base };
            var ord = [_]I{ base, -99, -99, 1 + base };
            var values = [_]T{ number(T, 2, 1), number(T, 777, 7), number(T, 777, 7), number(T, 3, -1) };
            var h: ?Handles.Handle = null;
            try Typed.createBorrowed(T, I, ctx, .{ .base = @intCast(base), .rows = 2, .columns = 2, .axis = axis, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
            var x: [6]T = @splat(number(T, 77, 9));
            var y = x;
            const alpha = number(T, 0.5, 0.25);
            for (0..2) |j| for (0..2) |i| {
                var diag = values[if (i == 0) 0 else 3];
                if (comptime complex) {
                    if (op == .adjoint) diag.im = -diag.im;
                }
                x[at(layout, 3, i, j)] = binding.Provider.Scalar.multiply(T, diag, number(T, @as(f64, @floatFromInt(i + j + 1)), 0.3));
            };
            const result = try binding.Solve.panel(T, I, h, op, .lower, .non_unit, layout, alpha, &x, 2, 3, &y, 3);
            try std.testing.expectEqual(@as(usize, 0), result.info);
            try std.testing.expectEqual(@as(usize, 2), result.completed_rhs);
            for (0..2) |j| for (0..2) |i| {
                const want = binding.Provider.Scalar.multiply(T, alpha, number(T, @as(f64, @floatFromInt(i + j + 1)), 0.3));
                const got = y[at(layout, 3, i, j)];
                if (comptime complex) {
                    try std.testing.expectApproxEqAbs(want.re, got.re, if (T == Core.sparse.ComplexF32) @as(@TypeOf(want.re), 3e-5) else 3e-12);
                    try std.testing.expectApproxEqAbs(want.im, got.im, if (T == Core.sparse.ComplexF32) @as(@TypeOf(want.im), 3e-5) else 3e-12);
                } else try std.testing.expectApproxEqAbs(want, got, if (T == f32) @as(T, 3e-5) else 3e-12);
            };
            try std.testing.expectEqual(number(T, 77, 9), y[2]);
            try std.testing.expectEqual(number(T, 77, 9), y[5]);
            try std.testing.expectEqual(@as(usize, 0), ctx.snapshot().active_operations);
            try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().resource_pins);
            try Typed.destroy(h);
            _ = ctx.requestClose();
            try std.testing.expectEqual(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
        };
    };
}

test "checked shared handle admitted solve finishes after close and permits staged in-place panel" {
    const T = Core.sparse.ComplexF64;
    const I = i32;
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    var starts = [_]I{0};
    var ends = [_]I{1};
    var ord = [_]I{0};
    var values = [_]T{number(T, 2, 1)};
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(T, I, ctx, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
    var operation = try ctx.begin();
    _ = ctx.requestClose();
    const prepared = try binding.Solve.prepareAdmitted(T, I, &operation, h.?, .normal, .lower, .non_unit, .column_major, 1, 1, 1);
    var xy = [_]T{number(T, 3, 4)};
    const result = try binding.Solve.executeAdmitted(T, I, &operation, h.?, prepared, number(T, 1, 0), &xy, &xy);
    try std.testing.expectEqual(@as(usize, 0), result.info);
    try std.testing.expectApproxEqAbs(@as(f64, 2), xy[0].re, 3e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), xy[0].im, 3e-12);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().active_operations);
    try std.testing.expectEqual(@as(usize, 1), ctx.snapshot().resource_pins);
    operation.end();
    try Typed.destroy(h);
    try std.testing.expectEqual(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
}

test "checked handle solve normalizes off diagonal CSC CSR and commits only numerical success" {
    const T = Core.sparse.ComplexF64;
    const I = i64;
    const M = binding.Provider.Scalar;
    for ([_]P.MajorAxis{ .row, .column }) |axis| for ([_]P.Operation{ .normal, .transposed, .adjoint }) |op| {
        var reference_ctx: FixtureReferenceContext = undefined;
        reference_ctx.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        var starts = [_]I{ 0, if (axis == .row) 2 else 3 };
        var ends = [_]I{ if (axis == .row) 1 else 2, 4 };
        var ord = [_]I{ 0, if (axis == .row) -99 else 1, if (axis == .row) 0 else -99, 1 };
        const d0 = number(T, 2, 1);
        const d1 = number(T, 3, -1);
        const c = number(T, 1, 0.5);
        var values = [_]T{ d0, if (axis == .row) number(T, 777, 9) else c, if (axis == .row) c else number(T, 777, 9), d1 };
        var h: ?Handles.Handle = null;
        try Typed.createBorrowed(T, I, ctx, .{ .base = 0, .rows = 2, .columns = 2, .axis = axis, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
        const want = [_]T{ number(T, 1, 2), number(T, 3, -0.5) };
        const coeff0 = if (op == .adjoint) M.maybeConjugate(T, d0, true) else d0;
        const coeff1 = if (op == .adjoint) M.maybeConjugate(T, d1, true) else d1;
        const coupling = if (op == .adjoint) M.maybeConjugate(T, c, true) else c;
        const x = [_]T{ M.add(T, M.multiply(T, coeff0, want[0]), if (op == .normal) number(T, 0, 0) else M.multiply(T, coupling, want[1])), M.add(T, M.multiply(T, coeff1, want[1]), if (op == .normal) M.multiply(T, coupling, want[0]) else number(T, 0, 0)) };
        var y = [_]T{ number(T, 99, 8), number(T, 99, 8) };
        const ok = try binding.Solve.panel(T, I, h, op, .lower, .non_unit, .column_major, number(T, 1, 0), &x, 1, 2, &y, 2);
        try equal(@as(usize, 0), ok.info);
        for (y, want) |got, target| {
            try std.testing.expectApproxEqAbs(target.re, got.re, 3e-12);
            try std.testing.expectApproxEqAbs(target.im, got.im, 3e-12);
        }
        values[3] = number(T, 0, 0);
        const before = y;
        const bad = try binding.Solve.panel(T, I, h, op, .lower, .non_unit, .column_major, number(T, 1, 0), &x, 1, 2, &y, 2);
        try equal(@as(usize, 2), bad.info);
        try std.testing.expectEqualSlices(T, &before, &y);
        try Typed.destroy(h);
        _ = ctx.requestClose();
        try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    };
}

test "checked handle solve all allocation failure points leave output and release earlier scratch" {
    for (0..5) |allocation| {
        var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
        var reference_ctx: FixtureReferenceContext = undefined;
        reference_ctx.initInPlace(allocator.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        var starts = [_]i32{0};
        var ends = [_]i32{1};
        var ord = [_]i32{0};
        var values = [_]f64{2};
        var h: ?Handles.Handle = null;
        try Typed.createBorrowed(f64, i32, ctx, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
        allocator.fail_index = allocator.alloc_index + allocation;
        const x = [_]f64{4};
        var y = [_]f64{99};
        try expectError(error.OutOfMemory, binding.Solve.panel(f64, i32, h, .normal, .lower, .non_unit, .column_major, 1, &x, 1, 1, &y, 1));
        try equal(@as(f64, 99), y[0]);
        try equal(@as(usize, 0), ctx.snapshot().active_operations);
        try equal(@as(usize, 1), ctx.snapshot().resource_pins);
        try Typed.destroy(h);
        try equal(allocator.allocated_bytes, allocator.freed_bytes);
        _ = ctx.requestClose();
        try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    }
}

test "checked raw TRSV TRSM four scalars both formats bases N T H and documented layouts" {
    @setEvalBranchQuota(20000);
    inline for (.{ f32, f64, Core.sparse.ComplexF32, Core.sparse.ComplexF64 }) |T| inline for (.{ i32, i64 }) |I| {
        const complex = comptime isComplex(T);
        for ([_]P.MajorAxis{ .row, .column }) |axis| for ([_]I{ 0, 1 }) |base| for ([_]P.Operation{ .normal, .transposed, .adjoint }) |op| {
            var reference_ctx: FixtureReferenceContext = undefined;
            reference_ctx.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
            const ctx = reference_ctx.borrowedContext();
            var starts = [_]I{ base, 1 + base };
            var ends = [_]I{ 1 + base, 2 + base };
            var ord = [_]I{ base, 1 + base };
            var values = [_]T{ number(T, 2, 1), number(T, 3, -1) };
            var h: ?Handles.Handle = null;
            try Typed.createBorrowed(T, I, ctx, .{ .base = @intCast(base), .rows = 2, .columns = 2, .axis = axis, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
            const descriptor: binding.RawSolve.Descriptor = .{ .matrix_type = 23, .mode = 40, .diag = 50 };
            var input: [2]T = undefined;
            var output: [2]T = undefined;
            for (0..2) |i| {
                var coefficient = values[i];
                if (comptime complex) {
                    if (op == .adjoint) coefficient.im = -coefficient.im;
                }
                input[i] = binding.Provider.Scalar.multiply(T, coefficient, number(T, @as(f64, @floatFromInt(i + 1)), 0.3));
            }
            const result = try binding.RawSolve.vector(T, I, h, opBits(op), descriptor, number(T, 1, 0), @ptrCast(&input), @ptrCast(&output));
            try equal(@as(usize, 0), result.info);
            const layout: Dense.Layout = if (base == 0) .row_major else .column_major;
            var x: [6]T = @splat(number(T, 77, 9));
            var y = x;
            for (0..2) |j| for (0..2) |i| {
                x[at(layout, 3, i, j)] = input[i];
            };
            const alpha = number(T, 0.5, 0.25);
            const panel_result = try binding.RawSolve.panel(T, I, h, opBits(op), descriptor, layoutBits(layout), alpha, @ptrCast(&x), 2, 3, @ptrCast(&y), 3);
            try equal(@as(usize, 0), panel_result.info);
            for (0..2) |i| {
                const want = number(T, @as(f64, @floatFromInt(i + 1)), 0.3);
                const tolerance = if (T == f32 or T == Core.sparse.ComplexF32) 3e-5 else 3e-12;
                if (comptime complex) {
                    try std.testing.expectApproxEqAbs(want.re, output[i].re, tolerance);
                    try std.testing.expectApproxEqAbs(want.im, output[i].im, tolerance);
                } else try std.testing.expectApproxEqAbs(want, output[i], tolerance);
                const scaled = binding.Provider.Scalar.multiply(T, alpha, want);
                for (0..2) |j| {
                    const got = y[at(layout, 3, i, j)];
                    if (comptime complex) {
                        try std.testing.expectApproxEqAbs(scaled.re, got.re, tolerance);
                        try std.testing.expectApproxEqAbs(scaled.im, got.im, tolerance);
                    } else try std.testing.expectApproxEqAbs(scaled, got, tolerance);
                }
            }
            try equal(number(T, 77, 9), y[2]);
            try equal(number(T, 77, 9), y[5]);
            try Typed.destroy(h);
            _ = ctx.requestClose();
            try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
        };
    };
}

test "checked raw diagonal descriptor ignores mode and off diagonal NaN without treating as triangular" {
    const T = Core.sparse.ComplexF64;
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    var starts = [_]i32{ 0, 2 };
    var ends = [_]i32{ 2, 4 };
    var ord = [_]i32{ 0, 1, 0, 1 };
    var values = [_]T{ number(T, 2, 1), number(T, std.math.nan(f64), std.math.inf(f64)), number(T, std.math.nan(f64), std.math.inf(f64)), number(T, 3, -1) };
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(T, i32, ctx, .{ .base = 0, .rows = 2, .columns = 2, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
    const input = [_]T{ number(T, 3, 4), number(T, 3, -1) };
    var output: [2]T = undefined;
    const descriptor: binding.RawSolve.Descriptor = .{ .matrix_type = 24, .mode = 0xfedcba98, .diag = 50 };
    const result = try binding.RawSolve.vector(T, i32, h, 10, descriptor, number(T, 1, 0), @ptrCast(&input), @ptrCast(&output));
    try equal(@as(usize, 0), result.info);
    try std.testing.expectApproxEqAbs(@as(f64, 2), output[0].re, 3e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), output[0].im, 3e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), output[1].re, 3e-12);
    try Typed.destroy(h);
    _ = ctx.requestClose();
    try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
}

test "checked raw solve enum geometry alignment aliases and closing precedence are explicit" {
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    var starts = [_]i32{0};
    var ends = [_]i32{1};
    var ord = [_]i32{0};
    var values = [_]f64{2};
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i32, ctx, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
    const d: binding.RawSolve.Descriptor = .{ .matrix_type = 23, .mode = 40, .diag = 50 };
    try expectError(error.NullHandle, binding.RawSolve.vector(f64, i32, null, 99, d, 1, null, null));
    try expectError(error.InvalidOperation, binding.RawSolve.vector(f64, i32, h, 99, d, 1, null, null));
    try expectError(error.UnreviewedSolveDescriptor, binding.RawSolve.vector(f64, i32, h, 10, .{ .matrix_type = 20, .mode = 40, .diag = 50 }, 1, null, null));
    try expectError(error.InvalidLayout, binding.RawSolve.panel(f64, i32, h, 10, d, 999, 1, null, 1, 1, null, 1));
    try expectError(error.UnsupportedConfiguration, binding.RawSolve.panel(f64, i32, h, 10, d, 102, 1, null, 1, 1, null, 1));
    try expectError(error.InvalidLeadingDimension, binding.RawSolve.panel(f64, i32, h, 10, d, 101, 1, null, 1, 0, null, 1));
    try expectError(error.NullInput, binding.RawSolve.vector(f64, i32, h, 10, d, 1, null, null));
    const x = [_]f64{4};
    try expectError(error.NullOutput, binding.RawSolve.vector(f64, i32, h, 10, d, 1, @ptrCast(&x), null));
    var bytes: [32]u8 align(16) = @splat(0);
    const unaligned: binding.Raw.ConstPointer(f64) = @ptrCast(bytes[1..].ptr);
    var y = [_]f64{99};
    try expectError(error.InvalidAlignment, binding.RawSolve.vector(f64, i32, h, 10, d, 1, unaligned, @ptrCast(&y)));
    try equal(@as(f64, 99), y[0]);
    try expectError(error.BufferOverlap, binding.RawSolve.vector(f64, i32, h, 10, d, 1, @ptrCast(&x), @ptrCast(&values)));
    try equal(@as(f64, 2), values[0]);
    var op = try ctx.begin();
    _ = ctx.requestClose();
    try expectError(error.Closing, binding.RawSolve.vector(f64, i32, h, 99, d, 1, null, null));
    const result = try binding.RawSolve.vectorAdmitted(f64, i32, &op, h.?, 10, d, 1, @ptrCast(&x), @ptrCast(&y));
    try equal(@as(usize, 0), result.info);
    try equal(@as(f64, 2), y[0]);
    op.end();
    try Typed.destroy(h);
    try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
}

test "raw solve zero columns has no dense prefix or scratch and failed solve leaves output" {
    var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocator.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    var starts = [_]i32{0};
    var ends = [_]i32{1};
    var ord = [_]i32{0};
    var values = [_]f64{std.math.nan(f64)};
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i32, ctx, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
    allocator.fail_index = allocator.alloc_index;
    const d: binding.RawSolve.Descriptor = .{ .matrix_type = 23, .mode = 40, .diag = 50 };
    const zero = try binding.RawSolve.panel(f64, i32, h, 10, d, 101, 1, null, 0, 0, null, 0);
    try equal(@as(usize, 0), zero.info);
    try equal(@as(usize, 0), zero.completed_rhs);
    allocator.fail_index = std.math.maxInt(usize);
    values[0] = 0;
    const x = [_]f64{4};
    var y = [_]f64{99};
    const bad = try binding.RawSolve.vector(f64, i32, h, 10, d, 1, @ptrCast(&x), @ptrCast(&y));
    try equal(@as(usize, 1), bad.info);
    try equal(@as(f64, 99), y[0]);
    try equal(@as(usize, 1), ctx.snapshot().resource_pins);
    try Typed.destroy(h);
    try equal(allocator.allocated_bytes, allocator.freed_bytes);
    _ = ctx.requestClose();
    try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
}

test "raw solve resource pin overflow cleans previous same operation leases" {
    var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocator.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    Context.testing.configure(ctx, 2, null);
    var starts = [_]i32{0};
    var ends = [_]i32{1};
    var ord = [_]i32{0};
    var values = [_]f64{2};
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i32, ctx, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
    const d: binding.RawSolve.Descriptor = .{ .matrix_type = 23, .mode = 40, .diag = 50 };
    const x = [_]f64{4};
    var y = [_]f64{99};
    try expectError(error.CounterOverflow, binding.RawSolve.vector(f64, i32, h, 10, d, 1, @ptrCast(&x), @ptrCast(&y)));
    try equal(@as(f64, 99), y[0]);
    try equal(@as(usize, 0), ctx.snapshot().active_operations);
    try equal(@as(usize, 1), ctx.snapshot().resource_pins);
    try Typed.destroy(h);
    try equal(allocator.allocated_bytes, allocator.freed_bytes);
    _ = ctx.requestClose();
    try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
}

test "solve IEEE permits NaN Inf while checked failure preserves output all scalar families" {
    inline for (.{ f32, f64, Core.sparse.ComplexF32, Core.sparse.ComplexF64 }) |T| {
        const R = if (T == f32 or T == Core.sparse.ComplexF32) f32 else f64;
        var reference_ctx: FixtureReferenceContext = undefined;
        reference_ctx.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        var starts = [_]i32{0};
        var ends = [_]i32{1};
        var ord = [_]i32{0};
        var values = [_]T{number(T, 2, 0)};
        var h: ?Handles.Handle = null;
        try Typed.createBorrowed(T, i32, ctx, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
        const d: binding.RawSolve.Descriptor = .{ .matrix_type = 23, .mode = 40, .diag = 50 };
        for ([_]R{ std.math.nan(R), std.math.inf(R) }) |bad| {
            const x = [_]T{number(T, bad, 0)};
            var y = [_]T{number(T, 99, 9)};
            try expectError(error.NonFiniteInput, binding.RawSolve.vector(T, i32, h, 10, d, number(T, 1, 0), @ptrCast(&x), @ptrCast(&y)));
            try equal(number(T, 99, 9), y[0]);
            const result = try binding.RawSolve.vectorWithContract(T, i32, h, 10, d, number(T, 1, 0), @ptrCast(&x), @ptrCast(&y), .sparse_ieee_v1);
            try equal(@as(usize, 0), result.info);
            const out = if (comptime isComplex(T)) y[0].re else y[0];
            if (std.math.isNan(bad)) try expect(std.math.isNan(out)) else try expect(std.math.isInf(out));
        }
        values[0] = number(T, std.math.nan(R), 0);
        const x = [_]T{number(T, 4, 0)};
        var y = [_]T{number(T, 99, 9)};
        const result = try binding.RawSolve.vectorWithContract(T, i32, h, 10, d, number(T, 1, 0), @ptrCast(&x), @ptrCast(&y), .sparse_ieee_v1);
        try equal(@as(usize, 0), result.info);
        try expect(std.math.isNan(if (comptime isComplex(T)) y[0].re else y[0]));
        try Typed.destroy(h);
        _ = ctx.requestClose();
        try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    }
}

test "solve IEEE zero scale skips singular nonfinite numbers and scratch but retains raw prefixes" {
    var allocator = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(allocator.allocator(), fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    var starts = [_]i32{0};
    var ends = [_]i32{1};
    var ord = [_]i32{0};
    var values = [_]f64{0};
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i32, ctx, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
    allocator.fail_index = allocator.alloc_index;
    const d: binding.RawSolve.Descriptor = .{ .matrix_type = 23, .mode = 40, .diag = 50 };
    const x = [_]f64{std.math.nan(f64)};
    var y = [_]f64{99};
    const result = try binding.RawSolve.vectorWithContract(f64, i32, h, 10, d, 0, @ptrCast(&x), @ptrCast(&y), .sparse_ieee_v1);
    try equal(@as(usize, 0), result.info);
    try equal(@as(f64, 0), y[0]);
    try expectError(error.NullInput, binding.RawSolve.vectorWithContract(f64, i32, h, 10, d, 0, null, @ptrCast(&y), .sparse_ieee_v1));
    try equal(@as(usize, 1), ctx.snapshot().resource_pins);
    try Typed.destroy(h);
    try equal(allocator.allocated_bytes, allocator.freed_bytes);
    _ = ctx.requestClose();
    try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
}

test "solve IEEE complex unit no division preserves Inf and ignored diagonal N T H" {
    inline for (.{ Core.sparse.ComplexF32, Core.sparse.ComplexF64 }) |T| {
        const R = @TypeOf(@as(T, undefined).re);
        var reference_ctx: FixtureReferenceContext = undefined;
        reference_ctx.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
        const ctx = reference_ctx.borrowedContext();
        var starts = [_]i32{0};
        var ends = [_]i32{1};
        var ord = [_]i32{0};
        var values = [_]T{number(T, std.math.nan(R), std.math.nan(R))};
        var h: ?Handles.Handle = null;
        try Typed.createBorrowed(T, i32, ctx, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
        const d: binding.RawSolve.Descriptor = .{ .matrix_type = 23, .mode = 40, .diag = 51 };
        const x = [_]T{number(T, std.math.inf(R), 0)};
        for ([_]u32{ 10, 11, 12 }) |op| {
            var y = [_]T{number(T, 99, 9)};
            const result = try binding.RawSolve.vectorWithContract(T, i32, h, op, d, number(T, 1, 0), @ptrCast(&x), @ptrCast(&y), .sparse_ieee_v1);
            try equal(@as(usize, 0), result.info);
            try expect(std.math.isInf(y[0].re));
            try equal(@as(R, 0), y[0].im);
        }
        try Typed.destroy(h);
        _ = ctx.requestClose();
        try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
    }
}

test "solve IEEE finite scaling overflow propagates and zero diagonal still returns numerical failure" {
    var reference_ctx: FixtureReferenceContext = undefined;
    reference_ctx.initInPlace(std.testing.allocator, fixture.policy, .owned_closable, @import("zynum-execution").reference.defaultCacheDomain());
    const ctx = reference_ctx.borrowedContext();
    var starts = [_]i32{0};
    var ends = [_]i32{1};
    var ord = [_]i32{0};
    var values = [_]f64{1};
    var h: ?Handles.Handle = null;
    try Typed.createBorrowed(f64, i32, ctx, .{ .base = 0, .rows = 1, .columns = 1, .axis = .row, .starts = &starts, .ends = &ends, .ordinals = &ord, .values = &values }, &h);
    const d: binding.RawSolve.Descriptor = .{ .matrix_type = 23, .mode = 40, .diag = 50 };
    const x = [_]f64{2};
    var y = [_]f64{99};
    try expectError(error.NonFiniteScaling, binding.RawSolve.vector(f64, i32, h, 10, d, std.math.floatMax(f64), @ptrCast(&x), @ptrCast(&y)));
    try equal(@as(f64, 99), y[0]);
    const overflow = try binding.RawSolve.vectorWithContract(f64, i32, h, 10, d, std.math.floatMax(f64), @ptrCast(&x), @ptrCast(&y), .sparse_ieee_v1);
    try equal(@as(usize, 0), overflow.info);
    try expect(std.math.isInf(y[0]));
    values[0] = 0;
    y[0] = 99;
    const singular = try binding.RawSolve.vectorWithContract(f64, i32, h, 10, d, 1, @ptrCast(&x), @ptrCast(&y), .sparse_ieee_v1);
    try equal(@as(usize, 1), singular.info);
    try equal(@as(f64, 99), y[0]);
    try Typed.destroy(h);
    _ = ctx.requestClose();
    try equal(execution.context.CloseResult.closed, fixtureProgressClose(ctx));
}

const FixtureReferenceContext = @import("zynum-execution").reference.ReferenceContext;
fn fixtureProgressClose(ctx: *@import("zynum-execution").Context) @import("zynum-execution").CloseResult {
    var report: @import("zynum-execution").CallReport = .{};
    return ctx.tryProgressClose(&report);
}
