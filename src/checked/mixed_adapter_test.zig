const std = @import("std");
const api = @import("implementation.zig");
const abi = @import("abi_types.zig");
const sparse = @import("zynum-core").sparse;

pub fn checkAdapter0() !void {
    inline for (.{ sparse.ComplexF32, sparse.ComplexF64 }) |T| {
        inline for (.{ .{ i64, i32 }, .{ u64, u32 }, .{ i32, i64 }, .{ u32, u64 } }) |types| {
            const O = types[0];
            const I = types[1];
            const offsets = [_]O{ 0, 1, 2 };
            const indices = [_]I{ 0, 1 };
            const values = [_]T{ .{ .re = 1, .im = 1 }, .{ .re = 2, .im = -1 } };
            const a: abi.csr = .{ .row_offsets = .{ .data = &offsets, .length = 3 }, .column_indices = .{ .data = &indices, .length = 2 }, .values = .{ .data = &values, .length = 2 }, .rows = 2, .cols = 2 };
            const x = [_]T{ .{ .re = 2, .im = 1 }, .{ .re = 1, .im = -2 }, .{ .re = 77, .im = 88 }, .{ .re = 2, .im = 1 }, .{ .re = 1, .im = -2 } };
            for ([_]i32{ 0, 2 }) |op| {
                var y: [5]T = @splat(.{ .re = 91, .im = 92 });
                const one: T = .{ .re = 1, .im = 0 };
                const zero: T = .{ .re = 0, .im = 0 };
                const first: T = if (op == 0) .{ .re = 1, .im = 3 } else .{ .re = 3, .im = -1 };
                const second: T = if (op == 0) .{ .re = 0, .im = -5 } else .{ .re = 4, .im = -3 };
                const mv = api.spmvWithIndexTypes(T, O, I, a, op, one, .{ .data = &x, .length = 2 }, zero, .{ .data = &y, .length = 2 });
                try std.testing.expectEqual(abi.OK, mv.status);
                try std.testing.expectEqual(first, y[0]);
                try std.testing.expectEqual(second, y[1]);
                const mm = api.spmmWithIndexTypes(T, O, I, a, op, one, .{ .storage = .{ .data = &x, .length = 5 }, .rows = 2, .cols = 2, .ld = 3 }, zero, .{ .storage = .{ .data = &y, .length = 5 }, .rows = 2, .cols = 2, .ld = 3 });
                try std.testing.expectEqual(abi.OK, mm.status);
                try std.testing.expectEqual(first, y[3]);
                try std.testing.expectEqual(second, y[4]);
                try std.testing.expectEqual(T{ .re = 91, .im = 92 }, y[2]);
                const before = y;
                const bad = api.spmmWithIndexTypes(T, O, I, a, op, zero, .{ .storage = .{ .data = @ptrFromInt(1), .length = 5 }, .rows = 2, .cols = 2, .ld = 3 }, zero, .{ .storage = .{ .data = &y, .length = 5 }, .rows = 2, .cols = 2, .ld = 3 });
                try std.testing.expectEqual(abi.MISALIGNED_POINTER, bad.status);
                try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&y));
            }
        }
    }
}

pub fn checkAdapter1() !void {
    const a: abi.csr = .{ .row_offsets = .{ .data = null, .length = 0 }, .column_indices = .{ .data = null, .length = 0 }, .values = .{ .data = null, .length = 0 }, .rows = 0, .cols = 0 };
    const r = api.spmv(f64, u32, a, 2, 1, .{ .data = null, .length = 0 }, 0, .{ .data = null, .length = 0 });
    try std.testing.expectEqual(abi.INVALID_FLAG, r.status);
}

pub fn checkAdapter2() !void {
    var offsets = [_]i64{ 0, 99, 2 };
    const indices = [_]i32{ 0, 1 };
    const values = [_]f64{ 2, 3 };
    var a: abi.csr = .{ .row_offsets = .{ .data = &offsets, .length = 3 }, .column_indices = .{ .data = &indices, .length = 2 }, .values = .{ .data = &values, .length = 2 }, .rows = 2, .cols = 2 };
    const x = [_]f64{ 5, 7 };
    var y = [_]f64{ 91, 92 };
    const bad_output = api.spmvWithIndexTypes(f64, i64, i32, a, 0, 1, .{ .data = &x, .length = 2 }, 0, .{ .data = @ptrFromInt(1), .length = 2 });
    try std.testing.expectEqual(abi.MISALIGNED_POINTER, bad_output.status);
    const bad_structure = api.spmvWithIndexTypes(f64, i64, i32, a, 0, 1, .{ .data = &x, .length = 2 }, 0, .{ .data = &y, .length = 2 });
    try std.testing.expectEqual(abi.INVALID_SPARSE_STRUCTURE, bad_structure.status);
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &y);
    offsets[1] = 1;
    a.row_offsets.data = null;
    const null_offsets = api.spmvWithIndexTypes(f64, i64, i32, a, 0, 1, .{ .data = &x, .length = 2 }, 0, .{ .data = &y, .length = 2 });
    try std.testing.expectEqual(abi.NULL_POINTER, null_offsets.status);
    a.row_offsets.data = &offsets;
    a.values.length = std.math.maxInt(i64);
    a.column_indices.length = std.math.maxInt(i64);
    const huge = api.spmvWithIndexTypes(f64, i64, i32, a, 0, 1, .{ .data = &x, .length = 2 }, 0, .{ .data = &y, .length = 2 });
    try std.testing.expectEqual(abi.SIZE_OVERFLOW, huge.status);
    try std.testing.expectEqualSlices(f64, &.{ 91, 92 }, &y);
}
