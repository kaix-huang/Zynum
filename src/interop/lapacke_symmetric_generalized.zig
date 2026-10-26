// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Real SYGV protocol over the configured Hermitian numerical owner.
//! All conversion and numerical scratch borrows the caller's admitted operation.
const std = @import("std");
const common = @import("lapacke_common.zig");

pub fn Adapter(comptime Core: type) type {
    return Engine(Core, .typed);
}
pub fn RawAdapter(comptime Core: type) type {
    return Engine(Core, .raw);
}
fn Engine(comptime Core: type, comptime kind: common.access.Kind) type {
    return struct {
        const n = Core.internal.numerics;
        const context = n.Execution.context;
        const H = n.Hermitian;
        const M = H.Managed(n.Execution);
        pub const OwnerIdentity = Core;
        pub const Error = M.Error || common.access.Fault(kind) || error{NumericalOutcomePendingABI};
        pub fn Arguments(comptime T: type) type {
            if (T != f32 and T != f64) @compileError("SYGV uses f32 or f64");
            return struct {
                layout: c_int,
                itype: i32 = 1,
                jobz: u8,
                uplo: u8,
                n: i32,
                a: common.access.Operand(T, true, kind) = null,
                lda: i32,
                b: common.access.Operand(T, true, kind) = null,
                ldb: i32,
                w: common.access.Operand(T, true, kind) = null,
                work: common.access.Operand(T, true, kind) = null,
                lwork: i32 = 0,
                native_options: H.core.Options = .{},
            };
        }
        fn capacity(comptime T: type, values: common.access.Operand(T, true, kind), count: usize) Error![]T {
            return common.access.capacity(n.Storage, kind, T, true, values, count);
        }
        fn operand(comptime T: type, values: []T) common.access.Operand(T, true, kind) {
            return if (kind == .typed) values else @ptrCast(values.ptr);
        }
        fn disjoint(comptime T: type, a: []const T, b: []const T) Error!void {
            const first = try n.Storage.byteRange(@intFromPtr(a.ptr), try n.Storage.byteLength(a.len, @sizeOf(T)));
            const second = try n.Storage.byteRange(@intFromPtr(b.ptr), try n.Storage.byteLength(b.len, @sizeOf(T)));
            if (n.Storage.overlaps(first, second)) return error.AliasingNotAllowed;
        }
        pub fn minimumWorkspace(size: i32) Error!i32 {
            if (size <= 0) return 1;
            return std.math.cast(i32, @as(i64, size) * 3 - 1) orelse error.IntegerNotRepresentable;
        }
        fn queryValue(comptime T: type, count: i32) T {
            var value: T = @floatFromInt(count);
            if (@as(f64, value) < @as(f64, @floatFromInt(count))) value = std.math.nextAfter(T, value, std.math.inf(T));
            return value;
        }
        fn upper(value: u8) ?bool {
            return switch (value) {
                'U', 'u' => true,
                'L', 'l' => false,
                else => null,
            };
        }
        fn wantsVectors(value: u8) bool {
            return value == 'V' or value == 'v';
        }
        fn controls(comptime T: type, args: Arguments(T)) i32 {
            if (args.itype < 1 or args.itype > 3) return -2;
            if (!wantsVectors(args.jobz) and args.jobz != 'N' and args.jobz != 'n') return -3;
            if (upper(args.uplo) == null) return -4;
            if (args.n < 0) return -5;
            if (args.lda < @max(1, args.n)) return -7;
            if (args.ldb < @max(1, args.n)) return -9;
            return 0;
        }
        fn matrixCount(size: usize, ld: i32) Error!usize {
            return n.Storage.columnMajorExtent(size, size, @intCast(ld));
        }
        fn copySelectedIn(comptime T: type, input: []const T, output: anytype, size: usize, ld: usize, is_upper: bool) void {
            @memset(output, .{ .re = 0, .im = 0 });
            for (0..size) |j| for (0..size) |i| {
                if (if (is_upper) i <= j else i >= j) output[i + j * size] = .{ .re = input[i + j * ld], .im = 0 };
            };
        }
        fn copySelectedOut(comptime T: type, input: anytype, output: []T, size: usize, ld: usize, is_upper: bool) void {
            for (0..size) |j| for (0..size) |i| {
                if (if (is_upper) i <= j else i >= j) output[i + j * ld] = input[i + j * size].re;
            };
        }
        fn columnWork(comptime T: type, operation: *const context.Operation, args: Arguments(T), original_a: []const T, original_b: []const T) Error!i32 {
            const invalid = controls(T, args);
            if (invalid != 0) return invalid;
            const minimum = try minimumWorkspace(args.n);
            const work_head = try capacity(T, args.work, 1);
            work_head[0] = queryValue(T, minimum);
            if (args.lwork != -1 and args.lwork < minimum) return -12;
            if (args.lwork == -1 or args.n == 0) return 0;
            const size: usize = @intCast(args.n);
            const work = try capacity(T, args.work, @intCast(args.lwork));
            const b = try capacity(T, args.b, try matrixCount(size, args.ldb));
            try disjoint(T, b, work);
            const square = try n.Storage.byteLength(size, size);
            const Z = if (T == f32) n.Scalar.ComplexF32 else n.Scalar.ComplexF64;
            var stage = context.ContextScratch(Z).acquire(operation, try n.Storage.byteLength(square, 3)) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            defer stage.deinit();
            const av = stage.lease.values[0..square];
            const bv = stage.lease.values[square .. 2 * square];
            const q = stage.lease.values[2 * square .. 3 * square];
            const is_upper = upper(args.uplo).?;
            copySelectedIn(T, b, bv, size, @intCast(args.ldb), is_upper);
            @memcpy(q, bv);
            const pivot = n.ComplexCholesky.factorizeWithContract(Z, .{ .values = q, .row_count = size, .column_count = size, .leading_dimension = size }, if (is_upper) .upper else .lower, .lapack_potrf);
            if (pivot != 0) {
                copySelectedOut(T, q, b, size, @intCast(args.ldb), is_upper);
                return args.n + @as(i32, @intCast(pivot));
            }
            // The column driver factors B before accessing A or eigenvalues.
            const a = try capacity(T, args.a, try matrixCount(size, args.lda));
            const values = try capacity(T, args.w, size);
            try disjoint(T, a, b);
            try disjoint(T, a, values);
            try disjoint(T, a, work);
            try disjoint(T, b, values);
            try disjoint(T, values, work);
            try disjoint(T, original_a, original_b);
            inline for (.{ original_a, original_b }) |original| {
                try disjoint(T, original, values);
                try disjoint(T, original, work);
            }
            copySelectedOut(T, q, b, size, @intCast(args.ldb), is_upper);
            copySelectedIn(T, a, av, size, @intCast(args.lda), is_upper);
            const result = M.runScratchAdmitted(Z, operation, .{
                .a = .{ .values = av, .row_count = size, .column_count = size, .leading_dimension = size },
                .b = .{ .values = bv, .row_count = size, .column_count = size, .leading_dimension = size },
                .problem = switch (args.itype) {
                    1 => .ax_lambda_bx,
                    2 => .abx_lambda_x,
                    3 => .bax_lambda_x,
                    else => unreachable,
                },
                .triangle = if (is_upper) .upper else .lower,
                .eigenvalues = values,
                .vectors = if (wantsVectors(args.jobz)) .{ .values = q, .row_count = size, .column_count = size, .leading_dimension = size } else null,
                .contract = .heev_ieee_v1,
                .options = args.native_options,
            }) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            if (result.status == .not_positive_definite) return args.n + @as(i32, @intCast(result.failing_pivot));
            if (result.status == .range_limit) return error.NumericalOutcomePendingABI;
            if (wantsVectors(args.jobz)) for (0..size) |j| for (0..size) |i| {
                a[i + j * @as(usize, @intCast(args.lda))] = q[i + j * size].re;
            };
            return @intCast(result.unconverged);
        }
        pub fn workAdmitted(comptime T: type, operation: *const context.Operation, args: Arguments(T)) Error!i32 {
            _ = operation.borrowedContext();
            if (args.layout != 101 and args.layout != 102) return -1;
            if (args.layout == 102) return columnWork(T, operation, args, &.{}, &.{});
            if (args.lda < args.n) return -7;
            if (args.ldb < args.n) return -9;
            var call = args;
            call.layout = 102;
            call.lda = @max(1, args.n);
            call.ldb = @max(1, args.n);
            if (args.lwork == -1) return columnWork(T, operation, call, &.{}, &.{});
            const ld: usize = @intCast(call.lda);
            const count = try n.Storage.byteLength(ld, ld);
            var row_a = context.ContextScratch(T).acquire(operation, count) catch |err| switch (err) {
                error.OutOfMemory => return -1011,
                else => return err,
            };
            defer row_a.deinit();
            var row_b = context.ContextScratch(T).acquire(operation, count) catch |err| switch (err) {
                error.OutOfMemory => return -1011,
                else => return err,
            };
            defer row_b.deinit();
            call.a = operand(T, row_a.lease.values);
            call.b = operand(T, row_b.lease.values);
            const size: usize = @intCast(@max(0, args.n));
            var a: []T = &.{};
            var b: []T = &.{};
            if (size != 0) {
                a = try capacity(T, args.a, try matrixCount(size, args.lda));
                b = try capacity(T, args.b, try matrixCount(size, args.ldb));
                for (0..size) |j| for (0..size) |i| {
                    row_a.lease.values[i + j * ld] = a[i * @as(usize, @intCast(args.lda)) + j];
                    row_b.lease.values[i + j * ld] = b[i * @as(usize, @intCast(args.ldb)) + j];
                };
            }
            const info = try columnWork(T, operation, call, a, b);
            if (info != -1010) for (0..size) |j| for (0..size) |i| {
                a[i * @as(usize, @intCast(args.lda)) + j] = row_a.lease.values[i + j * ld];
                b[i * @as(usize, @intCast(args.ldb)) + j] = row_b.lease.values[i + j * ld];
            };
            return info;
        }
        fn callbacks(borrowed: anytype) @TypeOf(.{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event }) {
            return .{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
        }
        fn event(comptime T: type, services: anytype, channel: common.HookKind, entry: common.Entry, argument: i32) void {
            services.error_event(services.context, .{ .kind = channel, .entry = entry, .scalar = if (T == f32) 's' else 'd', .argument = argument, .routine = .sygv });
        }
        pub fn workWithDiagnosticsAdmitted(comptime T: type, operation: *const context.Operation, borrowed: anytype, args: Arguments(T)) Error!i32 {
            const services = callbacks(borrowed);
            const info = try workAdmitted(T, operation, args);
            if (info < 0) {
                if (info == -1 or info == -1010 or info == -1011 or (args.layout == 101 and (args.lda < args.n or args.ldb < args.n))) {
                    event(T, services, .lapacke, .work, info);
                } else {
                    event(T, services, .fortran_returning_test, .fortran_sygv, -info - 1);
                }
            }
            return info;
        }
        fn highCompute(comptime T: type, operation: *const context.Operation, services: anytype, args: Arguments(T)) Error!i32 {
            var query: [1]T = undefined;
            var call = args;
            call.work = operand(T, &query);
            call.lwork = -1;
            const info = try workWithDiagnosticsAdmitted(T, operation, services, call);
            if (info != 0) return info;
            if (!std.math.isFinite(query[0]) or @as(f64, query[0]) > std.math.maxInt(i32)) return error.IntegerNotRepresentable;
            call.lwork = @intFromFloat(query[0]);
            var work = try context.ContextScratch(T).acquire(operation, @intCast(call.lwork));
            defer work.deinit();
            call.work = operand(T, work.lease.values);
            return workWithDiagnosticsAdmitted(T, operation, services, call);
        }
        pub fn highAdmitted(comptime compile_nan_scan: bool, comptime T: type, operation: *const context.Operation, borrowed: anytype, args: Arguments(T)) Error!i32 {
            _ = operation.borrowedContext();
            const services = callbacks(borrowed);
            if (args.layout != 101 and args.layout != 102) {
                event(T, services, .lapacke, .high, -1);
                return -1;
            }
            if (comptime compile_nan_scan) {
                if (services.nan_enabled(services.context)) {
                    const Provider = struct {
                        pub const Services = @TypeOf(services);
                        pub const Storage = n.Storage;
                        pub const Scalar = n.Scalar;
                    };
                    const helper = common.HelpersWithAccess(Provider, kind);
                    const a: common.access.Operand(T, false, kind) = args.a;
                    const b: common.access.Operand(T, false, kind) = args.b;
                    if (try helper.scanTriangle(T, i32, .{ .layout = args.layout, .uplo = args.uplo, .n = args.n, .lda = args.lda, .values = a })) return -6;
                    if (try helper.scanTriangle(T, i32, .{ .layout = args.layout, .uplo = args.uplo, .n = args.n, .lda = args.ldb, .values = b })) return -8;
                }
            }
            const info = highCompute(T, operation, services, args) catch |err| switch (err) {
                error.OutOfMemory => {
                    event(T, services, .lapacke, .high, -1010);
                    return -1010;
                },
                else => return err,
            };
            return info;
        }
    };
}
