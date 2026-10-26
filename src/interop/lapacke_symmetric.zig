// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Typed/raw real-symmetric frontend over the configured Hermitian owner.
//! Real input is embedded in the existing complex scalar representation. The
//! shared IEEE tridiagonal driver supplies actual unconverged-edge INFO counts.
//! Every temporary borrows the caller's existing Context and Operation.
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
        const execution = n.Execution;
        const context = execution.context;
        const H = n.Hermitian;
        const M = H.Managed(execution);
        pub const OwnerIdentity = Core;
        pub const Error = M.Error || common.access.Fault(kind) || error{NumericalOutcomePendingABI};
        pub fn Arguments(comptime T: type) type {
            if (T != f32 and T != f64) @compileError("SYEV uses f32 or f64");
            return struct {
                layout: c_int,
                jobz: u8,
                uplo: u8,
                n: i32,
                a: common.access.Operand(T, true, kind) = null,
                lda: i32,
                w: common.access.Operand(T, true, kind) = null,
                work: common.access.Operand(T, true, kind) = null,
                lwork: i32 = 0,
                // Internal numerical-test controls; no extra C ABI parameter.
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
        fn callbacks(borrowed: anytype) @TypeOf(.{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event }) {
            return .{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
        }
        fn event(comptime T: type, services: anytype, channel: common.HookKind, entry: common.Entry, argument: i32) void {
            services.error_event(services.context, .{ .kind = channel, .entry = entry, .scalar = if (T == f32) 's' else 'd', .argument = argument, .routine = .syev });
        }
        fn notify(comptime T: type, services: anytype, args: Arguments(T), info: i32) void {
            if (info >= 0) return;
            if (info == -1 or info == -1010 or info == -1011 or (args.layout == 101 and args.lda < args.n)) {
                event(T, services, .lapacke, .work, info);
            } else {
                event(T, services, .fortran_returning_test, .fortran_syev, -info - 1);
            }
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
        fn upper(uplo: u8) ?bool {
            return switch (uplo) {
                'U', 'u' => true,
                'L', 'l' => false,
                else => null,
            };
        }
        fn wantsVectors(jobz: u8) bool {
            return jobz == 'V' or jobz == 'v';
        }
        fn matrixCount(size: usize, ld: i32) Error!usize {
            return n.Storage.columnMajorExtent(size, size, @intCast(ld));
        }
        fn controls(comptime T: type, args: Arguments(T)) i32 {
            if (!wantsVectors(args.jobz) and args.jobz != 'N' and args.jobz != 'n') return -2;
            if (upper(args.uplo) == null) return -3;
            if (args.n < 0) return -4;
            if (args.lda < @max(1, args.n)) return -6;
            return 0;
        }
        fn compute(comptime T: type, operation: *const context.Operation, args: Arguments(T), a: []T, values: []T) Error!i32 {
            const Z = if (T == f32) n.Scalar.ComplexF32 else n.Scalar.ComplexF64;
            const size: usize = @intCast(args.n);
            const square = try n.Storage.byteLength(size, size);
            const vector_output = wantsVectors(args.jobz);
            const count = try n.Storage.byteLength(square, if (vector_output) 2 else 1);
            var stage = context.ContextScratch(Z).acquire(operation, count) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            defer stage.deinit();
            const input = stage.lease.values[0..square];
            const vectors = stage.lease.values[square..count];
            @memset(input, .{ .re = 0, .im = 0 });
            const is_upper = upper(args.uplo).?;
            const ld: usize = @intCast(args.lda);
            for (0..size) |j| for (0..size) |i| {
                if (if (is_upper) i <= j else i >= j) input[i + j * size] = .{ .re = a[i + j * ld], .im = 0 };
            };
            const result = M.runScratchAdmitted(Z, operation, .{
                .a = .{ .values = input, .row_count = size, .column_count = size, .leading_dimension = size },
                .triangle = if (is_upper) .upper else .lower,
                .eigenvalues = values,
                .vectors = if (vector_output) .{ .values = vectors, .row_count = size, .column_count = size, .leading_dimension = size } else null,
                .contract = .heev_ieee_v1,
                .options = args.native_options,
            }) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            if (result.status == .range_limit) return error.NumericalOutcomePendingABI;
            if (vector_output) for (0..size) |j| for (0..size) |i| {
                a[i + j * ld] = vectors[i + j * size].re;
            };
            return @intCast(result.unconverged);
        }
        fn columnWork(comptime T: type, operation: *const context.Operation, args: Arguments(T), original: []const T) Error!i32 {
            const invalid = controls(T, args);
            if (invalid != 0) return invalid;
            const minimum = try minimumWorkspace(args.n);
            const work = try capacity(T, args.work, 1);
            work[0] = queryValue(T, minimum);
            if (args.lwork != -1 and args.lwork < minimum) return -9;
            if (args.lwork == -1 or args.n == 0) return 0;
            const size: usize = @intCast(args.n);
            const a = try capacity(T, args.a, try matrixCount(size, args.lda));
            const values = try capacity(T, args.w, size);
            try disjoint(T, a, values);
            try disjoint(T, a, work);
            try disjoint(T, values, work);
            try disjoint(T, original, values);
            try disjoint(T, original, work);
            if (size == 1) {
                values[0] = a[0];
                work[0] = 2;
                if (wantsVectors(args.jobz)) a[0] = 1;
                return 0;
            }
            return compute(T, operation, args, a, values);
        }
        /// The row-major conversion precedes Fortran scalar validation on a
        /// nonquery call. Query and invalid-row-LDA paths do not access A.
        pub fn workAdmitted(comptime T: type, operation: *const context.Operation, args: Arguments(T)) Error!i32 {
            _ = operation.borrowedContext();
            if (args.layout != 101 and args.layout != 102) return -1;
            if (args.layout == 102) return columnWork(T, operation, args, &.{});
            if (args.lda < args.n) return -6;
            var converted = args;
            converted.layout = 102;
            converted.lda = @max(1, args.n);
            if (args.lwork == -1) return columnWork(T, operation, converted, &.{});
            const ld: usize = @intCast(converted.lda);
            var row_stage = context.ContextScratch(T).acquire(operation, try n.Storage.byteLength(ld, ld)) catch |err| switch (err) {
                error.OutOfMemory => return -1011,
                else => return err,
            };
            defer row_stage.deinit();
            @memset(row_stage.lease.values, 0);
            converted.a = operand(T, row_stage.lease.values);
            const size: usize = @intCast(@max(0, args.n));
            const selected = upper(args.uplo);
            var original: []T = &.{};
            if (size != 0 and selected != null) {
                original = try capacity(T, args.a, try matrixCount(size, args.lda));
                const source_ld: usize = @intCast(args.lda);
                for (0..size) |j| for (0..size) |i| {
                    if (if (selected.?) i <= j else i >= j) row_stage.lease.values[i + j * ld] = original[i * source_ld + j];
                };
            }
            const info = try columnWork(T, operation, converted, original);
            const whole = wantsVectors(args.jobz);
            if (info != -1010 and size != 0 and (whole or selected != null)) {
                if (original.len == 0) original = try capacity(T, args.a, try matrixCount(size, args.lda));
                const output_ld: usize = @intCast(args.lda);
                for (0..size) |j| for (0..size) |i| {
                    if (whole or (if (selected.?) i <= j else i >= j)) original[i * output_ld + j] = row_stage.lease.values[i + j * ld];
                };
            }
            return info;
        }
        pub fn workWithDiagnosticsAdmitted(comptime T: type, operation: *const context.Operation, borrowed: anytype, args: Arguments(T)) Error!i32 {
            const services = callbacks(borrowed);
            const result = try workAdmitted(T, operation, args);
            notify(T, services, args, result);
            return result;
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
                    const source: common.access.Operand(T, false, kind) = args.a;
                    if (try helper.scanTriangle(T, i32, .{ .layout = args.layout, .uplo = args.uplo, .n = args.n, .lda = args.lda, .values = source })) return -5;
                }
            }
            const info = highCompute(T, operation, services, args) catch |err| switch (err) {
                error.OutOfMemory => @as(i32, -1010),
                else => return err,
            };
            if (info == -1010) event(T, services, .lapacke, .high, info);
            return info;
        }
    };
}
