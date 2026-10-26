// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! SYEVD/HEEVD and SYGVD/HEGVD over the existing Hermitian numerical owner. Caller
//! workspaces and native scratch remain distinct and share one admitted operation.
const std = @import("std");
const common = @import("lapacke_common.zig");
pub const policy = @import("lapacke_evd_policy.zig");

pub fn Adapter(comptime Core: type) type {
    return Engine(Core, .typed, false);
}
pub fn RawAdapter(comptime Core: type) type {
    return Engine(Core, .raw, false);
}
pub fn GeneralizedAdapter(comptime Core: type) type {
    return Engine(Core, .typed, true);
}
pub fn RawGeneralizedAdapter(comptime Core: type) type {
    return Engine(Core, .raw, true);
}
fn Engine(comptime Core: type, comptime kind: common.access.Kind, comptime generalized: bool) type {
    return struct {
        const N = Core.internal.numerics;
        const S = N.Scalar;
        const context = N.Execution.context;
        const H = N.Hermitian;
        const M = H.Managed(N.Execution);
        pub const OwnerIdentity = Core;
        pub const Error = M.Error || common.access.Fault(kind) || policy.Error || error{NumericalOutcomePendingABI};

        pub fn Arguments(comptime T: type) type {
            if (T != f32 and T != f64 and T != S.ComplexF32 and T != S.ComplexF64) @compileError("EVD scalar type");
            return struct {
                layout: c_int,
                itype: i32 = 1,
                jobz: u8,
                uplo: u8,
                n: i32,
                a: common.access.Operand(T, true, kind) = null,
                lda: i32,
                b: common.access.Operand(T, true, kind) = null,
                ldb: i32 = 0,
                w: common.access.Operand(S.Real(T), true, kind) = null,
                work: common.access.Operand(T, true, kind) = null,
                lwork: i32 = 0,
                rwork: common.access.Operand(S.Real(T), true, kind) = null,
                lrwork: i32 = 0,
                iwork: common.access.Operand(i32, true, kind) = null,
                liwork: i32 = 0,
                native_options: H.core.Options = .{},
            };
        }
        fn family(comptime T: type) policy.Family {
            return if (S.isComplex(T)) .complex else .real;
        }
        fn vectors(jobz: u8) bool {
            return jobz == 'V' or jobz == 'v';
        }
        fn triangle(uplo: u8) ?bool {
            return switch (uplo) {
                'U', 'u' => true,
                'L', 'l' => false,
                else => null,
            };
        }
        fn lengths(comptime T: type, args: Arguments(T)) policy.Workspace(i32) {
            return .{ .work = args.lwork, .rwork = args.lrwork, .iwork = args.liwork };
        }
        fn capacity(comptime T: type, values: common.access.Operand(T, true, kind), count: usize) Error![]T {
            return common.access.capacity(N.Storage, kind, T, true, values, count);
        }
        fn operand(comptime T: type, values: []T) common.access.Operand(T, true, kind) {
            return if (kind == .typed) values else @ptrCast(values.ptr);
        }
        fn disjoint(comptime A: type, a: []const A, comptime B: type, b: []const B) Error!void {
            const first = try N.Storage.byteRange(@intFromPtr(a.ptr), try N.Storage.byteLength(a.len, @sizeOf(A)));
            const second = try N.Storage.byteRange(@intFromPtr(b.ptr), try N.Storage.byteLength(b.len, @sizeOf(B)));
            if (N.Storage.overlaps(first, second)) return error.AliasingNotAllowed;
        }
        fn controls(comptime T: type, args: Arguments(T)) i32 {
            if (generalized and (args.itype < 1 or args.itype > 3)) return -2;
            if (!vectors(args.jobz) and args.jobz != 'N' and args.jobz != 'n') return if (generalized) -3 else -2;
            if (triangle(args.uplo) == null) return if (generalized) -4 else -3;
            if (args.n < 0) return if (generalized) -5 else -4;
            if (args.lda < @max(1, args.n)) return if (generalized) -7 else -6;
            if (generalized and args.ldb < @max(1, args.n)) return -9;
            return 0;
        }
        fn event(comptime T: type, services: anytype, channel: common.HookKind, entry: common.Entry, info: i32) void {
            services.error_event(services.context, .{
                .kind = channel,
                .entry = entry,
                .scalar = if (T == f32) 's' else if (T == f64) 'd' else if (T == S.ComplexF32) 'c' else 'z',
                .argument = info,
                .routine = if (generalized) (if (S.isComplex(T)) .hegvd else .sygvd) else (if (S.isComplex(T)) .heevd else .syevd),
            });
        }
        fn notify(comptime T: type, services: anytype, args: Arguments(T), info: i32) void {
            if (info >= 0) return;
            if (info == -1 or info == -1010 or info == -1011 or (args.layout == 101 and (args.lda < args.n or (generalized and args.ldb < args.n)))) {
                event(T, services, .lapacke, .work, info);
            } else {
                const entry: common.Entry = if (generalized) (if (S.isComplex(T)) .fortran_hegvd else .fortran_sygvd) else (if (S.isComplex(T)) .fortran_heevd else .fortran_syevd);
                event(T, services, .fortran_returning_test, entry, -info - 1);
            }
        }
        fn callbacks(borrowed: anytype) @TypeOf(.{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event }) {
            return .{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
        }
        fn queryScalar(comptime T: type, count: i32) Error!T {
            const value = try policy.queryValue(S.Real(T), i32, count);
            return if (comptime S.isComplex(T)) .{ .re = value, .im = 0 } else value;
        }
        fn columnWork(comptime T: type, operation: *const context.Operation, args: Arguments(T), original: []const T, original_b: []const T) Error!i32 {
            const invalid = controls(T, args);
            if (invalid != 0) return invalid;
            const required = try policy.minimumWorkspace(i32, args.n, vectors(args.jobz), family(T));
            const work_head = try capacity(T, args.work, 1);
            const integer_head = try capacity(i32, args.iwork, 1);
            const R = S.Real(T);
            const real_head = if (comptime S.isComplex(T)) try capacity(R, args.rwork, 1) else &.{};
            // Validate every query destination, including disjointness, before
            // publishing any head. Query never admits an input matrix or W.
            try disjoint(T, work_head, i32, integer_head);
            if (comptime S.isComplex(T)) {
                try disjoint(T, work_head, R, real_head);
                try disjoint(i32, integer_head, R, real_head);
            }
            const work_value = try queryScalar(T, required.work);
            const real_value = if (comptime S.isComplex(T)) try policy.queryValue(R, i32, required.rwork) else 1;
            work_head[0] = work_value;
            integer_head[0] = required.iwork;
            if (comptime S.isComplex(T)) real_head[0] = real_value;
            const workspace_info = policy.workspaceInfo(i32, family(T), lengths(T, args), required);
            if (workspace_info != 0) return workspace_info - (if (generalized) @as(i32, 3) else 0);
            if (policy.isQuery(i32, family(T), lengths(T, args)) or args.n == 0) return 0;
            if (comptime generalized) return computeGeneralized(T, operation, args, original, original_b);

            const size: usize = @intCast(args.n);
            const a = try capacity(T, args.a, try N.Storage.columnMajorExtent(size, size, @intCast(args.lda)));
            const values = try capacity(R, args.w, size);
            const work = try capacity(T, args.work, if (size == 1) 1 else @intCast(args.lwork));
            const integers = try capacity(i32, args.iwork, if (size == 1) 1 else @intCast(args.liwork));
            const real = if (comptime S.isComplex(T)) try capacity(R, args.rwork, if (size == 1) 1 else @intCast(args.lrwork)) else &.{};
            try disjoint(T, a, R, values);
            inline for (.{ a, original }) |input| {
                try disjoint(T, input, T, work);
                try disjoint(T, input, i32, integers);
                if (comptime S.isComplex(T)) try disjoint(T, input, R, real);
            }
            try disjoint(T, original, R, values);
            try disjoint(R, values, T, work);
            try disjoint(R, values, i32, integers);
            try disjoint(T, work, i32, integers);
            if (comptime S.isComplex(T)) {
                try disjoint(R, values, R, real);
                try disjoint(T, work, R, real);
                try disjoint(i32, integers, R, real);
            }
            if (size == 1) {
                values[0] = if (comptime S.isComplex(T)) a[0].re else a[0];
                if (vectors(args.jobz)) a[0] = if (comptime S.isComplex(T)) .{ .re = 1, .im = 0 } else 1;
                return 0;
            }
            return compute(T, operation, args, a, values);
        }
        fn compute(comptime T: type, operation: *const context.Operation, args: Arguments(T), a: []T, values: []S.Real(T)) Error!i32 {
            const C = if (comptime S.isComplex(T)) T else if (T == f32) S.ComplexF32 else S.ComplexF64;
            const size: usize = @intCast(args.n);
            const square = try N.Storage.byteLength(size, size);
            const count = try N.Storage.byteLength(square, if (vectors(args.jobz)) 2 else 1);
            var stage = context.ContextScratch(C).acquire(operation, count) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            defer stage.deinit();
            const input = stage.lease.values[0..square];
            const output = stage.lease.values[square..count];
            @memset(input, .{ .re = 0, .im = 0 });
            const selected = triangle(args.uplo).?;
            const ld: usize = @intCast(args.lda);
            for (0..size) |j| for (0..size) |i| {
                if (if (selected) i <= j else i >= j) input[i + j * size] = if (comptime S.isComplex(T)) a[i + j * ld] else .{ .re = a[i + j * ld], .im = 0 };
            };
            const result = M.runScratchAdmitted(C, operation, .{
                .a = .{ .values = input, .row_count = size, .column_count = size, .leading_dimension = size },
                .triangle = if (selected) .upper else .lower,
                .eigenvalues = values,
                .vectors = if (vectors(args.jobz)) .{ .values = output, .row_count = size, .column_count = size, .leading_dimension = size } else null,
                .contract = .heev_ieee_v1,
                .options = args.native_options,
            }) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            if (result.status == .range_limit) return error.NumericalOutcomePendingABI;
            if (vectors(args.jobz)) for (0..size) |j| for (0..size) |i| {
                a[i + j * ld] = if (comptime S.isComplex(T)) output[i + j * size] else output[i + j * size].re;
            };
            return policy.iterationInfo(i32, args.n, vectors(args.jobz), result.unconverged, result.failed_block);
        }
        fn copySelectedIn(comptime T: type, source: []const T, target: anytype, size: usize, ld: usize, upper: bool) void {
            @memset(target, .{ .re = 0, .im = 0 });
            for (0..size) |j| for (0..size) |i| {
                if (if (upper) i <= j else i >= j) target[i + j * size] = if (comptime S.isComplex(T)) source[i + j * ld] else .{ .re = source[i + j * ld], .im = 0 };
            };
        }
        fn copySelectedOut(comptime T: type, source: anytype, target: []T, size: usize, ld: usize, upper: bool) void {
            for (0..size) |j| for (0..size) |i| {
                if (if (upper) i <= j else i >= j) target[i + j * ld] = if (comptime S.isComplex(T)) source[i + j * size] else source[i + j * size].re;
            };
        }
        fn computeGeneralized(comptime T: type, operation: *const context.Operation, args: Arguments(T), original_a: []const T, original_b: []const T) Error!i32 {
            const R = S.Real(T);
            const C = if (comptime S.isComplex(T)) T else if (T == f32) S.ComplexF32 else S.ComplexF64;
            const size: usize = @intCast(args.n);
            const square = try N.Storage.byteLength(size, size);
            const lda: usize = @intCast(args.lda);
            const ldb: usize = @intCast(args.ldb);
            const selected = triangle(args.uplo).?;
            const b = try capacity(T, args.b, try N.Storage.columnMajorExtent(size, size, ldb));
            try disjoint(T, b, T, try capacity(T, args.work, 1));
            try disjoint(T, b, i32, try capacity(i32, args.iwork, 1));
            if (comptime S.isComplex(T)) try disjoint(T, b, R, try capacity(R, args.rwork, 1));
            var b_stage = context.ContextScratch(C).acquire(operation, try N.Storage.byteLength(square, 2)) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            defer b_stage.deinit();
            const input_b = b_stage.lease.values[0..square];
            const factor = b_stage.lease.values[square..];
            copySelectedIn(T, b, input_b, size, ldb, selected);
            @memcpy(factor, input_b);
            const pivot = N.ComplexCholesky.factorizeWithContract(C, .{ .values = factor, .row_count = size, .column_count = size, .leading_dimension = size }, if (selected) .upper else .lower, .lapack_potrf);
            copySelectedOut(T, factor, b, size, ldb, selected);
            // The column driver factors B and publishes its partial factor
            // before it reads A, W or the unused tails of the work arrays.
            if (pivot != 0) return args.n + @as(i32, @intCast(pivot));
            const a = try capacity(T, args.a, try N.Storage.columnMajorExtent(size, size, lda));
            const values = try capacity(R, args.w, size);
            const work = try capacity(T, args.work, if (size == 1) 1 else @intCast(args.lwork));
            const integers = try capacity(i32, args.iwork, if (size == 1) 1 else @intCast(args.liwork));
            const real = if (comptime S.isComplex(T)) try capacity(R, args.rwork, if (size == 1) 1 else @intCast(args.lrwork)) else &.{};
            try disjoint(T, a, T, b);
            inline for (.{ a, b, original_a, original_b }) |input| {
                try disjoint(T, input, R, values);
                try disjoint(T, input, T, work);
                try disjoint(T, input, i32, integers);
                if (comptime S.isComplex(T)) try disjoint(T, input, R, real);
            }
            try disjoint(R, values, T, work);
            try disjoint(R, values, i32, integers);
            try disjoint(T, work, i32, integers);
            if (comptime S.isComplex(T)) {
                try disjoint(R, values, R, real);
                try disjoint(T, work, R, real);
                try disjoint(i32, integers, R, real);
            }
            const count = try N.Storage.byteLength(square, if (vectors(args.jobz)) 2 else 1);
            var a_stage = context.ContextScratch(C).acquire(operation, count) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            defer a_stage.deinit();
            const input_a = a_stage.lease.values[0..square];
            const output = a_stage.lease.values[square..];
            copySelectedIn(T, a, input_a, size, lda, selected);
            var options = args.native_options;
            options.generalized_failure_vectors = .keep_reduced;
            const result = M.runScratchAdmitted(C, operation, .{
                .a = .{ .values = input_a, .row_count = size, .column_count = size, .leading_dimension = size },
                .b = .{ .values = input_b, .row_count = size, .column_count = size, .leading_dimension = size },
                .problem = switch (args.itype) {
                    1 => .ax_lambda_bx,
                    2 => .abx_lambda_x,
                    3 => .bax_lambda_x,
                    else => unreachable,
                },
                .triangle = if (selected) .upper else .lower,
                .eigenvalues = values,
                .vectors = if (vectors(args.jobz)) .{ .values = output, .row_count = size, .column_count = size, .leading_dimension = size } else null,
                .contract = .heev_ieee_v1,
                .options = options,
            }) catch |err| switch (err) {
                error.OutOfMemory => return -1010,
                else => return err,
            };
            if (result.status == .not_positive_definite) return args.n + @as(i32, @intCast(result.failing_pivot));
            if (result.status == .range_limit) return error.NumericalOutcomePendingABI;
            if (vectors(args.jobz)) for (0..size) |j| for (0..size) |i| {
                a[i + j * lda] = if (comptime S.isComplex(T)) output[i + j * size] else output[i + j * size].re;
            };
            return policy.iterationInfo(i32, args.n, vectors(args.jobz), result.unconverged, result.failed_block);
        }
        fn generalizedRowWork(comptime T: type, operation: *const context.Operation, args: Arguments(T)) Error!i32 {
            if (args.lda < args.n) return -7;
            if (args.ldb < args.n) return -9;
            var converted = args;
            converted.layout = 102;
            converted.lda = @max(1, args.n);
            converted.ldb = @max(1, args.n);
            if (policy.isQuery(i32, family(T), lengths(T, args))) return columnWork(T, operation, converted, &.{}, &.{});
            const ld: usize = @intCast(converted.lda);
            const count = try N.Storage.byteLength(ld, ld);
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
            @memset(row_a.lease.values, std.mem.zeroes(T));
            @memset(row_b.lease.values, std.mem.zeroes(T));
            converted.a = operand(T, row_a.lease.values);
            converted.b = operand(T, row_b.lease.values);
            const size: usize = @intCast(@max(0, args.n));
            const original_a: []T = if (size == 0) &.{} else try capacity(T, args.a, try N.Storage.columnMajorExtent(size, size, @intCast(args.lda)));
            const original_b: []T = if (size == 0) &.{} else try capacity(T, args.b, try N.Storage.columnMajorExtent(size, size, @intCast(args.ldb)));
            try disjoint(T, original_a, T, original_b);
            const lda: usize = @intCast(@max(0, args.lda));
            const ldb: usize = @intCast(@max(0, args.ldb));
            // GVD's LAPACKE row wrapper uses a full GE transpose for both
            // matrices, including the unselected triangle and failure outputs.
            for (0..size) |j| for (0..size) |i| {
                row_a.lease.values[i + j * ld] = original_a[i * lda + j];
                row_b.lease.values[i + j * ld] = original_b[i * ldb + j];
            };
            const info = try columnWork(T, operation, converted, original_a, original_b);
            for (0..size) |j| for (0..size) |i| {
                original_a[i * lda + j] = row_a.lease.values[i + j * ld];
                original_b[i * ldb + j] = row_b.lease.values[i + j * ld];
            };
            return info;
        }
        /// Row conversion and its allocation precede the Fortran scalar checks
        /// on a nonquery call. Queries and invalid row LDA access no matrix.
        pub fn workAdmitted(comptime T: type, operation: *const context.Operation, args: Arguments(T)) Error!i32 {
            _ = operation.borrowedContext();
            if (args.layout != 101 and args.layout != 102) return -1;
            if (args.layout == 102) return columnWork(T, operation, args, &.{}, &.{});
            if (comptime generalized) return generalizedRowWork(T, operation, args);
            if (args.lda < args.n) return -6;
            var converted = args;
            converted.layout = 102;
            converted.lda = @max(1, args.n);
            if (policy.isQuery(i32, family(T), lengths(T, args))) return columnWork(T, operation, converted, &.{}, &.{});
            const ld: usize = @intCast(converted.lda);
            var row = context.ContextScratch(T).acquire(operation, try N.Storage.byteLength(ld, ld)) catch |err| switch (err) {
                error.OutOfMemory => return -1011,
                else => return err,
            };
            defer row.deinit();
            @memset(row.lease.values, std.mem.zeroes(T));
            converted.a = operand(T, row.lease.values);
            const size: usize = @intCast(@max(0, args.n));
            const selected = triangle(args.uplo);
            var original: []T = &.{};
            if (size != 0 and selected != null) {
                original = try capacity(T, args.a, try N.Storage.columnMajorExtent(size, size, @intCast(args.lda)));
                const source_ld: usize = @intCast(args.lda);
                for (0..size) |j| for (0..size) |i| {
                    if (if (selected.?) i <= j else i >= j) row.lease.values[i + j * ld] = original[i * source_ld + j];
                };
            }
            const info = try columnWork(T, operation, converted, original, &.{});
            if (info != -1010 and size != 0 and (vectors(args.jobz) or selected != null)) {
                if (original.len == 0) original = try capacity(T, args.a, try N.Storage.columnMajorExtent(size, size, @intCast(args.lda)));
                const output_ld: usize = @intCast(args.lda);
                for (0..size) |j| for (0..size) |i| {
                    if (vectors(args.jobz) or (if (selected.?) i <= j else i >= j)) original[i * output_ld + j] = row.lease.values[i + j * ld];
                };
            }
            return info;
        }
        pub fn workWithDiagnosticsAdmitted(comptime T: type, operation: *const context.Operation, borrowed: anytype, args: Arguments(T)) Error!i32 {
            const info = try workAdmitted(T, operation, args);
            notify(T, callbacks(borrowed), args, info);
            return info;
        }
        fn highCompute(comptime T: type, operation: *const context.Operation, services: anytype, args: Arguments(T)) Error!i32 {
            const R = S.Real(T);
            var work_query: [1]T = undefined;
            var real_query: [1]R = undefined;
            var integer_query: [1]i32 = undefined;
            var call = args;
            call.work = operand(T, &work_query);
            call.rwork = operand(R, &real_query);
            call.iwork = operand(i32, &integer_query);
            call.lwork = -1;
            call.lrwork = -1;
            call.liwork = -1;
            const queried = try workWithDiagnosticsAdmitted(T, operation, services, call);
            if (queried != 0) return queried;
            call.lwork = @intFromFloat(if (comptime S.isComplex(T)) work_query[0].re else work_query[0]);
            call.liwork = integer_query[0];
            call.lrwork = if (comptime S.isComplex(T)) @intFromFloat(real_query[0]) else 0;
            var integers = try context.ContextScratch(i32).acquire(operation, @intCast(call.liwork));
            defer integers.deinit();
            var real: ?context.ContextScratch(R) = null;
            defer if (real) |*lease| lease.deinit();
            if (comptime S.isComplex(T)) real = try context.ContextScratch(R).acquire(operation, @intCast(call.lrwork));
            var work = try context.ContextScratch(T).acquire(operation, @intCast(call.lwork));
            defer work.deinit();
            call.iwork = operand(i32, integers.lease.values);
            call.rwork = if (comptime S.isComplex(T)) operand(R, real.?.lease.values) else null;
            call.work = operand(T, work.lease.values);
            const info = try workAdmitted(T, operation, call);
            if (info != -1010) notify(T, services, call, info);
            return info;
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
                        pub const Storage = N.Storage;
                        pub const Scalar = S;
                    };
                    const helpers = common.HelpersWithAccess(Provider, kind);
                    if (try helpers.scanTriangle(T, i32, .{ .layout = args.layout, .uplo = args.uplo, .n = args.n, .lda = args.lda, .values = @as(common.access.Operand(T, false, kind), args.a) })) return if (generalized) -6 else -5;
                    if (comptime generalized) {
                        if (try helpers.scanTriangle(T, i32, .{ .layout = args.layout, .uplo = args.uplo, .n = args.n, .lda = args.ldb, .values = @as(common.access.Operand(T, false, kind), args.b) })) return -8;
                    }
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
