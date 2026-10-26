// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Internal typed/raw high/work Hermitian adapter candidate, not installed C ABI.
//! Uses our explicit Hermitian IEEE numerical contract; production profile,
//! vendor nonfinite trajectory equivalence and distributed acceptance are separate.
const std = @import("std");
const common = @import("lapacke_common.zig");
pub const Routine = enum { heev, hegv };
pub fn Adapter(comptime Core: type) type {
    return Engine(Core, .typed);
}
pub fn RawAdapter(comptime Core: type) type {
    return Engine(Core, .raw);
}
fn Engine(comptime Core: type, comptime access_kind: common.access.Kind) type {
    return struct {
        const n = Core.internal.numerics;
        const execution = n.Execution;
        const context = execution.context;
        const H = n.Hermitian;
        const M = H.Managed(execution);
        pub const Error = M.Error || common.access.Fault(access_kind) || error{NumericalOutcomePendingABI};
        pub fn Arguments(comptime C: type) type {
            return struct {
                layout: c_int,
                itype: i32 = 1,
                jobz: u8,
                uplo: u8,
                n: i32,
                a: common.access.Operand(C, true, access_kind) = null,
                lda: i32,
                b: common.access.Operand(C, true, access_kind) = null,
                ldb: i32 = 1,
                w: common.access.Operand(n.Scalar.Real(C), true, access_kind) = null,
                work: common.access.Operand(C, true, access_kind) = null,
                lwork: i32 = 0,
                rwork: common.access.Operand(n.Scalar.Real(C), true, access_kind) = null,
            };
        }
        fn capacity(comptime T: type, values: common.access.Operand(T, true, access_kind), count: usize) Error![]T {
            return common.access.capacity(n.Storage, access_kind, T, true, values, count);
        }
        fn operand(comptime T: type, values: []T) common.access.Operand(T, true, access_kind) {
            return if (access_kind == .typed) values else @ptrCast(values.ptr);
        }
        fn event(comptime routine: Routine, comptime C: type, services: anytype, kind: common.HookKind, entry: common.Entry, argument: i32) void {
            services.error_event(services.context, .{ .kind = kind, .entry = entry, .scalar = if (C == n.Scalar.ComplexF32) 'c' else 'z', .argument = argument, .routine = if (routine == .heev) .heev else .hegv });
        }
        fn callbacks(borrowed: anytype) @TypeOf(.{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event }) {
            // Capture callbacks before invoking user code; never inspect allocator.
            return .{ .context = borrowed.context, .nan_enabled = borrowed.nan_enabled, .error_event = borrowed.error_event };
        }
        fn notifyInfo(comptime routine: Routine, comptime C: type, services: anytype, args: Arguments(C), info: i32) void {
            if (info >= 0) return;
            const row_ld = args.layout == 101 and (args.lda < args.n or (routine == .hegv and args.ldb < args.n));
            if (info == -1 or row_ld or info == -1010 or info == -1011) event(routine, C, services, .lapacke, .work, info) else event(routine, C, services, .fortran_returning_test, if (routine == .heev) .fortran_heev else .fortran_hegv, -info - 1);
        }
        /// Raw/typed caller storage is admitted lazily at the original access stage.
        pub fn workWithDiagnosticsAdmitted(comptime routine: Routine, comptime C: type, operation: *const context.Operation, borrowed: anytype, args: Arguments(C)) Error!i32 {
            _ = operation.borrowedContext();
            const services = callbacks(borrowed);
            const info = workAdmitted(routine, C, operation, args) catch |err| switch (err) {
                error.OutOfMemory => if (args.layout == 101) @as(i32, -1011) else @as(i32, -1010),
                else => return err,
            };
            notifyInfo(routine, C, services, args, info);
            return info;
        }
        fn highCompute(comptime routine: Routine, comptime C: type, operation: *const context.Operation, services: anytype, args: Arguments(C)) Error!i32 {
            const R = n.Scalar.Real(C);
            const rcount: usize = @intCast(@max(@as(i64, 1), @as(i64, args.n) * 3 - 2));
            var rlease = try context.ContextScratch(R).acquire(operation, rcount);
            defer rlease.deinit();
            var query: [1]C = undefined;
            var call = args;
            call.work = operand(C, &query);
            call.rwork = operand(R, rlease.lease.values);
            call.lwork = -1;
            const info = try workAdmitted(routine, C, operation, call);
            notifyInfo(routine, C, services, call, info);
            if (info != 0) return info;
            const optimal = query[0].re;
            if (!std.math.isFinite(optimal) or @as(f64, optimal) > std.math.maxInt(i32)) return error.IntegerNotRepresentable;
            call.lwork = @intFromFloat(optimal);
            var lease = try context.ContextScratch(C).acquire(operation, @intCast(call.lwork));
            defer lease.deinit();
            call.work = operand(C, lease.lease.values);
            const result = try workAdmitted(routine, C, operation, call);
            notifyInfo(routine, C, services, call, result);
            return result;
        }
        /// Reference high-level order: layout, optional A/B NaN scans, real
        /// workspace allocation, _work query, complex allocation and compute.
        /// One operation owns all scratch; callbacks run after allocation cleanup.
        pub fn highAdmitted(comptime compile_nan_scan: bool, comptime routine: Routine, comptime C: type, operation: *const context.Operation, borrowed: anytype, args: Arguments(C)) Error!i32 {
            _ = operation.borrowedContext();
            const services = callbacks(borrowed);
            if (args.layout != 101 and args.layout != 102) {
                event(routine, C, services, .lapacke, .high, -1);
                return -1;
            }
            if (comptime compile_nan_scan) {
                if (services.nan_enabled(services.context)) {
                    const Provider = struct {
                        pub const Services = @TypeOf(services);
                        pub const Storage = n.Storage;
                        pub const Scalar = n.Scalar;
                    };
                    const helpers = common.HelpersWithAccess(Provider, access_kind);
                    if (try helpers.scanTriangle(C, i32, .{ .values = args.a, .layout = args.layout, .uplo = args.uplo, .n = args.n, .lda = args.lda })) return if (routine == .heev) -5 else -6;
                    if (routine == .hegv and try helpers.scanTriangle(C, i32, .{ .values = args.b, .layout = args.layout, .uplo = args.uplo, .n = args.n, .lda = args.ldb })) return -8;
                }
            }
            return highCompute(routine, C, operation, services, args) catch |err| switch (err) {
                error.OutOfMemory => {
                    event(routine, C, services, .lapacke, .high, -1010);
                    return -1010;
                },
                else => return err,
            };
        }
        fn disjoint(comptime A: type, a: []const A, comptime B: type, b: []const B) Error!void {
            const ar = try n.Storage.byteRange(@intFromPtr(a.ptr), try n.Storage.byteLength(a.len, @sizeOf(A)));
            const br = try n.Storage.byteRange(@intFromPtr(b.ptr), try n.Storage.byteLength(b.len, @sizeOf(B)));
            if (n.Storage.overlaps(ar, br)) return error.AliasingNotAllowed;
        }
        fn admitAliases(comptime C: type, a: []C, b: []C, values: []n.Scalar.Real(C), work: []C, rwork: []n.Scalar.Real(C)) Error!void {
            try disjoint(C, a, C, work);
            try disjoint(C, a, n.Scalar.Real(C), rwork);
            try disjoint(C, a, n.Scalar.Real(C), values);
            try disjoint(C, work, n.Scalar.Real(C), rwork);
            try disjoint(C, work, n.Scalar.Real(C), values);
            try disjoint(n.Scalar.Real(C), rwork, n.Scalar.Real(C), values);
            try disjoint(C, b, n.Scalar.Real(C), rwork);
            try disjoint(C, a, C, b);
            try disjoint(C, b, C, work);
            try disjoint(C, b, n.Scalar.Real(C), values);
        }
        fn matrixCount(size: usize, ld: i32) Error!usize {
            return n.Storage.columnMajorExtent(size, size, @intCast(ld));
        }
        fn index(layout: c_int, ld: usize, i: usize, j: usize) usize {
            return if (layout == 102) i + j * ld else i * ld + j;
        }
        fn copyIn(comptime C: type, source: []const C, dest: []C, size: usize, ld: usize, layout: c_int, upper: bool) void {
            @memset(dest, .{ .re = 0, .im = 0 });
            for (0..size) |j| for (0..size) |i| {
                if (if (upper) i <= j else i >= j) dest[i + j * size] = source[index(layout, ld, i, j)];
            };
        }
        fn copyOut(comptime C: type, source: []const C, dest: []C, size: usize, ld: usize, layout: c_int, upper: bool, whole: bool) void {
            for (0..size) |j| for (0..size) |i| {
                if (whole or (if (upper) i <= j else i >= j)) dest[index(layout, ld, i, j)] = source[i + j * size];
            };
        }
        pub fn minimumWorkspace(size: i32) Error!i32 {
            if (size <= 0) return 1;
            return std.math.cast(i32, @as(i64, size) * 2 - 1) orelse error.IntegerNotRepresentable;
        }
        fn queryValue(comptime C: type, count: i32) C {
            const R = n.Scalar.Real(C);
            var value: R = @floatFromInt(count);
            if (@as(f64, value) < @as(f64, @floatFromInt(count))) value = std.math.nextAfter(R, value, std.math.inf(R));
            return .{ .re = value, .im = 0 };
        }
        /// Borrows one already admitted actual-Core operation. Queries allocate
        /// nothing and never read A/B/W/RWORK. No new admission or owner exists.
        /// INFO is the LAPACKE positional result; typed access faults stay errors.
        pub fn workAdmitted(comptime routine: Routine, comptime C: type, operation: *const context.Operation, args: Arguments(C)) Error!i32 {
            _ = operation.borrowedContext();
            if (args.layout != 101 and args.layout != 102) return -1;
            const generalized = routine == .hegv;
            const shift: i32 = if (generalized) 1 else 0;
            // Row-major wrapper precedence precedes Fortran parameter validation.
            if (args.layout == 101 and args.lda < args.n) return -6 - shift;
            if (generalized and args.layout == 101 and args.ldb < args.n) return -9;
            if (generalized and (args.itype < 1 or args.itype > 3)) return -2;
            const vectors = args.jobz == 'V' or args.jobz == 'v';
            if (!vectors and args.jobz != 'N' and args.jobz != 'n') return -2 - shift;
            const upper = args.uplo == 'U' or args.uplo == 'u';
            if (!upper and args.uplo != 'L' and args.uplo != 'l') return -3 - shift;
            if (args.n < 0) return -4 - shift;
            if (args.layout == 102 and args.lda < @max(1, args.n)) return -6 - shift;
            if (generalized and args.layout == 102 and args.ldb < @max(1, args.n)) return -9;
            const minimum = try minimumWorkspace(args.n);
            const work_head = try capacity(C, args.work, 1);
            work_head[0] = queryValue(C, minimum);
            if (args.lwork != -1 and args.lwork < minimum) return if (generalized) -12 else -9;
            if (args.lwork == -1 or args.n == 0) return 0;
            const size: usize = @intCast(args.n);
            const work = try capacity(C, args.work, @intCast(args.lwork));
            var a: []C = &.{};
            var values: []n.Scalar.Real(C) = &.{};
            var rwork: []n.Scalar.Real(C) = &.{};
            var b: []C = &.{};
            if (!generalized) {
                a = try capacity(C, args.a, try matrixCount(size, args.lda));
                values = try capacity(n.Scalar.Real(C), args.w, size);
                if (size == 1) {
                    try disjoint(C, a, C, work);
                    try disjoint(C, a, n.Scalar.Real(C), values);
                    try disjoint(C, work, n.Scalar.Real(C), values);
                    values[0] = a[0].re;
                    if (vectors) a[0] = .{ .re = 1, .im = 0 };
                    work_head[0] = .{ .re = 1, .im = 0 };
                    return 0;
                }
                rwork = try capacity(n.Scalar.Real(C), args.rwork, try n.Storage.byteLength(size, 3) - 2);
                try admitAliases(C, a, b, values, work, rwork);
            } else {
                // Row-major conversion accesses A before the Fortran driver;
                // column-major POTRF does not access A/W/RWORK on B failure.
                if (args.layout == 101) a = try capacity(C, args.a, try matrixCount(size, args.lda));
                b = try capacity(C, args.b, try matrixCount(size, args.ldb));
                try disjoint(C, b, C, work);
            }
            const square = try n.Storage.byteLength(size, size);
            const count = try n.Storage.byteLength(square, if (generalized) 3 else 2);
            var stage = try context.ContextScratch(C).acquire(operation, count);
            defer stage.deinit();
            const av = stage.lease.values[0..square];
            const bv = if (generalized) stage.lease.values[square .. 2 * square] else &.{};
            const q = stage.lease.values[count - square .. count];
            if (!generalized or args.layout == 101) copyIn(C, a, av, size, @intCast(args.lda), args.layout, upper);
            if (generalized) {
                copyIn(C, b, bv, size, @intCast(args.ldb), args.layout, upper);
                var factor = try context.ContextScratch(C).acquire(operation, square);
                defer factor.deinit();
                @memcpy(factor.lease.values, bv);
                const pivot = n.ComplexCholesky.factorizeWithContract(C, .{ .values = factor.lease.values, .row_count = size, .column_count = size, .leading_dimension = size }, if (upper) .upper else .lower, .lapack_potrf);
                if (pivot != 0) {
                    copyOut(C, factor.lease.values, b, size, @intCast(args.ldb), args.layout, upper, false);
                    return args.n + @as(i32, @intCast(pivot));
                }
                if (args.layout == 102) a = try capacity(C, args.a, try matrixCount(size, args.lda));
                values = try capacity(n.Scalar.Real(C), args.w, size);
                // Scalar HEEV never uses the real workspace, including HEGV's
                // nested scalar solve. Neither null nor misaligned RWORK is read.
                rwork = try capacity(n.Scalar.Real(C), args.rwork, if (size == 1) 0 else try n.Storage.byteLength(size, 3) - 2);
                try admitAliases(C, a, b, values, work, rwork);
                if (args.layout == 102) copyIn(C, a, av, size, @intCast(args.lda), args.layout, upper);
                copyOut(C, factor.lease.values, b, size, @intCast(args.ldb), args.layout, upper, false);
            }
            var request: H.Request(C) = .{
                .a = .{ .values = av, .row_count = size, .column_count = size, .leading_dimension = size },
                .triangle = if (upper) .upper else .lower,
                .eigenvalues = values,
                .contract = .heev_ieee_v1,
                .vectors = if (vectors) .{ .values = q, .row_count = size, .column_count = size, .leading_dimension = size } else null,
            };
            if (generalized) {
                request.b = .{ .values = bv, .row_count = size, .column_count = size, .leading_dimension = size };
                request.problem = switch (args.itype) {
                    1 => .ax_lambda_bx,
                    2 => .abx_lambda_x,
                    3 => .bax_lambda_x,
                    else => unreachable,
                };
            }
            const result = try M.runScratchAdmitted(C, operation, request);
            if (result.status == .not_positive_definite) return args.n + @as(i32, @intCast(result.failing_pivot));
            if (result.status == .range_limit) return error.NumericalOutcomePendingABI;
            if (vectors) copyOut(C, q, a, size, @intCast(args.lda), args.layout, upper, true);
            // JOBZ=N permits destruction but preserving A is also unspecified output.
            if (args.n == 1 and !generalized) work_head[0] = .{ .re = 1, .im = 0 };
            return @intCast(result.unconverged);
        }
    };
}
