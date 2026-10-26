// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Typed FEAST4 LP64 CSR Hermitian driver seam. External FPM/INFO translation
//! lives here; the native contour algorithm, sparse kernels and all resources
//! come from one injected Core. No new executor, handle family or global state.
const std = @import("std");
pub const Parameters = @import("feast_parameters.zig");
pub fn Adapter(comptime Provider: type) type {
    const S = Provider.Scalar;
    const T = S.ComplexF64;
    const D = Provider.Dense;
    const Store = Provider.Storage;
    const C = Provider.Execution.context;
    const F = Provider.Contour;
    const Backend = Provider.SparseOperator.PatternBackend(T, i32, i32);
    return struct {
        pub const Arguments = struct {
            uplo: u8,
            dimension: i32,
            capacity: i32,
            matrix: []const T,
            rows: []const i32,
            columns: []const i32,
            fpm: []i32,
            lower: f64,
            upper: f64,
            eigenvalues: []f64,
            vectors: []T,
            residuals: []f64,
        };
        pub const Result = struct {
            info: i32,
            found: ?usize = null,
            loop: ?usize = null,
            epsout: ?f64 = null,
            native: ?F.Result(T) = null,
        };
        fn range(values: anytype) !Store.ByteRange {
            return Store.byteRange(@intFromPtr(values.ptr), try Store.byteLength(values.len, @sizeOf(@typeInfo(@TypeOf(values)).pointer.child)));
        }
        fn separate(a: anytype, b: anytype) !void {
            if (Store.overlaps(try range(a), try range(b))) return error.AliasingNotAllowed;
        }
        fn buffers(args: Arguments, n: usize, m: usize) !usize {
            const size = std.math.mul(usize, n, m) catch return error.SizeOverflow;
            _ = try Store.byteLength(size, @sizeOf(T));
            if (args.eigenvalues.len < m or args.vectors.len < size or args.residuals.len < m) return error.BufferTooSmall;
            const x = args.vectors[0..size];
            const e = args.eigenvalues[0..m];
            const r = args.residuals[0..m];
            try separate(x, e);
            try separate(x, r);
            try separate(e, r);
            try separate(args.matrix, x);
            try separate(args.matrix, e);
            try separate(args.matrix, r);
            try separate(args.rows, x);
            try separate(args.rows, e);
            try separate(args.rows, r);
            try separate(args.columns, x);
            try separate(args.columns, e);
            try separate(args.columns, r);
            try separate(args.fpm, x);
            try separate(args.fpm, e);
            try separate(args.fpm, r);
            try separate(args.fpm, args.matrix);
            try separate(args.fpm, args.rows);
            try separate(args.fpm, args.columns);
            return size;
        }
        pub fn solveAdmitted(op: *const C.Operation, args: Arguments) !Result {
            _ = op.borrowedContext();
            const uplo = std.ascii.toUpper(args.uplo);
            if (uplo != 'F' and uplo != 'L' and uplo != 'U') return .{ .info = -101 };
            if (args.dimension <= 0) return .{ .info = -102 };
            const n: usize = @intCast(args.dimension);
            if (args.capacity <= 1 or args.capacity > args.dimension) return .{ .info = 201 };
            const m: usize = @intCast(args.capacity);
            if (!std.math.isFinite(args.lower) or !std.math.isFinite(args.upper) or !(args.lower < args.upper)) return .{ .info = 200 };
            const controls = switch (Parameters.decode(args.fpm)) {
                .invalid => |code| return .{ .info = code },
                .valid => |v| v,
            };
            const size = try buffers(args, n, m);
            // Full Hermitian storage has the same mathematical lower triangle.
            // Opposite-half entries and diagonal imaginary parts are not used.
            var backend = Backend{ .operand = .{ .pattern = .{ .ranges = .{ .contiguous = args.rows }, .ordinals = args.columns, .row_count = n, .column_count = n, .index_base = .one }, .values = args.matrix }, .semantics = .{ .hermitian = .{ .triangle = if (uplo == 'U') .upper else .lower, .diagonal = .stored } } };
            _ = backend.operand.check() catch |err| return .{ .info = if (err == error.InvalidOrdinal) -105 else -104 };
            for (0..n) |row| {
                const entries = try backend.operand.pattern.checkedMajorRange(row);
                for (entries.begin..entries.end) |entry| {
                    const column = try backend.operand.pattern.checkedOrdinalAt(entry);
                    if ((uplo == 'U' and row > column) or (uplo != 'U' and row < column)) continue;
                    const value = args.matrix[entry];
                    if (!std.math.isFinite(value.re) or (row != column and !std.math.isFinite(value.im))) return .{ .info = -103 };
                }
            }
            var initial = try C.ContextScratch(T).acquire(op, if (controls.reuse) 0 else size);
            defer initial.deinit();
            var vectors = try C.ContextScratch(T).acquire(op, size);
            defer vectors.deinit();
            var values = try C.ContextScratch(f64).acquire(op, try std.math.mul(usize, m, 2));
            defer values.deinit();
            if (!controls.reuse) {
                var prng = std.Random.DefaultPrng.init(142242);
                const random = prng.random();
                for (initial.lease.values) |*value| value.* = .{ .re = random.float(f64) - 0.5, .im = random.float(f64) - 0.5 };
            }
            const scale = @max(@abs(args.lower), @abs(args.upper));
            const tolerance = std.math.pow(f64, 10, -@as(f64, @floatFromInt(controls.exponent)));
            const shape = D.PanelShape{ .row_count = n, .column_count = m, .leading_dimension = n, .layout = .column_major };
            const native = try F.Managed(Provider.Execution).runScratchAdmitted(T, op, .{ .action = backend.borrow(), .global_dimension = n, .initial = .{ .values = if (controls.reuse) args.vectors[0..size] else initial.lease.values, .shape = shape }, .vectors = .{ .values = vectors.lease.values, .shape = shape }, .eigenvalues = values.lease.values[0..m], .residuals = values.lease.values[m..], .contour = .{ .lower = args.lower, .upper = args.upper, .vertical_ratio = controls.ratio, .half_nodes = controls.nodes, .rule = if (controls.rule == .gauss) .gauss_legendre else .midpoint_trapezoidal }, .max_iterations = try std.math.add(usize, controls.refinements, 1), .relative_tolerance = 0, .absolute_tolerance = tolerance * scale, .linear_max_iterations = if (controls.inexact) controls.linear_iterations else @max(@as(usize, 64), try std.math.mul(usize, n, 4)), .linear_relative_tolerance = if (controls.inexact) controls.linear_tolerance else @min(@as(f64, 1e-12), tolerance * 0.01), .linear_accuracy = .fixed, .method = .residual_inverse, .allow_inexact_iteration_limit = controls.inexact, .output_selection = .all_ritz, .publish_finite_stops = true, .convergence = if (controls.trace) .trace else .residual, .trace_tolerance = tolerance });
            if (native.status == .linear_stop) return .{ .info = -2, .native = native };
            if (native.status == .ritz_stop) return .{ .info = -3, .native = native };
            // Translate all reported candidate residuals before publication.
            for (values.lease.values[m..][0..native.published]) |*residual| {
                residual.* = @floatCast(@as(f128, residual.*) / scale);
                if (!std.math.isFinite(residual.*)) return error.NonFiniteResult;
            }
            _ = std.math.cast(i32, native.linear_iterations) orelse return error.SizeOverflow;
            try Parameters.publish(args.fpm, controls, native.linear_iterations);
            @memcpy(args.eigenvalues[0..native.published], values.lease.values[0..native.published]);
            @memcpy(args.residuals[0..native.published], values.lease.values[m..][0..native.published]);
            @memcpy(args.vectors[0 .. n * native.published], vectors.lease.values[0 .. n * native.published]);
            // Filling the subspace is a capacity warning only when additional
            // independent vectors can exist outside it. M0=N spans the full space.
            return .{ .info = if (native.found == 0) 1 else if (native.found >= m and m < n) 3 else if (native.status == .iteration_limit) 2 else 0, .found = native.found, .loop = if (native.iterations == 0) 0 else native.iterations - 1, .epsout = native.trace_change, .native = native };
        }
        pub fn solve(owner: *C.Context, args: Arguments) !Result {
            var operation = try owner.begin();
            defer operation.end();
            return solveAdmitted(&operation, args);
        }
    };
}
