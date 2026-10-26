// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shared private LAPACKE interface helpers, never numerical kernels or owners.
const std = @import("std");

/// A WORK(1) length must not round below its integer requirement.
pub fn workspaceQueryValue(comptime T: type, count: usize) T {
    var value: T = @floatFromInt(count);
    if (@as(f128, value) < @as(f128, @floatFromInt(count)))
        value = std.math.nextAfter(T, value, std.math.inf(T));
    return value;
}
pub const access = @import("lapacke_access.zig");
pub const Level = enum { high, work };
pub const HookKind = enum { lapacke, fortran_returning_test };
pub const Entry = enum { high, work, fortran_potrf, fortran_potrs, fortran_getrf, fortran_getrs, fortran_gesv, fortran_stev, fortran_gelsd, fortran_heev, fortran_hegv, fortran_geqrf, fortran_ormqr, fortran_unmqr, fortran_orgqr, fortran_ungqr, fortran_lalsd, fortran_gesvd, fortran_posv, fortran_trtrs, fortran_trtri, fortran_getri, fortran_potri, fortran_syev, fortran_sygv, fortran_syevd, fortran_heevd, fortran_sygvd, fortran_hegvd, fortran_gels, fortran_gelss };
pub const Routine = enum { potrf, potrs, getrf, getrs, gesv, stev, gelsd, heev, hegv, geqrf, ormqr, unmqr, orgqr, ungqr, lalsd, gesvd, posv, trtrs, trtri, getri, potri, syev, sygv, syevd, heevd, sygvd, hegvd, gels, gelss };
pub const ErrorEvent = struct { kind: HookKind, entry: Entry, scalar: u8, argument: i32, routine: Routine = .potrf };
pub const AdmissionFault = access.TypedFault;
pub fn Helpers(comptime Core: type) type {
    return HelpersWithAccess(Core, .typed);
}
pub fn HelpersWithAccess(comptime Core: type, comptime access_kind: access.Kind) type {
    return struct {
        const Fault = access.Fault(access_kind);
        const Services = Core.Services;
        pub fn integer(comptime I: type) void {
            if (I != i32 and I != i64) @compileError("private POTRF input integers are i32 or i64");
        }

        pub fn scalar(comptime T: type) void {
            if (T != f32 and T != f64 and T != Core.Scalar.ComplexF32 and T != Core.Scalar.ComplexF64)
                @compileError("private POTRF supports the owner's f32/f64 and complex pair types");
        }

        pub fn prefix(comptime T: type) u8 {
            return if (T == f32) 's' else if (T == f64) 'd' else if (T == Core.Scalar.ComplexF32) 'c' else 'z';
        }

        pub fn triangle(uplo: u8) ?bool {
            return switch (uplo) {
                'U', 'u' => true,
                'L', 'l' => false,
                else => null,
            };
        }

        pub fn hook(comptime T: type, services: Services, kind: HookKind, entry: Entry, argument: i32, routine: Routine) void {
            services.error_event(services.context, .{ .kind = kind, .entry = entry, .scalar = prefix(T), .argument = argument, .routine = routine });
        }

        pub fn native(value: anytype) Fault!usize {
            return std.math.cast(usize, value) orelse error.IntegerNotRepresentable;
        }

        pub fn capacity(comptime T: type, values: access.Operand(T, true, access_kind), count: usize) Fault![]T {
            return access.capacity(Core.Storage, access_kind, T, true, values, count);
        }
        pub fn constCapacity(comptime T: type, values: access.Operand(T, false, access_kind), count: usize) Fault![]const T {
            return access.capacity(Core.Storage, access_kind, T, false, values, count);
        }

        pub fn nan(comptime T: type, value: T) bool {
            if (comptime Core.Scalar.isComplex(T)) return std.math.isNan(value.re) or std.math.isNan(value.im);
            return std.math.isNan(value);
        }

        pub fn scanTriangle(comptime T: type, comptime I: type, args: anytype) Fault!bool {
            _ = I;
            // Preserve Reference helper nonreads for null/invalid uplo and no
            // visited elements. Private admission only guards actual scan spans.
            if (args.values == null) return false;
            const upper = triangle(args.uplo) orelse return false;
            if (args.n <= 0 or args.lda <= 0) return false;
            const n = try native(args.n);
            const ld = try native(args.lda);
            const limit = @min(n, ld);
            const leading_part = (args.layout == 102) == upper;
            const columns = if (leading_part) n else limit;
            const extent = try Core.Storage.columnMajorExtent(limit, columns, ld);
            const v = try constCapacity(T, args.values, extent);
            for (0..columns) |j| {
                const start = if (leading_part) 0 else j;
                const end = if (leading_part) @min(j + 1, ld) else limit;
                for (start..end) |i| if (nan(T, v[i + j * ld])) return true;
            }
            return false;
        }

        pub fn unitDiagonal(diag: u8) ?bool {
            return switch (diag) {
                'U', 'u' => true,
                'N', 'n' => false,
                else => null,
            };
        }
        /// Bounding span of selected elements; a unit singleton reads no A.
        /// Computation callers have already validated the leading dimension.
        pub fn triangularReadExtent(n: usize, ld: usize, layout: c_int, upper: bool, unit: bool) Fault!usize {
            if (n == 0 or (unit and n == 1)) return 0;
            const leading = (layout == 102) == upper;
            const rows = if (unit and leading) n - 1 else n;
            const columns = if (unit and !leading) n - 1 else n;
            return Core.Storage.columnMajorExtent(rows, columns, ld);
        }
        /// TR utility semantics: invalid triangle/diag and empty visited sets
        /// do not access A. Unit diagonal values are excluded, including NaNs.
        pub fn scanTriangular(comptime T: type, comptime I: type, args: anytype) Fault!bool {
            _ = I;
            if (args.values == null or (args.layout != 101 and args.layout != 102)) return false;
            const upper = triangle(args.uplo) orelse return false;
            const unit = unitDiagonal(args.diag) orelse return false;
            if (args.n <= 0 or args.lda <= 0) return false;
            const n = try native(args.n);
            const ld = try native(args.lda);
            const st: usize = if (unit) 1 else 0;
            const limit = @min(n, ld);
            const leading = (args.layout == 102) == upper;
            // A short positive lda still visits later columns in the leading
            // part. Only the opposite part uses limit to bound empty scans.
            if (leading) {
                if (n <= st) return false;
            } else if (limit <= st) return false;
            const rows = if (leading) @min(n - st, ld) else limit;
            const columns = if (leading) n else limit - st;
            const extent = try Core.Storage.columnMajorExtent(rows, columns, ld);
            const values = try constCapacity(T, args.values, extent);
            const first: usize = if (leading) st else 0;
            for (first..columns) |j| {
                const start = if (leading) 0 else j + st;
                const end = if (leading) @min(j + 1 - st, ld) else limit;
                for (start..end) |i| if (nan(T, values[i + j * ld])) return true;
            }
            return false;
        }
        pub fn copyTriangularToColumn(comptime T: type, original: []const T, temporary: []T, n: usize, ld: usize, upper: bool, unit: bool) void {
            for (0..n) |row| for (0..n) |column| {
                if (selected(row, column, upper) and !(unit and row == column)) temporary[row + column * n] = original[row * ld + column];
            };
        }

        pub fn selected(row: usize, column: usize, upper: bool) bool {
            return if (upper) row <= column else row >= column;
        }

        pub fn transfer(comptime T: type, original: anytype, temporary: []T, n: usize, ld: usize, upper: bool, comptime to_column: bool) void {
            // Coordinate-preserving layout copy, never conjugation. Only the
            // selected triangle is accessed; its diagonal bounds both spans.
            for (0..n) |row| for (0..n) |column| {
                if (selected(row, column, upper)) {
                    const r = row * ld + column;
                    const c = row + column * n;
                    if (to_column) temporary[c] = original[r] else original[r] = temporary[c];
                }
            };
        }

        pub fn scanGeneral(comptime T: type, layout: c_int, n_input: anytype, nrhs_input: @TypeOf(n_input), ld_input: @TypeOf(n_input), values: access.Operand(T, false, access_kind)) Fault!bool {
            if (values == null or n_input <= 0 or nrhs_input <= 0 or ld_input <= 0) return false;
            const n = try native(n_input);
            const nrhs = try native(nrhs_input);
            const ld = try native(ld_input);
            const rows = @min(if (layout == 102) n else nrhs, ld);
            const columns = if (layout == 102) nrhs else n;
            const extent = try Core.Storage.columnMajorExtent(rows, columns, ld);
            const v = try constCapacity(T, values, extent);
            for (0..columns) |j| for (0..rows) |i| {
                if (nan(T, v[i + j * ld])) return true;
            };
            return false;
        }
        pub fn transferGeneral(comptime T: type, original: []T, temporary: []T, n: usize, nrhs: usize, ld: usize, comptime to_column: bool) void {
            // Match Reference general-copy loop order, with admitted full extents.
            const outer = if (to_column) nrhs else n;
            const inner = if (to_column) n else nrhs;
            for (0..outer) |i| for (0..inner) |j| {
                if (to_column) temporary[i * n + j] = original[j * ld + i] else original[i * ld + j] = temporary[j * n + i];
            };
        }
        /// Readonly GETRS factor copy, with the same coordinate/loop order as
        /// transferGeneral(..., true); no const cast or numerical transform.
        pub fn copyGeneralToColumn(comptime T: type, original: []const T, temporary: []T, n: usize, columns: usize, ld: usize) void {
            for (0..columns) |column| for (0..n) |row| {
                temporary[column * n + row] = original[row * ld + column];
            };
        }
        /// Mixed operand widths retain byte geometry; never reinterpret pivot
        /// storage as a numerical scalar. Only already-admitted used prefixes.
        pub fn disjointMixed(comptime A: type, a: []const A, comptime B: type, b: []const B) Fault!void {
            const ar = try Core.Storage.byteRange(@intFromPtr(a.ptr), try Core.Storage.byteLength(a.len, @sizeOf(A)));
            const br = try Core.Storage.byteRange(@intFromPtr(b.ptr), try Core.Storage.byteLength(b.len, @sizeOf(B)));
            if (Core.Storage.overlaps(ar, br)) return error.AliasingNotAllowed;
        }
        pub fn disjoint(comptime T: type, a: []const T, b: []T) Fault!void {
            const ar = try Core.Storage.byteRange(@intFromPtr(a.ptr), try Core.Storage.byteLength(a.len, @sizeOf(T)));
            const br = try Core.Storage.byteRange(@intFromPtr(b.ptr), try Core.Storage.byteLength(b.len, @sizeOf(T)));
            if (Core.Storage.overlaps(ar, br)) return error.AliasingNotAllowed;
        }
    };
}

/// Closed ownership selection; this only binds existing leases to one engine.
pub const Lifetime = enum { unmanaged, context_managed };
pub fn EngineResult(comptime lifetime: Lifetime, comptime R: type) type {
    return if (lifetime == .context_managed) error{CounterOverflow}!R else R;
}
/// An affine invocation: capture once, then end only after the engine's cleanup.
/// Managed allocator authority is exclusively the admitted Context. The input
/// Services allocator field is deliberately not read (its ptr may be undefined).
pub fn ManagedInvocation(comptime Core: type) type {
    return struct {
        operation: Core.Execution.context.Operation,
        services: Core.Services,
        pub fn begin(context: *Core.Execution.context.Context, borrowed: *const Core.Services) Core.Execution.context.AdmissionError!@This() {
            const operation = try context.begin();
            return .{ .operation = operation, .services = .{
                .allocator = operation.allocator(),
                .context = borrowed.context,
                .nan_enabled = borrowed.nan_enabled,
                .error_event = borrowed.error_event,
            } };
        }
    };
}
/// No lease copy, extra allocation, resource owner or numerical policy. values
/// borrows the active inner slice and cannot outlive the selected lease.
pub fn Scratch(comptime Core: type, comptime T: type, comptime lifetime: Lifetime) type {
    return struct {
        pub const Lease = if (lifetime == .context_managed) Core.Execution.context.ContextScratch(T) else Core.ScratchLease(T);
        pub const Source = if (lifetime == .context_managed) *const Core.Execution.context.Operation else std.mem.Allocator;
        pub const AcquireError = if (lifetime == .context_managed) error{ OutOfMemory, CounterOverflow } else error{OutOfMemory};
        pub fn acquire(source: Source, count: usize) AcquireError!Lease {
            return Lease.acquire(source, count);
        }
        pub fn values(lease: *const Lease) []T {
            return if (lifetime == .context_managed) lease.lease.values else lease.values;
        }
    };
}
