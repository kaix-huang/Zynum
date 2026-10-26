// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Generated finite native-arm64 interface subset; not a complete vendor profile.
const std = @import("std");
const builtin = @import("builtin");
const binding = @import("lapacke-interface-owner");
pub const WireInt = c_int;
const InternalInt = i32;
const C = binding.implementation.ComplexF32;
const Z = binding.implementation.ComplexF64;
const Level = binding.common.Level;

comptime {
    if (builtin.is_test or builtin.target.cpu.arch != .aarch64 or builtin.target.os.tag != .macos)
        @compileError("wire probe requires a non-test native Apple arm64 product");
    if (@sizeOf(c_int) != 4 or @sizeOf(WireInt) != 4) @compileError("integer representation");
    if (@typeInfo(WireInt).int.signedness != .signed or @typeInfo(InternalInt).int.signedness != .signed or
        std.math.minInt(WireInt) != std.math.minInt(InternalInt) or std.math.maxInt(WireInt) != std.math.maxInt(InternalInt))
        @compileError("wire/internal integer ranges must be identical");
    if (@typeInfo(c_int).int.signedness != @typeInfo(i32).int.signedness or
        std.math.minInt(c_int) != std.math.minInt(i32) or std.math.maxInt(c_int) != std.math.maxInt(i32))
        @compileError("layout c_int/i32 ranges must be identical");
    if (@sizeOf(C) != 8 or @alignOf(C) != 4 or @offsetOf(C, "re") != 0 or @offsetOf(C, "im") != 4) @compileError("complex8 representation");
    if (@sizeOf(Z) != 16 or @alignOf(Z) != 8 or @offsetOf(Z, "re") != 0 or @offsetOf(Z, "im") != 8) @compileError("complex16 representation");
}

fn terminal(reason: []const u8) noreturn {
    // This function is reached only after the managed call has fully returned:
    // its scratch defers and Operation.end have already completed.
    _ = reason; // No production hook ABI or private fault observer is introduced.
    std.c.abort();
}
fn finish(result: anytype) WireInt {
    return switch (result) {
        .lapacke_info => |info| @as(WireInt, @intCast(info)),
        .admission_fault => |err| terminal(@errorName(err)),
    };
}
const Diagnostic = binding.Diagnostics(InternalInt);
extern fn LAPACKE_xerbla(name: [*:0]const u8, info: WireInt) callconv(.c) void;
extern fn xerbla_(name: [*]const u8, info: *const WireInt) callconv(.c) void;
const DiagnosticHandlers = struct {
    pub fn lapacke(_: @This(), name: [:0]const u8, info: InternalInt) void {
        LAPACKE_xerbla(name.ptr, @intCast(info));
    }
    pub fn fortran(_: @This(), name: [:0]const u8, info: InternalInt) void {
        const wire_info: WireInt = @intCast(info);
        xerbla_(name.ptr, &wire_info);
    }
};
fn diagnosticEvent(_: *anyopaque, event: binding.common.ErrorEvent) void {
    Diagnostic.dispatch(event, DiagnosticHandlers{}) catch terminal("invalid internal diagnostic");
}
fn profileDefaults() binding.shared_services.Callbacks {
    const callback: binding.shared_services.Callback = .{
        .userdata = binding.Provider.Execution.getContext(),
        .function = diagnosticEvent,
    };
    return .{ .lapacke = callback, .fortran_returning_test = callback };
}

fn Factor(comptime T: type, comptime level: Level) type {
    return struct {
        pub fn call(layout: c_int, uplo: i8, n: WireInt, a: ?[*]align(1) T, lda: WireInt) callconv(.c) WireInt {
            const normalized_layout: i32 = @intCast(layout);
            const result = binding.Shared.factorRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                // The existing engine descriptor retains its c_int layout field.
                .layout = @as(c_int, @intCast(normalized_layout)),
                .uplo = @bitCast(uplo),
                .n = @as(InternalInt, @intCast(n)),
                .values = a,
                .lda = @as(InternalInt, @intCast(lda)),
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return finish(result);
        }
    };
}
fn Solve(comptime T: type, comptime level: Level) type {
    return struct {
        pub fn call(layout: c_int, uplo: i8, n: WireInt, nrhs: WireInt, a: ?[*]align(1) const T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt) callconv(.c) WireInt {
            const normalized_layout: i32 = @intCast(layout);
            const result = binding.Shared.solveRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                // The existing engine descriptor retains its c_int layout field.
                .layout = @as(c_int, @intCast(normalized_layout)),
                .uplo = @bitCast(uplo),
                .n = @as(InternalInt, @intCast(n)),
                .nrhs = @as(InternalInt, @intCast(nrhs)),
                .a = a,
                .lda = @as(InternalInt, @intCast(lda)),
                .b = b,
                .ldb = @as(InternalInt, @intCast(ldb)),
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return finish(result);
        }
    };
}

// Borrowed pivot elements have the same representation on this exact target.
// Pointers stay nullable/align(1); only the existing raw engine may form spans.
comptime {
    if (@sizeOf(WireInt) != @sizeOf(InternalInt) or @alignOf(WireInt) != @alignOf(InternalInt))
        @compileError("wire/internal pivot element representation");
}
fn LuFactor(comptime T: type, comptime level: Level) type {
    return struct {
        pub fn call(layout: c_int, m: WireInt, n: WireInt, a: ?[*]align(1) T, lda: WireInt, ipiv: ?[*]align(1) WireInt) callconv(.c) WireInt {
            const normalized_layout: i32 = @intCast(layout);
            const result = binding.Shared.getrfRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                .layout = @as(c_int, @intCast(normalized_layout)),
                .m = @as(InternalInt, @intCast(m)),
                .n = @as(InternalInt, @intCast(n)),
                .values = a,
                .lda = @as(InternalInt, @intCast(lda)),
                .pivots = if (ipiv) |pointer| @as([*]align(1) InternalInt, @ptrCast(pointer)) else null,
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return finish(result);
        }
    };
}
fn LuSolve(comptime T: type, comptime level: Level) type {
    return struct {
        pub fn call(layout: c_int, trans: i8, n: WireInt, nrhs: WireInt, a: ?[*]align(1) const T, lda: WireInt, ipiv: ?[*]align(1) const WireInt, b: ?[*]align(1) T, ldb: WireInt) callconv(.c) WireInt {
            const normalized_layout: i32 = @intCast(layout);
            const result = binding.Shared.getrsRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                .layout = @as(c_int, @intCast(normalized_layout)),
                .trans = @bitCast(trans),
                .n = @as(InternalInt, @intCast(n)),
                .nrhs = @as(InternalInt, @intCast(nrhs)),
                .a = a,
                .lda = @as(InternalInt, @intCast(lda)),
                .pivots = if (ipiv) |pointer| @as([*]align(1) const InternalInt, @ptrCast(pointer)) else null,
                .b = b,
                .ldb = @as(InternalInt, @intCast(ldb)),
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return finish(result);
        }
    };
}

fn LuDriver(comptime T: type, comptime level: Level) type {
    return struct {
        pub fn call(layout: c_int, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, ipiv: ?[*]align(1) WireInt, b: ?[*]align(1) T, ldb: WireInt) callconv(.c) WireInt {
            const normalized_layout: i32 = @intCast(layout);
            const result = binding.Shared.gesvRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                .layout = @as(c_int, @intCast(normalized_layout)),
                .n = @as(InternalInt, @intCast(n)),
                .nrhs = @as(InternalInt, @intCast(nrhs)),
                .a = a,
                .lda = @as(InternalInt, @intCast(lda)),
                .pivots = if (ipiv) |pointer| @as([*]align(1) InternalInt, @ptrCast(pointer)) else null,
                .b = b,
                .ldb = @as(InternalInt, @intCast(ldb)),
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return finish(result);
        }
    };
}
// Generated boundary only; the existing owner supplies mathematics and work.
fn TriangularInverse(comptime T: type, comptime level: Level) type {
    return struct {
        pub fn call(layout: c_int, uplo: i8, diag: i8, n: WireInt, a: ?[*]align(1) T, lda: WireInt) callconv(.c) WireInt {
            const result = binding.Shared.trtriRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                .layout = layout,
                .uplo = @bitCast(uplo),
                .diag = @bitCast(diag),
                .n = @intCast(n),
                .a = a,
                .lda = @intCast(lda),
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return finish(result);
        }
    };
}
fn TridiagonalEigen(comptime T: type, comptime level: Level) type {
    return struct {
        fn invoke(layout: c_int, jobz: i8, n: WireInt, d: ?[*]align(1) T, e: ?[*]align(1) T, z: ?[*]align(1) T, ldz: WireInt, work: ?[*]align(1) T) WireInt {
            const result = binding.Shared.stevRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                .layout = layout,
                .jobz = @bitCast(jobz),
                .n = @intCast(n),
                .d = d,
                .e = e,
                .z = z,
                .ldz = @intCast(ldz),
                .work = work,
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return finish(result);
        }
        fn high(layout: c_int, jobz: i8, n: WireInt, d: ?[*]align(1) T, e: ?[*]align(1) T, z: ?[*]align(1) T, ldz: WireInt) callconv(.c) WireInt {
            return invoke(layout, jobz, n, d, e, z, ldz, null);
        }
        fn workEntry(layout: c_int, jobz: i8, n: WireInt, d: ?[*]align(1) T, e: ?[*]align(1) T, z: ?[*]align(1) T, ldz: WireInt, work: ?[*]align(1) T) callconv(.c) WireInt {
            return invoke(layout, jobz, n, d, e, z, ldz, work);
        }
        pub const call = if (level == .high) high else workEntry;
    };
}
fn RankLeastSquares(comptime T: type, comptime level: Level) type {
    return struct {
        fn invoke(layout: c_int, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, singular: ?[*]align(1) T, rcond: T, rank: ?[*]align(1) WireInt, work: ?[*]align(1) T, lwork: WireInt, iwork: ?[*]align(1) WireInt) WireInt {
            const result = binding.Shared.gelsdRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                .layout = layout,
                .column = .{
                    .m = @intCast(m),
                    .n = @intCast(n),
                    .nrhs = @intCast(nrhs),
                    .a = a,
                    .lda = @intCast(lda),
                    .b = b,
                    .ldb = @intCast(ldb),
                    .singular = singular,
                    .rcond = rcond,
                    .rank = if (rank) |ptr| @as([*]align(1) InternalInt, @ptrCast(ptr)) else null,
                    .work = work,
                    .lwork = @intCast(lwork),
                    .iwork = if (iwork) |ptr| @as([*]align(1) InternalInt, @ptrCast(ptr)) else null,
                },
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            // All managed cleanup has completed before any terminal invariant fault.
            // Only actual unfinished bidiagonal edges are returned as positive INFO.
            return switch (result) {
                .lapacke_info => |info| @intCast(info),
                .admission_fault => |err| terminal(@errorName(err)),
                .numerical_fault => |err| terminal(@errorName(err)),
            };
        }
        fn high(layout: c_int, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, singular: ?[*]align(1) T, rcond: T, rank: ?[*]align(1) WireInt) callconv(.c) WireInt {
            return invoke(layout, m, n, nrhs, a, lda, b, ldb, singular, rcond, rank, null, 0, null);
        }
        fn workEntry(layout: c_int, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, singular: ?[*]align(1) T, rcond: T, rank: ?[*]align(1) WireInt, work: ?[*]align(1) T, lwork: WireInt, iwork: ?[*]align(1) WireInt) callconv(.c) WireInt {
            return invoke(layout, m, n, nrhs, a, lda, b, ldb, singular, rcond, rank, work, lwork, iwork);
        }
        pub const call = if (level == .high) high else workEntry;
    };
}
// Generated representation adapter; all numerical state belongs to the binding.
// Finite QR exhaustion returns unfinished edges; other numeric faults retain their own token.
fn ComplexRankLeastSquares(comptime T: type, comptime level: Level) type {
    if (InternalInt != i32) @compileError("complex GELSD generation acceptance is LP64 only");
    const R = binding.Provider.Scalar.Real(T);
    return struct {
        fn invoke(layout: c_int, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, singular: ?[*]align(1) R, rcond: R, rank: ?[*]align(1) WireInt, work: ?[*]align(1) T, lwork: WireInt, rwork: ?[*]align(1) R, iwork: ?[*]align(1) WireInt) WireInt {
            const result = binding.Shared.complexGelsdRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                .layout = layout,
                .column = .{
                    .m = @intCast(m),
                    .n = @intCast(n),
                    .nrhs = @intCast(nrhs),
                    .a = a,
                    .lda = @intCast(lda),
                    .b = b,
                    .ldb = @intCast(ldb),
                    .singular = singular,
                    .rcond = rcond,
                    .rank = if (rank) |ptr| @as([*]align(1) InternalInt, @ptrCast(ptr)) else null,
                    .work = work,
                    .lwork = @intCast(lwork),
                    .rwork = rwork,
                    .iwork = if (iwork) |ptr| @as([*]align(1) InternalInt, @ptrCast(ptr)) else null,
                },
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            // Managed leases and the single admitted operation have ended.
            return switch (result) {
                .lapacke_info => |info| @intCast(info),
                .admission_fault => |err| terminal(@errorName(err)),
                .numerical_fault => |err| if (binding.complex_gelsd_policy.standardNumericalInfo(err)) |info| @intCast(info) else terminal(@errorName(err)),
            };
        }
        fn high(layout: c_int, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, singular: ?[*]align(1) R, rcond: R, rank: ?[*]align(1) WireInt) callconv(.c) WireInt {
            return invoke(layout, m, n, nrhs, a, lda, b, ldb, singular, rcond, rank, null, 0, null, null);
        }
        fn workEntry(layout: c_int, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, singular: ?[*]align(1) R, rcond: R, rank: ?[*]align(1) WireInt, work: ?[*]align(1) T, lwork: WireInt, rwork: ?[*]align(1) R, iwork: ?[*]align(1) WireInt) callconv(.c) WireInt {
            return invoke(layout, m, n, nrhs, a, lda, b, ldb, singular, rcond, rank, work, lwork, rwork, iwork);
        }
        pub const call = if (level == .high) high else workEntry;
    };
}
// Generated representation only. One injected binding owns Core and resources.
// QR exhaustion returns upper bidiagonal factors and edge counts; separate numeric faults retain their token.
fn SingularValueDecomposition(comptime T: type, comptime level: Level) type {
    if (InternalInt != i32) @compileError("GESVD generation acceptance is LP64 only");
    const R = binding.Provider.Scalar.Real(T);
    return struct {
        fn invoke(layout: c_int, jobu: i8, jobvt: i8, m: WireInt, n: WireInt, a: ?[*]align(1) T, lda: WireInt, singular: ?[*]align(1) R, u: ?[*]align(1) T, ldu: WireInt, vt: ?[*]align(1) T, ldvt: WireInt, superb: ?[*]align(1) R, work: ?[*]align(1) T, lwork: WireInt, rwork: ?[*]align(1) R) WireInt {
            const result = binding.Shared.gesvdRawManagedWithDefaults(true, T, level, binding.Provider.Execution.getContext(), .{
                .layout = layout,
                .jobu = @bitCast(jobu),
                .jobvt = @bitCast(jobvt),
                .m = @intCast(m),
                .n = @intCast(n),
                .a = a,
                .lda = @intCast(lda),
                .singular = singular,
                .u = u,
                .ldu = @intCast(ldu),
                .vt = vt,
                .ldvt = @intCast(ldvt),
                .superb = superb,
                .work = work,
                .lwork = @intCast(lwork),
                .rwork = rwork,
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            // The borrowed scratch leases and sole operation have ended here.
            return switch (result) {
                .lapacke_info => |info| @intCast(info),
                .admission_fault => |err| terminal(@errorName(err)),
                .numerical_fault => |err| if (binding.gesvd_policy.numericalInfo(err)) |info| @intCast(info) else terminal(@errorName(err)),
            };
        }
        fn high(layout: c_int, jobu: i8, jobvt: i8, m: WireInt, n: WireInt, a: ?[*]align(1) T, lda: WireInt, singular: ?[*]align(1) R, u: ?[*]align(1) T, ldu: WireInt, vt: ?[*]align(1) T, ldvt: WireInt, superb: ?[*]align(1) R) callconv(.c) WireInt {
            return invoke(layout, jobu, jobvt, m, n, a, lda, singular, u, ldu, vt, ldvt, superb, null, 0, null);
        }
        fn realWork(layout: c_int, jobu: i8, jobvt: i8, m: WireInt, n: WireInt, a: ?[*]align(1) T, lda: WireInt, singular: ?[*]align(1) R, u: ?[*]align(1) T, ldu: WireInt, vt: ?[*]align(1) T, ldvt: WireInt, work: ?[*]align(1) T, lwork: WireInt) callconv(.c) WireInt {
            return invoke(layout, jobu, jobvt, m, n, a, lda, singular, u, ldu, vt, ldvt, null, work, lwork, null);
        }
        fn complexWork(layout: c_int, jobu: i8, jobvt: i8, m: WireInt, n: WireInt, a: ?[*]align(1) T, lda: WireInt, singular: ?[*]align(1) R, u: ?[*]align(1) T, ldu: WireInt, vt: ?[*]align(1) T, ldvt: WireInt, work: ?[*]align(1) T, lwork: WireInt, rwork: ?[*]align(1) R) callconv(.c) WireInt {
            return invoke(layout, jobu, jobvt, m, n, a, lda, singular, u, ldu, vt, ldvt, null, work, lwork, rwork);
        }
        pub const call = if (level == .high) high else if (binding.Provider.Scalar.isComplex(T)) complexWork else realWork;
    };
}
// Interface-only template. Numerical/data/resource policy comes from one binding.
fn HermitianReal(comptime T: type) type {
    return if (T == C) f32 else if (T == Z) f64 else @compileError("Hermitian complex scalar required");
}
fn HermitianStandard(comptime T: type, comptime level: Level) type {
    if (InternalInt != i32) @compileError("Hermitian interface acceptance is LP64 only");
    return struct {
        fn invoke(layout: c_int, jobz: i8, uplo: i8, n: WireInt, a: ?[*]align(1) T, lda: WireInt, w: ?[*]align(1) HermitianReal(T), work: ?[*]align(1) T, lwork: WireInt, rwork: ?[*]align(1) HermitianReal(T)) WireInt {
            const info = binding.Shared.hermitianRawManagedWithDefaults(true, T, .heev, level, binding.Provider.Execution.getContext(), .{
                .layout = layout,
                .jobz = @bitCast(jobz),
                .uplo = @bitCast(uplo),
                .n = @intCast(n),
                .a = a,
                .lda = @intCast(lda),
                .w = w,
                .work = work,
                .lwork = @intCast(lwork),
                .rwork = rwork,
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return @intCast(info);
        }
        fn high(layout: c_int, jobz: i8, uplo: i8, n: WireInt, a: ?[*]align(1) T, lda: WireInt, w: ?[*]align(1) HermitianReal(T)) callconv(.c) WireInt {
            return invoke(layout, jobz, uplo, n, a, lda, w, null, 0, null);
        }
        fn workEntry(layout: c_int, jobz: i8, uplo: i8, n: WireInt, a: ?[*]align(1) T, lda: WireInt, w: ?[*]align(1) HermitianReal(T), work: ?[*]align(1) T, lwork: WireInt, rwork: ?[*]align(1) HermitianReal(T)) callconv(.c) WireInt {
            return invoke(layout, jobz, uplo, n, a, lda, w, work, lwork, rwork);
        }
        pub const call = if (level == .high) high else workEntry;
    };
}
fn HermitianGeneralized(comptime T: type, comptime level: Level) type {
    if (InternalInt != i32) @compileError("Hermitian interface acceptance is LP64 only");
    return struct {
        fn invoke(layout: c_int, itype: WireInt, jobz: i8, uplo: i8, n: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, w: ?[*]align(1) HermitianReal(T), work: ?[*]align(1) T, lwork: WireInt, rwork: ?[*]align(1) HermitianReal(T)) WireInt {
            const info = binding.Shared.hermitianRawManagedWithDefaults(true, T, .hegv, level, binding.Provider.Execution.getContext(), .{
                .layout = layout,
                .itype = @intCast(itype),
                .jobz = @bitCast(jobz),
                .uplo = @bitCast(uplo),
                .n = @intCast(n),
                .a = a,
                .lda = @intCast(lda),
                .b = b,
                .ldb = @intCast(ldb),
                .w = w,
                .work = work,
                .lwork = @intCast(lwork),
                .rwork = rwork,
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return @intCast(info);
        }
        fn high(layout: c_int, itype: WireInt, jobz: i8, uplo: i8, n: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, w: ?[*]align(1) HermitianReal(T)) callconv(.c) WireInt {
            return invoke(layout, itype, jobz, uplo, n, a, lda, b, ldb, w, null, 0, null);
        }
        fn workEntry(layout: c_int, itype: WireInt, jobz: i8, uplo: i8, n: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, w: ?[*]align(1) HermitianReal(T), work: ?[*]align(1) T, lwork: WireInt, rwork: ?[*]align(1) HermitianReal(T)) callconv(.c) WireInt {
            return invoke(layout, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork);
        }
        pub const call = if (level == .high) high else workEntry;
    };
}
comptime {
    @export(&Factor(f32, .high).call, .{ .name = "LAPACKE_spotrf" });
    @export(&Factor(f32, .work).call, .{ .name = "LAPACKE_spotrf_work" });
    @export(&Solve(f32, .high).call, .{ .name = "LAPACKE_spotrs" });
    @export(&Solve(f32, .work).call, .{ .name = "LAPACKE_spotrs_work" });
    @export(&Factor(f64, .high).call, .{ .name = "LAPACKE_dpotrf" });
    @export(&Factor(f64, .work).call, .{ .name = "LAPACKE_dpotrf_work" });
    @export(&Solve(f64, .high).call, .{ .name = "LAPACKE_dpotrs" });
    @export(&Solve(f64, .work).call, .{ .name = "LAPACKE_dpotrs_work" });
    @export(&Factor(C, .high).call, .{ .name = "LAPACKE_cpotrf" });
    @export(&Factor(C, .work).call, .{ .name = "LAPACKE_cpotrf_work" });
    @export(&Solve(C, .high).call, .{ .name = "LAPACKE_cpotrs" });
    @export(&Solve(C, .work).call, .{ .name = "LAPACKE_cpotrs_work" });
    @export(&Factor(Z, .high).call, .{ .name = "LAPACKE_zpotrf" });
    @export(&Factor(Z, .work).call, .{ .name = "LAPACKE_zpotrf_work" });
    @export(&Solve(Z, .high).call, .{ .name = "LAPACKE_zpotrs" });
    @export(&Solve(Z, .work).call, .{ .name = "LAPACKE_zpotrs_work" });
    @export(&LuFactor(f32, .high).call, .{ .name = "LAPACKE_sgetrf" });
    @export(&LuFactor(f32, .work).call, .{ .name = "LAPACKE_sgetrf_work" });
    @export(&LuSolve(f32, .high).call, .{ .name = "LAPACKE_sgetrs" });
    @export(&LuSolve(f32, .work).call, .{ .name = "LAPACKE_sgetrs_work" });
    @export(&LuFactor(f64, .high).call, .{ .name = "LAPACKE_dgetrf" });
    @export(&LuFactor(f64, .work).call, .{ .name = "LAPACKE_dgetrf_work" });
    @export(&LuSolve(f64, .high).call, .{ .name = "LAPACKE_dgetrs" });
    @export(&LuSolve(f64, .work).call, .{ .name = "LAPACKE_dgetrs_work" });
    @export(&LuFactor(C, .high).call, .{ .name = "LAPACKE_cgetrf" });
    @export(&LuFactor(C, .work).call, .{ .name = "LAPACKE_cgetrf_work" });
    @export(&LuSolve(C, .high).call, .{ .name = "LAPACKE_cgetrs" });
    @export(&LuSolve(C, .work).call, .{ .name = "LAPACKE_cgetrs_work" });
    @export(&LuFactor(Z, .high).call, .{ .name = "LAPACKE_zgetrf" });
    @export(&LuFactor(Z, .work).call, .{ .name = "LAPACKE_zgetrf_work" });
    @export(&LuSolve(Z, .high).call, .{ .name = "LAPACKE_zgetrs" });
    @export(&LuSolve(Z, .work).call, .{ .name = "LAPACKE_zgetrs_work" });
    @export(&LuDriver(f32, .high).call, .{ .name = "LAPACKE_sgesv" });
    @export(&LuDriver(f32, .work).call, .{ .name = "LAPACKE_sgesv_work" });
    @export(&LuDriver(f64, .high).call, .{ .name = "LAPACKE_dgesv" });
    @export(&LuDriver(f64, .work).call, .{ .name = "LAPACKE_dgesv_work" });
    @export(&LuDriver(C, .high).call, .{ .name = "LAPACKE_cgesv" });
    @export(&LuDriver(C, .work).call, .{ .name = "LAPACKE_cgesv_work" });
    @export(&LuDriver(Z, .high).call, .{ .name = "LAPACKE_zgesv" });
    @export(&LuDriver(Z, .work).call, .{ .name = "LAPACKE_zgesv_work" });
    @export(&TridiagonalEigen(f32, .high).call, .{ .name = "LAPACKE_sstev" });
    @export(&TridiagonalEigen(f32, .work).call, .{ .name = "LAPACKE_sstev_work" });
    @export(&TridiagonalEigen(f64, .high).call, .{ .name = "LAPACKE_dstev" });
    @export(&TridiagonalEigen(f64, .work).call, .{ .name = "LAPACKE_dstev_work" });
    @export(&RankLeastSquares(f32, .high).call, .{ .name = "LAPACKE_sgelsd" });
    @export(&RankLeastSquares(f32, .work).call, .{ .name = "LAPACKE_sgelsd_work" });
    @export(&RankLeastSquares(f64, .high).call, .{ .name = "LAPACKE_dgelsd" });
    @export(&RankLeastSquares(f64, .work).call, .{ .name = "LAPACKE_dgelsd_work" });
    @export(&HermitianStandard(C, .high).call, .{ .name = "LAPACKE_cheev" });
    @export(&HermitianStandard(C, .work).call, .{ .name = "LAPACKE_cheev_work" });
    @export(&HermitianGeneralized(C, .high).call, .{ .name = "LAPACKE_chegv" });
    @export(&HermitianGeneralized(C, .work).call, .{ .name = "LAPACKE_chegv_work" });
    @export(&HermitianStandard(Z, .high).call, .{ .name = "LAPACKE_zheev" });
    @export(&HermitianStandard(Z, .work).call, .{ .name = "LAPACKE_zheev_work" });
    @export(&HermitianGeneralized(Z, .high).call, .{ .name = "LAPACKE_zhegv" });
    @export(&HermitianGeneralized(Z, .work).call, .{ .name = "LAPACKE_zhegv_work" });
    @export(&ComplexRankLeastSquares(C, .high).call, .{ .name = "LAPACKE_cgelsd" });
    @export(&ComplexRankLeastSquares(C, .work).call, .{ .name = "LAPACKE_cgelsd_work" });
    @export(&ComplexRankLeastSquares(Z, .high).call, .{ .name = "LAPACKE_zgelsd" });
    @export(&ComplexRankLeastSquares(Z, .work).call, .{ .name = "LAPACKE_zgelsd_work" });
    @export(&SingularValueDecomposition(f32, .high).call, .{ .name = "LAPACKE_sgesvd" });
    @export(&SingularValueDecomposition(f32, .work).call, .{ .name = "LAPACKE_sgesvd_work" });
    @export(&SingularValueDecomposition(f64, .high).call, .{ .name = "LAPACKE_dgesvd" });
    @export(&SingularValueDecomposition(f64, .work).call, .{ .name = "LAPACKE_dgesvd_work" });
    @export(&SingularValueDecomposition(C, .high).call, .{ .name = "LAPACKE_cgesvd" });
    @export(&SingularValueDecomposition(C, .work).call, .{ .name = "LAPACKE_cgesvd_work" });
    @export(&SingularValueDecomposition(Z, .high).call, .{ .name = "LAPACKE_zgesvd" });
    @export(&SingularValueDecomposition(Z, .work).call, .{ .name = "LAPACKE_zgesvd_work" });
    @export(&TriangularInverse(f32, .high).call, .{ .name = "LAPACKE_strtri" });
    @export(&TriangularInverse(f32, .work).call, .{ .name = "LAPACKE_strtri_work" });
    @export(&TriangularInverse(f64, .high).call, .{ .name = "LAPACKE_dtrtri" });
    @export(&TriangularInverse(f64, .work).call, .{ .name = "LAPACKE_dtrtri_work" });
    @export(&TriangularInverse(C, .high).call, .{ .name = "LAPACKE_ctrtri" });
    @export(&TriangularInverse(C, .work).call, .{ .name = "LAPACKE_ctrtri_work" });
    @export(&TriangularInverse(Z, .high).call, .{ .name = "LAPACKE_ztrtri" });
    @export(&TriangularInverse(Z, .work).call, .{ .name = "LAPACKE_ztrtri_work" });
}
