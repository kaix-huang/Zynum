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
// Generated boundary only. The injected binding owns the sole operation.
fn CholeskyDriver(comptime T: type, comptime level: Level) type {
    return struct {
        pub fn call(layout: c_int, uplo: i8, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt) callconv(.c) WireInt {
            const result = binding.Shared.posvRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                .layout = layout,
                .uplo = @bitCast(uplo),
                .n = @intCast(n),
                .nrhs = @intCast(nrhs),
                .a = a,
                .lda = @intCast(lda),
                .b = b,
                .ldb = @intCast(ldb),
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return finish(result);
        }
    };
}
// Generated boundary only. Existing binding owns the sole operation and Core.
fn TriangularSolve(comptime T: type, comptime level: Level) type {
    return struct {
        pub fn call(layout: c_int, uplo: i8, trans: i8, diag: i8, n: WireInt, nrhs: WireInt, a: ?[*]align(1) const T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt) callconv(.c) WireInt {
            const result = binding.Shared.trtrsRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
                .layout = layout,
                .uplo = @bitCast(uplo),
                .trans = @bitCast(trans),
                .diag = @bitCast(diag),
                .n = @intCast(n),
                .nrhs = @intCast(nrhs),
                .a = a,
                .lda = @intCast(lda),
                .b = b,
                .ldb = @intCast(ldb),
            }, profileDefaults()) catch |err| terminal(@errorName(err));
            return finish(result);
        }
    };
}
// Generated forwarding only. Mathematics, Context and every lease belong to the owner.
fn invokeLuInverse(comptime T: type, comptime level: Level, layout: c_int, n: WireInt, a: ?[*]align(1) T, lda: WireInt, ipiv: ?[*]align(1) const WireInt, work: ?[*]align(1) T, lwork: WireInt) WireInt {
    const result = binding.Shared.getriRawManagedWithDefaults(true, T, InternalInt, level, binding.Provider.Execution.getContext(), .{
        .layout = layout,
        .n = @intCast(n),
        .a = a,
        .lda = @intCast(lda),
        .pivots = if (ipiv) |pointer| @as([*]align(1) const InternalInt, @ptrCast(pointer)) else null,
        .work = work,
        .lwork = @intCast(lwork),
    }, profileDefaults()) catch |err| terminal(@errorName(err));
    return finish(result);
}
fn LuInverse(comptime T: type, comptime level: Level) type {
    return if (level == .high) struct {
        pub fn call(layout: c_int, n: WireInt, a: ?[*]align(1) T, lda: WireInt, ipiv: ?[*]align(1) const WireInt) callconv(.c) WireInt {
            return invokeLuInverse(T, level, layout, n, a, lda, ipiv, null, 0);
        }
    } else struct {
        pub fn call(layout: c_int, n: WireInt, a: ?[*]align(1) T, lda: WireInt, ipiv: ?[*]align(1) const WireInt, work: ?[*]align(1) T, lwork: WireInt) callconv(.c) WireInt {
            return invokeLuInverse(T, level, layout, n, a, lda, ipiv, work, lwork);
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
// Interface-only QR template: types, arithmetic and resources come from one binding.
fn QrFactor(comptime T: type, comptime level: Level) type {
    if (InternalInt != i32) @compileError("QR interface acceptance is LP64 only");
    return struct {
        fn invoke(layout: c_int, m: WireInt, n: WireInt, a: ?[*]align(1) T, lda: WireInt, tau: ?[*]align(1) T, work: ?[*]align(1) T, lwork: WireInt) WireInt {
            const info = binding.Shared.qrRawManagedWithDefaults(true, T, level, binding.Provider.Execution.getContext(), .{ .layout = layout, .m = @intCast(m), .n = @intCast(n), .a = a, .lda = @intCast(lda), .tau = tau, .work = work, .lwork = @intCast(lwork) }, profileDefaults()) catch |err| terminal(@errorName(err));
            return @intCast(info);
        }
        fn high(layout: c_int, m: WireInt, n: WireInt, a: ?[*]align(1) T, lda: WireInt, tau: ?[*]align(1) T) callconv(.c) WireInt {
            return invoke(layout, m, n, a, lda, tau, null, 0);
        }
        fn workEntry(layout: c_int, m: WireInt, n: WireInt, a: ?[*]align(1) T, lda: WireInt, tau: ?[*]align(1) T, work: ?[*]align(1) T, lwork: WireInt) callconv(.c) WireInt {
            return invoke(layout, m, n, a, lda, tau, work, lwork);
        }
        pub const call = if (level == .high) high else workEntry;
    };
}
fn QrApply(comptime T: type, comptime level: Level) type {
    if (InternalInt != i32) @compileError("QR interface acceptance is LP64 only");
    return struct {
        fn invoke(layout: c_int, side: i8, trans: i8, m: WireInt, n: WireInt, k: WireInt, a: ?[*]align(1) const T, lda: WireInt, tau: ?[*]align(1) const T, c: ?[*]align(1) T, ldc: WireInt, work: ?[*]align(1) T, lwork: WireInt) WireInt {
            const info = binding.Shared.qApplyRawManagedWithDefaults(true, T, level, binding.Provider.Execution.getContext(), .{ .layout = layout, .side = @bitCast(side), .trans = @bitCast(trans), .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .a = a, .lda = @intCast(lda), .tau = tau, .c = c, .ldc = @intCast(ldc), .work = work, .lwork = @intCast(lwork) }, profileDefaults()) catch |err| terminal(@errorName(err));
            return @intCast(info);
        }
        fn high(layout: c_int, side: i8, trans: i8, m: WireInt, n: WireInt, k: WireInt, a: ?[*]align(1) const T, lda: WireInt, tau: ?[*]align(1) const T, c: ?[*]align(1) T, ldc: WireInt) callconv(.c) WireInt {
            return invoke(layout, side, trans, m, n, k, a, lda, tau, c, ldc, null, 0);
        }
        fn workEntry(layout: c_int, side: i8, trans: i8, m: WireInt, n: WireInt, k: WireInt, a: ?[*]align(1) const T, lda: WireInt, tau: ?[*]align(1) const T, c: ?[*]align(1) T, ldc: WireInt, work: ?[*]align(1) T, lwork: WireInt) callconv(.c) WireInt {
            return invoke(layout, side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork);
        }
        pub const call = if (level == .high) high else workEntry;
    };
}
// Interface-only explicit Q generation; one implementation and resource owner.
fn QrGenerate(comptime T: type, comptime level: Level) type {
    if (InternalInt != i32) @compileError("Q generation interface acceptance is LP64 only");
    return struct {
        fn invoke(layout: c_int, m: WireInt, n: WireInt, k: WireInt, a: ?[*]align(1) T, lda: WireInt, tau: ?[*]align(1) const T, work: ?[*]align(1) T, lwork: WireInt) WireInt {
            const info = binding.Shared.qGenerateRawManagedWithDefaults(true, T, level, binding.Provider.Execution.getContext(), .{ .layout = layout, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .a = a, .lda = @intCast(lda), .tau = tau, .work = work, .lwork = @intCast(lwork) }, profileDefaults()) catch |err| terminal(@errorName(err));
            return @intCast(info);
        }
        fn high(layout: c_int, m: WireInt, n: WireInt, k: WireInt, a: ?[*]align(1) T, lda: WireInt, tau: ?[*]align(1) const T) callconv(.c) WireInt {
            return invoke(layout, m, n, k, a, lda, tau, null, 0);
        }
        fn workEntry(layout: c_int, m: WireInt, n: WireInt, k: WireInt, a: ?[*]align(1) T, lda: WireInt, tau: ?[*]align(1) const T, work: ?[*]align(1) T, lwork: WireInt) callconv(.c) WireInt {
            return invoke(layout, m, n, k, a, lda, tau, work, lwork);
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
    @export(&QrFactor(f32, .high).call, .{ .name = "LAPACKE_sgeqrf" });
    @export(&QrFactor(f32, .work).call, .{ .name = "LAPACKE_sgeqrf_work" });
    @export(&QrApply(f32, .high).call, .{ .name = "LAPACKE_sormqr" });
    @export(&QrApply(f32, .work).call, .{ .name = "LAPACKE_sormqr_work" });
    @export(&QrFactor(f64, .high).call, .{ .name = "LAPACKE_dgeqrf" });
    @export(&QrFactor(f64, .work).call, .{ .name = "LAPACKE_dgeqrf_work" });
    @export(&QrApply(f64, .high).call, .{ .name = "LAPACKE_dormqr" });
    @export(&QrApply(f64, .work).call, .{ .name = "LAPACKE_dormqr_work" });
    @export(&QrFactor(C, .high).call, .{ .name = "LAPACKE_cgeqrf" });
    @export(&QrFactor(C, .work).call, .{ .name = "LAPACKE_cgeqrf_work" });
    @export(&QrApply(C, .high).call, .{ .name = "LAPACKE_cunmqr" });
    @export(&QrApply(C, .work).call, .{ .name = "LAPACKE_cunmqr_work" });
    @export(&QrFactor(Z, .high).call, .{ .name = "LAPACKE_zgeqrf" });
    @export(&QrFactor(Z, .work).call, .{ .name = "LAPACKE_zgeqrf_work" });
    @export(&QrApply(Z, .high).call, .{ .name = "LAPACKE_zunmqr" });
    @export(&QrApply(Z, .work).call, .{ .name = "LAPACKE_zunmqr_work" });
    @export(&QrGenerate(f32, .high).call, .{ .name = "LAPACKE_sorgqr" });
    @export(&QrGenerate(f32, .work).call, .{ .name = "LAPACKE_sorgqr_work" });
    @export(&QrGenerate(f64, .high).call, .{ .name = "LAPACKE_dorgqr" });
    @export(&QrGenerate(f64, .work).call, .{ .name = "LAPACKE_dorgqr_work" });
    @export(&QrGenerate(C, .high).call, .{ .name = "LAPACKE_cungqr" });
    @export(&QrGenerate(C, .work).call, .{ .name = "LAPACKE_cungqr_work" });
    @export(&QrGenerate(Z, .high).call, .{ .name = "LAPACKE_zungqr" });
    @export(&QrGenerate(Z, .work).call, .{ .name = "LAPACKE_zungqr_work" });
    @export(&CholeskyDriver(f32, .high).call, .{ .name = "LAPACKE_sposv" });
    @export(&CholeskyDriver(f32, .work).call, .{ .name = "LAPACKE_sposv_work" });
    @export(&CholeskyDriver(f64, .high).call, .{ .name = "LAPACKE_dposv" });
    @export(&CholeskyDriver(f64, .work).call, .{ .name = "LAPACKE_dposv_work" });
    @export(&CholeskyDriver(C, .high).call, .{ .name = "LAPACKE_cposv" });
    @export(&CholeskyDriver(C, .work).call, .{ .name = "LAPACKE_cposv_work" });
    @export(&CholeskyDriver(Z, .high).call, .{ .name = "LAPACKE_zposv" });
    @export(&CholeskyDriver(Z, .work).call, .{ .name = "LAPACKE_zposv_work" });
    @export(&TriangularSolve(f32, .high).call, .{ .name = "LAPACKE_strtrs" });
    @export(&TriangularSolve(f32, .work).call, .{ .name = "LAPACKE_strtrs_work" });
    @export(&TriangularSolve(f64, .high).call, .{ .name = "LAPACKE_dtrtrs" });
    @export(&TriangularSolve(f64, .work).call, .{ .name = "LAPACKE_dtrtrs_work" });
    @export(&TriangularSolve(C, .high).call, .{ .name = "LAPACKE_ctrtrs" });
    @export(&TriangularSolve(C, .work).call, .{ .name = "LAPACKE_ctrtrs_work" });
    @export(&TriangularSolve(Z, .high).call, .{ .name = "LAPACKE_ztrtrs" });
    @export(&TriangularSolve(Z, .work).call, .{ .name = "LAPACKE_ztrtrs_work" });
    @export(&LuInverse(f32, .high).call, .{ .name = "LAPACKE_sgetri" });
    @export(&LuInverse(f32, .work).call, .{ .name = "LAPACKE_sgetri_work" });
    @export(&LuInverse(f64, .high).call, .{ .name = "LAPACKE_dgetri" });
    @export(&LuInverse(f64, .work).call, .{ .name = "LAPACKE_dgetri_work" });
    @export(&LuInverse(C, .high).call, .{ .name = "LAPACKE_cgetri" });
    @export(&LuInverse(C, .work).call, .{ .name = "LAPACKE_cgetri_work" });
    @export(&LuInverse(Z, .high).call, .{ .name = "LAPACKE_zgetri" });
    @export(&LuInverse(Z, .work).call, .{ .name = "LAPACKE_zgetri_work" });
}
