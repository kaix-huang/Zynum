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
comptime {
    @export(&SingularValueDecomposition(f32, .high).call, .{ .name = "LAPACKE_sgesvd" });
    @export(&SingularValueDecomposition(f32, .work).call, .{ .name = "LAPACKE_sgesvd_work" });
    @export(&SingularValueDecomposition(f64, .high).call, .{ .name = "LAPACKE_dgesvd" });
    @export(&SingularValueDecomposition(f64, .work).call, .{ .name = "LAPACKE_dgesvd_work" });
    @export(&SingularValueDecomposition(C, .high).call, .{ .name = "LAPACKE_cgesvd" });
    @export(&SingularValueDecomposition(C, .work).call, .{ .name = "LAPACKE_cgesvd_work" });
    @export(&SingularValueDecomposition(Z, .high).call, .{ .name = "LAPACKE_zgesvd" });
    @export(&SingularValueDecomposition(Z, .work).call, .{ .name = "LAPACKE_zgesvd_work" });
}
