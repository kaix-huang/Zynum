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
}
