// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Generated interface forwarding only. The existing owner supplies all state.
const std = @import("std");
const builtin = @import("builtin");
const binding = @import("lapacke-interface-owner");
pub const OwnerIdentity = binding.Provider.OwnerIdentity;
pub const BindingOwner = binding;
pub const WireInt = c_int;
const InternalInt = i32;
const C = binding.implementation.ComplexF32;
const Z = binding.implementation.ComplexF64;
const Level = binding.common.Level;
comptime {
    if (builtin.is_test or builtin.target.cpu.arch != .aarch64 or builtin.target.os.tag != .macos) @compileError("component requires a non-test Apple arm64 product");
    if (@sizeOf(c_int) != 4 or @sizeOf(WireInt) != 4 or @sizeOf(InternalInt) != 4) @compileError("integer representation");
    if (std.math.minInt(WireInt) != std.math.minInt(InternalInt) or std.math.maxInt(WireInt) != std.math.maxInt(InternalInt)) @compileError("wire/internal integer range");
    if (@sizeOf(C) != 8 or @alignOf(C) != 4 or @offsetOf(C, "re") != 0 or @offsetOf(C, "im") != 4) @compileError("complex8 representation");
    if (@sizeOf(Z) != 16 or @alignOf(Z) != 8 or @offsetOf(Z, "re") != 0 or @offsetOf(Z, "im") != 8) @compileError("complex16 representation");
}
fn terminal(reason: []const u8) noreturn {
    _ = reason; // The managed operation and every lease have already ended.
    std.c.abort();
}
extern fn LAPACKE_xerbla(name: [*:0]const u8, info: WireInt) callconv(.c) void;
extern fn xerbla_(name: [*]const u8, info: *const WireInt) callconv(.c) void;
const Diagnostic = binding.Diagnostics(InternalInt);
const Handlers = struct {
    pub fn lapacke(_: @This(), name: [:0]const u8, info: InternalInt) void {
        LAPACKE_xerbla(name.ptr, @intCast(info));
    }
    pub fn fortran(_: @This(), name: [:0]const u8, info: InternalInt) void {
        const value: WireInt = @intCast(info);
        xerbla_(name.ptr, &value);
    }
};
fn diagnostic(_: *anyopaque, event: binding.common.ErrorEvent) void {
    Diagnostic.dispatch(event, Handlers{}) catch terminal("diagnostic");
}
fn defaults(context: *binding.Provider.Execution.context.Context) binding.shared_services.Callbacks {
    const callback: binding.shared_services.Callback = .{ .userdata = context, .function = diagnostic };
    return .{ .lapacke = callback, .fortran_returning_test = callback };
}
fn Real(comptime T: type) type {
    return if (T == f32 or T == C) f32 else f64;
}
fn GelssStandard(comptime T: type, comptime level: Level) type {
    if (InternalInt != i32) @compileError("GELSS acceptance is LP64 only");
    return struct {
        fn invoke(layout: c_int, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, s: ?[*]align(1) Real(T), rcond: Real(T), rank: ?[*]align(1) InternalInt, work: ?[*]align(1) T, lwork: WireInt, rwork: ?[*]align(1) Real(T)) WireInt {
            const context = binding.Provider.Execution.getContext();
            const info = binding.Shared.gelssRawManagedWithDefaults(true, T, level, context, .{
                .layout = layout,
                .m = @intCast(m),
                .n = @intCast(n),
                .nrhs = @intCast(nrhs),
                .a = a,
                .lda = @intCast(lda),
                .b = b,
                .ldb = @intCast(ldb),
                .singular = s,
                .rcond = rcond,
                .rank = rank,
                .work = work,
                .lwork = @intCast(lwork),
                .rwork = rwork,
            }, defaults(context)) catch |err| terminal(@errorName(err));
            return @intCast(info);
        }
        fn high(layout: c_int, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, s: ?[*]align(1) Real(T), rcond: Real(T), rank: ?[*]align(1) InternalInt) callconv(.c) WireInt {
            return invoke(layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank, null, 0, null);
        }
        fn realWork(layout: c_int, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, s: ?[*]align(1) Real(T), rcond: Real(T), rank: ?[*]align(1) InternalInt, work: ?[*]align(1) T, lwork: WireInt) callconv(.c) WireInt {
            return invoke(layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, null);
        }
        fn complexWork(layout: c_int, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, s: ?[*]align(1) Real(T), rcond: Real(T), rank: ?[*]align(1) InternalInt, work: ?[*]align(1) T, lwork: WireInt, rwork: ?[*]align(1) Real(T)) callconv(.c) WireInt {
            return invoke(layout, m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, rwork);
        }
        pub const call = if (level == .high) high else if (T == C or T == Z) complexWork else realWork;
    };
}
comptime {
    @export(&GelssStandard(f32, .high).call, .{ .name = "LAPACKE_sgelss" });
    @export(&GelssStandard(f32, .work).call, .{ .name = "LAPACKE_sgelss_work" });
    @export(&GelssStandard(f64, .high).call, .{ .name = "LAPACKE_dgelss" });
    @export(&GelssStandard(f64, .work).call, .{ .name = "LAPACKE_dgelss_work" });
    @export(&GelssStandard(C, .high).call, .{ .name = "LAPACKE_cgelss" });
    @export(&GelssStandard(C, .work).call, .{ .name = "LAPACKE_cgelss_work" });
    @export(&GelssStandard(Z, .high).call, .{ .name = "LAPACKE_zgelss" });
    @export(&GelssStandard(Z, .work).call, .{ .name = "LAPACKE_zgelss_work" });
}
