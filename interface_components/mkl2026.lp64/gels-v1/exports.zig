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
fn GelsStandard(comptime T: type, comptime level: Level) type {
    if (InternalInt != i32) @compileError("GELS acceptance is LP64 only");
    return struct {
        fn invoke(layout: c_int, trans: i8, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, work: ?[*]align(1) T, lwork: WireInt) WireInt {
            const context = binding.Provider.Execution.getContext();
            const info = binding.Shared.gelsRawManagedWithDefaults(true, T, level, context, .{
                .layout = layout,
                .trans = @bitCast(trans),
                .m = @intCast(m),
                .n = @intCast(n),
                .nrhs = @intCast(nrhs),
                .a = a,
                .lda = @intCast(lda),
                .b = b,
                .ldb = @intCast(ldb),
                .work = work,
                .lwork = @intCast(lwork),
            }, defaults(context)) catch |err| terminal(@errorName(err));
            return @intCast(info);
        }
        fn high(layout: c_int, trans: i8, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt) callconv(.c) WireInt {
            return invoke(layout, trans, m, n, nrhs, a, lda, b, ldb, null, 0);
        }
        fn workCall(layout: c_int, trans: i8, m: WireInt, n: WireInt, nrhs: WireInt, a: ?[*]align(1) T, lda: WireInt, b: ?[*]align(1) T, ldb: WireInt, work: ?[*]align(1) T, lwork: WireInt) callconv(.c) WireInt {
            return invoke(layout, trans, m, n, nrhs, a, lda, b, ldb, work, lwork);
        }
        pub const call = if (level == .high) high else workCall;
    };
}
comptime {
    @export(&GelsStandard(f32, .high).call, .{ .name = "LAPACKE_sgels" });
    @export(&GelsStandard(f32, .work).call, .{ .name = "LAPACKE_sgels_work" });
    @export(&GelsStandard(f64, .high).call, .{ .name = "LAPACKE_dgels" });
    @export(&GelsStandard(f64, .work).call, .{ .name = "LAPACKE_dgels_work" });
    @export(&GelsStandard(C, .high).call, .{ .name = "LAPACKE_cgels" });
    @export(&GelsStandard(C, .work).call, .{ .name = "LAPACKE_cgels_work" });
    @export(&GelsStandard(Z, .high).call, .{ .name = "LAPACKE_zgels" });
    @export(&GelsStandard(Z, .work).call, .{ .name = "LAPACKE_zgels_work" });
}
