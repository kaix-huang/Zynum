// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Fixed-layout private ABI for isolated structured Level 3 candidates.

const std = @import("std");

pub const Operation = enum(u8) {
    symm_dense,
    trmm_right,
    trsm_right,
};

pub const Scalar = enum(u8) {
    f32,
    f64,
    complex_f32,
    complex_f64,
};

pub const Side = enum(u8) { left, right };
pub const Uplo = enum(u8) { upper, lower };
pub const Transpose = enum(u8) { no_trans, trans, conj_trans };
pub const Diagonal = enum(u8) { unit, non_unit };

pub const Request = extern struct {
    operation: u8,
    scalar: u8,
    side: u8,
    uplo: u8,
    transpose: u8,
    diagonal: u8,
    flags: u8,
    reserved0: u8 = 0,
    m: i32,
    n: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    reserved1: i32 = 0,
    alpha_re: u64,
    alpha_im: u64,
    beta_re: u64,
    beta_im: u64,
    a: *const anyopaque,
    b: *const anyopaque,
    c: *anyopaque,
};

pub const GemmRequest = extern struct {
    transa: u8,
    transb: u8,
    scalar: u8,
    reserved0: [5]u8 = .{ 0, 0, 0, 0, 0 },
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    alpha_re: u64,
    alpha_im: u64,
    beta_re: u64,
    beta_im: u64,
    a: *const anyopaque,
    b: *const anyopaque,
    c: *anyopaque,
};

test "structured candidate private ABIs have fixed 64-bit layouts" {
    try std.testing.expectEqual(@as(usize, 88), @sizeOf(Request));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(Request, "m"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(Request, "alpha_re"));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(Request, "a"));

    try std.testing.expectEqual(@as(usize, 88), @sizeOf(GemmRequest));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(GemmRequest, "m"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(GemmRequest, "alpha_re"));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(GemmRequest, "a"));
}

/// Matched-build, synchronous x86_64 service borrow. The opaque host binding
/// and every callback remain valid until execute_v2 returns. This is not a
/// public ABI, allocator vtable, registration mechanism or async capability.
pub const ServicesV1 = extern struct {
    version: u32,
    struct_bytes: u32,
    borrow: *const anyopaque,
    alloc: *const fn (*const anyopaque, usize, u8) callconv(.c) ?[*]u8,
    free: *const fn (*const anyopaque, [*]u8, usize, u8) callconv(.c) void,
    task_count: *const fn (*const anyopaque, usize, usize) callconv(.c) usize,
    run_low_latency: *const fn (*const anyopaque, usize, *const anyopaque, usize) callconv(.c) u8,
    gemm: *const fn (*const anyopaque, *GemmRequest) callconv(.c) void,
};

/// Inspect only the descriptor header, before request payload or callbacks.
pub fn servicesSupported(services: *const ServicesV1) bool {
    return services.version == 1 and services.struct_bytes == @sizeOf(ServicesV1);
}

comptime {
    // These declarations are also imported by non-x86 host graphs. Only the
    // matched x86_64 protocol carries the fixed 64-bit layout obligation.
    if (@import("builtin").target.cpu.arch == .x86_64) {
        if (@sizeOf(Request) != 88 or @sizeOf(GemmRequest) != 88 or
            @alignOf(Request) != 8 or @alignOf(GemmRequest) != 8)
            @compileError("structured mathematical packet layout changed");
        if (@sizeOf(ServicesV1) != 56 or @alignOf(ServicesV1) != 8 or
            @offsetOf(ServicesV1, "version") != 0 or
            @offsetOf(ServicesV1, "struct_bytes") != 4 or
            @offsetOf(ServicesV1, "borrow") != 8 or
            @offsetOf(ServicesV1, "alloc") != 16 or
            @offsetOf(ServicesV1, "free") != 24 or
            @offsetOf(ServicesV1, "task_count") != 32 or
            @offsetOf(ServicesV1, "run_low_latency") != 40 or
            @offsetOf(ServicesV1, "gemm") != 48)
            @compileError("structured service v1 layout changed");
    }
    if (@TypeOf(@as(ServicesV1, undefined).alloc) != *const fn (*const anyopaque, usize, u8) callconv(.c) ?[*]u8 or
        @TypeOf(@as(ServicesV1, undefined).free) != *const fn (*const anyopaque, [*]u8, usize, u8) callconv(.c) void or
        @TypeOf(@as(ServicesV1, undefined).task_count) != *const fn (*const anyopaque, usize, usize) callconv(.c) usize or
        @TypeOf(@as(ServicesV1, undefined).run_low_latency) != *const fn (*const anyopaque, usize, *const anyopaque, usize) callconv(.c) u8 or
        @TypeOf(@as(ServicesV1, undefined).gemm) != *const fn (*const anyopaque, *GemmRequest) callconv(.c) void)
        @compileError("structured services must retain their C callback types");
}

test "structured service v1 checks only the version and size header" {
    // Deliberately leave borrow/callbacks undefined: the header predicate must
    // neither inspect them nor invoke anything, even for a rejected version.
    var services: ServicesV1 = undefined;
    services.version = 1;
    services.struct_bytes = @sizeOf(ServicesV1);
    try std.testing.expect(servicesSupported(&services));
    services.version = 2;
    try std.testing.expect(!servicesSupported(&services));
    services.version = 1;
    services.struct_bytes = @sizeOf(ServicesV1) - 1;
    try std.testing.expect(!servicesSupported(&services));
    services.struct_bytes = @sizeOf(ServicesV1) + 1;
    try std.testing.expect(!servicesSupported(&services));
}
