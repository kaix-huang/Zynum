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
