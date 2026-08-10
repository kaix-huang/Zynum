// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Fixed-layout private ABI for isolated x86_64 Level 2 width candidates.

const std = @import("std");

pub const Operation = enum(u8) {
    gemv_no_trans_unit,
    gemv_no_trans_full,
    gemv_trans_unit,
    gemv_trans_full,
    gemv_no_trans_complex,
    gemv_trans_complex,
    ger_real,
    ger_complex,
};

pub const Scalar = enum(u8) {
    f32,
    f64,
    complex_f32,
    complex_f64,
};

pub const Request = extern struct {
    operation: u8,
    scalar: u8,
    flags: u8,
    reserved0: u8 = 0,
    lda: i32,
    m: u64,
    n: u64,
    alpha_re: u64,
    alpha_im: u64,
    beta_re: u64,
    beta_im: u64,
    input0: *const anyopaque,
    input1: *const anyopaque,
    output: *anyopaque,
};

test "Level 2 width private request ABI has fixed 64-bit layout" {
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(Request));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(Request, "m"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(Request, "alpha_re"));
    try std.testing.expectEqual(@as(usize, 56), @offsetOf(Request, "input0"));
    try std.testing.expectEqual(@as(usize, 72), @offsetOf(Request, "output"));
}
