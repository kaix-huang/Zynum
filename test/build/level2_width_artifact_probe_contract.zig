// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Test-only declaration of the isolated Level 2 width object's private ABI.

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

pub extern var zynum_internal_x86_64_level2_width_enabled: u8;
pub extern fn zynum_internal_x86_64_level2_width_execute(request: *Request) callconv(.c) u8;

comptime {
    if (@sizeOf(Request) != 80 or
        @offsetOf(Request, "m") != 8 or
        @offsetOf(Request, "alpha_re") != 24 or
        @offsetOf(Request, "input0") != 56 or
        @offsetOf(Request, "output") != 72)
    {
        @compileError("x86 Level 2 width request ABI layout changed");
    }
}
