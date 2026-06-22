// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

pub const BlasInt = i32;

/// Complex scalar with f32 real and imaginary components.
/// This matches Python/NumPy `complex64` total-width naming.
pub const ComplexF32 = extern struct {
    re: f32,
    im: f32,
};

/// Complex scalar with f64 real and imaginary components.
/// This matches Python/NumPy `complex128` total-width naming.
pub const ComplexF64 = extern struct {
    re: f64,
    im: f64,
};

pub const Layout = enum(c_int) {
    row_major = 101,
    col_major = 102,
};

pub const Transpose = enum(c_int) {
    no_trans = 111,
    trans = 112,
    conj_trans = 113,
};

pub const Uplo = enum(c_int) {
    upper = 121,
    lower = 122,
};

pub const Diag = enum(c_int) {
    non_unit = 131,
    unit = 132,
};

pub const Side = enum(c_int) {
    left = 141,
    right = 142,
};

pub fn complexF32(re: f32, im: f32) ComplexF32 {
    return .{ .re = re, .im = im };
}

pub fn complexF64(re: f64, im: f64) ComplexF64 {
    return .{ .re = re, .im = im };
}
