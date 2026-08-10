// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Fixed-layout private ABI for the isolated x86_64 compact-triangular object.

const std = @import("std");

pub const Operation = enum(u8) {
    tpmv,
    tpsv,
    tbsv,
};

pub const Scalar = enum(u8) {
    f32,
    f64,
    complex_f32,
    complex_f64,
};

pub const Uplo = enum(u8) {
    upper,
    lower,
};

pub const Transpose = enum(u8) {
    no_trans,
    trans,
    conj_trans,
};

pub const Diagonal = enum(u8) {
    unit,
    non_unit,
};

pub const Request = extern struct {
    operation: u8,
    scalar: u8,
    uplo: u8,
    transpose: u8,
    diagonal: u8,
    reserved: [3]u8,
    n: i32,
    k: i32,
    lda: i32,
    incx: i32,
    matrix: *const anyopaque,
    vector: *anyopaque,
};

pub fn init(
    operation: Operation,
    scalar: Scalar,
    uplo: Uplo,
    transpose: Transpose,
    diagonal: Diagonal,
    n: i32,
    k: i32,
    lda: i32,
    matrix: *const anyopaque,
    vector: *anyopaque,
    incx: i32,
) Request {
    return .{
        .operation = @intFromEnum(operation),
        .scalar = @intFromEnum(scalar),
        .uplo = @intFromEnum(uplo),
        .transpose = @intFromEnum(transpose),
        .diagonal = @intFromEnum(diagonal),
        .reserved = .{ 0, 0, 0 },
        .n = n,
        .k = k,
        .lda = lda,
        .incx = incx,
        .matrix = matrix,
        .vector = vector,
    };
}

test "compact triangular private request ABI has fixed 64-bit layout" {
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(Request));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(Request, "n"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(Request, "matrix"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(Request, "vector"));
}
