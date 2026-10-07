// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Private fixed-layout ABI shared by the main graph and isolated object.

pub const Operation = enum(u8) {
    scal,
    rscal,
    swap,
    axpy,
    axpby,
    dot,
    dot_f32_acc_f64,
    asum,
    nrm2,
    iamax,
    rot,
    rotm,
};

pub const Scalar = enum(u8) {
    f32,
    f64,
    complex_f32,
    complex_f64,
};

pub const conjugate_x: u16 = 1;

pub const Request = extern struct {
    operation: u8,
    scalar: u8,
    flags: u16,
    reserved: u32,
    n: usize,
    x: *anyopaque,
    y: ?*anyopaque,
    args: [8]u64,
    result: [2]u64,
    result_index: i64,
};

pub fn init(operation: Operation, scalar: Scalar, n: usize, x: *anyopaque, y: ?*anyopaque) Request {
    return .{
        .operation = @backingInt(operation),
        .scalar = @backingInt(scalar),
        .flags = 0,
        .reserved = 0,
        .n = n,
        .x = x,
        .y = y,
        .args = @as([8]u64, @splat(0)),
        .result = @as([2]u64, @splat(0)),
        .result_index = 0,
    };
}
