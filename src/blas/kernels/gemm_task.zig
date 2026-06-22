// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const types = @import("../types.zig");

pub const BlasInt = types.BlasInt;

pub fn Task(comptime T: type) type {
    return struct {
        m: usize,
        n0: usize,
        n1: usize,
        k: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        b: [*]const T,
        ldb: BlasInt,
        beta: T,
        c: [*]T,
        ldc: BlasInt,
        allow_sme: bool = false,
    };
}

pub inline fn toUsize(x: BlasInt) usize {
    return @intCast(x);
}

pub inline fn matIndex(lda: BlasInt, row: usize, col: usize) usize {
    return row + col * toUsize(lda);
}
