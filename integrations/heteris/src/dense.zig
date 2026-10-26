// Copyright (C) 2026 Zynum contributors
// Copyright (C) 2026 Scientific Platform contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Heteris admission for the adapted upstream portable GEMM reference loop.
const std = @import("std");
const v = @import("views.zig");
const c = v.c;
const scalar = @import("scientific_sources").scalar;
const reference = @import("adapted/gemm.zig");

pub fn Matrix(comptime T: type) type {
    return struct {
        view: *const v.View,
        mapping: v.Mapping,
        op: u32 = c.ZYNUM_OP_NONE,
        pub fn at(self: @This(), i: usize, j: usize) *T {
            const row = if (self.op == c.ZYNUM_OP_NONE) i else j;
            const col = if (self.op == c.ZYNUM_OP_NONE) j else i;
            const offset: i128 = @as(i128, self.view.byte_offset) + @as(i128, row) * self.view.strides[0] + @as(i128, col) * self.view.strides[1];
            return @ptrFromInt(self.mapping.base + @as(usize, @intCast(offset)));
        }
        pub fn get(self: @This(), i: usize, j: usize) T {
            return scalar.maybeConj(T, self.at(i, j).*, self.op == c.ZYNUM_OP_CONJUGATE_TRANSPOSE);
        }
    };
}

fn finite(comptime T: type, value: T) bool {
    return if (comptime scalar.isComplex(T)) std.math.isFinite(value.re) and std.math.isFinite(value.im) else std.math.isFinite(value);
}

fn gemm(comptime T: type, r_arg: ?*const v.Resources, op_a: u32, op_b: u32, a_arg: ?*const v.View, b_arg: ?*const v.View, c_arg: ?*const v.View, alpha: T, beta: T) v.Status {
    // Copy every descriptor and the callback table before invoking any provider.
    const resources = (r_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const a = (a_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const b = (b_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    const out = (c_arg orelse return c.HETERIS_INVALID_ARGUMENT).*;
    if (!finite(T, alpha) or !finite(T, beta) or op_a > c.ZYNUM_OP_CONJUGATE_TRANSPOSE or op_b > c.ZYNUM_OP_CONJUGATE_TRANSPOSE) return c.HETERIS_INVALID_ARGUMENT;
    if (!v.resourcesValid(&resources)) return c.HETERIS_INVALID_ARGUMENT;
    const dtype: u32 = if (T == f64) c.HETERIS_F64 else c.HETERIS_COMPLEX128;
    var status = v.validateType(&a, 2, dtype, c.HETERIS_VIEW_READ);
    if (status != c.HETERIS_OK) return status;
    status = v.validateType(&b, 2, dtype, c.HETERIS_VIEW_READ);
    if (status != c.HETERIS_OK) return status;
    const out_flags: u32 = @as(u32, c.HETERIS_VIEW_WRITE) | (if (scalar.isZero(T, beta)) @as(u32, 0) else @as(u32, c.HETERIS_VIEW_READ));
    status = v.validateType(&out, 2, dtype, out_flags);
    if (status != c.HETERIS_OK) return status;
    const m = a.shape[if (op_a == c.ZYNUM_OP_NONE) 0 else 1];
    const k = a.shape[if (op_a == c.ZYNUM_OP_NONE) 1 else 0];
    const n = b.shape[if (op_b == c.ZYNUM_OP_NONE) 1 else 0];
    if (b.shape[if (op_b == c.ZYNUM_OP_NONE) 0 else 1] != k or out.shape[0] != m or out.shape[1] != n) return c.HETERIS_INVALID_ARGUMENT;
    if (m > std.math.maxInt(usize) or n > std.math.maxInt(usize) or k > std.math.maxInt(usize)) return c.HETERIS_OVERFLOW;
    var am: v.Mapping = undefined;
    var bm: v.Mapping = undefined;
    var cm: v.Mapping = undefined;
    status = v.mapped(&resources, &a, &am);
    if (status != c.HETERIS_OK) return status;
    status = v.mapped(&resources, &b, &bm);
    if (status != c.HETERIS_OK) return status;
    status = v.mapped(&resources, &out, &cm);
    if (status != c.HETERIS_OK) return status;
    if (v.intersects(cm, am) or v.intersects(cm, bm)) return c.HETERIS_OVERLAPPING_WRITE;
    reference.gemm(T, Matrix(T), @intCast(m), @intCast(n), @intCast(k), alpha, .{ .view = &a, .mapping = am, .op = op_a }, .{ .view = &b, .mapping = bm, .op = op_b }, beta, .{ .view = &out, .mapping = cm });
    return c.HETERIS_OK;
}

pub export fn zynum_gemm_f64(r: ?*const v.Resources, op_a: u32, op_b: u32, a: ?*const v.View, b: ?*const v.View, out: ?*const v.View, alpha: f64, beta: f64) v.Status {
    return gemm(f64, r, op_a, op_b, a, b, out, alpha, beta);
}
pub fn zynum_gemm_c128(r: ?*const v.Resources, op_a: u32, op_b: u32, a: ?*const v.View, b: ?*const v.View, out: ?*const v.View, alpha: c.zynum_complex128, beta: c.zynum_complex128) v.Status {
    return gemm(scalar.ComplexF64, r, op_a, op_b, a, b, out, .{ .re = alpha.real, .im = alpha.imag }, .{ .re = beta.real, .im = beta.imag });
}

// Internal scalar boundary for the C ABI wrapper in descriptor.c. Zig 0.17's
// x86_64 SysV lowering incorrectly spills these two-double structs after six
// integer-register arguments; scalar f64 arguments retain the correct C ABI.
pub export fn zynum_internal_gemm_c128_components(r: ?*const v.Resources, op_a: u32, op_b: u32, a: ?*const v.View, b: ?*const v.View, out: ?*const v.View, ar: f64, ai: f64, br: f64, bi: f64) v.Status {
    return zynum_gemm_c128(r, op_a, op_b, a, b, out, .{ .real = ar, .imag = ai }, .{ .real = br, .imag = bi });
}
