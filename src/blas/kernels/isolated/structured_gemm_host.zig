// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Main-graph GEMM callback used by the isolated structured candidate object.

const abi = @import("x86_64_structured_abi.zig");
const gemm_impl = @import("../../core/matrix_matrix/gemm.zig");
const scalar = @import("../../core/shared/scalar.zig");
const CallResources = @import("../../core/execution/thread_pool.zig").CallResources;

fn realValue(comptime T: type, bits: u64) T {
    if (T == f32) return @bitCast(@as(u32, @truncate(bits)));
    if (T == f64) return @bitCast(bits);
    @compileError("structured GEMM host expects a real component");
}

fn scalarValue(comptime T: type, re: u64, im: u64) T {
    if (T == f32 or T == f64) return realValue(T, re);
    const Real = scalar.Real(T);
    return .{ .re = realValue(Real, re), .im = realValue(Real, im) };
}

fn order(raw: u8) scalar.Order {
    return switch (@as(abi.Transpose, @fromBackingInt(@intCast(raw)))) {
        .no_trans => .no_trans,
        .trans => .trans,
        .conj_trans => .conj_trans,
    };
}

fn run(comptime T: type, resources: CallResources, request: *abi.GemmRequest) void {
    const a: [*]const T = @ptrCast(@alignCast(request.a));
    const b: [*]const T = @ptrCast(@alignCast(request.b));
    const c: [*]T = @ptrCast(@alignCast(request.c));
    gemm_impl.gemmWithResources(
        T,
        resources,
        order(request.transa),
        order(request.transb),
        request.m,
        request.n,
        request.k,
        scalarValue(T, request.alpha_re, request.alpha_im),
        a,
        request.lda,
        b,
        request.ldb,
        scalarValue(T, request.beta_re, request.beta_im),
        c,
        request.ldc,
    );
}

pub fn execute(resources: CallResources, request: *abi.GemmRequest) void {
    const scalar_tag: abi.Scalar = @fromBackingInt(@intCast(request.scalar));
    switch (scalar_tag) {
        .f32 => run(f32, resources, request),
        .f64 => run(f64, resources, request),
        .complex_f32 => run(scalar.ComplexF32, resources, request),
        .complex_f64 => run(scalar.ComplexF64, resources, request),
    }
}
