// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const gemm_task = @import("gemm_task.zig");
const aarch64_asimd = @import("aarch64/asimd.zig");
const aarch64_sme = @import("aarch64/sme.zig");
const aarch64_sve2 = @import("aarch64/sve2.zig");
const generic = @import("generic/gemm.zig");
const x86_64_simd = @import("x86_64/simd.zig");

pub const Task = gemm_task.Task;

pub const Backend = enum {
    generic_asimd,
    aarch64_asimd_fma,
    aarch64_sve2_asimd_fma,
    aarch64_sme_asimd_fma,
    x86_64_sse2,
    x86_64_avx,
    x86_64_avx2,
    x86_64_avx512f,
};

pub const active_backend: Backend = if (aarch64_sme.enabled)
    .aarch64_sme_asimd_fma
else if (aarch64_sve2.enabled)
    .aarch64_sve2_asimd_fma
else if (aarch64_asimd.enabled)
    .aarch64_asimd_fma
else if (x86_64_simd.supports_avx512f)
    .x86_64_avx512f
else if (x86_64_simd.supports_avx2)
    .x86_64_avx2
else if (x86_64_simd.supports_avx)
    .x86_64_avx
else if (x86_64_simd.enabled)
    .x86_64_sse2
else
    .generic_asimd;

pub fn backendName() []const u8 {
    return switch (active_backend) {
        .generic_asimd => "generic_asimd",
        .aarch64_asimd_fma => "aarch64_asimd_fma",
        .aarch64_sve2_asimd_fma => "aarch64_sve2_asimd_fma",
        .aarch64_sme_asimd_fma => "aarch64_sme_asimd_fma",
        .x86_64_sse2 => "x86_64_sse2",
        .x86_64_avx => if (x86_64_simd.supports_fma) "x86_64_avx_fma" else "x86_64_avx",
        .x86_64_avx2 => if (x86_64_simd.supports_fma) "x86_64_avx2_fma" else "x86_64_avx2",
        .x86_64_avx512f => "x86_64_avx512f_fma",
    };
}

pub fn preferredColumnBlock(comptime T: type) usize {
    return switch (active_backend) {
        .aarch64_sme_asimd_fma => aarch64_sme.preferredColumnBlock(T),
        .aarch64_sve2_asimd_fma => aarch64_sve2.preferredColumnBlock(T),
        .aarch64_asimd_fma => aarch64_asimd.preferredColumnBlock(T),
        .x86_64_sse2, .x86_64_avx, .x86_64_avx2, .x86_64_avx512f => x86_64_simd.preferredColumnBlock(T),
        .generic_asimd => generic.preferredColumnBlock(T),
    };
}

pub fn noTransReal(comptime T: type, task: Task(T)) void {
    if (T == f32) {
        noTransRealF32(task);
    } else if (T == f64) {
        noTransRealF64(task);
    } else {
        @compileError("GEMM kernels support f32 and f64");
    }
}

pub fn noTransRealF32(task: Task(f32)) void {
    switch (active_backend) {
        .aarch64_sme_asimd_fma => aarch64_sme.noTransRealF32(task),
        .aarch64_sve2_asimd_fma => aarch64_sve2.noTransRealF32(task),
        .aarch64_asimd_fma => aarch64_asimd.noTransRealF32(task),
        .x86_64_sse2, .x86_64_avx, .x86_64_avx2, .x86_64_avx512f => x86_64_simd.noTransRealF32(task),
        .generic_asimd => generic.noTransRealF32(task),
    }
}

pub fn noTransRealF64(task: Task(f64)) void {
    switch (active_backend) {
        .aarch64_sme_asimd_fma => aarch64_sme.noTransRealF64(task),
        .aarch64_sve2_asimd_fma => aarch64_sve2.noTransRealF64(task),
        .aarch64_asimd_fma => aarch64_asimd.noTransRealF64(task),
        .x86_64_sse2, .x86_64_avx, .x86_64_avx2, .x86_64_avx512f => x86_64_simd.noTransRealF64(task),
        .generic_asimd => generic.noTransRealF64(task),
    }
}
