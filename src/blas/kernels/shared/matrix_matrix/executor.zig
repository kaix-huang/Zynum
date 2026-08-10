// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Runtime executor for selected GEMM kernel ids.
//!
//! Dispatch chooses a descriptor. Tasks carry the descriptor's `KernelId`, and
//! this file maps that id to an implementation module. Branches are guarded by
//! compile-time feature constants so unsupported architecture paths do not pull
//! in unavailable symbols.

const aarch64_asimd = @import("../../arch/aarch64/matrix_matrix/asimd.zig");
const aarch64_sme = @import("../../arch/aarch64/matrix_matrix/sme.zig");
const aarch64_sve2 = @import("../../arch/aarch64/matrix_matrix/sve2.zig");
const catalog = @import("catalog.zig");
const generic = @import("generic.zig");
const gemm_task = @import("task.zig");
const x86_64_simd = @import("../../arch/x86_64/matrix_matrix/simd.zig");

fn runAsimd(comptime T: type, task: gemm_task.Task(T)) void {
    if (T == f32) return aarch64_asimd.noTransRealF32(task);
    if (T == f64) return aarch64_asimd.noTransRealF64(task);
    @compileError("GEMM executor supports f32 and f64");
}

fn runSve2(comptime T: type, task: gemm_task.Task(T)) void {
    if (T == f32) return aarch64_sve2.noTransRealF32(task);
    if (T == f64) return aarch64_sve2.noTransRealF64(task);
    @compileError("GEMM executor supports f32 and f64");
}

fn runSme(comptime T: type, task: gemm_task.Task(T)) void {
    if (T == f32) return aarch64_sme.noTransRealF32(task);
    if (T == f64) return aarch64_sme.noTransRealF64(task);
    @compileError("GEMM executor supports f32 and f64");
}

fn runX86F32(comptime tier: x86_64_simd.Tier, task: gemm_task.Task(f32)) void {
    if (comptime x86_64_simd.tierAvailable(tier)) return x86_64_simd.noTransRealF32For(tier, task);
    generic.noTransRealF32(task);
}

fn runX86F64(comptime tier: x86_64_simd.Tier, task: gemm_task.Task(f64)) void {
    if (comptime x86_64_simd.tierAvailable(tier)) return x86_64_simd.noTransRealF64For(tier, task);
    generic.noTransRealF64(task);
}

pub const ExecutorRoute = catalog.KernelId;

/// Single typed authority consumed by production execution and the registry
/// snapshot. A new kernel ID must be classified here before either surface can
/// compile.
pub fn executorRoute(kernel: catalog.KernelId) ExecutorRoute {
    return switch (kernel) {
        .auto => .auto,
        .generic_f32_4x4 => .generic_f32_4x4,
        .generic_f64_4x4 => .generic_f64_4x4,
        .aarch64_asimd_f32_12x8 => .aarch64_asimd_f32_12x8,
        .aarch64_asimd_f64_6x8 => .aarch64_asimd_f64_6x8,
        .aarch64_sve2_asimd_f32_12x8 => .aarch64_sve2_asimd_f32_12x8,
        .aarch64_sve2_asimd_f64_6x8 => .aarch64_sve2_asimd_f64_6x8,
        .aarch64_sme_f32_2mx2n => .aarch64_sme_f32_2mx2n,
        .aarch64_sme_f64_4mx2n => .aarch64_sme_f64_4mx2n,
        .x86_64_sse2_f32_packed => .x86_64_sse2_f32_packed,
        .x86_64_sse2_f64_packed => .x86_64_sse2_f64_packed,
        .x86_64_avx_f32_packed => .x86_64_avx_f32_packed,
        .x86_64_avx_f64_packed => .x86_64_avx_f64_packed,
        .x86_64_avx2_fma_f32_packed => .x86_64_avx2_fma_f32_packed,
        .x86_64_avx2_fma_f64_packed => .x86_64_avx2_fma_f64_packed,
        .x86_64_avx512f_fma_f32_packed => .x86_64_avx512f_fma_f32_packed,
        .x86_64_avx512f_fma_f64_packed => .x86_64_avx512f_fma_f64_packed,
    };
}

pub fn executorRouteMatchesKernel(kernel: catalog.KernelId, route: ExecutorRoute) bool {
    return kernel != .auto and route == kernel;
}

pub fn hasRegisteredExecutorMapping(kernel: catalog.KernelId) bool {
    return executorRouteMatchesKernel(kernel, executorRoute(kernel));
}

pub fn availableFor(comptime T: type, kernel: catalog.KernelId) bool {
    if (T == f32) return switch (kernel) {
        .auto, .generic_f32_4x4 => true,
        .aarch64_asimd_f32_12x8 => aarch64_asimd.enabled,
        .aarch64_sve2_asimd_f32_12x8 => aarch64_sve2.enabled,
        .aarch64_sme_f32_2mx2n => aarch64_sme.enabled,
        .x86_64_sse2_f32_packed => x86_64_simd.tierAvailable(.sse2),
        .x86_64_avx_f32_packed => x86_64_simd.tierAvailable(.avx),
        .x86_64_avx2_fma_f32_packed => x86_64_simd.tierAvailable(.avx2_fma),
        .x86_64_avx512f_fma_f32_packed => x86_64_simd.tierAvailable(.avx512f_fma),
        .generic_f64_4x4,
        .aarch64_asimd_f64_6x8,
        .aarch64_sve2_asimd_f64_6x8,
        .aarch64_sme_f64_4mx2n,
        .x86_64_sse2_f64_packed,
        .x86_64_avx_f64_packed,
        .x86_64_avx2_fma_f64_packed,
        .x86_64_avx512f_fma_f64_packed,
        => false,
    };
    if (T == f64) return switch (kernel) {
        .auto, .generic_f64_4x4 => true,
        .aarch64_asimd_f64_6x8 => aarch64_asimd.enabled,
        .aarch64_sve2_asimd_f64_6x8 => aarch64_sve2.enabled,
        .aarch64_sme_f64_4mx2n => aarch64_sme.enabled and aarch64_sme.supports_f64_accumulate,
        .x86_64_sse2_f64_packed => x86_64_simd.tierAvailable(.sse2),
        .x86_64_avx_f64_packed => x86_64_simd.tierAvailable(.avx),
        .x86_64_avx2_fma_f64_packed => x86_64_simd.tierAvailable(.avx2_fma),
        .x86_64_avx512f_fma_f64_packed => x86_64_simd.tierAvailable(.avx512f_fma),
        .generic_f32_4x4,
        .aarch64_asimd_f32_12x8,
        .aarch64_sve2_asimd_f32_12x8,
        .aarch64_sme_f32_2mx2n,
        .x86_64_sse2_f32_packed,
        .x86_64_avx_f32_packed,
        .x86_64_avx2_fma_f32_packed,
        .x86_64_avx512f_fma_f32_packed,
        => false,
    };
    @compileError("GEMM executor supports f32 and f64");
}

pub fn runF32(task: gemm_task.Task(f32)) void {
    switch (executorRoute(task.kernel)) {
        .auto, .generic_f32_4x4 => generic.noTransRealF32(task),
        .aarch64_asimd_f32_12x8 => if (comptime aarch64_asimd.enabled) runAsimd(f32, task) else generic.noTransRealF32(task),
        .aarch64_sve2_asimd_f32_12x8 => if (comptime aarch64_sve2.enabled) runSve2(f32, task) else generic.noTransRealF32(task),
        .aarch64_sme_f32_2mx2n => if (comptime aarch64_sme.enabled) runSme(f32, task) else generic.noTransRealF32(task),
        .x86_64_sse2_f32_packed => runX86F32(.sse2, task),
        .x86_64_avx_f32_packed => runX86F32(.avx, task),
        .x86_64_avx2_fma_f32_packed => runX86F32(.avx2_fma, task),
        .x86_64_avx512f_fma_f32_packed => runX86F32(.avx512f_fma, task),
        .generic_f64_4x4,
        .aarch64_asimd_f64_6x8,
        .aarch64_sve2_asimd_f64_6x8,
        .aarch64_sme_f64_4mx2n,
        .x86_64_sse2_f64_packed,
        .x86_64_avx_f64_packed,
        .x86_64_avx2_fma_f64_packed,
        .x86_64_avx512f_fma_f64_packed,
        => generic.noTransRealF32(task),
    }
}

pub fn runF64(task: gemm_task.Task(f64)) void {
    switch (executorRoute(task.kernel)) {
        .auto, .generic_f64_4x4 => generic.noTransRealF64(task),
        .aarch64_asimd_f64_6x8 => if (comptime aarch64_asimd.enabled) runAsimd(f64, task) else generic.noTransRealF64(task),
        .aarch64_sve2_asimd_f64_6x8 => if (comptime aarch64_sve2.enabled) runSve2(f64, task) else generic.noTransRealF64(task),
        .aarch64_sme_f64_4mx2n => if (comptime aarch64_sme.enabled and aarch64_sme.supports_f64_accumulate) runSme(f64, task) else generic.noTransRealF64(task),
        .x86_64_sse2_f64_packed => runX86F64(.sse2, task),
        .x86_64_avx_f64_packed => runX86F64(.avx, task),
        .x86_64_avx2_fma_f64_packed => runX86F64(.avx2_fma, task),
        .x86_64_avx512f_fma_f64_packed => runX86F64(.avx512f_fma, task),
        .generic_f32_4x4,
        .aarch64_asimd_f32_12x8,
        .aarch64_sve2_asimd_f32_12x8,
        .aarch64_sme_f32_2mx2n,
        .x86_64_sse2_f32_packed,
        .x86_64_avx_f32_packed,
        .x86_64_avx2_fma_f32_packed,
        .x86_64_avx512f_fma_f32_packed,
        => generic.noTransRealF64(task),
    }
}

pub fn run(comptime T: type, task: gemm_task.Task(T)) void {
    if (T == f32) return runF32(task);
    if (T == f64) return runF64(task);
    @compileError("GEMM executor supports f32 and f64");
}
