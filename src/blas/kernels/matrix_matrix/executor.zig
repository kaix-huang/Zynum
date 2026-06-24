// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Runtime executor for selected GEMM kernel ids.
//!
//! Dispatch chooses a descriptor. Tasks carry the descriptor's `KernelId`, and
//! this file maps that id to an implementation module. Branches are guarded by
//! compile-time feature constants so unsupported architecture paths do not pull
//! in unavailable symbols.

const aarch64_asimd = @import("../aarch64/asimd.zig");
const aarch64_sme = @import("../aarch64/sme.zig");
const aarch64_sve2 = @import("../aarch64/sve2.zig");
const catalog = @import("catalog.zig");
const generic = @import("../generic/matrix_matrix.zig");
const gemm_task = @import("task.zig");
const x86_64_simd = @import("../x86_64/simd.zig");

pub fn implementationFor(desc: catalog.Descriptor) gemm_task.Implementation {
    return switch (desc.family) {
        .generic => .generic,
        .packed_simd => .arch_simd,
        .streaming_matrix => .streaming_matrix,
    };
}

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

fn runX86Simd(comptime T: type, task: gemm_task.Task(T)) void {
    if (T == f32) return x86_64_simd.noTransRealF32(task);
    if (T == f64) return x86_64_simd.noTransRealF64(task);
    @compileError("GEMM executor supports f32 and f64");
}

pub fn run(comptime T: type, task: gemm_task.Task(T)) void {
    switch (task.kernel) {
        .auto, .generic_basic => generic.noTransReal(T, task),
        .aarch64_asimd_packed => if (comptime aarch64_asimd.enabled) runAsimd(T, task) else generic.noTransReal(T, task),
        .aarch64_sve2_asimd_packed => if (comptime aarch64_sve2.enabled) runSve2(T, task) else if (comptime aarch64_asimd.enabled) runAsimd(T, task) else generic.noTransReal(T, task),
        .aarch64_sme_streaming => if (comptime aarch64_sme.enabled) runSme(T, task) else if (comptime aarch64_sve2.enabled) runSve2(T, task) else if (comptime aarch64_asimd.enabled) runAsimd(T, task) else generic.noTransReal(T, task),
        .x86_64_simd_packed => if (comptime x86_64_simd.enabled) runX86Simd(T, task) else generic.noTransReal(T, task),
    }
}

pub fn runF32(task: gemm_task.Task(f32)) void {
    run(f32, task);
}

pub fn runF64(task: gemm_task.Task(f64)) void {
    run(f64, task);
}
