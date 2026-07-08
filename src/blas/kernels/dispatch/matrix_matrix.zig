// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const gemm_task = @import("../shared/matrix_matrix/task.zig");
const aarch64_asimd = @import("../arch/aarch64/matrix_matrix/asimd.zig");
const aarch64_sme = @import("../arch/aarch64/matrix_matrix/sme.zig");
const aarch64_sve2 = @import("../arch/aarch64/matrix_matrix/sve2.zig");
pub const catalog = @import("../shared/matrix_matrix/catalog.zig");
const executor = @import("../shared/matrix_matrix/executor.zig");
const generic = @import("../shared/matrix_matrix/generic.zig");
const x86_64_simd = @import("../arch/x86_64/matrix_matrix/simd.zig");

pub const Task = gemm_task.Task;
pub const Implementation = gemm_task.Implementation;
pub const ExecutionPlan = gemm_task.ExecutionPlan;

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

pub fn descriptor(comptime T: type) catalog.Descriptor {
    return candidates(T).at(0);
}

pub fn candidates(comptime T: type) catalog.CandidateList {
    return switch (active_backend) {
        .aarch64_sme_asimd_fma => if (T == f64 and !aarch64_sme.supports_f64_accumulate)
            if (comptime aarch64_sve2.enabled)
                catalog.candidateList(.{ catalog.aarch64Sve2Descriptor(T), catalog.genericDescriptor(T) })
            else
                catalog.candidateList(.{ catalog.aarch64AsimdDescriptor(T), catalog.genericDescriptor(T) })
        else
            catalog.candidateList(.{
                catalog.aarch64SmeDescriptor(T, aarch64_sme.preferredColumnBlock(f32) / 2 * @sizeOf(f32)),
                if (comptime aarch64_sve2.enabled) catalog.aarch64Sve2Descriptor(T) else catalog.aarch64AsimdDescriptor(T),
                catalog.genericDescriptor(T),
            }),
        .aarch64_sve2_asimd_fma => catalog.candidateList(.{ catalog.aarch64Sve2Descriptor(T), catalog.aarch64AsimdDescriptor(T), catalog.genericDescriptor(T) }),
        .aarch64_asimd_fma => catalog.candidateList(.{ catalog.aarch64AsimdDescriptor(T), catalog.genericDescriptor(T) }),
        .x86_64_sse2 => catalog.candidateList(.{ catalog.x86Descriptor(T, .x86_64_sse2), catalog.genericDescriptor(T) }),
        .x86_64_avx => catalog.candidateList(.{ catalog.x86Descriptor(T, .x86_64_avx), catalog.genericDescriptor(T) }),
        .x86_64_avx2 => catalog.candidateList(.{ catalog.x86Descriptor(T, if (x86_64_simd.supports_fma) .x86_64_avx2_fma else .x86_64_avx), catalog.genericDescriptor(T) }),
        .x86_64_avx512f => catalog.candidateList(.{ catalog.x86Descriptor(T, .x86_64_avx512f_fma), catalog.genericDescriptor(T) }),
        .generic_asimd => catalog.candidateList(.{catalog.genericDescriptor(T)}),
    };
}

pub fn candidateCount(comptime T: type) usize {
    return candidates(T).len;
}

pub fn candidate(comptime T: type, index: usize) catalog.Descriptor {
    return candidates(T).at(index);
}

pub fn implementationFor(desc: catalog.Descriptor) Implementation {
    return executor.implementationFor(desc);
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

pub fn tryNoTransRealF32Fast(m: gemm_task.BlasInt, n: gemm_task.BlasInt, k: gemm_task.BlasInt, alpha: f32, a: [*]const f32, lda: gemm_task.BlasInt, b: [*]const f32, ldb: gemm_task.BlasInt, beta: f32, c: [*]f32, ldc: gemm_task.BlasInt) bool {
    return switch (active_backend) {
        .aarch64_sme_asimd_fma => aarch64_sme.tryNoTransRealF32Fast(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        else => false,
    };
}

pub fn freeCurrentThreadCaches() void {
    if (comptime aarch64_sme.enabled) aarch64_sme.freeCurrentThreadCaches();
}

fn archSimdKernel() catalog.KernelId {
    return switch (active_backend) {
        .aarch64_sme_asimd_fma, .aarch64_sve2_asimd_fma => if (comptime aarch64_sve2.enabled) .aarch64_sve2_asimd_packed else .aarch64_asimd_packed,
        .aarch64_asimd_fma => .aarch64_asimd_packed,
        .x86_64_sse2, .x86_64_avx, .x86_64_avx2, .x86_64_avx512f => .x86_64_simd_packed,
        .generic_asimd => .generic_basic,
    };
}

fn defaultKernel(comptime T: type, implementation: Implementation) catalog.KernelId {
    return switch (implementation) {
        .auto => descriptor(T).kernel,
        .generic => .generic_basic,
        .arch_simd => archSimdKernel(),
        .streaming_matrix => descriptor(T).kernel,
    };
}

pub fn noTransRealF32(task: Task(f32)) void {
    var selected = task;
    if (selected.kernel == .auto) selected.kernel = defaultKernel(f32, selected.implementation);
    executor.runF32(selected);
}

pub fn noTransRealF64(task: Task(f64)) void {
    var selected = task;
    if (selected.kernel == .auto) selected.kernel = defaultKernel(f64, selected.implementation);
    executor.runF64(selected);
}
