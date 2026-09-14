// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const builtin = @import("builtin");
const root = @import("root");

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const matrix_vector_ops = @import("../matrix_vector.zig");
const core_pool = @import("../execution/thread_pool.zig");
const blocked = @import("structured_blocked.zig");
const gemm_dispatch = @import("../../kernels/dispatch/matrix_matrix.zig");
const runtime = @import("../../runtime.zig");
const isolated_structured = @import("../../kernels/isolated/x86_64_structured_bridge.zig");
const structured_tuning = @import("../../kernels/tuning/structured.zig");

const use_isolated_structured = if (@hasDecl(root, "zynum_structured_object_candidates"))
    root.zynum_structured_object_candidates
else
    false;

pub const BlasInt = scalar.BlasInt;
pub const Order = scalar.Order;
pub const Uplo = scalar.Uplo;
pub const Side = scalar.Side;

const zero = scalar.zero;
const add = scalar.add;
const mul = scalar.mul;
const conj = scalar.conj;
const isComplex = scalar.isComplex;
const isZero = scalar.isZero;

const toUsize = indexing.toUsize;
const matIndex = indexing.matIndex;
const matrixValue = matrix_vector_ops.matrixValue;
const symValue = matrix_vector_ops.symValue;

fn SymmTask(comptime T: type) type {
    return struct {
        side: Side,
        uplo: Uplo,
        m: usize,
        n: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        b: [*]const T,
        ldb: BlasInt,
        beta: T,
        c: [*]T,
        ldc: BlasInt,
        herm: bool,
        task_index: usize,
        task_count: usize,
    };
}

fn runSymmColumns(comptime T: type, task: SymmTask(T)) void {
    var j = task.task_index;
    while (j < task.n) : (j += task.task_count) {
        for (0..task.m) |i| {
            var sum = zero(T);
            if (task.side == .left) {
                for (0..task.m) |p| sum = add(T, sum, mul(T, symValue(T, task.uplo, task.a, task.lda, i, p, task.herm), task.b[matIndex(task.ldb, p, j)]));
            } else {
                for (0..task.n) |p| sum = add(T, sum, mul(T, task.b[matIndex(task.ldb, i, p)], symValue(T, task.uplo, task.a, task.lda, p, j, task.herm)));
            }
            const idxc = matIndex(task.ldc, i, j);
            task.c[idxc] = add(T, mul(T, task.alpha, sum), if (isZero(T, task.beta)) zero(T) else mul(T, task.beta, task.c[idxc]));
        }
    }
}

fn runSymmTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const SymmTask(T) = @ptrCast(@alignCast(raw_tasks));
    runSymmColumns(T, tasks[index]);
}

fn runSymmTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runSymmTask(f32, raw_tasks, index);
}

fn runSymmTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runSymmTask(f64, raw_tasks, index);
}

fn runSymmTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runSymmTask(scalar.ComplexF32, raw_tasks, index);
}

fn runSymmTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runSymmTask(scalar.ComplexF64, raw_tasks, index);
}

fn runParallelSymm(comptime T: type, tasks: []const SymmTask(T)) bool {
    const runner = if (T == f32)
        runSymmTaskF32
    else if (T == f64)
        runSymmTaskF64
    else if (T == scalar.ComplexF32)
        runSymmTaskC32
    else if (T == scalar.ComplexF64)
        runSymmTaskC64
    else
        @compileError("parallel SYMM supports BLAS scalar types");
    return core_pool.runLowLatency(runner, @ptrCast(tasks.ptr), tasks.len);
}

pub fn symm(comptime T: type, side: Side, uplo: Uplo, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) void {
    if (m_ <= 0 or n_ <= 0) return;
    if (isZero(T, alpha)) {
        if (scalar.isOne(T, beta)) return;
        for (0..toUsize(n_)) |j| {
            for (0..toUsize(m_)) |i| {
                const index = matIndex(ldc, i, j);
                c[index] = if (isZero(T, beta)) zero(T) else mul(T, beta, c[index]);
            }
        }
        return;
    }
    // Candidate: reuse bounded structured panels once both output dimensions
    // amortize a 64-wide GEMM update. The AArch64 blocked route never expands
    // the whole structured operand, and acquires its panel before any writes.
    if (comptime builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos) {
        const profile = structured_tuning.aarch64_macos_active_panel_candidate;
        if (m_ > 0 and n_ > 0 and profile.candidate(toUsize(m_), toUsize(n_))) {
            const capability = gemm_dispatch.activeCapability();
            if (capability == .aarch64_asimd_fma or capability == .aarch64_sme) {
                if (blocked.trySymm(T, .{ .block_size = profile.block_size }, side, uplo, m_, n_, alpha, a, lda, b, ldb, beta, c, ldc, herm)) return;
            }
        }
    }
    // Use larger updates for multi-thread execution; retain small-kernel fallbacks
    // for modes that did not improve in the threaded comparison.
    if (comptime builtin.cpu.arch == .x86_64) {
        if (m_ >= 128 and n_ >= 128 and (runtime.maxThreads() == 1 or (m_ >= 256 and n_ >= 256)) and gemm_dispatch.activeCapability() == .x86_64_avx2_fma) {
            if (blocked.trySymm(T, .{ .block_size = if (runtime.maxThreads() > 1) 256 else if (scalar.isComplex(T)) 128 else 64 }, side, uplo, m_, n_, alpha, a, lda, b, ldb, beta, c, ldc, herm)) return;
        }
    }
    if (m_ <= 0 or n_ <= 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (comptime use_isolated_structured and builtin.cpu.arch == .x86_64) {
        const profile = structured_tuning.x86_64_object_profile;
        const tuning_side: structured_tuning.Side = if (side == .left) .left else .right;
        if (profile.denseCandidate(structured_tuning.scalarKind(T), tuning_side, herm, m, n) and
            isolated_structured.trySymm(T, side, uplo, m_, n_, alpha, a, lda, b, ldb, beta, c, ldc, herm)) return;
    }
    var tasks: [core_pool.max_tasks]SymmTask(T) = undefined;
    const order = if (side == .left) m else n;
    const work = m *| n *| order;
    const task_count = if (work >= 8 * 1024 * 1024) @min(core_pool.taskCount(n, 4), 32) else 1;
    if (task_count > 1) {
        for (tasks[0..task_count], 0..) |*task, task_index| {
            task.* = .{
                .side = side,
                .uplo = uplo,
                .m = m,
                .n = n,
                .alpha = alpha,
                .a = a,
                .lda = lda,
                .b = b,
                .ldb = ldb,
                .beta = beta,
                .c = c,
                .ldc = ldc,
                .herm = herm,
                .task_index = task_index,
                .task_count = task_count,
            };
        }
        if (runParallelSymm(T, tasks[0..task_count])) return;
    }
    runSymmColumns(T, .{
        .side = side,
        .uplo = uplo,
        .m = m,
        .n = n,
        .alpha = alpha,
        .a = a,
        .lda = lda,
        .b = b,
        .ldb = ldb,
        .beta = beta,
        .c = c,
        .ldc = ldc,
        .herm = herm,
        .task_index = 0,
        .task_count = 1,
    });
}

fn SyrkTask(comptime T: type) type {
    return struct {
        uplo: Uplo,
        trans: Order,
        n: usize,
        k: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        beta: T,
        c: [*]T,
        ldc: BlasInt,
        herm: bool,
        task_index: usize,
        task_count: usize,
    };
}

fn scaledStoredValue(comptime T: type, beta: T, source: *const T, hermitian_diagonal: bool) T {
    if (scalar.isZero(T, beta)) return scalar.zero(T);
    if (comptime scalar.isComplex(T)) {
        if (hermitian_diagonal) return scalar.realScalar(T, beta.re * source.re);
    }
    if (scalar.isOne(T, beta)) return source.*;
    return scalar.mul(T, beta, source.*);
}

fn scaleStoredRank(comptime T: type, uplo: Uplo, n: usize, beta: T, c: [*]T, ldc: BlasInt, herm: bool) void {
    for (0..n) |j| {
        const first = if (uplo == .upper) 0 else j;
        const end = if (uplo == .upper) j + 1 else n;
        for (first..end) |i| {
            const index = matIndex(ldc, i, j);
            c[index] = scaledStoredValue(T, beta, &c[index], herm and i == j);
        }
    }
}

noinline fn runSyrkColumnsC32Mode(comptime trans: Order, comptime herm: bool, task: SyrkTask(scalar.ComplexF32)) void {
    const T = scalar.ComplexF32;
    var j = task.task_index;
    while (j < task.n) : (j += task.task_count) {
        const row0: usize = if (task.uplo == .upper) 0 else j;
        const row1: usize = if (task.uplo == .upper) j + 1 else task.n;
        for (row0..row1) |i| {
            var sum = zero(T);
            for (0..task.k) |p| {
                const ai = if (trans == .no_trans) task.a[matIndex(task.lda, i, p)] else matrixValue(T, trans, task.a, task.lda, i, p);
                var aj = if (trans == .no_trans) task.a[matIndex(task.lda, j, p)] else matrixValue(T, trans, task.a, task.lda, j, p);
                if (herm) aj = conj(T, aj);
                sum = add(T, sum, mul(T, ai, aj));
            }
            const idxc = matIndex(task.ldc, i, j);
            task.c[idxc] = add(T, mul(T, task.alpha, sum), scaledStoredValue(T, task.beta, &task.c[idxc], herm and i == j));
            if (herm and i == j) {
                if (comptime isComplex(T)) task.c[idxc].im = 0;
            }
        }
    }
}

fn runSyrkColumns(comptime T: type, task: SyrkTask(T)) void {
    if (T == scalar.ComplexF32) {
        switch (task.trans) {
            .no_trans => {
                if (task.herm) return runSyrkColumnsC32Mode(.no_trans, true, task);
                return runSyrkColumnsC32Mode(.no_trans, false, task);
            },
            .trans => {
                if (task.herm) return runSyrkColumnsC32Mode(.trans, true, task);
                return runSyrkColumnsC32Mode(.trans, false, task);
            },
            .conj_trans => {
                if (task.herm) return runSyrkColumnsC32Mode(.conj_trans, true, task);
                return runSyrkColumnsC32Mode(.conj_trans, false, task);
            },
        }
    }
    var j = task.task_index;
    while (j < task.n) : (j += task.task_count) {
        const row0: usize = if (task.uplo == .upper) 0 else j;
        const row1: usize = if (task.uplo == .upper) j + 1 else task.n;
        for (row0..row1) |i| {
            var sum = zero(T);
            for (0..task.k) |p| {
                const ai = if (task.trans == .no_trans) task.a[matIndex(task.lda, i, p)] else matrixValue(T, task.trans, task.a, task.lda, i, p);
                var aj = if (task.trans == .no_trans) task.a[matIndex(task.lda, j, p)] else matrixValue(T, task.trans, task.a, task.lda, j, p);
                if (task.herm) aj = conj(T, aj);
                sum = add(T, sum, mul(T, ai, aj));
            }
            const idxc = matIndex(task.ldc, i, j);
            task.c[idxc] = add(T, mul(T, task.alpha, sum), scaledStoredValue(T, task.beta, &task.c[idxc], task.herm and i == j));
            if (task.herm and i == j) {
                if (comptime isComplex(T)) task.c[idxc].im = 0;
            }
        }
    }
}

fn runSyrkTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const SyrkTask(T) = @ptrCast(@alignCast(raw_tasks));
    runSyrkColumns(T, tasks[index]);
}

fn runSyrkTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runSyrkTask(f32, raw_tasks, index);
}

fn runSyrkTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runSyrkTask(f64, raw_tasks, index);
}

fn runSyrkTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runSyrkTask(scalar.ComplexF32, raw_tasks, index);
}

fn runSyrkTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runSyrkTask(scalar.ComplexF64, raw_tasks, index);
}

fn runParallelSyrk(comptime T: type, tasks: []const SyrkTask(T)) bool {
    const runner = if (T == f32)
        runSyrkTaskF32
    else if (T == f64)
        runSyrkTaskF64
    else if (T == scalar.ComplexF32)
        runSyrkTaskC32
    else if (T == scalar.ComplexF64)
        runSyrkTaskC64
    else
        @compileError("parallel SYRK supports BLAS scalar types");
    return core_pool.runLowLatency(runner, @ptrCast(tasks.ptr), tasks.len);
}

// Keep the macOS candidate implementation out of the public fallback body.
noinline fn tryMacSyrk(comptime T: type, uplo: Uplo, trans_: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) bool {
    const profile = structured_tuning.aarch64_macos_rank_tile_candidate;
    return blocked.trySyrk(T, .{ .block_size = profile.block_size, .direct_rank_panels = profile.direct_panels, .packed_rank_panels = profile.packed_panels, .parallel_packed_rank_panels = profile.parallel_packed_panels }, uplo, trans_, n_, k_, alpha, a, lda, beta, c, ldc, herm);
}

pub fn syrk(comptime T: type, uplo: Uplo, trans_: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) void {
    if (n_ <= 0) return;
    if (k_ == 0 or isZero(T, alpha)) {
        scaleStoredRank(T, uplo, toUsize(n_), beta, c, ldc, herm);
        return;
    }
    // Candidate: compute one private output tile, then commit only its stored
    // triangle. All workspace is acquired before changing caller output.
    if (comptime builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos) {
        const profile = structured_tuning.aarch64_macos_rank_tile_candidate;
        if (n_ > 0 and k_ > 0 and profile.candidate(toUsize(n_), toUsize(k_), false) and (runtime.maxThreads() == 1 or (profile.candidate(toUsize(n_), toUsize(k_), true) and (T != scalar.ComplexF64 or n_ >= 128 or k_ >= 64)))) {
            const capability = gemm_dispatch.activeCapability();
            if (capability == .aarch64_asimd_fma or capability == .aarch64_sme) {
                if (tryMacSyrk(T, uplo, trans_, n_, k_, alpha, a, lda, beta, c, ldc, herm)) return;
            }
        }
    }
    // Use larger updates for multi-thread execution; retain small-kernel fallbacks
    // for modes that did not improve in the threaded comparison.
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n_ >= 128 and k_ >= 64 and (runtime.maxThreads() == 1 or n_ >= 256) and gemm_dispatch.activeCapability() == .x86_64_avx2_fma) {
            if (blocked.trySyrk(T, .{ .block_size = if (runtime.maxThreads() > 1) 256 else if (T == scalar.ComplexF32) 128 else 64 }, uplo, trans_, n_, k_, alpha, a, lda, beta, c, ldc, herm)) return;
        }
    }
    if (n_ <= 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    var tasks: [core_pool.max_tasks]SyrkTask(T) = undefined;
    const work = n *| n *| k;
    const task_count = if (work >= 128 * 1024) @min(core_pool.taskCount(n, 8), 32) else 1;
    if (task_count > 1) {
        for (tasks[0..task_count], 0..) |*task, task_index| {
            task.* = .{
                .uplo = uplo,
                .trans = trans_,
                .n = n,
                .k = k,
                .alpha = alpha,
                .a = a,
                .lda = lda,
                .beta = beta,
                .c = c,
                .ldc = ldc,
                .herm = herm,
                .task_index = task_index,
                .task_count = task_count,
            };
        }
        if (runParallelSyrk(T, tasks[0..task_count])) return;
    }
    runSyrkColumns(T, .{
        .uplo = uplo,
        .trans = trans_,
        .n = n,
        .k = k,
        .alpha = alpha,
        .a = a,
        .lda = lda,
        .beta = beta,
        .c = c,
        .ldc = ldc,
        .herm = herm,
        .task_index = 0,
        .task_count = 1,
    });
}

fn Syr2kTask(comptime T: type) type {
    return struct {
        uplo: Uplo,
        trans: Order,
        n: usize,
        k: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        b: [*]const T,
        ldb: BlasInt,
        beta: T,
        c: [*]T,
        ldc: BlasInt,
        herm: bool,
        task_index: usize,
        task_count: usize,
    };
}

fn runSyr2kColumns(comptime T: type, task: Syr2kTask(T)) void {
    var j = task.task_index;
    while (j < task.n) : (j += task.task_count) {
        const row0: usize = if (task.uplo == .upper) 0 else j;
        const row1: usize = if (task.uplo == .upper) j + 1 else task.n;
        for (row0..row1) |i| {
            var sum = zero(T);
            for (0..task.k) |p| {
                const ai = matrixValue(T, task.trans, task.a, task.lda, i, p);
                const bi = matrixValue(T, task.trans, task.b, task.ldb, i, p);
                var aj = matrixValue(T, task.trans, task.a, task.lda, j, p);
                var bj = matrixValue(T, task.trans, task.b, task.ldb, j, p);
                if (task.herm) {
                    aj = conj(T, aj);
                    bj = conj(T, bj);
                    sum = add(T, sum, add(T, mul(T, task.alpha, mul(T, ai, bj)), mul(T, conj(T, task.alpha), mul(T, bi, aj))));
                } else {
                    sum = add(T, sum, add(T, mul(T, ai, bj), mul(T, bi, aj)));
                }
            }
            const idxc = matIndex(task.ldc, i, j);
            const prod = if (task.herm) sum else mul(T, task.alpha, sum);
            task.c[idxc] = add(T, prod, scaledStoredValue(T, task.beta, &task.c[idxc], task.herm and i == j));
            if (task.herm and i == j) {
                if (comptime isComplex(T)) task.c[idxc].im = 0;
            }
        }
    }
}

fn runSyr2kTask(comptime T: type, raw_tasks: *const anyopaque, index: usize) void {
    const tasks: [*]const Syr2kTask(T) = @ptrCast(@alignCast(raw_tasks));
    runSyr2kColumns(T, tasks[index]);
}

fn runSyr2kTaskF32(raw_tasks: *const anyopaque, index: usize) void {
    runSyr2kTask(f32, raw_tasks, index);
}

fn runSyr2kTaskF64(raw_tasks: *const anyopaque, index: usize) void {
    runSyr2kTask(f64, raw_tasks, index);
}

fn runSyr2kTaskC32(raw_tasks: *const anyopaque, index: usize) void {
    runSyr2kTask(scalar.ComplexF32, raw_tasks, index);
}

fn runSyr2kTaskC64(raw_tasks: *const anyopaque, index: usize) void {
    runSyr2kTask(scalar.ComplexF64, raw_tasks, index);
}

fn runParallelSyr2k(comptime T: type, tasks: []const Syr2kTask(T)) bool {
    const runner = if (T == f32)
        runSyr2kTaskF32
    else if (T == f64)
        runSyr2kTaskF64
    else if (T == scalar.ComplexF32)
        runSyr2kTaskC32
    else if (T == scalar.ComplexF64)
        runSyr2kTaskC64
    else
        @compileError("parallel SYR2K supports BLAS scalar types");
    return core_pool.runLowLatency(runner, @ptrCast(tasks.ptr), tasks.len);
}

// Keep the macOS candidate implementation out of the public fallback body.
noinline fn tryMacSyr2k(comptime T: type, uplo: Uplo, trans_: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) bool {
    const profile = structured_tuning.aarch64_macos_rank_tile_candidate;
    return blocked.trySyr2k(T, .{ .block_size = profile.block_size, .direct_rank_panels = profile.direct_panels, .packed_rank_panels = profile.packed_panels, .parallel_packed_rank_panels = profile.parallel_packed_panels }, uplo, trans_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc, herm);
}

pub fn syr2k(comptime T: type, uplo: Uplo, trans_: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, herm: bool) void {
    if (n_ <= 0) return;
    if (k_ == 0 or isZero(T, alpha)) {
        scaleStoredRank(T, uplo, toUsize(n_), beta, c, ldc, herm);
        return;
    }
    // Candidate: combine both products in one private tile, then apply beta
    // once while committing only the stored triangle.
    if (comptime builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos) {
        const profile = structured_tuning.aarch64_macos_rank_tile_candidate;
        if (n_ > 0 and k_ > 0 and profile.candidate(toUsize(n_), toUsize(k_), false) and (runtime.maxThreads() == 1 or profile.candidate(toUsize(n_), toUsize(k_), true))) {
            const capability = gemm_dispatch.activeCapability();
            if (capability == .aarch64_asimd_fma or capability == .aarch64_sme) {
                if (tryMacSyr2k(T, uplo, trans_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc, herm)) return;
            }
        }
    }
    // Use larger updates for multi-thread execution; retain small-kernel fallbacks
    // for modes that did not improve in the threaded comparison.
    if (comptime builtin.cpu.arch == .x86_64) {
        if (n_ >= 128 and k_ >= 64 and (runtime.maxThreads() == 1 or n_ >= 256) and gemm_dispatch.activeCapability() == .x86_64_avx2_fma) {
            if (blocked.trySyr2k(T, .{ .block_size = if (runtime.maxThreads() > 1) 256 else if (T == scalar.ComplexF32) 128 else 64 }, uplo, trans_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc, herm)) return;
        }
    }
    if (n_ <= 0) return;
    const n = toUsize(n_);
    const k = toUsize(k_);
    var tasks: [core_pool.max_tasks]Syr2kTask(T) = undefined;
    const work = n *| n *| k;
    const task_count = if (work >= 128 * 1024) @min(core_pool.taskCount(n, 8), 32) else 1;
    if (task_count > 1) {
        for (tasks[0..task_count], 0..) |*task, task_index| {
            task.* = .{
                .uplo = uplo,
                .trans = trans_,
                .n = n,
                .k = k,
                .alpha = alpha,
                .a = a,
                .lda = lda,
                .b = b,
                .ldb = ldb,
                .beta = beta,
                .c = c,
                .ldc = ldc,
                .herm = herm,
                .task_index = task_index,
                .task_count = task_count,
            };
        }
        if (runParallelSyr2k(T, tasks[0..task_count])) return;
    }
    runSyr2kColumns(T, .{
        .uplo = uplo,
        .trans = trans_,
        .n = n,
        .k = k,
        .alpha = alpha,
        .a = a,
        .lda = lda,
        .b = b,
        .ldb = ldb,
        .beta = beta,
        .c = c,
        .ldc = ldc,
        .herm = herm,
        .task_index = 0,
        .task_count = 1,
    });
}
