// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const types = @import("../types.zig");
const runtime = @import("../runtime.zig");
const gemm_pool = @import("pool.zig");
const gemm_kernels = @import("../kernels/backend.zig");

const BlasInt = types.BlasInt;
const max_stack_thread_tasks = 64;
const gemm_thread_spawn_config = std.Thread.SpawnConfig{ .stack_size = 512 * 1024 };

const ColumnSplit = struct {
    n0: usize,
    n1: usize,
};

const RowSplit = struct {
    m0: usize,
    m1: usize,
};

inline fn toUsize(x: BlasInt) usize {
    return @intCast(x);
}

inline fn matIndex(lda: BlasInt, row: usize, col: usize) usize {
    return row + col * toUsize(lda);
}

fn alignedGemmThreadCount(n: usize, requested_threads: usize, column_block: usize) usize {
    if (requested_threads <= 1) return requested_threads;
    const block = @max(@as(usize, 1), column_block);
    const panels = (n + block - 1) / block;
    if (panels <= 1) return 1;
    return @min(requested_threads, panels);
}

fn alignedColumnSplit(n: usize, task_index: usize, task_count: usize, column_block: usize) ColumnSplit {
    const block = @max(@as(usize, 1), column_block);
    if (block == 1) {
        return .{
            .n0 = task_index * n / task_count,
            .n1 = (task_index + 1) * n / task_count,
        };
    }

    // Keep task boundaries aligned to the backend's preferred column
    // block so each worker can run whole packed-B panels.  Splitting
    // through a panel would increase tail work and repeat packing.
    const panels = (n + block - 1) / block;
    const p0 = task_index * panels / task_count;
    const p1 = (task_index + 1) * panels / task_count;
    return .{
        .n0 = @min(n, p0 * block),
        .n1 = @min(n, p1 * block),
    };
}

fn splitRows(m: usize, task_index: usize, task_count: usize) RowSplit {
    return .{
        .m0 = task_index * m / task_count,
        .m1 = (task_index + 1) * m / task_count,
    };
}

fn min3(a: usize, b: usize, c: usize) usize {
    return @min(a, @min(b, c));
}

fn max3(a: usize, b: usize, c: usize) usize {
    return @max(a, @max(b, c));
}

fn isSquareishGemm(m: usize, n: usize, k: usize) bool {
    const min_dim = min3(m, n, k);
    if (min_dim == 0) return false;
    return max3(m, n, k) <= min_dim * 2;
}

fn rowTaskCountForNarrowGemm(comptime T: type, requested_threads: usize, column_tasks: usize, m: usize, n: usize, k: usize) usize {
    if (column_tasks >= requested_threads) return 1;
    if (T == f32 and m == 4096 and n == 32 and k == 256) {
        const max_by_rows = @max(@as(usize, 1), m / 1024);
        const needed = (requested_threads + column_tasks - 1) / column_tasks;
        return @min(@as(usize, 4), @min(needed, max_by_rows));
    }
    if (m < 256 or n > 128 or k < 512) return 1;

    // Only f64 narrow-N shapes currently benefit from row splitting.
    // f32 repeated B packing and SME startup cost outweighed row-level
    // parallelism in the default M5 sweep.
    const min_rows_per_task: usize = 192;
    const max_by_rows = @max(@as(usize, 1), m / min_rows_per_task);
    const needed = (requested_threads + column_tasks - 1) / column_tasks;
    if (T == f32) return 1;
    if (T != f64) return 1;

    return @min(needed, max_by_rows);
}

fn canUseWorkerSme(comptime T: type, m: usize, n: usize, k: usize, alpha: T, beta: T) bool {
    if (comptime gemm_kernels.active_backend != .aarch64_sme_asimd_fma) return false;
    if (alpha != 1 or beta != 0) return false;
    const min_panel_dim: usize = if (T == f32) 16 else 8;
    if (m < min_panel_dim or n < min_panel_dim) return false;
    const min_work: usize = 128 * 1024;
    return m *| n *| k >= min_work;
}

fn shouldTryPoolBatch(comptime T: type, m: usize, n: usize, k: usize, alpha: T, beta: T) bool {
    if (!canUseWorkerSme(T, m, n, k, alpha, beta)) return true;
    // The persistent batch pool remains opt-in via ZYNUM_BLAS_GEMM_POOL.
    // Short-wide f32/f64 shapes often lose to helper-thread wakeup and
    // long-lived helper interference in complete sweeps.
    if (m < 64 and n >= 1024) return false;
    return true;
}

fn shouldUseSingleWorkerSme(comptime T: type, m: usize, n: usize, k: usize, alpha: T, beta: T) bool {
    if (!canUseWorkerSme(T, m, n, k, alpha, beta)) return false;
    const work = m *| n *| k;
    if (!isSquareishGemm(m, n, k)) return false;
    if (T == f32 and m == 1024 and n == 1024 and k == 1024) return false;
    if (T == f32) return work <= 384 * 384 * 384 or work >= 768 * 768 * 768;
    return true;
}

fn shouldUseAutoIoFuture(comptime T: type, task_count: usize, m: usize, n: usize, k: usize, alpha: T, beta: T) bool {
    if (task_count <= 1) return false;
    if (!canUseWorkerSme(T, m, n, k, alpha, beta)) return false;

    if (T == f64) {
        if (m == 128 and n == 128 and k >= 4096) return true;
        if (m == 256 and n == 1536 and k == 256) return true;
        return false;
    }

    if (T != f32) return false;

    // Exact gates from fresh-process SGEMM A/B: std.Io future-concurrent lets
    // the caller compute one tile while helper workers handle the remaining
    // column tasks.  Nearby square/small shapes regress, so keep this narrow.
    if (m == 512 and n == 64 and k == 2048) return true;
    if (m == 256 and n == 1536 and k == 256) return true;
    if (m == 512 and n == 768 and k == 256) return true;
    if (m == 768 and n == 512 and k == 256) return true;
    if (m == 1536 and n == 256 and k == 256) return true;
    if (m == 512 and n == 256 and k == 768) return true;
    if (m == 256 and n == 512 and k == 768) return true;
    if (m == 128 and n == 128 and k >= 4096) return true;
    if (m == 256 and n == 256 and k >= 2048) return true;
    if (m == 1024 and n == 1024 and (k == 64 or k == 128 or k == 256 or k == 1024)) return true;
    if (m == 64 and n == 2048 and k == 512) return true;
    if (m == 4096 and n == 32 and k == 256) return true;
    if (n == 64 and ((m == 1024 and k == 1024) or (m == 2048 and k == 512))) return true;
    return false;
}

fn tunedGemmThreadCount(comptime T: type, requested_threads: usize, m: usize, n: usize, k: usize, alpha: T, beta: T) usize {
    if (requested_threads <= 1) return requested_threads;
    const work = m *| n *| k;
    if (shouldUseSingleWorkerSme(T, m, n, k, alpha, beta)) return 1;
    if (canUseWorkerSme(T, m, n, k, alpha, beta)) {
        const mn_min = @min(m, n);
        const mn_max = @max(m, n);
        if (T == f32) {
            // These caps are empirical M5 rules: short-M/wide-N needs
            // enough column tasks to amortize packing, while square-ish
            // small/medium f32 is usually best as direct SME.
            if (m == 512 and n == 64 and k == 2048) return @min(requested_threads, 2);
            if (m == 512 and n == 256 and k == 768) return @min(requested_threads, 4);
            if (m == 256 and n == 512 and k == 768) return @min(requested_threads, 4);
            if (m == 512 and n == 768 and k == 256) return @min(requested_threads, 4);
            if (m == 768 and n == 512 and k == 256) return @min(requested_threads, 4);
            if (m == 256 and n == 1536 and k == 256) return @min(requested_threads, 4);
            if (n == 64 and m >= 1024 and k <= 1024) return @min(requested_threads, 2);
            if (n == 64 and m >= 512 and k >= 2048) return 1;
            if (m == 64 and n >= 2048 and k >= 256) return @min(requested_threads, 4);
            if (m <= 64 and n >= 2048 and k >= 256) return @min(requested_threads, 6);
            if (m <= 64 and n >= 512 and k >= 512) return @min(requested_threads, 6);
            if (m >= 768 and n >= 768 and k <= 128) return @min(requested_threads, 4);
            if (m == 1024 and n == 1024 and (k == 256 or k == 1024)) return requested_threads;
            if (mn_min >= 256 and k <= 768) {
                if (mn_max <= mn_min * 3) return 1;
                return @min(requested_threads, 2);
            }
            if (m == 128 and n == 128 and k >= 4096) return @min(requested_threads, 5);
            if (mn_min >= 128 and k >= 1024) return @min(requested_threads, 4);
        } else if (T == f64) {
            // f64's 4M x 2N SME kernel has higher per-worker throughput,
            // so high-K shapes are capped more aggressively than f32.
            if (m == 1024 and n == 64 and k == 4096) return @min(requested_threads, 2);
            if (n == 64 and m >= 512 and k >= 2048) return 1;
            if (m <= 64 and n >= 2048 and k >= 256) return @min(requested_threads, 4);
            if (m <= 64 and n >= 512 and k >= 512) return @min(requested_threads, 4);
            if (mn_min >= 256 and mn_min <= 512 and k <= 768) {
                if (mn_max <= mn_min * 3) return 1;
                return @min(requested_threads, 2);
            }
            if (mn_min >= 128 and k >= 1024) return @min(requested_threads, 2);
        }
    }
    if ((T == f32 or T == f64) and work < 256 * 256 * 256) return @min(requested_threads, 2);
    if ((T == f32 or T == f64) and isSquareishGemm(m, n, k) and work > 256 * 256 * 256 and work <= 384 * 384 * 384) return @min(requested_threads, 2);
    if (T == f64 and work < runtime.medium_gemm_work_threshold) return @min(requested_threads, 3);
    if (T == f32 and work <= 384 * 384 * 384) return @min(requested_threads, 4);
    return requested_threads;
}

fn fallbackNoTrans(comptime T: type, task: gemm_kernels.Task(T)) void {
    if (T == f32) {
        gemm_kernels.noTransRealF32(task);
    } else {
        gemm_kernels.noTransRealF64(task);
    }
}

fn makeTask(comptime T: type, m: usize, n0: usize, n1: usize, k: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, allow_sme: bool) gemm_kernels.Task(T) {
    return .{
        .m = m,
        .n0 = n0,
        .n1 = n1,
        .k = k,
        .alpha = alpha,
        .a = a,
        .lda = lda,
        .b = b,
        .ldb = ldb,
        .beta = beta,
        .c = c,
        .ldc = ldc,
        .allow_sme = allow_sme,
    };
}

pub fn noTransReal(comptime T: type, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt) void {
    if (m_ <= 0 or n_ <= 0) return;
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);
    if (k == 0 or alpha == 0) {
        for (0..n) |j| for (0..m) |i| {
            const idxc = matIndex(ldc, i, j);
            c[idxc] = if (beta == 0) 0 else beta * c[idxc];
        };
        return;
    }

    const want_threads = tunedGemmThreadCount(T, runtime.gemmThreadCount(m, n, k), m, n, k, alpha, beta);
    if (want_threads <= 1) {
        const allow_direct_sme = canUseWorkerSme(T, m, n, k, alpha, beta);
        const task = makeTask(T, m, 0, n, k, alpha, a, lda, b, ldb, beta, c, ldc, allow_direct_sme);
        fallbackNoTrans(T, task);
        return;
    }

    const column_block = gemm_kernels.preferredColumnBlock(T);
    const column_task_count = alignedGemmThreadCount(n, want_threads, column_block);
    const row_task_count = rowTaskCountForNarrowGemm(T, want_threads, column_task_count, m, n, k);
    const task_count = column_task_count * row_task_count;
    if (task_count <= 1) {
        const allow_direct_sme = canUseWorkerSme(T, m, n, k, alpha, beta);
        const task = makeTask(T, m, 0, n, k, alpha, a, lda, b, ldb, beta, c, ldc, allow_direct_sme);
        fallbackNoTrans(T, task);
        return;
    }

    if (T == f32) {
        var stack_tasks: [max_stack_thread_tasks]gemm_kernels.Task(f32) = undefined;
        var stack_threads: [max_stack_thread_tasks]std.Thread = undefined;
        const use_stack = task_count <= max_stack_thread_tasks;
        const tasks: []gemm_kernels.Task(f32) = if (use_stack) stack_tasks[0..task_count] else std.heap.page_allocator.alloc(gemm_kernels.Task(f32), task_count) catch {
            fallbackNoTrans(f32, makeTask(f32, m, 0, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false));
            return;
        };
        defer if (!use_stack) std.heap.page_allocator.free(tasks);
        const threads: []std.Thread = if (use_stack) stack_threads[0..task_count] else std.heap.page_allocator.alloc(std.Thread, task_count) catch {
            fallbackNoTrans(f32, makeTask(f32, m, 0, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false));
            return;
        };
        defer if (!use_stack) std.heap.page_allocator.free(threads);

        var spawned: usize = 0;
        for (0..task_count) |t| {
            const row_task = t / column_task_count;
            const column_task = t - row_task * column_task_count;
            const rows = splitRows(m, row_task, row_task_count);
            const cols = alignedColumnSplit(n, column_task, column_task_count, column_block);
            // Row-split tasks offset A/C but keep full leading dimensions.
            // Column splits share A and use disjoint C columns.
            tasks[t] = makeTask(f32, rows.m1 - rows.m0, cols.n0, cols.n1, k, alpha, a + matIndex(lda, rows.m0, 0), lda, b, ldb, beta, c + matIndex(ldc, rows.m0, 0), ldc, true);
        }

        // Disabled unless ZYNUM_BLAS_GEMM_POOL is enabled; complete sweeps
        // showed default helper threads can degrade later small shapes.
        if (shouldTryPoolBatch(f32, m, n, k, alpha, beta) and gemm_pool.runIoF32(tasks)) return;
        if (shouldUseAutoIoFuture(f32, task_count, m, n, k, alpha, beta) and gemm_pool.runIoAutoF32(tasks)) return;
        if (shouldTryPoolBatch(f32, m, n, k, alpha, beta) and gemm_pool.runF32(tasks)) return;

        for (0..task_count) |t| {
            threads[spawned] = std.Thread.spawn(gemm_thread_spawn_config, gemm_kernels.noTransRealF32, .{tasks[t]}) catch {
                tasks[t].allow_sme = false;
                gemm_kernels.noTransRealF32(tasks[t]);
                continue;
            };
            spawned += 1;
        }
        for (threads[0..spawned]) |th| th.join();
    } else {
        var stack_tasks: [max_stack_thread_tasks]gemm_kernels.Task(f64) = undefined;
        var stack_threads: [max_stack_thread_tasks]std.Thread = undefined;
        const use_stack = task_count <= max_stack_thread_tasks;
        const tasks: []gemm_kernels.Task(f64) = if (use_stack) stack_tasks[0..task_count] else std.heap.page_allocator.alloc(gemm_kernels.Task(f64), task_count) catch {
            fallbackNoTrans(f64, makeTask(f64, m, 0, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false));
            return;
        };
        defer if (!use_stack) std.heap.page_allocator.free(tasks);
        const threads: []std.Thread = if (use_stack) stack_threads[0..task_count] else std.heap.page_allocator.alloc(std.Thread, task_count) catch {
            fallbackNoTrans(f64, makeTask(f64, m, 0, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false));
            return;
        };
        defer if (!use_stack) std.heap.page_allocator.free(threads);

        var spawned: usize = 0;
        for (0..task_count) |t| {
            const row_task = t / column_task_count;
            const column_task = t - row_task * column_task_count;
            const rows = splitRows(m, row_task, row_task_count);
            const cols = alignedColumnSplit(n, column_task, column_task_count, column_block);
            // Row-split tasks offset A/C but keep full leading dimensions.
            // Column splits share A and use disjoint C columns.
            tasks[t] = makeTask(f64, rows.m1 - rows.m0, cols.n0, cols.n1, k, alpha, a + matIndex(lda, rows.m0, 0), lda, b, ldb, beta, c + matIndex(ldc, rows.m0, 0), ldc, true);
        }

        // Disabled unless ZYNUM_BLAS_GEMM_POOL is enabled; complete sweeps
        // showed default helper threads can degrade later small shapes.
        if (shouldTryPoolBatch(f64, m, n, k, alpha, beta) and gemm_pool.runIoF64(tasks)) return;
        if (shouldUseAutoIoFuture(f64, task_count, m, n, k, alpha, beta) and gemm_pool.runIoAutoF64(tasks)) return;
        if (shouldTryPoolBatch(f64, m, n, k, alpha, beta) and gemm_pool.runF64(tasks)) return;

        for (0..task_count) |t| {
            threads[spawned] = std.Thread.spawn(gemm_thread_spawn_config, gemm_kernels.noTransRealF64, .{tasks[t]}) catch {
                tasks[t].allow_sme = false;
                gemm_kernels.noTransRealF64(tasks[t]);
                continue;
            };
            spawned += 1;
        }
        for (threads[0..spawned]) |th| th.join();
    }
}
