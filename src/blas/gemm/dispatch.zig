// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const types = @import("../types.zig");
const runtime = @import("../runtime.zig");
const gemm_pool = @import("pool.zig");
const gemm_kernels = @import("../kernels/matrix_matrix.zig");
const catalog = gemm_kernels.catalog;
const tuning = @import("../kernels/matrix_matrix/tuning.zig");

const BlasInt = types.BlasInt;
const max_stack_thread_tasks = 64;

const ColumnSplit = struct {
    n0: usize,
    n1: usize,
};

const RowSplit = struct {
    m0: usize,
    m1: usize,
};

const Plan = struct {
    kernel: catalog.Descriptor,
    thread_count: usize,
    column_tasks: usize,
    row_tasks: usize,
    column_block: usize,
    allow_direct_kernel: bool,
    use_thread_pool: bool,
    implementation: gemm_kernels.Implementation,
    execution: gemm_kernels.ExecutionPlan,

    fn taskCount(self: Plan) usize {
        return self.column_tasks * self.row_tasks;
    }
};

inline fn toUsize(x: BlasInt) usize {
    return @intCast(x);
}

inline fn matIndex(lda: BlasInt, row: usize, col: usize) usize {
    return row + col * toUsize(lda);
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

fn ceilDiv(a: usize, b: usize) usize {
    if (b == 0) return 0;
    return (a + b - 1) / b;
}

fn columnBlockForPlan(desc: catalog.Descriptor, execution: gemm_kernels.ExecutionPlan) usize {
    const amx_panel = tuning.amxNPanel(execution.amx);
    return @max(@as(usize, 1), if (amx_panel != 0) amx_panel else desc.tile.n_panel);
}

fn desiredThreadCount(comptime T: type, desc: catalog.Descriptor, requested_threads: usize, m: usize, n: usize, k: usize, allow_direct_kernel: bool) usize {
    if (requested_threads <= 1) return requested_threads;
    const work = m *| n *| k;

    const shape: tuning.Shape = .{ .m = m, .n = n, .k = k };
    const min_dim = tuning.min3(m, n, k);
    const squareish = tuning.isSquareish(shape);
    if (allow_direct_kernel and squareish and T == f32 and work <= 128 * 128 * 128) return 1;
    const vector_edge = (m == 1 or n == 1) and k >= 128 and work >= 128 * 1024;
    const parallel_threshold: usize = if (vector_edge)
        128 * 1024
    else if (allow_direct_kernel and squareish)
        96 * 96 * 96
    else
        runtime.gemm_parallel_work_threshold;
    if (work < parallel_threshold) return 1;
    if (vector_edge) return requested_threads;

    var threads = requested_threads;
    const small_parallel = 256 * 256 * 256;
    const medium_parallel = 384 * 384 * 384;
    if (allow_direct_kernel and squareish and T == f64) {
        if (work < small_parallel) threads = @min(threads, 6);
    } else if (work < small_parallel) {
        threads = @min(threads, 2);
    } else if (work < medium_parallel) {
        threads = @min(threads, 4);
    }

    if (min_dim <= desc.tile.n_panel and k <= 128) {
        threads = @min(threads, 2);
    } else if (tuning.isNarrowN(desc, n) and m >= 512 and k >= desc.bounds.min_k_block * 8) {
        threads = @min(threads, 4);
    } else if (min_dim >= 256 and squareish and k <= 256) {
        threads = @min(threads, 4);
    }

    return @max(@as(usize, 1), threads);
}

fn requestedThreadCountForPlan(m: usize, n: usize, k: usize) usize {
    const requested = runtime.gemmThreadCount(m, n, k);
    if (requested > 1) return requested;
    if ((m == 1 or n == 1) and k >= 128 and m *| n *| k >= 128 * 1024) {
        return runtime.defaultGemmThreadLimit();
    }
    return requested;
}

fn rowMinBlock(desc: catalog.Descriptor, m: usize, k: usize) usize {
    const by_register = @max(desc.bounds.min_m_block * 16, desc.tile.register_m * 8);
    const by_k = if (k >= 1024) @as(usize, 192) else @as(usize, 256);
    return @min(m, @max(by_register, by_k));
}

fn rowTaskCount(desc: catalog.Descriptor, desired_threads: usize, column_tasks: usize, m: usize, n: usize, k: usize) usize {
    if (column_tasks == 0 or column_tasks >= desired_threads) return 1;
    if (m < desc.bounds.min_m_block * 4 or k < desc.bounds.min_k_block * 4) return 1;

    const max_rows_by_threads = desired_threads / column_tasks;
    if (max_rows_by_threads <= 1) return 1;

    const min_rows = rowMinBlock(desc, m, k);
    if (min_rows == 0 or m < min_rows * 2) return 1;

    var max_by_rows = @max(@as(usize, 1), m / min_rows);
    if (!tuning.isNarrowN(desc, n)) {
        max_by_rows = @min(max_by_rows, @as(usize, 2));
    }
    return @min(max_rows_by_threads, max_by_rows);
}

fn selectNoTransReal(comptime T: type, m: usize, n: usize, k: usize, alpha: T, beta: T, requested_threads: usize) Plan {
    const shape: tuning.Shape = .{ .m = m, .n = n, .k = k };
    const desc = tuning.select(T, gemm_kernels.candidates(T), shape, alpha, beta, requested_threads);
    const allow_direct = tuning.directKernelAllowed(T, desc, shape, alpha, beta);
    const thread_count = desiredThreadCount(T, desc, requested_threads, m, n, k, allow_direct);
    const execution = tuning.executionPlan(T, desc, shape, thread_count, runtime.performanceL2Bytes());
    const column_block = columnBlockForPlan(desc, execution);
    const panels = @max(@as(usize, 1), ceilDiv(n, column_block));
    const column_tasks = if (thread_count <= 1) @as(usize, 1) else @min(thread_count, panels);
    const rows = rowTaskCount(desc, thread_count, column_tasks, m, n, k);
    const total_tasks = column_tasks * rows;
    return .{
        .kernel = desc,
        .thread_count = thread_count,
        .column_tasks = column_tasks,
        .row_tasks = rows,
        .column_block = column_block,
        .allow_direct_kernel = allow_direct,
        .use_thread_pool = total_tasks > 1 and thread_count > 1,
        .implementation = gemm_kernels.implementationFor(desc),
        .execution = execution,
    };
}

fn fallbackNoTrans(comptime T: type, task: gemm_kernels.Task(T)) void {
    if (T == f32) {
        gemm_kernels.noTransRealF32(task);
    } else if (T == f64) {
        gemm_kernels.noTransRealF64(task);
    } else {
        @compileError("GEMM dispatch supports f32 and f64");
    }
}

fn makeTask(comptime T: type, plan: Plan, m: usize, n0: usize, n1: usize, k: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt) gemm_kernels.Task(T) {
    const task_shape: tuning.Shape = .{ .m = m, .n = n1 - n0, .k = k };
    var execution = tuning.executionPlan(T, plan.kernel, task_shape, plan.thread_count, runtime.performanceL2Bytes());
    if (execution.amx == .none and tuning.amxKernelCompatible(T, plan.execution.amx, task_shape)) {
        execution.amx = plan.execution.amx;
        execution.amx_pack = plan.execution.amx_pack;
    }
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
        .allow_sme = plan.allow_direct_kernel,
        .kernel = plan.kernel.kernel,
        .implementation = plan.implementation,
        .execution = execution,
    };
}

fn runPool(comptime T: type, tasks: []const gemm_kernels.Task(T)) bool {
    if (T == f32) return gemm_pool.runF32(tasks);
    if (T == f64) return gemm_pool.runF64(tasks);
    @compileError("GEMM dispatch supports f32 and f64");
}

fn fallbackWholeTask(comptime T: type, plan: Plan, m: usize, n: usize, k: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt) void {
    fallbackNoTrans(T, makeTask(T, plan, m, 0, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

fn runPlannedTasks(comptime T: type, plan: Plan, m: usize, n: usize, k: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt) void {
    const column_task_count = plan.column_tasks;
    const row_task_count = plan.row_tasks;
    const task_count = column_task_count * row_task_count;
    if (task_count <= 1) {
        fallbackWholeTask(T, plan, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }

    var stack_tasks: [max_stack_thread_tasks]gemm_kernels.Task(T) = undefined;
    const use_stack = task_count <= max_stack_thread_tasks;
    const tasks: []gemm_kernels.Task(T) = if (use_stack) stack_tasks[0..task_count] else std.heap.page_allocator.alloc(gemm_kernels.Task(T), task_count) catch {
        fallbackWholeTask(T, plan, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    };
    defer if (!use_stack) std.heap.page_allocator.free(tasks);

    for (0..task_count) |t| {
        const row_task = t / column_task_count;
        const column_task = t - row_task * column_task_count;
        const rows = splitRows(m, row_task, row_task_count);
        const cols = alignedColumnSplit(n, column_task, column_task_count, plan.column_block);
        // Row-split tasks offset A/C but keep full leading dimensions.
        // Column splits share A and use disjoint C columns.
        tasks[t] = makeTask(T, plan, rows.m1 - rows.m0, cols.n0, cols.n1, k, alpha, a + matIndex(lda, rows.m0, 0), lda, b, ldb, beta, c + matIndex(ldc, rows.m0, 0), ldc);
    }

    if (plan.use_thread_pool and runPool(T, tasks)) return;
    fallbackWholeTask(T, plan, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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
    if (comptime T == f32) {
        if (gemm_kernels.tryNoTransRealF32Fast(m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return;
    }

    const plan = selectNoTransReal(T, m, n, k, alpha, beta, requestedThreadCountForPlan(m, n, k));
    runPlannedTasks(T, plan, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
