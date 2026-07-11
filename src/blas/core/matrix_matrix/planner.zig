// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const types = @import("../../types.zig");
const runtime = @import("../../runtime.zig");
const core_pool = @import("../execution/thread_pool.zig");
const gemm_kernels = @import("../../kernels/dispatch/matrix_matrix.zig");
const catalog = gemm_kernels.catalog;
const packing = @import("../../kernels/shared/matrix_matrix/packing.zig");
const tuning = @import("../../kernels/shared/matrix_matrix/tuning.zig");

const BlasInt = types.BlasInt;
const parallel_work_threshold: usize = 192 * 192 * 192;
const max_stack_tasks = 64;

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
    use_parallel_tasks: bool,
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
    if (T == f32 and m == 128 and n == 128 and k >= 2048) return @min(requested_threads, 8);
    if (T == f64 and allow_direct_kernel and m >= 512 and n >= 512 and k <= 256) return @min(requested_threads, 8);
    if (T == f32 and allow_direct_kernel and m >= 1024 and m < 4096 and n <= 32 and k >= 128 and k <= 512) return 1;
    if (allow_direct_kernel and squareish and T == f32 and work <= 128 * 128 * 128) return 1;
    const vector_edge = (m == 1 or n == 1) and k >= 128 and work >= 128 * 1024;
    const parallel_threshold: usize = if (vector_edge)
        128 * 1024
    else if (allow_direct_kernel and squareish)
        96 * 96 * 96
    else
        parallel_work_threshold;
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

    if (T == f64 and allow_direct_kernel and !squareish and m >= 256 and n >= 256 and k >= 512 and k <= 1024) {
        threads = @min(threads, 8);
    }

    if (min_dim <= desc.tile.n_panel and k <= 128) {
        threads = @min(threads, 2);
    } else if (tuning.isNarrowN(desc, n) and m >= 512 and k >= desc.bounds.min_k_block * 8) {
        const narrow_cap: usize = if (T == f32 and allow_direct_kernel and n <= desc.tile.n_panel * 2 and k >= 512) 2 else 4;
        threads = @min(threads, narrow_cap);
    } else if (min_dim >= 256 and squareish and k <= 256 and !(T == f64 and allow_direct_kernel and m >= 512 and n >= 512)) {
        threads = @min(threads, 4);
    }

    return @max(@as(usize, 1), @min(threads, core_pool.max_tasks));
}

fn baseThreadCountForPlan(m: usize, n: usize, k: usize) usize {
    if (m == 0 or n < 2 or k == 0) return 1;

    const work = m *| n *| k;
    if (work < parallel_work_threshold) return 1;

    const limit = runtime.maxThreads();
    if (limit <= 1) return 1;
    return @max(@as(usize, 1), @min(limit, n));
}

fn requestedThreadCountForPlan(m: usize, n: usize, k: usize) usize {
    const requested = baseThreadCountForPlan(m, n, k);
    if (requested > 1) return @min(requested, core_pool.max_tasks);
    if ((m == 1 or n == 1) and k >= 128 and m *| n *| k >= 128 * 1024) {
        return @min(runtime.maxThreads(), core_pool.max_tasks);
    }
    return requested;
}

fn forceSingleThreadPlan(comptime T: type, m: usize, n: usize, k: usize, alpha: T, beta: T) bool {
    if (comptime switch (gemm_kernels.active_backend) {
        .x86_64_sse2, .x86_64_avx, .x86_64_avx2, .x86_64_avx512f => true,
        else => false,
    }) return false;
    return T == f32 and alpha == 1 and beta == 0 and m >= 1024 and m < 4096 and n <= 32 and k >= 128 and k <= 512;
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

    // Row splits duplicate B-panel packing in each row task. Keep high-K,
    // narrow-N packed kernels column-only until shared packed-B workspaces exist.
    if (desc.pack.kind != .none and tuning.isNarrowN(desc, n) and k >= 1024) return 1;

    var max_by_rows = @max(@as(usize, 1), m / min_rows);
    if (!tuning.isNarrowN(desc, n)) {
        max_by_rows = @min(max_by_rows, @as(usize, 2));
    }
    return @min(max_rows_by_threads, max_by_rows);
}

fn planForDescriptor(comptime T: type, desc: catalog.Descriptor, m: usize, n: usize, k: usize, alpha: T, beta: T, requested_threads: usize) Plan {
    const shape: tuning.Shape = .{ .m = m, .n = n, .k = k };
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
        .use_parallel_tasks = total_tasks > 1 and thread_count > 1,
        .implementation = gemm_kernels.implementationFor(desc),
        .execution = execution,
    };
}

fn selectNoTransReal(comptime T: type, m: usize, n: usize, k: usize, alpha: T, beta: T, requested_threads: usize) Plan {
    const shape: tuning.Shape = .{ .m = m, .n = n, .k = k };
    const desc = tuning.select(T, gemm_kernels.candidates(T), shape, alpha, beta, requested_threads);
    return planForDescriptor(T, desc, m, n, k, alpha, beta, requested_threads);
}

fn selectTransposedBReal(comptime T: type, m: usize, n: usize, k: usize, alpha: T, beta: T, requested_threads: usize) ?Plan {
    const shape: tuning.Shape = .{ .m = m, .n = n, .k = k };
    const candidates = gemm_kernels.candidates(T);
    var best: ?catalog.Descriptor = null;
    var best_score: i64 = std.math.minInt(i64);
    var index: usize = 0;
    while (index < candidates.len) : (index += 1) {
        const desc = candidates.at(index);
        const supported = desc.family == .packed_simd or (T == f32 and desc.family == .streaming_matrix);
        if (!supported) continue;
        const score = tuning.score(T, desc, shape, alpha, beta, requested_threads);
        if (best == null or score > best_score) {
            best = desc;
            best_score = score;
        }
    }
    const desc = best orelse return null;
    return planForDescriptor(T, desc, m, n, k, alpha, beta, requested_threads);
}

fn fallbackNoTrans(comptime T: type, task: gemm_kernels.Task(T)) void {
    if (T == f32) {
        gemm_kernels.noTransRealF32(task);
    } else if (T == f64) {
        gemm_kernels.noTransRealF64(task);
    } else {
        @compileError("GEMM planner supports f32 and f64");
    }
}

fn makeTask(comptime T: type, plan: Plan, m: usize, n0: usize, n1: usize, k: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, b_layout: gemm_kernels.BLayout, beta: T, c: [*]T, ldc: BlasInt) gemm_kernels.Task(T) {
    const task_shape: tuning.Shape = .{ .m = m, .n = n1 - n0, .k = k };
    var execution = tuning.executionPlan(T, plan.kernel, task_shape, plan.thread_count, runtime.performanceL2Bytes());
    if (execution.amx == .none and tuning.amxKernelCompatible(T, plan.execution.amx, task_shape)) {
        execution.amx = plan.execution.amx;
        execution.amx_pack = plan.execution.amx_pack;
    }
    if (b_layout == .trans) {
        // The SME kernels consume packed B panels and can therefore handle
        // op(B) == B^T.  f32 AMX has a layout-specific contiguous packer,
        // while f64 AMX and the f32 transpose4 pack remain NN-only here.
        if (T == f64) {
            execution.amx = .none;
            execution.amx_pack = .{};
        }
        execution.amx_partial_n16 = false;
        execution.b_pack = .dynamic;
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
        .b_layout = b_layout,
        .beta = beta,
        .c = c,
        .ldc = ldc,
        .allow_sme = plan.allow_direct_kernel,
        .kernel = plan.kernel.kernel,
        .implementation = plan.implementation,
        .execution = execution,
    };
}

fn runSharedTasks(comptime T: type, tasks: []const gemm_kernels.Task(T)) bool {
    if (T == f32) return core_pool.runTyped(gemm_kernels.Task(f32), gemm_kernels.noTransRealF32, tasks);
    if (T == f64) return core_pool.runTyped(gemm_kernels.Task(f64), gemm_kernels.noTransRealF64, tasks);
    @compileError("GEMM planner supports f32 and f64");
}

fn fallbackWholeTask(comptime T: type, plan: Plan, m: usize, n: usize, k: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, b_layout: gemm_kernels.BLayout, beta: T, c: [*]T, ldc: BlasInt) void {
    fallbackNoTrans(T, makeTask(T, plan, m, 0, n, k, alpha, a, lda, b, ldb, b_layout, beta, c, ldc));
}

fn runPlannedTasks(comptime T: type, plan: Plan, m: usize, n: usize, k: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, b_layout: gemm_kernels.BLayout, beta: T, c: [*]T, ldc: BlasInt) void {
    const column_task_count = plan.column_tasks;
    const row_task_count = plan.row_tasks;
    const task_count = column_task_count * row_task_count;
    if (task_count <= 1) {
        fallbackWholeTask(T, plan, m, n, k, alpha, a, lda, b, ldb, b_layout, beta, c, ldc);
        return;
    }

    var stack_tasks: [max_stack_tasks]gemm_kernels.Task(T) = undefined;
    const use_stack = task_count <= max_stack_tasks;
    const tasks: []gemm_kernels.Task(T) = if (use_stack) stack_tasks[0..task_count] else std.heap.page_allocator.alloc(gemm_kernels.Task(T), task_count) catch {
        fallbackWholeTask(T, plan, m, n, k, alpha, a, lda, b, ldb, b_layout, beta, c, ldc);
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
        tasks[t] = makeTask(T, plan, rows.m1 - rows.m0, cols.n0, cols.n1, k, alpha, a + matIndex(lda, rows.m0, 0), lda, b, ldb, b_layout, beta, c + matIndex(ldc, rows.m0, 0), ldc);
    }

    if (plan.use_parallel_tasks and runSharedTasks(T, tasks)) return;
    fallbackWholeTask(T, plan, m, n, k, alpha, a, lda, b, ldb, b_layout, beta, c, ldc);
}

fn noTransRealWithBLayout(comptime T: type, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, b_layout: gemm_kernels.BLayout, beta: T, c: [*]T, ldc: BlasInt, require_packed_simd: bool) bool {
    if (m_ <= 0 or n_ <= 0) return true;
    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);
    if (k == 0 or alpha == 0) {
        for (0..n) |j| for (0..m) |i| {
            const idxc = matIndex(ldc, i, j);
            c[idxc] = if (beta == 0) 0 else beta * c[idxc];
        };
        return true;
    }
    if (comptime T == f32) {
        if (b_layout == .no_trans and gemm_kernels.tryNoTransRealF32Fast(m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc)) return true;
    }

    const requested_threads = if (forceSingleThreadPlan(T, m, n, k, alpha, beta)) @as(usize, 1) else requestedThreadCountForPlan(m, n, k);
    const plan = if (require_packed_simd)
        selectTransposedBReal(T, m, n, k, alpha, beta, requested_threads) orelse return false
    else
        selectNoTransReal(T, m, n, k, alpha, beta, requested_threads);
    runPlannedTasks(T, plan, m, n, k, alpha, a, lda, b, ldb, b_layout, beta, c, ldc);
    return true;
}

pub fn noTransReal(comptime T: type, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt) void {
    _ = noTransRealWithBLayout(T, m_, n_, k_, alpha, a, lda, b, ldb, .no_trans, beta, c, ldc, false);
}

pub fn noTransTransposedBReal(comptime T: type, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt) bool {
    return noTransRealWithBLayout(T, m_, n_, k_, alpha, a, lda, b, ldb, .trans, beta, c, ldc, true);
}

pub fn transposedAReal(comptime T: type, m_: BlasInt, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, b_layout: gemm_kernels.BLayout, beta: T, c: [*]T, ldc: BlasInt) bool {
    if (m_ <= 0 or n_ <= 0) return true;
    if (k_ <= 0 or alpha == 0) return false;

    const m = toUsize(m_);
    const n = toUsize(n_);
    const k = toUsize(k_);
    if (m == 1 or n == 1) return false;
    if (m <= 33 and n <= 33 and k <= 33) return false;
    const requested_threads = if (forceSingleThreadPlan(T, m, n, k, alpha, beta)) @as(usize, 1) else requestedThreadCountForPlan(m, n, k);
    const plan = if (b_layout == .trans)
        selectTransposedBReal(T, m, n, k, alpha, beta, requested_threads) orelse return false
    else
        selectNoTransReal(T, m, n, k, alpha, beta, requested_threads);
    if (plan.kernel.family == .generic) return false;

    const packed_len = std.math.mul(usize, m, k) catch return false;
    const a_pack = std.heap.c_allocator.alloc(T, packed_len) catch return false;
    defer std.heap.c_allocator.free(a_pack);

    packing.packTransposedA(T, m, k, a, lda, a_pack);
    runPlannedTasks(T, plan, m, n, k, alpha, a_pack.ptr, m_, b, ldb, b_layout, beta, c, ldc);
    return true;
}
