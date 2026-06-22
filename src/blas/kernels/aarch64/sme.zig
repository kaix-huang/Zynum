// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");
const gemm_task = @import("../gemm_task.zig");
const asimd = @import("asimd.zig");
const amx = @import("amx_gemm.zig");
const features = @import("features.zig");
const runtime = @import("../../runtime.zig");
const sve2 = @import("sve2.zig");

const matIndex = gemm_task.matIndex;
const sme_f32_thread_panel_count = 2;
const sme_f32_panel_batch_count = 8;
const sme_f64_thread_panel_count = 2;
const sme_f64_panel_batch_count = 8;
const sme_m_tile_count = 4;
const sme_f32_min_work = 128 * 1024;
const sme_f64_min_work = 128 * 1024;
const sme_stack_pack_bytes = 256 * 1024;
const sme_stack_pack_f32_elems = sme_stack_pack_bytes / @sizeOf(f32);
const sme_stack_pack_f64_elems = sme_stack_pack_bytes / @sizeOf(f64);
const max_sme_f32_tile = 256;
const max_sme_f64_tile = 256;

pub const enabled: bool = features.has_sme;

pub const supports_f64_accumulate: bool = features.has_sme_f64f64;

pub const supports_sme2: bool = features.has_sme2;
pub const supports_sme2p1: bool = features.has_sme2p1;
pub const supports_sme_tmop: bool = features.has_sme_tmop;

var amx_env_state = std.atomic.Value(u8).init(0);

fn amxMode() u8 {
    switch (amx_env_state.load(.acquire)) {
        1, 2, 3 => |state| return state,
        else => {},
    }
    const mode: u8 = blk: {
        const raw = std.c.getenv("ZYNUM_BLAS_AMX") orelse break :blk 3;
        const value = std.mem.span(raw);
        if (std.mem.eql(u8, value, "1") or
            std.ascii.eqlIgnoreCase(value, "true") or
            std.ascii.eqlIgnoreCase(value, "on"))
        {
            break :blk 2;
        }
        if (std.mem.eql(u8, value, "0") or
            std.ascii.eqlIgnoreCase(value, "false") or
            std.ascii.eqlIgnoreCase(value, "off") or
            std.ascii.eqlIgnoreCase(value, "no"))
        {
            break :blk 1;
        }
        break :blk 1;
    };
    amx_env_state.store(mode, .release);
    return mode;
}

extern fn zynum_blas_sme_sgemm_panel_f32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.c) void;
extern fn zynum_blas_sme_sgemm_panel4m_f32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.c) void;
extern fn zynum_blas_sme_sgemm_panel2x2_f32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.c) void;
extern fn zynum_blas_sme_sgemm_panel1x2_f32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.c) void;
extern fn zynum_blas_sme_sgemm_panels2x2_f32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize, panel_count: usize) callconv(.c) void;
extern fn zynum_blas_sme_sgemm_panels2x2_u4_f32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize, panel_count: usize) callconv(.c) void;
extern fn zynum_blas_sme_dgemm_panel_f64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.c) void;
extern fn zynum_blas_sme_dgemm_panel4m_f64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.c) void;
extern fn zynum_blas_sme_dgemm_panel2x2_f64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.c) void;
extern fn zynum_blas_sme_dgemm_panel4x2_f64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.c) void;
extern fn zynum_blas_sme_dgemm_panels4x2_f64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize, panel_count: usize) callconv(.c) void;

pub fn preferredColumnBlock(comptime T: type) usize {
    if (T == f32) return f32TileRows() * sme_f32_thread_panel_count;
    if (T == f64 and comptime supports_f64_accumulate) return f64TileRows() * sme_f64_thread_panel_count;
    if (comptime sve2.enabled) return sve2.preferredColumnBlock(T);
    return asimd.preferredColumnBlock(T);
}

fn f32TileRows() usize {
    const svl_bytes = features.streamingVectorBytes();
    if (svl_bytes == 0 or svl_bytes % @sizeOf(f32) != 0) return 0;
    return svl_bytes / @sizeOf(f32);
}

fn f64TileRows() usize {
    const svl_bytes = features.streamingVectorBytes();
    if (svl_bytes == 0 or svl_bytes % @sizeOf(f64) != 0) return 0;
    return svl_bytes / @sizeOf(f64);
}

fn panelBatchCount(comptime T: type, task: gemm_task.Task(T), panel_cols: usize, panels_left: usize) usize {
    if (panels_left <= 1) return panels_left;
    return @min(panels_left, panelBatchCapacity(T, task, panel_cols));
}

fn panelBatchCapacity(comptime T: type, task: gemm_task.Task(T), panel_cols: usize) usize {
    const max_batch = if (T == f32) sme_f32_panel_batch_count else sme_f64_panel_batch_count;
    const per_panel_bytes = task.k *| panel_cols *| @sizeOf(T);
    if (per_panel_bytes == 0) return 1;

    // B-pack batch size is cache-limited, not just call-overhead-driven.
    // Too many adjacent panels can evict packed B and A streams from the
    // shared P-core L2 when several GEMM workers run concurrently.
    const threads = @max(@as(usize, 1), runtime.defaultGemmThreadLimit());
    const l2_budget = runtime.performanceL2Bytes() / (threads * 2);
    const pack_budget = @max(@as(usize, 256 * 1024), l2_budget);
    const cache_batch = @max(@as(usize, 1), pack_budget / per_panel_bytes);
    return @min(max_batch, cache_batch);
}

inline fn packBPanelF32N(comptime tile: usize, task: gemm_task.Task(f32), j: usize, b_pack: []f32) void {
    // Packed B is K-major and panel-contiguous: b_pack[p*tile + col].
    // The SME kernels load one packed column vector per K step.
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * tile;
        inline for (0..tile) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanelF32Dynamic(task: gemm_task.Task(f32), j: usize, tile: usize, b_pack: []f32) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * tile;
        var col: usize = 0;
        while (col < tile) : (col += 1) {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanelF32(task: gemm_task.Task(f32), j: usize, tile: usize, b_pack: []f32) void {
    switch (tile) {
        16 => packBPanelF32N(16, task, j, b_pack),
        else => packBPanelF32Dynamic(task, j, tile, b_pack),
    }
}

inline fn packBPanel2F32N(comptime tile: usize, task: gemm_task.Task(f32), j: usize, b_pack: []f32) void {
    const panel_cols = tile * 2;
    // Two adjacent N panels share a single K loop in the 2x2 SME kernel.
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * panel_cols;
        inline for (0..panel_cols) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

noinline fn packBPanel2F32Transpose4N(comptime tile: usize, task: gemm_task.Task(f32), j: usize, b_pack: []f32) void {
    const panel_cols = tile * 2;
    var p: usize = 0;
    while (p + 4 <= task.k) : (p += 4) {
        const row0 = (p + 0) * panel_cols;
        const row1 = (p + 1) * panel_cols;
        const row2 = (p + 2) * panel_cols;
        const row3 = (p + 3) * panel_cols;
        inline for (0..(panel_cols / 4)) |col_block| {
            const col = col_block * 4;
            const b0 = task.b + matIndex(task.ldb, p, j + col + 0);
            const b1 = task.b + matIndex(task.ldb, p, j + col + 1);
            const b2 = task.b + matIndex(task.ldb, p, j + col + 2);
            const b3 = task.b + matIndex(task.ldb, p, j + col + 3);
            b_pack[row0 + col + 0] = b0[0];
            b_pack[row0 + col + 1] = b1[0];
            b_pack[row0 + col + 2] = b2[0];
            b_pack[row0 + col + 3] = b3[0];
            b_pack[row1 + col + 0] = b0[1];
            b_pack[row1 + col + 1] = b1[1];
            b_pack[row1 + col + 2] = b2[1];
            b_pack[row1 + col + 3] = b3[1];
            b_pack[row2 + col + 0] = b0[2];
            b_pack[row2 + col + 1] = b1[2];
            b_pack[row2 + col + 2] = b2[2];
            b_pack[row2 + col + 3] = b3[2];
            b_pack[row3 + col + 0] = b0[3];
            b_pack[row3 + col + 1] = b1[3];
            b_pack[row3 + col + 2] = b2[3];
            b_pack[row3 + col + 3] = b3[3];
        }
    }
}

fn packBPanel2F32Dynamic(task: gemm_task.Task(f32), j: usize, tile: usize, b_pack: []f32) void {
    const panel_cols = tile * 2;
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * panel_cols;
        var col: usize = 0;
        while (col < panel_cols) : (col += 1) {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn shouldUseTransposeF32Pack(task: gemm_task.Task(f32)) bool {
    const n = task.n1 - task.n0;
    if (task.k % 4 != 0) return false;
    if (task.m == 512 and task.k == 2048 and (n == 32 or n == 64)) return true;
    if (task.m <= 64 and n >= 64 and task.k >= 256) return true;
    return task.m == 256 and n >= 64 and n <= 128 and task.k >= 2048;
}

fn packBPanel2F32(task: gemm_task.Task(f32), j: usize, tile: usize, b_pack: []f32) void {
    switch (tile) {
        16 => packBPanel2F32N(16, task, j, b_pack),
        else => packBPanel2F32Dynamic(task, j, tile, b_pack),
    }
}

fn packBPanel2F32ForTask(task: gemm_task.Task(f32), j: usize, tile: usize, b_pack: []f32) void {
    if (tile == 16 and shouldUseTransposeF32Pack(task)) {
        packBPanel2F32Transpose4N(16, task, j, b_pack);
    } else {
        packBPanel2F32(task, j, tile, b_pack);
    }
}

inline fn packBPanelF64N(comptime tile: usize, task: gemm_task.Task(f64), j: usize, b_pack: []f64) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * tile;
        inline for (0..tile) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanelF64Dynamic(task: gemm_task.Task(f64), j: usize, tile: usize, b_pack: []f64) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * tile;
        var col: usize = 0;
        while (col < tile) : (col += 1) {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanelF64(task: gemm_task.Task(f64), j: usize, tile: usize, b_pack: []f64) void {
    switch (tile) {
        8 => packBPanelF64N(8, task, j, b_pack),
        else => packBPanelF64Dynamic(task, j, tile, b_pack),
    }
}

inline fn packBPanel2F64N(comptime tile: usize, task: gemm_task.Task(f64), j: usize, b_pack: []f64) void {
    const panel_cols = tile * 2;
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * panel_cols;
        inline for (0..panel_cols) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanel2F64Dynamic(task: gemm_task.Task(f64), j: usize, tile: usize, b_pack: []f64) void {
    const panel_cols = tile * 2;
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * panel_cols;
        var col: usize = 0;
        while (col < panel_cols) : (col += 1) {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanel2F64(task: gemm_task.Task(f64), j: usize, tile: usize, b_pack: []f64) void {
    switch (tile) {
        8 => packBPanel2F64N(8, task, j, b_pack),
        else => packBPanel2F64Dynamic(task, j, tile, b_pack),
    }
}

fn shouldUseDynamicF64Pack(task: gemm_task.Task(f64)) bool {
    const n = task.n1 - task.n0;
    return task.m <= 64 and n >= 64 and task.k >= 256;
}

fn packBPanel2F64ForTask(task: gemm_task.Task(f64), j: usize, tile: usize, b_pack: []f64) void {
    if (shouldUseDynamicF64Pack(task)) {
        packBPanel2F64Dynamic(task, j, tile, b_pack);
    } else {
        packBPanel2F64(task, j, tile, b_pack);
    }
}

fn tailRowsF32Direct(task: gemm_task.Task(f32), b_pack: []const f32, row_start: usize, j: usize, tile: usize) void {
    if (row_start >= task.m) return;
    // Scalar tail rows are intentionally simple: they are for fewer than
    // one SME tile of M after the assembly kernels consume full rows.
    var acc_storage: [max_sme_f32_tile]f32 = undefined;
    var i = row_start;
    while (i < task.m) : (i += 1) {
        const acc = acc_storage[0..tile];
        @memset(acc, 0);

        var p: usize = 0;
        while (p < task.k) : (p += 1) {
            const av = task.a[matIndex(task.lda, i, p)];
            const b_base = p * tile;
            var col: usize = 0;
            while (col < tile) : (col += 1) {
                acc[col] = @mulAdd(f32, av, b_pack[b_base + col], acc[col]);
            }
        }

        var col: usize = 0;
        while (col < tile) : (col += 1) {
            task.c[matIndex(task.ldc, i, j + col)] = acc[col];
        }
    }
}

fn tailRowsF64Direct(task: gemm_task.Task(f64), b_pack: []const f64, row_start: usize, j: usize, tile: usize) void {
    var acc_storage: [max_sme_f64_tile]f64 = undefined;
    var i = row_start;
    while (i < task.m) : (i += 1) {
        const acc = acc_storage[0..tile];
        @memset(acc, 0);

        var p: usize = 0;
        while (p < task.k) : (p += 1) {
            const av = task.a[matIndex(task.lda, i, p)];
            const b_base = p * tile;
            var col: usize = 0;
            while (col < tile) : (col += 1) {
                acc[col] = @mulAdd(f64, av, b_pack[b_base + col], acc[col]);
            }
        }

        var col: usize = 0;
        while (col < tile) : (col += 1) {
            task.c[matIndex(task.ldc, i, j + col)] = acc[col];
        }
    }
}

fn shouldUseF32Panels2x2U4(task: gemm_task.Task(f32), tile: usize) bool {
    if (tile != 16) return false;
    if (task.k % 4 != 0) return false;

    const n = task.n1 - task.n0;
    const panel2_cols = tile * 2;
    if (n < panel2_cols or n % panel2_cols != 0) return false;

    const small_square = task.n0 == 0 and
        task.m == n and
        task.k == n and
        (task.m == 96 or task.m == 128 or task.m == 192 or task.m == 256);
    if (small_square) return true;

    // The dispatcher splits m128_n128_k4096 into 32-column tasks on the
    // current M-series profile.  Gate on the visible task chunk so this stays
    // backend-local and does not change global thread policy.
    return task.m == 128 and task.k == 4096 and n == panel2_cols;
}

fn noTransRealF32SmeDirectWithPack(task: gemm_task.Task(f32), tile: usize, b_pack: []f32) void {
    const panel2_cols = tile * 2;
    const panel2_elems = task.k * panel2_cols;
    const lda_bytes = gemm_task.toUsize(task.lda) * @sizeOf(f32);
    const ldc_bytes = gemm_task.toUsize(task.ldc) * @sizeOf(f32);
    const full_rows = task.m - task.m % tile;
    const panel2m_rows = tile * 2;
    const full_rows_2m = task.m - task.m % panel2m_rows;
    const panel4m_rows = tile * sme_m_tile_count;
    const full_rows_4m = task.m - task.m % panel4m_rows;

    // Prefer 2M x 2N f32 panels for the general path.  The 4M x 1N path is
    // kept for a final single-N-panel tail.
    var j = task.n0;
    while (j + panel2_cols <= task.n1) {
        const panels_left = (task.n1 - j) / panel2_cols;
        const batch_panels = panelBatchCount(f32, task, panel2_cols, panels_left);
        const batch_j = j;

        var panel_index: usize = 0;
        while (panel_index < batch_panels) : (panel_index += 1) {
            const panel_offset = panel_index * panel2_elems;
            packBPanel2F32ForTask(task, j, tile, b_pack[panel_offset .. panel_offset + panel2_elems]);
            j += panel2_cols;
        }

        if (full_rows_2m != 0) {
            const c_panel = task.c + matIndex(task.ldc, 0, batch_j);
            if (shouldUseF32Panels2x2U4(task, tile)) {
                zynum_blas_sme_sgemm_panels2x2_u4_f32(task.a, b_pack.ptr, c_panel, full_rows_2m, task.k, lda_bytes, ldc_bytes, batch_panels);
            } else if (batch_panels == 1) {
                zynum_blas_sme_sgemm_panel2x2_f32(task.a, b_pack.ptr, c_panel, full_rows_2m, task.k, lda_bytes, ldc_bytes);
            } else {
                zynum_blas_sme_sgemm_panels2x2_f32(task.a, b_pack.ptr, c_panel, full_rows_2m, task.k, lda_bytes, ldc_bytes, batch_panels);
            }
        }

        var scalar_tail_row = full_rows_2m;
        if (full_rows_2m < full_rows) {
            const a_panel = task.a + matIndex(task.lda, full_rows_2m, 0);
            const c_panel = task.c + matIndex(task.ldc, full_rows_2m, batch_j);
            if (batch_panels == 1) {
                zynum_blas_sme_sgemm_panel1x2_f32(a_panel, b_pack.ptr, c_panel, full_rows - full_rows_2m, task.k, lda_bytes, ldc_bytes);
            } else {
                var residual_panel_index: usize = 0;
                var tail_j = batch_j;
                while (residual_panel_index < batch_panels) : (residual_panel_index += 1) {
                    const panel_offset = residual_panel_index * panel2_elems;
                    const panel_c = task.c + matIndex(task.ldc, full_rows_2m, tail_j);
                    zynum_blas_sme_sgemm_panel1x2_f32(a_panel, b_pack.ptr + panel_offset, panel_c, full_rows - full_rows_2m, task.k, lda_bytes, ldc_bytes);
                    tail_j += panel2_cols;
                }
            }
            scalar_tail_row = full_rows;
        }

        var tail_j = batch_j;
        panel_index = 0;
        while (panel_index < batch_panels) : (panel_index += 1) {
            const panel_offset = panel_index * panel2_elems;
            tailRowsF32Direct(task, b_pack[panel_offset .. panel_offset + panel2_elems], scalar_tail_row, tail_j, panel2_cols);
            tail_j += panel2_cols;
        }
    }

    while (j + tile <= task.n1) : (j += tile) {
        packBPanelF32(task, j, tile, b_pack);

        if (full_rows_4m != 0) {
            const c_panel = task.c + matIndex(task.ldc, 0, j);
            zynum_blas_sme_sgemm_panel4m_f32(task.a, b_pack.ptr, c_panel, full_rows_4m, task.k, lda_bytes, ldc_bytes);
        }
        if (full_rows_4m < full_rows) {
            const a_panel = task.a + matIndex(task.lda, full_rows_4m, 0);
            const c_panel = task.c + matIndex(task.ldc, full_rows_4m, j);
            zynum_blas_sme_sgemm_panel_f32(a_panel, b_pack.ptr, c_panel, full_rows - full_rows_4m, task.k, lda_bytes, ldc_bytes);
        }
        tailRowsF32Direct(task, b_pack, full_rows, j, tile);
    }

    if (j < task.n1) {
        var tail = task;
        tail.n0 = j;
        asimd.noTransRealF32(tail);
    }
}

fn noTransRealF32Amx(task: gemm_task.Task(f32)) bool {
    const mode = amxMode();
    if (mode == 1) return false;
    if (!task.allow_sme) return false;
    if (task.alpha != 1 or task.beta != 0) return false;
    const n = task.n1 - task.n0;
    if (task.m % 16 != 0 or n % 16 != 0 or task.k == 0) return false;

    const auto = mode == 3;
    if (auto and comptime builtin.target.os.tag != .macos) return false;

    const short_wide = if (auto)
        (task.m < 64 and n >= 512 and task.k >= 128) or
            (task.m == 64 and n >= 1024 and task.k <= 1024)
    else
        task.m <= 64 and n >= 512 and task.k >= 128;
    const low_k_large = !auto and task.m >= 512 and n >= 512 and task.k <= 128;
    const square = task.m == n and task.k == n and task.m >= 64 and task.m <= 512;
    const high_k_small = if (auto)
        task.m == 128 and n == 32 and task.k >= 4096
    else
        task.m <= 256 and n == 32 and task.k >= 4096;
    if (!short_wide and !low_k_large and !square and !high_k_small) return false;

    const b_panel = task.b + matIndex(task.ldb, 0, task.n0);
    const c_panel = task.c + matIndex(task.ldc, 0, task.n0);
    return amx.zynum_blas_amx_sgemm_nn_f32(
        @intCast(task.m),
        @intCast(n),
        @intCast(task.k),
        task.a,
        @intCast(gemm_task.toUsize(task.lda)),
        b_panel,
        @intCast(gemm_task.toUsize(task.ldb)),
        c_panel,
        @intCast(gemm_task.toUsize(task.ldc)),
    ) != 0;
}

fn noTransRealF32SmeDirect(task: gemm_task.Task(f32)) bool {
    if (!task.allow_sme) return false;
    if (task.alpha != 1 or task.beta != 0) return false;
    if (noTransRealF32Amx(task)) return true;
    if (task.m *| (task.n1 - task.n0) *| task.k < sme_f32_min_work) return false;

    const tile = f32TileRows();
    if (tile == 0 or tile > max_sme_f32_tile) return false;
    const panel2_cols = tile * 2;
    if (panel2_cols == 0 or panel2_cols > max_sme_f32_tile) return false;
    if (task.m < tile or task.n1 - task.n0 < tile) return false;

    const panel2_count = (task.n1 - task.n0) / panel2_cols;
    const pack_batch = @max(@as(usize, 1), panelBatchCount(f32, task, panel2_cols, panel2_count));
    const pack_elems = task.k * panel2_cols * pack_batch;
    // Keep the stack frame fixed at 256 KiB.  Smaller tiered frames were
    // tested and looked good on probes, but regressed full real/complex
    // sweeps due to code/layout and cache effects.
    if (pack_elems <= sme_stack_pack_f32_elems) {
        var stack_pack: [sme_stack_pack_f32_elems]f32 = undefined;
        noTransRealF32SmeDirectWithPack(task, tile, stack_pack[0..pack_elems]);
        return true;
    }

    const b_pack = std.heap.c_allocator.alloc(f32, pack_elems) catch return false;
    defer std.heap.c_allocator.free(b_pack);
    noTransRealF32SmeDirectWithPack(task, tile, b_pack);

    return true;
}

fn noTransRealF64SmeDirectWithPack(task: gemm_task.Task(f64), tile: usize, b_pack: []f64) void {
    const panel2_cols = tile * 2;
    const panel_elems = task.k * panel2_cols;
    const lda_bytes = gemm_task.toUsize(task.lda) * @sizeOf(f64);
    const ldc_bytes = gemm_task.toUsize(task.ldc) * @sizeOf(f64);
    const full_rows = task.m - task.m % tile;
    const panel4m_rows = tile * sme_m_tile_count;
    const full_rows_4m = task.m - task.m % panel4m_rows;
    const full_rows_2m = task.m - task.m % (tile * 2);

    // f64 uses 4M x 2N as its main SME shape.  It has enough encodable
    // ZA .d tiles on M5 and benefits from B-side prefetch in assembly.
    var j = task.n0;
    while (j + panel2_cols <= task.n1) {
        const panels_left = (task.n1 - j) / panel2_cols;
        const batch_panels = panelBatchCount(f64, task, panel2_cols, panels_left);
        const batch_j = j;

        var panel_index: usize = 0;
        while (panel_index < batch_panels) : (panel_index += 1) {
            const panel_offset = panel_index * panel_elems;
            packBPanel2F64ForTask(task, j, tile, b_pack[panel_offset .. panel_offset + panel_elems]);
            j += panel2_cols;
        }

        if (full_rows_4m != 0) {
            const c_panel = task.c + matIndex(task.ldc, 0, batch_j);
            if (batch_panels == 1) {
                zynum_blas_sme_dgemm_panel4x2_f64(task.a, b_pack.ptr, c_panel, full_rows_4m, task.k, lda_bytes, ldc_bytes);
            } else {
                zynum_blas_sme_dgemm_panels4x2_f64(task.a, b_pack.ptr, c_panel, full_rows_4m, task.k, lda_bytes, ldc_bytes, batch_panels);
            }
        }

        var tail_j = batch_j;
        panel_index = 0;
        while (panel_index < batch_panels) : (panel_index += 1) {
            const panel_offset = panel_index * panel_elems;
            if (full_rows_4m < full_rows_2m) {
                const a_panel = task.a + matIndex(task.lda, full_rows_4m, 0);
                const c_panel = task.c + matIndex(task.ldc, full_rows_4m, tail_j);
                zynum_blas_sme_dgemm_panel2x2_f64(a_panel, b_pack.ptr + panel_offset, c_panel, full_rows_2m - full_rows_4m, task.k, lda_bytes, ldc_bytes);
            }
            tailRowsF64Direct(task, b_pack[panel_offset .. panel_offset + panel_elems], full_rows_2m, tail_j, panel2_cols);
            tail_j += panel2_cols;
        }
    }

    while (j + tile <= task.n1) : (j += tile) {
        packBPanelF64(task, j, tile, b_pack);

        if (full_rows_4m != 0) {
            const c_panel = task.c + matIndex(task.ldc, 0, j);
            zynum_blas_sme_dgemm_panel4m_f64(task.a, b_pack.ptr, c_panel, full_rows_4m, task.k, lda_bytes, ldc_bytes);
        }
        if (full_rows_4m < full_rows) {
            const a_panel = task.a + matIndex(task.lda, full_rows_4m, 0);
            const c_panel = task.c + matIndex(task.ldc, full_rows_4m, j);
            zynum_blas_sme_dgemm_panel_f64(a_panel, b_pack.ptr, c_panel, full_rows - full_rows_4m, task.k, lda_bytes, ldc_bytes);
        }
        tailRowsF64Direct(task, b_pack, full_rows, j, tile);
    }

    if (j < task.n1) {
        var tail = task;
        tail.n0 = j;
        asimd.noTransRealF64(tail);
    }
}

fn noTransRealF64Amx(task: gemm_task.Task(f64)) bool {
    const mode = amxMode();
    if (mode == 1) return false;
    if (!task.allow_sme) return false;
    if (task.alpha != 1 or task.beta != 0) return false;
    const n = task.n1 - task.n0;
    if (task.m % 8 != 0 or n % 8 != 0 or task.k == 0) return false;

    const auto = mode == 3;
    if (auto and comptime builtin.target.os.tag != .macos) return false;

    const short_wide = !auto and task.m <= 64 and n >= 512 and task.k >= 128;
    const square = task.m == n and task.k == n and task.m >= 64 and task.m <= 384;
    if (!short_wide and !square) return false;

    const b_panel = task.b + matIndex(task.ldb, 0, task.n0);
    const c_panel = task.c + matIndex(task.ldc, 0, task.n0);
    return amx.zynum_blas_amx_dgemm_nn_f64(
        @intCast(task.m),
        @intCast(n),
        @intCast(task.k),
        task.a,
        @intCast(gemm_task.toUsize(task.lda)),
        b_panel,
        @intCast(gemm_task.toUsize(task.ldb)),
        c_panel,
        @intCast(gemm_task.toUsize(task.ldc)),
    ) != 0;
}

fn noTransRealF64SmeDirect(task: gemm_task.Task(f64)) bool {
    if (comptime !supports_f64_accumulate) return false;
    if (!task.allow_sme) return false;
    if (task.alpha != 1 or task.beta != 0) return false;
    if (noTransRealF64Amx(task)) return true;
    if (task.m *| (task.n1 - task.n0) *| task.k < sme_f64_min_work) return false;

    const tile = f64TileRows();
    if (tile == 0 or tile > max_sme_f64_tile) return false;
    const panel2_cols = tile * 2;
    if (panel2_cols == 0 or panel2_cols > max_sme_f64_tile) return false;
    if (task.m < tile or task.n1 - task.n0 < tile) return false;

    const panel2_count = (task.n1 - task.n0) / panel2_cols;
    const pack_batch = @max(@as(usize, 1), panelBatchCount(f64, task, panel2_cols, panel2_count));
    const pack_elems = task.k * panel2_cols * pack_batch;
    // Match f32: a single stack-pack size is more stable than tiered
    // stack frames in complete sweeps.
    if (pack_elems <= sme_stack_pack_f64_elems) {
        var stack_pack: [sme_stack_pack_f64_elems]f64 = undefined;
        noTransRealF64SmeDirectWithPack(task, tile, stack_pack[0..pack_elems]);
        return true;
    }

    const b_pack = std.heap.c_allocator.alloc(f64, pack_elems) catch return false;
    defer std.heap.c_allocator.free(b_pack);
    noTransRealF64SmeDirectWithPack(task, tile, b_pack);

    return true;
}

pub fn noTransRealF32(task: gemm_task.Task(f32)) void {
    if (noTransRealF32SmeDirect(task)) return;
    if (comptime sve2.enabled) {
        sve2.noTransRealF32(task);
    } else {
        asimd.noTransRealF32(task);
    }
}

pub fn noTransRealF64(task: gemm_task.Task(f64)) void {
    if (noTransRealF64SmeDirect(task)) return;
    if (comptime supports_f64_accumulate and sve2.enabled) {
        sve2.noTransRealF64(task);
    } else {
        asimd.noTransRealF64(task);
    }
}
