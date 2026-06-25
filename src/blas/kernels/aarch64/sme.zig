// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");
const gemm_task = @import("../matrix_matrix/task.zig");
const asimd = @import("asimd.zig");
const amx = @import("amx_gemm.zig");
const features = @import("features.zig");
const sme_gemm_asm = @import("sme_gemm_asm.zig");
const sve2 = @import("sve2.zig");

const matIndex = gemm_task.matIndex;
const sme_f32_thread_panel_count = 2;
const sme_f64_thread_panel_count = 2;
const sme_m_tile_count = 4;
const max_sme_stack_pack_bytes = 256 * 1024;
const max_sme_stack_pack_f32_elems = max_sme_stack_pack_bytes / @sizeOf(f32);
const max_sme_stack_pack_f64_elems = max_sme_stack_pack_bytes / @sizeOf(f64);
const max_sme_f32_tile = 256;
const max_sme_f64_tile = 256;
const Vec4f = @Vector(4, f32);

pub const enabled: bool = features.has_sme;

pub const supports_f64_accumulate: bool = features.has_sme_f64f64;

pub const supports_sme2: bool = features.has_sme2;
pub const supports_sme2p1: bool = features.has_sme2p1;
pub const supports_sme_tmop: bool = features.has_sme_tmop;

threadlocal var sme_f32_pack_ptr: ?[*]f32 = null;
threadlocal var sme_f32_pack_len: usize = 0;
threadlocal var sme_f64_pack_ptr: ?[*]f64 = null;
threadlocal var sme_f64_pack_len: usize = 0;

const amx_fast_f32_pack_plan = gemm_task.PackWorkspacePlan{
    .stack_bytes = 128 * 1024,
    .cache_bytes = 8 * 1024 * 1024,
};

fn PackWorkspace(comptime T: type) type {
    return struct {
        data: []T,
        cached: bool,

        fn deinit(self: @This()) void {
            if (!self.cached) std.heap.c_allocator.free(self.data);
        }
    };
}

fn acquireSmePack(comptime T: type, len: usize, cache_bytes: usize) ?PackWorkspace(T) {
    const max_cached_elems = cache_bytes / @sizeOf(T);
    if (len > max_cached_elems) {
        const data = std.heap.c_allocator.alloc(T, len) catch return null;
        return .{ .data = data, .cached = false };
    }

    if (T == f32) {
        if (sme_f32_pack_len < len) {
            const data = std.heap.c_allocator.alloc(f32, len) catch return null;
            if (sme_f32_pack_ptr) |old| std.heap.c_allocator.free(old[0..sme_f32_pack_len]);
            sme_f32_pack_ptr = data.ptr;
            sme_f32_pack_len = len;
        }
        return .{ .data = sme_f32_pack_ptr.?[0..len], .cached = true };
    }
    if (T == f64) {
        if (sme_f64_pack_len < len) {
            const data = std.heap.c_allocator.alloc(f64, len) catch return null;
            if (sme_f64_pack_ptr) |old| std.heap.c_allocator.free(old[0..sme_f64_pack_len]);
            sme_f64_pack_ptr = data.ptr;
            sme_f64_pack_len = len;
        }
        return .{ .data = sme_f64_pack_ptr.?[0..len], .cached = true };
    }
    @compileError("SME pack workspace supports f32 and f64");
}

inline fn callSmeGemmPanel(
    comptime T: type,
    comptime kernel: anytype,
    a: [*]const T,
    b_pack: [*]const T,
    c: [*]T,
    m_full: usize,
    k: usize,
    lda_bytes: usize,
    ldc_bytes: usize,
) void {
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSmZa();
    defer sm_state.stopSmZa();

    const Kernel = *const fn ([*]const T, [*]const T, [*]T, usize, usize, usize, usize) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(a, b_pack, c, m_full, k, lda_bytes, ldc_bytes);
}

inline fn callSmeGemmPanelBatch(
    comptime T: type,
    comptime kernel: anytype,
    a: [*]const T,
    b_pack: [*]const T,
    c: [*]T,
    m_full: usize,
    k: usize,
    lda_bytes: usize,
    ldc_bytes: usize,
    panel_count: usize,
) void {
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSmZa();
    defer sm_state.stopSmZa();

    const Kernel = *const fn ([*]const T, [*]const T, [*]T, usize, usize, usize, usize, usize) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(a, b_pack, c, m_full, k, lda_bytes, ldc_bytes, panel_count);
}

pub fn preferredColumnBlock(comptime T: type) usize {
    if (T == f32) return f32TileRows() * sme_f32_thread_panel_count;
    if (T == f64 and comptime supports_f64_accumulate) return f64TileRows() * sme_f64_thread_panel_count;
    if (comptime sve2.enabled) return sve2.preferredColumnBlock(T);
    return asimd.preferredColumnBlock(T);
}

pub fn tryNoTransRealF32Fast(m_: gemm_task.BlasInt, n_: gemm_task.BlasInt, k_: gemm_task.BlasInt, alpha: f32, a: [*]const f32, lda_: gemm_task.BlasInt, b: [*]const f32, ldb_: gemm_task.BlasInt, beta: f32, c: [*]f32, ldc_: gemm_task.BlasInt) bool {
    if (comptime builtin.target.os.tag != .macos) return false;
    if (alpha != 1 or beta != 0) return false;
    if (m_ != 128 or n_ != 128 or k_ != 128) return false;
    return amx.sgemmN32(m_, n_, k_, a, lda_, b, ldb_, c, ldc_, amx_fast_f32_pack_plan) != 0;
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

inline fn loadF32x4(ptr: [*]const f32) Vec4f {
    return @as(*align(1) const Vec4f, @ptrCast(ptr)).*;
}

inline fn storeF32x4(ptr: [*]f32, value: Vec4f) void {
    @as(*align(1) Vec4f, @ptrCast(ptr)).* = value;
}

inline fn trn1F32(a: Vec4f, b: Vec4f) Vec4f {
    return @shuffle(f32, a, b, @Vector(4, i32){ 0, ~@as(i32, 0), 2, ~@as(i32, 2) });
}

inline fn trn2F32(a: Vec4f, b: Vec4f) Vec4f {
    return @shuffle(f32, a, b, @Vector(4, i32){ 1, ~@as(i32, 1), 3, ~@as(i32, 3) });
}

inline fn combineLowF32(a: Vec4f, b: Vec4f) Vec4f {
    return @shuffle(f32, a, b, @Vector(4, i32){ 0, 1, ~@as(i32, 0), ~@as(i32, 1) });
}

inline fn combineHighF32(a: Vec4f, b: Vec4f) Vec4f {
    return @shuffle(f32, a, b, @Vector(4, i32){ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
}

fn panelBatchCount(comptime T: type, task: gemm_task.Task(T), panels_left: usize) usize {
    if (panels_left <= 1) return panels_left;
    return @min(panels_left, @max(@as(usize, 1), task.execution.sme_panel_batch));
}

fn maxSmeStackPackElems(comptime T: type) comptime_int {
    if (T == f32) return max_sme_stack_pack_f32_elems;
    if (T == f64) return max_sme_stack_pack_f64_elems;
    @compileError("SME stack pack supports f32 and f64");
}

fn maxSmeTile(comptime T: type) comptime_int {
    if (T == f32) return max_sme_f32_tile;
    if (T == f64) return max_sme_f64_tile;
    @compileError("SME tiles support f32 and f64");
}

fn plannedStackPackElems(comptime T: type, task: gemm_task.Task(T)) usize {
    return @min(@as(usize, maxSmeStackPackElems(T)), task.execution.pack.stack_bytes / @sizeOf(T));
}

fn canUseSmeTile(comptime T: type, task: gemm_task.Task(T), tile: usize) bool {
    if (tile == 0 or tile > maxSmeTile(T)) return false;
    const panel2_cols = tile * 2;
    if (panel2_cols == 0 or panel2_cols > maxSmeTile(T)) return false;
    return task.m >= tile and task.n1 - task.n0 >= tile;
}

fn directPackElems(comptime T: type, task: gemm_task.Task(T), tile: usize) usize {
    const panel2_cols = tile * 2;
    const panel2_count = (task.n1 - task.n0) / panel2_cols;
    const pack_batch = @max(@as(usize, 1), panelBatchCount(T, task, panel2_count));
    return task.k * panel2_cols * pack_batch;
}

fn runSmeDirectWithWorkspace(
    comptime T: type,
    task: gemm_task.Task(T),
    tile: usize,
    comptime run: fn (gemm_task.Task(T), usize, []T) void,
) bool {
    const pack_elems = directPackElems(T, task, tile);
    if (pack_elems <= plannedStackPackElems(T, task)) {
        var stack_pack: [maxSmeStackPackElems(T)]T = undefined;
        run(task, tile, stack_pack[0..pack_elems]);
        return true;
    }

    const workspace = acquireSmePack(T, pack_elems, task.execution.pack.cache_bytes) orelse return false;
    defer workspace.deinit();
    run(task, tile, workspace.data);
    return true;
}

inline fn packBPanelN(comptime T: type, comptime panel_cols: usize, task: gemm_task.Task(T), j: usize, b_pack: []T) void {
    // Packed B is K-major and panel-contiguous: b_pack[p*tile + col].
    // The SME kernels load one packed column vector per K step.
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * panel_cols;
        inline for (0..panel_cols) |col| {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanelDynamic(comptime T: type, task: gemm_task.Task(T), j: usize, panel_cols: usize, b_pack: []T) void {
    var p: usize = 0;
    while (p < task.k) : (p += 1) {
        const base = p * panel_cols;
        var col: usize = 0;
        while (col < panel_cols) : (col += 1) {
            b_pack[base + col] = task.b[matIndex(task.ldb, p, j + col)];
        }
    }
}

fn packBPanelWithDefault(comptime T: type, comptime default_panel_cols: usize, task: gemm_task.Task(T), j: usize, panel_cols: usize, b_pack: []T) void {
    switch (panel_cols) {
        default_panel_cols => packBPanelN(T, default_panel_cols, task, j, b_pack),
        else => packBPanelDynamic(T, task, j, panel_cols, b_pack),
    }
}

fn packBPanelF32(task: gemm_task.Task(f32), j: usize, tile: usize, b_pack: []f32) void {
    packBPanelWithDefault(f32, 16, task, j, tile, b_pack);
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
            const v0 = loadF32x4(task.b + matIndex(task.ldb, p, j + col + 0));
            const v1 = loadF32x4(task.b + matIndex(task.ldb, p, j + col + 1));
            const v2 = loadF32x4(task.b + matIndex(task.ldb, p, j + col + 2));
            const v3 = loadF32x4(task.b + matIndex(task.ldb, p, j + col + 3));
            const t0 = trn1F32(v0, v1);
            const t1 = trn2F32(v0, v1);
            const t2 = trn1F32(v2, v3);
            const t3 = trn2F32(v2, v3);
            storeF32x4(b_pack.ptr + row0 + col, combineLowF32(t0, t2));
            storeF32x4(b_pack.ptr + row1 + col, combineLowF32(t1, t3));
            storeF32x4(b_pack.ptr + row2 + col, combineHighF32(t0, t2));
            storeF32x4(b_pack.ptr + row3 + col, combineHighF32(t1, t3));
        }
    }
}

fn canUseTransposeF32Pack(task: gemm_task.Task(f32)) bool {
    const n = task.n1 - task.n0;
    if (task.k % 4 != 0) return false;
    if (n < 32) return false;
    return true;
}

fn packBPanel2F32(task: gemm_task.Task(f32), j: usize, tile: usize, b_pack: []f32) void {
    packBPanelWithDefault(f32, 32, task, j, tile * 2, b_pack);
}

fn packBPanel2F32ForTask(task: gemm_task.Task(f32), j: usize, tile: usize, b_pack: []f32) void {
    if (task.execution.b_pack == .transpose4 and tile == 16 and canUseTransposeF32Pack(task)) {
        packBPanel2F32Transpose4N(16, task, j, b_pack);
    } else {
        packBPanel2F32(task, j, tile, b_pack);
    }
}

fn packBPanelF64(task: gemm_task.Task(f64), j: usize, tile: usize, b_pack: []f64) void {
    packBPanelWithDefault(f64, 8, task, j, tile, b_pack);
}

fn packBPanel2F64(task: gemm_task.Task(f64), j: usize, tile: usize, b_pack: []f64) void {
    packBPanelWithDefault(f64, 16, task, j, tile * 2, b_pack);
}

fn packBPanel2F64ForTask(task: gemm_task.Task(f64), j: usize, tile: usize, b_pack: []f64) void {
    if (task.execution.b_pack == .dynamic) {
        packBPanelDynamic(f64, task, j, tile * 2, b_pack);
    } else {
        packBPanel2F64(task, j, tile, b_pack);
    }
}

fn tailRowsDirect(comptime T: type, task: gemm_task.Task(T), b_pack: []const T, row_start: usize, j: usize, tile: usize) void {
    if (row_start >= task.m) return;
    // Scalar tail rows are intentionally simple: they are for fewer than
    // one SME tile of M after the assembly kernels consume full rows.
    var acc_storage: [maxSmeTile(T)]T = undefined;
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
                acc[col] = @mulAdd(T, av, b_pack[b_base + col], acc[col]);
            }
        }

        var col: usize = 0;
        while (col < tile) : (col += 1) {
            task.c[matIndex(task.ldc, i, j + col)] = acc[col];
        }
    }
}

fn canUseF32Panels2x2U4(task: gemm_task.Task(f32), tile: usize) bool {
    if (tile != 16) return false;
    if (task.k % 4 != 0) return false;

    const n = task.n1 - task.n0;
    const panel2_cols = tile * 2;
    if (n < panel2_cols or n % panel2_cols != 0) return false;
    return true;
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
        const batch_panels = panelBatchCount(f32, task, panels_left);
        const batch_j = j;

        var panel_index: usize = 0;
        while (panel_index < batch_panels) : (panel_index += 1) {
            const panel_offset = panel_index * panel2_elems;
            packBPanel2F32ForTask(task, j, tile, b_pack[panel_offset .. panel_offset + panel2_elems]);
            j += panel2_cols;
        }

        if (full_rows_2m != 0) {
            const c_panel = task.c + matIndex(task.ldc, 0, batch_j);
            if (task.execution.f32_panel == .panels2x2_u4 and canUseF32Panels2x2U4(task, tile)) {
                callSmeGemmPanelBatch(f32, sme_gemm_asm.sgemmPanels2x2U4F32, task.a, b_pack.ptr, c_panel, full_rows_2m, task.k, lda_bytes, ldc_bytes, batch_panels);
            } else if (batch_panels == 1) {
                callSmeGemmPanel(f32, sme_gemm_asm.sgemmPanel2x2F32, task.a, b_pack.ptr, c_panel, full_rows_2m, task.k, lda_bytes, ldc_bytes);
            } else {
                callSmeGemmPanelBatch(f32, sme_gemm_asm.sgemmPanels2x2F32, task.a, b_pack.ptr, c_panel, full_rows_2m, task.k, lda_bytes, ldc_bytes, batch_panels);
            }
        }

        var scalar_tail_row = full_rows_2m;
        if (full_rows_2m < full_rows) {
            const a_panel = task.a + matIndex(task.lda, full_rows_2m, 0);
            const c_panel = task.c + matIndex(task.ldc, full_rows_2m, batch_j);
            if (batch_panels == 1) {
                callSmeGemmPanel(f32, sme_gemm_asm.sgemmPanel1x2F32, a_panel, b_pack.ptr, c_panel, full_rows - full_rows_2m, task.k, lda_bytes, ldc_bytes);
            } else {
                var residual_panel_index: usize = 0;
                var tail_j = batch_j;
                while (residual_panel_index < batch_panels) : (residual_panel_index += 1) {
                    const panel_offset = residual_panel_index * panel2_elems;
                    const panel_c = task.c + matIndex(task.ldc, full_rows_2m, tail_j);
                    callSmeGemmPanel(f32, sme_gemm_asm.sgemmPanel1x2F32, a_panel, b_pack.ptr + panel_offset, panel_c, full_rows - full_rows_2m, task.k, lda_bytes, ldc_bytes);
                    tail_j += panel2_cols;
                }
            }
            scalar_tail_row = full_rows;
        }

        var tail_j = batch_j;
        panel_index = 0;
        while (panel_index < batch_panels) : (panel_index += 1) {
            const panel_offset = panel_index * panel2_elems;
            tailRowsDirect(f32, task, b_pack[panel_offset .. panel_offset + panel2_elems], scalar_tail_row, tail_j, panel2_cols);
            tail_j += panel2_cols;
        }
    }

    while (j + tile <= task.n1) : (j += tile) {
        packBPanelF32(task, j, tile, b_pack);

        if (full_rows_4m != 0) {
            const c_panel = task.c + matIndex(task.ldc, 0, j);
            callSmeGemmPanel(f32, sme_gemm_asm.sgemmPanel4mF32, task.a, b_pack.ptr, c_panel, full_rows_4m, task.k, lda_bytes, ldc_bytes);
        }
        if (full_rows_4m < full_rows) {
            const a_panel = task.a + matIndex(task.lda, full_rows_4m, 0);
            const c_panel = task.c + matIndex(task.ldc, full_rows_4m, j);
            callSmeGemmPanel(f32, sme_gemm_asm.sgemmPanelF32, a_panel, b_pack.ptr, c_panel, full_rows - full_rows_4m, task.k, lda_bytes, ldc_bytes);
        }
        tailRowsDirect(f32, task, b_pack, full_rows, j, tile);
    }

    if (j < task.n1) {
        var tail = task;
        tail.n0 = j;
        asimd.noTransRealF32(tail);
    }
}

fn noTransRealF32Amx(task: gemm_task.Task(f32)) bool {
    if (comptime builtin.target.os.tag != .macos) return false;
    if (!task.allow_sme) return false;
    if (task.alpha != 1 or task.beta != 0) return false;
    const n = task.n1 - task.n0;
    if (task.m % 16 != 0 or n % 16 != 0 or task.k == 0) return false;
    if (task.k > 512) return false;
    if (task.execution.amx != .f32_n16 and task.execution.amx != .f32_n32) return false;

    const b_panel = task.b + matIndex(task.ldb, 0, task.n0);
    const c_panel = task.c + matIndex(task.ldc, 0, task.n0);
    const m_c: c_int = @intCast(task.m);
    const n_c: c_int = @intCast(n);
    const k_c: c_int = @intCast(task.k);
    const lda_c: c_int = @intCast(gemm_task.toUsize(task.lda));
    const ldb_c: c_int = @intCast(gemm_task.toUsize(task.ldb));
    const ldc_c: c_int = @intCast(gemm_task.toUsize(task.ldc));

    return switch (task.execution.amx) {
        .f32_n32 => amx.sgemmN32(m_c, n_c, k_c, task.a, lda_c, b_panel, ldb_c, c_panel, ldc_c, task.execution.amx_pack) != 0,
        .f32_n16 => amx.sgemmN16(m_c, n_c, k_c, task.a, lda_c, b_panel, ldb_c, c_panel, ldc_c, task.execution.amx_pack) != 0,
        else => false,
    };
}

fn noTransRealF32AmxFullPanels(task: gemm_task.Task(f32)) bool {
    if (comptime builtin.target.os.tag != .macos) return false;
    if (!task.allow_sme) return false;
    if (task.alpha != 1 or task.beta != 0) return false;
    const n = task.n1 - task.n0;
    const n_full = n - n % 16;
    if (n_full == 0 or n_full == n) return false;
    if (task.m % 16 != 0 or task.k == 0) return false;
    if (task.k > 512) return false;

    const b_panel = task.b + matIndex(task.ldb, 0, task.n0);
    const c_panel = task.c + matIndex(task.ldc, 0, task.n0);
    const ok = amx.sgemmN16(
        @intCast(task.m),
        @intCast(n_full),
        @intCast(task.k),
        task.a,
        @intCast(gemm_task.toUsize(task.lda)),
        b_panel,
        @intCast(gemm_task.toUsize(task.ldb)),
        c_panel,
        @intCast(gemm_task.toUsize(task.ldc)),
        amx_fast_f32_pack_plan,
    ) != 0;
    if (!ok) return false;

    var tail = task;
    tail.n0 += n_full;
    if (tail.n0 < tail.n1) asimd.noTransRealF32(tail);
    return true;
}

fn noTransRealF32SmeDirect(task: gemm_task.Task(f32)) bool {
    if (!task.allow_sme) return false;
    if (task.alpha != 1 or task.beta != 0) return false;
    if (noTransRealF32Amx(task)) return true;
    if (noTransRealF32AmxFullPanels(task)) return true;

    const tile = f32TileRows();
    if (!canUseSmeTile(f32, task, tile)) return false;
    return runSmeDirectWithWorkspace(f32, task, tile, noTransRealF32SmeDirectWithPack);
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

    // f64 uses a 4M x 2N main SME shape, with 2M x 2N handling residual rows.
    var j = task.n0;
    while (j + panel2_cols <= task.n1) {
        const panels_left = (task.n1 - j) / panel2_cols;
        const batch_panels = panelBatchCount(f64, task, panels_left);
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
                callSmeGemmPanel(f64, sme_gemm_asm.dgemmPanel4x2F64, task.a, b_pack.ptr, c_panel, full_rows_4m, task.k, lda_bytes, ldc_bytes);
            } else {
                callSmeGemmPanelBatch(f64, sme_gemm_asm.dgemmPanels4x2F64, task.a, b_pack.ptr, c_panel, full_rows_4m, task.k, lda_bytes, ldc_bytes, batch_panels);
            }
        }

        var tail_j = batch_j;
        panel_index = 0;
        while (panel_index < batch_panels) : (panel_index += 1) {
            const panel_offset = panel_index * panel_elems;
            if (full_rows_4m < full_rows_2m) {
                const a_panel = task.a + matIndex(task.lda, full_rows_4m, 0);
                const c_panel = task.c + matIndex(task.ldc, full_rows_4m, tail_j);
                callSmeGemmPanel(f64, sme_gemm_asm.dgemmPanel2x2F64, a_panel, b_pack.ptr + panel_offset, c_panel, full_rows_2m - full_rows_4m, task.k, lda_bytes, ldc_bytes);
            }
            tailRowsDirect(f64, task, b_pack[panel_offset .. panel_offset + panel_elems], full_rows_2m, tail_j, panel2_cols);
            tail_j += panel2_cols;
        }
    }

    while (j + tile <= task.n1) : (j += tile) {
        packBPanelF64(task, j, tile, b_pack);

        if (full_rows_4m != 0) {
            const c_panel = task.c + matIndex(task.ldc, 0, j);
            callSmeGemmPanel(f64, sme_gemm_asm.dgemmPanel4mF64, task.a, b_pack.ptr, c_panel, full_rows_4m, task.k, lda_bytes, ldc_bytes);
        }
        if (full_rows_4m < full_rows) {
            const a_panel = task.a + matIndex(task.lda, full_rows_4m, 0);
            const c_panel = task.c + matIndex(task.ldc, full_rows_4m, j);
            callSmeGemmPanel(f64, sme_gemm_asm.dgemmPanelF64, a_panel, b_pack.ptr, c_panel, full_rows - full_rows_4m, task.k, lda_bytes, ldc_bytes);
        }
        tailRowsDirect(f64, task, b_pack, full_rows, j, tile);
    }

    if (j < task.n1) {
        var tail = task;
        tail.n0 = j;
        asimd.noTransRealF64(tail);
    }
}

fn noTransRealF64Amx(task: gemm_task.Task(f64)) bool {
    if (comptime builtin.target.os.tag != .macos) return false;
    if (!task.allow_sme) return false;
    if (task.alpha != 1 or task.beta != 0) return false;
    const n = task.n1 - task.n0;
    if (task.m % 8 != 0 or n % 8 != 0 or task.k == 0) return false;
    if (task.execution.amx != .f64_n8 and task.execution.amx != .f64_n16 and task.execution.amx != .f64_n32) return false;

    const b_panel = task.b + matIndex(task.ldb, 0, task.n0);
    const c_panel = task.c + matIndex(task.ldc, 0, task.n0);
    const m_c: c_int = @intCast(task.m);
    const n_c: c_int = @intCast(n);
    const k_c: c_int = @intCast(task.k);
    const lda_c: c_int = @intCast(gemm_task.toUsize(task.lda));
    const ldb_c: c_int = @intCast(gemm_task.toUsize(task.ldb));
    const ldc_c: c_int = @intCast(gemm_task.toUsize(task.ldc));

    return switch (task.execution.amx) {
        .f64_n32 => amx.dgemmN32(m_c, n_c, k_c, task.a, lda_c, b_panel, ldb_c, c_panel, ldc_c, task.execution.amx_pack) != 0,
        .f64_n16 => amx.dgemmN16(m_c, n_c, k_c, task.a, lda_c, b_panel, ldb_c, c_panel, ldc_c, task.execution.amx_pack) != 0,
        .f64_n8 => amx.dgemmN8(m_c, n_c, k_c, task.a, lda_c, b_panel, ldb_c, c_panel, ldc_c, task.execution.amx_pack) != 0,
        else => false,
    };
}

fn noTransRealF64SmeDirect(task: gemm_task.Task(f64)) bool {
    if (comptime !supports_f64_accumulate) return false;
    if (!task.allow_sme) return false;
    if (task.alpha != 1 or task.beta != 0) return false;
    if (noTransRealF64Amx(task)) return true;

    const tile = f64TileRows();
    if (!canUseSmeTile(f64, task, tile)) return false;
    return runSmeDirectWithWorkspace(f64, task, tile, noTransRealF64SmeDirectWithPack);
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
