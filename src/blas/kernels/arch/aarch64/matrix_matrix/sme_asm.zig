// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! SME GEMM microkernels expressed as Zig-owned whole-function inline asm.

const builders = @import("../asm/builders.zig");

pub noinline fn sgemmPanelF32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (builders.smeGemmPanel1mAsm("s") ::: .{ .memory = true });
}

pub noinline fn sgemmPanel4mF32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (builders.smeGemmPanel4mAsm("s") ::: .{ .memory = true });
}

pub noinline fn sgemmPanel2x2F32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (builders.smeGemmPanel2x2Asm("s") ::: .{ .memory = true });
}

pub noinline fn sgemmPanel1x2F32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (builders.smeGemmPanel1x2Asm("s") ::: .{ .memory = true });
}

pub noinline fn sgemmPanels2x2F32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize, panel_count: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    _ = panel_count;
    asm volatile (builders.smeGemmPanels2x2Asm("s", 2, false) ::: .{ .memory = true });
}

pub noinline fn sgemmPanels2x2U4F32(a: [*]const f32, b_pack: [*]const f32, c: [*]f32, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize, panel_count: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    _ = panel_count;
    asm volatile (builders.smeGemmPanels2x2Asm("s", 4, true) ::: .{ .memory = true });
}

pub noinline fn dgemmPanelF64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (builders.smeGemmPanel1mAsm("d") ::: .{ .memory = true });
}

pub noinline fn dgemmPanel4mF64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (builders.smeGemmPanel4mAsm("d") ::: .{ .memory = true });
}

pub noinline fn dgemmPanel2x2F64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (builders.smeGemmPanel2x2Asm("d") ::: .{ .memory = true });
}

pub noinline fn dgemmPanel4x2F64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    asm volatile (builders.smeGemmPanel4x2F64Asm() ::: .{ .memory = true });
}

pub noinline fn dgemmPanels4x2F64(a: [*]const f64, b_pack: [*]const f64, c: [*]f64, m_full: usize, k: usize, lda_bytes: usize, ldc_bytes: usize, panel_count: usize) callconv(.naked) void {
    _ = a;
    _ = b_pack;
    _ = c;
    _ = m_full;
    _ = k;
    _ = lda_bytes;
    _ = ldc_bytes;
    _ = panel_count;
    asm volatile (builders.smeGemmPanels4x2F64Asm() ::: .{ .memory = true });
}
