// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const builtin = @import("builtin");
const std = @import("std");

// Changing PSTATE.SM makes the architectural Z/P/FFR contents unknown.  The
// compiler cannot infer that effect from an inline `smstart`/`smstop`, so model
// it explicitly at both state boundaries.  In particular, this prevents a
// scalar coefficient held in the low lane of a Z register from being reused
// after entering streaming mode.
const streaming_mode_clobbers: std.builtin.assembly.Clobbers = if (builtin.cpu.arch == .aarch64) .{
    .z0 = true,
    .z1 = true,
    .z2 = true,
    .z3 = true,
    .z4 = true,
    .z5 = true,
    .z6 = true,
    .z7 = true,
    .z8 = true,
    .z9 = true,
    .z10 = true,
    .z11 = true,
    .z12 = true,
    .z13 = true,
    .z14 = true,
    .z15 = true,
    .z16 = true,
    .z17 = true,
    .z18 = true,
    .z19 = true,
    .z20 = true,
    .z21 = true,
    .z22 = true,
    .z23 = true,
    .z24 = true,
    .z25 = true,
    .z26 = true,
    .z27 = true,
    .z28 = true,
    .z29 = true,
    .z30 = true,
    .z31 = true,
    .p0 = true,
    .p1 = true,
    .p2 = true,
    .p3 = true,
    .p4 = true,
    .p5 = true,
    .p6 = true,
    .p7 = true,
    .p8 = true,
    .p9 = true,
    .p10 = true,
    .p11 = true,
    .p12 = true,
    .p13 = true,
    .p14 = true,
    .p15 = true,
    .ffr = true,
    .memory = true,
} else .{ .memory = true };

pub const has_asimd: bool =
    builtin.cpu.arch == .aarch64 and
    builtin.cpu.hasAll(.aarch64, &.{ .neon, .fp_armv8 });

pub const has_sve: bool =
    builtin.cpu.arch == .aarch64 and
    builtin.cpu.has(.aarch64, .sve);

pub const has_sve2: bool =
    has_sve and builtin.cpu.has(.aarch64, .sve2);

pub const has_sme: bool =
    builtin.cpu.arch == .aarch64 and
    builtin.cpu.has(.aarch64, .sme);

pub const has_sme2: bool =
    has_sme and builtin.cpu.has(.aarch64, .sme2);

pub const has_sme2p1: bool =
    has_sme2 and builtin.cpu.has(.aarch64, .sme2p1);

pub const has_mops: bool =
    builtin.cpu.arch == .aarch64 and
    builtin.cpu.has(.aarch64, .mops);

pub const has_complxnum: bool =
    builtin.cpu.arch == .aarch64 and
    builtin.cpu.has(.aarch64, .complxnum);

pub const has_sme_f64f64: bool =
    has_sme and builtin.cpu.has(.aarch64, .sme_f64f64);

pub const has_sme_tmop: bool =
    has_sme2 and builtin.cpu.has(.aarch64, .sme_tmop);

pub fn sveVectorBytes() usize {
    if (comptime !has_sve) return 16;
    return asm volatile ("rdvl %[out], #1"
        : [out] "=r" (-> usize),
    );
}

pub fn streamingVectorBytes() usize {
    if (comptime !has_sme) return 16;
    return asm volatile ("rdsvl %[out], #1"
        : [out] "=r" (-> usize),
    );
}

pub const StreamingModeState = struct {
    pub inline fn startSm(self: *StreamingModeState) void {
        if (comptime !has_sme) return;
        _ = self;
        asm volatile (
            \\smstart sm
            ::: streaming_mode_clobbers);
    }

    pub inline fn startSmZa(self: *StreamingModeState) void {
        if (comptime !has_sme) return;
        _ = self;
        asm volatile (
            \\smstart sm
            \\smstart za
            ::: streaming_mode_clobbers);
    }

    pub inline fn stopSm(self: *StreamingModeState) void {
        if (comptime !has_sme) return;
        _ = self;
        asm volatile (
            \\smstop sm
            ::: streaming_mode_clobbers);
    }

    pub inline fn stopSmRetU64(self: *StreamingModeState, result_bits: u64) u64 {
        if (comptime !has_sme) return result_bits;
        _ = self;
        return asm volatile (
            \\mov x0, x11
            \\smstop sm
            : [result] "={x0}" (-> u64),
            : [result_bits] "{x11}" (result_bits),
            : streaming_mode_clobbers);
    }

    pub inline fn stopSmZa(self: *StreamingModeState) void {
        if (comptime !has_sme) return;
        _ = self;
        asm volatile (
            \\smstop za
            \\smstop sm
            ::: streaming_mode_clobbers);
    }

    pub inline fn stopSmZaRetU64(self: *StreamingModeState, result_bits: u64) u64 {
        if (comptime !has_sme) return result_bits;
        _ = self;
        return asm volatile (
            \\mov x0, x11
            \\smstop za
            \\smstop sm
            : [result] "={x0}" (-> u64),
            : [result_bits] "{x11}" (result_bits),
            : streaming_mode_clobbers);
    }

    pub inline fn stopSmZaRetU32(self: *StreamingModeState, result_bits: u32) u32 {
        if (comptime !has_sme) return result_bits;
        _ = self;
        return asm volatile (
            \\mov w0, w11
            \\smstop za
            \\smstop sm
            : [result] "={w0}" (-> u32),
            : [result_bits] "{w11}" (result_bits),
            : streaming_mode_clobbers);
    }
};
