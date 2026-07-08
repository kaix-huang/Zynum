// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const builtin = @import("builtin");

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
    saved_d8_d15: [8]u64 = undefined,

    pub inline fn startSm(self: *StreamingModeState) void {
        if (comptime !has_sme) return;
        asm volatile (
            \\stp d8, d9, [x10]
            \\stp d10, d11, [x10, #16]
            \\stp d12, d13, [x10, #32]
            \\stp d14, d15, [x10, #48]
            \\smstart sm
            :
            : [saved] "{x10}" (@intFromPtr(&self.saved_d8_d15)),
            : .{ .memory = true });
    }

    pub inline fn startSmZa(self: *StreamingModeState) void {
        if (comptime !has_sme) return;
        asm volatile (
            \\stp d8, d9, [x10]
            \\stp d10, d11, [x10, #16]
            \\stp d12, d13, [x10, #32]
            \\stp d14, d15, [x10, #48]
            \\smstart sm
            \\smstart za
            :
            : [saved] "{x10}" (@intFromPtr(&self.saved_d8_d15)),
            : .{ .memory = true });
    }

    pub inline fn stopSm(self: *StreamingModeState) void {
        if (comptime !has_sme) return;
        asm volatile (
            \\smstop sm
            \\ldp d8, d9, [x10]
            \\ldp d10, d11, [x10, #16]
            \\ldp d12, d13, [x10, #32]
            \\ldp d14, d15, [x10, #48]
            :
            : [saved] "{x10}" (@intFromPtr(&self.saved_d8_d15)),
            : .{ .memory = true });
    }

    pub inline fn stopSmRetU64(self: *StreamingModeState, result_bits: u64) u64 {
        if (comptime !has_sme) return result_bits;
        return asm volatile (
            \\mov x0, x11
            \\smstop sm
            \\ldp d8, d9, [x10]
            \\ldp d10, d11, [x10, #16]
            \\ldp d12, d13, [x10, #32]
            \\ldp d14, d15, [x10, #48]
            : [result] "={x0}" (-> u64),
            : [saved] "{x10}" (@intFromPtr(&self.saved_d8_d15)),
              [result_bits] "{x11}" (result_bits),
            : .{ .memory = true });
    }

    pub inline fn stopSmZa(self: *StreamingModeState) void {
        if (comptime !has_sme) return;
        asm volatile (
            \\smstop za
            \\smstop sm
            \\ldp d8, d9, [x10]
            \\ldp d10, d11, [x10, #16]
            \\ldp d12, d13, [x10, #32]
            \\ldp d14, d15, [x10, #48]
            :
            : [saved] "{x10}" (@intFromPtr(&self.saved_d8_d15)),
            : .{ .memory = true });
    }

    pub inline fn stopSmZaRetU64(self: *StreamingModeState, result_bits: u64) u64 {
        if (comptime !has_sme) return result_bits;
        return asm volatile (
            \\mov x0, x11
            \\smstop za
            \\smstop sm
            \\ldp d8, d9, [x10]
            \\ldp d10, d11, [x10, #16]
            \\ldp d12, d13, [x10, #32]
            \\ldp d14, d15, [x10, #48]
            : [result] "={x0}" (-> u64),
            : [saved] "{x10}" (@intFromPtr(&self.saved_d8_d15)),
              [result_bits] "{x11}" (result_bits),
            : .{ .memory = true });
    }

    pub inline fn stopSmZaRetU32(self: *StreamingModeState, result_bits: u32) u32 {
        if (comptime !has_sme) return result_bits;
        return asm volatile (
            \\mov w0, w11
            \\smstop za
            \\smstop sm
            \\ldp d8, d9, [x10]
            \\ldp d10, d11, [x10, #16]
            \\ldp d12, d13, [x10, #32]
            \\ldp d14, d15, [x10, #48]
            : [result] "={w0}" (-> u32),
            : [saved] "{x10}" (@intFromPtr(&self.saved_d8_d15)),
              [result_bits] "{w11}" (result_bits),
            : .{ .memory = true });
    }
};
