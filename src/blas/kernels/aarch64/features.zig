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

pub fn smstartZa() void {
    if (comptime has_sme) asm volatile ("smstart za" ::: .{ .memory = true });
}

pub fn smstopZa() void {
    if (comptime has_sme) asm volatile ("smstop za" ::: .{ .memory = true });
}
