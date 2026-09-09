// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Baseline-only resolver. ISA objects never run before CPU/OS admission.
const std = @import("std");
const builtin = @import("builtin");
const hardware = @import("../../hardware.zig");
const protocol = @import("protocol.zig");
const catalog = @import("../shared/matrix_matrix/catalog.zig");

pub const Tier = enum(u8) { baseline, aarch64_sve2, aarch64_sme, aarch64_sme2, aarch64_sme2p1, x86_avx, x86_avx2_fma, x86_avx512 };
const Entry = *const fn (u32, u32, *anyopaque) callconv(.c) void;
extern fn zynum_internal_kernel_baseline(u32, u32, *anyopaque) callconv(.c) void;
extern fn zynum_internal_kernel_aarch64_sve2(u32, u32, *anyopaque) callconv(.c) void;
extern fn zynum_internal_kernel_aarch64_sme(u32, u32, *anyopaque) callconv(.c) void;
extern fn zynum_internal_kernel_aarch64_sme2(u32, u32, *anyopaque) callconv(.c) void;
extern fn zynum_internal_kernel_aarch64_sme2p1(u32, u32, *anyopaque) callconv(.c) void;
extern fn zynum_internal_kernel_x86_avx(u32, u32, *anyopaque) callconv(.c) void;
extern fn zynum_internal_kernel_x86_avx2_fma(u32, u32, *anyopaque) callconv(.c) void;
extern fn zynum_internal_kernel_x86_avx512(u32, u32, *anyopaque) callconv(.c) void;

var cached_tier = std.atomic.Value(u8).init(255);

pub fn select(cap: hardware.Capabilities, ceiling: Tier) Tier {
    if (ceiling == .baseline) return .baseline;
    if (comptime builtin.cpu.arch == .aarch64) {
        if (@intFromEnum(ceiling) > 4) return .baseline;
        if (@intFromEnum(ceiling) >= @intFromEnum(Tier.aarch64_sme2p1) and cap.supportsSme2p1() and cap.sme_f64f64) return .aarch64_sme2p1;
        if (@intFromEnum(ceiling) >= @intFromEnum(Tier.aarch64_sme2) and cap.supportsSme2() and cap.sme_f64f64) return .aarch64_sme2;
        if (@intFromEnum(ceiling) >= @intFromEnum(Tier.aarch64_sme) and cap.supportsSme()) return .aarch64_sme;
        if (cap.supportsSve2()) return .aarch64_sve2;
    } else if (comptime builtin.cpu.arch == .x86_64) {
        if (@intFromEnum(ceiling) < 5) return .baseline;
        if (@intFromEnum(ceiling) >= @intFromEnum(Tier.x86_avx512) and cap.supportsAvx512()) return .x86_avx512;
        if (@intFromEnum(ceiling) >= @intFromEnum(Tier.x86_avx2_fma) and cap.supportsAvx2Fma()) return .x86_avx2_fma;
        if (cap.avx and cap.sse and cap.sse2 and cap.sse3 and cap.ssse3 and cap.sse4_1 and cap.sse4_2) return .x86_avx;
    }
    return .baseline;
}

fn ceilingFromEnvironment() Tier {
    const maximum: Tier = switch (builtin.cpu.arch) {
        .aarch64 => .aarch64_sme2p1,
        .x86_64 => .x86_avx512,
        else => .baseline,
    };
    const raw = std.c.getenv("ZYNUM_MAX_ISA") orelse return maximum;
    const name = std.mem.span(raw);
    const value = std.meta.stringToEnum(Tier, name) orelse return .baseline;
    // An override only removes capabilities, never grants them. Wrong-arch
    // names and unknown values conservatively select the baseline object.
    if (value == .baseline) return .baseline;
    if (builtin.cpu.arch == .aarch64 and @intFromEnum(value) <= 4) return value;
    if (builtin.cpu.arch == .x86_64 and @intFromEnum(value) >= 5) return value;
    return .baseline;
}

pub fn selectedTier() Tier {
    const cached = cached_tier.load(.acquire);
    if (cached != 255) return @enumFromInt(cached);
    const detected = select(hardware.detected(), ceilingFromEnvironment());
    // Racing initial callers may compute identical immutable evidence. The
    // first publication wins; changes to the environment after init do not
    // switch live TLS workspaces between ISA objects.
    const previous = cached_tier.cmpxchgStrong(255, @intFromEnum(detected), .acq_rel, .acquire);
    return if (previous) |value| @enumFromInt(value) else detected;
}

fn entry(tier: Tier) Entry {
    if (comptime builtin.cpu.arch == .aarch64) return switch (tier) {
        .aarch64_sve2 => &zynum_internal_kernel_aarch64_sve2,
        .aarch64_sme => &zynum_internal_kernel_aarch64_sme,
        .aarch64_sme2 => &zynum_internal_kernel_aarch64_sme2,
        .aarch64_sme2p1 => &zynum_internal_kernel_aarch64_sme2p1,
        else => &zynum_internal_kernel_baseline,
    };
    if (comptime builtin.cpu.arch == .x86_64) return switch (tier) {
        .x86_avx => &zynum_internal_kernel_x86_avx,
        .x86_avx2_fma => &zynum_internal_kernel_x86_avx2_fma,
        .x86_avx512 => &zynum_internal_kernel_x86_avx512,
        else => &zynum_internal_kernel_baseline,
    };
    return &zynum_internal_kernel_baseline;
}

pub fn call(comptime operation: protocol.Operation, comptime T: type, args: anytype) protocol.Result(operation, T) {
    var packet: protocol.Packet(operation, T) = .{ .args = args, .result = undefined };
    entry(selectedTier())(@intFromEnum(operation), @intFromEnum(protocol.scalarId(T)), &packet);
    return packet.result;
}

pub fn activeCapability() catalog.IsaCapability {
    return switch (selectedTier()) {
        .aarch64_sme, .aarch64_sme2, .aarch64_sme2p1 => .aarch64_sme,
        .aarch64_sve2 => .aarch64_sve2,
        .x86_avx => .x86_64_avx,
        .x86_avx2_fma => .x86_64_avx2_fma,
        .x86_avx512 => .x86_64_avx512f_fma,
        .baseline => switch (builtin.cpu.arch) {
            .aarch64 => .aarch64_asimd_fma,
            .x86_64 => .x86_64_sse2,
            else => .generic,
        },
    };
}
