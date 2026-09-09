// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! OS-usable instruction capabilities, independent of the compilation target.
//! Keep this module at the baseline ISA: probing never executes optional vector
//! instructions. A process-wide atomic snapshot avoids repeated OS queries.

const std = @import("std");
const builtin = @import("builtin");

pub const Capabilities = struct {
    neon: bool = false,
    fp_armv8: bool = false,
    fullfp16: bool = false,
    bf16: bool = false,
    sve: bool = false,
    sve2: bool = false,
    sme: bool = false,
    sme2: bool = false,
    sme2p1: bool = false,
    sme_f64f64: bool = false,
    sse: bool = false,
    sse2: bool = false,
    sse3: bool = false,
    ssse3: bool = false,
    sse4_1: bool = false,
    sse4_2: bool = false,
    f16c: bool = false,
    avx: bool = false,
    avx2: bool = false,
    fma: bool = false,
    avx512f: bool = false,
    avx512dq: bool = false,
    avx512bw: bool = false,
    avx512vl: bool = false,

    pub fn supportsAvx2Fma(self: Capabilities) bool {
        // LLVM AVX implies the full SSE chain; SSE4.2 also enables CRC32.
        return self.sse and self.sse2 and self.sse3 and self.ssse3 and self.sse4_1 and self.sse4_2 and
            self.avx and self.avx2 and self.fma;
    }

    pub fn supportsAvx512(self: Capabilities) bool {
        return self.supportsAvx2Fma() and self.f16c and self.avx512f and self.avx512dq and self.avx512bw and self.avx512vl;
    }

    pub fn supportsSve2(self: Capabilities) bool {
        return self.neon and self.fp_armv8 and self.fullfp16 and self.sve and self.sve2;
    }

    pub fn supportsSme(self: Capabilities) bool {
        // These are compiler dependencies of +sme, including non-streaming BF16.
        return self.neon and self.fp_armv8 and self.fullfp16 and self.bf16 and self.sme;
    }

    pub fn supportsSme2(self: Capabilities) bool {
        return self.supportsSme() and self.sme2;
    }

    pub fn supportsSme2p1(self: Capabilities) bool {
        return self.supportsSme2() and self.sme2p1;
    }
};

comptime {
    if (std.meta.fields(Capabilities).len > 31) @compileError("capability cache needs a wider atomic");
}

const initialized: u32 = 1 << 31;
var cache = std.atomic.Value(u32).init(0);

fn encode(value: Capabilities) u32 {
    var bits: u32 = initialized;
    inline for (std.meta.fields(Capabilities), 0..) |field, index| {
        if (@field(value, field.name)) bits |= @as(u32, 1) << index;
    }
    return bits;
}

fn decode(bits: u32) Capabilities {
    var result: Capabilities = .{};
    inline for (std.meta.fields(Capabilities), 0..) |field, index| {
        @field(result, field.name) = (bits & (@as(u32, 1) << index)) != 0;
    }
    return result;
}

pub fn detected() Capabilities {
    const previous = cache.load(.monotonic);
    if (previous != 0) return decode(previous);
    const result = detect();
    // Concurrent first callers may probe twice; the complete value is published
    // atomically, with no separately initialized state for readers to race on.
    cache.store(encode(result), .monotonic);
    return result;
}

fn has(value: u64, comptime bit: u6) bool {
    return value & (@as(u64, 1) << bit) != 0;
}

/// Linux arm64 ELF HWCAP values describe features enabled for userspace by the
/// kernel, rather than features inferred from a processor name or MIDR register.
pub fn fromLinuxAarch64Hwcap(hwcap: u64, hwcap2: u64) Capabilities {
    const sve = has(hwcap, 22);
    const sme = has(hwcap2, 23);
    const sme2 = sme and has(hwcap2, 37);
    return .{
        .neon = has(hwcap, 1),
        .fp_armv8 = has(hwcap, 0),
        .fullfp16 = has(hwcap, 9) and has(hwcap, 10),
        .bf16 = has(hwcap2, 14),
        .sve = sve,
        .sve2 = sve and has(hwcap2, 1),
        .sme = sme,
        .sme2 = sme2,
        .sme2p1 = sme2 and has(hwcap2, 38),
        .sme_f64f64 = sme and has(hwcap2, 25),
    };
}

pub const X86Evidence = struct {
    leaf1_ecx: u32 = 0,
    leaf1_edx: u32 = 0,
    leaf7_ebx: u32 = 0,
    xcr0: u64 = 0,
};

/// CPUID alone is insufficient: AVX needs XMM/YMM state and AVX-512 additionally
/// needs opmask, ZMM high halves, and the high sixteen ZMM registers in XCR0.
pub fn fromX86Evidence(evidence: X86Evidence) Capabilities {
    const xsave = has(evidence.leaf1_ecx, 26) and has(evidence.leaf1_ecx, 27);
    const avx = xsave and has(evidence.leaf1_ecx, 28) and evidence.xcr0 & 0x6 == 0x6;
    const avx512 = avx and evidence.xcr0 & 0xe6 == 0xe6 and has(evidence.leaf7_ebx, 16);
    return .{
        .sse = has(evidence.leaf1_edx, 25),
        .sse2 = has(evidence.leaf1_edx, 26),
        .sse3 = has(evidence.leaf1_ecx, 0),
        .ssse3 = has(evidence.leaf1_ecx, 9),
        .sse4_1 = has(evidence.leaf1_ecx, 19),
        .sse4_2 = has(evidence.leaf1_ecx, 20),
        .f16c = avx and has(evidence.leaf1_ecx, 29),
        .avx = avx,
        .avx2 = avx and has(evidence.leaf7_ebx, 5),
        .fma = avx and has(evidence.leaf1_ecx, 12),
        .avx512f = avx512,
        .avx512dq = avx512 and has(evidence.leaf7_ebx, 17),
        .avx512bw = avx512 and has(evidence.leaf7_ebx, 30),
        .avx512vl = avx512 and has(evidence.leaf7_ebx, 31),
    };
}

fn sysctlFeature(comptime name: [:0]const u8) bool {
    var value: c_int = 0;
    var length: usize = @sizeOf(c_int);
    return std.c.sysctlbyname(name.ptr, &value, &length, null, 0) == 0 and
        length == @sizeOf(c_int) and value == 1;
}

const CpuidLeaf = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

fn cpuid(leaf: u32) CpuidLeaf {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [eax] "={eax}" (eax),
          [ebx] "={ebx}" (ebx),
          [ecx] "={ecx}" (ecx),
          [edx] "={edx}" (edx),
        : [leaf] "{eax}" (leaf),
          [subleaf] "{ecx}" (@as(u32, 0)),
    );
    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

fn detectX86() Capabilities {
    const maximum = cpuid(0).eax;
    if (maximum < 1) return .{};
    const leaf1 = cpuid(1);
    var evidence: X86Evidence = .{ .leaf1_ecx = leaf1.ecx, .leaf1_edx = leaf1.edx };
    if (maximum >= 7) evidence.leaf7_ebx = cpuid(7).ebx;
    if (has(leaf1.ecx, 26) and has(leaf1.ecx, 27) and has(leaf1.ecx, 28)) {
        var low: u32 = undefined;
        var high: u32 = undefined;
        asm volatile ("xgetbv"
            : [low] "={eax}" (low),
              [high] "={edx}" (high),
            : [index] "{ecx}" (@as(u32, 0)),
        );
        evidence.xcr0 = @as(u64, high) << 32 | low;
    }
    return fromX86Evidence(evidence);
}

fn detect() Capabilities {
    if (comptime builtin.cpu.arch == .x86_64) return detectX86();
    if (comptime builtin.cpu.arch != .aarch64) return .{};
    if (comptime builtin.os.tag == .linux) {
        // Shared libraries do not run Zig executable startup, so its private
        // auxv pointer is unset. Use libc's process auxv when libc is linked.
        const getauxval = if (builtin.link_libc) std.c.getauxval else std.os.linux.getauxval;
        return fromLinuxAarch64Hwcap(
            getauxval(std.elf.AT_HWCAP),
            getauxval(std.elf.AT_HWCAP2),
        );
    }
    if (comptime builtin.os.tag == .macos) {
        const sve = sysctlFeature("hw.optional.arm.FEAT_SVE");
        const sme = sysctlFeature("hw.optional.arm.FEAT_SME");
        const sme2 = sme and sysctlFeature("hw.optional.arm.FEAT_SME2");
        return .{
            .neon = sysctlFeature("hw.optional.neon"),
            // Scalar FP is part of the arm64 macOS userspace ABI.
            .fp_armv8 = true,
            .fullfp16 = sysctlFeature("hw.optional.arm.FEAT_FP16"),
            .bf16 = sysctlFeature("hw.optional.arm.FEAT_BF16"),
            .sve = sve,
            .sve2 = sve and sysctlFeature("hw.optional.arm.FEAT_SVE2"),
            .sme = sme,
            .sme2 = sme2,
            .sme2p1 = sme2 and sysctlFeature("hw.optional.arm.FEAT_SME2p1"),
            .sme_f64f64 = sme and sysctlFeature("hw.optional.arm.FEAT_SME_F64F64"),
        };
    }
    return .{};
}

test "x86 vector tiers require CPU support and every OS state component" {
    const full: X86Evidence = .{
        .leaf1_ecx = (1 << 0) | (1 << 9) | (1 << 12) | (1 << 19) | (1 << 20) | (1 << 26) | (1 << 27) | (1 << 28) | (1 << 29),
        .leaf1_edx = (1 << 25) | (1 << 26),
        .leaf7_ebx = (1 << 5) | (1 << 16) | (1 << 17) | (1 << 30) | (1 << 31),
        .xcr0 = 0xe6,
    };
    try std.testing.expect(fromX86Evidence(full).supportsAvx512());
    for ([_]u6{ 26, 27, 28 }) |bit| {
        var missing = full;
        missing.leaf1_ecx &= ~(@as(u32, 1) << @as(u5, @intCast(bit)));
        try std.testing.expect(!fromX86Evidence(missing).avx);
    }
    for ([_]u6{ 1, 2, 5, 6, 7 }) |bit| {
        var missing = full;
        missing.xcr0 &= ~(@as(u64, 1) << bit);
        try std.testing.expect(!fromX86Evidence(missing).supportsAvx512());
        try std.testing.expectEqual(bit >= 5, fromX86Evidence(missing).supportsAvx2Fma());
    }
    for ([_]u5{ 5, 16, 17, 30, 31 }) |bit| {
        var missing = full;
        missing.leaf7_ebx &= ~(@as(u32, 1) << bit);
        try std.testing.expect(!fromX86Evidence(missing).supportsAvx512());
    }
    for ([_]u5{ 0, 9, 19, 20, 29 }) |bit| {
        var missing = full;
        missing.leaf1_ecx &= ~(@as(u32, 1) << bit);
        try std.testing.expect(!fromX86Evidence(missing).supportsAvx512());
        try std.testing.expectEqual(bit == 29, fromX86Evidence(missing).supportsAvx2Fma());
    }
    var no_fma = full;
    no_fma.leaf1_ecx &= ~@as(u32, 1 << 12);
    try std.testing.expect(!fromX86Evidence(no_fma).supportsAvx2Fma());
}

test "aarch64 capability tiers fail closed without parent features" {
    const hwcap: u64 = (1 << 0) | (1 << 1) | (1 << 9) | (1 << 10) | (1 << 22);
    const hwcap2: u64 = (1 << 1) | (1 << 14) | (1 << 23) | (1 << 25) | (1 << 37) | (1 << 38);
    const full = fromLinuxAarch64Hwcap(hwcap, hwcap2);
    try std.testing.expect(full.supportsSme2p1() and full.sme_f64f64 and full.sve2);
    try std.testing.expect(full.supportsSve2());
    for ([_]u6{ 0, 1, 9, 10 }) |bit| {
        const missing = fromLinuxAarch64Hwcap(hwcap & ~(@as(u64, 1) << bit), hwcap2);
        try std.testing.expect(!missing.supportsSve2());
        try std.testing.expect(!missing.supportsSme2());
    }
    const no_bf16 = fromLinuxAarch64Hwcap(hwcap, hwcap2 & ~@as(u64, 1 << 14));
    try std.testing.expect(no_bf16.supportsSve2() and !no_bf16.supportsSme());
    const no_sme = fromLinuxAarch64Hwcap(hwcap, hwcap2 & ~@as(u64, 1 << 23));
    try std.testing.expect(!no_sme.sme2 and !no_sme.sme2p1 and !no_sme.sme_f64f64);
    const no_sme2 = fromLinuxAarch64Hwcap(hwcap, hwcap2 & ~@as(u64, 1 << 37));
    try std.testing.expect(no_sme2.supportsSme() and !no_sme2.supportsSme2p1());
    try std.testing.expect(!fromLinuxAarch64Hwcap(0, hwcap2).sve2);
    try std.testing.expect(!fromLinuxAarch64Hwcap(0, hwcap2).supportsSme());
}

test "capability cache representation preserves empty and complete snapshots" {
    const empty: Capabilities = .{};
    try std.testing.expect(encode(empty) != 0);
    try std.testing.expectEqualDeep(empty, decode(encode(empty)));
    var all: Capabilities = .{};
    inline for (std.meta.fields(Capabilities)) |field| @field(all, field.name) = true;
    try std.testing.expectEqualDeep(all, decode(encode(all)));
    try std.testing.expectEqualDeep(detected(), detected());
    if (comptime builtin.cpu.arch == .aarch64 and builtin.os.tag == .linux and builtin.link_libc) {
        // This also checks the library-safe provider against libc directly.
        const expected = fromLinuxAarch64Hwcap(
            std.c.getauxval(std.elf.AT_HWCAP),
            std.c.getauxval(std.elf.AT_HWCAP2),
        );
        try std.testing.expectEqualDeep(expected, detected());
    }
}
