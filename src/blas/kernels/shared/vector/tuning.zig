// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Named Level 1 preference profiles.
//!
//! This module owns measured enablement and length/shape cutoffs. ISA presence,
//! streaming-vector length compatibility, aliasing, alignment, and state
//! ownership remain hard feasibility checks at the implementation boundary.

const std = @import("std");
const root = @import("root");

pub const Aarch64Profile = struct {
    enable_asimd_scal_f64: bool,
    enable_sme_scal: bool,
    enable_sve_scal_f32: bool,
    enable_sve_scal_f64: bool,
    enable_sme_asum: bool,
    enable_sve_asum: bool,
    enable_mops_copy: bool,
    enable_sme_copy: bool,
    enable_sme_swap: bool,
    enable_sme_axpy: bool,
    enable_sve_axpy_complex_f32: bool,
    enable_sve_axpy_complex_f64: bool,
    enable_sme_axpby: bool,
    enable_fixed_axpby: bool,
    enable_sme_dot: bool,
    enable_sve_dot_f32: bool,
    enable_sve_dot: bool,
    enable_sve_dot_complex_f32: bool,
    enable_sve_dot_complex_f64: bool,
    enable_fixed_dot_f32_acc_f64: bool,
    enable_fixed_complex_iamax: bool,
    enable_sme_linear_transform: bool,
    enable_fixed_rot: bool,
    enable_fixed_rotm: bool,

    short_vector_min_elements: usize,
    streaming_min_elements: usize,
    sve_complex_dot_min_elements: usize,
    asimd_swap_min_bytes: usize,
    sme_copy_exact_bytes: usize,
    sme_copy_min_bytes: usize,
    sme_copy_max_bytes_exclusive: usize,
    sme_swap_min_bytes: usize,
    sme_swap_max_bytes_inclusive: usize,
    sme_axpy_f64_max_bytes_exclusive: usize,
    sme_rot_f32_max_bytes_inclusive: usize,
    fixed_complex_iamax_max_elements: usize,

    pub fn preferAsimdScal(self: Aarch64Profile, comptime T: type, n: usize) bool {
        return self.enable_asimd_scal_f64 and T == f64 and n >= self.short_vector_min_elements;
    }

    pub fn preferSmeScal(self: Aarch64Profile, comptime T: type, n: usize) bool {
        return self.enable_sme_scal and (T == f32 or T == f64) and n >= self.streaming_min_elements;
    }

    pub fn preferSveScal(self: Aarch64Profile, comptime T: type, n: usize) bool {
        return ((self.enable_sve_scal_f32 and T == f32) or
            (self.enable_sve_scal_f64 and T == f64)) and
            n >= self.short_vector_min_elements;
    }

    pub fn preferSmeAsum(self: Aarch64Profile, comptime T: type, n: usize) bool {
        return self.enable_sme_asum and (T == f32 or T == f64) and n >= self.streaming_min_elements;
    }

    pub fn preferSveAsum(self: Aarch64Profile, comptime T: type, n: usize) bool {
        if (!self.enable_sve_asum) return false;
        if (T == f32) return n >= self.streaming_min_elements;
        return T == f64 and n >= self.short_vector_min_elements;
    }

    pub fn preferSmeCopyExact(self: Aarch64Profile, n_bytes: usize) bool {
        return self.enable_sme_copy and n_bytes == self.sme_copy_exact_bytes;
    }

    pub fn preferAsimdCopy(self: Aarch64Profile, n_bytes: usize) bool {
        return n_bytes < self.sme_copy_min_bytes;
    }

    pub fn preferMopsCopy(self: Aarch64Profile, n_bytes: usize) bool {
        return self.enable_mops_copy and n_bytes >= self.sme_copy_min_bytes;
    }

    pub fn preferSmeCopy(self: Aarch64Profile, n_bytes: usize) bool {
        return self.enable_sme_copy and
            n_bytes >= self.sme_copy_min_bytes and
            n_bytes < self.sme_copy_max_bytes_exclusive;
    }

    pub fn preferSmeSwap(self: Aarch64Profile, n_bytes: usize) bool {
        return self.enable_sme_swap and
            n_bytes >= self.sme_swap_min_bytes and
            n_bytes <= self.sme_swap_max_bytes_inclusive;
    }

    pub fn preferAsimdSwap(self: Aarch64Profile, n_bytes: usize) bool {
        return n_bytes >= self.asimd_swap_min_bytes;
    }

    pub fn preferSmeAxpy(self: Aarch64Profile, comptime T: type, n: usize) bool {
        if (!self.enable_sme_axpy or n < self.streaming_min_elements) return false;
        if (T == f32) return true;
        return T == f64 and n * @sizeOf(f64) < self.sme_axpy_f64_max_bytes_exclusive;
    }

    pub fn preferSveAxpyComplexF64(self: Aarch64Profile, n: usize) bool {
        return self.enable_sve_axpy_complex_f64 and n >= self.streaming_min_elements;
    }

    pub fn preferSveAxpyComplexF32(self: Aarch64Profile, n: usize) bool {
        return self.enable_sve_axpy_complex_f32 and n >= self.streaming_min_elements;
    }

    pub fn preferSmeAxpby(self: Aarch64Profile, comptime T: type, n: usize) bool {
        return self.enable_sme_axpby and T == f32 and n >= self.streaming_min_elements;
    }

    pub fn preferSmeDot(self: Aarch64Profile, comptime T: type, n: usize) bool {
        return self.enable_sme_dot and (T == f32 or T == f64) and n >= self.streaming_min_elements;
    }

    pub fn preferSveDot(self: Aarch64Profile, comptime T: type, n: usize) bool {
        return ((self.enable_sve_dot_f32 and T == f32) or
            (self.enable_sve_dot and T == f64)) and
            n >= self.short_vector_min_elements;
    }

    pub fn preferSveDotComplexF64(self: Aarch64Profile, n: usize) bool {
        return self.enable_sve_dot_complex_f64 and n >= self.sve_complex_dot_min_elements;
    }

    pub fn preferSveDotComplexF32(self: Aarch64Profile, n: usize) bool {
        return self.enable_sve_dot_complex_f32 and n >= self.sve_complex_dot_min_elements;
    }

    pub fn preferSmeRot(self: Aarch64Profile, comptime T: type, n: usize) bool {
        return self.enable_sme_linear_transform and T == f32 and
            n >= self.streaming_min_elements and
            n * @sizeOf(f32) <= self.sme_rot_f32_max_bytes_inclusive;
    }

    pub fn preferSmeRotm(self: Aarch64Profile, comptime T: type, n: usize) bool {
        return self.enable_sme_linear_transform and T == f32 and n >= self.streaming_min_elements;
    }

    pub fn preferFixedComplexIamax(self: Aarch64Profile, n: usize) bool {
        return self.enable_fixed_complex_iamax and n <= self.fixed_complex_iamax_max_elements;
    }
};

pub const X86_64Profile = struct {
    enable_fixed_dot_f32_acc_f64: bool,
    enable_fixed_complex_iamax: bool,
    enable_fixed_rotm: bool,
    use_avx2_width_axpy_on_avx512: bool,
    use_avx2_width_f32_dot_on_avx512: bool,
    use_avx2_width_f32_asum_on_avx512: bool,
    use_avx2_width_f32_complex_asum_on_avx512: bool,
    core_copy_exact_bytes: usize,
    core_copy_min_bytes: usize,
    core_copy_max_bytes_exclusive: usize,
    stride2_parallel_min_elements: usize,
    fixed_dot_f32_acc_f64_min_elements: usize,
    fixed_complex_iamax_min_elements: usize,

    pub fn preferCoreCopy(self: X86_64Profile, n_bytes: usize) bool {
        return n_bytes == self.core_copy_exact_bytes or
            (n_bytes >= self.core_copy_min_bytes and n_bytes < self.core_copy_max_bytes_exclusive);
    }

    pub fn preferStride2Parallel(self: X86_64Profile, n: usize) bool {
        return n >= self.stride2_parallel_min_elements;
    }

    pub fn preferFixedDotF32AccF64(self: X86_64Profile, n: usize) bool {
        return self.enable_fixed_dot_f32_acc_f64 and
            n >= self.fixed_dot_f32_acc_f64_min_elements;
    }

    pub fn preferFixedComplexIamax(self: X86_64Profile, n: usize) bool {
        return self.enable_fixed_complex_iamax and
            n >= self.fixed_complex_iamax_min_elements;
    }

    pub fn preferAvx2WidthAxpy(self: X86_64Profile) bool {
        return self.use_avx2_width_axpy_on_avx512;
    }

    pub fn preferAvx2WidthDot(self: X86_64Profile, comptime T: type) bool {
        return self.use_avx2_width_f32_dot_on_avx512 and T == f32;
    }

    pub fn preferAvx2WidthAsum(self: X86_64Profile, comptime T: type) bool {
        return self.use_avx2_width_f32_asum_on_avx512 and T == f32;
    }

    pub fn preferAvx2WidthComplexAsum(self: X86_64Profile, comptime T: type) bool {
        return self.use_avx2_width_f32_complex_asum_on_avx512 and T == f32;
    }
};

pub const Profile = struct {
    aarch64: Aarch64Profile,
    x86_64: X86_64Profile,
};

/// Production choices immediately before the Level 1 registry migration.
/// Keep this name stable so future native profiles can be compared explicitly.
pub const production_2026_07_17: Profile = .{
    .aarch64 = .{
        .enable_asimd_scal_f64 = false,
        .enable_sme_scal = true,
        .enable_sve_scal_f32 = false,
        .enable_sve_scal_f64 = false,
        .enable_sme_asum = true,
        .enable_sve_asum = false,
        .enable_mops_copy = true,
        .enable_sme_copy = true,
        .enable_sme_swap = true,
        .enable_sme_axpy = true,
        .enable_sve_axpy_complex_f32 = false,
        .enable_sve_axpy_complex_f64 = false,
        .enable_sme_axpby = true,
        .enable_fixed_axpby = false,
        .enable_sme_dot = true,
        .enable_sve_dot_f32 = false,
        .enable_sve_dot = false,
        .enable_sve_dot_complex_f32 = false,
        .enable_sve_dot_complex_f64 = false,
        .enable_fixed_dot_f32_acc_f64 = false,
        // Fresh-process crossover controls retain the fixed ASIMD leaf through
        // n=256 for both complex precisions. At n=512 the portable wide-vector
        // body wins again, so the production gate has a closed upper bound.
        .enable_fixed_complex_iamax = true,
        .enable_sme_linear_transform = true,
        .enable_fixed_rot = false,
        .enable_fixed_rotm = false,
        .short_vector_min_elements = 16,
        .streaming_min_elements = 64 * 1024,
        .sve_complex_dot_min_elements = 64,
        .asimd_swap_min_bytes = 128,
        .sme_copy_exact_bytes = 8 * 1024,
        .sme_copy_min_bytes = 8 * 1024,
        .sme_copy_max_bytes_exclusive = 16 * 1024 * 1024,
        .sme_swap_min_bytes = 64 * 1024,
        .sme_swap_max_bytes_inclusive = 8 * 1024 * 1024,
        .sme_axpy_f64_max_bytes_exclusive = 8 * 1024 * 1024,
        .sme_rot_f32_max_bytes_inclusive = 4 * 1024 * 1024,
        .fixed_complex_iamax_max_elements = 256,
    },
    .x86_64 = .{
        // Forced-path correctness and same-ISA fresh-process controls cover
        // these leaves. The retained lower bounds exclude the short mixed-DOT
        // regression and the marginal tiny complex-IAMAX range; isolated
        // boundary controls confirm the production gates.
        .enable_fixed_dot_f32_acc_f64 = true,
        .enable_fixed_complex_iamax = true,
        .enable_fixed_rotm = false,
        // Fresh-process AVX-512F/FMA target-tier controls showed that the
        // AVX2/FMA-width leaf repeatedly beats the AVX-512-width leaf for real
        // AXPY, f32 DOT, and f32 ASUM from 64 Ki to 1 Mi elements. A runtime
        // f64 ASUM width gate was rejected because retaining both bodies
        // perturbed code layout; f64 ASUM, NRM2, and f64 DOT retain AVX-512
        // width throughout.
        .use_avx2_width_axpy_on_avx512 = true,
        .use_avx2_width_f32_dot_on_avx512 = true,
        .use_avx2_width_f32_asum_on_avx512 = true,
        // Fresh-process complex-f32 ASUM controls retain 512-bit width in both
        // single-task and parallel component leaves. Narrowing repeated ratios
        // of 0.683 at 64 Ki and 0.931 at 512 Ki complex elements against the
        // retained AVX-512F/FMA body.
        .use_avx2_width_f32_complex_asum_on_avx512 = false,
        .core_copy_exact_bytes = 8 * 1024,
        .core_copy_min_bytes = 32 * 1024,
        .core_copy_max_bytes_exclusive = 128 * 1024,
        .stride2_parallel_min_elements = 512 * 1024,
        .fixed_dot_f32_acc_f64_min_elements = 64 * 1024,
        .fixed_complex_iamax_min_elements = 4 * 1024,
    },
};

fn enableFixedCandidates(base: Profile) Profile {
    var result = base;
    result.aarch64.enable_fixed_axpby = true;
    result.aarch64.enable_fixed_dot_f32_acc_f64 = true;
    result.aarch64.enable_fixed_complex_iamax = true;
    result.aarch64.enable_fixed_rot = true;
    result.aarch64.enable_fixed_rotm = true;
    result.aarch64.fixed_complex_iamax_max_elements = std.math.maxInt(usize);
    result.x86_64.enable_fixed_dot_f32_acc_f64 = true;
    result.x86_64.enable_fixed_complex_iamax = true;
    result.x86_64.enable_fixed_rotm = true;
    result.x86_64.fixed_dot_f32_acc_f64_min_elements = 0;
    result.x86_64.fixed_complex_iamax_min_elements = 0;
    return result;
}

/// Explicit benchmark profile for registered experimental leaves. Selection
/// is rooted at a distinct ABI export module so ordinary builds and direct
/// tests keep the production profile without generated global options.
pub const fixed_candidates_2026_07_17 = enableFixedCandidates(production_2026_07_17);

fn enableSveCandidates(base: Profile) Profile {
    var result = base;
    result.aarch64.enable_sve_scal_f32 = true;
    result.aarch64.enable_sve_scal_f64 = true;
    result.aarch64.enable_sve_asum = true;
    result.aarch64.enable_sve_axpy_complex_f32 = true;
    result.aarch64.enable_sve_axpy_complex_f64 = true;
    result.aarch64.enable_sve_dot_f32 = true;
    result.aarch64.enable_sve_dot = true;
    result.aarch64.enable_sve_dot_complex_f32 = true;
    result.aarch64.enable_sve_dot_complex_f64 = true;
    return result;
}

/// Explicit benchmark profile for SVE operation/type counterparts whose
/// production promotion still requires fresh-process native evidence.
pub const sve_candidates_2026_07_17 = enableSveCandidates(production_2026_07_17);

const use_fixed_candidates = if (@hasDecl(root, "zynum_level1_fixed_candidates"))
    root.zynum_level1_fixed_candidates
else
    false;

const use_sve_candidates = if (@hasDecl(root, "zynum_level1_sve_candidates"))
    root.zynum_level1_sve_candidates
else
    false;

pub const active = if (use_sve_candidates)
    sve_candidates_2026_07_17
else if (use_fixed_candidates)
    fixed_candidates_2026_07_17
else
    production_2026_07_17;

test "named Level 1 production profile preserves boundary behavior" {
    const arm = production_2026_07_17.aarch64;
    try std.testing.expect(!arm.preferSmeScal(f32, arm.streaming_min_elements - 1));
    try std.testing.expect(arm.preferSmeScal(f32, arm.streaming_min_elements));
    try std.testing.expect(arm.preferSmeCopy(arm.sme_copy_min_bytes));
    try std.testing.expect(!arm.preferSmeCopy(arm.sme_copy_max_bytes_exclusive));
    try std.testing.expect(arm.preferSmeSwap(arm.sme_swap_max_bytes_inclusive));
    try std.testing.expect(!arm.preferSmeSwap(arm.sme_swap_max_bytes_inclusive + 1));
    try std.testing.expect(arm.preferFixedComplexIamax(arm.fixed_complex_iamax_max_elements));
    try std.testing.expect(!arm.preferFixedComplexIamax(arm.fixed_complex_iamax_max_elements + 1));
    try std.testing.expect(!arm.enable_sve_scal_f32);
    try std.testing.expect(!arm.enable_sve_scal_f64);
    try std.testing.expect(!arm.enable_sve_asum);
    try std.testing.expect(!arm.enable_sve_axpy_complex_f32);
    try std.testing.expect(!arm.enable_sve_axpy_complex_f64);
    try std.testing.expect(!arm.enable_sve_dot_f32);
    try std.testing.expect(!arm.enable_sve_dot);
    try std.testing.expect(!arm.enable_sve_dot_complex_f32);
    try std.testing.expect(!arm.enable_sve_dot_complex_f64);

    const x86 = production_2026_07_17.x86_64;
    try std.testing.expect(x86.enable_fixed_dot_f32_acc_f64);
    try std.testing.expect(x86.enable_fixed_complex_iamax);
    try std.testing.expect(!x86.enable_fixed_rotm);
    try std.testing.expect(!x86.preferFixedDotF32AccF64(x86.fixed_dot_f32_acc_f64_min_elements - 1));
    try std.testing.expect(x86.preferFixedDotF32AccF64(x86.fixed_dot_f32_acc_f64_min_elements));
    try std.testing.expect(!x86.preferFixedComplexIamax(x86.fixed_complex_iamax_min_elements - 1));
    try std.testing.expect(x86.preferFixedComplexIamax(x86.fixed_complex_iamax_min_elements));
    try std.testing.expect(x86.preferAvx2WidthAxpy());
    try std.testing.expect(x86.preferAvx2WidthDot(f32));
    try std.testing.expect(!x86.preferAvx2WidthDot(f64));
    try std.testing.expect(x86.preferAvx2WidthAsum(f32));
    try std.testing.expect(!x86.preferAvx2WidthAsum(f64));
    try std.testing.expect(!x86.preferAvx2WidthComplexAsum(f32));
    try std.testing.expect(x86.preferCoreCopy(x86.core_copy_exact_bytes));
    try std.testing.expect(!x86.preferCoreCopy(x86.core_copy_exact_bytes - 1));
    try std.testing.expect(x86.preferCoreCopy(x86.core_copy_min_bytes));
    try std.testing.expect(!x86.preferCoreCopy(x86.core_copy_max_bytes_exclusive));
}

test "experimental fixed candidate profile changes only candidate switches" {
    const candidate = fixed_candidates_2026_07_17;
    try std.testing.expect(candidate.aarch64.enable_fixed_axpby);
    try std.testing.expect(candidate.aarch64.enable_fixed_dot_f32_acc_f64);
    try std.testing.expect(candidate.aarch64.enable_fixed_complex_iamax);
    try std.testing.expect(candidate.aarch64.enable_fixed_rot);
    try std.testing.expect(candidate.aarch64.enable_fixed_rotm);
    try std.testing.expect(candidate.x86_64.enable_fixed_dot_f32_acc_f64);
    try std.testing.expect(candidate.x86_64.enable_fixed_complex_iamax);
    try std.testing.expect(candidate.x86_64.enable_fixed_rotm);

    var restored = candidate;
    restored.aarch64.enable_fixed_axpby = production_2026_07_17.aarch64.enable_fixed_axpby;
    restored.aarch64.enable_fixed_dot_f32_acc_f64 = false;
    restored.aarch64.enable_fixed_complex_iamax = production_2026_07_17.aarch64.enable_fixed_complex_iamax;
    restored.aarch64.enable_fixed_rot = production_2026_07_17.aarch64.enable_fixed_rot;
    restored.aarch64.enable_fixed_rotm = false;
    restored.aarch64.fixed_complex_iamax_max_elements = production_2026_07_17.aarch64.fixed_complex_iamax_max_elements;
    restored.x86_64.enable_fixed_dot_f32_acc_f64 = production_2026_07_17.x86_64.enable_fixed_dot_f32_acc_f64;
    restored.x86_64.enable_fixed_complex_iamax = production_2026_07_17.x86_64.enable_fixed_complex_iamax;
    restored.x86_64.enable_fixed_rotm = production_2026_07_17.x86_64.enable_fixed_rotm;
    restored.x86_64.fixed_dot_f32_acc_f64_min_elements = production_2026_07_17.x86_64.fixed_dot_f32_acc_f64_min_elements;
    restored.x86_64.fixed_complex_iamax_min_elements = production_2026_07_17.x86_64.fixed_complex_iamax_min_elements;
    try std.testing.expectEqualDeep(production_2026_07_17, restored);
}

test "experimental SVE candidate profile changes only SVE candidate switches" {
    const candidate = sve_candidates_2026_07_17;
    try std.testing.expect(candidate.aarch64.enable_sve_scal_f32);
    try std.testing.expect(candidate.aarch64.enable_sve_scal_f64);
    try std.testing.expect(candidate.aarch64.enable_sve_asum);
    try std.testing.expect(candidate.aarch64.enable_sve_axpy_complex_f32);
    try std.testing.expect(candidate.aarch64.enable_sve_axpy_complex_f64);
    try std.testing.expect(candidate.aarch64.enable_sve_dot_f32);
    try std.testing.expect(candidate.aarch64.enable_sve_dot);
    try std.testing.expect(candidate.aarch64.enable_sve_dot_complex_f32);
    try std.testing.expect(candidate.aarch64.enable_sve_dot_complex_f64);

    var restored = candidate;
    restored.aarch64.enable_sve_scal_f32 = false;
    restored.aarch64.enable_sve_scal_f64 = false;
    restored.aarch64.enable_sve_asum = false;
    restored.aarch64.enable_sve_axpy_complex_f32 = false;
    restored.aarch64.enable_sve_axpy_complex_f64 = false;
    restored.aarch64.enable_sve_dot_f32 = false;
    restored.aarch64.enable_sve_dot = false;
    restored.aarch64.enable_sve_dot_complex_f32 = false;
    restored.aarch64.enable_sve_dot_complex_f64 = false;
    try std.testing.expectEqualDeep(production_2026_07_17, restored);
}
