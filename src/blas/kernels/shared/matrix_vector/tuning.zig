// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Level 2 hard feasibility and named production preference policy.
//!
//! The catalog describes executable facts. This module first filters those
//! facts using only semantic constraints, then applies the retained measured
//! shape, workspace, and task-count gates. Traversal order and all writes stay
//! in the core operation implementations.

const std = @import("std");
const builtin = @import("builtin");
const root = @import("root");

const contract = @import("../../contract.zig");
const catalog = @import("catalog.zig");
const types = @import("../../../types.zig");

pub const Request = struct {
    m: usize,
    n: usize,
    bandwidth: ?usize = null,
    incx: types.BlasInt,
    incy: types.BlasInt = 0,
};

pub const FeasibilityReason = enum {
    feasible,
    empty_problem,
    invalid_x_stride,
    invalid_y_stride,
    missing_bandwidth,
};

pub const Feasibility = struct {
    reason: FeasibilityReason,

    pub fn isFeasible(self: Feasibility) bool {
        return self.reason == .feasible;
    }
};

fn strideAllows(rule: contract.VectorStrideRule, stride: types.BlasInt) bool {
    return switch (rule) {
        .not_applicable => true,
        .any_nonzero => stride != 0,
        .positive => stride > 0,
        .unit => stride == 1,
        .exactly_two => stride == 2,
    };
}

pub inline fn hardFeasibility(descriptor: catalog.Descriptor, request: Request) Feasibility {
    if (request.m == 0 or request.n == 0) return .{ .reason = .empty_problem };
    if (!strideAllows(descriptor.strides.x, request.incx)) return .{ .reason = .invalid_x_stride };
    if (!strideAllows(descriptor.strides.y, request.incy)) return .{ .reason = .invalid_y_stride };
    if ((descriptor.storage == .general_band or
        descriptor.storage == .symmetric_band or
        descriptor.storage == .triangular_band) and request.bandwidth == null)
    {
        return .{ .reason = .missing_bandwidth };
    }
    return .{ .reason = .feasible };
}

pub fn scalarKind(comptime T: type) contract.ScalarKind {
    return switch (T) {
        f32 => .f32,
        f64 => .f64,
        types.ComplexF32 => .complex_f32,
        types.ComplexF64 => .complex_f64,
        else => @compileError("unsupported Level 2 scalar type"),
    };
}

/// Choose between the unit-stride execution layer and its terminal semantic
/// fallback. Performance thresholds never affect this hard selection.
pub inline fn selectDefault(
    comptime T: type,
    comptime operation: catalog.Level2Operation,
    request: Request,
) catalog.Descriptor {
    const scalar = comptime scalarKind(T);
    if (comptime catalog.findImplementation(operation, scalar, .core_unit)) |unit| {
        if (unit.lifecycle.defaultEligible() and hardFeasibility(unit, request).isFeasible()) return unit;
    }
    return comptime catalog.findImplementation(operation, scalar, .portable_scalar).?;
}

pub inline fn coreUnitSelected(
    comptime T: type,
    comptime operation: catalog.Level2Operation,
    request: Request,
) bool {
    return selectDefault(T, operation, request).kernel.implementation == .core_unit;
}

pub const GemvProfile = struct {
    enable_x86_narrow_width: bool,
    private_workspace_max_bytes: usize,
    no_trans_packed_min_work: usize,
    no_trans_parallel_min_work: usize,
    trans_f32_min_work: usize,
    trans_f64_min_work: usize,
    complex_parallel_min_work: usize,
    gbmv_min_dimension: usize,
    gbmv_min_bandwidth: usize,
    gbmv_c32_conjugate_min_dimension: usize,

    pub fn noTransPackedMinBlocks(self: GemvProfile) usize {
        _ = self;
        return 8;
    }

    pub fn noTransRowMinRows(self: GemvProfile) usize {
        _ = self;
        return 128;
    }

    pub fn noTransColumnMinColumns(self: GemvProfile) usize {
        _ = self;
        return 256;
    }

    pub fn transMinColumns(self: GemvProfile) usize {
        _ = self;
        return 64;
    }

    pub fn transBlockColumns(self: GemvProfile) usize {
        _ = self;
        return 16;
    }

    pub fn capTransTasks(self: GemvProfile, task_count: usize, n: usize) usize {
        _ = self;
        return if (n <= 1536) @min(task_count, 8) else task_count;
    }

    pub fn complexMinColumns(self: GemvProfile, comptime T: type) usize {
        _ = self;
        return if (T == types.ComplexF32) 64 else 48;
    }

    pub fn complexBlockColumns(self: GemvProfile) usize {
        _ = self;
        return 4;
    }

    pub fn capComplexTasks(
        self: GemvProfile,
        comptime T: type,
        m: usize,
        n: usize,
        task_count: usize,
        no_trans: bool,
    ) usize {
        _ = self;
        const exact_c64_512 = no_trans and T == types.ComplexF64 and m == 512 and n == 512;
        const cap: usize = if (exact_c64_512)
            10
        else if (T == types.ComplexF64 and n >= 256 and n < 512)
            10
        else if (n < 512)
            4
        else if (T == types.ComplexF32)
            10
        else
            8;
        return @min(task_count, cap);
    }

    pub fn useC64BalancedColumnRanges(
        self: GemvProfile,
        comptime T: type,
        m: usize,
        n: usize,
        task_count: usize,
    ) bool {
        _ = self;
        return T == types.ComplexF64 and m == 512 and n == 512 and task_count == 10;
    }

    pub fn preferComplexTiledFull(self: GemvProfile, comptime T: type, m: usize, n: usize) bool {
        _ = self;
        return (T == types.ComplexF32 or T == types.ComplexF64) and m == 256 and n == 256;
    }

    pub fn preferNoTransRowsComplex(self: GemvProfile, comptime T: type, m: usize, n: usize, lda: types.BlasInt) bool {
        _ = self;
        return T == types.ComplexF64 and m == 512 and n == 512 and lda == 512;
    }

    pub fn preferTransTaskFullComplex(self: GemvProfile, comptime T: type, m: usize, n: usize) bool {
        _ = self;
        return T == types.ComplexF64 and m == 512 and n == 512;
    }

    pub fn preferNoTransComplexWideColumns(self: GemvProfile, m: usize) bool {
        _ = self;
        return m >= 128;
    }

    pub fn preferTransComplexWideColumns(self: GemvProfile, comptime T: type, m: usize) bool {
        _ = self;
        return if (T == types.ComplexF32)
            (m == 128 or m == 256 or m >= 512)
        else
            T == types.ComplexF64 and m >= 128;
    }

    pub fn preferC64TransTask(self: GemvProfile, comptime T: type, m: usize, n: usize, conjugate: bool) bool {
        _ = self;
        return T == types.ComplexF64 and m == 512 and n == 64 and !conjugate;
    }

    pub fn preferC64TransTiledTask(self: GemvProfile, comptime T: type, m: usize, conjugate: bool) bool {
        _ = self;
        return T == types.ComplexF64 and m == 256 and !conjugate;
    }

    pub fn preferNoTransPacked(self: GemvProfile, m: usize, n: usize) bool {
        return m *| n >= self.no_trans_packed_min_work;
    }

    pub fn preferNoTransParallel(self: GemvProfile, m: usize, n: usize) bool {
        return m *| n >= self.no_trans_parallel_min_work;
    }

    pub fn preferTransParallel(self: GemvProfile, comptime T: type, m: usize, n: usize) bool {
        const min_work = if (T == f32 or (T == f64 and m == 512 and n == 512))
            self.trans_f32_min_work
        else
            self.trans_f64_min_work;
        return m *| n >= min_work;
    }

    pub fn preferComplexParallel(self: GemvProfile, comptime T: type, m: usize, n: usize) bool {
        if (T == types.ComplexF32 and m == 128 and n == 128) return false;
        return m *| n >= self.complex_parallel_min_work;
    }

    pub fn preferWideF64Parallel(self: GemvProfile, comptime T: type, m: usize, n: usize) bool {
        _ = self;
        return T == f64 and m == 256 and n >= 8192;
    }

    pub fn preferGbmvUnit(
        self: GemvProfile,
        comptime T: type,
        conjugate: bool,
        m: usize,
        n: usize,
        bandwidth: usize,
    ) bool {
        if (m < self.gbmv_min_dimension or n < self.gbmv_min_dimension or
            bandwidth < self.gbmv_min_bandwidth) return false;
        return !(T == types.ComplexF32 and conjugate and
            (m < self.gbmv_c32_conjugate_min_dimension or n < self.gbmv_c32_conjugate_min_dimension));
    }

    pub fn workspaceAllowed(self: GemvProfile, element_count: usize, element_size: usize) bool {
        return element_count <= self.private_workspace_max_bytes / element_size;
    }
};

pub const SymmetricProfile = struct {
    enable_fixed_columns: bool,
    private_workspace_max_bytes: usize,
    real_parallel_min_work: usize,
    hermitian_parallel_min_work: usize,
    band_min_n: usize,
    band_min_k: usize,
    packed_parallel_min_n: usize,

    pub fn preferFourColumnRealBody(self: SymmetricProfile, n: usize) bool {
        _ = self;
        return n >= 256;
    }

    pub fn realMinColumns(self: SymmetricProfile, comptime T: type, n: usize) usize {
        _ = self;
        return if ((T == f32 and n == 512) or n >= 768) 64 else 128;
    }

    pub fn capRealTasks(self: SymmetricProfile, comptime T: type, n: usize, task_count: usize) usize {
        _ = self;
        if (n > 1536) return task_count;
        const cap: usize = if (T == f32 and n == 512) 8 else 6;
        return @min(task_count, cap);
    }

    pub fn useUpperRangedRealWorkspace(self: SymmetricProfile, comptime T: type, upper: bool, n: usize) bool {
        _ = self;
        return T == f32 and upper and n == 512;
    }

    pub fn useRealLowLatency(self: SymmetricProfile, comptime T: type, n: usize) bool {
        _ = self;
        return (T == f32 or T == f64) and n <= 1024;
    }

    pub fn hermitianMinColumns(self: SymmetricProfile) usize {
        _ = self;
        return 48;
    }

    pub fn capHermitianTasks(self: SymmetricProfile, comptime T: type, n: usize, task_count: usize) usize {
        _ = self;
        var result = if (n <= 512) @min(task_count, 10) else task_count;
        if (T == types.ComplexF64 and n == 512) result = @min(result, 8);
        return result;
    }

    pub fn balanceUpperHermitian(self: SymmetricProfile, upper: bool, n: usize) bool {
        _ = self;
        return upper and n >= 256;
    }

    pub fn packedMinColumns(self: SymmetricProfile) usize {
        _ = self;
        return 64;
    }

    pub fn preferRealParallel(self: SymmetricProfile, n: usize) bool {
        return n *| n >= self.real_parallel_min_work;
    }

    pub fn preferHermitianParallel(self: SymmetricProfile, n: usize) bool {
        return n *| n >= self.hermitian_parallel_min_work;
    }

    pub fn preferBandUnit(self: SymmetricProfile, n: usize, k: usize) bool {
        return n >= self.band_min_n and k >= self.band_min_k;
    }

    pub fn preferPackedParallel(self: SymmetricProfile, n: usize) bool {
        return n >= self.packed_parallel_min_n;
    }

    pub fn workspaceAllowed(self: SymmetricProfile, element_count: usize, element_size: usize) bool {
        return element_count <= self.private_workspace_max_bytes / element_size;
    }
};

pub const RankUpdateProfile = struct {
    enable_fixed_complex_ger: bool,
    enable_x86_narrow_width: bool,
    real_parallel_min_work: usize,
    complex_parallel_min_work: usize,
    dense_structured_parallel_min_work: usize,
    packed_task_min_n: usize,
    packed_structured_parallel_min_n: usize,

    pub fn realGerMinColumns(self: RankUpdateProfile, n: usize) usize {
        _ = self;
        return if (n >= 768) 32 else if (n >= 512) 64 else if (n >= 256) 80 else 256;
    }

    pub fn capRealGerTasks(self: RankUpdateProfile, n: usize, task_count: usize) usize {
        _ = self;
        const cap: usize = if (n >= 1536) 10 else if (n >= 512) 4 else if (n >= 384) 10 else 8;
        return @min(task_count, cap);
    }

    pub fn useRealGerLowLatency(self: RankUpdateProfile, comptime T: type, n: usize) bool {
        _ = self;
        return (T == f32 or T == f64) and n < 1536;
    }

    pub fn complexGerMinColumns(self: RankUpdateProfile, comptime T: type, exact_c64_ger128: bool) usize {
        _ = self;
        return if (exact_c64_ger128 or T == types.ComplexF32) 64 else 48;
    }

    pub fn capComplexGerTasks(
        self: RankUpdateProfile,
        comptime T: type,
        exact_c64_ger128: bool,
        n: usize,
        task_count: usize,
    ) usize {
        _ = self;
        const cap: usize = if (exact_c64_ger128)
            2
        else if (T == types.ComplexF64 and n >= 256 and n < 512)
            5
        else if (n < 512)
            4
        else
            8;
        return @min(task_count, cap);
    }

    pub fn structuredMinColumns(self: RankUpdateProfile) usize {
        _ = self;
        return 64;
    }

    pub fn capStructuredTasks(self: RankUpdateProfile, n: usize, task_count: usize) usize {
        _ = self;
        return if (n <= 1536) @min(task_count, 8) else task_count;
    }

    pub fn preferRealGerWideF64(self: RankUpdateProfile, comptime T: type, m: usize) bool {
        _ = self;
        return T == f64 and m >= 512 and m < 1024;
    }

    pub fn preferComplexGerSpecialized(self: RankUpdateProfile, comptime T: type, m: usize) bool {
        _ = self;
        return (T == types.ComplexF32 and m >= 128) or
            (T == types.ComplexF64 and (m == 128 or m == 256));
    }

    pub fn preferRealGerParallel(self: RankUpdateProfile, m: usize, n: usize) bool {
        return m *| n >= self.real_parallel_min_work;
    }

    pub fn preferComplexGerParallel(self: RankUpdateProfile, exact_c64_ger128: bool, m: usize, n: usize) bool {
        return exact_c64_ger128 or m *| n >= self.complex_parallel_min_work;
    }

    pub fn preferDenseStructuredParallel(self: RankUpdateProfile, n: usize) bool {
        return n *| n >= self.dense_structured_parallel_min_work;
    }

    pub fn preferPackedStructuredParallel(self: RankUpdateProfile, n: usize) bool {
        return n >= self.packed_structured_parallel_min_n;
    }

    pub fn preferPackedTaskComposition(self: RankUpdateProfile, n: usize) bool {
        return n >= self.packed_task_min_n;
    }
};

pub const TriangularProfile = struct {
    enable_fixed_bodies: bool,
    dense_vector_min: usize,
    complex_vector_parallel_min: usize,
    band_window_min_n: usize,
    band_window_max_fraction_denominator: usize,
    packed_x86_min_n: usize,

    pub fn preferDenseVector(self: TriangularProfile, n: usize) bool {
        return n >= self.dense_vector_min;
    }

    pub fn keepComplexDependencySerial(self: TriangularProfile, n: usize) bool {
        return n < self.complex_vector_parallel_min;
    }

    pub fn preferBandWindow(self: TriangularProfile, n: usize, k: usize) bool {
        return n >= self.band_window_min_n and
            k <= n / self.band_window_max_fraction_denominator;
    }

    pub fn bandWindowAllowed(
        self: TriangularProfile,
        require_x86: bool,
        n: types.BlasInt,
        k: types.BlasInt,
        incx: types.BlasInt,
    ) bool {
        if (incx != 1 or n <= 0 or k < 0) return false;
        if (require_x86 and builtin.cpu.arch != .x86_64) return false;
        return self.preferBandWindow(@intCast(n), @intCast(k));
    }

    pub fn preferPackedX86(self: TriangularProfile, n: usize) bool {
        return builtin.cpu.arch == .x86_64 and n >= self.packed_x86_min_n;
    }
};

pub const Profile = struct {
    gemv: GemvProfile,
    symmetric: SymmetricProfile,
    rank_update: RankUpdateProfile,
    triangular: TriangularProfile,
};

/// Production choices immediately before the Level 2 registry migration.
pub const production_2026_07_17: Profile = .{
    .gemv = .{
        .enable_x86_narrow_width = false,
        .private_workspace_max_bytes = 64 * 1024 * 1024,
        .no_trans_packed_min_work = 768 * 768,
        .no_trans_parallel_min_work = 512 * 512,
        .trans_f32_min_work = 512 * 512,
        .trans_f64_min_work = 768 * 768,
        .complex_parallel_min_work = 128 * 128,
        .gbmv_min_dimension = 512,
        .gbmv_min_bandwidth = 17,
        .gbmv_c32_conjugate_min_dimension = 1024,
    },
    .symmetric = .{
        .enable_fixed_columns = false,
        .private_workspace_max_bytes = 64 * 1024 * 1024,
        .real_parallel_min_work = 512 * 512,
        .hermitian_parallel_min_work = 256 * 256,
        .band_min_n = 512,
        .band_min_k = 8,
        .packed_parallel_min_n = 512,
    },
    .rank_update = .{
        .enable_fixed_complex_ger = false,
        .enable_x86_narrow_width = false,
        .real_parallel_min_work = 0,
        .complex_parallel_min_work = 256 * 256,
        .dense_structured_parallel_min_work = 512 * 512,
        .packed_task_min_n = 512,
        .packed_structured_parallel_min_n = 2048,
    },
    .triangular = .{
        .enable_fixed_bodies = false,
        .dense_vector_min = 64,
        .complex_vector_parallel_min = 512 * 1024,
        .band_window_min_n = 512,
        .band_window_max_fraction_denominator = 16,
        .packed_x86_min_n = 128,
    },
};

fn enableFixedCandidates(base: Profile) Profile {
    var result = base;
    result.rank_update.enable_fixed_complex_ger = true;
    result.symmetric.enable_fixed_columns = true;
    result.triangular.enable_fixed_bodies = true;
    return result;
}

pub const fixed_candidates_2026_07_17 = enableFixedCandidates(production_2026_07_17);

fn enableWidthCandidates(base: Profile) Profile {
    var result = base;
    result.gemv.enable_x86_narrow_width = true;
    result.rank_update.enable_x86_narrow_width = true;
    return result;
}

pub const width_candidates_2026_07_17 = enableWidthCandidates(production_2026_07_17);

const use_fixed_candidates = if (@hasDecl(root, "zynum_level2_fixed_candidates"))
    root.zynum_level2_fixed_candidates
else
    false;

const use_width_candidates = if (@hasDecl(root, "zynum_level2_width_candidates"))
    root.zynum_level2_width_candidates
else
    false;

pub const active = if (use_fixed_candidates)
    fixed_candidates_2026_07_17
else if (use_width_candidates)
    width_candidates_2026_07_17
else
    production_2026_07_17;

pub fn capTaskCountByWork(task_count: usize, work: usize, min_work_per_task: usize) usize {
    const by_work = @max(@as(usize, 1), (work +| (min_work_per_task - 1)) / min_work_per_task);
    return @min(task_count, by_work);
}

test "experimental Level 2 width profile changes only width switches" {
    const candidate = width_candidates_2026_07_17;
    try std.testing.expect(candidate.gemv.enable_x86_narrow_width);
    try std.testing.expect(candidate.rank_update.enable_x86_narrow_width);

    var restored = candidate;
    restored.gemv.enable_x86_narrow_width = false;
    restored.rank_update.enable_x86_narrow_width = false;
    try std.testing.expectEqualDeep(production_2026_07_17, restored);
}

test "experimental Level 2 fixed profile changes only its candidate switch" {
    const candidate = fixed_candidates_2026_07_17;
    try std.testing.expect(candidate.rank_update.enable_fixed_complex_ger);

    var restored = candidate;
    restored.rank_update.enable_fixed_complex_ger = false;
    restored.symmetric.enable_fixed_columns = false;
    restored.triangular.enable_fixed_bodies = false;
    try std.testing.expectEqualDeep(production_2026_07_17, restored);
}

test "Level 2 hard feasibility is independent from shape preference" {
    const unit = catalog.findImplementation(.gemv, .f64, .core_unit).?;
    const short: Request = .{ .m = 1, .n = 1, .incx = 1, .incy = 1 };
    try std.testing.expect(hardFeasibility(unit, short).isFeasible());
    try std.testing.expect(!active.gemv.preferNoTransParallel(short.m, short.n));

    const strided: Request = .{ .m = 512, .n = 512, .incx = -2, .incy = 3 };
    try std.testing.expectEqual(FeasibilityReason.invalid_x_stride, hardFeasibility(unit, strided).reason);
    try std.testing.expectEqual(catalog.Implementation.portable_scalar, selectDefault(f64, .gemv, strided).kernel.implementation);
}

test "Level 2 selector preserves unit and signed-stride default boundaries" {
    const corpus = [_]Request{
        .{ .m = 1, .n = 1, .incx = 1, .incy = 1 },
        .{ .m = 512, .n = 512, .incx = 1, .incy = 1 },
        .{ .m = 512, .n = 512, .incx = 2, .incy = 1 },
        .{ .m = 4096, .n = 4096, .incx = -1, .incy = -3 },
    };
    for (corpus, 0..) |request, index| {
        const expected: catalog.Implementation = if (index < 2) .core_unit else .portable_scalar;
        try std.testing.expectEqual(expected, selectDefault(types.ComplexF64, .gemv, request).kernel.implementation);
    }
}

test "named Level 2 production profile preserves retained thresholds" {
    const profile = production_2026_07_17;
    try std.testing.expect(!profile.gemv.preferNoTransParallel(511, 512));
    try std.testing.expect(profile.gemv.preferNoTransParallel(512, 512));
    try std.testing.expect(!profile.gemv.preferGbmvUnit(f32, false, 512, 512, 16));
    try std.testing.expect(profile.gemv.preferGbmvUnit(f32, false, 512, 512, 17));
    try std.testing.expect(!profile.symmetric.preferBandUnit(512, 7));
    try std.testing.expect(profile.symmetric.preferBandUnit(512, 8));
    try std.testing.expect(!profile.triangular.preferBandWindow(512, 33));
    try std.testing.expect(profile.triangular.preferBandWindow(512, 32));
    try std.testing.expect(!profile.rank_update.preferPackedStructuredParallel(2047));
    try std.testing.expect(profile.rank_update.preferPackedStructuredParallel(2048));
}
