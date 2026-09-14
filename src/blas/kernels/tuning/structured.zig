// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shape policy for structured Level 3 candidates.
//!
//! Intrinsic ABI, storage, and workspace constraints stay in the object and
//! registry contracts. These predicates are outer dispatch preferences so a
//! rejected shape does not pay the hidden bridge call.

const std = @import("std");
const contract = @import("../contract.zig");

pub const scalarKind = contract.scalarKind;

pub const Side = enum {
    left,
    right,
};

/// Bounded-panel preference, separate from packing and capability contracts.
pub const ActivePanelProfile = struct {
    min_output_dimension: usize,
    block_size: usize,

    pub fn candidate(self: ActivePanelProfile, m: usize, n: usize) bool {
        return m >= self.min_output_dimension and n >= self.min_output_dimension;
    }
};

/// Unmeasured candidate: retain only after same-source native boundary sweeps.
pub const aarch64_macos_active_panel_candidate: ActivePanelProfile = .{
    .min_output_dimension = 128,
    .block_size = 64,
};

/// Private triangular output tiles; preference only, not an ISA contract.
pub const RankTileProfile = struct {
    min_order: usize,
    min_reduction: usize,
    block_size: usize,
    direct_panels: bool,
    packed_panels: bool = false,
    parallel_packed_panels: bool = false,

    threaded_min_order: usize = 96,
    threaded_min_work: usize = 512 * 1024,

    pub fn candidate(self: RankTileProfile, n: usize, k: usize, threaded: bool) bool {
        if (n < self.min_order or k < self.min_reduction) return false;
        return !threaded or (n >= self.threaded_min_order and n *| n *| k >= self.threaded_min_work);
    }
};

/// Native macOS preference for measured rank-update shapes; other shapes retain fallbacks.
pub const aarch64_macos_rank_tile_candidate: RankTileProfile = .{
    .min_order = 64,
    .min_reduction = 32,
    .block_size = 64,
    .direct_panels = false,
    .packed_panels = true,
    .parallel_packed_panels = true,
};

/// Shared with the retained left-column parallel executor.
pub const triangular_parallel_left_min_work: usize = 8 * 1024 * 1024;

pub const TriangularProfile = struct {
    min_output_dimension: usize = 128,
    right_min_output_dimension: usize = 128,
    block_size: usize = 64,
    solve: bool,

    pub fn candidate(self: TriangularProfile, kind: contract.ScalarKind, side: Side, threaded: bool, m: usize, n: usize) bool {
        if (side == .right) return m >= self.right_min_output_dimension and n >= self.right_min_output_dimension;
        if (m < self.min_output_dimension or n < self.min_output_dimension) return false;
        // The retained executor parallelizes independent left RHS columns.
        if (threaded and m *| m *| n >= triangular_parallel_left_min_work) return false;
        // Both-cap sweeps retain clear left solve wins only for complex f64.
        return !self.solve or kind == .complex_f64;
    }
};

pub const aarch64_macos_trmm_candidate: TriangularProfile = .{ .solve = false, .right_min_output_dimension = 64 };
pub const aarch64_macos_trsm_candidate: TriangularProfile = .{ .solve = true };

pub const Profile = struct {
    dense_min_dimension: usize,
    dense_c64_left_symm_gap_start: usize,
    dense_c64_left_symm_gap_end: usize,
    right_min_dimension: usize,
    right_min_aspect_ratio: usize,
    right_min_work: usize,

    pub fn denseCandidate(self: Profile, scalar: contract.ScalarKind, side: Side, hermitian: bool, m: usize, n: usize) bool {
        if (m != n or m < self.dense_min_dimension) return false;
        if (scalar == .complex_f64 and side == .left and !hermitian and
            m > self.dense_c64_left_symm_gap_start and
            m < self.dense_c64_left_symm_gap_end)
        {
            return false;
        }
        return true;
    }

    pub fn rightTriangularCandidate(self: Profile, m: usize, n: usize) bool {
        const short = @min(m, n);
        const long = @max(m, n);
        if (short < self.right_min_dimension) return false;
        if (long < short *| self.right_min_aspect_ratio) return false;
        return m *| n *| n >= self.right_min_work;
    }
};

/// Fresh-process isolated-object/control measurements on AVX-512F/FMA targets:
/// - dense square 128 and 512 passed every same-ABI cell;
/// - c64 left SYMM at 256 lost while right SYMM and HEMM retained wins;
/// - rectangular dense classes regressed;
/// - right TRMM/TRSM passed all 80 cells at 512x128 and 128x512;
/// - square128 rejected before the bridge after short-case overhead was found.
pub const x86_64_object_profile: Profile = .{
    .dense_min_dimension = 128,
    .dense_c64_left_symm_gap_start = 128,
    .dense_c64_left_symm_gap_end = 512,
    .right_min_dimension = 128,
    .right_min_aspect_ratio = 4,
    .right_min_work = 8 * 1024 * 1024,
};

test "structured object profile keeps only measured dense shape classes" {
    const profile = x86_64_object_profile;
    try std.testing.expect(profile.denseCandidate(.f32, .left, false, 128, 128));
    try std.testing.expect(profile.denseCandidate(.complex_f64, .left, false, 128, 128));
    try std.testing.expect(!profile.denseCandidate(.complex_f64, .left, false, 256, 256));
    try std.testing.expect(profile.denseCandidate(.complex_f64, .right, false, 256, 256));
    try std.testing.expect(profile.denseCandidate(.complex_f64, .left, true, 256, 256));
    try std.testing.expect(profile.denseCandidate(.complex_f64, .left, false, 512, 512));
    try std.testing.expect(!profile.denseCandidate(.f64, .left, false, 128, 512));
    try std.testing.expect(!profile.denseCandidate(.f64, .left, false, 64, 64));
}

test "structured object profile rejects short and near-square triangular work" {
    const multiply = aarch64_macos_trmm_candidate;
    const solve = aarch64_macos_trsm_candidate;
    try std.testing.expect(!multiply.candidate(.f32, .left, false, 127, 128));
    try std.testing.expect(multiply.candidate(.f32, .left, true, 128, 511));
    try std.testing.expect(!multiply.candidate(.f32, .left, true, 128, 512));
    try std.testing.expect(multiply.candidate(.f32, .left, false, 128, 512));
    try std.testing.expect(multiply.candidate(.f32, .right, true, 128, 512));
    try std.testing.expect(!solve.candidate(.f64, .left, false, 128, 128));
    try std.testing.expect(!solve.candidate(.complex_f32, .left, true, 128, 128));
    try std.testing.expect(solve.candidate(.complex_f64, .left, true, 129, 131));
    try std.testing.expect(!solve.candidate(.complex_f64, .left, true, 128, 512));
    try std.testing.expect(solve.candidate(.complex_f64, .left, false, 128, 512));
    try std.testing.expect(solve.candidate(.f32, .right, true, 128, 512));

    const profile = x86_64_object_profile;
    try std.testing.expect(profile.rightTriangularCandidate(512, 128));
    try std.testing.expect(profile.rightTriangularCandidate(128, 512));
    try std.testing.expect(!profile.rightTriangularCandidate(128, 128));
    try std.testing.expect(!profile.rightTriangularCandidate(256, 256));
    try std.testing.expect(!profile.rightTriangularCandidate(1024, 64));
}
