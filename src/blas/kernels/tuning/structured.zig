// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Measured shape policy for isolated structured Level 3 candidates.
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
    const profile = x86_64_object_profile;
    try std.testing.expect(profile.rightTriangularCandidate(512, 128));
    try std.testing.expect(profile.rightTriangularCandidate(128, 512));
    try std.testing.expect(!profile.rightTriangularCandidate(128, 128));
    try std.testing.expect(!profile.rightTriangularCandidate(256, 256));
    try std.testing.expect(!profile.rightTriangularCandidate(1024, 64));
}
