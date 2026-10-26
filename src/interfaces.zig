// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Optional combined export root; no additional API, numerical state or algorithm.
const blas_exports = @import("zynum-blas-exports");
const checked_exports = @import("zynum-checked-exports");

// Preserve the selected legacy root's compile-time experimental policy markers.
pub const zynum_structured_object_candidates = if (@hasDecl(blas_exports, "zynum_structured_object_candidates")) blas_exports.zynum_structured_object_candidates else false;
pub const zynum_level1_sve_candidates = if (@hasDecl(blas_exports, "zynum_level1_sve_candidates")) blas_exports.zynum_level1_sve_candidates else false;
pub const zynum_level1_fixed_candidates = if (@hasDecl(blas_exports, "zynum_level1_fixed_candidates")) blas_exports.zynum_level1_fixed_candidates else false;
pub const zynum_level2_fixed_candidates = if (@hasDecl(blas_exports, "zynum_level2_fixed_candidates")) blas_exports.zynum_level2_fixed_candidates else false;
pub const zynum_level2_width_candidates = if (@hasDecl(blas_exports, "zynum_level2_width_candidates")) blas_exports.zynum_level2_width_candidates else false;

comptime {
    _ = blas_exports;
    _ = checked_exports;
}
