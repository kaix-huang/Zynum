// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! One separately compiled ISA object, with no public ABI or task runtime.
const object = @import("kernels/multiversion/object.zig");
pub const zynum_level1_sve_candidates = object.zynum_level1_sve_candidates;
pub const zynum_level1_fixed_candidates = object.zynum_level1_fixed_candidates;
pub const zynum_level2_fixed_candidates = object.zynum_level2_fixed_candidates;
pub const zynum_level2_width_candidates = object.zynum_level2_width_candidates;
comptime {
    _ = object;
}
