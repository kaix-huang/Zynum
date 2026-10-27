// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! The sole reference image cache bridge. Execution calls its hidden C symbol;
//! no execution module imports mathematical code.
const std = @import("std");
const builtin = @import("builtin");
const options = @import("zynum-build-options");
fn releaseWorkerCaches() callconv(.c) void {
    // Preserve the original freeWorkerCaches target/build conditions exactly.
    if (comptime options.dynamic_dispatch or
        (builtin.target.cpu.arch == .aarch64 and builtin.target.cpu.features.isEnabled(@backingInt(std.Target.aarch64.Feature.sme))))
        @import("kernels/dispatch/matrix_matrix.zig").freeCurrentThreadCaches();
}
comptime {
    if (!options.reference_compat) @compileError("reference cache bridge is forbidden in integrated_native");
    @export(&releaseWorkerCaches, .{ .name = "zynum_internal_reference_release_worker_caches", .visibility = .hidden });
}
