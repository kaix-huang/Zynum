// SPDX-License-Identifier: LGPL-3.0-or-later
// This adapter is reachable only from unit tests, including with standalone OFF.
const core = @import("root.zig").testing;
const standalone = @import("standalone.zig");
pub const abi = core.abi;
pub const gemm = core.gemm;
pub const gemm_complex = core.gemm_complex;
pub const axpy = core.axpy;
pub const dot = core.dot;
pub const laplacian = core.laplacian;
pub const create = standalone.zynum_resources_create_naive;
pub const destroy = standalone.zynum_resources_destroy_naive;
