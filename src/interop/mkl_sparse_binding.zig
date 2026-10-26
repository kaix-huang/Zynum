// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Single typed interface binding, with no ABI exports or profile readiness.
//! Core owns common storage/numerical/resource/Context types and state. The shell
//! and its semantic tags belong here, never in Core or an Intel layout mirror.
pub const implementation = @import("zynum-core");
const n = implementation.internal.numerics;
pub const Provider = struct {
    pub const OwnerIdentity = implementation;
    pub const Pattern = n.SparsePattern;
    pub const Scalar = n.SparseScalar;
    pub const Product = n.SparseProduct;
    pub const Dense = n.Dense;
    pub const Object = n.SparseObject;
    pub const Storage = n.Storage;
    pub const Execution = n.Execution;
    pub const Resources = n.Resources;
};
pub const Handles = @import("sparse_handle.zig").Family(Provider);
pub const Adapter = @import("mkl_sparse.zig").TypedAdapter(Provider, Handles);
pub const Raw = @import("mkl_sparse_raw.zig").RawAdapter(Provider, Handles, Adapter);

pub const Solve = @import("mkl_sparse_solve.zig").Adapter(Provider, Handles, Adapter);

pub const RawSolve = @import("mkl_sparse_solve_raw.zig").Adapter(Provider, Handles, Adapter, Solve, Raw);
