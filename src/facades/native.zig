// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Explicit native facade. No default owner or compatibility API is forwarded.
const core = @import("zynum-core");
const execution = @import("zynum-execution");
pub const Context = execution.Context;
pub const Operation = execution.Operation;
pub const ContextDesc = execution.ContextDesc;
pub const Algorithm = execution.Algorithm;
pub const CallReport = execution.CallReport;
pub const ExecutionError = execution.ExecutionError;
pub const MatrixProductF64 = core.native.MatrixProductF64;
pub const NativeError = core.native.NativeError;
pub const matrixMultiplyF64NN = core.native.matrixMultiplyF64NN;
pub const coreIdentity = core.internal.nativeCoreIdentity;
