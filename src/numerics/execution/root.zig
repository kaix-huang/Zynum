// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Unique execution identities. This root never imports the mathematical Core.
const options = @import("zynum-build-options");
pub const contracts = @import("contracts.zig");
pub const results = @import("results.zig");
pub const context = @import("context.zig");
pub const Context = context.Context;
pub const Operation = context.Operation;
pub const ResourcePin = context.ResourcePin;
pub const OperationScratch = context.OperationScratch;
pub const acquireOperationScratch = context.acquireOperationScratch;
pub const ContextDesc = results.ContextDesc;
pub const Algorithm = results.Algorithm;
pub const CallReport = results.CallReport;
pub const ProviderDiagnostic = results.ProviderDiagnostic;
pub const ResourceStage = results.ResourceStage;
pub const SubmissionState = results.SubmissionState;
pub const OutputState = results.OutputState;
pub const ExecutionError = results.ExecutionError;
pub const InitError = results.InitError;
pub const AdmissionError = context.AdmissionError;
pub const CloseResult = context.CloseResult;
pub const RequestResult = context.RequestResult;
pub const TaskFn = contracts.TaskFn;
pub const Batch = contracts.Batch;
pub const Policy = contracts.Policy;
pub const WorkerCacheDomain = contracts.WorkerCacheDomain;
pub const max_tasks = contracts.max_tasks;
pub const RunMode = contracts.RunMode;
pub const AdmittedCallResources = @import("call_resources.zig").AdmittedCallResources;
pub const admittedResources = @import("call_resources.zig").admittedResources;
pub const reference = if (options.reference_compat) @import("reference_root.zig") else struct {};
pub const CallResources = if (options.reference_compat) reference.ReferenceCallResources else AdmittedCallResources;
// Lazy internal reference aliases: actually using one in the integrated graph
// fails compilation because the false branch has no matching declaration.
pub const owner = reference.owner;
pub const policy = reference.policy;
pub const get = reference.get;
pub const getContext = reference.getContext;
pub const beginDefault = reference.beginDefault;
pub const test_support = reference.test_support;
// Address identity only: private storage is never mutated and creates no owner.
// A var backing prevents identical-constant folding across module instances.
var identity_anchor: u8 = 0xe7;
pub fn executionIdentity() *const anyopaque {
    return &identity_anchor;
}
