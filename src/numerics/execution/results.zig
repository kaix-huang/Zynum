// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
pub const Algorithm = enum(u32) { portable_generic = 1 };
pub const ContextDesc = struct { algorithm: Algorithm = .portable_generic, max_parallel_tasks: u32 = 0 };
pub const ResourceStage = enum(u32) { none = 0, descriptor = 1, retain = 2, query = 3, scratch_allocate = 4, submit = 5, scratch_release = 6, provider_release = 7, rollback_release = 8 };
pub const SubmissionState = enum(u32) { not_submitted = 0, rejected = 1, quiescent = 2 };
pub const OutputState = enum(u32) { unchanged = 0, complete = 1, partial_or_invalid = 2 };
pub const ProviderDiagnostic = struct {
    stage: ResourceStage = .none,
    returned_status: i32 = 0,
    execution_status: i32 = 0,
    callback_status: i32 = 0,
    cleanup_status: i32 = 0,
    state: SubmissionState = .not_submitted,
    callbacks_started: u64 = 0,
    callbacks_completed: u64 = 0,
};
pub const CallReport = struct { provider: ProviderDiagnostic = .{}, output: OutputState = .unchanged };
pub const ExecutionError = error{ InvalidProvider, Unsupported, ContextClosing, CounterOverflow, ResourceBudgetExceeded, OutOfMemory, ProviderBusy, ProviderClosed, WouldDeadlock, ProviderRejected, ProviderExecutionFailed, ProviderCallbackFailed, ExtentOverflow };
pub const InitError = ExecutionError;
