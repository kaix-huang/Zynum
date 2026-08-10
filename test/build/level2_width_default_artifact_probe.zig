// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Non-test artifact probe for the default x86 Level 2 width production graph.

const std = @import("std");
const builtin = @import("builtin");
const contract = @import("level2_width_artifact_probe_contract.zig");

const ProbeError = error{
    BuiltAsTest,
    UnexpectedTarget,
    CandidateEnabled,
    StubHandledRequest,
    StubChangedOutput,
    GuardChanged,
};

fn rawF32(value: f32) u64 {
    return @as(u32, @bitCast(value));
}

fn sameStorage(actual: []const f32, expected: []const f32) bool {
    return std.mem.eql(u8, std.mem.sliceAsBytes(actual), std.mem.sliceAsBytes(expected));
}

fn verifyStubGemvWitness() ProbeError!void {
    const matrix = [_]f32{ 1, -2, 3, -4, 5, -6, 7, -8 };
    const x = [_]f32{2};
    var output_storage = [_]f32{ 1001, 10, 20, 30, 40, 50, 60, 70, 80, -1003 };
    const before = output_storage;
    var request: contract.Request = .{
        .operation = @intFromEnum(contract.Operation.gemv_no_trans_unit),
        .scalar = @intFromEnum(contract.Scalar.f32),
        .flags = 0,
        .lda = 8,
        .m = 8,
        .n = 1,
        .alpha_re = rawF32(0.5),
        .alpha_im = 0,
        .beta_re = rawF32(1),
        .beta_im = 0,
        .input0 = @ptrCast(&matrix),
        .input1 = @ptrCast(&x),
        .output = @ptrCast(&output_storage[1]),
    };
    if (contract.zynum_internal_x86_64_level2_width_execute(&request) != 0) return error.StubHandledRequest;
    if (output_storage[0] != before[0] or output_storage[output_storage.len - 1] != before[before.len - 1]) {
        return error.GuardChanged;
    }
    if (!sameStorage(&output_storage, &before)) return error.StubChangedOutput;
}

fn verifyStubGerWitness() ProbeError!void {
    const x = [_]f32{ 1, -2, 3, -4, 5, -6, 7, -8 };
    const y = [_]f32{2};
    var output_storage = [_]f32{ 2001, 11, 12, 13, 14, 15, 16, 17, 18, -2003 };
    const before = output_storage;
    var request: contract.Request = .{
        .operation = @intFromEnum(contract.Operation.ger_real),
        .scalar = @intFromEnum(contract.Scalar.f32),
        .flags = 0,
        .lda = 8,
        .m = 8,
        .n = 1,
        .alpha_re = rawF32(0.5),
        .alpha_im = 0,
        .beta_re = 0,
        .beta_im = 0,
        .input0 = @ptrCast(&x),
        .input1 = @ptrCast(&y),
        .output = @ptrCast(&output_storage[1]),
    };
    if (contract.zynum_internal_x86_64_level2_width_execute(&request) != 0) return error.StubHandledRequest;
    if (output_storage[0] != before[0] or output_storage[output_storage.len - 1] != before[before.len - 1]) {
        return error.GuardChanged;
    }
    if (!sameStorage(&output_storage, &before)) return error.StubChangedOutput;
}

pub fn main() ProbeError!void {
    if (builtin.is_test) return error.BuiltAsTest;
    if (builtin.cpu.arch != .x86_64) return error.UnexpectedTarget;
    if (contract.zynum_internal_x86_64_level2_width_enabled != 0) return error.CandidateEnabled;

    try verifyStubGemvWitness();
    try verifyStubGerWitness();
}
