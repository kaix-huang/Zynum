// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const aarch64_features = @import("kernels/arch/aarch64/features.zig");
const apple_amx = @import("kernels/arch/aarch64/matrix_matrix/amx.zig");
const apple_amx_ops = @import("kernels/arch/aarch64/matrix_matrix/amx_ops.zig");
const aarch64_matrix_vector = @import("kernels/arch/aarch64/matrix_vector.zig");
const complex_gemm = @import("core/matrix_matrix/gemm.zig");
const scalar = @import("core/shared/scalar.zig");
const catalog = @import("kernels/shared/matrix_matrix/catalog.zig");
const coverage = @import("kernels/shared/matrix_matrix/coverage.zig");
const executor = @import("kernels/shared/matrix_matrix/executor.zig");
const gemm_task = @import("kernels/shared/matrix_matrix/task.zig");
const tuning = @import("kernels/shared/matrix_matrix/tuning.zig");

const Layout = enum {
    no_trans,
    transposed_b,
};

const ComplexLayout = struct {
    transa: scalar.Order,
    transb: scalar.Order,
};

fn streamingVectorBytes() usize {
    return aarch64_features.streamingVectorBytes();
}

fn sample(comptime T: type, index: usize, salt: usize) T {
    const centered: i32 = @as(i32, @intCast((index * 17 + salt * 11) % 29)) - 14;
    return @as(T, @floatFromInt(centered)) / 16;
}

fn complexValue(comptime T: type, re: scalar.Real(T), im: scalar.Real(T)) T {
    return .{ .re = re, .im = im };
}

fn complexSample(comptime T: type, index: usize, salt: usize) T {
    const Real = scalar.Real(T);
    return complexValue(T, sample(Real, index, salt), sample(Real, index * 7 + 3, salt + 5));
}

fn complexOperand(comptime T: type, trans: scalar.Order, matrix: []const T, ld: gemm_task.BlasInt, row: usize, col: usize) T {
    const value = switch (trans) {
        .no_trans => matrix[gemm_task.matIndex(ld, row, col)],
        .trans, .conj_trans => matrix[gemm_task.matIndex(ld, col, row)],
    };
    return if (trans == .conj_trans) scalar.conj(T, value) else value;
}

fn complexReference(
    comptime T: type,
    layout: ComplexLayout,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: []const T,
    lda: gemm_task.BlasInt,
    b: []const T,
    ldb: gemm_task.BlasInt,
    beta: T,
    c_initial: []const T,
    expected: []T,
    ldc: gemm_task.BlasInt,
) void {
    for (0..n) |j| {
        for (0..m) |i| {
            var acc = scalar.zero(T);
            for (0..k) |p| {
                acc = scalar.add(T, acc, scalar.mul(T, complexOperand(T, layout.transa, a, lda, i, p), complexOperand(T, layout.transb, b, ldb, p, j)));
            }
            const c_index = gemm_task.matIndex(ldc, i, j);
            expected[c_index] = scalar.add(T, scalar.mul(T, alpha, acc), scalar.mul(T, beta, c_initial[c_index]));
        }
    }
}

fn expectComplexClose(comptime T: type, expected: T, actual: T) !void {
    const Real = scalar.Real(T);
    const absolute_tolerance: Real = if (T == complex_gemm.ComplexF32) 3e-3 else 2e-11;
    const relative_tolerance: Real = if (T == complex_gemm.ComplexF32) 3e-3 else 2e-11;
    const scale = @max(@abs(expected.re), @abs(expected.im));
    const tolerance = absolute_tolerance + relative_tolerance * scale;
    try std.testing.expect(@abs(expected.re - actual.re) <= tolerance);
    try std.testing.expect(@abs(expected.im - actual.im) <= tolerance);
}

fn runForcedComplexCase(
    comptime T: type,
    kernel: catalog.ComplexKernelId,
    options: complex_gemm.ComplexExecutionOptions,
    layout: ComplexLayout,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    beta: T,
) !complex_gemm.ComplexExecutionResult {
    const a_rows = if (layout.transa == .no_trans) m else k;
    const a_cols = if (layout.transa == .no_trans) k else m;
    const b_rows = if (layout.transb == .no_trans) k else n;
    const b_cols = if (layout.transb == .no_trans) n else k;
    const lda: gemm_task.BlasInt = @intCast(a_rows + 2);
    const ldb: gemm_task.BlasInt = @intCast(b_rows + 3);
    const ldc: gemm_task.BlasInt = @intCast(m + 4);
    const a_len = @as(usize, @intCast(lda)) * a_cols;
    const b_len = @as(usize, @intCast(ldb)) * b_cols;
    const c_len = @as(usize, @intCast(ldc)) * n;

    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, a_len);
    defer allocator.free(a);
    const b = try allocator.alloc(T, b_len);
    defer allocator.free(b);
    const c = try allocator.alloc(T, c_len);
    defer allocator.free(c);
    const c_initial = try allocator.alloc(T, c_len);
    defer allocator.free(c_initial);
    const expected = try allocator.alloc(T, c_len);
    defer allocator.free(expected);

    for (a, 0..) |*value, index| value.* = complexSample(T, index, 1);
    for (b, 0..) |*value, index| value.* = complexSample(T, index, 2);
    for (c, 0..) |*value, index| value.* = complexSample(T, index, 3);
    @memcpy(c_initial, c);
    @memcpy(expected, c);
    complexReference(T, layout, m, n, k, alpha, a, lda, b, ldb, beta, c_initial, expected, ldc);

    const result = complex_gemm.executeForcedComplexKernel(
        T,
        kernel,
        options,
        layout.transa,
        layout.transb,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        alpha,
        a.ptr,
        lda,
        b.ptr,
        ldb,
        beta,
        c.ptr,
        ldc,
    );
    try std.testing.expect(result.executed != null);
    for (0..n) |j| {
        for (0..m) |i| {
            const index = gemm_task.matIndex(ldc, i, j);
            try expectComplexClose(T, expected[index], c[index]);
        }
        for (m..@as(usize, @intCast(ldc))) |i| {
            const index = gemm_task.matIndex(ldc, i, j);
            try std.testing.expectEqual(c_initial[index], c[index]);
        }
    }
    return result;
}

fn reference(
    comptime T: type,
    layout: Layout,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: []const T,
    lda: gemm_task.BlasInt,
    b: []const T,
    ldb: gemm_task.BlasInt,
    beta: T,
    c_initial: []const T,
    expected: []T,
    ldc: gemm_task.BlasInt,
) void {
    for (0..n) |j| {
        for (0..m) |i| {
            var acc: T = 0;
            for (0..k) |p| {
                const b_index = switch (layout) {
                    .no_trans => gemm_task.matIndex(ldb, p, j),
                    .transposed_b => gemm_task.matIndex(ldb, j, p),
                };
                acc = @mulAdd(T, a[gemm_task.matIndex(lda, i, p)], b[b_index], acc);
            }
            const c_index = gemm_task.matIndex(ldc, i, j);
            expected[c_index] = alpha * acc + beta * c_initial[c_index];
        }
    }
}

fn expectClose(comptime T: type, expected: T, actual: T) !void {
    const absolute_tolerance: T = if (T == f32) 4e-4 else 2e-12;
    const relative_tolerance: T = if (T == f32) 4e-4 else 2e-12;
    const tolerance = absolute_tolerance + relative_tolerance * @abs(expected);
    try std.testing.expect(@abs(expected - actual) <= tolerance);
}

fn runCase(comptime T: type, desc: catalog.Descriptor, layout: Layout, alpha: T, beta: T) !void {
    const m = @max(desc.bounds.min_m_block, desc.tile.register_m) + 3;
    const n = @max(desc.bounds.min_n_block, desc.tile.n_panel) + 3;
    const k = @max(desc.bounds.min_k_block, desc.tile.k_unroll * 2) + 1;
    try runCaseShape(T, desc, layout, alpha, beta, m, n, k, 0);
}

fn runCaseShape(comptime T: type, desc: catalog.Descriptor, layout: Layout, alpha: T, beta: T, m: usize, n: usize, k: usize, n0: usize) !void {
    const lda: gemm_task.BlasInt = @intCast(m + 3);
    const ldb: gemm_task.BlasInt = @intCast(switch (layout) {
        .no_trans => k + 2,
        .transposed_b => n + 2,
    });
    const ldc: gemm_task.BlasInt = @intCast(m + 5);
    const a_len = @as(usize, @intCast(lda)) * k;
    const b_len = @as(usize, @intCast(ldb)) * switch (layout) {
        .no_trans => n,
        .transposed_b => k,
    };
    const c_len = @as(usize, @intCast(ldc)) * n;

    const allocator = std.testing.allocator;
    const a = try allocator.alloc(T, a_len);
    defer allocator.free(a);
    const b = try allocator.alloc(T, b_len);
    defer allocator.free(b);
    const c = try allocator.alloc(T, c_len);
    defer allocator.free(c);
    const c_initial = try allocator.alloc(T, c_len);
    defer allocator.free(c_initial);
    const expected = try allocator.alloc(T, c_len);
    defer allocator.free(expected);

    for (a, 0..) |*value, index| value.* = sample(T, index, 1);
    for (b, 0..) |*value, index| value.* = sample(T, index, 2);
    for (c, 0..) |*value, index| value.* = sample(T, index, 3);
    @memcpy(c_initial, c);
    @memcpy(expected, c);

    reference(T, layout, m, n, k, alpha, a, lda, b, ldb, beta, c_initial, expected, ldc);
    const prefix_len = @as(usize, @intCast(ldc)) * n0;
    @memcpy(expected[0..prefix_len], c_initial[0..prefix_len]);

    const task: gemm_task.Task(T) = .{
        .m = m,
        .n0 = n0,
        .n1 = n,
        .k = k,
        .alpha = alpha,
        .a = a.ptr,
        .lda = lda,
        .b = b.ptr,
        .ldb = ldb,
        .b_layout = if (layout == .transposed_b) .trans else .no_trans,
        .beta = beta,
        .c = c.ptr,
        .ldc = ldc,
        .allow_sme = desc.family == .streaming_matrix,
        .kernel = desc.kernel,
        .execution = .{
            .selected_kernel = desc.kernel,
            .fallback_kernel = desc.fallback,
            .pack = .{
                .stack_bytes = desc.pack.stack_bytes,
                .cache_bytes = desc.pack.cache_bytes,
            },
        },
    };
    executor.run(T, task);
    try std.testing.expectEqualSlices(T, c_initial[0..prefix_len], c[0..prefix_len]);
    try std.testing.expectEqual(aarch64_features.TestStreamingDepths{ .sm = 0, .za = 0 }, aarch64_features.testStreamingDepths());
    try std.testing.expectEqual(@as(usize, 0), apple_amx_ops.testStateDepth());
    if (comptime aarch64_features.has_sme) try std.testing.expectEqual(@as(u2, 0), aarch64_features.streamingModeBits());

    for (0..n) |j| {
        for (0..m) |i| {
            const index = gemm_task.matIndex(ldc, i, j);
            try expectClose(T, expected[index], c[index]);
        }
        for (m..@as(usize, @intCast(ldc))) |i| {
            const index = gemm_task.matIndex(ldc, i, j);
            try std.testing.expectEqual(c_initial[index], c[index]);
        }
    }
}

fn testForcedPaths(comptime T: type) !void {
    const descriptors = catalog.registeredDescriptors(streamingVectorBytes());
    for (descriptors) |desc| {
        if (desc.scalar != catalog.contractScalarKind(T)) continue;
        if (!executor.availableFor(T, desc.kernel)) continue;

        try runCase(T, desc, .no_trans, 1, 0);
        if (desc.epilogue.arbitrary_alpha and desc.epilogue.arbitrary_beta) {
            try runCase(T, desc, .no_trans, @as(T, 0.75), @as(T, -0.25));
        }
        if (desc.epilogue.alpha_zero) {
            try runCase(T, desc, .no_trans, 0, @as(T, 0.5));
        }
        if (desc.layouts.transposed_b) {
            const trans_alpha: T = if (desc.epilogue.arbitrary_alpha) 0.75 else 1;
            const trans_beta: T = if (desc.epilogue.arbitrary_beta) -0.25 else 0;
            try runCase(T, desc, .transposed_b, trans_alpha, trans_beta);
        }
    }
}

test "forced f32 real GEMM paths match a scalar reference on tails and epilogues" {
    try testForcedPaths(f32);
}

test "forced f64 real GEMM paths match a scalar reference on tails and epilogues" {
    try testForcedPaths(f64);
    const desc = catalog.aarch64AsimdDescriptor(f64);
    if (executor.availableFor(f64, desc.kernel)) {
        // Cover every K remainder in the low-K path with row/column tails,
        // padded leading dimensions, and an untouched column prefix.
        for ([_]usize{ 28, 29, 30, 31, 32, 33 }) |k| {
            for ([_]Layout{ .no_trans, .transposed_b }) |layout| {
                try runCaseShape(f64, desc, layout, 1, 0, 49, 55, k, 3);
                try runCaseShape(f64, desc, layout, 0.75, -0.25, 49, 55, k, 3);
            }
        }
    }
}

test "descriptor packing and workspace contracts flow unchanged into execution plans" {
    const descriptors = catalog.registeredDescriptors(64);
    for (descriptors) |desc| {
        const plan = switch (desc.scalar) {
            .f32 => tuning.executionPlan(f32, desc, .{ .m = 257, .n = 259, .k = 263 }, 4, 4 * 1024 * 1024),
            .f64 => tuning.executionPlan(f64, desc, .{ .m = 257, .n = 259, .k = 263 }, 4, 4 * 1024 * 1024),
            else => unreachable,
        };
        try std.testing.expectEqual(desc.kernel, plan.selected_kernel);
        try std.testing.expectEqual(desc.fallback, plan.fallback_kernel);
        try std.testing.expectEqual(desc.pack.stack_bytes, plan.pack.stack_bytes);
        try std.testing.expectEqual(desc.pack.cache_bytes, plan.pack.cache_bytes);
        if (desc.pack.kind == .none) {
            try std.testing.expectEqual(@as(usize, 0), plan.pack.stack_bytes);
            try std.testing.expectEqual(@as(usize, 0), plan.pack.cache_bytes);
        } else {
            try std.testing.expect(plan.pack.stack_bytes != 0);
        }
    }
}

test "Apple AMX benchmark shapes have explicit selected subkernel ids" {
    const shape: tuning.Shape = .{ .m = 256, .n = 256, .k = 256 };
    const requested_threads: usize = 10;
    const performance_l2_bytes: usize = 16 * 1024 * 1024;

    const f32_candidates = catalog.candidateList(.{
        catalog.aarch64SmeDescriptor(f32, 64),
        catalog.aarch64AsimdDescriptor(f32),
        catalog.genericDescriptor(f32),
    });
    const f32_desc = tuning.select(f32, f32_candidates, shape, 1, 0, requested_threads);
    try std.testing.expectEqual(catalog.KernelId.aarch64_sme_f32_2mx2n, f32_desc.kernel);
    const f32_plan = tuning.executionPlan(f32, f32_desc, shape, requested_threads, performance_l2_bytes);
    try std.testing.expectEqual(gemm_task.AppleAmxKernelId.apple_amx_f32_n32, f32_plan.amx);

    // Preserve the K cap, panel exclusions, and fringe boundaries when
    // simplifying preference rules. These inspect policy without executing AMX.
    const f32_cases = [_]struct {
        shape: tuning.Shape,
        amx: gemm_task.AppleAmxKernelId,
        partial_n16: bool = false,
    }{
        .{ .shape = .{ .m = 128, .n = 32, .k = 0 }, .amx = .none },
        .{ .shape = .{ .m = 128, .n = 32, .k = 255 }, .amx = .none },
        .{ .shape = .{ .m = 128, .n = 32, .k = 256 }, .amx = .apple_amx_f32_n32 },
        .{ .shape = .{ .m = 128, .n = 32, .k = 512 }, .amx = .apple_amx_f32_n32 },
        .{ .shape = .{ .m = 128, .n = 32, .k = 513 }, .amx = .none },
        .{ .shape = .{ .m = 128, .n = 32, .k = 4096 }, .amx = .none },
        .{ .shape = .{ .m = 512, .n = 32, .k = 511 }, .amx = .apple_amx_f32_n32 },
        .{ .shape = .{ .m = 512, .n = 32, .k = 512 }, .amx = .none },
        .{ .shape = .{ .m = 512, .n = 32, .k = 2048 }, .amx = .none },
        .{ .shape = .{ .m = 1024, .n = 32, .k = 512 }, .amx = .apple_amx_f32_n32 },
        .{ .shape = .{ .m = 1024, .n = 32, .k = 1024 }, .amx = .none },
        .{ .shape = .{ .m = 512, .n = 512, .k = 512 }, .amx = .apple_amx_f32_n32 },
        .{ .shape = .{ .m = 768, .n = 768, .k = 768 }, .amx = .none },
        .{ .shape = .{ .m = 512, .n = 16, .k = 512 }, .amx = .apple_amx_f32_n16 },
        .{ .shape = .{ .m = 512, .n = 16, .k = 513 }, .amx = .none },
        .{ .shape = .{ .m = 127, .n = 129, .k = 64 }, .amx = .none },
        .{ .shape = .{ .m = 127, .n = 129, .k = 65 }, .amx = .apple_amx_f32_n16 },
        .{ .shape = .{ .m = 127, .n = 129, .k = 256 }, .amx = .apple_amx_f32_n16 },
        .{ .shape = .{ .m = 127, .n = 129, .k = 257 }, .amx = .none },
        .{ .shape = .{ .m = 128, .n = 33, .k = 512 }, .amx = .none, .partial_n16 = true },
        .{ .shape = .{ .m = 128, .n = 33, .k = 513 }, .amx = .none },
    };
    for (f32_cases) |case| {
        const plan = tuning.executionPlan(f32, f32_desc, case.shape, requested_threads, performance_l2_bytes);
        try std.testing.expectEqual(case.amx, plan.amx);
        try std.testing.expectEqual(case.partial_n16, plan.amx_partial_n16);
        if (case.shape.k > 512) {
            try std.testing.expect(!tuning.amxKernelCompatible(f32, .apple_amx_f32_n16, case.shape));
            try std.testing.expect(!tuning.amxKernelCompatible(f32, .apple_amx_f32_n32, case.shape));
        }
    }

    const f64_candidates = catalog.candidateList(.{
        catalog.aarch64SmeDescriptor(f64, 64),
        catalog.aarch64AsimdDescriptor(f64),
        catalog.genericDescriptor(f64),
    });
    const f64_desc = tuning.select(f64, f64_candidates, shape, 1, 0, requested_threads);
    try std.testing.expectEqual(catalog.KernelId.aarch64_sme_f64_4mx2n, f64_desc.kernel);
    const f64_plan = tuning.executionPlan(f64, f64_desc, shape, requested_threads, performance_l2_bytes);
    try std.testing.expectEqual(gemm_task.AppleAmxKernelId.apple_amx_f64_n16, f64_plan.amx);
    const f64_high_k_shape: tuning.Shape = .{ .m = 128, .n = 32, .k = 4096 };
    const f64_high_k_plan = tuning.executionPlan(f64, f64_desc, f64_high_k_shape, requested_threads, performance_l2_bytes);
    try std.testing.expectEqual(gemm_task.AppleAmxKernelId.apple_amx_f64_n32, f64_high_k_plan.amx);
    try std.testing.expect(tuning.amxKernelCompatible(f64, f64_high_k_plan.amx, f64_high_k_shape));
}

test "SME epilogue rejection falls back without entering streaming state" {
    if (comptime !aarch64_features.has_sme) return;
    const before = aarch64_features.testStreamingEntries();
    try runCase(f32, catalog.aarch64SmeDescriptor(f32, streamingVectorBytes()), .no_trans, 0.75, -0.25);
    try std.testing.expectEqual(before, aarch64_features.testStreamingEntries());
    try std.testing.expectEqual(aarch64_features.TestStreamingDepths{ .sm = 0, .za = 0 }, aarch64_features.testStreamingDepths());
    try std.testing.expectEqual(@as(u2, 0), aarch64_features.streamingModeBits());
}

test "Apple AMX success and rejection exits balance state before return" {
    if (comptime builtin.cpu.arch != .aarch64 or builtin.target.os.tag != .macos) return;

    const a: [16]f32 = @splat(1);
    const b: [16]f32 = @splat(1);
    var c: [16 * 16]f32 = @splat(0);
    const workspace: gemm_task.PackWorkspacePlan = .{
        .stack_bytes = 128 * 1024,
        .cache_bytes = 8 * 1024 * 1024,
    };
    defer apple_amx.freeCurrentThreadCaches();

    if (comptime !apple_amx.enabled) {
        const before_disabled = c;
        const entries_before_disabled = apple_amx_ops.testStateEntries();
        try std.testing.expectEqual(@as(c_int, 0), apple_amx.sgemmN16(16, 16, 1, &a, 16, &b, 1, &c, 16, workspace));
        try std.testing.expectEqualSlices(f32, &before_disabled, &c);
        try std.testing.expectEqual(entries_before_disabled, apple_amx_ops.testStateEntries());
        try std.testing.expectEqual(@as(usize, 0), apple_amx_ops.testStateDepth());

        const x = [_]f64{2};
        var pack: [8]f64 = @splat(17);
        const pack_before_disabled = pack;
        try std.testing.expect(!aarch64_matrix_vector.gemvNoTransPackUnitReal(f64, 1, 1, &x, &pack));
        try std.testing.expectEqualSlices(f64, &pack_before_disabled, &pack);
        return;
    }

    const before_success = apple_amx_ops.testStateEntries();
    try std.testing.expectEqual(@as(c_int, 1), apple_amx.sgemmN16(16, 16, 1, &a, 16, &b, 1, &c, 16, workspace));
    try std.testing.expectEqual(before_success + 1, apple_amx_ops.testStateEntries());
    try std.testing.expectEqual(@as(usize, 0), apple_amx_ops.testStateDepth());
    for (c) |value| try std.testing.expectEqual(@as(f32, 1), value);

    const before_rejection = c;
    const entries_before_rejection = apple_amx_ops.testStateEntries();
    try std.testing.expectEqual(@as(c_int, 0), apple_amx.sgemmN16(15, 16, 1, &a, 16, &b, 1, &c, 16, workspace));
    try std.testing.expectEqualSlices(f32, &before_rejection, &c);
    try std.testing.expectEqual(entries_before_rejection, apple_amx_ops.testStateEntries());
    try std.testing.expectEqual(@as(usize, 0), apple_amx_ops.testStateDepth());
}

test "coverage contracts expose every descriptor without executor inspection" {
    const report = coverage.entries(64);
    for (report) |entry| {
        const kernel = entry.kernel orelse continue;
        const desc = catalog.descriptorForKernel(kernel, 64) orelse return error.MissingDescriptor;
        const declared = entry.contract orelse return error.MissingCoverageContract;
        try std.testing.expectEqual(desc.state, entry.state);
        try std.testing.expectEqual(desc.layouts, declared.layouts);
        try std.testing.expectEqual(desc.tails, declared.tails);
        try std.testing.expectEqual(desc.epilogue, declared.epilogue);
        try std.testing.expectEqual(desc.pack.kind, declared.pack_kind);
        try std.testing.expectEqual(desc.pack.stack_bytes, declared.stack_workspace_bytes);
        try std.testing.expectEqual(desc.pack.cache_bytes, declared.cache_workspace_bytes);
        try std.testing.expectEqual(desc.tile, declared.tile);
    }
}

test "structured Level 3 registry is reachable from the matrix-matrix catalog" {
    try std.testing.expectEqual(@as(usize, 138), catalog.structured.registry.len);
    try std.testing.expectEqual(
        catalog.structured.StructuredOperation.trsm,
        catalog.structured.descriptorForKernel(.ztrsm_left_column_parallel).?.operation,
    );
}

fn testForcedComplexPaths(comptime T: type) !void {
    const portable: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .portable_c32 else .portable_c64;
    const compact: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .compact_c32 else .compact_c64;
    const three_m: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .three_m_c32 else .three_m_c64;
    const expanded: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .expanded_real_c32 else .expanded_real_c64;
    const vector_edge: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .vector_edge_c32 else .vector_edge_c64;
    const arbitrary_alpha = complexValue(T, 0.75, -0.25);
    const arbitrary_beta = complexValue(T, -0.375, 0.125);
    const one = scalar.one(T);
    const zero = scalar.zero(T);
    const layouts = [_]ComplexLayout{
        .{ .transa = .no_trans, .transb = .no_trans },
        .{ .transa = .no_trans, .transb = .trans },
        .{ .transa = .trans, .transb = .no_trans },
        .{ .transa = .trans, .transb = .trans },
        .{ .transa = .conj_trans, .transb = .no_trans },
        .{ .transa = .no_trans, .transb = .conj_trans },
        .{ .transa = .conj_trans, .transb = .conj_trans },
    };

    for (layouts) |layout| {
        const result = try runForcedComplexCase(T, portable, .{}, layout, 7, 5, 9, arbitrary_alpha, arbitrary_beta);
        try std.testing.expect(result.usedRequested());
        if (T == complex_gemm.ComplexF64) {
            for ([_]usize{ 3, 4, 5, 7 }) |wide_n| {
                const wide = try runForcedComplexCase(T, portable, .{}, layout, 1, wide_n, 9, arbitrary_alpha, arbitrary_beta);
                try std.testing.expect(wide.usedRequested());
                const wide_zero_beta = try runForcedComplexCase(T, portable, .{}, layout, 1, wide_n, 9, one, zero);
                try std.testing.expect(wide_zero_beta.usedRequested());
                const wide_zero_alpha = try runForcedComplexCase(T, portable, .{}, layout, 1, wide_n, 9, zero, arbitrary_beta);
                try std.testing.expect(wide_zero_alpha.usedRequested());
            }
        }
    }

    const compact_result = try runForcedComplexCase(T, compact, .{}, layouts[0], 7, 5, 9, arbitrary_alpha, arbitrary_beta);
    try std.testing.expect(compact_result.usedRequested());

    for (layouts) |layout| {
        const result = try runForcedComplexCase(T, three_m, .{}, layout, 7, 5, 9, one, zero);
        try std.testing.expect(result.usedRequested());
    }

    const expanded_layout_count: usize = if (T == complex_gemm.ComplexF32) layouts.len else 1;
    for (layouts[0..expanded_layout_count]) |layout| {
        const result = try runForcedComplexCase(T, expanded, .{}, layout, 7, 5, 9, one, zero);
        try std.testing.expect(result.usedRequested());
    }
    const expanded_epilogue_fallback = try runForcedComplexCase(T, expanded, .{}, layouts[0], 7, 5, 9, arbitrary_alpha, arbitrary_beta);
    try std.testing.expectEqual(three_m, expanded_epilogue_fallback.executed.?);
    const three_m_epilogue = try runForcedComplexCase(T, three_m, .{}, layouts[1], 7, 5, 9, one, arbitrary_beta);
    try std.testing.expect(three_m_epilogue.usedRequested());
    if (T == complex_gemm.ComplexF64) {
        const rejected_layout = try runForcedComplexCase(T, expanded, .{}, layouts[2], 7, 5, 9, one, zero);
        try std.testing.expect(!rejected_layout.usedRequested());
        try std.testing.expectEqual(catalog.ComplexKernelId.three_m_c64, rejected_layout.executed.?);
    }

    const row_edge = try runForcedComplexCase(T, vector_edge, .{}, layouts[0], 1, 7, 129, arbitrary_alpha, arbitrary_beta);
    try std.testing.expect(row_edge.usedRequested());
    const column_edge = try runForcedComplexCase(T, vector_edge, .{}, layouts[0], 7, 1, 129, arbitrary_alpha, arbitrary_beta);
    try std.testing.expect(column_edge.usedRequested());
}

test "forced complex GEMM stable IDs cover layouts epilogues vector edges and odd tails" {
    defer complex_gemm.freeCurrentThreadCaches();
    try testForcedComplexPaths(complex_gemm.ComplexF32);
    try testForcedComplexPaths(complex_gemm.ComplexF64);
    const T = complex_gemm.ComplexF64;
    // Three independent real products share the original 3M reduction and
    // epilogue; cover padded and unpadded small-K panels in every layout.
    for ([_]scalar.Order{ .no_trans, .trans, .conj_trans }) |ta| {
        for ([_]scalar.Order{ .no_trans, .trans, .conj_trans }) |tb| {
            for ([_][3]usize{ .{ 96, 96, 32 }, .{ 127, 129, 31 }, .{ 127, 129, 33 } }) |shape| {
                const result = try runForcedComplexCase(T, .three_m_c64, .{}, .{ .transa = ta, .transb = tb }, shape[0], shape[1], shape[2], complexValue(T, 0.75, -0.25), complexValue(T, -0.375, 0.125));
                try std.testing.expect(result.usedRequested());
            }
        }
    }
    const nan = complexValue(T, std.math.nan(f64), std.math.nan(f64));
    const guard = complexValue(T, 73, -19);
    const one = scalar.one(T);
    const zero = scalar.zero(T);
    for ([_]scalar.Order{ .trans, .conj_trans }) |ta| {
        for ([_]scalar.Order{ .trans, .conj_trans }) |tb| {
            for (0..3) |special_case| {
                var a = [_]T{nan} ** 4;
                var b = [_]T{nan} ** 14;
                var c = [_]T{guard} ** 15;
                a[0] = if (special_case == 0) complexValue(T, std.math.inf(f64), 0) else complexValue(T, 2, 0);
                a[1] = complexValue(T, 3, 0);
                for (0..2) |p_index| {
                    for (0..5) |j| b[p_index * 7 + j] = if (special_case == 0) zero else one;
                }
                for (0..5) |j| c[j * 3] = if (special_case == 1) nan else complexValue(T, 4, 0);
                const beta = if (special_case == 2) complexValue(T, 2, 0) else zero;
                const result = complex_gemm.executeForcedComplexKernel(T, .portable_c64, .{}, ta, tb, 1, 5, 2, one, &a, 4, &b, 7, beta, &c, 3);
                try std.testing.expect(result.usedRequested());
                for (0..5) |j| {
                    if (special_case == 0) {
                        try std.testing.expect(std.math.isNan(c[j * 3].re));
                        try std.testing.expect(std.math.isNan(c[j * 3].im));
                    } else {
                        try expectComplexClose(T, complexValue(T, if (special_case == 2) 13 else 5, 0), c[j * 3]);
                    }
                    try std.testing.expectEqual(guard, c[j * 3 + 1]);
                    try std.testing.expectEqual(guard, c[j * 3 + 2]);
                }
            }
        }
    }
}

fn testComplexWorkspaceFallback(comptime T: type) !void {
    const expanded: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .expanded_real_c32 else .expanded_real_c64;
    const three_m: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .three_m_c32 else .three_m_c64;
    const compact: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .compact_c32 else .compact_c64;
    const portable: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .portable_c32 else .portable_c64;
    const unavailable = complex_gemm.ComplexExecutionOptions{ .workspace_available = false };
    const nn = ComplexLayout{ .transa = .no_trans, .transb = .no_trans };
    const nt = ComplexLayout{ .transa = .no_trans, .transb = .trans };

    const expanded_nn = try runForcedComplexCase(T, expanded, unavailable, nn, 7, 5, 9, scalar.one(T), scalar.zero(T));
    try std.testing.expectEqual(compact, expanded_nn.executed.?);
    try std.testing.expect(expanded_nn.fallback_count >= 2);

    const three_m_nt = try runForcedComplexCase(T, three_m, unavailable, nt, 7, 5, 9, scalar.one(T), scalar.zero(T));
    try std.testing.expectEqual(portable, three_m_nt.executed.?);
    try std.testing.expect(three_m_nt.fallback_count >= 2);
}

test "complex workspace denial falls back before output mutation" {
    defer complex_gemm.freeCurrentThreadCaches();
    try testComplexWorkspaceFallback(complex_gemm.ComplexF32);
    try testComplexWorkspaceFallback(complex_gemm.ComplexF64);
}

fn expectComplexSelection(comptime T: type, expected: catalog.ComplexKernelId, layout: ComplexLayout, m: usize, n: usize, k: usize, alpha: T, beta: T) !void {
    const selected = complex_gemm.selectComplexKernel(T, layout.transa, layout.transb, @intCast(m), @intCast(n), @intCast(k), alpha, beta);
    try std.testing.expectEqual(expected, selected);
}

test "complex GEMM selection is deterministic across layouts and scalar classes" {
    const nn = ComplexLayout{ .transa = .no_trans, .transb = .no_trans };
    const nt = ComplexLayout{ .transa = .no_trans, .transb = .trans };
    const ct = ComplexLayout{ .transa = .conj_trans, .transb = .trans };

    inline for (.{ complex_gemm.ComplexF32, complex_gemm.ComplexF64 }) |T| {
        const portable: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .portable_c32 else .portable_c64;
        const compact: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .compact_c32 else .compact_c64;
        const vector_edge: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .vector_edge_c32 else .vector_edge_c64;
        const three_m: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32) .three_m_c32 else .three_m_c64;
        const materialized_non_nn: catalog.ComplexKernelId = if (T == complex_gemm.ComplexF32 and builtin.cpu.arch != .x86_64) .expanded_real_c32 else three_m;
        const arbitrary_alpha = complexValue(T, 0.75, -0.25);
        const arbitrary_beta = complexValue(T, -0.375, 0.125);

        try expectComplexSelection(T, compact, nn, 7, 5, 9, arbitrary_alpha, arbitrary_beta);
        try expectComplexSelection(T, vector_edge, nn, 1, 7, 129, arbitrary_alpha, arbitrary_beta);
        try expectComplexSelection(T, portable, nt, 7, 5, 9, scalar.one(T), scalar.zero(T));
        try expectComplexSelection(T, materialized_non_nn, nt, 32, 32, 128, scalar.one(T), scalar.zero(T));
        try expectComplexSelection(T, materialized_non_nn, ct, 32, 32, 128, scalar.one(T), scalar.zero(T));
        try expectComplexSelection(T, three_m, nt, 32, 32, 128, arbitrary_alpha, scalar.zero(T));
        try expectComplexSelection(T, three_m, nt, 32, 32, 128, scalar.one(T), arbitrary_beta);
        try expectComplexSelection(T, portable, nn, 32, 32, 128, scalar.zero(T), arbitrary_beta);
    }

    try expectComplexSelection(complex_gemm.ComplexF32, .three_m_c32, nn, 64, 64, 64, scalar.one(complex_gemm.ComplexF32), scalar.zero(complex_gemm.ComplexF32));
    try expectComplexSelection(complex_gemm.ComplexF64, .expanded_real_c64, nn, 64, 64, 64, scalar.one(complex_gemm.ComplexF64), scalar.zero(complex_gemm.ComplexF64));
}
