// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Static Level 1 implementation contracts.
//!
//! This catalog records executable facts only: operation/type coverage and
//! hard stride, alignment, alias, tail, reduction, state, and fallback
//! semantics. Length thresholds and architecture preferences belong in named
//! tuning profiles. Precise ISA microkernels are added as distinct entries;
//! the portable contiguous execution layer must not be mistaken for one.

const std = @import("std");
const contract = @import("../../contract.zig");

pub const OperationKind = contract.OperationKind;
pub const ScalarKind = contract.ScalarKind;
pub const IsaCapability = contract.IsaCapability;
pub const Lifecycle = contract.Lifecycle;
pub const TailStrategy = contract.TailStrategy;
pub const VectorStrideRule = contract.VectorStrideRule;
pub const VectorStrideContract = contract.VectorStrideContract;
pub const AlignmentContract = contract.AlignmentContract;
pub const AliasContract = contract.AliasContract;
pub const ReductionKind = contract.ReductionKind;
pub const ReductionContract = contract.ReductionContract;
pub const StateContract = contract.StateContract;
pub const FallbackSemantics = contract.FallbackSemantics;

pub const VectorOperation = enum {
    scal,
    rscal,
    copy,
    swap,
    axpy,
    axpby,
    dot,
    dotu,
    dotc,
    dot_f32_acc_f64,
    asum,
    nrm2,
    iamax,
    rot,
    rotg,
    rotm,
    rotmg,
};

pub const Implementation = enum {
    portable_scalar,
    portable_contiguous,
    aarch64_asimd,
    aarch64_sve,
    aarch64_sme2_streaming,
    x86_64_fixed_simd,
    x86_64_stride2_parallel,
};

/// Hard SME feasibility requirements that are independent of tuning policy.
/// A zero streaming-vector length means the implementation is not an SME
/// streaming body. The optional FP64 ZA feature is deliberately separate from
/// the SME2 capability because Arm exposes it as an independent feature bit.
pub const SmeRequirements = struct {
    streaming_vector_bytes: usize = 0,
    requires_f64f64: bool = false,
};

pub const EntryScope = enum {
    core_and_abi,
    abi_only,
};

pub const KernelId = struct {
    operation: VectorOperation,
    scalar: ScalarKind,
    implementation: Implementation,
    capability: IsaCapability,
};

pub const Descriptor = struct {
    name: []const u8,
    kernel: KernelId,
    fallback: ?KernelId,
    operation_kind: OperationKind,
    lifecycle: Lifecycle,
    entry_scope: EntryScope,
    strides: VectorStrideContract,
    alignment: AlignmentContract,
    aliasing: AliasContract,
    tails: TailStrategy,
    reduction: ReductionContract,
    state: StateContract,
    sme: SmeRequirements,
    fallback_semantics: FallbackSemantics,
};

pub const max_descriptors = 640;

pub const Registry = struct {
    len: usize = 0,
    items: [max_descriptors]Descriptor = undefined,

    pub fn slice(self: *const Registry) []const Descriptor {
        return self.items[0..self.len];
    }

    fn append(self: *Registry, descriptor: Descriptor) void {
        if (self.len == self.items.len) @compileError("Level 1 registry capacity is too small");
        self.items[self.len] = descriptor;
        self.len += 1;
    }
};

const all_scalars = [_]ScalarKind{ .f32, .f64, .complex_f32, .complex_f64 };
const real_scalars = [_]ScalarKind{ .f32, .f64 };
const complex_scalars = [_]ScalarKind{ .complex_f32, .complex_f64 };

const all_scalar_operations = [_]VectorOperation{
    .scal,
    .copy,
    .swap,
    .axpy,
    .axpby,
    .asum,
    .nrm2,
    .iamax,
    .rot,
    .rotg,
};

const all_contiguous_operations = [_]VectorOperation{
    .scal,
    .copy,
    .swap,
    .axpy,
    .axpby,
    .asum,
    .nrm2,
    .iamax,
    .rot,
};

const x86_stride2_all_scalars = [_]VectorOperation{
    .scal,
    .swap,
    .axpy,
    .axpby,
    .asum,
    .nrm2,
    .iamax,
    .rot,
};

// These lists describe architecture entrypoints that exist today.  They do
// not infer ISA ownership merely because the portable layer also uses Zig
// vectors.  Complex operations that deliberately reuse a real-lane kernel are
// included because the architecture entrypoint is the executable body.
const fixed_arch_all_scalars = [_]VectorOperation{
    .scal,
    .copy,
    .swap,
    .axpy,
    .axpby,
    .asum,
    .nrm2,
    .rot,
};

const x86_fixed_capabilities = [_]IsaCapability{
    .x86_64_sse2,
    .x86_64_avx,
    .x86_64_fma,
    .x86_64_avx2,
    .x86_64_avx2_fma,
    .x86_64_avx512f_fma,
};

pub const registry = buildRegistry();
pub const registered_descriptor_count = registry.len;

fn buildRegistry() Registry {
    @setEvalBranchQuota(2_000_000);
    var out = Registry{};

    appendCrossProduct(&out, all_scalar_operations, all_scalars, .portable_scalar);
    appendCrossProduct(&out, &.{ .dot, .rotm, .rotmg }, real_scalars, .portable_scalar);
    appendCrossProduct(&out, &.{ .dotu, .dotc, .rscal }, complex_scalars, .portable_scalar);
    out.append(makeDescriptor(.dot_f32_acc_f64, .f32, .portable_scalar));

    appendCrossProduct(&out, all_contiguous_operations, all_scalars, .portable_contiguous);
    appendCrossProduct(&out, &.{ .dot, .rotm }, real_scalars, .portable_contiguous);
    appendCrossProduct(&out, &.{ .dotu, .dotc, .rscal }, complex_scalars, .portable_contiguous);
    out.append(makeDescriptor(.dot_f32_acc_f64, .f32, .portable_contiguous));

    appendFixedArchitecture(&out, .aarch64_asimd, .aarch64_asimd_fma);
    appendSveArchitecture(&out, .aarch64_sve);
    // SVE2 is a strict architectural superset for these floating-point
    // kernels. A separate target-tier descriptor names the executable SVE2
    // build without pretending that the shared instruction body uses an
    // SVE2-only instruction.
    appendSveArchitecture(&out, .aarch64_sve2);
    appendSme2Architecture(&out);
    inline for (x86_fixed_capabilities) |capability| {
        appendFixedArchitecture(&out, .x86_64_fixed_simd, capability);
    }

    appendCrossProduct(&out, x86_stride2_all_scalars, all_scalars, .x86_64_stride2_parallel);
    appendCrossProduct(&out, &.{ .dot, .rotm }, real_scalars, .x86_64_stride2_parallel);
    appendCrossProduct(&out, &.{ .dotu, .dotc, .rscal }, complex_scalars, .x86_64_stride2_parallel);
    out.append(makeDescriptor(.dot_f32_acc_f64, .f32, .x86_64_stride2_parallel));

    validateRegistry(&out);
    return out;
}

fn appendSme2Architecture(out: *Registry) void {
    // Register only complete public-operation cells. Complex SCAL/AXPY/AXPBY
    // with general complex coefficients are not represented by the real-lane
    // SME bodies and therefore remain explicit coverage gaps.
    appendArchitectureCrossProduct(out, &.{.scal}, real_scalars, .aarch64_sme2_streaming, .aarch64_sme2);
    appendArchitectureCrossProduct(out, &.{.rscal}, complex_scalars, .aarch64_sme2_streaming, .aarch64_sme2);
    appendArchitectureCrossProduct(out, &.{ .copy, .swap }, all_scalars, .aarch64_sme2_streaming, .aarch64_sme2);
    appendArchitectureCrossProduct(out, &.{.axpy}, real_scalars, .aarch64_sme2_streaming, .aarch64_sme2);
    out.append(makeArchitectureDescriptor(.axpby, .f32, .aarch64_sme2_streaming, .aarch64_sme2));
    appendArchitectureCrossProduct(out, &.{.dot}, real_scalars, .aarch64_sme2_streaming, .aarch64_sme2);
    appendArchitectureCrossProduct(out, &.{.asum}, all_scalars, .aarch64_sme2_streaming, .aarch64_sme2);
    out.append(makeArchitectureDescriptor(.rot, .f32, .aarch64_sme2_streaming, .aarch64_sme2));
    out.append(makeArchitectureDescriptor(.rotm, .f32, .aarch64_sme2_streaming, .aarch64_sme2));
}

fn appendFixedArchitecture(
    out: *Registry,
    comptime implementation: Implementation,
    comptime capability: IsaCapability,
) void {
    inline for (fixed_arch_all_scalars) |operation| {
        inline for (all_scalars) |scalar| {
            out.append(makeArchitectureDescriptor(operation, scalar, implementation, capability));
        }
    }
    appendArchitectureCrossProduct(out, &.{.iamax}, all_scalars, implementation, capability);
    appendArchitectureCrossProduct(out, &.{.dot}, real_scalars, implementation, capability);
    appendArchitectureCrossProduct(out, &.{.rotm}, real_scalars, implementation, capability);
    appendArchitectureCrossProduct(out, &.{ .dotu, .dotc, .rscal }, complex_scalars, implementation, capability);
    out.append(makeArchitectureDescriptor(.dot_f32_acc_f64, .f32, implementation, capability));
}

fn appendSveArchitecture(out: *Registry, comptime capability: IsaCapability) void {
    inline for (fixed_arch_all_scalars) |operation| {
        inline for (all_scalars) |scalar| {
            out.append(makeArchitectureDescriptor(operation, scalar, .aarch64_sve, capability));
        }
    }
    appendArchitectureCrossProduct(out, &.{.iamax}, all_scalars, .aarch64_sve, capability);
    appendArchitectureCrossProduct(out, &.{ .dot, .rotm }, real_scalars, .aarch64_sve, capability);
    appendArchitectureCrossProduct(out, &.{ .dotu, .dotc, .rscal }, complex_scalars, .aarch64_sve, capability);
    out.append(makeArchitectureDescriptor(.dot_f32_acc_f64, .f32, .aarch64_sve, capability));
}

fn appendArchitectureCrossProduct(
    out: *Registry,
    comptime operations: anytype,
    comptime scalars: anytype,
    comptime implementation: Implementation,
    comptime capability: IsaCapability,
) void {
    inline for (operations) |operation| {
        inline for (scalars) |scalar| {
            out.append(makeArchitectureDescriptor(operation, scalar, implementation, capability));
        }
    }
}

fn appendCrossProduct(
    out: *Registry,
    comptime operations: anytype,
    comptime scalars: anytype,
    comptime implementation: Implementation,
) void {
    inline for (operations) |operation| {
        inline for (scalars) |scalar| {
            out.append(makeDescriptor(operation, scalar, implementation));
        }
    }
}

fn makeDescriptor(
    comptime operation: VectorOperation,
    comptime scalar: ScalarKind,
    comptime implementation: Implementation,
) Descriptor {
    const capability: IsaCapability = switch (implementation) {
        .portable_scalar, .portable_contiguous => .generic,
        .aarch64_asimd, .aarch64_sve, .aarch64_sme2_streaming, .x86_64_fixed_simd => unreachable,
        // x86-64 guarantees SSE2. The implementation expresses 512-bit Zig
        // vectors but intentionally permits the compiler to lower them into
        // narrower instructions, so claiming AVX-512 would be incorrect.
        .x86_64_stride2_parallel => .x86_64_sse2,
    };
    return makeDescriptorWithCapability(operation, scalar, implementation, capability);
}

fn makeArchitectureDescriptor(
    comptime operation: VectorOperation,
    comptime scalar: ScalarKind,
    comptime implementation: Implementation,
    comptime capability: IsaCapability,
) Descriptor {
    return makeDescriptorWithCapability(operation, scalar, implementation, capability);
}

fn makeDescriptorWithCapability(
    comptime operation: VectorOperation,
    comptime scalar: ScalarKind,
    comptime implementation: Implementation,
    comptime capability: IsaCapability,
) Descriptor {
    const terminal = implementation == .portable_scalar;
    const kernel: KernelId = .{
        .operation = operation,
        .scalar = scalar,
        .implementation = implementation,
        .capability = capability,
    };
    return .{
        .name = std.fmt.comptimePrint("level1.{s}.{s}.{s}.{s}", .{
            @tagName(operation),
            @tagName(scalar),
            @tagName(implementation),
            @tagName(capability),
        }),
        .kernel = kernel,
        .fallback = fallbackFor(operation, scalar, implementation),
        .operation_kind = operationKind(operation),
        .lifecycle = lifecycleFor(operation, scalar, implementation, capability),
        .entry_scope = if (implementation == .x86_64_stride2_parallel) .abi_only else .core_and_abi,
        .strides = strideContract(operation, implementation),
        .alignment = .{},
        .aliasing = aliasContract(operation, implementation),
        .tails = if (terminal)
            .native
        else if (implementation == .aarch64_sve or implementation == .aarch64_sme2_streaming)
            .predicated
        else
            .scalar_cleanup,
        .reduction = reductionContract(operation),
        .state = if (implementation == .aarch64_sme2_streaming)
            contract.stateContract(smeStateKind(operation))
        else
            .{},
        .sme = if (implementation == .aarch64_sme2_streaming) .{
            .streaming_vector_bytes = 64,
            .requires_f64f64 = smeRequiresF64F64(operation, scalar),
        } else .{},
        .fallback_semantics = if (terminal) .terminal_portable else .whole_operation,
    };
}

fn smeStateKind(comptime operation: VectorOperation) contract.StateKind {
    return switch (operation) {
        .copy, .swap => .aarch64_streaming_sm,
        else => .aarch64_streaming_za,
    };
}

fn smeRequiresF64F64(comptime operation: VectorOperation, comptime scalar: ScalarKind) bool {
    return switch (operation) {
        .scal, .axpy, .dot => scalar == .f64,
        .rscal => scalar == .complex_f64,
        .asum => scalar == .f64 or scalar == .complex_f64,
        else => false,
    };
}

fn lifecycleFor(
    comptime operation: VectorOperation,
    comptime scalar: ScalarKind,
    comptime implementation: Implementation,
    comptime capability: IsaCapability,
) Lifecycle {
    if (implementation == .portable_scalar) return .portable_fallback;
    if (implementation == .aarch64_sve) return .experimental;
    if (operation == .rotm and
        (implementation == .aarch64_asimd or implementation == .x86_64_fixed_simd))
    {
        return .experimental;
    }
    if (operation == .dot_f32_acc_f64 and implementation == .aarch64_asimd) return .experimental;
    if (operation == .dot_f32_acc_f64 and implementation == .x86_64_fixed_simd and
        capability != .x86_64_avx512f_fma) return .experimental;
    if (implementation == .x86_64_fixed_simd and operation == .iamax and
        (scalar == .complex_f32 or scalar == .complex_f64) and
        capability != .x86_64_avx512f_fma) return .experimental;
    // The narrow AArch64 fixed-width ROT and real AXPBY leaves lost to the
    // wider portable vector bodies in retained native measurements.  Keep the
    // executable cells visible without making them default-eligible.
    if (implementation == .aarch64_asimd and operation == .rot) return .rejected;
    if (implementation == .aarch64_asimd and operation == .axpby and
        (scalar == .f32 or scalar == .f64)) return .rejected;
    return .production;
}

fn fallbackFor(
    comptime operation: VectorOperation,
    comptime scalar: ScalarKind,
    comptime implementation: Implementation,
) ?KernelId {
    return switch (implementation) {
        .portable_scalar => null,
        .portable_contiguous, .x86_64_stride2_parallel => portableFallback(operation, scalar),
        .aarch64_asimd, .aarch64_sve, .aarch64_sme2_streaming, .x86_64_fixed_simd => .{
            .operation = operation,
            .scalar = scalar,
            .implementation = .portable_contiguous,
            .capability = .generic,
        },
    };
}

fn portableFallback(comptime operation: VectorOperation, comptime scalar: ScalarKind) KernelId {
    return .{
        .operation = operation,
        .scalar = scalar,
        .implementation = .portable_scalar,
        .capability = .generic,
    };
}

fn operationKind(operation: VectorOperation) OperationKind {
    return switch (operation) {
        .rotg, .rotmg => .scalar_generator,
        .scal, .rscal, .asum, .nrm2, .iamax => .vector_unary,
        .copy, .swap, .axpy, .axpby, .dot, .dotu, .dotc, .dot_f32_acc_f64, .rot, .rotm => .vector_binary,
    };
}

fn strideContract(operation: VectorOperation, implementation: Implementation) VectorStrideContract {
    if (operation == .rotg or operation == .rotmg) return .{ .x = .not_applicable };

    const unary = operationKind(operation) == .vector_unary;
    return switch (implementation) {
        .portable_scalar => .{
            .x = if (operation == .iamax) .positive else .any_nonzero,
            .y = if (unary) .not_applicable else .any_nonzero,
        },
        .portable_contiguous, .aarch64_asimd, .aarch64_sve, .aarch64_sme2_streaming, .x86_64_fixed_simd => .{
            .x = .unit,
            .y = if (unary) .not_applicable else .unit,
        },
        .x86_64_stride2_parallel => .{
            .x = .exactly_two,
            .y = if (unary) .not_applicable else .exactly_two,
        },
    };
}

fn aliasContract(operation: VectorOperation, implementation: Implementation) AliasContract {
    if (implementation == .aarch64_sme2_streaming and operation == .swap) return .non_overlapping;
    if (implementation != .x86_64_stride2_parallel) return .blas_valid;
    return switch (operation) {
        .swap, .axpy, .axpby, .rot, .rotm => .non_overlapping,
        else => .blas_valid,
    };
}

fn reductionContract(operation: VectorOperation) ReductionContract {
    return switch (operation) {
        .dot, .dotu, .dotc, .dot_f32_acc_f64, .asum => .{ .kind = .sum },
        .nrm2 => .{ .kind = .scaled_sum_of_squares },
        .iamax => .{ .kind = .first_index_maximum, .preserves_first_index_on_ties = true },
        else => .{},
    };
}

fn validateRegistry(out: *const Registry) void {
    for (out.slice(), 0..) |descriptor, index| {
        validateDescriptor(descriptor);
        for (out.slice()[index + 1 ..]) |other| {
            if (std.mem.eql(u8, descriptor.name, other.name)) @compileError("duplicate Level 1 kernel name");
            if (std.meta.eql(descriptor.kernel, other.kernel)) @compileError("duplicate Level 1 kernel id");
        }
        if (descriptor.fallback) |fallback| {
            if (!containsKernel(out, fallback)) @compileError("Level 1 fallback is absent from the registry");
        }
    }
}

fn validateDescriptor(descriptor: Descriptor) void {
    if (descriptor.name.len == 0) @compileError("Level 1 kernel name must not be empty");
    if (!std.math.isPowerOfTwo(descriptor.alignment.x_bytes) or !std.math.isPowerOfTwo(descriptor.alignment.y_bytes)) {
        @compileError("Level 1 alignment must be a nonzero power of two");
    }
    if (descriptor.fallback_semantics == .terminal_portable) {
        if (descriptor.fallback != null or descriptor.lifecycle != .portable_fallback) {
            @compileError("terminal Level 1 implementation must be a portable fallback");
        }
    } else if (descriptor.fallback == null) {
        @compileError("nonterminal Level 1 implementation lacks a fallback");
    }
    if (descriptor.entry_scope == .abi_only and descriptor.kernel.implementation != .x86_64_stride2_parallel) {
        @compileError("unexpected ABI-only Level 1 implementation");
    }
    switch (descriptor.kernel.implementation) {
        .aarch64_asimd => if (descriptor.kernel.capability != .aarch64_asimd_fma) {
            @compileError("ASIMD Level 1 descriptor has the wrong capability");
        },
        .aarch64_sve => if (descriptor.kernel.capability != .aarch64_sve and
            descriptor.kernel.capability != .aarch64_sve2)
        {
            @compileError("SVE Level 1 descriptor has the wrong capability");
        },
        .aarch64_sme2_streaming => if (descriptor.kernel.capability != .aarch64_sme2) {
            @compileError("SME2 Level 1 descriptor has the wrong capability");
        },
        .x86_64_fixed_simd => switch (descriptor.kernel.capability) {
            .x86_64_sse2,
            .x86_64_avx,
            .x86_64_fma,
            .x86_64_avx2,
            .x86_64_avx2_fma,
            .x86_64_avx512f,
            .x86_64_avx512f_fma,
            => {},
            else => @compileError("x86 fixed-SIMD Level 1 descriptor has the wrong capability"),
        },
        else => {},
    }
    contract.validateStateContract(descriptor.state);
    if (descriptor.kernel.implementation == .aarch64_sme2_streaming) {
        if (descriptor.sme.streaming_vector_bytes != 64) {
            @compileError("SME2 Level 1 descriptor lacks the retained 64-byte SVL requirement");
        }
        if (descriptor.state.kind != smeStateKind(descriptor.kernel.operation)) {
            @compileError("SME2 Level 1 descriptor has the wrong state ownership");
        }
    } else if (!std.meta.eql(descriptor.sme, SmeRequirements{})) {
        @compileError("non-SME Level 1 descriptor declares SME requirements");
    }
    if (descriptor.reduction.preserves_first_index_on_ties and descriptor.reduction.kind != .first_index_maximum) {
        @compileError("first-index tie semantics require an index reduction");
    }
}

fn containsKernel(out: *const Registry, kernel: KernelId) bool {
    for (out.slice()) |descriptor| {
        if (std.meta.eql(descriptor.kernel, kernel)) return true;
    }
    return false;
}

fn find(operation: VectorOperation, scalar: ScalarKind, implementation: Implementation) ?Descriptor {
    for (registry.slice()) |descriptor| {
        if (descriptor.kernel.operation == operation and
            descriptor.kernel.scalar == scalar and
            descriptor.kernel.implementation == implementation)
        {
            return descriptor;
        }
    }
    return null;
}

test "Level 1 registry keeps every public operation and datatype total" {
    try std.testing.expectEqual(@as(usize, 589), registered_descriptor_count);
    try std.testing.expectEqual(@as(usize, 53), countImplementation(.portable_scalar));
    try std.testing.expectEqual(@as(usize, 47), countImplementation(.portable_contiguous));
    try std.testing.expectEqual(@as(usize, 47), countImplementation(.aarch64_asimd));
    try std.testing.expectEqual(@as(usize, 94), countImplementation(.aarch64_sve));
    try std.testing.expectEqual(@as(usize, 23), countImplementation(.aarch64_sme2_streaming));
    try std.testing.expectEqual(@as(usize, 282), countImplementation(.x86_64_fixed_simd));
    try std.testing.expectEqual(@as(usize, 43), countImplementation(.x86_64_stride2_parallel));
    for (registry.slice()) |descriptor| {
        if (descriptor.fallback) |fallback| try std.testing.expect(find(
            fallback.operation,
            fallback.scalar,
            fallback.implementation,
        ) != null);
    }
}

test "Level 1 hard contracts distinguish contiguous and ABI-only stride two paths" {
    const contiguous = find(.axpy, .complex_f64, .portable_contiguous).?;
    try std.testing.expectEqual(VectorStrideRule.unit, contiguous.strides.x);
    try std.testing.expectEqual(EntryScope.core_and_abi, contiguous.entry_scope);

    const stride2 = find(.axpy, .complex_f64, .x86_64_stride2_parallel).?;
    try std.testing.expectEqual(VectorStrideRule.exactly_two, stride2.strides.x);
    try std.testing.expectEqual(AliasContract.non_overlapping, stride2.aliasing);
    try std.testing.expectEqual(EntryScope.abi_only, stride2.entry_scope);
    try std.testing.expectEqual(TailStrategy.scalar_cleanup, stride2.tails);
    try std.testing.expectEqual(@as(usize, 1), stride2.alignment.x_bytes);
}

test "Level 1 reduction contracts preserve required semantics" {
    const iamax = find(.iamax, .complex_f32, .portable_scalar).?;
    try std.testing.expectEqual(ReductionKind.first_index_maximum, iamax.reduction.kind);
    try std.testing.expect(iamax.reduction.preserves_first_index_on_ties);
    try std.testing.expectEqual(VectorStrideRule.positive, iamax.strides.x);

    const nrm2 = find(.nrm2, .f64, .portable_contiguous).?;
    try std.testing.expectEqual(ReductionKind.scaled_sum_of_squares, nrm2.reduction.kind);
}

test "Level 1 architecture contracts expose precise build tiers" {
    const asimd = findCapability(.axpy, .f32, .aarch64_asimd, .aarch64_asimd_fma).?;
    try std.testing.expectEqual(VectorStrideRule.unit, asimd.strides.x);
    try std.testing.expectEqual(Implementation.portable_contiguous, asimd.fallback.?.implementation);

    const sve = findCapability(.dotc, .complex_f64, .aarch64_sve, .aarch64_sve).?;
    try std.testing.expectEqual(TailStrategy.predicated, sve.tails);
    try std.testing.expectEqual(Lifecycle.experimental, sve.lifecycle);

    const sme_copy = findCapability(.copy, .complex_f64, .aarch64_sme2_streaming, .aarch64_sme2).?;
    try std.testing.expectEqual(contract.StateKind.aarch64_streaming_sm, sme_copy.state.kind);
    try std.testing.expectEqual(@as(usize, 64), sme_copy.sme.streaming_vector_bytes);
    try std.testing.expect(!sme_copy.sme.requires_f64f64);
    try std.testing.expectEqual(TailStrategy.predicated, sme_copy.tails);

    const sme_dot = findCapability(.dot, .f64, .aarch64_sme2_streaming, .aarch64_sme2).?;
    try std.testing.expectEqual(contract.StateKind.aarch64_streaming_za, sme_dot.state.kind);
    try std.testing.expect(sme_dot.sme.requires_f64f64);
    try std.testing.expectEqual(Implementation.portable_contiguous, sme_dot.fallback.?.implementation);

    const avx2 = findCapability(.dot, .f64, .x86_64_fixed_simd, .x86_64_avx2).?;
    const avx2_fma = findCapability(.dot, .f64, .x86_64_fixed_simd, .x86_64_avx2_fma).?;
    try std.testing.expect(!std.meta.eql(avx2.kernel, avx2_fma.kernel));

    const rejected_rot = findCapability(.rot, .f32, .aarch64_asimd, .aarch64_asimd_fma).?;
    try std.testing.expectEqual(Lifecycle.rejected, rejected_rot.lifecycle);

    const experimental_rotm = findCapability(.rotm, .f64, .x86_64_fixed_simd, .x86_64_avx2_fma).?;
    try std.testing.expectEqual(Lifecycle.experimental, experimental_rotm.lifecycle);
}

fn findCapability(
    operation: VectorOperation,
    scalar: ScalarKind,
    implementation: Implementation,
    capability: IsaCapability,
) ?Descriptor {
    for (registry.slice()) |descriptor| {
        if (descriptor.kernel.operation == operation and
            descriptor.kernel.scalar == scalar and
            descriptor.kernel.implementation == implementation and
            descriptor.kernel.capability == capability)
        {
            return descriptor;
        }
    }
    return null;
}

fn countImplementation(implementation: Implementation) usize {
    var count: usize = 0;
    for (registry.slice()) |descriptor| {
        if (descriptor.kernel.implementation == implementation) count += 1;
    }
    return count;
}
