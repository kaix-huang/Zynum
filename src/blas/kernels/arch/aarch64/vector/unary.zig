// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 single-vector specialized kernels.

const std = @import("std");

const builders = @import("../asm/builders.zig");
const features = @import("../features.zig");
const simd_config = @import("../simd_config.zig");
const fixed_simd = @import("../../../shared/vector/fixed_simd.zig");
const tuning = @import("../../../shared/vector/tuning.zig");
const types = @import("../../../../types.zig");
const vector_asm = @import("../asm/vector.zig");

const profile = tuning.active.aarch64;

fn asimdDscalF64(n: usize, alpha: f64, x: [*]f64) void {
    const V = @Vector(32, f64);
    const alpha_v: V = @splat(alpha);
    var i: usize = 0;
    while (i + 64 <= n) : (i += 64) {
        inline for (0..2) |k| {
            const offset = i + 32 * k;
            const xv: V = @as(*align(1) const V, @ptrCast(x + offset)).*;
            @as(*align(1) V, @ptrCast(x + offset)).* = xv * alpha_v;
        }
    }
    while (i + 32 <= n) : (i += 32) {
        const xv: V = @as(*align(1) const V, @ptrCast(x + i)).*;
        @as(*align(1) V, @ptrCast(x + i)).* = xv * alpha_v;
    }
    while (i < n) : (i += 1) x[i] *= alpha;
}

inline fn callScalF64Kernel(comptime kernel: anytype, n: usize, alpha: f64, x: [*]f64) void {
    const Kernel = *const fn (usize, f64, [*]f64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, x);
}

inline fn callScalF32Kernel(comptime kernel: anytype, n: usize, alpha: f32, x: [*]f32) void {
    const Kernel = *const fn (usize, f32, [*]f32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, x);
}

inline fn callSmScalF64Kernel(comptime kernel: anytype, n: usize, alpha_bits: u64, x: [*]f64) void {
    const Kernel = *const fn (usize, [*]f64, u64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, alpha_bits);
}

inline fn callSmScalF32Kernel(comptime kernel: anytype, n: usize, alpha_bits: u32, x: [*]f32) void {
    const Kernel = *const fn (usize, [*]f32, u32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, alpha_bits);
}

inline fn callAsumF64Kernel(comptime kernel: anytype, n: usize, x: [*]const f64) u64 {
    const Kernel = *const fn (usize, [*]const f64) callconv(.c) u64;
    return @as(Kernel, @ptrCast(&kernel))(n, x);
}

inline fn callAsumF32Kernel(comptime kernel: anytype, n: usize, x: [*]const f32) u32 {
    const Kernel = *const fn (usize, [*]const f32) callconv(.c) u32;
    return @as(Kernel, @ptrCast(&kernel))(n, x);
}

inline fn callComplexScalF32Kernel(comptime kernel: anytype, n: usize, alpha: types.ComplexF32, x: [*]types.ComplexF32) void {
    const Kernel = *const fn (usize, f32, f32, [*]u8) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha.re, alpha.im, @ptrCast(x));
}

inline fn callComplexScalF64Kernel(comptime kernel: anytype, n: usize, alpha: types.ComplexF64, x: [*]types.ComplexF64) void {
    const Kernel = *const fn (usize, f64, f64, [*]u8) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha.re, alpha.im, @ptrCast(x));
}

inline fn callMaxAbsF32Kernel(comptime kernel: anytype, n: usize, x: anytype) f32 {
    const Kernel = *const fn (usize, [*]const u8) callconv(.c) u32;
    return @bitCast(@as(Kernel, @ptrCast(&kernel))(n, @ptrCast(x)));
}

inline fn callMaxAbsF64Kernel(comptime kernel: anytype, n: usize, x: anytype) f64 {
    const Kernel = *const fn (usize, [*]const u8) callconv(.c) u64;
    return @bitCast(@as(Kernel, @ptrCast(&kernel))(n, @ptrCast(x)));
}

inline fn callScaledSsqF32Kernel(comptime kernel: anytype, n: usize, scale: f32, x: [*]const f32) f32 {
    const Kernel = *const fn (usize, f32, [*]const f32) callconv(.c) u32;
    return @bitCast(@as(Kernel, @ptrCast(&kernel))(n, scale, x));
}

inline fn callScaledSsqF64Kernel(comptime kernel: anytype, n: usize, scale: f64, x: [*]const f64) f64 {
    const Kernel = *const fn (usize, f64, [*]const f64) callconv(.c) u64;
    return @bitCast(@as(Kernel, @ptrCast(&kernel))(n, scale, x));
}

inline fn scalUnitRealDisabled(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    _ = n;
    _ = alpha;
    _ = x;
    return false;
}

inline fn asumUnitRealDisabled(comptime T: type, n: usize, x: [*]const T) ?T {
    _ = n;
    _ = x;
    return null;
}

pub fn scalUnitReal(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime !features.has_asimd and !(profile.enable_sme_scal and features.has_sme2) and
        !((profile.enable_sve_scal_f32 or profile.enable_sve_scal_f64) and features.has_sve))
    {
        return scalUnitRealDisabled(T, n, alpha, x);
    }
    if (comptime profile.enable_asimd_scal_f64 and features.has_asimd) {
        if (profile.preferAsimdScal(T, n)) {
            asimdDscalF64(n, alpha, x);
            return true;
        }
    }
    if (comptime profile.enable_sme_scal and features.has_sme2) {
        if (T == f32 and profile.preferSmeScal(T, n) and features.streamingVectorBytes() == 64) {
            const alpha_bits: u32 = @bitCast(alpha);
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();

            callSmScalF32Kernel(smeSscalF32Streaming, n, alpha_bits, x);
            return true;
        }
        if (T == f64 and comptime features.has_sme_f64f64) {
            if (profile.preferSmeScal(T, n) and features.streamingVectorBytes() == 64) {
                const alpha_bits: u64 = @bitCast(alpha);
                var sm_state: features.StreamingModeState = undefined;
                sm_state.startSmZa();
                defer sm_state.stopSmZa();

                callSmScalF64Kernel(smeDscalF64Streaming, n, alpha_bits, x);
                return true;
            }
        }
    }
    if (comptime (profile.enable_sve_scal_f32 or profile.enable_sve_scal_f64) and features.has_sve) {
        if (T == f32 and profile.preferSveScal(T, n)) {
            return sveScalF32Candidate(n, alpha, x);
        }
        if (T == f64 and profile.preferSveScal(T, n)) {
            callScalF64Kernel(vector_asm.dscalSveF64, n, alpha, x);
            return true;
        }
    }
    if (comptime features.has_asimd) return fixed_simd.scalUnitReal(T, simd_config.vectorConfig(T), n, alpha, x);
    return false;
}

pub fn sveScalF32Candidate(n: usize, alpha: f32, x: [*]f32) bool {
    if (comptime !features.has_sve) return false;
    callScalF32Kernel(vector_asm.sscalSveF32, n, alpha, x);
    return true;
}

pub fn sveScalRealCandidate(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime !features.has_sve) return false;
    if (T == f32) {
        callScalF32Kernel(vector_asm.sscalSveF32, n, alpha, x);
    } else if (T == f64) {
        callScalF64Kernel(vector_asm.dscalSveF64, n, alpha, x);
    } else {
        @compileError("SVE real SCAL supports f32 and f64");
    }
    return true;
}

pub fn sveScalComplexCandidate(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime !features.has_sve) return false;
    if (T == types.ComplexF32) {
        callComplexScalF32Kernel(vector_asm.cscalSveC32, n, alpha, x);
    } else if (T == types.ComplexF64) {
        callComplexScalF64Kernel(vector_asm.zscalSveC64, n, alpha, x);
    } else {
        @compileError("SVE complex SCAL supports ComplexF32 and ComplexF64");
    }
    return true;
}

pub fn scalUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime !features.has_asimd) return false;
    return fixed_simd.scalUnitComplex(T, simd_config.vectorConfig(T), n, alpha, x);
}

pub fn asumUnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !features.has_asimd and !(profile.enable_sme_asum and features.has_sme) and !(profile.enable_sve_asum and features.has_sve)) {
        return asumUnitRealDisabled(T, n, x);
    }
    if (comptime profile.enable_sve_asum and features.has_sve) {
        if (T == f32 and profile.preferSveAsum(T, n) and features.sveVectorBytes() == 64) {
            const result_bits = callAsumF32Kernel(sveSasumF32Bits, n, x);
            return @bitCast(result_bits);
        }
    }
    if (comptime profile.enable_sme_asum and features.has_sme2) {
        if (T == f32 and profile.preferSmeAsum(T, n) and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            const result_bits = callAsumF32Kernel(smeSasumF32StreamingBits, n, x);
            const stopped_result_bits = sm_state.stopSmZaRetU32(result_bits);
            return @bitCast(stopped_result_bits);
        }
        if (T == f64 and comptime features.has_sme_f64f64) {
            if (profile.preferSmeAsum(T, n) and features.streamingVectorBytes() == 64) {
                var sm_state: features.StreamingModeState = undefined;
                sm_state.startSmZa();
                const result_bits = callAsumF64Kernel(smeDasumF64StreamingBits, n, x);
                const stopped_result_bits = sm_state.stopSmZaRetU64(result_bits);
                return @bitCast(stopped_result_bits);
            }
        }
    }
    if (comptime profile.enable_sve_asum and features.has_sve) {
        if (T == f64 and profile.preferSveAsum(T, n)) {
            const result_bits = callAsumF64Kernel(vector_asm.dasumSveF64Bits, n, x);
            return @bitCast(result_bits);
        }
    }
    if (comptime features.has_asimd) return fixed_simd.asumUnitReal(T, simd_config.vectorConfig(T), n, x);
    return null;
}

pub fn asumUnitComplexComponents(comptime T: type, n: usize, x: [*]const T) ?T {
    return asumUnitReal(T, n, x);
}

pub fn sveAsumRealCandidate(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !features.has_sve) return null;
    if (T == f32) return @bitCast(callAsumF32Kernel(sveSasumF32Bits, n, x));
    if (T == f64) return @bitCast(callAsumF64Kernel(vector_asm.dasumSveF64Bits, n, x));
    @compileError("SVE ASUM supports f32 and f64 component arrays");
}

pub fn sveNrm2UnitRealCandidate(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !features.has_sve) return null;
    const scale: T = if (T == f32)
        callMaxAbsF32Kernel(vector_asm.smaxabsSveF32Bits, n, x)
    else if (T == f64)
        callMaxAbsF64Kernel(vector_asm.dmaxabsSveF64Bits, n, x)
    else
        @compileError("SVE NRM2 supports f32 and f64 component arrays");
    if (scale == 0 or std.math.isInf(scale) or std.math.isNan(scale)) return scale;
    const ssq: T = if (T == f32)
        callScaledSsqF32Kernel(vector_asm.sscaledSsqSveF32Bits, n, scale, x)
    else
        callScaledSsqF64Kernel(vector_asm.dscaledSsqSveF64Bits, n, scale, x);
    return scale * @sqrt(ssq);
}

pub fn sveIamaxUnitRealCandidate(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !features.has_sve) return null;
    if (n == 0) return 0;
    const maximum: T = if (T == f32)
        callMaxAbsF32Kernel(vector_asm.smaxabsSveF32Bits, n, x)
    else if (T == f64)
        callMaxAbsF64Kernel(vector_asm.dmaxabsSveF64Bits, n, x)
    else
        @compileError("SVE IAMAX supports f32 and f64");
    for (0..n) |i| if (@abs(x[i]) == maximum) return @intCast(i + 1);
    return 1;
}

pub fn sveIamaxUnitComplexCandidate(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !features.has_sve) return null;
    if (n == 0) return 0;
    const maximum = if (T == types.ComplexF32)
        callMaxAbsF32Kernel(vector_asm.cmaxabs1SveC32Bits, n, x)
    else if (T == types.ComplexF64)
        callMaxAbsF64Kernel(vector_asm.zmaxabs1SveC64Bits, n, x)
    else
        @compileError("SVE complex IAMAX supports ComplexF32 and ComplexF64");
    for (0..n) |i| {
        if (@abs(x[i].re) + @abs(x[i].im) == maximum) return @intCast(i + 1);
    }
    return 1;
}

pub fn nrm2UnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !features.has_asimd) return null;
    return fixed_simd.nrm2UnitReal(T, simd_config.vectorConfig(T), n, x);
}

pub fn iamaxUnitReal(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !features.has_asimd) return null;
    return fixed_simd.iamaxUnitReal(T, simd_config.vectorConfig(T), n, x);
}

pub fn iamaxUnitComplex(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !features.has_asimd) return null;
    if (!profile.preferFixedComplexIamax(n)) return null;
    return fixedIamaxUnitComplexCandidate(T, n, x);
}

pub fn fixedIamaxUnitComplexCandidate(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !features.has_asimd) return null;
    return fixed_simd.iamaxUnitComplex(T, simd_config.vectorConfig(T), n, x);
}

noinline fn sveSasumF32Bits(n: usize, x: [*]const f32) callconv(.naked) u32 {
    _ = n;
    _ = x;
    asm volatile (builders.sveRealAsumAsm("s", 8) ::: .{ .memory = true });
}

noinline fn smeSscalF32Streaming(n: usize, x: [*]f32, alpha_bits: u32) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = alpha_bits;
    asm volatile (builders.smeScalStreamingAsm("s") ::: .{ .memory = true });
}

noinline fn smeDscalF64Streaming(n: usize, x: [*]f64, alpha_bits: u64) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = alpha_bits;
    asm volatile (builders.smeScalStreamingAsm("d") ::: .{ .memory = true });
}

noinline fn smeSasumF32StreamingBits(n: usize, x: [*]const f32) callconv(.naked) u32 {
    _ = n;
    _ = x;
    asm volatile (builders.smeAsumStreamingAsm("s") ::: .{ .memory = true });
}

noinline fn smeDasumF64StreamingBits(n: usize, x: [*]const f64) callconv(.naked) u64 {
    _ = n;
    _ = x;
    asm volatile (builders.smeAsumStreamingAsm("d") ::: .{ .memory = true });
}
