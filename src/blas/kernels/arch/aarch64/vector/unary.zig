// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 single-vector specialized kernels.

const builders = @import("../asm/builders.zig");
const features = @import("../features.zig");
const simd_config = @import("../simd_config.zig");
const fixed_simd = @import("../../../shared/vector/fixed_simd.zig");
const types = @import("../../../../types.zig");
const vector_asm = @import("../asm/vector.zig");

const enable_asimd_dscal = false;
const enable_sme_dscal = true;
const enable_sve_dscal = true;
const enable_sme_dasum = true;
const enable_sve_dasum = true;

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
    if (comptime !features.has_asimd and !(enable_sme_dscal and features.has_sme2) and !(enable_sve_dscal and features.has_sve)) {
        return scalUnitRealDisabled(T, n, alpha, x);
    }
    if (comptime enable_asimd_dscal and features.has_asimd) {
        if (T == f64 and n >= 16) {
            asimdDscalF64(n, alpha, x);
            return true;
        }
    }
    if (comptime enable_sme_dscal and features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            const alpha_bits: u32 = @bitCast(alpha);
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();

            callSmScalF32Kernel(smeSscalF32Streaming, n, alpha_bits, x);
            return true;
        }
        if (T == f64 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            const alpha_bits: u64 = @bitCast(alpha);
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();

            callSmScalF64Kernel(smeDscalF64Streaming, n, alpha_bits, x);
            return true;
        }
    }
    if (comptime enable_sve_dscal and features.has_sve) {
        if (T == f64 and n >= 16) {
            callScalF64Kernel(vector_asm.dscalSveF64, n, alpha, x);
            return true;
        }
    }
    if (comptime features.has_asimd) return fixed_simd.scalUnitReal(T, simd_config.vectorConfig(T), n, alpha, x);
    return false;
}

pub fn scalUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]T) bool {
    if (comptime !features.has_asimd) return false;
    return fixed_simd.scalUnitComplex(T, simd_config.vectorConfig(T), n, alpha, x);
}

pub fn asumUnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !features.has_asimd and !(enable_sme_dasum and features.has_sme) and !(enable_sve_dasum and features.has_sve)) {
        return asumUnitRealDisabled(T, n, x);
    }
    if (comptime enable_sve_dasum and features.has_sve) {
        if (T == f32 and n >= 64 * 1024 and features.sveVectorBytes() == 64) {
            const result_bits = callAsumF32Kernel(sveSasumF32Bits, n, x);
            return @bitCast(result_bits);
        }
    }
    if (comptime enable_sme_dasum and features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            const result_bits = callAsumF32Kernel(smeSasumF32StreamingBits, n, x);
            const stopped_result_bits = sm_state.stopSmZaRetU32(result_bits);
            return @bitCast(stopped_result_bits);
        }
        if (T == f64 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            const result_bits = callAsumF64Kernel(smeDasumF64StreamingBits, n, x);
            const stopped_result_bits = sm_state.stopSmZaRetU64(result_bits);
            return @bitCast(stopped_result_bits);
        }
    }
    if (comptime enable_sve_dasum and features.has_sve) {
        if (T == f64 and n >= 16) {
            const result_bits = callAsumF64Kernel(vector_asm.dasumSveF64Bits, n, x);
            return @bitCast(result_bits);
        }
    }
    if (comptime features.has_asimd) return fixed_simd.asumUnitReal(T, simd_config.vectorConfig(T), n, x);
    return null;
}

pub fn nrm2UnitReal(comptime T: type, n: usize, x: [*]const T) ?T {
    if (comptime !features.has_asimd) return null;
    return fixed_simd.nrm2UnitReal(T, simd_config.vectorConfig(T), n, x);
}

pub fn iamaxUnitReal(comptime T: type, n: usize, x: [*]const T) ?types.BlasInt {
    if (comptime !features.has_asimd) return null;
    return fixed_simd.iamaxUnitReal(T, simd_config.vectorConfig(T), n, x);
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
