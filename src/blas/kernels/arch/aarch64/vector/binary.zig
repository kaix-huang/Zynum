// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 two-vector specialized kernels.

const builders = @import("../asm/builders.zig");
const features = @import("../features.zig");
const simd_config = @import("../simd_config.zig");
const fixed_simd = @import("../../../shared/vector/fixed_simd.zig");
const types = @import("../../../../types.zig");
const vector_asm = @import("../asm/vector.zig");

const enable_mops_copy_bytes = true;
const enable_sme_copy_bytes = true;
const enable_sme_daxpy = true;
const enable_sme_ddot = true;
const enable_sve_ddot = true;
const enable_sve_zaxpy_f64 = true;

inline fn callCopyBytesKernel(comptime kernel: anytype, n_bytes: usize, x: [*]const u8, y: [*]u8) void {
    const Kernel = *const fn (usize, [*]const u8, [*]u8) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n_bytes, x, y);
}

inline fn callDotF64Kernel(comptime kernel: anytype, n: usize, x: [*]const f64, y: [*]const f64) u64 {
    const Kernel = *const fn (usize, [*]const f64, [*]const f64) callconv(.c) u64;
    return @as(Kernel, @ptrCast(&kernel))(n, x, y);
}

inline fn callDotF32Kernel(comptime kernel: anytype, n: usize, x: [*]const f32, y: [*]const f32) u32 {
    const Kernel = *const fn (usize, [*]const f32, [*]const f32) callconv(.c) u32;
    return @as(Kernel, @ptrCast(&kernel))(n, x, y);
}

inline fn callAxpyF32Kernel(comptime kernel: anytype, n: usize, alpha: f32, x: [*]const f32, y: [*]f32) void {
    const Kernel = *const fn (usize, f32, [*]const f32, [*]f32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, x, y);
}

inline fn callAxpyF64Kernel(comptime kernel: anytype, n: usize, alpha: f64, x: [*]const f64, y: [*]f64) void {
    const Kernel = *const fn (usize, f64, [*]const f64, [*]f64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, x, y);
}

inline fn callZaxpyF64Kernel(comptime kernel: anytype, n: usize, alpha_re: f64, alpha_im: f64, x: [*]const types.ComplexF64, y: [*]types.ComplexF64) void {
    const Kernel = *const fn (usize, f64, f64, [*]const types.ComplexF64, [*]types.ComplexF64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha_re, alpha_im, x, y);
}

inline fn callZdotF64Kernel(comptime kernel: anytype, n: usize, x: [*]const types.ComplexF64, y: [*]const types.ComplexF64, out: *types.ComplexF64) void {
    const Kernel = *const fn (usize, [*]const types.ComplexF64, [*]const types.ComplexF64, *types.ComplexF64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, y, out);
}

noinline fn smeDcopyBytesStreaming(n_bytes: usize, x: [*]const u8, y: [*]u8) callconv(.naked) void {
    _ = n_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.smeCopyBytesStreamingAsm() ::: .{ .memory = true });
}

inline fn mopsCopyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) void {
    @memcpy(y[0..n_bytes], x[0..n_bytes]);
}

noinline fn smeDdotF64StreamingBits(n: usize, x: [*]const f64, y: [*]const f64) callconv(.naked) u64 {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (builders.smeDotStreamingAsm("d") ::: .{ .memory = true });
}

noinline fn smeSdotF32StreamingBits(n: usize, x: [*]const f32, y: [*]const f32) callconv(.naked) u32 {
    _ = n;
    _ = x;
    _ = y;
    asm volatile (builders.smeDotStreamingAsm("s") ::: .{ .memory = true });
}

noinline fn sveZdotuF64(n: usize, x: [*]const types.ComplexF64, y: [*]const types.ComplexF64, out: *types.ComplexF64) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = out;
    asm volatile (builders.sveComplexDotAsm("d", false) ::: .{ .memory = true });
}

noinline fn sveZdotcF64(n: usize, x: [*]const types.ComplexF64, y: [*]const types.ComplexF64, out: *types.ComplexF64) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = out;
    asm volatile (builders.sveComplexDotAsm("d", true) ::: .{ .memory = true });
}

noinline fn smeSaxpyF32Streaming(n: usize, alpha: f32, x: [*]const f32, y: [*]f32) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    asm volatile (builders.smeAxpyStreamingAsm("s") ::: .{ .memory = true });
}

noinline fn sveZaxpyF64(n: usize, alpha_re: f64, alpha_im: f64, x: [*]const types.ComplexF64, y: [*]types.ComplexF64) callconv(.naked) void {
    _ = n;
    _ = alpha_re;
    _ = alpha_im;
    _ = x;
    _ = y;
    asm volatile (builders.sveComplexAxpyAsm("d") ::: .{ .memory = true });
}

noinline fn smeDaxpyF64Streaming(n: usize, alpha: f64, x: [*]const f64, y: [*]f64) callconv(.naked) void {
    _ = n;
    _ = alpha;
    _ = x;
    _ = y;
    asm volatile (builders.smeAxpyStreamingAsm("d") ::: .{ .memory = true });
}

pub fn copyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (n_bytes == 0) return true;
    if (comptime features.has_asimd) {
        if (n_bytes < 8 * 1024 * 1024) return fixed_simd.copyBytes(simd_config.byte_config, n_bytes, x, y);
    }
    if (comptime enable_mops_copy_bytes and features.has_mops) {
        if (n_bytes >= 8 * 1024) {
            mopsCopyBytes(n_bytes, x, y);
            return true;
        }
    }
    if (comptime enable_sme_copy_bytes and features.has_sme2) {
        if (n_bytes < 8 * 1024 or n_bytes >= 16 * 1024 * 1024 or features.streamingVectorBytes() != 64) return false;
        var sm_state: features.StreamingModeState = undefined;
        sm_state.startSm();
        defer sm_state.stopSm();
        callCopyBytesKernel(smeDcopyBytesStreaming, n_bytes, x, y);
        return true;
    }
    if (comptime features.has_asimd) return fixed_simd.copyBytes(simd_config.byte_config, n_bytes, x, y);
    return false;
}

pub fn copyUnit(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    return copyBytes(n * @sizeOf(T), @ptrCast(x), @ptrCast(y));
}

pub fn copyUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]T) bool {
    if (comptime T != f32 and T != f64) return false;
    return copyUnit(T, n, x, y);
}

pub fn swapUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T) bool {
    if (comptime features.has_asimd) return fixed_simd.swapUnitReal(T, simd_config.vectorConfig(T), n, x, y);
    return false;
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime enable_sme_daxpy and features.has_sme2) {
        if (T == f64 and n >= 64 * 1024 and n < (8 * 1024 * 1024) / @sizeOf(f64) and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();
            callAxpyF64Kernel(smeDaxpyF64Streaming, n, alpha, x, y);
            return true;
        }
    }
    if (comptime features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();
            callAxpyF32Kernel(smeSaxpyF32Streaming, n, alpha, x, y);
            return true;
        }
    }
    if (comptime features.has_asimd) return fixed_simd.axpyUnitReal(T, simd_config.vectorConfig(T), n, alpha, x, y);
    return false;
}

pub fn axpyUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime enable_sve_zaxpy_f64 and features.has_sve and T == types.ComplexF64) {
        if (n >= 64 * 1024 and features.sveVectorBytes() == 64) {
            callZaxpyF64Kernel(sveZaxpyF64, n, alpha.re, alpha.im, x, y);
            return true;
        }
    }
    if (comptime features.has_asimd) return fixed_simd.axpyUnitComplex(T, simd_config.vectorConfig(T), n, alpha, x, y);
    return false;
}

pub fn axpbyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime features.has_asimd) return fixed_simd.axpbyUnitReal(T, simd_config.vectorConfig(T), n, alpha, x, beta, y);
    return false;
}

pub fn axpbyUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime features.has_asimd) return fixed_simd.axpbyUnitComplex(T, simd_config.vectorConfig(T), n, alpha, x, beta, y);
    return false;
}

pub fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    if (comptime enable_sme_ddot and features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            const result_bits = callDotF32Kernel(smeSdotF32StreamingBits, n, x, y);
            const stopped_result_bits = sm_state.stopSmZaRetU32(result_bits);
            return @bitCast(stopped_result_bits);
        }
        if (T == f64 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            const result_bits = callDotF64Kernel(smeDdotF64StreamingBits, n, x, y);
            const stopped_result_bits = sm_state.stopSmZaRetU64(result_bits);
            return @bitCast(stopped_result_bits);
        }
    }
    if (comptime enable_sve_ddot and features.has_sve) {
        if (T == f64 and n >= 16) {
            const result_bits = callDotF64Kernel(vector_asm.ddotSveF64Bits, n, x, y);
            return @bitCast(result_bits);
        }
    }
    if (comptime features.has_asimd) return fixed_simd.dotUnitReal(T, simd_config.vectorConfig(T), n, x, y);
    return null;
}

pub fn dotUnitComplex(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    if (comptime features.has_sve and T == types.ComplexF64) {
        if (n >= 64) {
            var out: T = undefined;
            if (conjx) {
                callZdotF64Kernel(sveZdotcF64, n, x, y, &out);
            } else {
                callZdotF64Kernel(sveZdotuF64, n, x, y, &out);
            }
            return out;
        }
    }
    if (comptime features.has_asimd) return fixed_simd.dotUnitComplex(T, simd_config.vectorConfig(T), n, x, y, conjx);
    return null;
}

pub fn rotUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    if (comptime features.has_asimd) return fixed_simd.rotUnitReal(T, simd_config.vectorConfig(T), n, x, y, c, s);
    return false;
}
