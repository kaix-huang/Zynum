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
const enable_sme_swap_bytes = true;
const enable_sme_daxpy = true;
const enable_sme_axpby = true;
const enable_sme_linear_transform = true;
const enable_sme_ddot = true;
const enable_sve_ddot = true;
const enable_sve_zaxpy_f64 = true;
const enable_fixed_axpby = false;
const enable_fixed_rot = false;

const sme_copy_min_bytes = 8 * 1024;
const sme_copy_max_bytes = 16 * 1024 * 1024;

pub fn fixedCopyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (comptime !features.has_asimd) return false;
    return fixed_simd.copyBytes(simd_config.byte_config, n_bytes, x, y);
}

pub fn nonTemporalCopyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    _ = n_bytes;
    _ = x;
    _ = y;
    return false;
}

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

inline fn callAxpyF32Kernel(comptime kernel: anytype, n: usize, alpha_bits: u32, x: [*]const f32, y: [*]f32) void {
    const Kernel = *const fn (usize, [*]const f32, [*]f32, u32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, y, alpha_bits);
}

inline fn callAxpyF64Kernel(comptime kernel: anytype, n: usize, alpha_bits: u64, x: [*]const f64, y: [*]f64) void {
    const Kernel = *const fn (usize, [*]const f64, [*]f64, u64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, y, alpha_bits);
}

inline fn callAxpbyF32Kernel(comptime kernel: anytype, n: usize, alpha_bits: u32, beta_bits: u32, x: [*]const f32, y: [*]f32) void {
    const Kernel = *const fn (usize, [*]const f32, [*]f32, u32, u32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, y, alpha_bits, beta_bits);
}

inline fn callLinearTransformF32Kernel(
    comptime kernel: anytype,
    n: usize,
    x: [*]f32,
    y: [*]f32,
    a_bits: u32,
    b_bits: u32,
    c_bits: u32,
    d_bits: u32,
) void {
    const Kernel = *const fn (usize, [*]f32, [*]f32, u32, u32, u32, u32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, y, a_bits, b_bits, c_bits, d_bits);
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

noinline fn smeCopy8KiBStreaming(n_bytes: usize, x: [*]const u8, y: [*]u8) callconv(.naked) void {
    _ = n_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.smeCopy8KiBStreamingAsm() ::: .{ .memory = true });
}

noinline fn smeSwapBytesStreaming(n_bytes: usize, x: [*]u8, y: [*]u8) callconv(.naked) void {
    _ = n_bytes;
    _ = x;
    _ = y;
    asm volatile (builders.smeSwapBytesStreamingAsm() ::: .{ .memory = true });
}

noinline fn asimdSwapBytes(n_bytes: usize, x: [*]u8, y: [*]u8) callconv(.naked) void {
    _ = n_bytes;
    _ = x;
    _ = y;
    asm volatile (
        \\
        \\lsr x5, x0, #7
        \\cbz x5, 1f
        \\0:
        \\ld1 { v0.4s, v1.4s }, [x1]
        \\ld1 { v2.4s, v3.4s }, [x2]
        \\st1 { v0.4s, v1.4s }, [x2], #32
        \\st1 { v2.4s, v3.4s }, [x1], #32
        \\ld1 { v0.4s, v1.4s }, [x1]
        \\ld1 { v2.4s, v3.4s }, [x2]
        \\st1 { v0.4s, v1.4s }, [x2], #32
        \\st1 { v2.4s, v3.4s }, [x1], #32
        \\ld1 { v0.4s, v1.4s }, [x1]
        \\ld1 { v2.4s, v3.4s }, [x2]
        \\st1 { v0.4s, v1.4s }, [x2], #32
        \\st1 { v2.4s, v3.4s }, [x1], #32
        \\ld1 { v0.4s, v1.4s }, [x1]
        \\ld1 { v2.4s, v3.4s }, [x2]
        \\st1 { v0.4s, v1.4s }, [x2], #32
        \\st1 { v2.4s, v3.4s }, [x1], #32
        \\subs x5, x5, #1
        \\b.ne 0b
        \\1:
        \\and x0, x0, #127
        \\cmp x0, #64
        \\b.lo 2f
        \\ld1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x1]
        \\ld1 { v16.16b, v17.16b, v18.16b, v19.16b }, [x2]
        \\st1 { v16.16b, v17.16b, v18.16b, v19.16b }, [x1], #64
        \\st1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x2], #64
        \\sub x0, x0, #64
        \\2:
        \\cmp x0, #32
        \\b.lo 3f
        \\ld1 { v0.16b, v1.16b }, [x1]
        \\ld1 { v16.16b, v17.16b }, [x2]
        \\st1 { v16.16b, v17.16b }, [x1], #32
        \\st1 { v0.16b, v1.16b }, [x2], #32
        \\sub x0, x0, #32
        \\3:
        \\cmp x0, #16
        \\b.lo 4f
        \\ldr q0, [x1]
        \\ldr q16, [x2]
        \\str q16, [x1], #16
        \\str q0, [x2], #16
        \\sub x0, x0, #16
        \\4:
        \\cmp x0, #8
        \\b.lo 5f
        \\ldr d0, [x1]
        \\ldr d16, [x2]
        \\str d16, [x1], #8
        \\str d0, [x2], #8
        \\sub x0, x0, #8
        \\5:
        \\cmp x0, #4
        \\b.lo 6f
        \\ldr s0, [x1]
        \\ldr s16, [x2]
        \\str s16, [x1]
        \\str s0, [x2]
        \\6:
        \\ret
        ::: .{ .memory = true });
}

inline fn callSwapBytesKernel(comptime kernel: anytype, n_bytes: usize, x: [*]u8, y: [*]u8) void {
    const Kernel = *const fn (usize, [*]u8, [*]u8) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n_bytes, x, y);
}

noinline fn copyBytesSmeStreaming(n_bytes: usize, x: [*]const u8, y: [*]u8) void {
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSm();
    defer sm_state.stopSm();
    callCopyBytesKernel(smeDcopyBytesStreaming, n_bytes, x, y);
}

noinline fn copy8KiBSmeStreaming(x: [*]const u8, y: [*]u8) void {
    var sm_state: features.StreamingModeState = undefined;
    sm_state.startSm();
    defer sm_state.stopSm();
    callCopyBytesKernel(smeCopy8KiBStreaming, 8 * 1024, x, y);
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

noinline fn smeSaxpyF32Streaming(n: usize, x: [*]const f32, y: [*]f32, alpha_bits: u32) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = alpha_bits;
    asm volatile (builders.smeAxpyStreamingAsm("s") ::: .{ .memory = true });
}

noinline fn smeSaxpbyF32Streaming(n: usize, x: [*]const f32, y: [*]f32, alpha_bits: u32, beta_bits: u32) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = alpha_bits;
    _ = beta_bits;
    asm volatile (builders.smeAxpbyStreamingAsm("s") ::: .{ .memory = true });
}

noinline fn smeSlinearTransformF32Streaming(
    n: usize,
    x: [*]f32,
    y: [*]f32,
    a_bits: u32,
    b_bits: u32,
    c_bits: u32,
    d_bits: u32,
) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = a_bits;
    _ = b_bits;
    _ = c_bits;
    _ = d_bits;
    asm volatile (builders.smeLinearTransformStreamingAsm("s") ::: .{ .memory = true });
}

noinline fn sveZaxpyF64(n: usize, alpha_re: f64, alpha_im: f64, x: [*]const types.ComplexF64, y: [*]types.ComplexF64) callconv(.naked) void {
    _ = n;
    _ = alpha_re;
    _ = alpha_im;
    _ = x;
    _ = y;
    asm volatile (builders.sveComplexAxpyAsm("d") ::: .{ .memory = true });
}

noinline fn smeDaxpyF64Streaming(n: usize, x: [*]const f64, y: [*]f64, alpha_bits: u64) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = alpha_bits;
    asm volatile (builders.smeAxpyStreamingAsm("d") ::: .{ .memory = true });
}

pub fn copyBytes(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (n_bytes == 0) return true;
    if (comptime enable_sme_copy_bytes and features.has_sme2) {
        if (n_bytes == 8 * 1024 and features.streamingVectorBytes() == 64) {
            copy8KiBSmeStreaming(x, y);
            return true;
        }
    }
    if (comptime features.has_asimd) {
        if (n_bytes < 8 * 1024) return fixed_simd.copyBytes(simd_config.byte_config, n_bytes, x, y);
    }
    if (comptime enable_mops_copy_bytes and features.has_mops) {
        if (n_bytes >= 8 * 1024) {
            mopsCopyBytes(n_bytes, x, y);
            return true;
        }
    }
    if (comptime enable_sme_copy_bytes and features.has_sme2) {
        if (features.streamingVectorBytes() != 64 or n_bytes < sme_copy_min_bytes or n_bytes >= sme_copy_max_bytes) return false;
        copyBytesSmeStreaming(n_bytes, x, y);
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

pub fn swapUnitRealStreaming(comptime T: type, n: usize, x: [*]T, y: [*]T) bool {
    if (comptime enable_sme_swap_bytes and features.has_sme2) {
        const n_bytes = n * @sizeOf(T);
        if (n_bytes >= 64 * 1024 and n_bytes <= 8 * 1024 * 1024 and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSm();
            defer sm_state.stopSm();
            callSwapBytesKernel(smeSwapBytesStreaming, n_bytes, @ptrCast(x), @ptrCast(y));
            return true;
        }
    }
    return false;
}

pub fn swapUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T) bool {
    if (comptime features.has_asimd) {
        const n_bytes = n * @sizeOf(T);
        if (n_bytes >= 128) {
            callSwapBytesKernel(asimdSwapBytes, n_bytes, @ptrCast(x), @ptrCast(y));
            return true;
        }
    }
    if (comptime features.has_asimd) return fixed_simd.swapUnitReal(T, simd_config.vectorConfig(T), n, x, y);
    return false;
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime enable_sme_daxpy and features.has_sme2) {
        if (T == f64 and n >= 64 * 1024 and n < (8 * 1024 * 1024) / @sizeOf(f64) and features.streamingVectorBytes() == 64) {
            const alpha_bits: u64 = @bitCast(alpha);
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();
            callAxpyF64Kernel(smeDaxpyF64Streaming, n, alpha_bits, x, y);
            return true;
        }
    }
    if (comptime features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            const alpha_bits: u32 = @bitCast(alpha);
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();
            callAxpyF32Kernel(smeSaxpyF32Streaming, n, alpha_bits, x, y);
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
    if (comptime enable_sme_axpby and features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            const alpha_bits: u32 = @bitCast(alpha);
            const beta_bits: u32 = @bitCast(beta);
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();
            callAxpbyF32Kernel(smeSaxpbyF32Streaming, n, alpha_bits, beta_bits, x, y);
            return true;
        }
    }
    if (comptime enable_fixed_axpby and features.has_asimd) return fixed_simd.axpbyUnitReal(T, simd_config.vectorConfig(T), n, alpha, x, beta, y);
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

pub fn rotUnitRealStreaming(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    if (comptime enable_sme_linear_transform and features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and n * @sizeOf(T) <= 4 * 1024 * 1024 and features.streamingVectorBytes() == 64) {
            const a_bits: u32 = @bitCast(c);
            const b_bits: u32 = @bitCast(s);
            const c_bits: u32 = @bitCast(-s);
            const d_bits: u32 = a_bits;
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();
            callLinearTransformF32Kernel(smeSlinearTransformF32Streaming, n, x, y, a_bits, b_bits, c_bits, d_bits);
            return true;
        }
    }
    return false;
}

pub fn rotUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    if (comptime enable_fixed_rot and features.has_asimd) return fixed_simd.rotUnitReal(T, simd_config.vectorConfig(T), n, x, y, c, s);
    return false;
}

pub fn rotmUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    if (comptime enable_sme_linear_transform and features.has_sme2) {
        if (T == f32 and n >= 64 * 1024 and features.streamingVectorBytes() == 64) {
            var a: T = undefined;
            var b: T = undefined;
            var c: T = undefined;
            var d: T = undefined;
            if (flag < 0) {
                a = h11;
                b = h12;
                c = h21;
                d = h22;
            } else if (flag == 0) {
                a = 1;
                b = h12;
                c = h21;
                d = 1;
            } else {
                a = h11;
                b = 1;
                c = -1;
                d = h22;
            }
            const a_bits: u32 = @bitCast(a);
            const b_bits: u32 = @bitCast(b);
            const c_bits: u32 = @bitCast(c);
            const d_bits: u32 = @bitCast(d);
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();
            callLinearTransformF32Kernel(smeSlinearTransformF32Streaming, n, x, y, a_bits, b_bits, c_bits, d_bits);
            return true;
        }
    }
    return false;
}
