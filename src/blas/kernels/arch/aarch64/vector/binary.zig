// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! AArch64 two-vector specialized kernels.

const builders = @import("../asm/builders.zig");
const features = @import("../features.zig");
const simd_config = @import("../simd_config.zig");
const fixed_simd = @import("../../../shared/vector/fixed_simd.zig");
const tuning = @import("../../../shared/vector/tuning.zig");
const types = @import("../../../../types.zig");
const vector_asm = @import("../asm/vector.zig");

const profile = tuning.active.aarch64;

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

inline fn callDotF32AccF64Kernel(comptime kernel: anytype, n: usize, x: [*]const f32, y: [*]const f32) u64 {
    const Kernel = *const fn (usize, [*]const f32, [*]const f32) callconv(.c) u64;
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

inline fn callCaxpyF32Kernel(comptime kernel: anytype, n: usize, alpha_re: f32, alpha_im: f32, x: [*]const types.ComplexF32, y: [*]types.ComplexF32) void {
    const Kernel = *const fn (usize, f32, f32, [*]const types.ComplexF32, [*]types.ComplexF32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha_re, alpha_im, x, y);
}

inline fn callZdotF64Kernel(comptime kernel: anytype, n: usize, x: [*]const types.ComplexF64, y: [*]const types.ComplexF64, out: *types.ComplexF64) void {
    const Kernel = *const fn (usize, [*]const types.ComplexF64, [*]const types.ComplexF64, *types.ComplexF64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, y, out);
}

inline fn callCdotF32Kernel(comptime kernel: anytype, n: usize, x: [*]const types.ComplexF32, y: [*]const types.ComplexF32, out: *types.ComplexF32) void {
    const Kernel = *const fn (usize, [*]const types.ComplexF32, [*]const types.ComplexF32, *types.ComplexF32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, y, out);
}

inline fn callRealAxpyF32Kernel(comptime kernel: anytype, n: usize, alpha: f32, x: [*]const f32, y: [*]f32) void {
    const Kernel = *const fn (usize, f32, [*]const f32, [*]f32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, x, y);
}

inline fn callRealAxpyF64Kernel(comptime kernel: anytype, n: usize, alpha: f64, x: [*]const f64, y: [*]f64) void {
    const Kernel = *const fn (usize, f64, [*]const f64, [*]f64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, x, y);
}

inline fn callRealAxpbyF32Kernel(comptime kernel: anytype, n: usize, alpha: f32, beta: f32, x: [*]const f32, y: [*]f32) void {
    const Kernel = *const fn (usize, f32, f32, [*]const f32, [*]f32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, beta, x, y);
}

inline fn callRealAxpbyF64Kernel(comptime kernel: anytype, n: usize, alpha: f64, beta: f64, x: [*]const f64, y: [*]f64) void {
    const Kernel = *const fn (usize, f64, f64, [*]const f64, [*]f64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha, beta, x, y);
}

inline fn callLinearF32Kernel(comptime kernel: anytype, n: usize, x: [*]f32, y: [*]f32, a: f32, b: f32, c: f32, d: f32) void {
    const Kernel = *const fn (usize, [*]f32, [*]f32, f32, f32, f32, f32) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, y, a, b, c, d);
}

inline fn callLinearF64Kernel(comptime kernel: anytype, n: usize, x: [*]f64, y: [*]f64, a: f64, b: f64, c: f64, d: f64) void {
    const Kernel = *const fn (usize, [*]f64, [*]f64, f64, f64, f64, f64) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, x, y, a, b, c, d);
}

inline fn callComplexAxpbyF32Kernel(comptime kernel: anytype, n: usize, alpha: types.ComplexF32, beta: types.ComplexF32, x: [*]const types.ComplexF32, y: [*]types.ComplexF32) void {
    const Kernel = *const fn (usize, f32, f32, f32, f32, [*]const u8, [*]u8) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha.re, alpha.im, beta.re, beta.im, @ptrCast(x), @ptrCast(y));
}

inline fn callComplexAxpbyF64Kernel(comptime kernel: anytype, n: usize, alpha: types.ComplexF64, beta: types.ComplexF64, x: [*]const types.ComplexF64, y: [*]types.ComplexF64) void {
    const Kernel = *const fn (usize, f64, f64, f64, f64, [*]const u8, [*]u8) callconv(.c) void;
    @as(Kernel, @ptrCast(&kernel))(n, alpha.re, alpha.im, beta.re, beta.im, @ptrCast(x), @ptrCast(y));
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
    callCopyBytesKernel(smeCopy8KiBStreaming, profile.sme_copy_exact_bytes, x, y);
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

noinline fn sveCdotuF32(n: usize, x: [*]const types.ComplexF32, y: [*]const types.ComplexF32, out: *types.ComplexF32) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = out;
    asm volatile (builders.sveComplexDotAsm("s", false) ::: .{ .memory = true });
}

noinline fn sveCdotcF32(n: usize, x: [*]const types.ComplexF32, y: [*]const types.ComplexF32, out: *types.ComplexF32) callconv(.naked) void {
    _ = n;
    _ = x;
    _ = y;
    _ = out;
    asm volatile (builders.sveComplexDotAsm("s", true) ::: .{ .memory = true });
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

noinline fn sveCaxpyF32(n: usize, alpha_re: f32, alpha_im: f32, x: [*]const types.ComplexF32, y: [*]types.ComplexF32) callconv(.naked) void {
    _ = n;
    _ = alpha_re;
    _ = alpha_im;
    _ = x;
    _ = y;
    asm volatile (builders.sveComplexAxpyAsm("s") ::: .{ .memory = true });
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
    if (comptime profile.enable_sme_copy and features.has_sme2) {
        if (profile.preferSmeCopyExact(n_bytes) and features.streamingVectorBytes() == 64) {
            copy8KiBSmeStreaming(x, y);
            return true;
        }
    }
    if (comptime features.has_asimd) {
        if (profile.preferAsimdCopy(n_bytes)) return fixed_simd.copyBytes(simd_config.byte_config, n_bytes, x, y);
    }
    if (comptime profile.enable_mops_copy and features.has_mops) {
        if (profile.preferMopsCopy(n_bytes)) {
            mopsCopyBytes(n_bytes, x, y);
            return true;
        }
    }
    if (comptime profile.enable_sme_copy and features.has_sme2) {
        if (features.streamingVectorBytes() != 64 or !profile.preferSmeCopy(n_bytes)) return false;
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

pub fn sveCopyBytesCandidate(n_bytes: usize, x: [*]const u8, y: [*]u8) bool {
    if (comptime !features.has_sve) return false;
    callCopyBytesKernel(vector_asm.copySveBytes, n_bytes, x, y);
    return true;
}

pub fn sveSwapBytesCandidate(n_bytes: usize, x: [*]u8, y: [*]u8) bool {
    if (comptime !features.has_sve) return false;
    callSwapBytesKernel(vector_asm.swapSveBytes, n_bytes, x, y);
    return true;
}

pub fn swapUnitRealStreaming(comptime T: type, n: usize, x: [*]T, y: [*]T) bool {
    if (comptime profile.enable_sme_swap and features.has_sme2) {
        const n_bytes = n * @sizeOf(T);
        if (profile.preferSmeSwap(n_bytes) and features.streamingVectorBytes() == 64) {
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
        if (profile.preferAsimdSwap(n_bytes)) {
            callSwapBytesKernel(asimdSwapBytes, n_bytes, @ptrCast(x), @ptrCast(y));
            return true;
        }
    }
    if (comptime features.has_asimd) return fixed_simd.swapUnitReal(T, simd_config.vectorConfig(T), n, x, y);
    return false;
}

pub fn axpyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime profile.enable_sme_axpy and features.has_sme2) {
        if (T == f64 and comptime features.has_sme_f64f64) {
            if (profile.preferSmeAxpy(T, n) and features.streamingVectorBytes() == 64) {
                const alpha_bits: u64 = @bitCast(alpha);
                var sm_state: features.StreamingModeState = undefined;
                sm_state.startSmZa();
                defer sm_state.stopSmZa();
                callAxpyF64Kernel(smeDaxpyF64Streaming, n, alpha_bits, x, y);
                return true;
            }
        }
    }
    if (comptime features.has_sme2) {
        if (T == f32 and profile.preferSmeAxpy(T, n) and features.streamingVectorBytes() == 64) {
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

pub fn sveAxpyRealCandidate(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime !features.has_sve) return false;
    if (T == f32) {
        callRealAxpyF32Kernel(vector_asm.saxpySveF32, n, alpha, x, y);
    } else if (T == f64) {
        callRealAxpyF64Kernel(vector_asm.daxpySveF64, n, alpha, x, y);
    } else {
        @compileError("SVE real AXPY supports f32 and f64");
    }
    return true;
}

pub fn axpyUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]const T, y: [*]T) bool {
    if (comptime profile.enable_sve_axpy_complex_f32 and features.has_sve and T == types.ComplexF32) {
        if (profile.preferSveAxpyComplexF32(n)) return sveAxpyComplexF32Candidate(n, alpha, x, y);
    }
    if (comptime profile.enable_sve_axpy_complex_f64 and features.has_sve and T == types.ComplexF64) {
        if (profile.preferSveAxpyComplexF64(n) and features.sveVectorBytes() == 64) {
            callZaxpyF64Kernel(sveZaxpyF64, n, alpha.re, alpha.im, x, y);
            return true;
        }
    }
    if (comptime features.has_asimd) return fixed_simd.axpyUnitComplex(T, simd_config.vectorConfig(T), n, alpha, x, y);
    return false;
}

pub fn sveAxpyComplexF32Candidate(n: usize, alpha: types.ComplexF32, x: [*]const types.ComplexF32, y: [*]types.ComplexF32) bool {
    if (comptime !features.has_sve) return false;
    callCaxpyF32Kernel(sveCaxpyF32, n, alpha.re, alpha.im, x, y);
    return true;
}

pub fn axpbyUnitReal(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime profile.enable_sme_axpby and features.has_sme2) {
        if (T == f32 and profile.preferSmeAxpby(T, n) and features.streamingVectorBytes() == 64) {
            const alpha_bits: u32 = @bitCast(alpha);
            const beta_bits: u32 = @bitCast(beta);
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            defer sm_state.stopSmZa();
            callAxpbyF32Kernel(smeSaxpbyF32Streaming, n, alpha_bits, beta_bits, x, y);
            return true;
        }
    }
    if (comptime profile.enable_fixed_axpby and features.has_asimd) return fixed_simd.axpbyUnitReal(T, simd_config.vectorConfig(T), n, alpha, x, beta, y);
    return false;
}

pub fn axpbyUnitComplex(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime features.has_asimd) return fixed_simd.axpbyUnitComplex(T, simd_config.vectorConfig(T), n, alpha, x, beta, y);
    return false;
}

pub fn sveAxpbyRealCandidate(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime !features.has_sve) return false;
    if (T == f32) {
        callRealAxpbyF32Kernel(vector_asm.saxpbySveF32, n, alpha, beta, x, y);
    } else if (T == f64) {
        callRealAxpbyF64Kernel(vector_asm.daxpbySveF64, n, alpha, beta, x, y);
    } else {
        @compileError("SVE real AXPBY supports f32 and f64");
    }
    return true;
}

pub fn sveAxpbyComplexCandidate(comptime T: type, n: usize, alpha: T, x: [*]const T, beta: T, y: [*]T) bool {
    if (comptime !features.has_sve) return false;
    if (T == types.ComplexF32) {
        callComplexAxpbyF32Kernel(vector_asm.caxpbySveC32, n, alpha, beta, x, y);
    } else if (T == types.ComplexF64) {
        callComplexAxpbyF64Kernel(vector_asm.zaxpbySveC64, n, alpha, beta, x, y);
    } else {
        @compileError("SVE complex AXPBY supports ComplexF32 and ComplexF64");
    }
    return true;
}

pub fn dotUnitReal(comptime T: type, n: usize, x: [*]const T, y: [*]const T) ?T {
    if (comptime profile.enable_sme_dot and features.has_sme2) {
        if (T == f32 and profile.preferSmeDot(T, n) and features.streamingVectorBytes() == 64) {
            var sm_state: features.StreamingModeState = undefined;
            sm_state.startSmZa();
            const result_bits = callDotF32Kernel(smeSdotF32StreamingBits, n, x, y);
            const stopped_result_bits = sm_state.stopSmZaRetU32(result_bits);
            return @bitCast(stopped_result_bits);
        }
        if (T == f64 and comptime features.has_sme_f64f64) {
            if (profile.preferSmeDot(T, n) and features.streamingVectorBytes() == 64) {
                var sm_state: features.StreamingModeState = undefined;
                sm_state.startSmZa();
                const result_bits = callDotF64Kernel(smeDdotF64StreamingBits, n, x, y);
                const stopped_result_bits = sm_state.stopSmZaRetU64(result_bits);
                return @bitCast(stopped_result_bits);
            }
        }
    }
    if (comptime (profile.enable_sve_dot_f32 or profile.enable_sve_dot) and features.has_sve) {
        if (T == f32 and profile.preferSveDot(T, n)) return sveDotF32Candidate(n, x, y);
        if (T == f64 and profile.preferSveDot(T, n)) {
            const result_bits = callDotF64Kernel(vector_asm.ddotSveF64Bits, n, x, y);
            return @bitCast(result_bits);
        }
    }
    if (comptime features.has_asimd) return fixed_simd.dotUnitReal(T, simd_config.vectorConfig(T), n, x, y);
    return null;
}

pub fn sveDotF32Candidate(n: usize, x: [*]const f32, y: [*]const f32) ?f32 {
    if (comptime !features.has_sve) return null;
    return @bitCast(callDotF32Kernel(vector_asm.sdotSveF32Bits, n, x, y));
}

pub fn sveDotF32AccF64Candidate(n: usize, x: [*]const f32, y: [*]const f32) ?f64 {
    if (comptime !features.has_sve) return null;
    return @bitCast(callDotF32AccF64Kernel(vector_asm.sdotSveF32AccF64Bits, n, x, y));
}

pub fn dotF32AccF64Unit(n: usize, x: [*]const f32, y: [*]const f32) ?f64 {
    if (comptime !profile.enable_fixed_dot_f32_acc_f64 or !features.has_asimd) return null;
    return fixedDotF32AccF64UnitCandidate(n, x, y);
}

pub fn fixedDotF32AccF64UnitCandidate(n: usize, x: [*]const f32, y: [*]const f32) ?f64 {
    if (comptime !features.has_asimd) return null;
    return fixed_simd.dotF32AccF64Unit(simd_config.vectorConfig(f64), n, x, y);
}

pub fn dotUnitComplex(comptime T: type, n: usize, x: [*]const T, y: [*]const T, conjx: bool) ?T {
    if (comptime profile.enable_sve_dot_complex_f32 and features.has_sve and T == types.ComplexF32) {
        if (profile.preferSveDotComplexF32(n)) return sveDotComplexF32Candidate(n, x, y, conjx);
    }
    if (comptime profile.enable_sve_dot_complex_f64 and features.has_sve and T == types.ComplexF64) {
        if (profile.preferSveDotComplexF64(n)) {
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

pub fn sveDotComplexF32Candidate(n: usize, x: [*]const types.ComplexF32, y: [*]const types.ComplexF32, conjx: bool) ?types.ComplexF32 {
    if (comptime !features.has_sve) return null;
    var out: types.ComplexF32 = undefined;
    if (conjx) {
        callCdotF32Kernel(sveCdotcF32, n, x, y, &out);
    } else {
        callCdotF32Kernel(sveCdotuF32, n, x, y, &out);
    }
    return out;
}

pub fn rotUnitRealStreaming(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    if (comptime profile.enable_sme_linear_transform and features.has_sme2) {
        if (profile.preferSmeRot(T, n) and features.streamingVectorBytes() == 64) {
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
    if (comptime profile.enable_fixed_rot and features.has_asimd) return fixed_simd.rotUnitReal(T, simd_config.vectorConfig(T), n, x, y, c, s);
    return false;
}

pub fn sveRotUnitRealCandidate(comptime T: type, n: usize, x: [*]T, y: [*]T, c: T, s: T) bool {
    if (comptime !features.has_sve) return false;
    if (T == f32) {
        callLinearF32Kernel(vector_asm.slinearSveF32, n, x, y, c, s, -s, c);
    } else if (T == f64) {
        callLinearF64Kernel(vector_asm.dlinearSveF64, n, x, y, c, s, -s, c);
    } else {
        @compileError("SVE ROT supports f32 and f64 component arrays");
    }
    return true;
}

pub fn rotmUnitReal(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    if (comptime profile.enable_sme_linear_transform and features.has_sme2) {
        if (T == f32 and profile.preferSmeRotm(T, n) and features.streamingVectorBytes() == 64) {
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
    if (comptime profile.enable_fixed_rotm and features.has_asimd) {
        return fixedRotmUnitRealCandidate(T, n, x, y, flag, h11, h21, h12, h22);
    }
    return false;
}

pub fn fixedRotmUnitRealCandidate(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    if (comptime !features.has_asimd) return false;
    return fixed_simd.rotmUnitReal(T, simd_config.vectorConfig(T), n, x, y, flag, h11, h21, h12, h22);
}

pub fn sveRotmUnitRealCandidate(comptime T: type, n: usize, x: [*]T, y: [*]T, flag: T, h11: T, h21: T, h12: T, h22: T) bool {
    if (comptime !features.has_sve) return false;
    const coefficients: [4]T = if (flag < 0)
        .{ h11, h12, h21, h22 }
    else if (flag == 0)
        .{ 1, h12, h21, 1 }
    else if (flag == 1)
        .{ h11, 1, -1, h22 }
    else
        .{ 1, 0, 0, 1 };
    if (T == f32) {
        callLinearF32Kernel(vector_asm.slinearSveF32, n, x, y, coefficients[0], coefficients[1], coefficients[2], coefficients[3]);
    } else if (T == f64) {
        callLinearF64Kernel(vector_asm.dlinearSveF64, n, x, y, coefficients[0], coefficients[1], coefficients[2], coefficients[3]);
    } else {
        @compileError("SVE ROTM supports f32 and f64");
    }
    return true;
}
