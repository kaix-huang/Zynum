// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared fixed-width SIMD BLAS Level 2 real matrix-vector microkernels.

const types = @import("../../../types.zig");

const BlasInt = types.BlasInt;

pub const Config = struct {
    lane_count: comptime_int,
    row_unroll_vectors: comptime_int = 4,
    col_unroll: comptime_int = 4,
    min_work: usize = 0,
    max_work: usize = 0,
};

fn checkConfig(comptime T: type, comptime cfg: Config) void {
    if (T != f32 and T != f64) @compileError("fixed SIMD matrix-vector kernels support f32 and f64");
    if (cfg.lane_count == 0) @compileError("fixed SIMD matrix-vector lane_count must be nonzero");
    if (cfg.row_unroll_vectors == 0) @compileError("fixed SIMD matrix-vector row_unroll_vectors must be nonzero");
    if (cfg.col_unroll == 0) @compileError("fixed SIMD matrix-vector col_unroll must be nonzero");
}

fn checkComplexConfig(comptime T: type, comptime cfg: Config) void {
    if (T != types.ComplexF32 and T != types.ComplexF64) @compileError("fixed SIMD matrix-vector complex kernels support ComplexF32 and ComplexF64");
    if (cfg.lane_count == 0 or cfg.lane_count % 2 != 0) @compileError("fixed SIMD matrix-vector complex kernels need an even real lane_count");
    if (cfg.row_unroll_vectors == 0) @compileError("fixed SIMD matrix-vector row_unroll_vectors must be nonzero");
    if (cfg.col_unroll == 0) @compileError("fixed SIMD matrix-vector col_unroll must be nonzero");
}

fn Real(comptime T: type) type {
    if (T == types.ComplexF32) return f32;
    if (T == types.ComplexF64) return f64;
    @compileError("fixed SIMD matrix-vector complex kernels support ComplexF32 and ComplexF64");
}

inline fn matIndex(lda: BlasInt, row: usize, col: usize) usize {
    return row + col * @as(usize, @intCast(lda));
}

inline fn loadVec(comptime T: type, comptime lanes: comptime_int, ptr: [*]const T, index: usize) @Vector(lanes, T) {
    return @as(*align(1) const @Vector(lanes, T), @ptrCast(ptr + index)).*;
}

inline fn storeVec(comptime T: type, comptime lanes: comptime_int, ptr: [*]T, index: usize, value: @Vector(lanes, T)) void {
    @as(*align(1) @Vector(lanes, T), @ptrCast(ptr + index)).* = value;
}

inline fn asRealPtr(comptime T: type, ptr: [*]T) [*]Real(T) {
    return @ptrCast(ptr);
}

inline fn asConstRealPtr(comptime T: type, ptr: [*]const T) [*]const Real(T) {
    return @ptrCast(ptr);
}

inline fn complexAdd(comptime T: type, a: T, b: T) T {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}

inline fn complexMul(comptime T: type, a: T, b: T) T {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}

fn pairSwapMask(comptime lanes: comptime_int) @Vector(lanes, i32) {
    comptime var values: [lanes]i32 = undefined;
    inline for (0..lanes) |i| {
        values[i] = if (i % 2 == 0) @intCast(i + 1) else @intCast(i - 1);
    }
    return values;
}

fn pairSignVector(comptime T: type, comptime lanes: comptime_int, im: T) @Vector(lanes, T) {
    comptime var signs: [lanes]T = undefined;
    inline for (0..lanes) |i| {
        signs[i] = if (i % 2 == 0) -1 else 1;
    }
    return @as(@Vector(lanes, T), signs) * @as(@Vector(lanes, T), @splat(im));
}

fn pairPatternVector(comptime T: type, comptime lanes: comptime_int, comptime even: T, comptime odd: T) @Vector(lanes, T) {
    comptime var values: [lanes]T = undefined;
    inline for (0..lanes) |i| {
        values[i] = if (i % 2 == 0) even else odd;
    }
    return values;
}

fn rowUnroll(comptime cfg: Config) comptime_int {
    return cfg.lane_count * cfg.row_unroll_vectors;
}

fn tailColumnBlocks() [3]comptime_int {
    return .{ 8, 4, 2 };
}

fn driveColumnBlocks(comptime cfg: Config, n: usize, ctx: anytype) usize {
    var j: usize = 0;
    while (j + cfg.col_unroll <= n) : (j += cfg.col_unroll) {
        ctx.run(cfg.col_unroll, j);
    }
    inline for (tailColumnBlocks()) |tail_cols| {
        if (comptime tail_cols < cfg.col_unroll) {
            while (j + tail_cols <= n) : (j += tail_cols) {
                ctx.run(tail_cols, j);
            }
        }
    }
    return j;
}

fn useful(comptime cfg: Config, m: usize, n: usize) bool {
    const work = m *| n;
    if (m < cfg.lane_count or n == 0 or work < cfg.min_work) return false;
    if (cfg.max_work != 0 and work > cfg.max_work) return false;
    return true;
}

fn usefulComplex(comptime cfg: Config, m: usize, n: usize) bool {
    const work = m *| n;
    if (m < (cfg.lane_count + 1) / 2 or n == 0 or work < cfg.min_work) return false;
    if (cfg.max_work != 0 and work > cfg.max_work) return false;
    return true;
}

fn axpyColumn(comptime T: type, comptime cfg: Config, m: usize, alpha: T, x: [*]const T, y: [*]T) void {
    const V = @Vector(cfg.lane_count, T);
    const alpha_v: V = @splat(alpha);
    var i: usize = 0;
    while (i + rowUnroll(cfg) <= m) : (i += rowUnroll(cfg)) {
        inline for (0..cfg.row_unroll_vectors) |k| {
            const offset = i + k * cfg.lane_count;
            const xv = loadVec(T, cfg.lane_count, x, offset);
            const yv = loadVec(T, cfg.lane_count, y, offset);
            storeVec(T, cfg.lane_count, y, offset, @mulAdd(V, xv, alpha_v, yv));
        }
    }
    while (i + cfg.lane_count <= m) : (i += cfg.lane_count) {
        const xv = loadVec(T, cfg.lane_count, x, i);
        const yv = loadVec(T, cfg.lane_count, y, i);
        storeVec(T, cfg.lane_count, y, i, @mulAdd(V, xv, alpha_v, yv));
    }
    while (i < m) : (i += 1) y[i] = @mulAdd(T, alpha, x[i], y[i]);
}

fn scaleUnit(comptime T: type, comptime cfg: Config, n: usize, beta: T, y: [*]T) void {
    if (beta == 1) return;
    if (beta == 0) {
        @memset(y[0..n], 0);
        return;
    }

    const V = @Vector(cfg.lane_count, T);
    const beta_v: V = @splat(beta);
    var i: usize = 0;
    while (i + rowUnroll(cfg) <= n) : (i += rowUnroll(cfg)) {
        inline for (0..cfg.row_unroll_vectors) |k| {
            const offset = i + k * cfg.lane_count;
            storeVec(T, cfg.lane_count, y, offset, loadVec(T, cfg.lane_count, y, offset) * beta_v);
        }
    }
    while (i + cfg.lane_count <= n) : (i += cfg.lane_count) {
        storeVec(T, cfg.lane_count, y, i, loadVec(T, cfg.lane_count, y, i) * beta_v);
    }
    while (i < n) : (i += 1) y[i] *= beta;
}

fn gemvNoTransCols(
    comptime T: type,
    comptime cfg: Config,
    comptime cols: comptime_int,
    m: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
    j: usize,
) void {
    const V = @Vector(cfg.lane_count, T);
    var scalars: [cols]V = undefined;
    var col_ptrs: [cols][*]const T = undefined;
    inline for (0..cols) |col| {
        scalars[col] = @splat(alpha * x[j + col]);
        col_ptrs[col] = a + matIndex(lda, 0, j + col);
    }

    var i: usize = 0;
    while (i + rowUnroll(cfg) <= m) : (i += rowUnroll(cfg)) {
        inline for (0..cfg.row_unroll_vectors) |k| {
            const offset = i + k * cfg.lane_count;
            var yv = loadVec(T, cfg.lane_count, y, offset);
            inline for (0..cols) |col| {
                yv = @mulAdd(V, loadVec(T, cfg.lane_count, col_ptrs[col], offset), scalars[col], yv);
            }
            storeVec(T, cfg.lane_count, y, offset, yv);
        }
    }
    while (i + cfg.lane_count <= m) : (i += cfg.lane_count) {
        var yv = loadVec(T, cfg.lane_count, y, i);
        inline for (0..cols) |col| {
            yv = @mulAdd(V, loadVec(T, cfg.lane_count, col_ptrs[col], i), scalars[col], yv);
        }
        storeVec(T, cfg.lane_count, y, i, yv);
    }
    while (i < m) : (i += 1) {
        var yi = y[i];
        inline for (0..cols) |col| {
            yi = @mulAdd(T, col_ptrs[col][i], alpha * x[j + col], yi);
        }
        y[i] = yi;
    }
}

fn gemvNoTransUnitRealBody(
    comptime T: type,
    comptime cfg: Config,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) void {
    const Driver = struct {
        m: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        x: [*]const T,
        y: [*]T,

        fn run(self: @This(), comptime cols: comptime_int, j: usize) void {
            gemvNoTransCols(T, cfg, cols, self.m, self.alpha, self.a, self.lda, self.x, self.y, j);
        }
    };

    var j = driveColumnBlocks(cfg, n, Driver{ .m = m, .alpha = alpha, .a = a, .lda = lda, .x = x, .y = y });
    while (j < n) : (j += 1) {
        const xj = alpha * x[j];
        if (xj != 0) axpyColumn(T, cfg, m, xj, a + matIndex(lda, 0, j), y);
    }
}

pub fn gemvNoTransUnitReal(
    comptime T: type,
    comptime cfg: Config,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    comptime checkConfig(T, cfg);
    if (!useful(cfg, m, n)) return false;
    gemvNoTransUnitRealBody(T, cfg, m, n, alpha, a, lda, x, y);
    return true;
}

pub fn gemvNoTransFullUnitReal(
    comptime T: type,
    comptime cfg: Config,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    comptime checkConfig(T, cfg);
    if (!useful(cfg, m, n)) return false;
    scaleUnit(T, cfg, m, beta, y);
    gemvNoTransUnitRealBody(T, cfg, m, n, alpha, a, lda, x, y);
    return true;
}

fn gemvTransCols(
    comptime T: type,
    comptime cfg: Config,
    comptime cols: comptime_int,
    m: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
    j: usize,
) void {
    const V = @Vector(cfg.lane_count, T);
    var col_ptrs: [cols][*]const T = undefined;
    var accs: [cols][cfg.row_unroll_vectors]V = undefined;
    inline for (0..cols) |col| {
        col_ptrs[col] = a + matIndex(lda, 0, j + col);
        inline for (0..cfg.row_unroll_vectors) |u| {
            accs[col][u] = @splat(0);
        }
    }

    var i: usize = 0;
    while (i + rowUnroll(cfg) <= m) : (i += rowUnroll(cfg)) {
        inline for (0..cfg.row_unroll_vectors) |u| {
            const offset = i + u * cfg.lane_count;
            const xv = loadVec(T, cfg.lane_count, x, offset);
            inline for (0..cols) |col| {
                accs[col][u] = @mulAdd(V, loadVec(T, cfg.lane_count, col_ptrs[col], offset), xv, accs[col][u]);
            }
        }
    }

    var sums: [cols]T = undefined;
    inline for (0..cols) |col| {
        var acc: V = @splat(0);
        inline for (0..cfg.row_unroll_vectors) |u| acc += accs[col][u];
        sums[col] = @reduce(.Add, acc);
    }
    while (i + cfg.lane_count <= m) : (i += cfg.lane_count) {
        const xv = loadVec(T, cfg.lane_count, x, i);
        inline for (0..cols) |col| {
            sums[col] += @reduce(.Add, loadVec(T, cfg.lane_count, col_ptrs[col], i) * xv);
        }
    }
    while (i < m) : (i += 1) {
        inline for (0..cols) |col| {
            sums[col] = @mulAdd(T, col_ptrs[col][i], x[i], sums[col]);
        }
    }
    inline for (0..cols) |col| {
        y[j + col] = @mulAdd(T, alpha, sums[col], y[j + col]);
    }
}

fn gemvTransUnitRealBody(
    comptime T: type,
    comptime cfg: Config,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) void {
    const Driver = struct {
        m: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        x: [*]const T,
        y: [*]T,

        fn run(self: @This(), comptime cols: comptime_int, j: usize) void {
            gemvTransCols(T, cfg, cols, self.m, self.alpha, self.a, self.lda, self.x, self.y, j);
        }
    };

    var j = driveColumnBlocks(cfg, n, Driver{ .m = m, .alpha = alpha, .a = a, .lda = lda, .x = x, .y = y });
    while (j < n) : (j += 1) {
        gemvTransCols(T, cfg, 1, m, alpha, a, lda, x, y, j);
    }
}

pub fn gemvTransUnitReal(
    comptime T: type,
    comptime cfg: Config,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    comptime checkConfig(T, cfg);
    if (!useful(cfg, m, n)) return false;
    gemvTransUnitRealBody(T, cfg, m, n, alpha, a, lda, x, y);
    return true;
}

pub fn gemvTransFullUnitReal(
    comptime T: type,
    comptime cfg: Config,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    beta: T,
    y: [*]T,
) bool {
    comptime checkConfig(T, cfg);
    if (!useful(cfg, m, n)) return false;
    scaleUnit(T, cfg, n, beta, y);
    gemvTransUnitRealBody(T, cfg, m, n, alpha, a, lda, x, y);
    return true;
}

fn complexGemvNoTransCols(
    comptime T: type,
    comptime cfg: Config,
    comptime cols: comptime_int,
    m: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
    j: usize,
) void {
    const R = Real(T);
    const V = @Vector(cfg.lane_count, R);
    const swap_mask = comptime pairSwapMask(cfg.lane_count);
    const real_y = asRealPtr(T, y);
    var coeffs: [cols]T = undefined;
    var col_ptrs: [cols][*]const R = undefined;
    inline for (0..cols) |col| {
        coeffs[col] = complexMul(T, alpha, x[j + col]);
        col_ptrs[col] = asConstRealPtr(T, a + matIndex(lda, 0, j + col));
    }

    const real_m = 2 * m;
    var i: usize = 0;
    while (i + rowUnroll(cfg) <= real_m) : (i += rowUnroll(cfg)) {
        inline for (0..cfg.row_unroll_vectors) |u| {
            const offset = i + u * cfg.lane_count;
            var yv = loadVec(R, cfg.lane_count, real_y, offset);
            inline for (0..cols) |col| {
                const av = loadVec(R, cfg.lane_count, col_ptrs[col], offset);
                const swapped = @shuffle(R, av, undefined, swap_mask);
                const re_v: V = @splat(coeffs[col].re);
                const im_sign_v = pairSignVector(R, cfg.lane_count, coeffs[col].im);
                yv += @mulAdd(V, av, re_v, swapped * im_sign_v);
            }
            storeVec(R, cfg.lane_count, real_y, offset, yv);
        }
    }
    while (i + cfg.lane_count <= real_m) : (i += cfg.lane_count) {
        var yv = loadVec(R, cfg.lane_count, real_y, i);
        inline for (0..cols) |col| {
            const av = loadVec(R, cfg.lane_count, col_ptrs[col], i);
            const swapped = @shuffle(R, av, undefined, swap_mask);
            const re_v: V = @splat(coeffs[col].re);
            const im_sign_v = pairSignVector(R, cfg.lane_count, coeffs[col].im);
            yv += @mulAdd(V, av, re_v, swapped * im_sign_v);
        }
        storeVec(R, cfg.lane_count, real_y, i, yv);
    }
    while (i < real_m) : (i += 2) {
        var re = real_y[i];
        var im = real_y[i + 1];
        inline for (0..cols) |col| {
            const ar = col_ptrs[col][i];
            const ai = col_ptrs[col][i + 1];
            re += coeffs[col].re * ar - coeffs[col].im * ai;
            im += coeffs[col].re * ai + coeffs[col].im * ar;
        }
        real_y[i] = re;
        real_y[i + 1] = im;
    }
}

pub fn gemvNoTransUnitComplex(
    comptime T: type,
    comptime cfg: Config,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
) bool {
    comptime checkComplexConfig(T, cfg);
    if (!usefulComplex(cfg, m, n)) return false;

    const Driver = struct {
        m: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        x: [*]const T,
        y: [*]T,

        fn run(self: @This(), comptime cols: comptime_int, j: usize) void {
            complexGemvNoTransCols(T, cfg, cols, self.m, self.alpha, self.a, self.lda, self.x, self.y, j);
        }
    };

    var j = driveColumnBlocks(cfg, n, Driver{ .m = m, .alpha = alpha, .a = a, .lda = lda, .x = x, .y = y });
    while (j < n) : (j += 1) {
        complexGemvNoTransCols(T, cfg, 1, m, alpha, a, lda, x, y, j);
    }
    return true;
}

fn complexDotUnit(
    comptime T: type,
    comptime cfg: Config,
    m: usize,
    a_col: [*]const T,
    x: [*]const T,
    do_conj: bool,
) T {
    const R = Real(T);
    const V = @Vector(cfg.lane_count, R);
    const real_m = 2 * m;
    const real_a = asConstRealPtr(T, a_col);
    const real_x = asConstRealPtr(T, x);
    const swap_mask = comptime pairSwapMask(cfg.lane_count);
    const re_sign: V = if (do_conj) @splat(1) else pairPatternVector(R, cfg.lane_count, 1, -1);
    const im_sign: V = if (do_conj) pairPatternVector(R, cfg.lane_count, 1, -1) else @splat(1);

    var re_accs: [cfg.row_unroll_vectors]V = [_]V{@splat(0)} ** cfg.row_unroll_vectors;
    var im_accs: [cfg.row_unroll_vectors]V = [_]V{@splat(0)} ** cfg.row_unroll_vectors;
    var i: usize = 0;
    while (i + rowUnroll(cfg) <= real_m) : (i += rowUnroll(cfg)) {
        inline for (0..cfg.row_unroll_vectors) |u| {
            const offset = i + u * cfg.lane_count;
            const av = loadVec(R, cfg.lane_count, real_a, offset);
            const xv = loadVec(R, cfg.lane_count, real_x, offset);
            re_accs[u] = @mulAdd(V, av * xv, re_sign, re_accs[u]);
            im_accs[u] = @mulAdd(V, av * @shuffle(R, xv, undefined, swap_mask), im_sign, im_accs[u]);
        }
    }

    var re_acc: V = @splat(0);
    var im_acc: V = @splat(0);
    inline for (0..cfg.row_unroll_vectors) |u| {
        re_acc += re_accs[u];
        im_acc += im_accs[u];
    }
    while (i + cfg.lane_count <= real_m) : (i += cfg.lane_count) {
        const av = loadVec(R, cfg.lane_count, real_a, i);
        const xv = loadVec(R, cfg.lane_count, real_x, i);
        re_acc = @mulAdd(V, av * xv, re_sign, re_acc);
        im_acc = @mulAdd(V, av * @shuffle(R, xv, undefined, swap_mask), im_sign, im_acc);
    }

    var re_sum: R = @reduce(.Add, re_acc);
    var im_sum: R = @reduce(.Add, im_acc);
    while (i < real_m) : (i += 2) {
        const ar = real_a[i];
        const ai = real_a[i + 1];
        const xr = real_x[i];
        const xi = real_x[i + 1];
        if (do_conj) {
            re_sum = @mulAdd(R, ai, xi, @mulAdd(R, ar, xr, re_sum));
            im_sum = @mulAdd(R, -ai, xr, @mulAdd(R, ar, xi, im_sum));
        } else {
            re_sum = @mulAdd(R, -ai, xi, @mulAdd(R, ar, xr, re_sum));
            im_sum = @mulAdd(R, ai, xr, @mulAdd(R, ar, xi, im_sum));
        }
    }
    return .{ .re = re_sum, .im = im_sum };
}

fn complexGemvTransCols(
    comptime T: type,
    comptime cfg: Config,
    comptime cols: comptime_int,
    m: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
    do_conj: bool,
    j: usize,
) void {
    inline for (0..cols) |col| {
        const sum = complexDotUnit(T, cfg, m, a + matIndex(lda, 0, j + col), x, do_conj);
        y[j + col] = complexAdd(T, y[j + col], complexMul(T, alpha, sum));
    }
}

pub fn gemvTransUnitComplex(
    comptime T: type,
    comptime cfg: Config,
    m: usize,
    n: usize,
    alpha: T,
    a: [*]const T,
    lda: BlasInt,
    x: [*]const T,
    y: [*]T,
    do_conj: bool,
) bool {
    comptime checkComplexConfig(T, cfg);
    if (!usefulComplex(cfg, m, n)) return false;

    const Driver = struct {
        m: usize,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        x: [*]const T,
        y: [*]T,
        do_conj: bool,

        fn run(self: @This(), comptime cols: comptime_int, j: usize) void {
            complexGemvTransCols(T, cfg, cols, self.m, self.alpha, self.a, self.lda, self.x, self.y, self.do_conj, j);
        }
    };

    var j = driveColumnBlocks(cfg, n, Driver{ .m = m, .alpha = alpha, .a = a, .lda = lda, .x = x, .y = y, .do_conj = do_conj });
    while (j < n) : (j += 1) {
        complexGemvTransCols(T, cfg, 1, m, alpha, a, lda, x, y, do_conj, j);
    }
    return true;
}

fn gerCols(
    comptime T: type,
    comptime cfg: Config,
    comptime cols: comptime_int,
    m: usize,
    alpha: T,
    x: [*]const T,
    y: [*]const T,
    a: [*]T,
    lda: BlasInt,
    j: usize,
) void {
    const V = @Vector(cfg.lane_count, T);
    var scalars: [cols]V = undefined;
    var col_ptrs: [cols][*]T = undefined;
    inline for (0..cols) |col| {
        scalars[col] = @splat(alpha * y[j + col]);
        col_ptrs[col] = a + matIndex(lda, 0, j + col);
    }

    var i: usize = 0;
    while (i + rowUnroll(cfg) <= m) : (i += rowUnroll(cfg)) {
        inline for (0..cfg.row_unroll_vectors) |u| {
            const offset = i + u * cfg.lane_count;
            const xv = loadVec(T, cfg.lane_count, x, offset);
            inline for (0..cols) |col| {
                storeVec(T, cfg.lane_count, col_ptrs[col], offset, @mulAdd(V, xv, scalars[col], loadVec(T, cfg.lane_count, col_ptrs[col], offset)));
            }
        }
    }
    while (i + cfg.lane_count <= m) : (i += cfg.lane_count) {
        const xv = loadVec(T, cfg.lane_count, x, i);
        inline for (0..cols) |col| {
            storeVec(T, cfg.lane_count, col_ptrs[col], i, @mulAdd(V, xv, scalars[col], loadVec(T, cfg.lane_count, col_ptrs[col], i)));
        }
    }
    while (i < m) : (i += 1) {
        const xv = x[i];
        inline for (0..cols) |col| {
            col_ptrs[col][i] = @mulAdd(T, xv, alpha * y[j + col], col_ptrs[col][i]);
        }
    }
}

pub fn gerUnitReal(
    comptime T: type,
    comptime cfg: Config,
    m: usize,
    n: usize,
    alpha: T,
    x: [*]const T,
    y: [*]const T,
    a: [*]T,
    lda: BlasInt,
) bool {
    comptime checkConfig(T, cfg);
    if (!useful(cfg, m, n)) return false;

    const Driver = struct {
        m: usize,
        alpha: T,
        x: [*]const T,
        y: [*]const T,
        a: [*]T,
        lda: BlasInt,

        fn run(self: @This(), comptime cols: comptime_int, j: usize) void {
            gerCols(T, cfg, cols, self.m, self.alpha, self.x, self.y, self.a, self.lda, j);
        }
    };

    var j = driveColumnBlocks(cfg, n, Driver{ .m = m, .alpha = alpha, .x = x, .y = y, .a = a, .lda = lda });
    while (j < n) : (j += 1) {
        const temp = alpha * y[j];
        if (temp != 0) axpyColumn(T, cfg, m, temp, x, a + matIndex(lda, 0, j));
    }
    return true;
}
