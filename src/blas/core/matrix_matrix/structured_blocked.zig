// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Experimental native-block structured Level 3 algorithms.
//!
//! Only active structured panels and private output tiles are materialized.
//! Workspace acquisition completes before caller output is changed, so failure
//! can fall back as a whole operation.

const std = @import("std");

const scalar = @import("../shared/scalar.zig");
const indexing = @import("../shared/indexing.zig");
const matrix_vector_ops = @import("../matrix_vector.zig");
const gemm_impl = @import("gemm.zig");
const packing = @import("../../kernels/shared/matrix_matrix/structured_packing.zig");

pub const BlasInt = scalar.BlasInt;
pub const Order = scalar.Order;
pub const Uplo = scalar.Uplo;
pub const Diag = scalar.Diag;
pub const Side = scalar.Side;

pub const Options = struct {
    block_size: usize = 64,
    workspace_available: bool = true,
};

const toUsize = indexing.toUsize;
const matIndex = indexing.matIndex;

fn packTriangle(uplo: Uplo) packing.Triangle {
    return if (uplo == .upper) .upper else .lower;
}

fn unpackTriangle(triangle: packing.Triangle) Uplo {
    return if (triangle == .upper) .upper else .lower;
}

fn packTranspose(transpose: Order) packing.Transpose {
    return switch (transpose) {
        .no_trans => .no_trans,
        .trans => .trans,
        .conj_trans => .conj_trans,
    };
}

fn packDiagonal(diagonal: Diag) packing.Diagonal {
    return if (diagonal == .unit) .unit else .non_unit;
}

fn validOptions(options: Options) bool {
    return options.block_size > 0 and options.block_size <= 256;
}

fn workspaceElements(options: Options, buffer_count: usize) ?usize {
    if (!validOptions(options) or !options.workspace_available) return null;
    const block_elements = std.math.mul(usize, options.block_size, options.block_size) catch return null;
    return std.math.mul(usize, block_elements, buffer_count) catch null;
}

fn acquireWorkspace(comptime T: type, options: Options, buffer_count: usize) ?[]T {
    const len = workspaceElements(options, buffer_count) orelse return null;
    return std.heap.c_allocator.alloc(T, len) catch null;
}

fn scaleGeneral(comptime T: type, m: usize, n: usize, beta: T, c: [*]T, ldc: BlasInt) void {
    for (0..n) |j| {
        for (0..m) |i| {
            const index = matIndex(ldc, i, j);
            c[index] = if (scalar.isZero(T, beta)) scalar.zero(T) else scalar.mul(T, beta, c[index]);
        }
    }
}

fn scaleStoredTriangle(comptime T: type, uplo: Uplo, n: usize, beta: T, c: [*]T, ldc: BlasInt, hermitian: bool) void {
    for (0..n) |j| {
        const row0: usize = if (uplo == .upper) 0 else j;
        const row1: usize = if (uplo == .upper) j + 1 else n;
        for (row0..row1) |i| {
            const index = matIndex(ldc, i, j);
            c[index] = if (scalar.isZero(T, beta)) scalar.zero(T) else scalar.mul(T, beta, c[index]);
            if (hermitian and i == j) {
                if (comptime scalar.isComplex(T)) c[index].im = 0;
            }
        }
    }
}

fn zeroGeneral(comptime T: type, m: usize, n: usize, b: [*]T, ldb: BlasInt) void {
    scaleGeneral(T, m, n, scalar.zero(T), b, ldb);
}

fn blockCount(size: usize, block_size: usize) usize {
    return (size + block_size - 1) / block_size;
}

fn blockExtent(size: usize, block_size: usize, block_index: usize) struct { start: usize, len: usize } {
    const start = block_index * block_size;
    return .{ .start = start, .len = @min(block_size, size - start) };
}

fn tryFullSymm(comptime T: type, options: Options, side: Side, uplo: Uplo, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, hermitian: bool) bool {
    const order = if (side == .left) m else n;
    if (builtin.cpu.arch != .x86_64 or @import("../../runtime.zig").maxThreads() <= 1 or order < 256 or order > 1024) return false;
    if (!validOptions(options) or !options.workspace_available) return false;
    const workspace = std.heap.c_allocator.alloc(T, order * order) catch return false;
    defer std.heap.c_allocator.free(workspace);
    const Context = struct {
        order: usize,
        uplo: Uplo,
        hermitian: bool,
        a: [*]const T,
        lda: BlasInt,
        output: []T,
        fn execute(raw: *const anyopaque, index: usize) void {
            const x: *const @This() = @ptrCast(@alignCast(raw));
            const col = index * 32;
            const cols = @min(@as(usize, 32), x.order - col);
            packing.packSymmetricBlock(T, packTriangle(x.uplo), x.hermitian, x.a, x.lda, 0, col, x.order, cols, x.output[col * x.order .. (col + cols) * x.order]);
        }
    };
    const context = Context{ .order = order, .uplo = uplo, .hermitian = hermitian, .a = a, .lda = lda, .output = workspace };
    const jobs = blockCount(order, 32);
    if (!@import("../execution/thread_pool.zig").runQueued(Context.execute, &context, jobs, 16)) {
        for (0..jobs) |index| Context.execute(&context, index);
    }
    const Product = struct {
        m: usize,
        n: usize,
        order: usize,
        side: Side,
        alpha: T,
        beta: T,
        structured: [*]const T,
        b: [*]const T,
        ldb: BlasInt,
        c: [*]T,
        ldc: BlasInt,
        fn execute(raw: *const anyopaque, index: usize) void {
            const x: *const @This() = @ptrCast(@alignCast(raw));
            const row_blocks = blockCount(x.m, 64);
            const row = index % row_blocks * 64;
            const col = index / row_blocks * 64;
            const rows = @min(@as(usize, 64), x.m - row);
            const cols = @min(@as(usize, 64), x.n - col);
            const ld: BlasInt = @intCast(x.order);
            gemm_impl.gemm(T, .no_trans, .no_trans, @intCast(rows), @intCast(cols), ld, x.alpha, if (x.side == .left) x.structured + row else x.b + row, if (x.side == .left) ld else x.ldb, if (x.side == .left) x.b + matIndex(x.ldb, 0, col) else x.structured + col * x.order, if (x.side == .left) x.ldb else ld, x.beta, x.c + matIndex(x.ldc, row, col), x.ldc);
        }
    };
    const product = Product{ .m = m, .n = n, .order = order, .side = side, .alpha = alpha, .beta = beta, .structured = workspace.ptr, .b = b, .ldb = ldb, .c = c, .ldc = ldc };
    if (@import("../execution/thread_pool.zig").runQueued(Product.execute, &product, blockCount(m, 64) * blockCount(n, 64), 16)) return true;
    gemm_impl.gemm(T, .no_trans, .no_trans, @intCast(m), @intCast(n), @intCast(order), alpha, if (side == .left) workspace.ptr else b, if (side == .left) @intCast(order) else ldb, if (side == .left) b else workspace.ptr, if (side == .left) ldb else @intCast(order), beta, c, ldc);
    return true;
}

/// Computes SYMM/HEMM by packing one logical block of the structured operand
/// and immediately consuming it through the stable GEMM path.
pub fn trySymm(comptime T: type, options: Options, side: Side, uplo: Uplo, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, hermitian: bool) bool {
    if (m_ <= 0 or n_ <= 0) return true;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (scalar.isZero(T, alpha)) {
        scaleGeneral(T, m, n, beta, c, ldc);
        return true;
    }
    if (tryFullSymm(T, options, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc, hermitian)) return true;
    const workspace = acquireWorkspace(T, options, 1) orelse return false;
    defer std.heap.c_allocator.free(workspace);
    const bs = options.block_size;
    const order = if (side == .left) m else n;

    if (side == .left) {
        var i_base: usize = 0;
        while (i_base < m) : (i_base += bs) {
            const ib = @min(bs, m - i_base);
            var p0: usize = 0;
            while (p0 < order) : (p0 += bs) {
                const pb = @min(bs, order - p0);
                const panel = workspace[0 .. ib * pb];
                packing.packSymmetricBlock(T, packTriangle(uplo), hermitian, a, lda, i_base, p0, ib, pb, panel);
                gemm_impl.gemm(T, .no_trans, .no_trans, @intCast(ib), n_, @intCast(pb), alpha, panel.ptr, @intCast(ib), b + p0, ldb, if (p0 == 0) beta else scalar.one(T), c + i_base, ldc);
            }
        }
    } else {
        var j0: usize = 0;
        while (j0 < n) : (j0 += bs) {
            const jb = @min(bs, n - j0);
            var p0: usize = 0;
            while (p0 < order) : (p0 += bs) {
                const pb = @min(bs, order - p0);
                const panel = workspace[0 .. pb * jb];
                packing.packSymmetricBlock(T, packTriangle(uplo), hermitian, a, lda, p0, j0, pb, jb, panel);
                gemm_impl.gemm(T, .no_trans, .no_trans, m_, @intCast(jb), @intCast(pb), alpha, b + matIndex(ldb, 0, p0), ldb, panel.ptr, @intCast(pb), if (p0 == 0) beta else scalar.one(T), c + matIndex(ldc, 0, j0), ldc);
            }
        }
    }
    return true;
}

fn rankOperandBase(comptime T: type, trans: Order, matrix: [*]const T, ld: BlasInt, output_offset: usize) [*]const T {
    return if (trans == .no_trans) matrix + output_offset else matrix + matIndex(ld, 0, output_offset);
}

fn rankProduct(comptime T: type, trans: Order, hermitian: bool, rows: usize, cols: usize, k_: BlasInt, alpha: T, left: [*]const T, ld_left: BlasInt, right: [*]const T, ld_right: BlasInt, beta: T, tile: [*]T, ld_tile: BlasInt) void {
    if (trans == .no_trans) {
        gemm_impl.gemm(T, .no_trans, if (hermitian) .conj_trans else .trans, @intCast(rows), @intCast(cols), k_, alpha, left, ld_left, right, ld_right, beta, tile, ld_tile);
    } else {
        gemm_impl.gemm(T, if (hermitian) .conj_trans else .trans, .no_trans, @intCast(rows), @intCast(cols), k_, alpha, left, ld_left, right, ld_right, beta, tile, ld_tile);
    }
}

fn commitRankTile(comptime T: type, uplo: Uplo, hermitian: bool, row0: usize, col0: usize, rows: usize, cols: usize, beta: T, tile: []const T, c: [*]T, ldc: BlasInt) void {
    for (0..cols) |j| {
        const global_col = col0 + j;
        for (0..rows) |i| {
            const global_row = row0 + i;
            const stored = if (uplo == .upper) global_row <= global_col else global_row >= global_col;
            if (!stored) continue;
            const c_index = matIndex(ldc, global_row, global_col);
            c[c_index] = scalar.add(T, tile[i + j * rows], if (scalar.isZero(T, beta)) scalar.zero(T) else scalar.mul(T, beta, c[c_index]));
            if (hermitian and global_row == global_col) {
                if (comptime scalar.isComplex(T)) c[c_index].im = 0;
            }
        }
    }
}

fn rankDiagonal(comptime T: type, comptime rank_two: bool, uplo: Uplo, trans: Order, hermitian: bool, start: usize, size: usize, k: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, workspace: []T) void {
    const second_alpha = if (hermitian) scalar.conj(T, alpha) else alpha;
    // Complex 3M needs larger tiles to amortize conversion. Only real,
    // single-thread diagonal updates benefited from smaller subtiles.
    if (scalar.isComplex(T) or @import("../../runtime.zig").maxThreads() > 1) {
        const da = rankOperandBase(T, trans, a, lda, start);
        const db = rankOperandBase(T, trans, b, ldb, start);
        const tile = workspace[0 .. size * size];
        rankProduct(T, trans, hermitian, size, size, k, alpha, da, lda, db, ldb, scalar.zero(T), tile.ptr, @intCast(size));
        if (rank_two) rankProduct(T, trans, hermitian, size, size, k, second_alpha, db, ldb, da, lda, scalar.one(T), tile.ptr, @intCast(size));
        commitRankTile(T, uplo, hermitian, start, start, size, size, beta, tile, c, ldc);
        return;
    }
    const leaf = @min(size, @as(usize, 32));
    var d = start;
    while (d < start + size) : (d += leaf) {
        const width = @min(leaf, start + size - d);
        const first = if (uplo == .upper) start else d + width;
        const rows = if (uplo == .upper) d - start else start + size - first;
        const da = rankOperandBase(T, trans, a, lda, d);
        const db = rankOperandBase(T, trans, b, ldb, d);
        if (rows > 0) {
            const output = c + matIndex(ldc, first, d);
            rankProduct(T, trans, hermitian, rows, width, k, alpha, rankOperandBase(T, trans, a, lda, first), lda, db, ldb, beta, output, ldc);
            if (rank_two) rankProduct(T, trans, hermitian, rows, width, k, second_alpha, rankOperandBase(T, trans, b, ldb, first), ldb, da, lda, scalar.one(T), output, ldc);
        }
        const tile = workspace[0 .. width * width];
        rankProduct(T, trans, hermitian, width, width, k, alpha, da, lda, db, ldb, scalar.zero(T), tile.ptr, @intCast(width));
        if (rank_two) rankProduct(T, trans, hermitian, width, width, k, second_alpha, db, ldb, da, lda, scalar.one(T), tile.ptr, @intCast(width));
        commitRankTile(T, uplo, hermitian, d, d, width, width, beta, tile, c, ldc);
    }
}

fn tryQueuedRank(comptime T: type, comptime rank_two: bool, options: Options, uplo: Uplo, trans: Order, n: usize, k: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, hermitian: bool) bool {
    if (builtin.cpu.arch != .x86_64 or n < 256 or k < 64 or @import("../../runtime.zig").maxThreads() <= 1) return false;
    if (!validOptions(options) or !options.workspace_available) return false;
    const bs: usize = if (n < 384) 32 else 64;
    const nb = blockCount(n, bs);
    const length = std.math.mul(usize, nb, bs * bs) catch return false;
    const workspace = std.heap.c_allocator.alloc(T, length) catch return false;
    defer std.heap.c_allocator.free(workspace);
    const Context = struct {
        n: usize,
        nb: usize,
        bs: usize,
        k: BlasInt,
        uplo: Uplo,
        trans: Order,
        hermitian: bool,
        alpha: T,
        beta: T,
        a: [*]const T,
        lda: BlasInt,
        b: [*]const T,
        ldb: BlasInt,
        c: [*]T,
        ldc: BlasInt,
        workspace: []T,
        fn execute(raw: *const anyopaque, index: usize) void {
            const x: *const @This() = @ptrCast(@alignCast(raw));
            const bi = index % x.nb;
            const bj = index / x.nb;
            if (if (x.uplo == .upper) bi > bj else bi < bj) return;
            const row = bi * x.bs;
            const col = bj * x.bs;
            const rows = @min(x.bs, x.n - row);
            const cols = @min(x.bs, x.n - col);
            const diagonal = bi == bj;
            const output = if (diagonal) x.workspace.ptr + bi * x.bs * x.bs else x.c + matIndex(x.ldc, row, col);
            const ld: BlasInt = if (diagonal) @intCast(rows) else x.ldc;
            rankProduct(T, x.trans, x.hermitian, rows, cols, x.k, x.alpha, rankOperandBase(T, x.trans, x.a, x.lda, row), x.lda, rankOperandBase(T, x.trans, x.b, x.ldb, col), x.ldb, if (diagonal) scalar.zero(T) else x.beta, output, ld);
            if (rank_two) rankProduct(T, x.trans, x.hermitian, rows, cols, x.k, if (x.hermitian) scalar.conj(T, x.alpha) else x.alpha, rankOperandBase(T, x.trans, x.b, x.ldb, row), x.ldb, rankOperandBase(T, x.trans, x.a, x.lda, col), x.lda, scalar.one(T), output, ld);
            if (diagonal) commitRankTile(T, x.uplo, x.hermitian, row, col, rows, cols, x.beta, output[0 .. rows * cols], x.c, x.ldc);
        }
    };
    const context = Context{ .n = n, .nb = nb, .bs = bs, .k = k, .uplo = uplo, .trans = trans, .hermitian = hermitian, .alpha = alpha, .beta = beta, .a = a, .lda = lda, .b = b, .ldb = ldb, .c = c, .ldc = ldc, .workspace = workspace };
    const jobs = std.math.mul(usize, nb, nb) catch return false;
    return @import("../execution/thread_pool.zig").runQueued(Context.execute, &context, jobs, @min(@as(usize, 16), nb * (nb + 1) / 2));
}

/// Computes SYRK/HERK into private block tiles and commits only the requested
/// triangle, preserving the unstored half bitwise.
pub fn trySyrk(comptime T: type, options: Options, uplo: Uplo, trans: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, beta: T, c: [*]T, ldc: BlasInt, hermitian: bool) bool {
    if (n_ <= 0) return true;
    const n = toUsize(n_);
    if (k_ <= 0 or scalar.isZero(T, alpha)) {
        scaleStoredTriangle(T, uplo, n, beta, c, ldc, hermitian);
        return true;
    }
    if (comptime builtin.cpu.arch == .x86_64 and (T == f32 or T == f64)) {
        if (trans != .no_trans and n >= 128 and k_ >= 64 and (@import("../../runtime.zig").maxThreads() == 1 or n < 384) and options.workspace_available and validOptions(options)) {
            const count = std.math.mul(usize, n, toUsize(k_)) catch return false;
            const converted = std.heap.c_allocator.alloc(T, count) catch return false;
            defer std.heap.c_allocator.free(converted);
            // Convert once for all rank panels instead of repacking transposed
            // A inside every small GEMM. The recursive no-trans path cannot
            // re-enter this conversion and preserves the unstored triangle.
            @import("../../kernels/shared/matrix_matrix/packing.zig").packTransposedA(T, n, toUsize(k_), a, lda, converted);
            return trySyrk(T, options, uplo, .no_trans, n_, k_, alpha, converted.ptr, n_, beta, c, ldc, hermitian);
        }
    }
    if (tryQueuedRank(T, false, options, uplo, trans, n, k_, alpha, a, lda, a, lda, beta, c, ldc, hermitian)) return true;
    const workspace = acquireWorkspace(T, options, 1) orelse return false;
    defer std.heap.c_allocator.free(workspace);
    const bs = options.block_size;

    var j0: usize = 0;
    while (j0 < n) : (j0 += bs) {
        const jb = @min(bs, n - j0);
        if (comptime builtin.cpu.arch == .x86_64) {
            // A whole off-diagonal column panel lies inside the stored
            // triangle. Update it directly, reusing packing across rows.
            const first = if (uplo == .upper) 0 else j0 + jb;
            const rows = if (uplo == .upper) j0 else n - first;
            if (rows > 0) rankProduct(T, trans, hermitian, rows, jb, k_, alpha, rankOperandBase(T, trans, a, lda, first), lda, rankOperandBase(T, trans, a, lda, j0), lda, beta, c + matIndex(ldc, first, j0), ldc);
            rankDiagonal(T, false, uplo, trans, hermitian, j0, jb, k_, alpha, a, lda, a, lda, beta, c, ldc, workspace);
            continue;
        }
        var i_base: usize = if (uplo == .upper) 0 else j0;
        const i_end = if (uplo == .upper) j0 + jb else n;
        while (i_base < i_end) : (i_base += bs) {
            const ib = @min(bs, n - i_base);
            const tile = workspace[0 .. ib * jb];
            rankProduct(T, trans, hermitian, ib, jb, k_, alpha, rankOperandBase(T, trans, a, lda, i_base), lda, rankOperandBase(T, trans, a, lda, j0), lda, scalar.zero(T), tile.ptr, @intCast(ib));
            commitRankTile(T, uplo, hermitian, i_base, j0, ib, jb, beta, tile, c, ldc);
        }
    }
    return true;
}

/// Computes SYR2K/HER2K with two GEMM updates per private output tile.
pub fn trySyr2k(comptime T: type, options: Options, uplo: Uplo, trans: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]const T, ldb: BlasInt, beta: T, c: [*]T, ldc: BlasInt, hermitian: bool) bool {
    if (n_ <= 0) return true;
    const n = toUsize(n_);
    if (k_ <= 0 or scalar.isZero(T, alpha)) {
        scaleStoredTriangle(T, uplo, n, beta, c, ldc, hermitian);
        return true;
    }
    if (comptime builtin.cpu.arch == .x86_64 and (T == f32 or T == f64)) {
        if (trans != .no_trans and n >= 128 and k_ >= 64 and (@import("../../runtime.zig").maxThreads() == 1 or n < 384) and options.workspace_available and validOptions(options)) {
            const count = std.math.mul(usize, n, toUsize(k_)) catch return false;
            const total = std.math.mul(usize, count, 2) catch return false;
            const converted = std.heap.c_allocator.alloc(T, total) catch return false;
            defer std.heap.c_allocator.free(converted);
            const general_packing = @import("../../kernels/shared/matrix_matrix/packing.zig");
            general_packing.packTransposedA(T, n, toUsize(k_), a, lda, converted[0..count]);
            general_packing.packTransposedA(T, n, toUsize(k_), b, ldb, converted[count..]);
            return trySyr2k(T, options, uplo, .no_trans, n_, k_, alpha, converted.ptr, n_, converted.ptr + count, n_, beta, c, ldc, hermitian);
        }
    }
    if (tryQueuedRank(T, true, options, uplo, trans, n, k_, alpha, a, lda, b, ldb, beta, c, ldc, hermitian)) return true;
    const workspace = acquireWorkspace(T, options, 1) orelse return false;
    defer std.heap.c_allocator.free(workspace);
    const bs = options.block_size;
    const second_alpha = if (hermitian) scalar.conj(T, alpha) else alpha;

    var j0: usize = 0;
    while (j0 < n) : (j0 += bs) {
        const jb = @min(bs, n - j0);
        if (comptime builtin.cpu.arch == .x86_64) {
            const first = if (uplo == .upper) 0 else j0 + jb;
            const rows = if (uplo == .upper) j0 else n - first;
            if (rows > 0) {
                const output = c + matIndex(ldc, first, j0);
                rankProduct(T, trans, hermitian, rows, jb, k_, alpha, rankOperandBase(T, trans, a, lda, first), lda, rankOperandBase(T, trans, b, ldb, j0), ldb, beta, output, ldc);
                rankProduct(T, trans, hermitian, rows, jb, k_, second_alpha, rankOperandBase(T, trans, b, ldb, first), ldb, rankOperandBase(T, trans, a, lda, j0), lda, scalar.one(T), output, ldc);
            }
            rankDiagonal(T, true, uplo, trans, hermitian, j0, jb, k_, alpha, a, lda, b, ldb, beta, c, ldc, workspace);
            continue;
        }
        var i_base: usize = if (uplo == .upper) 0 else j0;
        const i_end = if (uplo == .upper) j0 + jb else n;
        while (i_base < i_end) : (i_base += bs) {
            const ib = @min(bs, n - i_base);
            const tile = workspace[0 .. ib * jb];
            rankProduct(T, trans, hermitian, ib, jb, k_, alpha, rankOperandBase(T, trans, a, lda, i_base), lda, rankOperandBase(T, trans, b, ldb, j0), ldb, scalar.zero(T), tile.ptr, @intCast(ib));
            rankProduct(T, trans, hermitian, ib, jb, k_, second_alpha, rankOperandBase(T, trans, b, ldb, i_base), ldb, rankOperandBase(T, trans, a, lda, j0), lda, scalar.one(T), tile.ptr, @intCast(ib));
            commitRankTile(T, uplo, hermitian, i_base, j0, ib, jb, beta, tile, c, ldc);
        }
    }
    return true;
}

fn packTriangularPanel(comptime T: type, uplo: Uplo, trans: Order, diag: Diag, a: [*]const T, lda: BlasInt, row0: usize, col0: usize, rows: usize, cols: usize, panel: []T) void {
    packing.packTriangularOpBlock(T, packTriangle(uplo), packTranspose(trans), packDiagonal(diag), a, lda, row0, col0, rows, cols, panel);
}

fn storeScaledTile(comptime T: type, rows: usize, cols: usize, alpha: T, tile: []const T, b: [*]T, ldb: BlasInt, row0: usize, col0: usize) void {
    for (0..cols) |j| {
        for (0..rows) |i| b[matIndex(ldb, row0 + i, col0 + j)] = scalar.mul(T, alpha, tile[i + j * rows]);
    }
}

/// Blocked TRMM. Effective triangular panels are packed, multiplied into a
/// private output tile, then committed in dependency-preserving block order.
fn tryQueuedTriangular(comptime T: type, comptime solve: bool, options: Options, side: Side, uplo: Uplo, trans: Order, diag: Diag, m: usize, n: usize, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) bool {
    if (builtin.cpu.arch != .x86_64 or m < 256 or n < 256 or @import("../../runtime.zig").maxThreads() <= 1) return false;
    if (!validOptions(options) or !options.workspace_available) return false;
    const bs = 64;
    const rhs_width = 32;
    const jobs = blockCount(if (side == .left) n else m, rhs_width);
    const per_job = bs * bs * (if (solve) @as(usize, 1) else 2);
    const length = std.math.mul(usize, jobs, per_job) catch return false;
    const workspace = std.heap.c_allocator.alloc(T, length) catch return false;
    defer std.heap.c_allocator.free(workspace);
    const Context = struct {
        m: usize,
        n: usize,
        side: Side,
        uplo: Uplo,
        trans: Order,
        diag: Diag,
        alpha: T,
        a: [*]const T,
        lda: BlasInt,
        b: [*]T,
        ldb: BlasInt,
        workspace: []T,
        fn execute(raw: *const anyopaque, index: usize) void {
            const x: *const @This() = @ptrCast(@alignCast(raw));
            const first = index * rhs_width;
            const rows = if (x.side == .left) x.m else @min(rhs_width, x.m - first);
            const cols = if (x.side == .left) @min(rhs_width, x.n - first) else x.n;
            const output = x.b + if (x.side == .left) matIndex(x.ldb, 0, first) else first;
            const scratch = x.workspace[index * per_job .. (index + 1) * per_job];
            const operation = if (solve) trsmWorkspace else trmmWorkspace;
            _ = operation(T, bs, x.side, x.uplo, x.trans, x.diag, @intCast(rows), @intCast(cols), x.alpha, x.a, x.lda, output, x.ldb, scratch);
        }
    };
    const context = Context{ .m = m, .n = n, .side = side, .uplo = uplo, .trans = trans, .diag = diag, .alpha = alpha, .a = a, .lda = lda, .b = b, .ldb = ldb, .workspace = workspace };
    return @import("../execution/thread_pool.zig").runQueued(Context.execute, &context, jobs, 16);
}

pub fn tryTrmm(comptime T: type, options: Options, side: Side, uplo: Uplo, trans: Order, diag: Diag, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) bool {
    if (m_ <= 0 or n_ <= 0) return true;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (scalar.isZero(T, alpha)) {
        zeroGeneral(T, m, n, b, ldb);
        return true;
    }
    if (tryQueuedTriangular(T, false, options, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)) return true;
    const workspace = acquireWorkspace(T, options, 2) orelse return false;
    defer std.heap.c_allocator.free(workspace);
    return trmmWorkspace(T, options.block_size, side, uplo, trans, diag, m_, n_, alpha, a, lda, b, ldb, workspace);
}

fn trmmWorkspace(comptime T: type, bs: usize, side: Side, uplo: Uplo, trans: Order, diag: Diag, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt, workspace: []T) bool {
    const m = toUsize(m_);
    const n = toUsize(n_);
    const block_elements = bs * bs;
    const packed_panel = workspace[0..block_elements];
    const output = workspace[block_elements .. 2 * block_elements];
    const effective = packing.effectiveTriangle(packTriangle(uplo), packTranspose(trans));

    if (builtin.cpu.arch == .x86_64 and @import("../../runtime.zig").maxThreads() > 1) {
        const dimension = if (side == .left) m else n;
        const rhs = if (side == .left) n else m;
        const blocks = blockCount(dimension, bs);
        var rhs0: usize = 0;
        while (rhs0 < rhs) : (rhs0 += bs) {
            const width = @min(bs, rhs - rhs0);
            for (0..blocks) |step| {
                const forward = if (side == .left) effective == .upper else effective == .lower;
                const block = blockExtent(dimension, bs, if (forward) step else blocks - 1 - step);
                const after = if (side == .left) effective == .upper else effective == .lower;
                const first = if (after) block.start + block.len else 0;
                const count = if (after) dimension - first else block.start;
                const rows = if (side == .left) block.len else width;
                const cols = if (side == .left) width else block.len;
                if (count > 0) {
                    if (side == .left) {
                        const panel = a + if (trans == .no_trans) matIndex(lda, block.start, first) else matIndex(lda, first, block.start);
                        gemm_impl.gemm(T, trans, .no_trans, @intCast(rows), @intCast(cols), @intCast(count), scalar.one(T), panel, lda, b + matIndex(ldb, first, rhs0), ldb, scalar.zero(T), output.ptr, @intCast(rows));
                    } else {
                        const panel = a + if (trans == .no_trans) matIndex(lda, first, block.start) else matIndex(lda, block.start, first);
                        gemm_impl.gemm(T, .no_trans, trans, @intCast(rows), @intCast(cols), @intCast(count), scalar.one(T), b + matIndex(ldb, rhs0, first), ldb, panel, lda, scalar.zero(T), output.ptr, @intCast(rows));
                    }
                }
                const diagonal = packed_panel[0 .. block.len * block.len];
                packTriangularPanel(T, uplo, trans, diag, a, lda, block.start, block.start, block.len, block.len, diagonal);
                const row0 = if (side == .left) block.start else rhs0;
                const col0 = if (side == .left) rhs0 else block.start;
                const rhs_panel = b + matIndex(ldb, row0, col0);
                gemm_impl.gemm(T, .no_trans, .no_trans, @intCast(rows), @intCast(cols), @intCast(block.len), scalar.one(T), if (side == .left) diagonal.ptr else rhs_panel, if (side == .left) @intCast(block.len) else ldb, if (side == .left) rhs_panel else diagonal.ptr, if (side == .left) ldb else @intCast(block.len), if (count == 0) scalar.zero(T) else scalar.one(T), output.ptr, @intCast(rows));
                storeScaledTile(T, rows, cols, alpha, output, b, ldb, row0, col0);
            }
        }
        return true;
    }

    if (side == .left) {
        const blocks = blockCount(m, bs);
        var col0: usize = 0;
        while (col0 < n) : (col0 += bs) {
            const cols = @min(bs, n - col0);
            for (0..blocks) |step| {
                const bi = if (effective == .upper) step else blocks - 1 - step;
                const row_block = blockExtent(m, bs, bi);
                var first = true;
                const p_first: usize = if (effective == .upper) bi else 0;
                const p_end: usize = if (effective == .upper) blocks else bi + 1;
                for (p_first..p_end) |pi| {
                    const p_block = blockExtent(m, bs, pi);
                    const panel = packed_panel[0 .. row_block.len * p_block.len];
                    packTriangularPanel(T, uplo, trans, diag, a, lda, row_block.start, p_block.start, row_block.len, p_block.len, panel);
                    gemm_impl.gemm(T, .no_trans, .no_trans, @intCast(row_block.len), @intCast(cols), @intCast(p_block.len), scalar.one(T), panel.ptr, @intCast(row_block.len), b + matIndex(ldb, p_block.start, col0), ldb, if (first) scalar.zero(T) else scalar.one(T), output.ptr, @intCast(row_block.len));
                    first = false;
                }
                storeScaledTile(T, row_block.len, cols, alpha, output, b, ldb, row_block.start, col0);
            }
        }
    } else {
        const blocks = blockCount(n, bs);
        var row0: usize = 0;
        while (row0 < m) : (row0 += bs) {
            const rows = @min(bs, m - row0);
            for (0..blocks) |step| {
                const bj = if (effective == .upper) blocks - 1 - step else step;
                const col_block = blockExtent(n, bs, bj);
                var first = true;
                const p_first: usize = if (effective == .upper) 0 else bj;
                const p_end: usize = if (effective == .upper) bj + 1 else blocks;
                for (p_first..p_end) |pi| {
                    const p_block = blockExtent(n, bs, pi);
                    const panel = packed_panel[0 .. p_block.len * col_block.len];
                    packTriangularPanel(T, uplo, trans, diag, a, lda, p_block.start, col_block.start, p_block.len, col_block.len, panel);
                    gemm_impl.gemm(T, .no_trans, .no_trans, @intCast(rows), @intCast(col_block.len), @intCast(p_block.len), scalar.one(T), b + matIndex(ldb, row0, p_block.start), ldb, panel.ptr, @intCast(p_block.len), if (first) scalar.zero(T) else scalar.one(T), output.ptr, @intCast(rows));
                    first = false;
                }
                storeScaledTile(T, rows, col_block.len, alpha, output, b, ldb, row0, col_block.start);
            }
        }
    }
    return true;
}

const builtin = @import("builtin");

fn updateSolvePanel(comptime T: type, comptime columns: usize, count: usize, x: [*]const T, coeff: [columns]T, dest: [columns][*]T) void {
    const R = scalar.Real(T);
    const width = 32 / @sizeOf(R);
    const V = @Vector(width, R);
    const components = if (comptime scalar.isComplex(T)) 2 else 1;
    const xp: [*]const R = @ptrCast(x);
    var i: usize = 0;
    while (i + width <= count * components) : (i += width) {
        const xv = @as(*align(1) const V, @ptrCast(xp + i)).*;
        inline for (0..columns) |col| {
            const dp: [*]R = @ptrCast(dest[col]);
            var yv = @as(*align(1) const V, @ptrCast(dp + i)).*;
            if (comptime scalar.isComplex(T)) {
                const mask: @Vector(width, i32) = comptime blk: {
                    var result: @Vector(width, i32) = undefined;
                    for (0..width) |lane| result[lane] = @intCast(lane ^ 1);
                    break :blk result;
                };
                var imag: V = @splat(coeff[col].im);
                inline for (0..width / 2) |pair| imag[2 * pair] = -coeff[col].im;
                yv = @mulAdd(V, xv, @as(V, @splat(coeff[col].re)), yv);
                yv = @mulAdd(V, @shuffle(R, xv, undefined, mask), imag, yv);
            } else {
                yv = @mulAdd(V, xv, @as(V, @splat(coeff[col])), yv);
            }
            @as(*align(1) V, @ptrCast(dp + i)).* = yv;
        }
    }
    var row = i / components;
    while (row < count) : (row += 1) {
        inline for (0..columns) |col| {
            dest[col][row] = if (comptime scalar.isComplex(T)) scalar.add(T, dest[col][row], scalar.mul(T, coeff[col], x[row])) else @mulAdd(T, coeff[col], x[row], dest[col][row]);
        }
    }
}

fn solveLeftPanel(comptime T: type, comptime columns: usize, triangle: packing.Triangle, rows: usize, diagonal: []const T, reciprocal: []const T, b: [*]T, ldb: BlasInt, row0: usize, col0: usize) void {
    for (0..rows) |step| {
        const p = if (triangle == .upper) rows - 1 - step else step;
        const first = if (triangle == .upper) 0 else p + 1;
        const count = if (triangle == .upper) p else rows - first;
        var coeff: [columns]T = undefined;
        var dest: [columns][*]T = undefined;
        inline for (0..columns) |col| {
            const index = matIndex(ldb, row0 + p, col0 + col);
            b[index] = scalar.mul(T, b[index], reciprocal[p]);
            coeff[col] = scalar.neg(T, b[index]);
            dest[col] = b + matIndex(ldb, row0 + first, col0 + col);
        }
        updateSolvePanel(T, columns, count, diagonal.ptr + p * rows + first, coeff, dest);
    }
}

fn updateRightPanel(comptime T: type, comptime columns: usize, rows: usize, pivot: usize, col: usize, diagonal: []const T, stride: usize, b: [*]T, ldb: BlasInt, row0: usize, col0: usize) void {
    var coeff: [columns]T = undefined;
    var dest: [columns][*]T = undefined;
    inline for (0..columns) |j| {
        coeff[j] = scalar.neg(T, diagonal[pivot + (col + j) * stride]);
        dest[j] = b + matIndex(ldb, row0, col0 + col + j);
    }
    updateSolvePanel(T, columns, rows, b + matIndex(ldb, row0, col0 + pivot), coeff, dest);
}

fn solvePackedLeft(comptime T: type, triangle: packing.Triangle, rows: usize, cols: usize, diagonal_block: []const T, b: [*]T, ldb: BlasInt, row0: usize, col0: usize) void {
    if (comptime builtin.cpu.arch == .x86_64) {
        var reciprocal: [256]T = undefined;
        for (0..rows) |i| reciprocal[i] = scalar.divv(T, scalar.one(T), diagonal_block[i + i * rows]);
        const columns = if (comptime scalar.isComplex(T)) 2 else 4;
        var j: usize = 0;
        while (j + columns <= cols) : (j += columns) solveLeftPanel(T, columns, triangle, rows, diagonal_block, reciprocal[0..rows], b, ldb, row0, col0 + j);
        while (j < cols) : (j += 1) solveLeftPanel(T, 1, triangle, rows, diagonal_block, reciprocal[0..rows], b, ldb, row0, col0 + j);
        return;
    }
    for (0..cols) |j| matrix_vector_ops.trsv(T, unpackTriangle(triangle), .no_trans, .non_unit, @intCast(rows), diagonal_block.ptr, @intCast(rows), b + matIndex(ldb, row0, col0 + j), 1);
}

fn solvePackedRight(comptime T: type, triangle: packing.Triangle, rows: usize, cols: usize, diagonal_block: []const T, b: [*]T, ldb: BlasInt, row0: usize, col0: usize) void {
    if (comptime builtin.cpu.arch == .x86_64) {
        const columns = if (comptime scalar.isComplex(T)) 2 else 4;
        for (0..cols) |step| {
            const p = if (triangle == .upper) step else cols - 1 - step;
            const reciprocal = scalar.divv(T, scalar.one(T), diagonal_block[p + p * cols]);
            for (0..rows) |i| {
                const index = matIndex(ldb, row0 + i, col0 + p);
                b[index] = scalar.mul(T, b[index], reciprocal);
            }
            var j: usize = if (triangle == .upper) p + 1 else 0;
            const end = if (triangle == .upper) cols else p;
            while (j + columns <= end) : (j += columns) updateRightPanel(T, columns, rows, p, j, diagonal_block, cols, b, ldb, row0, col0);
            while (j < end) : (j += 1) updateRightPanel(T, 1, rows, p, j, diagonal_block, cols, b, ldb, row0, col0);
        }
        return;
    }
    for (0..rows) |i| matrix_vector_ops.trsv(T, unpackTriangle(triangle), .trans, .non_unit, @intCast(cols), diagonal_block.ptr, @intCast(cols), b + matIndex(ldb, row0 + i, col0), ldb);
}

/// Blocked TRSM. Off-diagonal updates use GEMM and diagonal blocks use the
/// existing triangular solve leaf after op(A) has been packed explicitly.
pub fn tryTrsm(comptime T: type, options: Options, side: Side, uplo: Uplo, trans: Order, diag: Diag, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) bool {
    if (m_ <= 0 or n_ <= 0) return true;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (scalar.isZero(T, alpha)) {
        zeroGeneral(T, m, n, b, ldb);
        return true;
    }
    if (tryQueuedTriangular(T, true, options, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)) return true;
    const workspace = acquireWorkspace(T, options, 1) orelse return false;
    defer std.heap.c_allocator.free(workspace);
    return trsmWorkspace(T, options.block_size, side, uplo, trans, diag, m_, n_, alpha, a, lda, b, ldb, workspace);
}

fn trsmWorkspace(comptime T: type, bs: usize, side: Side, uplo: Uplo, trans: Order, diag: Diag, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt, workspace: []T) bool {
    const m = toUsize(m_);
    const n = toUsize(n_);
    scaleGeneral(T, m, n, alpha, b, ldb);
    const effective = packing.effectiveTriangle(packTriangle(uplo), packTranspose(trans));
    const minus_one = scalar.neg(T, scalar.one(T));

    // Solve a diagonal block for all right-hand sides, then update the
    // remaining matrix once. Off-diagonal rectangles are already valid dense
    // operands; only the diagonal triangle needs materialization.
    if (comptime builtin.cpu.arch == .x86_64) {
        const dimension = if (side == .left) m else n;
        const blocks = blockCount(dimension, bs);
        for (0..blocks) |step| {
            const forward = if (side == .left) effective == .lower else effective == .upper;
            const bi = if (forward) step else blocks - 1 - step;
            const block = blockExtent(dimension, bs, bi);
            const diagonal = workspace[0 .. block.len * block.len];
            packTriangularPanel(T, uplo, trans, diag, a, lda, block.start, block.start, block.len, block.len, diagonal);
            const first = if (forward) block.start + block.len else 0;
            const count = if (forward) dimension - first else block.start;
            if (side == .left) {
                solvePackedLeft(T, effective, block.len, n, diagonal, b, ldb, block.start, 0);
                if (count > 0) {
                    const panel = a + if (trans == .no_trans) matIndex(lda, first, block.start) else matIndex(lda, block.start, first);
                    gemm_impl.gemm(T, trans, .no_trans, @intCast(count), n_, @intCast(block.len), minus_one, panel, lda, b + block.start, ldb, scalar.one(T), b + first, ldb);
                }
            } else {
                solvePackedRight(T, effective, m, block.len, diagonal, b, ldb, 0, block.start);
                if (count > 0) {
                    const panel = a + if (trans == .no_trans) matIndex(lda, block.start, first) else matIndex(lda, first, block.start);
                    gemm_impl.gemm(T, .no_trans, trans, m_, @intCast(count), @intCast(block.len), minus_one, b + matIndex(ldb, 0, block.start), ldb, panel, lda, scalar.one(T), b + matIndex(ldb, 0, first), ldb);
                }
            }
        }
        return true;
    }

    if (side == .left) {
        const blocks = blockCount(m, bs);
        var col0: usize = 0;
        while (col0 < n) : (col0 += bs) {
            const cols = @min(bs, n - col0);
            for (0..blocks) |step| {
                const bi = if (effective == .upper) blocks - 1 - step else step;
                const row_block = blockExtent(m, bs, bi);
                const p_first: usize = if (effective == .upper) bi + 1 else 0;
                const p_end: usize = if (effective == .upper) blocks else bi;
                for (p_first..p_end) |pi| {
                    const p_block = blockExtent(m, bs, pi);
                    const panel = workspace[0 .. row_block.len * p_block.len];
                    packTriangularPanel(T, uplo, trans, diag, a, lda, row_block.start, p_block.start, row_block.len, p_block.len, panel);
                    gemm_impl.gemm(T, .no_trans, .no_trans, @intCast(row_block.len), @intCast(cols), @intCast(p_block.len), minus_one, panel.ptr, @intCast(row_block.len), b + matIndex(ldb, p_block.start, col0), ldb, scalar.one(T), b + matIndex(ldb, row_block.start, col0), ldb);
                }
                const diagonal_block = workspace[0 .. row_block.len * row_block.len];
                packTriangularPanel(T, uplo, trans, diag, a, lda, row_block.start, row_block.start, row_block.len, row_block.len, diagonal_block);
                solvePackedLeft(T, effective, row_block.len, cols, diagonal_block, b, ldb, row_block.start, col0);
            }
        }
    } else {
        const blocks = blockCount(n, bs);
        var row0: usize = 0;
        while (row0 < m) : (row0 += bs) {
            const rows = @min(bs, m - row0);
            for (0..blocks) |step| {
                const bj = if (effective == .upper) step else blocks - 1 - step;
                const col_block = blockExtent(n, bs, bj);
                const p_first: usize = if (effective == .upper) 0 else bj + 1;
                const p_end: usize = if (effective == .upper) bj else blocks;
                for (p_first..p_end) |pi| {
                    const p_block = blockExtent(n, bs, pi);
                    const panel = workspace[0 .. p_block.len * col_block.len];
                    packTriangularPanel(T, uplo, trans, diag, a, lda, p_block.start, col_block.start, p_block.len, col_block.len, panel);
                    gemm_impl.gemm(T, .no_trans, .no_trans, @intCast(rows), @intCast(col_block.len), @intCast(p_block.len), minus_one, b + matIndex(ldb, row0, p_block.start), ldb, panel.ptr, @intCast(p_block.len), scalar.one(T), b + matIndex(ldb, row0, col_block.start), ldb);
                }
                const diagonal_block = workspace[0 .. col_block.len * col_block.len];
                packTriangularPanel(T, uplo, trans, diag, a, lda, col_block.start, col_block.start, col_block.len, col_block.len, diagonal_block);
                solvePackedRight(T, effective, rows, col_block.len, diagonal_block, b, ldb, row0, col_block.start);
            }
        }
    }
    return true;
}
