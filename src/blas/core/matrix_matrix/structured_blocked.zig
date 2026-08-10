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

/// Computes SYRK/HERK into private block tiles and commits only the requested
/// triangle, preserving the unstored half bitwise.
pub fn trySyrk(comptime T: type, options: Options, uplo: Uplo, trans: Order, n_: BlasInt, k_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, beta: T, c: [*]T, ldc: BlasInt, hermitian: bool) bool {
    if (n_ <= 0) return true;
    const n = toUsize(n_);
    if (k_ <= 0 or scalar.isZero(T, alpha)) {
        scaleStoredTriangle(T, uplo, n, beta, c, ldc, hermitian);
        return true;
    }
    const workspace = acquireWorkspace(T, options, 1) orelse return false;
    defer std.heap.c_allocator.free(workspace);
    const bs = options.block_size;

    var j0: usize = 0;
    while (j0 < n) : (j0 += bs) {
        const jb = @min(bs, n - j0);
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
    const workspace = acquireWorkspace(T, options, 1) orelse return false;
    defer std.heap.c_allocator.free(workspace);
    const bs = options.block_size;
    const second_alpha = if (hermitian) scalar.conj(T, alpha) else alpha;

    var j0: usize = 0;
    while (j0 < n) : (j0 += bs) {
        const jb = @min(bs, n - j0);
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
pub fn tryTrmm(comptime T: type, options: Options, side: Side, uplo: Uplo, trans: Order, diag: Diag, m_: BlasInt, n_: BlasInt, alpha: T, a: [*]const T, lda: BlasInt, b: [*]T, ldb: BlasInt) bool {
    if (m_ <= 0 or n_ <= 0) return true;
    const m = toUsize(m_);
    const n = toUsize(n_);
    if (scalar.isZero(T, alpha)) {
        zeroGeneral(T, m, n, b, ldb);
        return true;
    }
    const workspace = acquireWorkspace(T, options, 2) orelse return false;
    defer std.heap.c_allocator.free(workspace);
    const bs = options.block_size;
    const block_elements = bs * bs;
    const packed_panel = workspace[0..block_elements];
    const output = workspace[block_elements .. 2 * block_elements];
    const effective = packing.effectiveTriangle(packTriangle(uplo), packTranspose(trans));

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

fn solvePackedLeft(comptime T: type, triangle: packing.Triangle, rows: usize, cols: usize, diagonal_block: []const T, b: [*]T, ldb: BlasInt, row0: usize, col0: usize) void {
    for (0..cols) |j| matrix_vector_ops.trsv(T, unpackTriangle(triangle), .no_trans, .non_unit, @intCast(rows), diagonal_block.ptr, @intCast(rows), b + matIndex(ldb, row0, col0 + j), 1);
}

fn solvePackedRight(comptime T: type, triangle: packing.Triangle, rows: usize, cols: usize, diagonal_block: []const T, b: [*]T, ldb: BlasInt, row0: usize, col0: usize) void {
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
    const workspace = acquireWorkspace(T, options, 1) orelse return false;
    defer std.heap.c_allocator.free(workspace);
    scaleGeneral(T, m, n, alpha, b, ldb);
    const bs = options.block_size;
    const effective = packing.effectiveTriangle(packTriangle(uplo), packTranspose(trans));
    const minus_one = scalar.neg(T, scalar.one(T));

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
