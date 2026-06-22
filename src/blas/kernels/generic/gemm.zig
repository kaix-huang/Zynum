// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const gemm_task = @import("../gemm_task.zig");

const matIndex = gemm_task.matIndex;

pub fn preferredColumnBlock(comptime T: type) usize {
    if (T == f32 or T == f64) return 4;
    @compileError("generic GEMM kernels support f32 and f64");
}

inline fn oldF32x4(task: gemm_task.Task(f32), row: usize, col: usize) @Vector(4, f32) {
    if (task.beta == 0) return @splat(0);
    const old: @Vector(4, f32) = .{
        task.c[matIndex(task.ldc, row + 0, col)],
        task.c[matIndex(task.ldc, row + 1, col)],
        task.c[matIndex(task.ldc, row + 2, col)],
        task.c[matIndex(task.ldc, row + 3, col)],
    };
    return old * @as(@Vector(4, f32), @splat(task.beta));
}

inline fn oldF64x2(task: gemm_task.Task(f64), row: usize, col: usize) @Vector(2, f64) {
    if (task.beta == 0) return @splat(0);
    const old: @Vector(2, f64) = .{
        task.c[matIndex(task.ldc, row + 0, col)],
        task.c[matIndex(task.ldc, row + 1, col)],
    };
    return old * @as(@Vector(2, f64), @splat(task.beta));
}

pub fn noTransReal(comptime T: type, task: gemm_task.Task(T)) void {
    if (T == f32) {
        noTransRealF32(task);
    } else if (T == f64) {
        noTransRealF64(task);
    } else {
        @compileError("generic GEMM kernels support f32 and f64");
    }
}

pub fn noTransRealF32(task: gemm_task.Task(f32)) void {
    const lanes = 4;
    var j = task.n0;
    while (j + 4 <= task.n1) : (j += 4) {
        var i: usize = 0;
        while (i + lanes <= task.m) : (i += lanes) {
            var acc0: @Vector(lanes, f32) = @splat(0);
            var acc1: @Vector(lanes, f32) = @splat(0);
            var acc2: @Vector(lanes, f32) = @splat(0);
            var acc3: @Vector(lanes, f32) = @splat(0);
            var p: usize = 0;
            while (p < task.k) : (p += 1) {
                const av: @Vector(lanes, f32) = .{
                    task.a[matIndex(task.lda, i + 0, p)],
                    task.a[matIndex(task.lda, i + 1, p)],
                    task.a[matIndex(task.lda, i + 2, p)],
                    task.a[matIndex(task.lda, i + 3, p)],
                };
                acc0 = @mulAdd(@Vector(lanes, f32), av, @as(@Vector(lanes, f32), @splat(task.b[matIndex(task.ldb, p, j + 0)])), acc0);
                acc1 = @mulAdd(@Vector(lanes, f32), av, @as(@Vector(lanes, f32), @splat(task.b[matIndex(task.ldb, p, j + 1)])), acc1);
                acc2 = @mulAdd(@Vector(lanes, f32), av, @as(@Vector(lanes, f32), @splat(task.b[matIndex(task.ldb, p, j + 2)])), acc2);
                acc3 = @mulAdd(@Vector(lanes, f32), av, @as(@Vector(lanes, f32), @splat(task.b[matIndex(task.ldb, p, j + 3)])), acc3);
            }
            const alpha_v: @Vector(lanes, f32) = @splat(task.alpha);
            inline for (0..4) |col| {
                const acc = switch (col) {
                    0 => acc0,
                    1 => acc1,
                    2 => acc2,
                    else => acc3,
                };
                const old = oldF32x4(task, i, j + col);
                const out = @mulAdd(@Vector(lanes, f32), acc, alpha_v, old);
                inline for (0..lanes) |r| task.c[matIndex(task.ldc, i + r, j + col)] = out[r];
            }
        }
        while (i < task.m) : (i += 1) {
            var acc0: f32 = 0;
            var acc1: f32 = 0;
            var acc2: f32 = 0;
            var acc3: f32 = 0;
            for (0..task.k) |p| {
                const av = task.a[matIndex(task.lda, i, p)];
                acc0 = @mulAdd(f32, av, task.b[matIndex(task.ldb, p, j + 0)], acc0);
                acc1 = @mulAdd(f32, av, task.b[matIndex(task.ldb, p, j + 1)], acc1);
                acc2 = @mulAdd(f32, av, task.b[matIndex(task.ldb, p, j + 2)], acc2);
                acc3 = @mulAdd(f32, av, task.b[matIndex(task.ldb, p, j + 3)], acc3);
            }
            inline for (0..4) |col| {
                const acc = switch (col) {
                    0 => acc0,
                    1 => acc1,
                    2 => acc2,
                    else => acc3,
                };
                const idxc = matIndex(task.ldc, i, j + col);
                task.c[idxc] = @mulAdd(f32, task.alpha, acc, if (task.beta == 0) 0 else task.beta * task.c[idxc]);
            }
        }
    }
    while (j < task.n1) : (j += 1) {
        var i: usize = 0;
        while (i + lanes <= task.m) : (i += lanes) {
            var acc: @Vector(lanes, f32) = @splat(0);
            var p: usize = 0;
            while (p < task.k) : (p += 1) {
                const av: @Vector(lanes, f32) = .{
                    task.a[matIndex(task.lda, i + 0, p)],
                    task.a[matIndex(task.lda, i + 1, p)],
                    task.a[matIndex(task.lda, i + 2, p)],
                    task.a[matIndex(task.lda, i + 3, p)],
                };
                acc = @mulAdd(@Vector(lanes, f32), av, @as(@Vector(lanes, f32), @splat(task.b[matIndex(task.ldb, p, j)])), acc);
            }
            const old = oldF32x4(task, i, j);
            const out = @mulAdd(@Vector(lanes, f32), acc, @as(@Vector(lanes, f32), @splat(task.alpha)), old);
            inline for (0..lanes) |r| task.c[matIndex(task.ldc, i + r, j)] = out[r];
        }
        while (i < task.m) : (i += 1) {
            var acc: f32 = 0;
            for (0..task.k) |p| acc = @mulAdd(f32, task.a[matIndex(task.lda, i, p)], task.b[matIndex(task.ldb, p, j)], acc);
            const idxc = matIndex(task.ldc, i, j);
            task.c[idxc] = @mulAdd(f32, task.alpha, acc, if (task.beta == 0) 0 else task.beta * task.c[idxc]);
        }
    }
}

pub fn noTransRealF64(task: gemm_task.Task(f64)) void {
    const lanes = 2;
    var j = task.n0;
    while (j + 4 <= task.n1) : (j += 4) {
        var i: usize = 0;
        while (i + 4 <= task.m) : (i += 4) {
            var acc0_lo: @Vector(lanes, f64) = @splat(0);
            var acc1_lo: @Vector(lanes, f64) = @splat(0);
            var acc2_lo: @Vector(lanes, f64) = @splat(0);
            var acc3_lo: @Vector(lanes, f64) = @splat(0);
            var acc0_hi: @Vector(lanes, f64) = @splat(0);
            var acc1_hi: @Vector(lanes, f64) = @splat(0);
            var acc2_hi: @Vector(lanes, f64) = @splat(0);
            var acc3_hi: @Vector(lanes, f64) = @splat(0);
            var p: usize = 0;
            while (p < task.k) : (p += 1) {
                const av_lo: @Vector(lanes, f64) = .{
                    task.a[matIndex(task.lda, i + 0, p)],
                    task.a[matIndex(task.lda, i + 1, p)],
                };
                const av_hi: @Vector(lanes, f64) = .{
                    task.a[matIndex(task.lda, i + 2, p)],
                    task.a[matIndex(task.lda, i + 3, p)],
                };
                const b0: @Vector(lanes, f64) = @splat(task.b[matIndex(task.ldb, p, j + 0)]);
                const b1: @Vector(lanes, f64) = @splat(task.b[matIndex(task.ldb, p, j + 1)]);
                const b2: @Vector(lanes, f64) = @splat(task.b[matIndex(task.ldb, p, j + 2)]);
                const b3: @Vector(lanes, f64) = @splat(task.b[matIndex(task.ldb, p, j + 3)]);
                acc0_lo = @mulAdd(@Vector(lanes, f64), av_lo, b0, acc0_lo);
                acc1_lo = @mulAdd(@Vector(lanes, f64), av_lo, b1, acc1_lo);
                acc2_lo = @mulAdd(@Vector(lanes, f64), av_lo, b2, acc2_lo);
                acc3_lo = @mulAdd(@Vector(lanes, f64), av_lo, b3, acc3_lo);
                acc0_hi = @mulAdd(@Vector(lanes, f64), av_hi, b0, acc0_hi);
                acc1_hi = @mulAdd(@Vector(lanes, f64), av_hi, b1, acc1_hi);
                acc2_hi = @mulAdd(@Vector(lanes, f64), av_hi, b2, acc2_hi);
                acc3_hi = @mulAdd(@Vector(lanes, f64), av_hi, b3, acc3_hi);
            }
            const alpha_v: @Vector(lanes, f64) = @splat(task.alpha);
            inline for (0..4) |col| {
                const acc_lo = switch (col) {
                    0 => acc0_lo,
                    1 => acc1_lo,
                    2 => acc2_lo,
                    else => acc3_lo,
                };
                const acc_hi = switch (col) {
                    0 => acc0_hi,
                    1 => acc1_hi,
                    2 => acc2_hi,
                    else => acc3_hi,
                };
                const old_lo = oldF64x2(task, i, j + col);
                const old_hi = oldF64x2(task, i + 2, j + col);
                const out_lo = @mulAdd(@Vector(lanes, f64), acc_lo, alpha_v, old_lo);
                const out_hi = @mulAdd(@Vector(lanes, f64), acc_hi, alpha_v, old_hi);
                task.c[matIndex(task.ldc, i + 0, j + col)] = out_lo[0];
                task.c[matIndex(task.ldc, i + 1, j + col)] = out_lo[1];
                task.c[matIndex(task.ldc, i + 2, j + col)] = out_hi[0];
                task.c[matIndex(task.ldc, i + 3, j + col)] = out_hi[1];
            }
        }
        while (i + lanes <= task.m) : (i += lanes) {
            var acc0: @Vector(lanes, f64) = @splat(0);
            var acc1: @Vector(lanes, f64) = @splat(0);
            var acc2: @Vector(lanes, f64) = @splat(0);
            var acc3: @Vector(lanes, f64) = @splat(0);
            var p: usize = 0;
            while (p < task.k) : (p += 1) {
                const av: @Vector(lanes, f64) = .{
                    task.a[matIndex(task.lda, i + 0, p)],
                    task.a[matIndex(task.lda, i + 1, p)],
                };
                acc0 = @mulAdd(@Vector(lanes, f64), av, @as(@Vector(lanes, f64), @splat(task.b[matIndex(task.ldb, p, j + 0)])), acc0);
                acc1 = @mulAdd(@Vector(lanes, f64), av, @as(@Vector(lanes, f64), @splat(task.b[matIndex(task.ldb, p, j + 1)])), acc1);
                acc2 = @mulAdd(@Vector(lanes, f64), av, @as(@Vector(lanes, f64), @splat(task.b[matIndex(task.ldb, p, j + 2)])), acc2);
                acc3 = @mulAdd(@Vector(lanes, f64), av, @as(@Vector(lanes, f64), @splat(task.b[matIndex(task.ldb, p, j + 3)])), acc3);
            }
            const alpha_v: @Vector(lanes, f64) = @splat(task.alpha);
            inline for (0..4) |col| {
                const acc = switch (col) {
                    0 => acc0,
                    1 => acc1,
                    2 => acc2,
                    else => acc3,
                };
                const old = oldF64x2(task, i, j + col);
                const out = @mulAdd(@Vector(lanes, f64), acc, alpha_v, old);
                inline for (0..lanes) |r| task.c[matIndex(task.ldc, i + r, j + col)] = out[r];
            }
        }
        while (i < task.m) : (i += 1) {
            var acc0: f64 = 0;
            var acc1: f64 = 0;
            var acc2: f64 = 0;
            var acc3: f64 = 0;
            for (0..task.k) |p| {
                const av = task.a[matIndex(task.lda, i, p)];
                acc0 = @mulAdd(f64, av, task.b[matIndex(task.ldb, p, j + 0)], acc0);
                acc1 = @mulAdd(f64, av, task.b[matIndex(task.ldb, p, j + 1)], acc1);
                acc2 = @mulAdd(f64, av, task.b[matIndex(task.ldb, p, j + 2)], acc2);
                acc3 = @mulAdd(f64, av, task.b[matIndex(task.ldb, p, j + 3)], acc3);
            }
            inline for (0..4) |col| {
                const acc = switch (col) {
                    0 => acc0,
                    1 => acc1,
                    2 => acc2,
                    else => acc3,
                };
                const idxc = matIndex(task.ldc, i, j + col);
                task.c[idxc] = @mulAdd(f64, task.alpha, acc, if (task.beta == 0) 0 else task.beta * task.c[idxc]);
            }
        }
    }
    while (j < task.n1) : (j += 1) {
        var i: usize = 0;
        while (i + lanes <= task.m) : (i += lanes) {
            var acc: @Vector(lanes, f64) = @splat(0);
            var p: usize = 0;
            while (p < task.k) : (p += 1) {
                const av: @Vector(lanes, f64) = .{
                    task.a[matIndex(task.lda, i + 0, p)],
                    task.a[matIndex(task.lda, i + 1, p)],
                };
                acc = @mulAdd(@Vector(lanes, f64), av, @as(@Vector(lanes, f64), @splat(task.b[matIndex(task.ldb, p, j)])), acc);
            }
            const old = oldF64x2(task, i, j);
            const out = @mulAdd(@Vector(lanes, f64), acc, @as(@Vector(lanes, f64), @splat(task.alpha)), old);
            inline for (0..lanes) |r| task.c[matIndex(task.ldc, i + r, j)] = out[r];
        }
        while (i < task.m) : (i += 1) {
            var acc: f64 = 0;
            for (0..task.k) |p| acc = @mulAdd(f64, task.a[matIndex(task.lda, i, p)], task.b[matIndex(task.ldb, p, j)], acc);
            const idxc = matIndex(task.ldc, i, j);
            task.c[idxc] = @mulAdd(f64, task.alpha, acc, if (task.beta == 0) 0 else task.beta * task.c[idxc]);
        }
    }
}
