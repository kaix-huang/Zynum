// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Shared alpha/beta write-back helpers for real GEMM kernels.

const gemm_task = @import("task.zig");

const matIndex = gemm_task.matIndex;

pub inline fn scaleOldScalar(comptime T: type, beta: T, old: T) T {
    if (beta == 0) return @as(T, 0);
    if (beta == 1) return old;
    return beta * old;
}

pub inline fn applyScalar(comptime T: type, alpha: T, beta: T, acc: T, old: T) T {
    if (alpha == 1 and beta == 0) return acc;
    return @mulAdd(T, alpha, acc, scaleOldScalar(T, beta, old));
}

pub inline fn applyTaskScalar(comptime T: type, task: gemm_task.Task(T), acc: T, row: usize, col: usize) T {
    if (task.alpha == 1 and task.beta == 0) return acc;
    if (task.beta == 0) return @mulAdd(T, task.alpha, acc, @as(T, 0));
    const idxc = matIndex(task.ldc, row, col);
    return applyScalar(T, task.alpha, task.beta, acc, task.c[idxc]);
}

pub inline fn scaleOldVector(comptime T: type, comptime lanes: comptime_int, beta: T, old: @Vector(lanes, T)) @Vector(lanes, T) {
    const V = @Vector(lanes, T);
    if (beta == 0) return @splat(0);
    if (beta == 1) return old;
    return old * @as(V, @splat(beta));
}

pub inline fn applyVector(comptime T: type, comptime lanes: comptime_int, alpha: T, beta: T, acc: @Vector(lanes, T), old: @Vector(lanes, T)) @Vector(lanes, T) {
    const V = @Vector(lanes, T);
    if (alpha == 1 and beta == 0) return acc;
    return @mulAdd(V, acc, @as(V, @splat(alpha)), scaleOldVector(T, lanes, beta, old));
}
