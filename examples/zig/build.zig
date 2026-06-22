// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zynum_dep = b.dependency("zynum", .{});

    const exe = b.addExecutable(.{
        .name = "zynum-zig-matrix-multiply",
        .root_module = b.createModule(.{
            .root_source_file = b.path("matrix_multiply.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zynum", .module = zynum_dep.module("zynum") },
            },
        }),
    });

    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    if (b.args) |args| run.addArgs(args);

    const run_step = b.step("run", "Run the Zynum Zig matrix multiplication example");
    run_step.dependOn(&run.step);
}
