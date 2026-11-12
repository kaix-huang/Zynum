// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");
pub fn build(b: *std.Build) void {
    // This entrypoint reads SDK environment before configuring its dependency.
    // Recheck it on every invocation, including absent-to-present transitions.
    b.graph.poisonCache();
    const target = b.standardTargetOptions(.{ .default_target = .{ .cpu_model = .baseline } });
    const optimize = b.standardOptimizeOption(.{});
    const include = b.option([]const u8, "heteris-include", "Explicit absolute flat Heteris SDK or canonical resource-contract include directory") orelse @panic("Supply -Dheteris-include");
    const data_include = b.option([]const u8, "heteris-data-include", "Explicit separate canonical Heteris Data include directory") orelse b.graph.environ_map.get("ZYNUM_HETERIS_DATA_INCLUDE");
    const dependency = b.dependency("original_zynum", .{
        .target = target,
        .optimize = optimize,
        .@"execution-profile" = .integrated_native,
        .@"heteris-include" = include,
        .@"heteris-data-include" = data_include,
        .standalone = b.option(bool, "standalone", "Separate explicit standalone adapter provider") orelse false,
        .shared = b.option(bool, "shared", "One shared main libzynum") orelse false,
    });
    const main = dependency.artifact("zynum");
    if (main.root_module.import_table.get("zynum-core") != dependency.module("zynum-core"))
        @panic("main library does not use the original dependency Core");
    b.getInstallStep().dependOn(dependency.builder.getInstallStep());
    const tests = dependency.builder.top_level_steps.get("test-platform") orelse @panic("original graph lacks adapter tests");
    b.step("test", "Run the existing adapter tests in the original root graph").dependOn(&tests.step);
}
