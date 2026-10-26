// SPDX-License-Identifier: LGPL-3.0-or-later
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const heteris_include = b.option([]const u8, "heteris-include", "Absolute Heteris Data SDK include directory") orelse
        @panic("Supply -Dheteris-include=/path/to/heteris/include");
    const upstream = b.createModule(.{
        .root_source_file = b.path("../../src/scientific_platform_sources.zig"),
        .target = target,
        .optimize = optimize,
    });
    const module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .pic = true,
    });
    module.addIncludePath(b.path("include"));
    module.addIncludePath(.{ .cwd_relative = heteris_include });
    module.addImport("scientific_sources", upstream);
    // Zig 0.17 moves C translation into the build graph. Generate the binding
    // from the authoritative C headers instead of maintaining duplicate layouts.
    const translated = b.addTranslateC(.{
        .root_source_file = b.path("include/zynum.h"),
        .target = target,
        .optimize = optimize,
    });
    translated.addIncludePath(.{ .cwd_relative = heteris_include });
    module.addImport("zynum_c", translated.createModule());
    // Compile the authoritative C inline validator as C: Zig 0.16 translate-c
    // mishandles array member subscripts inside this particular inline function.
    module.addCSourceFile(.{ .file = b.path("src/descriptor.c"), .flags = &.{"-std=c11"} });
    const library = b.addLibrary(.{
        .name = "zynum",
        .root_module = module,
        .use_lld = true,
        .use_llvm = true,
        .linkage = if (b.option(bool, "shared", "Build a shared library (default: static)") orelse false) .dynamic else .static,
    });
    // Plain C/C++ linkers must receive Zig stack probes and C sanitizer runtime
    // inside the archive; the consumer does not invoke the Zig driver.
    library.bundle_compiler_rt = true;
    library.bundle_ubsan_rt = true;
    b.installArtifact(library);
    b.installFile("include/zynum.h", "include/zynum.h");
    b.installFile("include/zynum_spectral.h", "include/zynum_spectral.h");
    b.getInstallStep().dependOn(&b.addInstallFileWithDir(.{
        .cwd_relative = b.pathJoin(&.{ heteris_include, "heteris.h" }),
    }, .header, "heteris.h").step);
    const tests = b.addTest(.{ .root_module = module, .use_lld = true, .use_llvm = true });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run CPU arithmetic, ABI, stride, and validation tests").dependOn(&run_tests.step);
}
