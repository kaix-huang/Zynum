// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const zynum_mod = b.addModule("zynum", .{
        .root_source_file = b.path("src/zynum.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    _ = b.addModule("zynum-blas", .{
        .root_source_file = b.path("src/blas.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const blas_compat_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/compat.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const fortran_compat_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/compat_fortran.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const cblas_compat_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/compat_cblas.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const lib = b.addLibrary(.{
        .name = "zynum_blas",
        .linkage = .dynamic,
        .root_module = blas_compat_mod,
    });
    b.installArtifact(lib);

    const static_lib = b.addLibrary(.{
        .name = "zynum_blas",
        .linkage = .static,
        .root_module = blas_compat_mod,
    });
    b.installArtifact(static_lib);

    const install_compat_headers = b.option(bool, "compat-headers", "Install Zynum BLAS CBLAS and BLAS/Fortran compatibility headers/modules") orelse true;
    if (install_compat_headers) {
        b.installFile("include/zynum/blas/cblas.h", "include/zynum/blas/cblas.h");
        b.installFile("include/zynum/blas/blas.h", "include/zynum/blas/blas.h");
        b.installFile("include/zynum/blas/blas.f90", "include/zynum/blas/blas.f90");
    }

    const generate_headers_tool = b.addExecutable(.{
        .name = "generate_compat_headers",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/generate_compat_headers.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });
    const generate_headers = b.addRunArtifact(generate_headers_tool);
    generate_headers.addArg(b.pathFromRoot("."));
    const generate_headers_step = b.step("generate-headers", "Regenerate Zynum BLAS CBLAS and BLAS/Fortran compatibility headers/modules");
    generate_headers_step.dependOn(&generate_headers.step);

    const modern_tests = b.addTest(.{
        .name = "zynum-modern-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/api/zynum_test.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zynum", .module = zynum_mod }},
        }),
    });
    const run_modern_tests = b.addRunArtifact(modern_tests);

    const fortran_tests = b.addTest(.{
        .name = "zynum-blas-fortran-compat-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/abi/fortran_compat_test.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zynum_blas_fortran_compat", .module = fortran_compat_mod }},
        }),
    });
    const run_fortran_tests = b.addRunArtifact(fortran_tests);

    const cblas_tests = b.addTest(.{
        .name = "zynum-blas-cblas-compat-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/abi/cblas_compat_test.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zynum_blas_cblas_compat", .module = cblas_compat_mod }},
        }),
    });
    const run_cblas_tests = b.addRunArtifact(cblas_tests);

    const header_smoke_mod = b.createModule(.{
        .root_source_file = b.path("test/headers/compat_headers_smoke.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    header_smoke_mod.addIncludePath(b.path("include"));
    const header_smoke_tests = b.addTest(.{
        .name = "zynum-blas-header-smoke-tests",
        .root_module = header_smoke_mod,
    });
    const run_header_smoke_tests = b.addRunArtifact(header_smoke_tests);
    const fortran_module_smoke_test = b.addSystemCommand(&.{
        "sh",
        "-c",
        b.fmt("if command -v gfortran >/dev/null 2>&1; then mkdir -p '{s}' && gfortran -std=f2008 -J '{s}' -fsyntax-only '{s}' '{s}'; fi", .{
            b.pathFromRoot("zig-out/fortran-mod"),
            b.pathFromRoot("zig-out/fortran-mod"),
            b.pathFromRoot("include/zynum/blas/blas.f90"),
            b.pathFromRoot("test/headers/fortran_module_smoke.f90"),
        }),
    });

    const test_step = b.step("test", "Run correctness tests");
    test_step.dependOn(&run_modern_tests.step);
    test_step.dependOn(&run_fortran_tests.step);
    test_step.dependOn(&run_cblas_tests.step);
    test_step.dependOn(&run_header_smoke_tests.step);
    test_step.dependOn(&fortran_module_smoke_test.step);

    const bench = b.addExecutable(.{
        .name = "bench-zynum-blas",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/bench.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.installArtifact(bench);

    const run_bench = b.addRunArtifact(bench);
    run_bench.addArg("--zynum-blas");
    run_bench.addFileArg(lib.getEmittedBin());
    const bench_openblas = b.option([]const u8, "bench-openblas", "Path to an OpenBLAS shared library for the bench step");
    const bench_accelerate = b.option([]const u8, "bench-accelerate", "Path to Accelerate for the bench step");
    const bench_mkl = b.option([]const u8, "bench-mkl", "Path to an MKL shared library exporting Fortran BLAS symbols for the bench step");
    if (bench_openblas) |path| {
        run_bench.addArg("--openblas");
        run_bench.addArg(path);
    } else if (target.result.os.tag == .macos) {
        run_bench.addArg("--openblas");
        run_bench.addArg("/opt/homebrew/opt/openblas/lib/libopenblas.dylib");
    }
    if (bench_accelerate) |path| {
        run_bench.addArg("--accelerate");
        run_bench.addArg(path);
    } else if (target.result.os.tag == .macos) {
        run_bench.addArg("--accelerate");
        run_bench.addArg("/System/Library/Frameworks/Accelerate.framework/Accelerate");
    }
    if (bench_mkl) |path| {
        run_bench.addArg("--mkl");
        run_bench.addArg(path);
    }
    if (b.args) |args| run_bench.addArgs(args);
    run_bench.step.dependOn(b.getInstallStep());

    const bench_step = b.step("bench", "Benchmark Zynum BLAS against Accelerate and OpenBLAS");
    bench_step.dependOn(&run_bench.step);

    const gemm_sweep = b.addExecutable(.{
        .name = "gemm-sweep",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/gemm_sweep.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.installArtifact(gemm_sweep);

    const run_gemm_sweep = b.addRunArtifact(gemm_sweep);
    run_gemm_sweep.addArg("--zynum-blas");
    run_gemm_sweep.addFileArg(lib.getEmittedBin());
    run_gemm_sweep.addArg("--csv");
    run_gemm_sweep.addArg("zig-out/gemm_sweep.csv");
    if (bench_openblas) |path| {
        run_gemm_sweep.addArg("--openblas");
        run_gemm_sweep.addArg(path);
    } else if (target.result.os.tag == .macos) {
        run_gemm_sweep.addArg("--openblas");
        run_gemm_sweep.addArg("/opt/homebrew/opt/openblas/lib/libopenblas.dylib");
    }
    if (bench_accelerate) |path| {
        run_gemm_sweep.addArg("--accelerate");
        run_gemm_sweep.addArg(path);
    } else if (target.result.os.tag == .macos) {
        run_gemm_sweep.addArg("--accelerate");
        run_gemm_sweep.addArg("/System/Library/Frameworks/Accelerate.framework/Accelerate");
    }
    if (bench_mkl) |path| {
        run_gemm_sweep.addArg("--mkl");
        run_gemm_sweep.addArg(path);
    }
    if (b.args) |args| run_gemm_sweep.addArgs(args);
    run_gemm_sweep.step.dependOn(b.getInstallStep());

    const gemm_sweep_step = b.step("bench-gemm-sweep", "Sweep GEMM shapes and write CSV results");
    gemm_sweep_step.dependOn(&run_gemm_sweep.step);

    const level12_sweep = b.addExecutable(.{
        .name = "level12-sweep",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/level12_sweep.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.installArtifact(level12_sweep);

    const run_level12_sweep = b.addRunArtifact(level12_sweep);
    run_level12_sweep.addArg("--zynum-blas");
    run_level12_sweep.addFileArg(lib.getEmittedBin());
    if (bench_openblas) |path| {
        run_level12_sweep.addArg("--openblas");
        run_level12_sweep.addArg(path);
    } else if (target.result.os.tag == .macos) {
        run_level12_sweep.addArg("--openblas");
        run_level12_sweep.addArg("/opt/homebrew/opt/openblas/lib/libopenblas.dylib");
    }
    if (bench_accelerate) |path| {
        run_level12_sweep.addArg("--accelerate");
        run_level12_sweep.addArg(path);
    } else if (target.result.os.tag == .macos) {
        run_level12_sweep.addArg("--accelerate");
        run_level12_sweep.addArg("/System/Library/Frameworks/Accelerate.framework/Accelerate");
    }
    if (b.args) |args| run_level12_sweep.addArgs(args);
    run_level12_sweep.step.dependOn(b.getInstallStep());

    const level12_sweep_step = b.step("bench-level12-sweep", "Sweep representative BLAS Level 1/2 kernels");
    level12_sweep_step.dependOn(&run_level12_sweep.step);
}
