// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

fn disabledBenchPath(path: []const u8) bool {
    return path.len == 0 or std.mem.eql(u8, path, "none");
}

fn addOptionalBenchLibrary(run: *std.Build.Step.Run, flag: []const u8, explicit_path: ?[]const u8, default_path: ?[]const u8) void {
    if (explicit_path) |path| {
        if (disabledBenchPath(path)) return;
        run.addArg(flag);
        run.addArg(path);
        return;
    }
    if (default_path) |path| {
        run.addArg(flag);
        run.addArg(path);
    }
}

fn addOptionalIsolatedBenchLibrary(run: *std.Build.Step.Run, flag: []const u8, explicit_path: ?[]const u8, default_path: ?[]const u8) void {
    if (explicit_path) |path| {
        // The Python isolated runner has platform defaults; pass "none" through
        // so callers can explicitly disable a comparator.
        run.addArg(flag);
        run.addArg(path);
        return;
    }
    if (default_path) |path| {
        run.addArg(flag);
        run.addArg(path);
    }
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });
    const test_optimize = b.option(std.builtin.OptimizeMode, "test-optimize", "Optimize mode for correctness tests") orelse .ReleaseSafe;
    const host_tool_smoke = b.option(bool, "host-tool-smoke", "Run host Python/C/C++/Fortran smoke checks as part of the test step") orelse true;

    const zynum_mod = b.addModule("zynum", .{
        .root_source_file = b.path("src/zynum.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const zynum_blas_mod = b.addModule("zynum-blas", .{
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
    const zynum_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zynum.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    const zynum_blas_test_mod = b.createModule(.{
        .root_source_file = b.path("src/blas.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    const fortran_compat_test_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/compat_fortran.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    const cblas_compat_test_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/compat_cblas.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    _ = zynum_mod;
    _ = zynum_blas_mod;
    _ = fortran_compat_mod;
    _ = cblas_compat_mod;
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
        b.installFile("include/zynum/blas/abi_manifest.json", "include/zynum/blas/abi_manifest.json");
        b.installFile("pkgconfig/zynum_blas.pc", "lib/pkgconfig/zynum_blas.pc");
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
            .optimize = test_optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "zynum", .module = zynum_test_mod },
            },
        }),
    });
    const run_modern_tests = b.addRunArtifact(modern_tests);

    const blas_module_tests = b.addTest(.{
        .name = "zynum-blas-module-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/api/zynum_blas_module_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zynum-blas", .module = zynum_blas_test_mod }},
        }),
    });
    const run_blas_module_tests = b.addRunArtifact(blas_module_tests);

    const fortran_tests = b.addTest(.{
        .name = "zynum-blas-fortran-compat-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/abi/fortran_compat_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "zynum_blas_fortran_compat", .module = fortran_compat_test_mod },
                .{ .name = "zynum-blas", .module = zynum_blas_test_mod },
            },
        }),
    });
    const run_fortran_tests = b.addRunArtifact(fortran_tests);

    const cblas_tests = b.addTest(.{
        .name = "zynum-blas-cblas-compat-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/abi/cblas_compat_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "zynum_blas_cblas_compat", .module = cblas_compat_test_mod }},
        }),
    });
    const run_cblas_tests = b.addRunArtifact(cblas_tests);

    const symm_dense_gemm_tests = b.addTest(.{
        .name = "zynum-blas-symm-dense-gemm-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/symm_dense_gemm_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_symm_dense_gemm_tests = b.addRunArtifact(symm_dense_gemm_tests);

    const triangular_parallel_tests = b.addTest(.{
        .name = "zynum-blas-triangular-parallel-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/triangular_parallel_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_triangular_parallel_tests = b.addRunArtifact(triangular_parallel_tests);

    const packed_parallel_tests = b.addTest(.{
        .name = "zynum-blas-packed-parallel-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/packed_parallel_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_packed_parallel_tests = b.addRunArtifact(packed_parallel_tests);

    const triangular_dense_unit_tests = b.addTest(.{
        .name = "zynum-blas-triangular-dense-unit-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/triangular_dense_unit_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_triangular_dense_unit_tests = b.addRunArtifact(triangular_dense_unit_tests);

    const triangular_band_window_tests = b.addTest(.{
        .name = "zynum-blas-triangular-band-window-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/triangular_band_window_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_triangular_band_window_tests = b.addRunArtifact(triangular_band_window_tests);

    const triangular_packed_unit_tests = b.addTest(.{
        .name = "zynum-blas-triangular-packed-unit-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/triangular_packed_unit_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_triangular_packed_unit_tests = b.addRunArtifact(triangular_packed_unit_tests);

    const triangular_band_solve_tests = b.addTest(.{
        .name = "zynum-blas-triangular-band-solve-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/triangular_band_solve_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_triangular_band_solve_tests = b.addRunArtifact(triangular_band_solve_tests);

    const vector_stride2_parallel_tests = b.addTest(.{
        .name = "zynum-blas-vector-stride2-parallel-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/vector_stride2_parallel_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_vector_stride2_parallel_tests = b.addRunArtifact(vector_stride2_parallel_tests);

    const header_smoke_mod = b.createModule(.{
        .root_source_file = b.path("test/headers/compat_headers_smoke.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    header_smoke_mod.addIncludePath(b.path("include"));
    const header_smoke_tests = b.addTest(.{
        .name = "zynum-blas-header-smoke-tests",
        .root_module = header_smoke_mod,
    });
    const run_header_smoke_tests = b.addRunArtifact(header_smoke_tests);
    const abi_manifest_smoke_test = b.addSystemCommand(&.{
        "python3",
        "-c",
        "import json, pathlib, sys; data=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert data['schema'] == 1; assert data['blas_integer_abi'] == 'LP64'; assert data['fortran']['export_count'] == 161; assert data['cblas']['export_count'] == 150; names={item['name'] for section in ('fortran','cblas') for item in data[section]['exports']}; assert {'dgemm_', 'cdotc_sub_', 'cblas_dgemm', 'cblas_zher2k'} <= names",
        b.pathFromRoot("include/zynum/blas/abi_manifest.json"),
    });
    const c_header_smoke_test = b.addSystemCommand(&.{
        "sh",
        "-c",
        b.fmt("mkdir -p '{s}' && cc -std=c11 -I '{s}' -c '{s}' -o '{s}'", .{
            b.pathFromRoot("zig-out/header-smoke"),
            b.pathFromRoot("include"),
            b.pathFromRoot("test/headers/compat_headers_c_smoke.c"),
            b.pathFromRoot("zig-out/header-smoke/compat_headers_c_smoke.o"),
        }),
    });
    const cpp_header_smoke_test = b.addSystemCommand(&.{
        "sh",
        "-c",
        b.fmt("mkdir -p '{s}' && c++ -std=c++17 -I '{s}' -c '{s}' -o '{s}'", .{
            b.pathFromRoot("zig-out/header-smoke"),
            b.pathFromRoot("include"),
            b.pathFromRoot("test/headers/compat_headers_cpp_smoke.cpp"),
            b.pathFromRoot("zig-out/header-smoke/compat_headers_cpp_smoke.o"),
        }),
    });
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
    test_step.dependOn(&run_blas_module_tests.step);
    test_step.dependOn(&run_fortran_tests.step);
    test_step.dependOn(&run_cblas_tests.step);
    test_step.dependOn(&run_symm_dense_gemm_tests.step);
    test_step.dependOn(&run_triangular_parallel_tests.step);
    test_step.dependOn(&run_packed_parallel_tests.step);
    test_step.dependOn(&run_triangular_dense_unit_tests.step);
    test_step.dependOn(&run_triangular_band_window_tests.step);
    test_step.dependOn(&run_triangular_packed_unit_tests.step);
    test_step.dependOn(&run_triangular_band_solve_tests.step);
    test_step.dependOn(&run_vector_stride2_parallel_tests.step);
    test_step.dependOn(&run_header_smoke_tests.step);
    if (host_tool_smoke) {
        test_step.dependOn(&abi_manifest_smoke_test.step);
        test_step.dependOn(&c_header_smoke_test.step);
        test_step.dependOn(&cpp_header_smoke_test.step);
        test_step.dependOn(&fortran_module_smoke_test.step);
    }

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
    const bench_aocl_blis = b.option([]const u8, "bench-aocl-blis", "Path to an AOCL-BLIS shared library exporting Fortran BLAS symbols for the bench step");
    addOptionalBenchLibrary(run_bench, "--openblas", bench_openblas, if (target.result.os.tag == .macos) "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" else null);
    addOptionalBenchLibrary(run_bench, "--accelerate", bench_accelerate, if (target.result.os.tag == .macos) "/System/Library/Frameworks/Accelerate.framework/Accelerate" else null);
    addOptionalBenchLibrary(run_bench, "--mkl", bench_mkl, null);
    addOptionalBenchLibrary(run_bench, "--aocl-blis", bench_aocl_blis, null);
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
    addOptionalBenchLibrary(run_gemm_sweep, "--openblas", bench_openblas, if (target.result.os.tag == .macos) "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" else null);
    addOptionalBenchLibrary(run_gemm_sweep, "--accelerate", bench_accelerate, if (target.result.os.tag == .macos) "/System/Library/Frameworks/Accelerate.framework/Accelerate" else null);
    addOptionalBenchLibrary(run_gemm_sweep, "--mkl", bench_mkl, null);
    addOptionalBenchLibrary(run_gemm_sweep, "--aocl-blis", bench_aocl_blis, null);
    if (b.args) |args| run_gemm_sweep.addArgs(args);
    run_gemm_sweep.step.dependOn(b.getInstallStep());

    const gemm_sweep_step = b.step("bench-gemm-sweep", "Sweep GEMM shapes and write CSV results");
    gemm_sweep_step.dependOn(&run_gemm_sweep.step);

    const run_gemm_sweep_isolated = b.addSystemCommand(&.{
        "python3",
        "bench/tools/run_gemm_sweep_isolated.py",
        "--gemm-sweep",
    });
    run_gemm_sweep_isolated.addFileArg(gemm_sweep.getEmittedBin());
    run_gemm_sweep_isolated.addArg("--zynum-blas");
    run_gemm_sweep_isolated.addFileArg(lib.getEmittedBin());
    addOptionalIsolatedBenchLibrary(run_gemm_sweep_isolated, "--openblas", bench_openblas, if (target.result.os.tag == .macos) "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" else null);
    addOptionalIsolatedBenchLibrary(run_gemm_sweep_isolated, "--accelerate", bench_accelerate, if (target.result.os.tag == .macos) "/System/Library/Frameworks/Accelerate.framework/Accelerate" else null);
    addOptionalIsolatedBenchLibrary(run_gemm_sweep_isolated, "--mkl", bench_mkl, null);
    addOptionalIsolatedBenchLibrary(run_gemm_sweep_isolated, "--aocl-blis", bench_aocl_blis, null);
    run_gemm_sweep_isolated.addArg("--csv");
    run_gemm_sweep_isolated.addArg("zig-out/gemm_sweep_isolated.csv");
    run_gemm_sweep_isolated.addArg("--process-repeats");
    run_gemm_sweep_isolated.addArg("2");
    run_gemm_sweep_isolated.addArg("--check");
    run_gemm_sweep_isolated.addArg("--skip-missing");
    if (b.args) |args| run_gemm_sweep_isolated.addArgs(args);
    run_gemm_sweep_isolated.step.dependOn(b.getInstallStep());

    const gemm_sweep_isolated_step = b.step("bench-gemm-sweep-isolated", "Run reportable fresh-process GEMM sweep with correctness checks");
    gemm_sweep_isolated_step.dependOn(&run_gemm_sweep_isolated.step);

    const vector_matrix_sweep = b.addExecutable(.{
        .name = "vector-matrix-sweep",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/vector_matrix_sweep.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.installArtifact(vector_matrix_sweep);

    const run_vector_matrix_sweep = b.addRunArtifact(vector_matrix_sweep);
    run_vector_matrix_sweep.addArg("--zynum-blas");
    run_vector_matrix_sweep.addFileArg(lib.getEmittedBin());
    addOptionalBenchLibrary(run_vector_matrix_sweep, "--openblas", bench_openblas, if (target.result.os.tag == .macos) "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" else null);
    addOptionalBenchLibrary(run_vector_matrix_sweep, "--accelerate", bench_accelerate, if (target.result.os.tag == .macos) "/System/Library/Frameworks/Accelerate.framework/Accelerate" else null);
    addOptionalBenchLibrary(run_vector_matrix_sweep, "--mkl", bench_mkl, null);
    addOptionalBenchLibrary(run_vector_matrix_sweep, "--aocl-blis", bench_aocl_blis, null);
    if (b.args) |args| run_vector_matrix_sweep.addArgs(args);
    run_vector_matrix_sweep.step.dependOn(b.getInstallStep());

    const vector_matrix_sweep_step = b.step("bench-vector-matrix-sweep", "Sweep representative BLAS Level 1/2 kernels");
    vector_matrix_sweep_step.dependOn(&run_vector_matrix_sweep.step);

    const rank_k_probe = b.addExecutable(.{
        .name = "rank-k-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/rank_k_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const install_rank_k_probe = b.addInstallArtifact(rank_k_probe, .{});
    const build_rank_k_probe_step = b.step("build-rank-k-probe", "Build the opt-in Level 3 rank-k probe");
    build_rank_k_probe_step.dependOn(&install_rank_k_probe.step);

    const run_rank_k_report = b.addSystemCommand(&.{
        "python3",
        "bench/tools/run_rank_k_report.py",
        "--probe",
    });
    run_rank_k_report.addFileArg(rank_k_probe.getEmittedBin());
    run_rank_k_report.addArg("--zynum");
    run_rank_k_report.addFileArg(lib.getEmittedBin());
    addOptionalIsolatedBenchLibrary(run_rank_k_report, "--openblas", bench_openblas, if (target.result.os.tag == .macos) "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" else null);
    addOptionalIsolatedBenchLibrary(run_rank_k_report, "--accelerate", bench_accelerate, if (target.result.os.tag == .macos) "/System/Library/Frameworks/Accelerate.framework/Accelerate" else null);
    addOptionalIsolatedBenchLibrary(run_rank_k_report, "--mkl", bench_mkl, null);
    addOptionalIsolatedBenchLibrary(run_rank_k_report, "--aocl-blis", bench_aocl_blis, null);
    run_rank_k_report.addArg("--csv");
    run_rank_k_report.addArg("zig-out/rank_k_report.csv");
    run_rank_k_report.addArg("--skip-missing");
    if (b.args) |args| run_rank_k_report.addArgs(args);

    const rank_k_report_step = b.step("bench-rank-k-report", "Run the opt-in fresh-process SYRK/HERK comparator report");
    rank_k_report_step.dependOn(&run_rank_k_report.step);

    const symm_probe = b.addExecutable(.{
        .name = "symm-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/symm_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const install_symm_probe = b.addInstallArtifact(symm_probe, .{});
    const build_symm_probe_step = b.step("build-symm-probe", "Build the opt-in Level 3 SYMM/HEMM probe");
    build_symm_probe_step.dependOn(&install_symm_probe.step);

    const run_symm_report = b.addSystemCommand(&.{
        "python3",
        "bench/tools/run_symm_report.py",
        "--probe",
    });
    run_symm_report.addFileArg(symm_probe.getEmittedBin());
    run_symm_report.addArg("--zynum");
    run_symm_report.addFileArg(lib.getEmittedBin());
    addOptionalIsolatedBenchLibrary(run_symm_report, "--openblas", bench_openblas, if (target.result.os.tag == .macos) "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" else null);
    addOptionalIsolatedBenchLibrary(run_symm_report, "--accelerate", bench_accelerate, if (target.result.os.tag == .macos) "/System/Library/Frameworks/Accelerate.framework/Accelerate" else null);
    addOptionalIsolatedBenchLibrary(run_symm_report, "--mkl", bench_mkl, null);
    addOptionalIsolatedBenchLibrary(run_symm_report, "--aocl-blis", bench_aocl_blis, null);
    run_symm_report.addArg("--csv");
    run_symm_report.addArg("zig-out/symm_report.csv");
    run_symm_report.addArg("--skip-missing");
    if (b.args) |args| run_symm_report.addArgs(args);

    const symm_report_step = b.step("bench-symm-report", "Run the opt-in fresh-process SYMM/HEMM comparator report");
    symm_report_step.dependOn(&run_symm_report.step);

    const triangular_matrix_probe = b.addExecutable(.{
        .name = "triangular-matrix-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/triangular_matrix_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const install_triangular_matrix_probe = b.addInstallArtifact(triangular_matrix_probe, .{});
    const build_triangular_matrix_probe_step = b.step("build-triangular-matrix-probe", "Build the opt-in Level 3 TRMM/TRSM probe");
    build_triangular_matrix_probe_step.dependOn(&install_triangular_matrix_probe.step);

    const run_triangular_matrix_report = b.addSystemCommand(&.{
        "python3",
        "bench/tools/run_triangular_matrix_report.py",
        "--probe",
    });
    run_triangular_matrix_report.addFileArg(triangular_matrix_probe.getEmittedBin());
    run_triangular_matrix_report.addArg("--zynum");
    run_triangular_matrix_report.addFileArg(lib.getEmittedBin());
    addOptionalIsolatedBenchLibrary(run_triangular_matrix_report, "--openblas", bench_openblas, if (target.result.os.tag == .macos) "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" else null);
    addOptionalIsolatedBenchLibrary(run_triangular_matrix_report, "--accelerate", bench_accelerate, if (target.result.os.tag == .macos) "/System/Library/Frameworks/Accelerate.framework/Accelerate" else null);
    addOptionalIsolatedBenchLibrary(run_triangular_matrix_report, "--mkl", bench_mkl, null);
    addOptionalIsolatedBenchLibrary(run_triangular_matrix_report, "--aocl-blis", bench_aocl_blis, null);
    run_triangular_matrix_report.addArg("--csv");
    run_triangular_matrix_report.addArg("zig-out/triangular_matrix_report.csv");
    run_triangular_matrix_report.addArg("--skip-missing");
    if (b.args) |args| run_triangular_matrix_report.addArgs(args);

    const triangular_matrix_report_step = b.step("bench-triangular-matrix-report", "Run the opt-in fresh-process TRMM/TRSM comparator report");
    triangular_matrix_report_step.dependOn(&run_triangular_matrix_report.step);

    const rotg_latency_probe = b.addExecutable(.{
        .name = "rotg-latency-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/rotg_latency_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const install_rotg_latency_probe = b.addInstallArtifact(rotg_latency_probe, .{});
    const build_rotg_latency_probe_step = b.step("build-rotg-latency-probe", "Build the opt-in Level 1 ROTG/ROTMG latency probe");
    build_rotg_latency_probe_step.dependOn(&install_rotg_latency_probe.step);

    const run_rotg_latency_report = b.addSystemCommand(&.{
        "python3",
        "bench/tools/run_rotg_latency_report.py",
        "--probe",
    });
    run_rotg_latency_report.addFileArg(rotg_latency_probe.getEmittedBin());
    run_rotg_latency_report.addArg("--zynum");
    run_rotg_latency_report.addFileArg(lib.getEmittedBin());
    addOptionalIsolatedBenchLibrary(run_rotg_latency_report, "--openblas", bench_openblas, if (target.result.os.tag == .macos) "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" else null);
    addOptionalIsolatedBenchLibrary(run_rotg_latency_report, "--accelerate", bench_accelerate, if (target.result.os.tag == .macos) "/System/Library/Frameworks/Accelerate.framework/Accelerate" else null);
    addOptionalIsolatedBenchLibrary(run_rotg_latency_report, "--mkl", bench_mkl, null);
    addOptionalIsolatedBenchLibrary(run_rotg_latency_report, "--aocl-blis", bench_aocl_blis, null);
    run_rotg_latency_report.addArg("--csv");
    run_rotg_latency_report.addArg("zig-out/perf-report/rotg_latency_report.csv");
    run_rotg_latency_report.addArg("--skip-missing");
    if (b.args) |args| run_rotg_latency_report.addArgs(args);

    const rotg_latency_report_step = b.step("bench-rotg-latency-report", "Run the opt-in fresh-process ROTG/ROTMG latency report");
    rotg_latency_report_step.dependOn(&run_rotg_latency_report.step);

    const level1_probe = b.addExecutable(.{
        .name = "level1-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/level1_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.installArtifact(level1_probe);

    const dcopy_probe = b.addExecutable(.{
        .name = "dcopy-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/dcopy_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.installArtifact(dcopy_probe);
}
