// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const TestInventoryProfile = struct {
    environment_id: []const u8,
    enumeration_class_id: []const u8,
};

const ProductionProfile = enum {
    default,
    structured_object,
    level1_sve_candidates,
    level1_fixed_candidates,
    level2_fixed_candidates,
    level2_width_candidates,
};

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
    const target_query = b.standardTargetOptionsQueryOnly(.{});
    const target = b.resolveTargetQuery(target_query);
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });
    const test_optimize = b.option(std.builtin.OptimizeMode, "test-optimize", "Optimize mode for correctness tests") orelse .ReleaseSafe;
    const host_tool_smoke = b.option(bool, "host-tool-smoke", "Run host Python/C/C++/Fortran smoke checks as part of the test step") orelse true;
    const exact_baseline_request = target_query.cpu_model == .baseline and
        target_query.cpu_features_add.isEmpty() and target_query.cpu_features_sub.isEmpty();
    const expected_baseline_cpu = std.Target.Cpu.baseline(target.result.cpu.arch, target.result.os);
    const resolved_cpu_matches_canonical_baseline = target.result.cpu.model == expected_baseline_cpu.model and
        target.result.cpu.features.eql(expected_baseline_cpu.features);
    const inventory_profile: ?TestInventoryProfile = if (exact_baseline_request and
        resolved_cpu_matches_canonical_baseline and
        target.result.cpu.arch == .aarch64 and
        target.result.cpu.model == &std.Target.aarch64.cpu.apple_m1 and
        target.result.os.tag == .macos and target.result.abi == .none and target.result.ofmt == .macho)
        .{
            .environment_id = "env:aarch64-macos-baseline",
            .enumeration_class_id = "enumeration-class:aarch64-macos-system-macho",
        }
    else if (exact_baseline_request and resolved_cpu_matches_canonical_baseline and
        target.result.cpu.arch == .x86_64 and target.result.cpu.model == &std.Target.x86.cpu.x86_64 and
        target.result.os.tag == .linux and target.result.abi == .gnu and target.result.ofmt == .elf)
        .{
            .environment_id = "env:x86-64-linux-gnu-baseline",
            .enumeration_class_id = "enumeration-class:x86-64-linux-gnu-elf",
        }
    else if (exact_baseline_request and resolved_cpu_matches_canonical_baseline and
        target.result.cpu.arch == .aarch64 and target.result.cpu.model == &std.Target.aarch64.cpu.generic and
        target.result.os.tag == .linux and target.result.abi == .gnu and target.result.ofmt == .elf)
        .{
            .environment_id = "env:aarch64-linux-gnu-baseline",
            .enumeration_class_id = "enumeration-class:aarch64-linux-gnu-elf",
        }
    else if (exact_baseline_request and resolved_cpu_matches_canonical_baseline and
        target.result.cpu.arch == .x86_64 and target.result.cpu.model == &std.Target.x86.cpu.x86_64 and
        target.result.os.tag == .windows and target.result.abi == .gnu and target.result.ofmt == .coff)
        .{
            .environment_id = "env:x86-64-windows-gnu-baseline",
            .enumeration_class_id = "enumeration-class:x86-64-windows-gnu-coff",
        }
    else
        null;
    const native_canonical_windows_python_tooling = inventory_profile != null and
        target.result.os.tag == .windows and
        b.graph.host.result.cpu.arch == .x86_64 and
        b.graph.host.result.os.tag == .windows;
    const level1_fixed_candidates = b.option(
        bool,
        "level1-fixed-candidates",
        "Build the BLAS library with experimental fixed-width Level 1 candidates enabled",
    ) orelse false;
    const level1_sve_candidates = b.option(
        bool,
        "level1-sve-candidates",
        "Build the BLAS library with experimental SVE Level 1 candidates enabled",
    ) orelse false;
    const level2_fixed_candidates = b.option(
        bool,
        "level2-fixed-candidates",
        "Build the BLAS library with experimental fixed-width Level 2 candidates enabled",
    ) orelse false;
    const level2_width_candidates = b.option(
        bool,
        "level2-width-candidates",
        "Build the BLAS library with experimental narrow-width x86 Level 2 choices enabled",
    ) orelse false;
    const structured_object_candidates = b.option(
        bool,
        "structured-object-candidates",
        "Build the BLAS library with isolated dense-SYMM and right-triangular candidates enabled",
    ) orelse false;
    const structured_object_baseline = b.option(
        bool,
        "structured-object-baseline",
        "Build the same-ABI structured Level 3 object with its enable byte cleared",
    ) orelse false;
    if (structured_object_candidates and structured_object_baseline) {
        @panic("structured-object-candidates and structured-object-baseline are mutually exclusive");
    }
    const structured_object_requested = structured_object_candidates or structured_object_baseline;
    const structured_object_root = if (structured_object_baseline)
        "src/blas/structured_object_stub_root.zig"
    else
        "src/blas/structured_object_root.zig";
    const level2_compact_triangular_baseline = b.option(
        bool,
        "level2-compact-triangular-baseline",
        "Build the same-ABI compact-triangular control object that always falls back",
    ) orelse false;
    const compact_triangular_object_root = if (level2_compact_triangular_baseline)
        "src/blas/level2_compact_triangular_stub_root.zig"
    else
        "src/blas/level2_compact_triangular_object_root.zig";
    const selected_profile: ProductionProfile = if (structured_object_requested)
        .structured_object
    else if (level1_sve_candidates)
        .level1_sve_candidates
    else if (level1_fixed_candidates)
        .level1_fixed_candidates
    else if (level2_fixed_candidates)
        .level2_fixed_candidates
    else if (level2_width_candidates)
        .level2_width_candidates
    else
        .default;
    const level2_width_selected = selected_profile == .level2_width_candidates;
    const level2_width_object_root = if (level2_width_selected)
        "src/blas/level2_width_object_root.zig"
    else
        "src/blas/level2_width_stub_root.zig";

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
        .root_source_file = b.path(switch (selected_profile) {
            .structured_object => "src/blas/compat_structured_object_candidates.zig",
            .level1_sve_candidates => "src/blas/compat_level1_sve_candidates.zig",
            .level1_fixed_candidates => "src/blas/compat_level1_fixed_candidates.zig",
            .level2_fixed_candidates => "src/blas/compat_level2_fixed_candidates.zig",
            .level2_width_candidates => "src/blas/compat_level2_width_candidates.zig",
            .default => "src/blas/compat.zig",
        }),
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
    const stride2_isolated_library = if (target.result.cpu.arch == .x86_64) b.addLibrary(.{
        .name = "zynum-level1-x86-stride2-isolated",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/level1_stride2_object_root.zig"),
            .target = target,
            .optimize = optimize,
            .pic = true,
        }),
    }) else null;
    const stride2_isolated_test_library = if (target.result.cpu.arch == .x86_64) b.addLibrary(.{
        .name = "zynum-level1-x86-stride2-isolated-test",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/level1_stride2_object_root.zig"),
            .target = target,
            .optimize = test_optimize,
            .pic = true,
        }),
    }) else null;
    const compact_triangular_isolated_library = if (target.result.cpu.arch == .x86_64) b.addLibrary(.{
        .name = "zynum-level2-x86-compact-triangular-isolated",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path(compact_triangular_object_root),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    const compact_triangular_isolated_test_library = if (target.result.cpu.arch == .x86_64) b.addLibrary(.{
        .name = "zynum-level2-x86-compact-triangular-isolated-test",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path(compact_triangular_object_root),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    const level2_width_isolated_library = if (target.result.cpu.arch == .x86_64) b.addLibrary(.{
        .name = "zynum-level2-x86-width-isolated",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path(level2_width_object_root),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    const level2_width_isolated_test_library = if (target.result.cpu.arch == .x86_64) b.addLibrary(.{
        .name = "zynum-level2-x86-width-isolated-test",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/level2_width_object_root.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    const structured_isolated_library = if (target.result.cpu.arch == .x86_64 and structured_object_requested) b.addLibrary(.{
        .name = "zynum-level3-x86-structured-isolated",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path(structured_object_root),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    const structured_isolated_test_library = if (target.result.cpu.arch == .x86_64) b.addLibrary(.{
        .name = "zynum-level3-x86-structured-isolated-test",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/structured_object_root.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    if (stride2_isolated_library) |library| {
        zynum_mod.linkLibrary(library);
        zynum_blas_mod.linkLibrary(library);
        blas_compat_mod.linkLibrary(library);
    }
    if (stride2_isolated_test_library) |library| {
        zynum_test_mod.linkLibrary(library);
        zynum_blas_test_mod.linkLibrary(library);
        fortran_compat_test_mod.linkLibrary(library);
        cblas_compat_test_mod.linkLibrary(library);
    }
    if (compact_triangular_isolated_library) |library| {
        zynum_mod.linkLibrary(library);
        zynum_blas_mod.linkLibrary(library);
        blas_compat_mod.linkLibrary(library);
    }
    if (compact_triangular_isolated_test_library) |library| {
        zynum_test_mod.linkLibrary(library);
        zynum_blas_test_mod.linkLibrary(library);
        fortran_compat_test_mod.linkLibrary(library);
        cblas_compat_test_mod.linkLibrary(library);
    }
    if (level2_width_isolated_library) |library| {
        zynum_mod.linkLibrary(library);
        zynum_blas_mod.linkLibrary(library);
        blas_compat_mod.linkLibrary(library);
    }
    if (level2_width_isolated_test_library) |library| {
        zynum_test_mod.linkLibrary(library);
        zynum_blas_test_mod.linkLibrary(library);
        fortran_compat_test_mod.linkLibrary(library);
        cblas_compat_test_mod.linkLibrary(library);
    }
    if (structured_isolated_library) |library| {
        blas_compat_mod.linkLibrary(library);
    }
    _ = fortran_compat_mod;
    _ = cblas_compat_mod;
    const lib = b.addLibrary(.{
        .name = "zynum_blas",
        .linkage = .dynamic,
        .root_module = blas_compat_mod,
    });
    const install_dynamic_lib = b.addInstallArtifact(lib, .{});

    const static_lib = b.addLibrary(.{
        .name = "zynum_blas",
        .linkage = .static,
        .root_module = blas_compat_mod,
    });
    const static_install_options: std.Build.Step.InstallArtifact.Options = if (target.result.os.tag == .windows)
        .{ .dest_dir = .{ .override = .{ .custom = "lib/static" } } }
    else
        .{};
    const install_static_lib = b.addInstallArtifact(static_lib, static_install_options);
    b.getInstallStep().dependOn(&install_dynamic_lib.step);
    b.getInstallStep().dependOn(&install_static_lib.step);
    const install_libraries_step = b.step(
        "install-libraries",
        "Install the shared and static Zynum BLAS libraries without tools",
    );
    install_libraries_step.dependOn(&install_dynamic_lib.step);
    install_libraries_step.dependOn(&install_static_lib.step);

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

    const kernel_coverage_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/kernel_coverage_root.zig"),
        .target = b.graph.host,
        // The generator materializes every descriptor for reporting. Debug
        // avoids spending release-optimization time on a host-only tool.
        .optimize = .Debug,
    });
    const generate_kernel_coverage_tool = b.addExecutable(.{
        .name = "generate_kernel_coverage",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/generate_kernel_coverage.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
            .imports = &.{.{ .name = "kernel-coverage", .module = kernel_coverage_mod }},
        }),
    });
    const generate_kernel_coverage = b.addRunArtifact(generate_kernel_coverage_tool);
    generate_kernel_coverage.addArg(b.pathFromRoot("."));
    generate_kernel_coverage.addArg("64");
    const generate_kernel_coverage_step = b.step("generate-kernel-coverage", "Regenerate the normalized BLAS kernel coverage artifact");
    generate_kernel_coverage_step.dependOn(&generate_kernel_coverage.step);

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

    const zynum_public_surface_contract_options = b.addOptions();
    zynum_public_surface_contract_options.addOption(bool, "is_top_level", true);
    const zynum_public_surface_contract_tests = b.addTest(.{
        .name = "zynum-public-surface-contract-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/api/public_surface_contract_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "public-surface", .module = zynum_test_mod },
                .{ .name = "public-surface-contract-options", .module = zynum_public_surface_contract_options.createModule() },
            },
        }),
    });
    const run_zynum_public_surface_contract_tests = b.addRunArtifact(zynum_public_surface_contract_tests);

    const blas_public_surface_contract_options = b.addOptions();
    blas_public_surface_contract_options.addOption(bool, "is_top_level", false);
    const blas_public_surface_contract_tests = b.addTest(.{
        .name = "zynum-blas-public-surface-contract-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/api/public_surface_contract_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "public-surface", .module = zynum_blas_test_mod },
                .{ .name = "public-surface-contract-options", .module = blas_public_surface_contract_options.createModule() },
            },
        }),
    });
    const run_blas_public_surface_contract_tests = b.addRunArtifact(blas_public_surface_contract_tests);
    const public_surface_contract_test_step = b.step(
        "test-public-api-contract",
        "Run exhaustive public Zig surface inventory and reflection contracts",
    );
    public_surface_contract_test_step.dependOn(&run_zynum_public_surface_contract_tests.step);
    public_surface_contract_test_step.dependOn(&run_blas_public_surface_contract_tests.step);

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

    const gemm_registry_tests = b.addTest(.{
        .name = "zynum-blas-gemm-registry-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/gemm_registry_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_gemm_registry_tests = b.addRunArtifact(gemm_registry_tests);
    const gemm_registry_test_step = b.step("test-gemm-registry", "Run real GEMM registry and forced-path correctness tests");
    gemm_registry_test_step.dependOn(&run_gemm_registry_tests.step);

    const level1_registry_tests = b.addTest(.{
        .name = "zynum-blas-level1-registry-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/level1_registry_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_level1_registry_tests = b.addRunArtifact(level1_registry_tests);
    const level1_registry_test_step = b.step("test-level1-registry", "Run Level 1 registry and forced-path correctness tests");
    level1_registry_test_step.dependOn(&run_level1_registry_tests.step);

    const level2_fused_registry_tests = b.addTest(.{
        .name = "zynum-blas-level2-fused-registry-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/level2_fused_registry_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_level2_fused_registry_tests = b.addRunArtifact(level2_fused_registry_tests);
    const level2_fused_registry_test_step = b.step("test-level2-fused-registry", "Run Level 2 fused-body registry and forced-path tests");
    level2_fused_registry_test_step.dependOn(&run_level2_fused_registry_tests.step);

    const level2_compact_registry_tests = b.addTest(.{
        .name = "zynum-blas-level2-compact-registry-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/level2_compact_registry_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_level2_compact_registry_tests = b.addRunArtifact(level2_compact_registry_tests);
    const level2_compact_registry_test_step = b.step("test-level2-compact-registry", "Run compact packed/banded registry and forced-path tests");
    level2_compact_registry_test_step.dependOn(&run_level2_compact_registry_tests.step);

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

    const structured_blocked_tests = b.addTest(.{
        .name = "zynum-blas-structured-blocked-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/structured_blocked_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    const run_structured_blocked_tests = b.addRunArtifact(structured_blocked_tests);
    const structured_blocked_test_step = b.step("test-structured-blocked", "Run blocked structured Level 3 packing and update tests");
    structured_blocked_test_step.dependOn(&run_structured_blocked_tests.step);

    const structured_object_test_step = b.step("test-structured-object", "Run isolated x86 structured Level 3 ABI and correctness tests");
    const structured_object_tests = if (target.result.cpu.arch == .x86_64) object_tests: {
        const structured_object_test_mod = b.createModule(.{
            .root_source_file = b.path("src/blas/structured_object_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        });
        structured_object_test_mod.linkLibrary(structured_isolated_test_library.?);
        const structured_object_tests = b.addTest(.{
            .name = "zynum-blas-structured-object-tests",
            .root_module = structured_object_test_mod,
        });
        break :object_tests structured_object_tests;
    } else null;
    const run_structured_object_tests = if (structured_object_tests) |tests| object_tests: {
        const run = b.addRunArtifact(tests);
        structured_object_test_step.dependOn(&run.step);
        break :object_tests run;
    } else null;

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

    const triangular_packed_unit_test_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/triangular_packed_unit_test.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    if (compact_triangular_isolated_test_library) |library| triangular_packed_unit_test_mod.linkLibrary(library);
    const triangular_packed_unit_tests = b.addTest(.{
        .name = "zynum-blas-triangular-packed-unit-tests",
        .root_module = triangular_packed_unit_test_mod,
    });
    const run_triangular_packed_unit_tests = b.addRunArtifact(triangular_packed_unit_tests);

    const triangular_band_solve_test_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/triangular_band_solve_test.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    if (compact_triangular_isolated_test_library) |library| triangular_band_solve_test_mod.linkLibrary(library);
    const triangular_band_solve_tests = b.addTest(.{
        .name = "zynum-blas-triangular-band-solve-tests",
        .root_module = triangular_band_solve_test_mod,
    });
    const run_triangular_band_solve_tests = b.addRunArtifact(triangular_band_solve_tests);
    level2_compact_registry_test_step.dependOn(&run_packed_parallel_tests.step);
    level2_compact_registry_test_step.dependOn(&run_triangular_band_window_tests.step);
    level2_compact_registry_test_step.dependOn(&run_triangular_packed_unit_tests.step);
    level2_compact_registry_test_step.dependOn(&run_triangular_band_solve_tests.step);

    const vector_stride2_parallel_test_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/vector_stride2_parallel_test.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    if (stride2_isolated_test_library) |library| vector_stride2_parallel_test_mod.linkLibrary(library);
    const vector_stride2_parallel_tests = b.addTest(.{
        .name = "zynum-blas-vector-stride2-parallel-tests",
        .root_module = vector_stride2_parallel_test_mod,
    });
    const run_vector_stride2_parallel_tests = b.addRunArtifact(vector_stride2_parallel_tests);
    const vector_stride2_parallel_test_step = b.step("test-level1-stride2-isolated", "Run isolated Level 1 stride-two bridge tests");
    vector_stride2_parallel_test_step.dependOn(&run_vector_stride2_parallel_tests.step);

    const level2_width_default_artifact_probe_mod = if (target.result.cpu.arch == .x86_64 and selected_profile == .default)
        b.createModule(.{
            .root_source_file = b.path("test/build/level2_width_default_artifact_probe.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        })
    else
        null;
    if (level2_width_default_artifact_probe_mod) |probe_mod| {
        probe_mod.linkLibrary(level2_width_isolated_library.?);
    }
    const level2_width_default_artifact_probe = if (level2_width_default_artifact_probe_mod) |probe_mod|
        b.addExecutable(.{
            .name = "zynum-level2-width-default-artifact-probe",
            .root_module = probe_mod,
        })
    else
        null;
    if (level2_width_default_artifact_probe) |probe| probe.step.dependOn(&lib.step);
    const run_level2_width_default_artifact_probe = if (level2_width_default_artifact_probe) |probe|
        b.addRunArtifact(probe)
    else
        null;
    const level2_width_default_artifact_probe_step = b.step(
        "test-level2-width-default-artifact",
        "Run the default x86 Level 2 width production-artifact probe",
    );
    if (run_level2_width_default_artifact_probe) |run| level2_width_default_artifact_probe_step.dependOn(&run.step);

    const target_has_avx512f = target.result.cpu.arch == .x86_64 and
        target.result.cpu.features.isEnabled(@intFromEnum(std.Target.x86.Feature.avx512f));
    const level2_width_enabled_artifact_probe_mod = if (target_has_avx512f and level2_width_selected)
        b.createModule(.{
            .root_source_file = b.path("test/build/level2_width_enabled_artifact_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        })
    else
        null;
    if (level2_width_enabled_artifact_probe_mod) |probe_mod| {
        probe_mod.linkLibrary(level2_width_isolated_library.?);
    }
    const level2_width_enabled_artifact_probe = if (level2_width_enabled_artifact_probe_mod) |probe_mod|
        b.addExecutable(.{
            .name = "zynum-level2-width-enabled-artifact-probe",
            .root_module = probe_mod,
        })
    else
        null;
    const build_level2_width_enabled_artifact_probe_step = b.step(
        "build-level2-width-enabled-artifact",
        "Compile the enabled x86 Level 2 width production-artifact probe",
    );
    const run_level2_width_enabled_artifact_probe_step = b.step(
        "test-level2-width-enabled-artifact",
        "Run the enabled x86 Level 2 width production-artifact probe on AVX-512 hardware",
    );
    if (level2_width_enabled_artifact_probe) |probe| {
        build_level2_width_enabled_artifact_probe_step.dependOn(&probe.step);
        const run_probe = b.addRunArtifact(probe);
        run_level2_width_enabled_artifact_probe_step.dependOn(&run_probe.step);
    } else {
        const unsupported_probe = b.addFail(
            "the enabled Level 2 width artifact probe requires an x86_64 AVX-512 target and -Dlevel2-width-candidates=true",
        );
        build_level2_width_enabled_artifact_probe_step.dependOn(&unsupported_probe.step);
        run_level2_width_enabled_artifact_probe_step.dependOn(&unsupported_probe.step);
    }

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
    const abi_baseline_observer_tests = b.addSystemCommand(&.{
        "python3",
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        "test/abi/baseline",
        "-p",
        "test_*.py",
    });
    const abi_baseline_observer_test_step = b.step(
        "test-abi-baseline-observer",
        "Run whole-artifact ABI baseline observer tests",
    );
    abi_baseline_observer_test_step.dependOn(&abi_baseline_observer_tests.step);
    const abi_artifact_parity_verifier_tests = b.addSystemCommand(&.{
        "python3",
        "-B",
        b.pathFromRoot("test/abi/baseline/test_abi_artifact_parity.py"),
    });
    const abi_artifact_parity_verifier_test_step = b.step(
        "test-abi-artifact-parity-verifier",
        "Run fresh-artifact parity parser and verifier tests",
    );
    abi_artifact_parity_verifier_test_step.dependOn(&abi_artifact_parity_verifier_tests.step);
    const build_inventory_tests = b.addSystemCommand(&.{
        "python3",
        "-B",
        b.pathFromRoot("test/build/test_build_inventory.py"),
    });
    const build_inventory_test_step = b.step(
        "test-build-inventory",
        "Validate the repository build and launch inventory",
    );
    build_inventory_test_step.dependOn(&build_inventory_tests.step);
    const test_inventory_structure_check = b.addSystemCommand(&.{
        "python3",
        "-B",
        b.pathFromRoot("tools/check_test_inventory.py"),
        "--root",
        b.pathFromRoot("."),
        "--structure-only",
    });
    test_inventory_structure_check.removeEnvironmentVariable("GIT_PAGER");
    test_inventory_structure_check.removeEnvironmentVariable("PAGER");
    const test_inventory_security_tests = b.addSystemCommand(&.{
        "python3",
        "-B",
        b.pathFromRoot("test/build/test_test_inventory.py"),
    });
    test_inventory_security_tests.removeEnvironmentVariable("GIT_PAGER");
    test_inventory_security_tests.removeEnvironmentVariable("PAGER");
    const test_inventory_security_test_step = b.step(
        "test-test-inventory",
        "Run the complete test-inventory security regression suite",
    );
    test_inventory_security_test_step.dependOn(&test_inventory_security_tests.step);
    const python_tooling_tests = b.addSystemCommand(&.{
        "python3",
        "-B",
        b.pathFromRoot("tools/check_test_inventory.py"),
        "--root",
        b.pathFromRoot("."),
        "--inventory",
        b.pathFromRoot("tools/test_inventory.json"),
        "--run-python-tooling-root",
        "python-root:benchmark-tools-discovery",
    });
    python_tooling_tests.setCwd(b.path("."));
    python_tooling_tests.removeEnvironmentVariable("PYTHONHOME");
    python_tooling_tests.removeEnvironmentVariable("PYTHONPATH");
    python_tooling_tests.removeEnvironmentVariable("PYTHONINSPECT");
    python_tooling_tests.removeEnvironmentVariable("PYTHONSTARTUP");
    python_tooling_tests.removeEnvironmentVariable("GIT_PAGER");
    python_tooling_tests.removeEnvironmentVariable("PAGER");
    python_tooling_tests.removeEnvironmentVariable("LESS");
    python_tooling_tests.step.dependOn(&test_inventory_structure_check.step);
    const python_tooling_test_step = b.step(
        "test-python-tooling",
        "Run inventory-declared Python benchmark tooling unit tests",
    );
    python_tooling_test_step.dependOn(&python_tooling_tests.step);

    const InventoryCase = struct {
        root_id: []const u8,
        logical_tests: ?*std.Build.Step.Compile,
    };
    const inventory_cases = [21]InventoryCase{
        .{ .root_id = "zig-root:blas-module-tests", .logical_tests = blas_module_tests },
        .{ .root_id = "zig-root:blas-public-surface-contract-tests", .logical_tests = blas_public_surface_contract_tests },
        .{ .root_id = "zig-root:cblas-tests", .logical_tests = cblas_tests },
        .{ .root_id = "zig-root:fortran-tests", .logical_tests = fortran_tests },
        .{ .root_id = "zig-root:gemm-registry-tests", .logical_tests = gemm_registry_tests },
        .{ .root_id = "zig-root:header-smoke-tests", .logical_tests = header_smoke_tests },
        .{ .root_id = "zig-root:level1-registry-tests", .logical_tests = level1_registry_tests },
        .{ .root_id = "zig-root:level2-compact-registry-tests", .logical_tests = level2_compact_registry_tests },
        .{ .root_id = "zig-root:level2-fused-registry-tests", .logical_tests = level2_fused_registry_tests },
        .{ .root_id = "zig-root:modern-tests", .logical_tests = modern_tests },
        .{ .root_id = "zig-root:packed-parallel-tests", .logical_tests = packed_parallel_tests },
        .{ .root_id = "zig-root:structured-blocked-tests", .logical_tests = structured_blocked_tests },
        .{ .root_id = "zig-root:structured-object-tests", .logical_tests = structured_object_tests },
        .{ .root_id = "zig-root:symm-dense-gemm-tests", .logical_tests = symm_dense_gemm_tests },
        .{ .root_id = "zig-root:triangular-band-solve-tests", .logical_tests = triangular_band_solve_tests },
        .{ .root_id = "zig-root:triangular-band-window-tests", .logical_tests = triangular_band_window_tests },
        .{ .root_id = "zig-root:triangular-dense-unit-tests", .logical_tests = triangular_dense_unit_tests },
        .{ .root_id = "zig-root:triangular-packed-unit-tests", .logical_tests = triangular_packed_unit_tests },
        .{ .root_id = "zig-root:triangular-parallel-tests", .logical_tests = triangular_parallel_tests },
        .{ .root_id = "zig-root:vector-stride2-parallel-tests", .logical_tests = vector_stride2_parallel_tests },
        .{ .root_id = "zig-root:zynum-public-surface-contract-tests", .logical_tests = zynum_public_surface_contract_tests },
    };
    const test_inventory_link_step = b.step(
        "test-inventory-link",
        "Compile every applicable test-inventory enumerator without running it",
    );
    const test_inventory_step = b.step(
        "test-inventory",
        "Run and verify the exact native test inventory without executing test bodies",
    );
    test_inventory_link_step.dependOn(&test_inventory_structure_check.step);
    test_inventory_step.dependOn(&test_inventory_structure_check.step);
    if (inventory_profile) |resolved_inventory_profile| {
        for (inventory_cases) |inventory_case| {
            const official_tests = inventory_case.logical_tests orelse continue;
            const inventory_tests = b.addTest(.{
                .name = b.fmt("inventory-{s}", .{inventory_case.root_id}),
                .root_module = official_tests.root_module,
                .test_runner = .{
                    .path = b.path("tools/test_inventory_runner.zig"),
                    .mode = .simple,
                },
            });
            const run_inventory_tests = b.addRunArtifact(inventory_tests);
            inventory_tests.step.dependOn(&test_inventory_structure_check.step);
            run_inventory_tests.step.dependOn(&test_inventory_structure_check.step);
            run_inventory_tests.addFileArg(b.path("tools/test_inventory.json"));
            run_inventory_tests.addArgs(&.{
                "--inventory-root",
                inventory_case.root_id,
                "--inventory-mode",
                @tagName(test_optimize),
                "--inventory-environment",
                resolved_inventory_profile.environment_id,
                "--inventory-class",
                resolved_inventory_profile.enumeration_class_id,
            });
            test_inventory_link_step.dependOn(&inventory_tests.step);
            test_inventory_step.dependOn(&run_inventory_tests.step);
        }
    } else {
        const unsupported_test_inventory_target = b.addFail(
            "test inventory enumeration is unavailable for the requested target CPU profile",
        );
        test_inventory_link_step.dependOn(&unsupported_test_inventory_target.step);
        test_inventory_step.dependOn(&unsupported_test_inventory_target.step);
    }

    const test_native_feature_step = b.step(
        "test-native-feature",
        "Run correctness-only tests for an explicit host-supported non-baseline CPU profile; not inventory evidence",
    );
    test_native_feature_step.dependOn(&test_inventory_structure_check.step);
    const explicit_non_baseline_cpu_profile = switch (target_query.cpu_model) {
        .native, .explicit => true,
        .baseline, .determined_by_arch_os => false,
    };
    const native_feature_target_matches_host = target.result.cpu.arch == b.graph.host.result.cpu.arch and
        target.result.os.tag == b.graph.host.result.os.tag and
        target.result.abi == b.graph.host.result.abi and
        target.result.ofmt == b.graph.host.result.ofmt;
    const native_feature_external_executor_enabled = b.enable_qemu or
        b.enable_rosetta or b.enable_wine or b.enable_darling or b.enable_wasmtime;
    const native_feature_profile_guard: *std.Build.Step = if (!explicit_non_baseline_cpu_profile)
        &b.addFail("test-native-feature requires an explicit non-baseline CPU profile").step
    else if (!native_feature_target_matches_host)
        &b.addFail("test-native-feature requires target arch/os/abi/ofmt to match the build host exactly").step
    else if (native_feature_external_executor_enabled)
        &b.addFail("test-native-feature forbids external target executors").step
    else if (!b.graph.host.result.cpu.features.isSuperSetOf(target.result.cpu.features))
        &b.addFail("test-native-feature requested CPU features are not supported by the build host").step
    else
        &test_inventory_structure_check.step;
    for (inventory_cases) |inventory_case| {
        const official_tests = inventory_case.logical_tests orelse continue;
        const run_native_feature_tests = b.addRunArtifact(official_tests);
        run_native_feature_tests.step.dependOn(native_feature_profile_guard);
        test_native_feature_step.dependOn(&run_native_feature_tests.step);
    }

    run_blas_module_tests.step.dependOn(test_inventory_step);
    run_blas_public_surface_contract_tests.step.dependOn(test_inventory_step);
    run_cblas_tests.step.dependOn(test_inventory_step);
    run_fortran_tests.step.dependOn(test_inventory_step);
    run_gemm_registry_tests.step.dependOn(test_inventory_step);
    run_header_smoke_tests.step.dependOn(test_inventory_step);
    run_level1_registry_tests.step.dependOn(test_inventory_step);
    run_level2_compact_registry_tests.step.dependOn(test_inventory_step);
    run_level2_fused_registry_tests.step.dependOn(test_inventory_step);
    run_modern_tests.step.dependOn(test_inventory_step);
    run_packed_parallel_tests.step.dependOn(test_inventory_step);
    run_structured_blocked_tests.step.dependOn(test_inventory_step);
    if (run_structured_object_tests) |run| run.step.dependOn(test_inventory_step);
    run_symm_dense_gemm_tests.step.dependOn(test_inventory_step);
    run_triangular_band_solve_tests.step.dependOn(test_inventory_step);
    run_triangular_band_window_tests.step.dependOn(test_inventory_step);
    run_triangular_dense_unit_tests.step.dependOn(test_inventory_step);
    run_triangular_packed_unit_tests.step.dependOn(test_inventory_step);
    run_triangular_parallel_tests.step.dependOn(test_inventory_step);
    run_vector_stride2_parallel_tests.step.dependOn(test_inventory_step);
    run_zynum_public_surface_contract_tests.step.dependOn(test_inventory_step);

    const test_step = b.step("test", "Run correctness tests");
    test_step.dependOn(&test_inventory_structure_check.step);
    test_step.dependOn(test_inventory_step);
    test_step.dependOn(python_tooling_test_step);
    test_step.dependOn(&run_modern_tests.step);
    test_step.dependOn(&run_blas_module_tests.step);
    test_step.dependOn(&run_zynum_public_surface_contract_tests.step);
    test_step.dependOn(&run_blas_public_surface_contract_tests.step);
    test_step.dependOn(&run_fortran_tests.step);
    test_step.dependOn(&run_cblas_tests.step);
    test_step.dependOn(&run_gemm_registry_tests.step);
    test_step.dependOn(&run_level1_registry_tests.step);
    test_step.dependOn(&run_level2_fused_registry_tests.step);
    test_step.dependOn(&run_level2_compact_registry_tests.step);
    test_step.dependOn(&run_symm_dense_gemm_tests.step);
    test_step.dependOn(&run_triangular_parallel_tests.step);
    test_step.dependOn(&run_structured_blocked_tests.step);
    if (run_structured_object_tests) |run| test_step.dependOn(&run.step);
    test_step.dependOn(&run_packed_parallel_tests.step);
    test_step.dependOn(&run_triangular_dense_unit_tests.step);
    test_step.dependOn(&run_triangular_band_window_tests.step);
    test_step.dependOn(&run_triangular_packed_unit_tests.step);
    test_step.dependOn(&run_triangular_band_solve_tests.step);
    test_step.dependOn(&run_vector_stride2_parallel_tests.step);
    if (run_level2_width_default_artifact_probe != null) test_step.dependOn(level2_width_default_artifact_probe_step);
    test_step.dependOn(&run_header_smoke_tests.step);
    if (host_tool_smoke) {
        test_step.dependOn(&abi_manifest_smoke_test.step);
        test_step.dependOn(&c_header_smoke_test.step);
        test_step.dependOn(&cpp_header_smoke_test.step);
        test_step.dependOn(&fortran_module_smoke_test.step);
        test_step.dependOn(&abi_baseline_observer_tests.step);
        test_step.dependOn(&build_inventory_tests.step);
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
    if (target.result.os.tag != .windows) b.installArtifact(bench);

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
    if (target.result.os.tag != .windows) b.installArtifact(gemm_sweep);

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
    if (target.result.os.tag != .windows) b.installArtifact(vector_matrix_sweep);

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
            .root_source_file = b.path(if (target.result.os.tag == .windows)
                "test/build/windows_python_tooling_probe_fixture.zig"
            else
                "bench/rank_k_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const install_rank_k_probe = b.addInstallArtifact(rank_k_probe, .{});
    const build_rank_k_probe_step = b.step(
        "build-rank-k-probe",
        if (target.result.os.tag == .windows)
            "Build the Windows Python tooling executable fixture; not benchmark runtime evidence"
        else
            "Build the opt-in Level 3 rank-k probe",
    );
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
            .root_source_file = b.path(if (target.result.os.tag == .windows)
                "test/build/windows_python_tooling_probe_fixture.zig"
            else
                "bench/symm_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const install_symm_probe = b.addInstallArtifact(symm_probe, .{});
    const build_symm_probe_step = b.step(
        "build-symm-probe",
        if (target.result.os.tag == .windows)
            "Build the Windows Python tooling executable fixture; not benchmark runtime evidence"
        else
            "Build the opt-in Level 3 SYMM/HEMM probe",
    );
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
            .root_source_file = b.path(if (target.result.os.tag == .windows)
                "test/build/windows_python_tooling_probe_fixture.zig"
            else
                "bench/triangular_matrix_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const install_triangular_matrix_probe = b.addInstallArtifact(triangular_matrix_probe, .{});
    const build_triangular_matrix_probe_step = b.step(
        "build-triangular-matrix-probe",
        if (target.result.os.tag == .windows)
            "Build the Windows Python tooling executable fixture; not benchmark runtime evidence"
        else
            "Build the opt-in Level 3 TRMM/TRSM probe",
    );
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
            .root_source_file = b.path(if (target.result.os.tag == .windows)
                "test/build/windows_python_tooling_probe_fixture.zig"
            else
                "bench/rotg_latency_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const install_rotg_latency_probe = b.addInstallArtifact(rotg_latency_probe, .{});
    const build_rotg_latency_probe_step = b.step(
        "build-rotg-latency-probe",
        if (target.result.os.tag == .windows)
            "Build the Windows Python tooling executable fixture; not benchmark runtime evidence"
        else
            "Build the opt-in Level 1 ROTG/ROTMG latency probe",
    );
    build_rotg_latency_probe_step.dependOn(&install_rotg_latency_probe.step);

    if (native_canonical_windows_python_tooling) {
        python_tooling_tests.addArg("--windows-zynum-blas-build-output");
        python_tooling_tests.addFileArg(lib.getEmittedBin());
        python_tooling_tests.addArg("--windows-zynum-blas-installed-output");
        python_tooling_tests.addArg(b.getInstallPath(.bin, "zynum_blas.dll"));
        python_tooling_tests.step.dependOn(&install_dynamic_lib.step);
        python_tooling_tests.step.dependOn(&install_rank_k_probe.step);
        python_tooling_tests.step.dependOn(&install_rotg_latency_probe.step);
        python_tooling_tests.step.dependOn(&install_symm_probe.step);
        python_tooling_tests.step.dependOn(&install_triangular_matrix_probe.step);
    }

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
    if (target.result.os.tag != .windows) b.installArtifact(level1_probe);

    const dcopy_probe = b.addExecutable(.{
        .name = "dcopy-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/dcopy_probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    if (target.result.os.tag != .windows) b.installArtifact(dcopy_probe);
}
