// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");

const DispatchMode = enum { auto, dynamic, specialized };

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

pub const FFTInterfaceProfile = enum { none, fftw3 };
pub const OperatorInterfaceProfile = enum { none, zynum_v1 };
pub const ContourInterfaceProfile = enum { none, zynum_v1 };
pub const BLASInterfaceProfile = enum { none, lp64 };
pub const FEASTInterfaceProfile = enum { none, feast4_lp64_abplas };

/// Existing production objects from one configuration, never replacement owners.
pub const ImplementationGraph = struct {
    core: *std.Build.Module,
    execution: *std.Build.Module,
    resources: *std.Build.Module,
    options: *std.Build.Module,
    lapacke_binding: *std.Build.Module,
    mkl_sparse_binding: *std.Build.Module,
};

pub fn build(b: *std.Build) void {
    _ = configure(b);
}

/// Composable single-call configuration; callers must not also invoke build().
pub fn configure(b: *std.Build) ImplementationGraph {
    const blas_interface = b.option(BLASInterfaceProfile, "blas-interface", "Optional existing LP64 CBLAS and Fortran BLAS frontend for the unified package") orelse .none;
    const fft_interface = b.option(FFTInterfaceProfile, "fft-interface", "Optional FFT frontend for the unified LP64 package: none or FFTW3 f64 ABPLaS profile") orelse .none;
    const fft_f32 = b.option(bool, "fft-f32", "Include the optional FFTW single-precision companion over the same unified Core") orelse false;
    if (fft_f32 and fft_interface != .fftw3) @panic("-Dfft-f32=true requires -Dfft-interface=fftw3");
    const checked_complex_cholesky = b.option(bool, "checked-complex-cholesky", "Include additive complex checked Cholesky in checked/Sparse and unified LP64 packages") orelse false;
    const lapacke_qr = b.option(bool, "lapacke-qr", "Include generated LP64 GEQRF/ORMQR/UNMQR/ORGQR/UNGQR in the finite LAPACKE and unified packages") orelse false;
    const lapacke_complex_gelsd = b.option(bool, "lapacke-complex-gelsd", "Include generated finite LP64 CGELSD/ZGELSD high/work/query in the LAPACKE and unified packages") orelse false;
    const lapacke_svd = b.option(bool, "lapacke-svd", "Include generated finite LP64 SGESVD/DGESVD/CGESVD/ZGESVD high/work/query in the LAPACKE and unified packages") orelse false;
    const lapacke_posv = b.option(bool, "lapacke-posv", "Include generated four-scalar POSV high/work in finite LP64/ILP64 and unified LP64 packages") orelse false;
    const lapacke_trtrs = b.option(bool, "lapacke-trtrs", "Include generated four-scalar triangular solve high/work in finite LP64/ILP64 and unified LP64 packages") orelse false;
    const lapacke_trtri = b.option(bool, "lapacke-trtri", "Include generated four-scalar triangular inverse high/work in finite LP64/ILP64 and unified LP64 packages") orelse false;
    const lapacke_getri = b.option(bool, "lapacke-getri", "Include generated four-scalar LU inverse high/work/query in finite LP64/ILP64 and unified LP64 packages") orelse false;
    const lapacke_potri = b.option(bool, "lapacke-potri", "Compose generated four-scalar Cholesky inverse high/work over the same LAPACKE owner") orelse false;
    const lapacke_syev = b.option(bool, "lapacke-syev", "Compose generated LP64 real symmetric eigenvalue high/work entries over the same LAPACKE owner") orelse false;
    const lapacke_sygv = b.option(bool, "lapacke-sygv", "Compose generated LP64 generalized real symmetric eigenvalue high/work entries over the same LAPACKE owner") orelse false;
    const lapacke_controls = b.option(bool, "lapacke-controls", "Compose standard NaN checking controls over the exact existing LAPACKE binding") orelse false;
    const lapacke_evd = b.option(bool, "lapacke-evd", "Compose generated LP64 real and complex EVD entries over the existing LAPACKE owner") orelse false;
    const lapacke_gvd = b.option(bool, "lapacke-gvd", "Compose generated LP64 generalized real and complex EVD entries over the existing LAPACKE owner") orelse false;
    const lapacke_gels = b.option(bool, "lapacke-gels", "Compose generated LP64 four-scalar full-rank least-squares high/work/query entries over the existing LAPACKE owner") orelse false;
    const lapacke_gelss = b.option(bool, "lapacke-gelss", "Compose generated LP64 four-scalar SVD minimum-norm least-squares high/work/query entries over the existing LAPACKE owner") orelse false;
    const lapacke_lp64_base_set = if (lapacke_svd) (if (lapacke_complex_gelsd) (if (lapacke_qr) "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-gesvd-ordinary-lp64-v1" else "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-gesvd-ordinary-lp64-v1") else (if (lapacke_qr) "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-gesvd-ordinary-lp64-v1" else "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-gesvd-ordinary-lp64-v1")) else (if (lapacke_complex_gelsd) (if (lapacke_qr) "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-complex-gelsd-ordinary-lp64-v1" else "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-complex-gelsd-ordinary-lp64-v1") else (if (lapacke_qr) "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-geqrf-ormqr-unmqr-orgqr-ungqr-ordinary-lp64-v1" else "potrf-potrs-getrf-getrs-gesv-stev-gelsd-heev-hegv-ordinary-lp64-v1"));
    const lapacke_lp64_before_trtrs_set = if (lapacke_posv)
        b.fmt("{s}-posv-ordinary-lp64-v1", .{lapacke_lp64_base_set[0 .. lapacke_lp64_base_set.len - "-ordinary-lp64-v1".len]})
    else
        lapacke_lp64_base_set;
    const lapacke_lp64_before_trtri_set = if (lapacke_trtrs)
        b.fmt("{s}-trtrs-ordinary-lp64-v1", .{lapacke_lp64_before_trtrs_set[0 .. lapacke_lp64_before_trtrs_set.len - "-ordinary-lp64-v1".len]})
    else
        lapacke_lp64_before_trtrs_set;
    const lapacke_lp64_before_getri_set = if (lapacke_trtri)
        b.fmt("{s}-trtri-ordinary-lp64-v1", .{lapacke_lp64_before_trtri_set[0 .. lapacke_lp64_before_trtri_set.len - "-ordinary-lp64-v1".len]})
    else
        lapacke_lp64_before_trtri_set;
    const lapacke_lp64_set = if (lapacke_getri)
        b.fmt("{s}-getri-ordinary-lp64-v1", .{lapacke_lp64_before_getri_set[0 .. lapacke_lp64_before_getri_set.len - "-ordinary-lp64-v1".len]})
    else
        lapacke_lp64_before_getri_set;
    const lapacke_ilp64_before_trtrs_set = if (lapacke_posv) "potrf-potrs-getrf-getrs-gesv-stev-gelsd-posv-ordinary-ilp64-v1" else "potrf-potrs-getrf-getrs-gesv-stev-gelsd-ordinary-ilp64-v1";
    const lapacke_ilp64_before_trtri_set = if (lapacke_trtrs)
        b.fmt("{s}-trtrs-ordinary-ilp64-v1", .{lapacke_ilp64_before_trtrs_set[0 .. lapacke_ilp64_before_trtrs_set.len - "-ordinary-ilp64-v1".len]})
    else
        lapacke_ilp64_before_trtrs_set;
    const lapacke_ilp64_before_getri_set = if (lapacke_trtri)
        b.fmt("{s}-trtri-ordinary-ilp64-v1", .{lapacke_ilp64_before_trtri_set[0 .. lapacke_ilp64_before_trtri_set.len - "-ordinary-ilp64-v1".len]})
    else
        lapacke_ilp64_before_trtri_set;
    const lapacke_ilp64_set = if (lapacke_getri)
        b.fmt("{s}-getri-ordinary-ilp64-v1", .{lapacke_ilp64_before_getri_set[0 .. lapacke_ilp64_before_getri_set.len - "-ordinary-ilp64-v1".len]})
    else
        lapacke_ilp64_before_getri_set;
    const sparse_solves = b.option(bool, "sparse-solves", "Include CSR/CSC TRSV/TRSM in finite LP64/ILP64 Sparse and unified LP64 packages") orelse false;
    const sparse_lp64_set = if (sparse_solves) "sparse-csr-csc-mv-mm-trsv-trsm-ordinary-lp64-v1" else "sparse-csr-csc-mv-mm-ordinary-lp64-v1";
    const sparse_ilp64_set = if (sparse_solves) "sparse-csr-csc-mv-mm-trsv-trsm-ordinary-ilp64-v1" else "sparse-csr-csc-mv-mm-ordinary-ilp64-v1";
    const feast_interface = b.option(FEASTInterfaceProfile, "feast-interface", "Optional finite FEAST4 LP64 ABPLaS CSR frontend over the unified Core") orelse .none;
    const operator_interface = b.option(OperatorInterfaceProfile, "operator-interface", "Optional generated synchronous callback frontend for the unified LP64 package") orelse .none;
    const operator_phase_agreement = b.option(bool, "operator-phase-agreement", "Include additional generated synchronous distributed phase agreement entries") orelse false;
    const contour_interface = b.option(ContourInterfaceProfile, "contour-interface", "Optional generated neutral Contour extension over the agreed operator-v1 owner") orelse .none;
    if (operator_phase_agreement and operator_interface == .none) {
        std.log.err("operator-phase-agreement requires -Doperator-interface=zynum_v1", .{});
        b.invalid_user_input = true;
    }
    if (contour_interface != .none and (operator_interface != .zynum_v1 or !operator_phase_agreement)) {
        std.log.err("contour-interface=zynum_v1 requires operator-interface=zynum_v1 and operator-phase-agreement=true", .{});
        b.invalid_user_input = true;
    }
    var target_query = b.standardTargetOptionsQueryOnly(.{ .default_target = .{ .cpu_model = .baseline } });
    const optimize = b.option(std.lang.Optimize, "optimize", "Explicit optimization mode (also accepted from Zig dependencies)") orelse
        b.standardOptimizeOption(.{ .preferred_optimize_mode = .fast });
    const dispatch_mode = b.option(DispatchMode, "dispatch", "auto: dynamic by default, specialized when -Dcpu is explicit; or select dynamic/specialized") orelse .auto;
    const explicit_cpu = b.user_input_options.contains("cpu");
    const dynamic_dispatch = switch (dispatch_mode) {
        .auto => !explicit_cpu,
        .dynamic => true,
        .specialized => false,
    };
    if (dynamic_dispatch) {
        // Dependencies forward a synthetic CPU option with ResolvedTarget. An
        // explicit dynamic request must still compile the resolver at baseline.
        target_query.cpu_model = .baseline;
        target_query.cpu_features_add = .empty;
        target_query.cpu_features_sub = .empty;
    }
    const target = b.resolveTargetQuery(target_query);
    if (lapacke_syev and (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)) {
        std.log.err("lapacke-syev requires the reviewed Apple arm64 LP64 C ABI target", .{});
        b.invalid_user_input = true;
    }
    if (lapacke_sygv and (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)) {
        std.log.err("lapacke-sygv requires the reviewed Apple arm64 LP64 C ABI target", .{});
        b.invalid_user_input = true;
    }
    if (lapacke_evd and (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)) {
        std.log.err("lapacke-evd requires the reviewed Apple arm64 LP64 C ABI target", .{});
        b.invalid_user_input = true;
    }
    if (lapacke_gvd and (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)) {
        std.log.err("lapacke-gvd requires the reviewed Apple arm64 LP64 C ABI target", .{});
        b.invalid_user_input = true;
    }
    if (lapacke_gels and (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)) {
        std.log.err("lapacke-gels requires the reviewed Apple arm64 LP64 C ABI target", .{});
        b.invalid_user_input = true;
    }
    if (lapacke_gelss and (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)) {
        std.log.err("lapacke-gelss requires the reviewed Apple arm64 LP64 C ABI target", .{});
        b.invalid_user_input = true;
    }
    if (lapacke_controls and (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)) {
        std.log.err("lapacke-controls requires the reviewed Apple arm64 C int ABI target", .{});
        b.invalid_user_input = true;
    }
    if (contour_interface != .none and (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)) {
        std.log.err("contour-interface=zynum_v1 requires the reviewed Apple arm64 C ABI target", .{});
        b.invalid_user_input = true;
    }
    if (feast_interface != .none and (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)) {
        std.log.err("feast-interface=feast4_lp64_abplas requires the unified Apple arm64 LP64 target", .{});
        b.invalid_user_input = true;
    }
    if (checked_complex_cholesky and (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)) {
        @panic("-Dchecked-complex-cholesky=true requires the reviewed Apple arm64 interface target");
    }
    const thread_limit = b.option(usize, "thread-limit", "Compile-time ceiling for worker threads; zero uses runtime hardware capacity") orelse 0;
    const test_optimize = b.option(std.lang.Optimize, "test-optimize", "Optimize mode for correctness tests") orelse .safe;
    const host_tool_smoke = b.option(bool, "host-tool-smoke", "Include the test-host-tool-smoke aggregate in the default test step") orelse true;
    const apple_amx = b.option(
        bool,
        "apple-amx",
        "Enable the experimental private Apple AMX ISA on a validated AArch64 macOS deployment",
    ) orelse false;
    if (apple_amx and (target.result.cpu.arch != .aarch64 or target.result.os.tag != .macos)) {
        @panic("-Dapple-amx=true requires an aarch64-macos target");
    }
    const zynum_build_options = b.addOptions();
    zynum_build_options.addOption(bool, "apple_amx", apple_amx);
    zynum_build_options.addOption(bool, "dynamic_dispatch", dynamic_dispatch);
    zynum_build_options.addOption(usize, "thread_limit", thread_limit);
    const zynum_build_options_mod = zynum_build_options.createModule();
    const exact_baseline_request = explicit_cpu and !dynamic_dispatch and target_query.cpu_model == .baseline and
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
    const native_canonical_windows_inventory_link = exact_baseline_request and
        resolved_cpu_matches_canonical_baseline and
        target.result.cpu.arch == .x86_64 and target.result.cpu.model == &std.Target.x86.cpu.x86_64 and
        target.result.os.tag == .windows and target.result.abi == .gnu and target.result.ofmt == .coff and
        b.graph.host.result.cpu.arch == target.result.cpu.arch and
        b.graph.host.result.os.tag == target.result.os.tag;
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
    var requested_profile_count: usize = 0;
    for ([_]bool{
        structured_object_requested,
        level1_sve_candidates,
        level1_fixed_candidates,
        level2_fixed_candidates,
        level2_width_candidates,
    }) |requested| {
        requested_profile_count += @intFromBool(requested);
    }
    if (requested_profile_count > 1) {
        @panic("experimental profile flags are mutually exclusive: select at most one of " ++
            "-Dstructured-object-candidates, -Dstructured-object-baseline, " ++
            "-Dlevel1-sve-candidates, -Dlevel1-fixed-candidates, " ++
            "-Dlevel2-fixed-candidates, -Dlevel2-width-candidates");
    }
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

    const kernel_libraries = if (dynamic_dispatch)
        buildKernelTiers(b, target, optimize, apple_amx, level1_sve_candidates, level1_fixed_candidates, level2_fixed_candidates, level2_width_candidates)
    else
        &.{};
    const kernel_test_libraries = if (dynamic_dispatch)
        buildKernelTiers(b, target, test_optimize, apple_amx, level1_sve_candidates, level1_fixed_candidates, level2_fixed_candidates, level2_width_candidates)
    else
        &.{};

    // One stateless scratch service module per compatible resolved target.
    // Leave optimize inherited so production and tests keep their own modes.
    const numerics_resources_mod = b.createModule(.{
        .root_source_file = b.path("src/numerics/resources.zig"),
        .target = target,
    });
    // The coverage generator executes on the build host even when the library
    // target is foreign. Reuse the main module only for identical targets.
    const resources_host_matches_target = std.Target.Query.eql(
        std.Target.Query.fromTarget(&target.result),
        std.Target.Query.fromTarget(&b.graph.host.result),
    ) and target.result.ofmt == b.graph.host.result.ofmt and
        target.result.dynamic_linker.eql(b.graph.host.result.dynamic_linker);
    const host_numerics_resources_mod = if (resources_host_matches_target)
        numerics_resources_mod
    else
        b.createModule(.{
            .root_source_file = b.path("src/numerics/resources.zig"),
            .target = b.graph.host,
        });

    // One execution/configuration owner per compatible compilation mode.
    // Never recreate the options module from the same generated source file.
    const execution_mod = b.createModule(.{
        .root_source_file = b.path("src/numerics/execution/default.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    execution_mod.addImport("zynum-build-options", zynum_build_options_mod);
    execution_mod.addImport("zynum-numerics-resources", numerics_resources_mod);
    const execution_test_mod = if (optimize == test_optimize) execution_mod else b.createModule(.{
        .root_source_file = b.path("src/numerics/execution/default.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    execution_test_mod.addImport("zynum-build-options", zynum_build_options_mod);
    execution_test_mod.addImport("zynum-numerics-resources", numerics_resources_mod);

    // One implementation/type/runtime owner for this compilation configuration.
    // Facades below contain aliases only; keep the original numerical roots here.
    const zynum_core_mod = b.createModule(.{
        .root_source_file = b.path("src/zynum.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const zynum_mod = b.addModule("zynum", .{
        .root_source_file = b.path("src/facades/zynum.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const zynum_blas_mod = b.addModule("zynum-blas", .{
        .root_source_file = b.path("src/facades/blas.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const zynum_lapack_mod = b.addModule("zynum-lapack", .{
        .root_source_file = b.path("src/facades/lapack.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const zynum_sparse_mod = b.addModule("zynum-sparse", .{
        .root_source_file = b.path("src/facades/sparse.zig"),
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
    // One implementation/type/runtime owner for this compilation configuration.
    // Facades below contain aliases only; keep the original numerical roots here.
    const zynum_core_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zynum.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    const zynum_test_mod = b.createModule(.{
        .root_source_file = b.path("src/facades/zynum.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    const zynum_blas_test_mod = b.createModule(.{
        .root_source_file = b.path("src/facades/blas.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    const zynum_lapack_test_mod = b.createModule(.{
        .root_source_file = b.path("src/facades/lapack.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    const zynum_sparse_test_mod = b.createModule(.{
        .root_source_file = b.path("src/facades/sparse.zig"),
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
    const blas_abi_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/abi/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    blas_abi_mod.addImport("zynum-core", zynum_core_mod);
    const blas_abi_test_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/abi/root.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    blas_abi_test_mod.addImport("zynum-core", zynum_core_test_mod);
    blas_compat_mod.addImport("zynum-blas-abi", blas_abi_mod);
    fortran_compat_mod.addImport("zynum-blas-abi", blas_abi_mod);
    cblas_compat_mod.addImport("zynum-blas-abi", blas_abi_mod);
    fortran_compat_test_mod.addImport("zynum-blas-abi", blas_abi_test_mod);
    cblas_compat_test_mod.addImport("zynum-blas-abi", blas_abi_test_mod);
    zynum_core_mod.addImport("zynum-build-options", zynum_build_options_mod);
    zynum_core_mod.addImport("zynum-execution", execution_mod);
    zynum_core_mod.addImport("zynum-numerics-resources", numerics_resources_mod);
    zynum_mod.addImport("zynum-core", zynum_core_mod);
    zynum_blas_mod.addImport("zynum-core", zynum_core_mod);
    zynum_lapack_mod.addImport("zynum-core", zynum_core_mod);
    zynum_sparse_mod.addImport("zynum-core", zynum_core_mod);
    zynum_core_test_mod.addImport("zynum-build-options", zynum_build_options_mod);
    zynum_core_test_mod.addImport("zynum-execution", execution_test_mod);
    zynum_core_test_mod.addImport("zynum-numerics-resources", numerics_resources_mod);
    zynum_test_mod.addImport("zynum-core", zynum_core_test_mod);
    zynum_blas_test_mod.addImport("zynum-core", zynum_core_test_mod);
    zynum_lapack_test_mod.addImport("zynum-core", zynum_core_test_mod);
    zynum_sparse_test_mod.addImport("zynum-core", zynum_core_test_mod);

    // Internal semantic bindings borrow the corresponding existing owner.
    // Even equal optimize modes do not make the two Core Module objects equal.
    const lapacke_binding_mod = b.createModule(.{
        .root_source_file = b.path("src/interop/lapacke_binding.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    lapacke_binding_mod.addImport("zynum-core", zynum_core_mod);
    const lapacke_binding_test_mod = b.createModule(.{
        .root_source_file = b.path("src/interop/lapacke_binding.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    lapacke_binding_test_mod.addImport("zynum-core", zynum_core_test_mod);
    const fftw_binding_test_mod = b.createModule(.{
        .root_source_file = b.path("src/interop/fftw_binding.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    fftw_binding_test_mod.addImport("zynum-core", zynum_core_test_mod);
    // Finite LAPACKE packages share the configured numerical owner. Separate
    // widths export the same standard names and must not be linked together.
    const lapacke_lp64_base_manifest = b.path(b.fmt("interface_subset/mkl2026.lp64/{s}/abi_manifest.json", .{lapacke_lp64_set}));
    var lapacke_lp64_manifest: std.Build.LazyPath = lapacke_lp64_base_manifest;
    var lapacke_lp64_component_check: ?*std.Build.Step.Run = null;
    var lapacke_lp64_syev_check: ?*std.Build.Step.Run = null;
    var lapacke_lp64_sygv_check: ?*std.Build.Step.Run = null;
    var lapacke_lp64_controls_check: ?*std.Build.Step.Run = null;
    var lapacke_lp64_evd_check: ?*std.Build.Step.Run = null;
    var lapacke_lp64_gvd_check: ?*std.Build.Step.Run = null;
    var lapacke_lp64_gels_check: ?*std.Build.Step.Run = null;
    var lapacke_lp64_gelss_check: ?*std.Build.Step.Run = null;
    if (lapacke_evd) {
        const lapacke_lp64_evd_drift_check = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--profile", "mkl2026.lp64", "--component", "evd", "--target-contract", "apple-arm64-default-c-v1", "--output", "interface_components/mkl2026.lp64/evd-v1", "--zig", b.graph.zig_exe, "--check" });
        lapacke_lp64_evd_drift_check.has_side_effects = true;
        lapacke_lp64_evd_check = lapacke_lp64_evd_drift_check;
    }
    if (lapacke_gvd) {
        const lapacke_lp64_gvd_drift_check = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--profile", "mkl2026.lp64", "--component", "gvd", "--target-contract", "apple-arm64-default-c-v1", "--output", "interface_components/mkl2026.lp64/gvd-v1", "--zig", b.graph.zig_exe, "--check" });
        lapacke_lp64_gvd_drift_check.has_side_effects = true;
        lapacke_lp64_gvd_check = lapacke_lp64_gvd_drift_check;
    }
    if (lapacke_gels) {
        const lapacke_lp64_gels_drift_check = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--profile", "mkl2026.lp64", "--component", "gels", "--target-contract", "apple-arm64-default-c-v1", "--output", "interface_components/mkl2026.lp64/gels-v1", "--zig", b.graph.zig_exe, "--check" });
        lapacke_lp64_gels_drift_check.has_side_effects = true;
        lapacke_lp64_gels_check = lapacke_lp64_gels_drift_check;
    }
    if (lapacke_gelss) {
        const lapacke_lp64_gelss_drift_check = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--profile", "mkl2026.lp64", "--component", "gelss", "--target-contract", "apple-arm64-default-c-v1", "--output", "interface_components/mkl2026.lp64/gelss-v1", "--zig", b.graph.zig_exe, "--check" });
        lapacke_lp64_gelss_drift_check.has_side_effects = true;
        lapacke_lp64_gelss_check = lapacke_lp64_gelss_drift_check;
    }
    if (lapacke_controls) {
        const lapacke_lp64_controls_drift_check = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--profile", "mkl2026.lp64", "--component", "controls", "--target-contract", "apple-arm64-default-c-v1", "--output", "interface_components/mkl2026.lp64/controls-v1", "--zig", b.graph.zig_exe, "--check" });
        lapacke_lp64_controls_drift_check.has_side_effects = true;
        lapacke_lp64_controls_check = lapacke_lp64_controls_drift_check;
    }
    if (lapacke_potri) {
        const lapacke_lp64_potri_check = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--profile", "mkl2026.lp64", "--component", "potri", "--target-contract", "apple-arm64-default-c-v1", "--output", "interface_components/mkl2026.lp64/potri-v1", "--zig", b.graph.zig_exe, "--check" });
        lapacke_lp64_potri_check.has_side_effects = true;
        lapacke_lp64_component_check = lapacke_lp64_potri_check;
    }
    if (lapacke_syev) {
        const lapacke_lp64_syev_drift_check = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--profile", "mkl2026.lp64", "--component", "syev", "--target-contract", "apple-arm64-default-c-v1", "--output", "interface_components/mkl2026.lp64/syev-v1", "--zig", b.graph.zig_exe, "--check" });
        lapacke_lp64_syev_drift_check.has_side_effects = true;
        lapacke_lp64_syev_check = lapacke_lp64_syev_drift_check;
    }
    if (lapacke_sygv) {
        const lapacke_lp64_sygv_drift_check = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--profile", "mkl2026.lp64", "--component", "sygv", "--target-contract", "apple-arm64-default-c-v1", "--output", "interface_components/mkl2026.lp64/sygv-v1", "--zig", b.graph.zig_exe, "--check" });
        lapacke_lp64_sygv_drift_check.has_side_effects = true;
        lapacke_lp64_sygv_check = lapacke_lp64_sygv_drift_check;
    }
    if (lapacke_controls or lapacke_evd or lapacke_gels or lapacke_gelss or lapacke_gvd or lapacke_potri or lapacke_syev or lapacke_sygv) {
        const lapacke_lp64_components_compose = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--compose-base" });
        lapacke_lp64_components_compose.addFileArg(lapacke_lp64_base_manifest);
        if (lapacke_controls) {
            lapacke_lp64_components_compose.addArg("--component-manifest");
            lapacke_lp64_components_compose.addFileArg(b.path("interface_components/mkl2026.lp64/controls-v1/abi_manifest.json"));
            lapacke_lp64_components_compose.step.dependOn(&lapacke_lp64_controls_check.?.step);
        }
        if (lapacke_evd) {
            lapacke_lp64_components_compose.addArg("--component-manifest");
            lapacke_lp64_components_compose.addFileArg(b.path("interface_components/mkl2026.lp64/evd-v1/abi_manifest.json"));
            lapacke_lp64_components_compose.step.dependOn(&lapacke_lp64_evd_check.?.step);
        }
        if (lapacke_gels) {
            lapacke_lp64_components_compose.addArg("--component-manifest");
            lapacke_lp64_components_compose.addFileArg(b.path("interface_components/mkl2026.lp64/gels-v1/abi_manifest.json"));
            lapacke_lp64_components_compose.step.dependOn(&lapacke_lp64_gels_check.?.step);
        }
        if (lapacke_gelss) {
            lapacke_lp64_components_compose.addArg("--component-manifest");
            lapacke_lp64_components_compose.addFileArg(b.path("interface_components/mkl2026.lp64/gelss-v1/abi_manifest.json"));
            lapacke_lp64_components_compose.step.dependOn(&lapacke_lp64_gelss_check.?.step);
        }
        if (lapacke_gvd) {
            lapacke_lp64_components_compose.addArg("--component-manifest");
            lapacke_lp64_components_compose.addFileArg(b.path("interface_components/mkl2026.lp64/gvd-v1/abi_manifest.json"));
            lapacke_lp64_components_compose.step.dependOn(&lapacke_lp64_gvd_check.?.step);
        }
        if (lapacke_potri) {
            lapacke_lp64_components_compose.addArg("--component-manifest");
            lapacke_lp64_components_compose.addFileArg(b.path("interface_components/mkl2026.lp64/potri-v1/abi_manifest.json"));
            lapacke_lp64_components_compose.step.dependOn(&lapacke_lp64_component_check.?.step);
        }
        if (lapacke_syev) {
            lapacke_lp64_components_compose.addArg("--component-manifest");
            lapacke_lp64_components_compose.addFileArg(b.path("interface_components/mkl2026.lp64/syev-v1/abi_manifest.json"));
            lapacke_lp64_components_compose.step.dependOn(&lapacke_lp64_syev_check.?.step);
        }
        if (lapacke_sygv) {
            lapacke_lp64_components_compose.addArg("--component-manifest");
            lapacke_lp64_components_compose.addFileArg(b.path("interface_components/mkl2026.lp64/sygv-v1/abi_manifest.json"));
            lapacke_lp64_components_compose.step.dependOn(&lapacke_lp64_sygv_check.?.step);
        }
        lapacke_lp64_components_compose.addFileInput(b.path("tools/generate_lapacke_component.py"));
        lapacke_lp64_components_compose.addFileInput(b.path("tools/lapacke_components.py"));
        lapacke_lp64_components_compose.addFileInput(b.path("tools/checked_abi_schema.py"));
        lapacke_lp64_components_compose.addFileInput(b.path("tools/standard_abi_bindings.py"));
        lapacke_lp64_components_compose.addArg("--output-manifest");
        lapacke_lp64_manifest = lapacke_lp64_components_compose.addOutputFileArg("lapacke-lp64-composed-manifest.json");
    }
    const lapacke_ilp64_base_manifest = b.path(b.fmt("interface_subset/mkl2026.ilp64/{s}/abi_manifest.json", .{lapacke_ilp64_set}));
    var lapacke_ilp64_manifest: std.Build.LazyPath = lapacke_ilp64_base_manifest;
    var lapacke_ilp64_component_check: ?*std.Build.Step.Run = null;
    var lapacke_ilp64_controls_check: ?*std.Build.Step.Run = null;
    if (lapacke_controls) {
        const lapacke_ilp64_controls_drift_check = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--profile", "mkl2026.ilp64", "--component", "controls", "--target-contract", "apple-arm64-default-c-v1", "--output", "interface_components/mkl2026.ilp64/controls-v1", "--zig", b.graph.zig_exe, "--check" });
        lapacke_ilp64_controls_drift_check.has_side_effects = true;
        lapacke_ilp64_controls_check = lapacke_ilp64_controls_drift_check;
    }
    if (lapacke_potri) {
        const lapacke_ilp64_potri_check = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--profile", "mkl2026.ilp64", "--component", "potri", "--target-contract", "apple-arm64-default-c-v1", "--output", "interface_components/mkl2026.ilp64/potri-v1", "--zig", b.graph.zig_exe, "--check" });
        lapacke_ilp64_potri_check.has_side_effects = true;
        lapacke_ilp64_component_check = lapacke_ilp64_potri_check;
    }
    if (lapacke_controls or lapacke_potri) {
        const lapacke_ilp64_potri_compose = b.addSystemCommand(&.{ "python3", "tools/generate_lapacke_component.py", "--compose-base" });
        lapacke_ilp64_potri_compose.addFileArg(lapacke_ilp64_base_manifest);
        if (lapacke_controls) {
            lapacke_ilp64_potri_compose.addArg("--component-manifest");
            lapacke_ilp64_potri_compose.addFileArg(b.path("interface_components/mkl2026.ilp64/controls-v1/abi_manifest.json"));
            lapacke_ilp64_potri_compose.step.dependOn(&lapacke_ilp64_controls_check.?.step);
        }
        if (lapacke_potri) {
            lapacke_ilp64_potri_compose.addArg("--component-manifest");
            lapacke_ilp64_potri_compose.addFileArg(b.path("interface_components/mkl2026.ilp64/potri-v1/abi_manifest.json"));
            lapacke_ilp64_potri_compose.step.dependOn(&lapacke_ilp64_component_check.?.step);
        }
        lapacke_ilp64_potri_compose.addFileInput(b.path("tools/generate_lapacke_component.py"));
        lapacke_ilp64_potri_compose.addFileInput(b.path("tools/lapacke_components.py"));
        lapacke_ilp64_potri_compose.addFileInput(b.path("tools/checked_abi_schema.py"));
        lapacke_ilp64_potri_compose.addFileInput(b.path("tools/standard_abi_bindings.py"));
        lapacke_ilp64_potri_compose.addArg("--output-manifest");
        lapacke_ilp64_manifest = lapacke_ilp64_potri_compose.addOutputFileArg("lapacke-ilp64-composed-manifest.json");
    }
    const lapacke_lp64_header_files = b.addWriteFiles();
    const lapacke_lp64_selected_header = if (lapacke_controls or lapacke_evd or lapacke_gels or lapacke_gelss or lapacke_gvd or lapacke_potri or lapacke_syev or lapacke_sygv)
        lapacke_lp64_header_files.add("lapacke.h", b.fmt("#pragma once\n#include \"lapacke_base.h\"\n{s}{s}{s}{s}{s}{s}{s}{s}", .{ if (lapacke_controls) "#include \"lapacke_controls.h\"\n" else "", if (lapacke_potri) "#include \"lapacke_potri.h\"\n" else "", if (lapacke_syev) "#include \"lapacke_syev.h\"\n" else "", if (lapacke_sygv) "#include \"lapacke_sygv.h\"\n" else "", if (lapacke_evd) "#include \"lapacke_evd.h\"\n" else "", if (lapacke_gvd) "#include \"lapacke_gvd.h\"\n" else "", if (lapacke_gels) "#include \"lapacke_gels.h\"\n" else "", if (lapacke_gelss) "#include \"lapacke_gelss.h\"\n" else "" }))
    else
        b.path(b.fmt("interface_subset/mkl2026.lp64/{s}/subset.h", .{lapacke_lp64_set}));
    const lapacke_ilp64_header_files = b.addWriteFiles();
    const lapacke_ilp64_selected_header = if (lapacke_controls or lapacke_potri)
        lapacke_ilp64_header_files.add("lapacke.h", b.fmt("#pragma once\n#include \"lapacke_base.h\"\n{s}{s}", .{ if (lapacke_controls) "#include \"lapacke_controls.h\"\n" else "", if (lapacke_potri) "#include \"lapacke_potri.h\"\n" else "" }))
    else
        b.path(b.fmt("interface_subset/mkl2026.ilp64/{s}/subset.h", .{lapacke_ilp64_set}));
    const lapacke_lp64_header_install = b.addInstallFileWithDir(lapacke_lp64_selected_header, .header, "zynum/lapacke/lp64/lapacke.h");
    const lapacke_lp64_manifest_install = b.addInstallFileWithDir(lapacke_lp64_manifest, .header, "zynum/lapacke/lp64/abi_manifest.json");
    const lapacke_lp64_base_header_install = b.addInstallFileWithDir(b.path(b.fmt("interface_subset/mkl2026.lp64/{s}/subset.h", .{lapacke_lp64_set})), .header, "zynum/lapacke/lp64/lapacke_base.h");
    const lapacke_lp64_potri_header_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/potri-v1/lapacke_potri.h"), .header, "zynum/lapacke/lp64/lapacke_potri.h");
    const lapacke_lp64_potri_manifest_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/potri-v1/abi_manifest.json"), .header, "zynum/lapacke/lp64/potri_manifest.json");
    const lapacke_lp64_syev_header_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/syev-v1/lapacke_syev.h"), .header, "zynum/lapacke/lp64/lapacke_syev.h");
    const lapacke_lp64_sygv_header_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/sygv-v1/lapacke_sygv.h"), .header, "zynum/lapacke/lp64/lapacke_sygv.h");
    const lapacke_lp64_syev_manifest_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/syev-v1/abi_manifest.json"), .header, "zynum/lapacke/lp64/syev_manifest.json");
    const lapacke_lp64_sygv_manifest_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/sygv-v1/abi_manifest.json"), .header, "zynum/lapacke/lp64/sygv_manifest.json");
    const lapacke_lp64_evd_header_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/evd-v1/lapacke_evd.h"), .header, "zynum/lapacke/lp64/lapacke_evd.h");
    const lapacke_lp64_gvd_header_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/gvd-v1/lapacke_gvd.h"), .header, "zynum/lapacke/lp64/lapacke_gvd.h");
    const lapacke_lp64_gels_header_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/gels-v1/lapacke_gels.h"), .header, "zynum/lapacke/lp64/lapacke_gels.h");
    const lapacke_lp64_gelss_header_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/gelss-v1/lapacke_gelss.h"), .header, "zynum/lapacke/lp64/lapacke_gelss.h");
    const lapacke_lp64_evd_manifest_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/evd-v1/abi_manifest.json"), .header, "zynum/lapacke/lp64/evd_manifest.json");
    const lapacke_lp64_gvd_manifest_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/gvd-v1/abi_manifest.json"), .header, "zynum/lapacke/lp64/gvd_manifest.json");
    const lapacke_lp64_gels_manifest_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/gels-v1/abi_manifest.json"), .header, "zynum/lapacke/lp64/gels_manifest.json");
    const lapacke_lp64_gelss_manifest_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/gelss-v1/abi_manifest.json"), .header, "zynum/lapacke/lp64/gelss_manifest.json");
    const lapacke_lp64_controls_header_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/controls-v1/lapacke_controls.h"), .header, "zynum/lapacke/lp64/lapacke_controls.h");
    const lapacke_lp64_controls_manifest_install = b.addInstallFileWithDir(b.path("interface_components/mkl2026.lp64/controls-v1/abi_manifest.json"), .header, "zynum/lapacke/lp64/controls_manifest.json");
    const lapacke_lp64_install_step = b.step("install-lapacke-lp64", "Install finite LAPACKE lp64 interface");
    if (target.result.cpu.arch == .aarch64 and target.result.os.tag == .macos) {
        const lapacke_lp64_check = b.addSystemCommand(&.{ "python3", "tools/generate_checked_abi.py", "--profile", "mkl2026.lp64", "--binding-set", lapacke_lp64_set, "--target-contract", "apple-arm64-default-c-v1", "--zig", b.graph.zig_exe, "--check" });
        const lapacke_lp64_interface = b.createModule(.{ .root_source_file = b.path(b.fmt("interface_subset/mkl2026.lp64/{s}/exports.zig", .{lapacke_lp64_set})), .target = target, .optimize = optimize, .link_libc = true });
        lapacke_lp64_interface.addImport("lapacke-interface-owner", lapacke_binding_mod);
        const lapacke_lp64_defaults_mod = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
        lapacke_lp64_defaults_mod.addCSourceFile(.{ .file = b.path("src/interop/lapacke_default_errors.c"), .flags = &.{ "-std=c11", "-fno-sanitize=undefined" } });

        const lapacke_lp64_defaults = b.addObject(.{ .name = "lapacke-default-errors-lp64", .root_module = lapacke_lp64_defaults_mod });
        lapacke_lp64_interface.addObject(lapacke_lp64_defaults);
        var lapacke_lp64_selected = lapacke_lp64_interface;
        if (lapacke_potri) {
            const lapacke_lp64_potri_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/potri-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_potri_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const lapacke_lp64_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_composed_root.addImport("zynum-core", zynum_core_mod);
            lapacke_lp64_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            lapacke_lp64_composed_root.addImport("zynum-lapacke-base-exports", lapacke_lp64_interface);
            lapacke_lp64_composed_root.addImport("zynum-lapacke-component-exports", lapacke_lp64_potri_component);
            lapacke_lp64_selected = lapacke_lp64_composed_root;
        }
        const lapacke_lp64_before_syev = lapacke_lp64_selected;
        var lapacke_lp64_syev_selected = lapacke_lp64_before_syev;
        if (lapacke_syev) {
            const lapacke_lp64_syev_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/syev-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_syev_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const lapacke_lp64_syev_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_symmetric_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_syev_composed_root.addImport("zynum-core", zynum_core_mod);
            lapacke_lp64_syev_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            lapacke_lp64_syev_composed_root.addImport("zynum-lapacke-base-exports", lapacke_lp64_before_syev);
            lapacke_lp64_syev_composed_root.addImport("zynum-lapacke-symmetric-exports", lapacke_lp64_syev_component);
            lapacke_lp64_syev_selected = lapacke_lp64_syev_composed_root;
        }
        const lapacke_lp64_before_sygv = lapacke_lp64_syev_selected;
        var lapacke_lp64_sygv_selected = lapacke_lp64_before_sygv;
        if (lapacke_sygv) {
            const lapacke_lp64_sygv_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/sygv-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_sygv_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const lapacke_lp64_sygv_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_symmetric_generalized_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_sygv_composed_root.addImport("zynum-core", zynum_core_mod);
            lapacke_lp64_sygv_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            lapacke_lp64_sygv_composed_root.addImport("zynum-lapacke-base-exports", lapacke_lp64_before_sygv);
            lapacke_lp64_sygv_composed_root.addImport("zynum-lapacke-symmetric-generalized-exports", lapacke_lp64_sygv_component);
            lapacke_lp64_sygv_selected = lapacke_lp64_sygv_composed_root;
        }
        var lapacke_lp64_controls_selected = lapacke_lp64_sygv_selected;
        if (lapacke_controls) {
            const lapacke_lp64_controls_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/controls-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_controls_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const lapacke_lp64_controls_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_controls_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_controls_composed_root.addImport("zynum-core", zynum_core_mod);
            lapacke_lp64_controls_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            lapacke_lp64_controls_composed_root.addImport("zynum-lapacke-base-exports", lapacke_lp64_sygv_selected);
            lapacke_lp64_controls_composed_root.addImport("zynum-lapacke-controls-exports", lapacke_lp64_controls_component);
            lapacke_lp64_controls_selected = lapacke_lp64_controls_composed_root;
        }
        var lapacke_lp64_evd_selected = lapacke_lp64_controls_selected;
        if (lapacke_evd) {
            const lapacke_lp64_evd_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/evd-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_evd_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const lapacke_lp64_evd_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_evd_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_evd_composed_root.addImport("zynum-core", zynum_core_mod);
            lapacke_lp64_evd_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            lapacke_lp64_evd_composed_root.addImport("zynum-lapacke-base-exports", lapacke_lp64_controls_selected);
            lapacke_lp64_evd_composed_root.addImport("zynum-lapacke-evd-exports", lapacke_lp64_evd_component);
            lapacke_lp64_evd_selected = lapacke_lp64_evd_composed_root;
        }
        var lapacke_lp64_gvd_selected = lapacke_lp64_evd_selected;
        if (lapacke_gvd) {
            const lapacke_lp64_gvd_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/gvd-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_gvd_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const lapacke_lp64_gvd_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_gvd_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_gvd_composed_root.addImport("zynum-core", zynum_core_mod);
            lapacke_lp64_gvd_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            lapacke_lp64_gvd_composed_root.addImport("zynum-lapacke-base-exports", lapacke_lp64_evd_selected);
            lapacke_lp64_gvd_composed_root.addImport("zynum-lapacke-gvd-exports", lapacke_lp64_gvd_component);
            lapacke_lp64_gvd_selected = lapacke_lp64_gvd_composed_root;
        }
        var lapacke_lp64_gels_selected = lapacke_lp64_gvd_selected;
        if (lapacke_gels) {
            const lapacke_lp64_gels_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/gels-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_gels_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const lapacke_lp64_gels_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_gels_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_gels_composed_root.addImport("zynum-core", zynum_core_mod);
            lapacke_lp64_gels_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            lapacke_lp64_gels_composed_root.addImport("zynum-lapacke-base-exports", lapacke_lp64_gvd_selected);
            lapacke_lp64_gels_composed_root.addImport("zynum-lapacke-gels-exports", lapacke_lp64_gels_component);
            lapacke_lp64_gels_selected = lapacke_lp64_gels_composed_root;
        }
        var lapacke_lp64_gelss_selected = lapacke_lp64_gels_selected;
        if (lapacke_gelss) {
            const lapacke_lp64_gelss_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/gelss-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_gelss_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const lapacke_lp64_gelss_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_gelss_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_lp64_gelss_composed_root.addImport("zynum-core", zynum_core_mod);
            lapacke_lp64_gelss_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            lapacke_lp64_gelss_composed_root.addImport("zynum-lapacke-base-exports", lapacke_lp64_gels_selected);
            lapacke_lp64_gelss_composed_root.addImport("zynum-lapacke-gelss-exports", lapacke_lp64_gelss_component);
            lapacke_lp64_gelss_selected = lapacke_lp64_gelss_composed_root;
        }
        const lapacke_lp64_static = b.addLibrary(.{ .name = "zynum_lapacke_lp64", .linkage = .static, .root_module = lapacke_lp64_gelss_selected });
        lapacke_lp64_static.bundle_compiler_rt = true;
        if (lapacke_lp64_controls_check) |check| lapacke_lp64_static.step.dependOn(&check.step);
        lapacke_lp64_static.step.dependOn(&lapacke_lp64_check.step);
        if (lapacke_lp64_component_check) |check| lapacke_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_syev_check) |check| lapacke_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_sygv_check) |check| lapacke_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_evd_check) |check| lapacke_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_gvd_check) |check| lapacke_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_gels_check) |check| lapacke_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_gelss_check) |check| lapacke_lp64_static.step.dependOn(&check.step);
        const lapacke_lp64_repack = b.addSystemCommand(&.{ "python3", "tools/repack_darwin_archive.py", b.graph.zig_exe });
        lapacke_lp64_repack.addFileArg(lapacke_lp64_static.getEmittedBin());
        const lapacke_lp64_archive = lapacke_lp64_repack.addOutputFileArg("libzynum_lapacke_lp64.a");
        const lapacke_lp64_dynamic_link = b.addSystemCommand(&.{ "python3", "tools/link_darwin_interface.py" });
        lapacke_lp64_dynamic_link.addFileArg(lapacke_lp64_archive);
        lapacke_lp64_dynamic_link.addFileArg(lapacke_lp64_manifest);
        const lapacke_lp64_dynamic_output = lapacke_lp64_dynamic_link.addOutputFileArg("libzynum_lapacke_lp64.dylib");
        const lapacke_lp64_macos_min = target.result.os.version_range.semver.min;
        lapacke_lp64_dynamic_link.addArgs(&.{ "--macos-min", b.fmt("{d}.{d}.{d}", .{ lapacke_lp64_macos_min.major, lapacke_lp64_macos_min.minor, lapacke_lp64_macos_min.patch }) });
        lapacke_lp64_install_step.dependOn(&b.addInstallFileWithDir(lapacke_lp64_dynamic_output, .lib, "libzynum_lapacke_lp64.dylib").step);
        lapacke_lp64_install_step.dependOn(&b.addInstallFileWithDir(lapacke_lp64_archive, .lib, "libzynum_lapacke_lp64.a").step);
        lapacke_lp64_install_step.dependOn(&lapacke_lp64_header_install.step);
        lapacke_lp64_install_step.dependOn(&lapacke_lp64_manifest_install.step);
        if (lapacke_controls) {
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_base_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_controls_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_controls_manifest_install.step);
        }
        if (lapacke_potri) {
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_base_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_potri_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_potri_manifest_install.step);
        }
        if (lapacke_syev) {
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_base_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_syev_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_syev_manifest_install.step);
        }
        if (lapacke_evd) {
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_base_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_evd_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_evd_manifest_install.step);
        }
        if (lapacke_gvd) {
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_base_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_gvd_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_gvd_manifest_install.step);
        }
        if (lapacke_gels) {
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_base_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_gels_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_gels_manifest_install.step);
        }
        if (lapacke_gelss) {
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_base_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_gelss_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_gelss_manifest_install.step);
        }
        if (lapacke_sygv) {
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_base_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_sygv_header_install.step);
            lapacke_lp64_install_step.dependOn(&lapacke_lp64_sygv_manifest_install.step);
        }
        const lapacke_lp64_files = b.addWriteFiles();
        const lapacke_lp64_pc = lapacke_lp64_files.add("zynum_lapacke_lp64.pc", b.fmt("lapacke_gelss={s}\nlapacke_gels={s}\nlapacke_gvd={s}\nlapacke_evd={s}\nlapacke_controls={s}\nlapacke_sygv={s}\nprefix=${{pcfiledir}}/../..\nlibdir=${{prefix}}/lib\nincludedir=${{prefix}}/include/zynum/lapacke/lp64\nName: zynum_lapacke_lp64\nDescription: Zynum finite POTRF/POTRS/GETRF/GETRS/GESV/STEV interface\nVersion: 0.0.1-beta\nLibs: -L${{libdir}} -lzynum_lapacke_lp64\nCflags: -I${{includedir}}\n", .{ if (lapacke_gelss) "true" else "false", if (lapacke_gels) "true" else "false", if (lapacke_gvd) "true" else "false", if (lapacke_evd) "true" else "false", if (lapacke_controls) "true" else "false", if (lapacke_sygv) "true" else "false" }));
        lapacke_lp64_install_step.dependOn(&b.addInstallFileWithDir(lapacke_lp64_pc, .lib, "pkgconfig/zynum_lapacke_lp64.pc").step);
    } else {
        lapacke_lp64_install_step.dependOn(&b.addFail("LP64 finite LAPACKE requires the Apple arm64 declaration contract").step);
    }

    const lapacke_ilp64_install_step = b.step("install-lapacke-ilp64", "Install finite LAPACKE ilp64 interface");
    if (target.result.cpu.arch == .aarch64 and target.result.os.tag == .macos) {
        const lapacke_ilp64_check = b.addSystemCommand(&.{ "python3", "tools/generate_checked_abi.py", "--profile", "mkl2026.ilp64", "--binding-set", lapacke_ilp64_set, "--target-contract", "apple-arm64-default-c-v1", "--zig", b.graph.zig_exe, "--check" });
        const lapacke_ilp64_interface = b.createModule(.{ .root_source_file = b.path(b.fmt("interface_subset/mkl2026.ilp64/{s}/exports.zig", .{lapacke_ilp64_set})), .target = target, .optimize = optimize, .link_libc = true });
        lapacke_ilp64_interface.addImport("lapacke-interface-owner", lapacke_binding_mod);
        const lapacke_ilp64_defaults_mod = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
        lapacke_ilp64_defaults_mod.addCSourceFile(.{ .file = b.path("src/interop/lapacke_default_errors.c"), .flags = &.{ "-std=c11", "-fno-sanitize=undefined" } });
        lapacke_ilp64_defaults_mod.addCMacro("ZYNUM_LAPACKE_ILP64", "1");
        const lapacke_ilp64_defaults = b.addObject(.{ .name = "lapacke-default-errors-ilp64", .root_module = lapacke_ilp64_defaults_mod });
        lapacke_ilp64_interface.addObject(lapacke_ilp64_defaults);
        var lapacke_ilp64_selected = lapacke_ilp64_interface;
        if (lapacke_potri) {
            const lapacke_ilp64_potri_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.ilp64/potri-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_ilp64_potri_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const lapacke_ilp64_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_ilp64_composed_root.addImport("zynum-core", zynum_core_mod);
            lapacke_ilp64_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            lapacke_ilp64_composed_root.addImport("zynum-lapacke-base-exports", lapacke_ilp64_interface);
            lapacke_ilp64_composed_root.addImport("zynum-lapacke-component-exports", lapacke_ilp64_potri_component);
            lapacke_ilp64_selected = lapacke_ilp64_composed_root;
        }
        var lapacke_ilp64_controls_selected = lapacke_ilp64_selected;
        if (lapacke_controls) {
            const lapacke_ilp64_controls_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.ilp64/controls-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_ilp64_controls_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const lapacke_ilp64_controls_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_controls_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            lapacke_ilp64_controls_composed_root.addImport("zynum-core", zynum_core_mod);
            lapacke_ilp64_controls_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            lapacke_ilp64_controls_composed_root.addImport("zynum-lapacke-base-exports", lapacke_ilp64_selected);
            lapacke_ilp64_controls_composed_root.addImport("zynum-lapacke-controls-exports", lapacke_ilp64_controls_component);
            lapacke_ilp64_controls_selected = lapacke_ilp64_controls_composed_root;
        }
        const lapacke_ilp64_static = b.addLibrary(.{ .name = "zynum_lapacke_ilp64", .linkage = .static, .root_module = lapacke_ilp64_controls_selected });
        lapacke_ilp64_static.bundle_compiler_rt = true;
        if (lapacke_ilp64_controls_check) |check| lapacke_ilp64_static.step.dependOn(&check.step);
        lapacke_ilp64_static.step.dependOn(&lapacke_ilp64_check.step);
        if (lapacke_ilp64_component_check) |check| lapacke_ilp64_static.step.dependOn(&check.step);
        const lapacke_ilp64_repack = b.addSystemCommand(&.{ "python3", "tools/repack_darwin_archive.py", b.graph.zig_exe });
        lapacke_ilp64_repack.addFileArg(lapacke_ilp64_static.getEmittedBin());
        const lapacke_ilp64_archive = lapacke_ilp64_repack.addOutputFileArg("libzynum_lapacke_ilp64.a");
        const lapacke_ilp64_dynamic_link = b.addSystemCommand(&.{ "python3", "tools/link_darwin_interface.py" });
        lapacke_ilp64_dynamic_link.addFileArg(lapacke_ilp64_archive);
        lapacke_ilp64_dynamic_link.addFileArg(lapacke_ilp64_manifest);
        const lapacke_ilp64_dynamic_output = lapacke_ilp64_dynamic_link.addOutputFileArg("libzynum_lapacke_ilp64.dylib");
        const lapacke_ilp64_macos_min = target.result.os.version_range.semver.min;
        lapacke_ilp64_dynamic_link.addArgs(&.{ "--macos-min", b.fmt("{d}.{d}.{d}", .{ lapacke_ilp64_macos_min.major, lapacke_ilp64_macos_min.minor, lapacke_ilp64_macos_min.patch }) });
        lapacke_ilp64_install_step.dependOn(&b.addInstallFileWithDir(lapacke_ilp64_dynamic_output, .lib, "libzynum_lapacke_ilp64.dylib").step);
        lapacke_ilp64_install_step.dependOn(&b.addInstallFileWithDir(lapacke_ilp64_archive, .lib, "libzynum_lapacke_ilp64.a").step);
        lapacke_ilp64_install_step.dependOn(&b.addInstallFileWithDir(lapacke_ilp64_selected_header, .header, "zynum/lapacke/ilp64/lapacke.h").step);
        lapacke_ilp64_install_step.dependOn(&b.addInstallFileWithDir(lapacke_ilp64_manifest, .header, "zynum/lapacke/ilp64/abi_manifest.json").step);
        if (lapacke_controls or lapacke_potri) {
            lapacke_ilp64_install_step.dependOn(&b.addInstallFileWithDir(b.path(b.fmt("interface_subset/mkl2026.ilp64/{s}/subset.h", .{lapacke_ilp64_set})), .header, "zynum/lapacke/ilp64/lapacke_base.h").step);
        }
        if (lapacke_potri) {
            lapacke_ilp64_install_step.dependOn(&b.addInstallFileWithDir(b.path("interface_components/mkl2026.ilp64/potri-v1/lapacke_potri.h"), .header, "zynum/lapacke/ilp64/lapacke_potri.h").step);
            lapacke_ilp64_install_step.dependOn(&b.addInstallFileWithDir(b.path("interface_components/mkl2026.ilp64/potri-v1/abi_manifest.json"), .header, "zynum/lapacke/ilp64/potri_manifest.json").step);
        }
        if (lapacke_controls) {
            lapacke_ilp64_install_step.dependOn(&b.addInstallFileWithDir(b.path("interface_components/mkl2026.ilp64/controls-v1/lapacke_controls.h"), .header, "zynum/lapacke/ilp64/lapacke_controls.h").step);
            lapacke_ilp64_install_step.dependOn(&b.addInstallFileWithDir(b.path("interface_components/mkl2026.ilp64/controls-v1/abi_manifest.json"), .header, "zynum/lapacke/ilp64/controls_manifest.json").step);
        }
        const lapacke_ilp64_files = b.addWriteFiles();
        const lapacke_ilp64_pc = lapacke_ilp64_files.add("zynum_lapacke_ilp64.pc", b.fmt("lapacke_controls={s}\n{s}", .{ if (lapacke_controls) "true" else "false", "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include/zynum/lapacke/ilp64\nName: zynum_lapacke_ilp64\nDescription: Zynum finite POTRF/POTRS/GETRF/GETRS/GESV/STEV interface\nVersion: 0.0.1-beta\nLibs: -L${libdir} -lzynum_lapacke_ilp64\nCflags: -I${includedir} -DMKL_ILP64\n" }));
        lapacke_ilp64_install_step.dependOn(&b.addInstallFileWithDir(lapacke_ilp64_pc, .lib, "pkgconfig/zynum_lapacke_ilp64.pc").step);
    } else {
        lapacke_ilp64_install_step.dependOn(&b.addFail("ILP64 finite LAPACKE requires the Apple arm64 declaration contract").step);
    }

    // One interface-owned Sparse shell family per existing implementation owner.
    // No new options/resources/kernel-object or default-state owner is attached.
    const mkl_sparse_binding_mod = b.createModule(.{
        .root_source_file = b.path("src/interop/mkl_sparse_binding.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mkl_sparse_binding_mod.addImport("zynum-core", zynum_core_mod);
    const mkl_sparse_binding_test_mod = b.createModule(.{
        .root_source_file = b.path("src/interop/mkl_sparse_binding.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    mkl_sparse_binding_test_mod.addImport("zynum-core", zynum_core_test_mod);

    // Finite Sparse packages share the configured numerical owner. Separate
    // widths export the same standard names and must not be linked together.
    const sparse_lp64_install_step = b.step("install-sparse-lp64", "Install finite Sparse lp64 interface");
    const sparse_lp64_header_install = b.addInstallFileWithDir(b.path(if (sparse_solves) "interface_subset/mkl2026.lp64/sparse-csr-csc-mv-mm-trsv-trsm-ordinary-lp64-v1/subset.h" else "interface_subset/mkl2026.lp64/sparse-csr-csc-mv-mm-ordinary-lp64-v1/subset.h"), .header, "zynum/sparse/lp64/mkl_spblas.h");
    const sparse_lp64_manifest_install = b.addInstallFileWithDir(b.path(if (sparse_solves) "interface_subset/mkl2026.lp64/sparse-csr-csc-mv-mm-trsv-trsm-ordinary-lp64-v1/abi_manifest.json" else "interface_subset/mkl2026.lp64/sparse-csr-csc-mv-mm-ordinary-lp64-v1/abi_manifest.json"), .header, "zynum/sparse/lp64/abi_manifest.json");
    if (target.result.cpu.arch == .aarch64 and target.result.os.tag == .macos) {
        const sparse_lp64_check = b.addSystemCommand(&.{ "python3", "tools/generate_checked_abi.py", "--profile", "mkl2026.lp64", "--binding-set", sparse_lp64_set, "--target-contract", "apple-arm64-default-c-v1", "--zig", b.graph.zig_exe, "--check" });
        const sparse_lp64_interface = b.createModule(.{ .root_source_file = b.path(if (sparse_solves) "interface_subset/mkl2026.lp64/sparse-csr-csc-mv-mm-trsv-trsm-ordinary-lp64-v1/exports.zig" else "interface_subset/mkl2026.lp64/sparse-csr-csc-mv-mm-ordinary-lp64-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
        sparse_lp64_interface.addImport("sparse-interface-owner", mkl_sparse_binding_mod);

        const sparse_lp64_dynamic = b.addLibrary(.{ .name = "zynum_sparse_lp64", .linkage = .dynamic, .root_module = sparse_lp64_interface });
        const sparse_lp64_static = b.addLibrary(.{ .name = "zynum_sparse_lp64", .linkage = .static, .root_module = sparse_lp64_interface });
        sparse_lp64_static.bundle_compiler_rt = true;
        sparse_lp64_dynamic.step.dependOn(&sparse_lp64_check.step);
        sparse_lp64_static.step.dependOn(&sparse_lp64_check.step);
        const sparse_lp64_repack = b.addSystemCommand(&.{ "python3", "tools/repack_darwin_archive.py", b.graph.zig_exe });
        sparse_lp64_repack.addFileArg(sparse_lp64_static.getEmittedBin());
        const sparse_lp64_archive = sparse_lp64_repack.addOutputFileArg("libzynum_sparse_lp64.a");
        const sparse_lp64_dynamic_install = b.addInstallArtifact(sparse_lp64_dynamic, .{});
        sparse_lp64_install_step.dependOn(&sparse_lp64_dynamic_install.step);
        sparse_lp64_install_step.dependOn(&b.addInstallFileWithDir(sparse_lp64_archive, .lib, "libzynum_sparse_lp64.a").step);
        sparse_lp64_install_step.dependOn(&sparse_lp64_header_install.step);
        sparse_lp64_install_step.dependOn(&sparse_lp64_manifest_install.step);
        const sparse_lp64_files = b.addWriteFiles();
        const sparse_lp64_pc = sparse_lp64_files.add("zynum_sparse_lp64.pc", "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include/zynum/sparse/lp64\nName: zynum_sparse_lp64\nDescription: Zynum finite CSR/CSC create/export/MV/MM interface\nVersion: 0.0.1-beta\nLibs: -L${libdir} -lzynum_sparse_lp64\nCflags: -I${includedir}\n");
        sparse_lp64_install_step.dependOn(&b.addInstallFileWithDir(sparse_lp64_pc, .lib, "pkgconfig/zynum_sparse_lp64.pc").step);
    } else {
        sparse_lp64_install_step.dependOn(&b.addFail("LP64 finite Sparse requires the Apple arm64 declaration contract").step);
    }

    const sparse_ilp64_install_step = b.step("install-sparse-ilp64", "Install finite Sparse ilp64 interface");
    if (target.result.cpu.arch == .aarch64 and target.result.os.tag == .macos) {
        const sparse_ilp64_check = b.addSystemCommand(&.{ "python3", "tools/generate_checked_abi.py", "--profile", "mkl2026.ilp64", "--binding-set", sparse_ilp64_set, "--target-contract", "apple-arm64-default-c-v1", "--zig", b.graph.zig_exe, "--check" });
        const sparse_ilp64_interface = b.createModule(.{ .root_source_file = b.path(if (sparse_solves) "interface_subset/mkl2026.ilp64/sparse-csr-csc-mv-mm-trsv-trsm-ordinary-ilp64-v1/exports.zig" else "interface_subset/mkl2026.ilp64/sparse-csr-csc-mv-mm-ordinary-ilp64-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
        sparse_ilp64_interface.addImport("sparse-interface-owner", mkl_sparse_binding_mod);
        const sparse_ilp64_dynamic = b.addLibrary(.{ .name = "zynum_sparse_ilp64", .linkage = .dynamic, .root_module = sparse_ilp64_interface });
        const sparse_ilp64_static = b.addLibrary(.{ .name = "zynum_sparse_ilp64", .linkage = .static, .root_module = sparse_ilp64_interface });
        sparse_ilp64_static.bundle_compiler_rt = true;
        sparse_ilp64_dynamic.step.dependOn(&sparse_ilp64_check.step);
        sparse_ilp64_static.step.dependOn(&sparse_ilp64_check.step);
        const sparse_ilp64_repack = b.addSystemCommand(&.{ "python3", "tools/repack_darwin_archive.py", b.graph.zig_exe });
        sparse_ilp64_repack.addFileArg(sparse_ilp64_static.getEmittedBin());
        const sparse_ilp64_archive = sparse_ilp64_repack.addOutputFileArg("libzynum_sparse_ilp64.a");
        const sparse_ilp64_dynamic_install = b.addInstallArtifact(sparse_ilp64_dynamic, .{});
        sparse_ilp64_install_step.dependOn(&sparse_ilp64_dynamic_install.step);
        sparse_ilp64_install_step.dependOn(&b.addInstallFileWithDir(sparse_ilp64_archive, .lib, "libzynum_sparse_ilp64.a").step);
        sparse_ilp64_install_step.dependOn(&b.addInstallFileWithDir(b.path(if (sparse_solves) "interface_subset/mkl2026.ilp64/sparse-csr-csc-mv-mm-trsv-trsm-ordinary-ilp64-v1/subset.h" else "interface_subset/mkl2026.ilp64/sparse-csr-csc-mv-mm-ordinary-ilp64-v1/subset.h"), .header, "zynum/sparse/ilp64/mkl_spblas.h").step);
        sparse_ilp64_install_step.dependOn(&b.addInstallFileWithDir(b.path(if (sparse_solves) "interface_subset/mkl2026.ilp64/sparse-csr-csc-mv-mm-trsv-trsm-ordinary-ilp64-v1/abi_manifest.json" else "interface_subset/mkl2026.ilp64/sparse-csr-csc-mv-mm-ordinary-ilp64-v1/abi_manifest.json"), .header, "zynum/sparse/ilp64/abi_manifest.json").step);
        const sparse_ilp64_files = b.addWriteFiles();
        const sparse_ilp64_pc = sparse_ilp64_files.add("zynum_sparse_ilp64.pc", "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include/zynum/sparse/ilp64\nName: zynum_sparse_ilp64\nDescription: Zynum finite CSR/CSC create/export/MV/MM interface\nVersion: 0.0.1-beta\nLibs: -L${libdir} -lzynum_sparse_ilp64\nCflags: -I${includedir} -DMKL_ILP64\n");
        sparse_ilp64_install_step.dependOn(&b.addInstallFileWithDir(sparse_ilp64_pc, .lib, "pkgconfig/zynum_sparse_ilp64.pc").step);
    } else {
        sparse_ilp64_install_step.dependOn(&b.addFail("ILP64 finite Sparse requires the Apple arm64 declaration contract").step);
    }

    for (kernel_libraries) |library| {
        zynum_core_mod.addObject(library);
    }
    for (kernel_test_libraries) |library| zynum_core_test_mod.addObject(library);
    const stride2_isolated_library = if (target.result.cpu.arch == .x86_64) b.addObject(.{
        .name = "zynum-level1-x86-stride2-isolated",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/level1_stride2_object_root.zig"),
            .target = target,
            .optimize = optimize,
            .pic = true,
            .link_libc = true,
        }),
    }) else null;
    const stride2_isolated_test_library = if (target.result.cpu.arch == .x86_64) b.addObject(.{
        .name = "zynum-level1-x86-stride2-isolated-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/level1_stride2_object_root.zig"),
            .target = target,
            .optimize = test_optimize,
            .pic = true,
            .link_libc = true,
        }),
    }) else null;
    const compact_triangular_isolated_library = if (target.result.cpu.arch == .x86_64) b.addObject(.{
        .name = "zynum-level2-x86-compact-triangular-isolated",
        .root_module = b.createModule(.{
            .root_source_file = b.path(compact_triangular_object_root),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    const compact_triangular_isolated_test_library = if (target.result.cpu.arch == .x86_64) b.addObject(.{
        .name = "zynum-level2-x86-compact-triangular-isolated-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path(compact_triangular_object_root),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    const level2_width_isolated_library = if (target.result.cpu.arch == .x86_64) b.addObject(.{
        .name = "zynum-level2-x86-width-isolated",
        .root_module = b.createModule(.{
            .root_source_file = b.path(level2_width_object_root),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    const level2_width_isolated_test_library = if (target.result.cpu.arch == .x86_64) b.addObject(.{
        .name = "zynum-level2-x86-width-isolated-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/level2_width_object_root.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    const structured_isolated_library = if (target.result.cpu.arch == .x86_64 and structured_object_requested) b.addObject(.{
        .name = "zynum-level3-x86-structured-isolated",
        .root_module = b.createModule(.{
            .root_source_file = b.path(structured_object_root),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    const structured_isolated_test_library = if (target.result.cpu.arch == .x86_64) b.addObject(.{
        .name = "zynum-level3-x86-structured-isolated-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/structured_object_root.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .pic = true,
        }),
    }) else null;
    if (stride2_isolated_library) |library| library.root_module.addImport("zynum-build-options", zynum_build_options_mod);
    if (stride2_isolated_test_library) |library| library.root_module.addImport("zynum-build-options", zynum_build_options_mod);
    if (stride2_isolated_library) |library| {
        zynum_core_mod.addObject(library);
    }
    if (compact_triangular_isolated_library) |library| {
        zynum_core_mod.addObject(library);
    }
    if (level2_width_isolated_library) |library| {
        zynum_core_mod.addObject(library);
    }
    if (stride2_isolated_test_library) |library| zynum_core_test_mod.addObject(library);
    if (compact_triangular_isolated_test_library) |library| zynum_core_test_mod.addObject(library);
    if (level2_width_isolated_test_library) |library| zynum_core_test_mod.addObject(library);
    if (structured_isolated_library) |library| {
        zynum_core_mod.addObject(library);
    }
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
    // Zig's archive writer can leave embedded Mach-O members misaligned. Rebuild
    // the container with LLVM ar's Darwin format before installing the single library.
    const repack_static_lib = b.addSystemCommand(&.{ "python3", "tools/repack_darwin_archive.py", b.graph.zig_exe });
    repack_static_lib.addFileArg(static_lib.getEmittedBin());
    const repacked_static_archive = repack_static_lib.addOutputFileArg("libzynum_blas.a");
    const install_static_lib = if (target.result.ofmt == .macho)
        &b.addInstallFileWithDir(repacked_static_archive, .lib, "libzynum_blas.a").step
    else
        &b.addInstallArtifact(static_lib, static_install_options).step;
    b.getInstallStep().dependOn(&install_dynamic_lib.step);
    b.getInstallStep().dependOn(install_static_lib);
    const install_libraries_step = b.step(
        "install-libraries",
        "Install the shared and static Zynum BLAS libraries without tools",
    );
    install_libraries_step.dependOn(&install_dynamic_lib.step);
    install_libraries_step.dependOn(install_static_lib);

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
            .optimize = .safe,
        }),
    });
    const generate_headers = b.addRunArtifact(generate_headers_tool);
    generate_headers.addDirectoryArg2(b.path("."), .{});
    const generate_headers_step = b.step("generate-headers", "Regenerate Zynum BLAS CBLAS and BLAS/Fortran compatibility headers/modules");
    generate_headers_step.dependOn(&generate_headers.step);

    const kernel_coverage_mod = b.createModule(.{
        .root_source_file = b.path("src/blas/kernel_coverage_root.zig"),
        .target = b.graph.host,
        // The generator materializes every descriptor for reporting. Debug
        // avoids spending release-optimization time on a host-only tool.
        .optimize = .debug,
    });
    kernel_coverage_mod.addImport("zynum-numerics-resources", host_numerics_resources_mod);
    const generate_kernel_coverage_tool = b.addExecutable(.{
        .name = "generate_kernel_coverage",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/generate_kernel_coverage.zig"),
            .target = b.graph.host,
            .optimize = .debug,
            .imports = &.{.{ .name = "kernel-coverage", .module = kernel_coverage_mod }},
        }),
    });
    const generate_kernel_coverage = b.addRunArtifact(generate_kernel_coverage_tool);
    generate_kernel_coverage.addDirectoryArg2(b.path("."), .{});
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
                .{ .name = "zynum-blas", .module = zynum_blas_test_mod },
                .{ .name = "zynum-lapack", .module = zynum_lapack_test_mod },
                .{ .name = "zynum-sparse", .module = zynum_sparse_test_mod },
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
            .imports = &.{
                .{ .name = "zynum-blas", .module = zynum_blas_test_mod },
                .{ .name = "zynum-execution", .module = execution_test_mod },
            },
        }),
    });
    const run_blas_module_tests = b.addRunArtifact(blas_module_tests);

    // A source-owned root discovers internal contract tests without exposing
    // implementation types through the public LAPACK or Sparse modules.
    const numerics_contract_tests = b.addTest(.{
        .name = "zynum-numerics-contract-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/numerics_contract_test.zig"),
            .target = target,
            .optimize = test_optimize,
        }),
    });
    const run_numerics_contract_tests = b.addRunArtifact(numerics_contract_tests);
    const numerics_contract_test_step = b.step("test-numerics-contract", "Run shared numerical contract tests");
    numerics_contract_test_step.dependOn(&run_numerics_contract_tests.step);

    // Independent mechanism tests use their explicit fixture policy, never a
    // production default/options instance or architecture-specific kernel objects.
    const execution_contract_tests = b.addTest(.{
        .name = "zynum-execution-contract-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/numerics/execution/owner.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    execution_contract_tests.root_module.addImport("zynum-numerics-resources", numerics_resources_mod);
    const run_execution_contract_tests = b.addRunArtifact(execution_contract_tests);
    const execution_contract_test_step = b.step("test-execution-contract", "Run shared execution owner contract tests");
    execution_contract_test_step.dependOn(&run_execution_contract_tests.step);

    const contour_test_generate = b.addSystemCommand(&.{ "python3", "-B", "tools/generate_contour_interface.py", "--wire-header" });
    for ([_][]const u8{ "tools/generate_contour_interface.py", "tools/contour_wire_profile.json", "tools/interface_templates/contour_wire_types.zig.in" }) |input|
        contour_test_generate.addFileInput(b.path(input));
    _ = contour_test_generate.addOutputFileArg("contour_wire_v1.h");
    contour_test_generate.addArg("--types");
    const contour_test_types = contour_test_generate.addOutputFileArg("contour_v1_types.zig");
    contour_test_generate.addArg("--manifest");
    _ = contour_test_generate.addOutputFileArg("manifest.json");
    const contour_wire_test_mod = b.createModule(.{ .root_source_file = contour_test_types, .target = target, .optimize = test_optimize });
    const general_operator_test_mod = b.createModule(.{
        .root_source_file = b.path("src/general_operator_candidate_test.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    general_operator_test_mod.addImport("zynum-core", zynum_core_test_mod);
    general_operator_test_mod.addImport("zynum-execution", execution_test_mod);
    general_operator_test_mod.addImport("zynum-contour-wire", contour_wire_test_mod);
    general_operator_test_mod.addImport("numerical-owner", lapacke_binding_test_mod);

    const lapack_tests = b.addTest(.{
        .name = "zynum-lapack-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/lapack/lapack_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "general-operator-tests", .module = general_operator_test_mod },
                .{ .name = "zynum-lapack", .module = zynum_lapack_test_mod },
                .{ .name = "numerical-owner", .module = lapacke_binding_test_mod },
                .{ .name = "fftw-interface-owner", .module = fftw_binding_test_mod },
                .{ .name = "zynum-core", .module = zynum_core_test_mod },
                .{ .name = "zynum-execution", .module = execution_test_mod },
            },
        }),
    });
    const lapacke_posv_tests = b.addTest(.{
        .name = "zynum-lapacke-posv-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/lapack/interop/posv_adapter_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "numerical-owner", .module = lapacke_binding_test_mod }},
        }),
    });
    const lapacke_posv_test_step = b.step("test-lapacke-posv", "Run POSV numerical, interface and single-operation lifetime tests");
    lapacke_posv_test_step.dependOn(&b.addRunArtifact(lapacke_posv_tests).step);

    const run_lapack_tests = b.addRunArtifact(lapack_tests);
    const lapack_test_step = b.step("test-lapack", "Run LAPACK foundation tests");
    lapack_test_step.dependOn(&run_lapack_tests.step);
    lapack_test_step.dependOn(numerics_contract_test_step);
    lapack_test_step.dependOn(execution_contract_test_step);

    // Checked adapter tests borrow the same test core and execution owner.
    const checked_sparse_test_mod = b.createModule(.{
        .root_source_file = b.path("src/checked/mixed_adapter_test.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    checked_sparse_test_mod.addImport("zynum-core", zynum_core_test_mod);
    checked_sparse_test_mod.addImport("zynum-execution", execution_test_mod);

    const sparse_tests = b.addTest(.{
        .name = "zynum-sparse-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/sparse/sparse_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "checked-sparse-tests", .module = checked_sparse_test_mod },
                .{ .name = "zynum-sparse", .module = zynum_sparse_test_mod },
                .{ .name = "sparse-interface-owner", .module = mkl_sparse_binding_test_mod },
                .{ .name = "zynum-core", .module = zynum_core_test_mod },
                .{ .name = "zynum-execution", .module = execution_test_mod },
            },
        }),
    });
    const run_sparse_tests = b.addRunArtifact(sparse_tests);
    const sparse_test_step = b.step("test-sparse", "Run Sparse foundation tests");
    sparse_test_step.dependOn(&run_sparse_tests.step);
    sparse_test_step.dependOn(numerics_contract_test_step);
    sparse_test_step.dependOn(execution_contract_test_step);

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
    fortran_tests.root_module.addImport("zynum-execution", execution_test_mod);
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
    cblas_tests.root_module.addImport("zynum-execution", execution_test_mod);
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
        structured_object_test_mod.addObject(structured_isolated_test_library.?);
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
        probe_mod.addObject(level2_width_isolated_library.?);
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
        target.result.cpu.features.isEnabled(@backingInt(std.Target.x86.Feature.avx512f));
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
        probe_mod.addObject(level2_width_isolated_library.?);
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

    const translated_headers = b.addTranslateC(.{
        .root_source_file = b.path("test/headers/compat_headers.h"),
        .target = target,
        .optimize = test_optimize,
    });
    translated_headers.addIncludePath(b.path("include"));
    const header_smoke_mod = b.createModule(.{
        .root_source_file = b.path("test/headers/compat_headers_smoke.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    header_smoke_mod.addImport("compat-headers", translated_headers.createModule());
    if (target.result.cpu.arch == .x86_64) {
        const task_runtime_host_test_library = b.addObject(.{
            .name = "zynum-test-task-runtime-host",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/blas/task_runtime_host_object_root.zig"),
                .target = target,
                .optimize = test_optimize,
                .link_libc = true,
            }),
        });
        task_runtime_host_test_library.root_module.addImport("zynum-build-options", zynum_build_options_mod);
        task_runtime_host_test_library.root_module.addImport("zynum-execution", execution_test_mod);
        header_smoke_mod.addObject(task_runtime_host_test_library);
        triangular_band_window_tests.root_module.addObject(task_runtime_host_test_library);
        triangular_packed_unit_test_mod.addObject(task_runtime_host_test_library);
        triangular_band_solve_test_mod.addObject(task_runtime_host_test_library);
    }
    const header_smoke_tests = b.addTest(.{
        .name = "zynum-blas-header-smoke-tests",
        .root_module = header_smoke_mod,
    });
    const run_header_smoke_tests = b.addRunArtifact(header_smoke_tests);
    const abi_manifest_smoke_test = b.addSystemCommand(&.{
        "python3",
        "-c",
        "import json, pathlib, sys; data=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert data['schema'] == 1; assert data['blas_integer_abi'] == 'LP64'; assert data['fortran']['export_count'] == 161; assert data['cblas']['export_count'] == 150; names={item['name'] for section in ('fortran','cblas') for item in data[section]['exports']}; assert {'dgemm_', 'cdotc_sub_', 'cblas_dgemm', 'cblas_zher2k'} <= names",
    });
    abi_manifest_smoke_test.addFileArg2(b.path("include/zynum/blas/abi_manifest.json"), .{});
    const c_header_smoke_test = b.addSystemCommand(&.{ "cc", "-std=c11", "-I" });
    c_header_smoke_test.addDirectoryArg2(b.path("include"), .{});
    c_header_smoke_test.addArg("-c");
    c_header_smoke_test.addFileArg2(b.path("test/headers/compat_headers_c_smoke.c"), .{});
    c_header_smoke_test.addArg("-o");
    _ = c_header_smoke_test.addOutputFileArg2("compat_headers_c_smoke.o", .{});
    const cpp_header_smoke_test = b.addSystemCommand(&.{ "c++", "-std=c++17", "-I" });
    cpp_header_smoke_test.addDirectoryArg2(b.path("include"), .{});
    cpp_header_smoke_test.addArg("-c");
    cpp_header_smoke_test.addFileArg2(b.path("test/headers/compat_headers_cpp_smoke.cpp"), .{});
    cpp_header_smoke_test.addArg("-o");
    _ = cpp_header_smoke_test.addOutputFileArg2("compat_headers_cpp_smoke.o", .{});
    const fortran_module_smoke_test = b.addSystemCommand(&.{
        "sh",
        "-c",
        "if command -v gfortran >/dev/null 2>&1; then mkdir -p \"$1\" && gfortran -std=f2008 -J \"$1\" -fsyntax-only \"$2\" \"$3\"; fi",
        "zynum-fortran-smoke",
    });
    _ = fortran_module_smoke_test.addOutputDirectoryArg2("fortran-mod", .{});
    fortran_module_smoke_test.addFileArg2(b.path("include/zynum/blas/blas.f90"), .{});
    fortran_module_smoke_test.addFileArg2(b.path("test/headers/fortran_module_smoke.f90"), .{});
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
    });
    abi_artifact_parity_verifier_tests.addFileArg2(b.path("test/abi/baseline/test_abi_artifact_parity.py"), .{});
    const abi_artifact_parity_verifier_test_step = b.step(
        "test-abi-artifact-parity-verifier",
        "Run fresh-artifact parity parser and verifier tests",
    );
    abi_artifact_parity_verifier_test_step.dependOn(&abi_artifact_parity_verifier_tests.step);
    const build_profile_tests = b.addSystemCommand(&.{
        "python3",
        "-B",
    });
    build_profile_tests.addFileArg2(b.path("test/build/test_build_profiles.py"), .{});
    const build_profile_test_step = b.step(
        "test-build-profiles",
        "Check default, single, and conflicting experimental build profiles",
    );
    build_profile_test_step.dependOn(&build_profile_tests.step);
    const checked_sparse_profile_tests = b.addSystemCommand(&.{ "python3", "-B" });
    checked_sparse_profile_tests.addFileArg2(b.path("test/build/test_checked_sparse_profile.py"), .{});
    const checked_sparse_profile_step = b.step("test-checked-sparse-profile", "Validate additive Sparse generation and source contracts");
    checked_sparse_profile_step.dependOn(&checked_sparse_profile_tests.step);
    const lapacke_posv_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_posv_profile.py" });
    const lapacke_posv_profile_step = b.step("test-lapacke-posv-profile", "Validate finite four-scalar POSV generation and early source rejection");
    lapacke_posv_profile_step.dependOn(&lapacke_posv_profile_tests.step);
    const lapacke_trtrs_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_trtrs_profile.py" });
    const lapacke_trtrs_profile_step = b.step("test-lapacke-trtrs-profile", "Validate finite four-scalar triangular solve generation and early source rejection");
    lapacke_trtrs_profile_step.dependOn(&lapacke_trtrs_profile_tests.step);

    const lapacke_trtri_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_trtri_profile.py" });
    const lapacke_trtri_profile_step = b.step("test-lapacke-trtri-profile", "Validate finite four-scalar triangular inverse generation and early source rejection");
    lapacke_trtri_profile_step.dependOn(&lapacke_trtri_profile_tests.step);
    const lapacke_getri_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_getri_profile.py" });
    const lapacke_getri_profile_step = b.step("test-lapacke-getri-profile", "Validate finite four-scalar LU inverse generation and early source rejection");
    lapacke_getri_profile_step.dependOn(&lapacke_getri_profile_tests.step);
    const lapacke_potri_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_potri_profile.py" });
    const lapacke_potri_profile_step = b.step("test-lapacke-potri-profile", "Validate optional four-scalar Cholesky inverse components and accepted base compositions");
    lapacke_potri_profile_step.dependOn(&lapacke_potri_profile_tests.step);
    const lapacke_syev_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_syev_profile.py" });
    const lapacke_syev_profile_step = b.step("test-lapacke-syev-profile", "Validate generated LP64 real symmetric eigenvalue components and same-owner compositions");
    lapacke_syev_profile_step.dependOn(&lapacke_syev_profile_tests.step);
    const lapacke_sygv_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_sygv_profile.py" });
    const lapacke_sygv_profile_step = b.step("test-lapacke-sygv-profile", "Validate generated LP64 generalized real symmetric components and same-owner compositions");
    lapacke_sygv_profile_step.dependOn(&lapacke_sygv_profile_tests.step);
    const lapacke_controls_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_controls_profile.py" });
    const lapacke_controls_profile_step = b.step("test-lapacke-controls-profile", "Validate generated NaN controls, C integer widths and shared configuration ownership");
    lapacke_controls_profile_step.dependOn(&lapacke_controls_profile_tests.step);
    const lapacke_evd_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_evd_profile.py" });
    const lapacke_evd_profile_step = b.step("test-lapacke-evd-profile", "Validate generated EVD workspace signatures and exact shared owner composition");
    lapacke_evd_profile_step.dependOn(&lapacke_evd_profile_tests.step);
    const lapacke_gvd_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_gvd_profile.py" });
    const lapacke_gvd_profile_step = b.step("test-lapacke-gvd-profile", "Validate generated generalized EVD workspace signatures and exact shared owner composition");
    lapacke_gvd_profile_step.dependOn(&lapacke_gvd_profile_tests.step);
    const lapacke_gels_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_gels_profile.py" });
    const lapacke_gels_profile_step = b.step("test-lapacke-gels-profile", "Validate generated full-rank least-squares signatures and exact shared owner composition");
    lapacke_gels_profile_step.dependOn(&lapacke_gels_profile_tests.step);
    const lapacke_gelss_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_lapacke_gelss_profile.py" });
    const lapacke_gelss_profile_step = b.step("test-lapacke-gelss-profile", "Validate generated SVD least-squares signatures and exact shared owner composition");
    lapacke_gelss_profile_step.dependOn(&lapacke_gelss_profile_tests.step);
    const checked_cholesky_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_checked_cholesky_profile.py" });
    const checked_cholesky_profile_step = b.step("test-checked-cholesky-profile", "Validate additive checked Cholesky generation contracts");
    checked_cholesky_profile_step.dependOn(&checked_cholesky_profile_tests.step);
    const feast_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_feast_profile.py" });
    const feast_profile_step = b.step("test-feast-profile", "Run finite FEAST profile and shared Core build contract tests");
    feast_profile_step.dependOn(&feast_profile_tests.step);
    const contour_profile_tests = b.addSystemCommand(&.{ "python3", "-B", "test/build/test_contour_profile.py" });
    const contour_profile_step = b.step("test-operator-contour-profile", "Validate optional neutral Contour generation, wire and shared-owner contracts");
    contour_profile_step.dependOn(&contour_profile_tests.step);
    const checked_cholesky_tests = b.addTest(.{
        .name = "zynum-checked-cholesky-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/checked/cholesky_adapter_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "zynum-core", .module = zynum_core_test_mod },
                .{ .name = "zynum-execution", .module = execution_test_mod },
            },
        }),
    });
    const checked_cholesky_lifecycle_step = b.step("test-checked-cholesky-lifecycle", "Check generated complex calls and caller factor lifetime across Context close");
    checked_cholesky_lifecycle_step.dependOn(&b.addRunArtifact(checked_cholesky_tests).step);
    const build_inventory_tests = b.addSystemCommand(&.{
        "python3",
        "-B",
    });
    build_inventory_tests.addFileArg2(b.path("test/build/test_build_inventory.py"), .{});
    const build_inventory_test_step = b.step(
        "test-build-inventory",
        "Validate the repository build and launch inventory",
    );
    build_inventory_test_step.dependOn(&build_inventory_tests.step);
    const test_inventory_structure_check = b.addSystemCommand(&.{
        "python3",
        "-B",
    });
    test_inventory_structure_check.addFileArg2(b.path("tools/check_test_inventory.py"), .{});
    test_inventory_structure_check.addArg("--root");
    test_inventory_structure_check.addDirectoryArg2(b.path("."), .{});
    test_inventory_structure_check.addArg("--structure-only");
    test_inventory_structure_check.removeEnvironmentVariable("GIT_PAGER");
    test_inventory_structure_check.removeEnvironmentVariable("PAGER");
    const test_inventory_security_tests = b.addSystemCommand(&.{
        "python3",
        "-B",
    });
    test_inventory_security_tests.addFileArg2(b.path("test/build/test_test_inventory.py"), .{});
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
    });
    python_tooling_tests.addFileArg2(b.path("tools/check_test_inventory.py"), .{});
    python_tooling_tests.addArg("--root");
    python_tooling_tests.addDirectoryArg2(b.path("."), .{});
    python_tooling_tests.addArg("--inventory");
    python_tooling_tests.addFileArg2(b.path("tools/test_inventory.json"), .{});
    python_tooling_tests.addArg("--run-python-tooling-root");
    python_tooling_tests.addArg("python-root:benchmark-tools-discovery");
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
    const host_tool_smoke_test_step = b.step(
        "test-host-tool-smoke",
        "Run host Python, C/C++, available Fortran, and ABI observer smoke checks",
    );
    host_tool_smoke_test_step.dependOn(python_tooling_test_step);
    host_tool_smoke_test_step.dependOn(&abi_manifest_smoke_test.step);
    host_tool_smoke_test_step.dependOn(&c_header_smoke_test.step);
    host_tool_smoke_test_step.dependOn(&cpp_header_smoke_test.step);
    host_tool_smoke_test_step.dependOn(&fortran_module_smoke_test.step);
    host_tool_smoke_test_step.dependOn(abi_baseline_observer_test_step);
    host_tool_smoke_test_step.dependOn(build_profile_test_step);

    const InventoryCase = struct {
        root_id: []const u8,
        logical_tests: ?*std.Build.Step.Compile,
    };
    const inventory_cases = [25]InventoryCase{
        .{ .root_id = "zig-root:blas-module-tests", .logical_tests = blas_module_tests },
        .{ .root_id = "zig-root:blas-public-surface-contract-tests", .logical_tests = blas_public_surface_contract_tests },
        .{ .root_id = "zig-root:cblas-tests", .logical_tests = cblas_tests },
        .{ .root_id = "zig-root:execution-contract-tests", .logical_tests = execution_contract_tests },
        .{ .root_id = "zig-root:fortran-tests", .logical_tests = fortran_tests },
        .{ .root_id = "zig-root:gemm-registry-tests", .logical_tests = gemm_registry_tests },
        .{ .root_id = "zig-root:header-smoke-tests", .logical_tests = header_smoke_tests },
        .{ .root_id = "zig-root:lapack-tests", .logical_tests = lapack_tests },
        .{ .root_id = "zig-root:level1-registry-tests", .logical_tests = level1_registry_tests },
        .{ .root_id = "zig-root:level2-compact-registry-tests", .logical_tests = level2_compact_registry_tests },
        .{ .root_id = "zig-root:level2-fused-registry-tests", .logical_tests = level2_fused_registry_tests },
        .{ .root_id = "zig-root:modern-tests", .logical_tests = modern_tests },
        .{ .root_id = "zig-root:numerics-contract-tests", .logical_tests = numerics_contract_tests },
        .{ .root_id = "zig-root:packed-parallel-tests", .logical_tests = packed_parallel_tests },
        .{ .root_id = "zig-root:sparse-tests", .logical_tests = sparse_tests },
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
    for (inventory_cases) |inventory_case| {
        const official_tests = inventory_case.logical_tests orelse continue;
        // The modern consumer imports only public facades; its options belong
        // to their shared owner. Other roots retain their existing bindings.
        if (official_tests != modern_tests and official_tests != execution_contract_tests) {
            official_tests.root_module.addImport("zynum-build-options", zynum_build_options_mod);
        }
        if (official_tests != execution_contract_tests) official_tests.root_module.addImport("zynum-numerics-resources", numerics_resources_mod);
        const uses_direct_execution = official_tests == gemm_registry_tests or official_tests == level1_registry_tests or
            official_tests == level2_fused_registry_tests or official_tests == level2_compact_registry_tests or
            official_tests == symm_dense_gemm_tests or official_tests == triangular_parallel_tests or
            official_tests == structured_blocked_tests or official_tests == structured_object_tests or
            official_tests == packed_parallel_tests or official_tests == triangular_dense_unit_tests or
            official_tests == triangular_band_window_tests or official_tests == triangular_packed_unit_tests or
            official_tests == triangular_band_solve_tests or official_tests == vector_stride2_parallel_tests or
            official_tests == numerics_contract_tests;
        if (uses_direct_execution) official_tests.root_module.addImport("zynum-execution", execution_test_mod);
        // These consumers already inherit implementation objects from their
        // shared facade owner. Other internal/ABI roots retain their own links.
        const uses_facade_owner = official_tests == modern_tests or official_tests == blas_module_tests or
            official_tests == lapack_tests or official_tests == sparse_tests or official_tests == fortran_tests or official_tests == cblas_tests or
            official_tests == zynum_public_surface_contract_tests or official_tests == blas_public_surface_contract_tests;
        if (!uses_facade_owner and official_tests != numerics_contract_tests and official_tests != execution_contract_tests) {
            for (kernel_test_libraries) |library| official_tests.root_module.addObject(library);
            if (stride2_isolated_test_library) |library| official_tests.root_module.addObject(library);
            if (compact_triangular_isolated_test_library) |library| official_tests.root_module.addObject(library);
            if (level2_width_isolated_test_library) |library| official_tests.root_module.addObject(library);
        }
    }
    const test_inventory_link_step = b.step(
        "test-inventory-link",
        "Compile every applicable test-inventory enumerator without running it",
    );
    const test_inventory_link_windows_native_smoke_step = b.step(
        "test-inventory-link-windows-native-smoke",
        "Compile/link the 25 canonical native Windows GNU inventory artifacts without debug information as compatibility-only evidence",
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
            const root_id_prefix = "zig-root:";
            if (!std.mem.startsWith(u8, inventory_case.root_id, root_id_prefix)) {
                @panic("test inventory root ID must start with zig-root:");
            }
            const inventory_tests = b.addTest(.{
                .name = b.fmt("inventory-zig-root-{s}", .{inventory_case.root_id[root_id_prefix.len..]}),
                .root_module = official_tests.root_module,
                .test_runner = .{
                    .path = b.path("tools/test_inventory_runner.zig"),
                    .mode = .simple,
                },
            });
            if (target.result.os.tag == .windows and target.result.ofmt == .coff) {
                inventory_tests.root_module.strip = true;
            }
            const run_inventory_tests = b.addRunArtifact(inventory_tests);
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
            if (native_canonical_windows_inventory_link) {
                test_inventory_link_windows_native_smoke_step.dependOn(&inventory_tests.step);
            }
            test_inventory_step.dependOn(&run_inventory_tests.step);
        }
    } else {
        const unsupported_test_inventory_target = b.addFail(
            "test inventory enumeration is unavailable for the requested target CPU profile",
        );
        test_inventory_link_step.dependOn(&unsupported_test_inventory_target.step);
        test_inventory_step.dependOn(&unsupported_test_inventory_target.step);
    }
    if (!native_canonical_windows_inventory_link) {
        const unsupported_windows_native_inventory_link = b.addFail(
            "test-inventory-link-windows-native-smoke requires an x86_64 Windows host and the exact x86_64-windows-gnu COFF target requested with -Dcpu=baseline and no CPU feature modifiers",
        );
        test_inventory_link_windows_native_smoke_step.dependOn(&unsupported_windows_native_inventory_link.step);
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
    // Executor flags are make-time inputs in Zig 0.17. Check them in a Run
    // step so cached build configurations cannot bypass native-only evidence.
    const native_execution_check = b.addSystemCommand(&.{ "python3", "tools/check_test_inventory.py", "--check-native-executors" });
    native_execution_check.addThirdPartyEnabledArgQemu(.{ .enabled = "--native-executor=qemu" });
    native_execution_check.addThirdPartyEnabledArgRosetta(.{ .enabled = "--native-executor=rosetta" });
    native_execution_check.addThirdPartyEnabledArgWine(.{ .enabled = "--native-executor=wine" });
    native_execution_check.addThirdPartyEnabledArgDarling(.{ .enabled = "--native-executor=darling" });
    native_execution_check.addThirdPartyEnabledArgWasmtime(.{ .enabled = "--native-executor=wasmtime" });
    // Standalone correctness process: no default Zig test-runner Io can mask
    // the two owned pools' process-signal restoration. Always baseline here.
    const runtime_lifecycle_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .macos,
        .abi = .none,
        .cpu_model = .baseline,
    });
    const runtime_lifecycle_probe_options = b.addOptions();
    runtime_lifecycle_probe_options.addOption(bool, "dynamic_dispatch", false);
    runtime_lifecycle_probe_options.addOption(usize, "thread_limit", 2);
    const runtime_lifecycle_options_mod = runtime_lifecycle_probe_options.createModule();
    const runtime_lifecycle_execution_mod = b.createModule(.{
        .root_source_file = b.path("src/numerics/execution/default.zig"),
        .target = runtime_lifecycle_target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    runtime_lifecycle_execution_mod.addImport("zynum-build-options", runtime_lifecycle_options_mod);
    // This independent artifact has a fixed target, which can differ from
    // both the main target and host. Reuse only an identical target's owner.
    const runtime_lifecycle_resources_mod = if (std.Target.Query.eql(
        std.Target.Query.fromTarget(&runtime_lifecycle_target.result),
        std.Target.Query.fromTarget(&target.result),
    ) and runtime_lifecycle_target.result.ofmt == target.result.ofmt and
        runtime_lifecycle_target.result.dynamic_linker.eql(target.result.dynamic_linker))
        numerics_resources_mod
    else if (std.Target.Query.eql(
        std.Target.Query.fromTarget(&runtime_lifecycle_target.result),
        std.Target.Query.fromTarget(&b.graph.host.result),
    ) and runtime_lifecycle_target.result.ofmt == b.graph.host.result.ofmt and
        runtime_lifecycle_target.result.dynamic_linker.eql(b.graph.host.result.dynamic_linker))
        host_numerics_resources_mod
    else
        b.createModule(.{
            .root_source_file = b.path("src/numerics/resources.zig"),
            .target = runtime_lifecycle_target,
        });
    runtime_lifecycle_execution_mod.addImport("zynum-numerics-resources", runtime_lifecycle_resources_mod);
    const runtime_lifecycle_probe = b.addExecutable(.{
        .name = "zynum-runtime-lifecycle-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/runtime_lifecycle_probe.zig"),
            .target = runtime_lifecycle_target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    runtime_lifecycle_probe.root_module.addImport("zynum-build-options", runtime_lifecycle_options_mod);
    runtime_lifecycle_probe.root_module.addImport("zynum-execution", runtime_lifecycle_execution_mod);
    runtime_lifecycle_probe.root_module.addCSourceFile(.{
        .file = b.path("test/runtime/blas_io_lifecycle.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" },
    });
    const run_runtime_lifecycle_tests = b.addSystemCommand(&.{ "python3", "test/runtime/run_blas_io_lifecycle.py" });
    run_runtime_lifecycle_tests.addFileArg(runtime_lifecycle_probe.getEmittedBin());
    const runtime_lifecycle_native_supported = b.graph.host.result.cpu.arch == .aarch64 and
        b.graph.host.result.os.tag == .macos and
        target.result.cpu.arch == .aarch64 and target.result.os.tag == .macos and
        target.result.abi == .none and target.result.ofmt == .macho;
    const test_runtime_lifecycle_step = b.step("test-runtime-lifecycle", "Run isolated native macOS AArch64 baseline runtime lifecycle checks; not inventory evidence");
    if (runtime_lifecycle_native_supported) {
        run_runtime_lifecycle_tests.step.dependOn(&native_execution_check.step);
        run_runtime_lifecycle_tests.step.dependOn(&test_inventory_structure_check.step);
        test_runtime_lifecycle_step.dependOn(&run_runtime_lifecycle_tests.step);
    } else {
        const unsupported_runtime_lifecycle = b.addFail("test-runtime-lifecycle requires native macOS AArch64 and a macOS AArch64 requested target without external executors");
        test_runtime_lifecycle_step.dependOn(&unsupported_runtime_lifecycle.step);
    }

    // Common declarations are installed once by either interface selection.
    const checked_header_install = b.addInstallFileWithDir(b.path("include/zynum/checked/checked.h"), .header, "zynum/checked/checked.h");
    const checked_fortran_install = b.addInstallFileWithDir(b.path("include/zynum/checked/checked.f90"), .header, "zynum/checked/checked.f90");
    const checked_manifest_install = b.addInstallFileWithDir(b.path("include/zynum/checked/abi_manifest.json"), .header, "zynum/checked/abi_manifest.json");

    const abplas_lp64_install = b.step("install-abplas-lp64", "Install one-owner finite LP64 LAPACKE and checked Sparse interfaces");
    // Unified additive artifact, using the same numerical module owners.
    if (target.result.os.tag == .macos and target.result.cpu.arch == .aarch64) {
        const csp = b.createModule(.{ .root_source_file = b.path("src/checked_sparse.zig"), .target = target, .optimize = optimize, .link_libc = true });
        csp.addImport("zynum-core", zynum_core_mod);
        csp.addImport("zynum-execution", execution_mod);
        csp.addImport("zynum-numerics-resources", numerics_resources_mod);
        csp.addImport("zynum-build-options", zynum_build_options_mod);
        const csp_options = b.addOptions();
        csp_options.addOption(bool, "checked_complex_cholesky", checked_complex_cholesky);
        csp.addImport("zynum-checked-options", csp_options.createModule());
        const csp_check = b.addSystemCommand(&.{ "python3", "tools/generate_checked_sparse.py", "--profile", "checked-sparse-v1", "--binding-set", "csr-products-independent-indices-v1", "--target-contract", "apple-arm64-default-c-v1", "--zig", b.graph.zig_exe, "--check" });
        const csp_static = b.addLibrary(.{ .name = "zynum_checked_sparse", .linkage = .static, .root_module = csp });
        csp_static.bundle_compiler_rt = true;
        csp_static.step.dependOn(&csp_check.step);
        const cholesky_check = b.addSystemCommand(&.{ "python3", "tools/generate_checked_cholesky.py", "--profile", "checked-cholesky-v1", "--target-contract", "apple-arm64-default-c-v1", "--zig", b.graph.zig_exe, "--check" });
        if (checked_complex_cholesky) csp_static.step.dependOn(&cholesky_check.step);
        const csp_repack = b.addSystemCommand(&.{ "python3", "tools/repack_darwin_archive.py", b.graph.zig_exe });
        csp_repack.addFileArg(csp_static.getEmittedBin());
        const csp_archive = csp_repack.addOutputFileArg("libzynum_checked_sparse.a");
        const csp_shared = b.addSystemCommand(&.{ "python3", "tools/link_darwin_interface.py" });
        csp_shared.addFileArg(csp_archive);
        csp_shared.addFileArg(b.path("include/zynum/checked/sparse_manifest.json"));
        const csp_shared_output = csp_shared.addOutputFileArg("libzynum_checked_sparse.dylib");
        const csp_macos_min = target.result.os.version_range.semver.min;
        csp_shared.addArgs(&.{ "--macos-min", b.fmt("{d}.{d}.{d}", .{ csp_macos_min.major, csp_macos_min.minor, csp_macos_min.patch }), "--legacy-manifest" });
        csp_shared.addFileArg(b.path("include/zynum/checked/abi_manifest.json"));
        if (checked_complex_cholesky) {
            csp_shared.addArg("--checked-cholesky-manifest");
            csp_shared.addFileArg(b.path("include/zynum/checked/cholesky_manifest.json"));
        }
        const csp_build = b.step("build-checked-sparse", "Build unified experimental checked/Sparse libraries");
        csp_build.dependOn(&csp_static.step);
        csp_build.dependOn(&csp_shared.step);
        const csp_install = b.step("install-checked-sparse", "Install unified experimental checked/Sparse libraries");
        csp_install.dependOn(&b.addInstallFileWithDir(csp_archive, .lib, "libzynum_checked_sparse.a").step);
        const csp_shared_install = b.addInstallFileWithDir(csp_shared_output, .lib, "libzynum_checked_sparse.dylib");
        csp_install.dependOn(&csp_shared_install.step);
        csp_install.dependOn(&checked_header_install.step);
        const csp_sparse_header_install = b.addInstallFileWithDir(b.path("include/zynum/checked/sparse.h"), .header, "zynum/checked/sparse.h");
        csp_install.dependOn(&csp_sparse_header_install.step);
        const checked_complex_header_install = b.addInstallFileWithDir(b.path("include/zynum/checked/complex.h"), .header, "zynum/checked/complex.h");
        csp_install.dependOn(&checked_complex_header_install.step);
        const checked_cholesky_header_install = b.addInstallFileWithDir(b.path("include/zynum/checked/cholesky.h"), .header, "zynum/checked/cholesky.h");
        const checked_cholesky_manifest_install = b.addInstallFileWithDir(b.path("include/zynum/checked/cholesky_manifest.json"), .header, "zynum/checked/cholesky_manifest.json");
        if (checked_complex_cholesky) {
            csp_install.dependOn(&checked_cholesky_header_install.step);
            csp_install.dependOn(&checked_cholesky_manifest_install.step);
        }
        csp_install.dependOn(&checked_fortran_install.step);
        csp_install.dependOn(&checked_manifest_install.step);
        const csp_sparse_manifest_install = b.addInstallFileWithDir(b.path("include/zynum/checked/sparse_manifest.json"), .header, "zynum/checked/sparse_manifest.json");
        csp_install.dependOn(&csp_sparse_manifest_install.step);

        // Both export modules reuse the same exact Core and interface binding.
        const abplas_lp64_lapacke = b.createModule(.{ .root_source_file = b.path(b.fmt("interface_subset/mkl2026.lp64/{s}/exports.zig", .{lapacke_lp64_set})), .target = target, .optimize = optimize, .link_libc = true });
        abplas_lp64_lapacke.addImport("lapacke-interface-owner", lapacke_binding_mod);
        var abplas_lp64_lapacke_selected = abplas_lp64_lapacke;
        if (lapacke_potri) {
            const abplas_lapacke_potri_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/potri-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lapacke_potri_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const abplas_lapacke_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lapacke_composed_root.addImport("zynum-core", zynum_core_mod);
            abplas_lapacke_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            abplas_lapacke_composed_root.addImport("zynum-lapacke-base-exports", abplas_lp64_lapacke);
            abplas_lapacke_composed_root.addImport("zynum-lapacke-component-exports", abplas_lapacke_potri_component);
            abplas_lp64_lapacke_selected = abplas_lapacke_composed_root;
        }
        const abplas_lp64_lapacke_before_syev = abplas_lp64_lapacke_selected;
        var abplas_lp64_lapacke_syev_selected = abplas_lp64_lapacke_before_syev;
        if (lapacke_syev) {
            const abplas_lp64_lapacke_syev_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/syev-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_syev_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const abplas_lp64_lapacke_syev_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_symmetric_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_syev_composed_root.addImport("zynum-core", zynum_core_mod);
            abplas_lp64_lapacke_syev_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            abplas_lp64_lapacke_syev_composed_root.addImport("zynum-lapacke-base-exports", abplas_lp64_lapacke_before_syev);
            abplas_lp64_lapacke_syev_composed_root.addImport("zynum-lapacke-symmetric-exports", abplas_lp64_lapacke_syev_component);
            abplas_lp64_lapacke_syev_selected = abplas_lp64_lapacke_syev_composed_root;
        }
        const abplas_lp64_lapacke_before_sygv = abplas_lp64_lapacke_syev_selected;
        var abplas_lp64_lapacke_sygv_selected = abplas_lp64_lapacke_before_sygv;
        if (lapacke_sygv) {
            const abplas_lp64_lapacke_sygv_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/sygv-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_sygv_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const abplas_lp64_lapacke_sygv_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_symmetric_generalized_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_sygv_composed_root.addImport("zynum-core", zynum_core_mod);
            abplas_lp64_lapacke_sygv_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            abplas_lp64_lapacke_sygv_composed_root.addImport("zynum-lapacke-base-exports", abplas_lp64_lapacke_before_sygv);
            abplas_lp64_lapacke_sygv_composed_root.addImport("zynum-lapacke-symmetric-generalized-exports", abplas_lp64_lapacke_sygv_component);
            abplas_lp64_lapacke_sygv_selected = abplas_lp64_lapacke_sygv_composed_root;
        }
        var abplas_lp64_lapacke_controls_selected = abplas_lp64_lapacke_sygv_selected;
        if (lapacke_controls) {
            const abplas_lp64_lapacke_controls_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/controls-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_controls_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const abplas_lp64_lapacke_controls_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_controls_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_controls_composed_root.addImport("zynum-core", zynum_core_mod);
            abplas_lp64_lapacke_controls_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            abplas_lp64_lapacke_controls_composed_root.addImport("zynum-lapacke-base-exports", abplas_lp64_lapacke_sygv_selected);
            abplas_lp64_lapacke_controls_composed_root.addImport("zynum-lapacke-controls-exports", abplas_lp64_lapacke_controls_component);
            abplas_lp64_lapacke_controls_selected = abplas_lp64_lapacke_controls_composed_root;
        }
        var abplas_lp64_lapacke_evd_selected = abplas_lp64_lapacke_controls_selected;
        if (lapacke_evd) {
            const abplas_lp64_lapacke_evd_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/evd-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_evd_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const abplas_lp64_lapacke_evd_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_evd_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_evd_composed_root.addImport("zynum-core", zynum_core_mod);
            abplas_lp64_lapacke_evd_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            abplas_lp64_lapacke_evd_composed_root.addImport("zynum-lapacke-base-exports", abplas_lp64_lapacke_controls_selected);
            abplas_lp64_lapacke_evd_composed_root.addImport("zynum-lapacke-evd-exports", abplas_lp64_lapacke_evd_component);
            abplas_lp64_lapacke_evd_selected = abplas_lp64_lapacke_evd_composed_root;
        }
        var abplas_lp64_lapacke_gvd_selected = abplas_lp64_lapacke_evd_selected;
        if (lapacke_gvd) {
            const abplas_lp64_lapacke_gvd_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/gvd-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_gvd_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const abplas_lp64_lapacke_gvd_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_gvd_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_gvd_composed_root.addImport("zynum-core", zynum_core_mod);
            abplas_lp64_lapacke_gvd_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            abplas_lp64_lapacke_gvd_composed_root.addImport("zynum-lapacke-base-exports", abplas_lp64_lapacke_evd_selected);
            abplas_lp64_lapacke_gvd_composed_root.addImport("zynum-lapacke-gvd-exports", abplas_lp64_lapacke_gvd_component);
            abplas_lp64_lapacke_gvd_selected = abplas_lp64_lapacke_gvd_composed_root;
        }
        var abplas_lp64_lapacke_gels_selected = abplas_lp64_lapacke_gvd_selected;
        if (lapacke_gels) {
            const abplas_lp64_lapacke_gels_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/gels-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_gels_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const abplas_lp64_lapacke_gels_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_gels_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_gels_composed_root.addImport("zynum-core", zynum_core_mod);
            abplas_lp64_lapacke_gels_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            abplas_lp64_lapacke_gels_composed_root.addImport("zynum-lapacke-base-exports", abplas_lp64_lapacke_gvd_selected);
            abplas_lp64_lapacke_gels_composed_root.addImport("zynum-lapacke-gels-exports", abplas_lp64_lapacke_gels_component);
            abplas_lp64_lapacke_gels_selected = abplas_lp64_lapacke_gels_composed_root;
        }
        var abplas_lp64_lapacke_gelss_selected = abplas_lp64_lapacke_gels_selected;
        if (lapacke_gelss) {
            const abplas_lp64_lapacke_gelss_component = b.createModule(.{ .root_source_file = b.path("interface_components/mkl2026.lp64/gelss-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_gelss_component.addImport("lapacke-interface-owner", lapacke_binding_mod);
            const abplas_lp64_lapacke_gelss_composed_root = b.createModule(.{ .root_source_file = b.path("src/lapacke_gelss_composed.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_lp64_lapacke_gelss_composed_root.addImport("zynum-core", zynum_core_mod);
            abplas_lp64_lapacke_gelss_composed_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
            abplas_lp64_lapacke_gelss_composed_root.addImport("zynum-lapacke-base-exports", abplas_lp64_lapacke_gels_selected);
            abplas_lp64_lapacke_gelss_composed_root.addImport("zynum-lapacke-gelss-exports", abplas_lp64_lapacke_gelss_component);
            abplas_lp64_lapacke_gelss_selected = abplas_lp64_lapacke_gelss_composed_root;
        }
        const abplas_lp64_root = b.createModule(.{ .root_source_file = b.path(if (fft_interface == .fftw3) "src/abplas_fftw_lp64.zig" else "src/abplas_lp64.zig"), .target = target, .optimize = optimize, .link_libc = true });
        abplas_lp64_root.addImport("zynum-core", zynum_core_mod);
        abplas_lp64_root.addImport("lapacke-interface-owner", lapacke_binding_mod);
        abplas_lp64_root.addImport("zynum-lapacke-exports", abplas_lp64_lapacke_gelss_selected);
        abplas_lp64_root.addImport("zynum-checked-sparse-exports", csp);
        const abplas_lp64_defaults_mod = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
        abplas_lp64_defaults_mod.addCSourceFile(.{ .file = b.path("src/interop/lapacke_default_errors.c"), .flags = &.{ "-std=c11", "-fno-sanitize=undefined" } });
        const abplas_lp64_defaults = b.addObject(.{ .name = "abplas-lp64-default-errors", .root_module = abplas_lp64_defaults_mod });
        abplas_lp64_root.addObject(abplas_lp64_defaults);
        // One named FFT binding Module is shared by both optional frontends.
        // It is compiled only when an enabled frontend imports it.
        const fftw_binding_mod = b.createModule(.{ .root_source_file = b.path("src/interop/fftw_binding.zig"), .target = target, .optimize = optimize, .link_libc = true });
        fftw_binding_mod.addImport("zynum-core", zynum_core_mod);
        var abplas_lp64_selected_root = abplas_lp64_root;
        var operator_manifest: ?std.Build.LazyPath = null;
        var contour_manifest: ?std.Build.LazyPath = null;
        var operator_binding_owner: ?*std.Build.Module = null;
        if (operator_interface == .zynum_v1) {
            const operator_generate = b.addSystemCommand(&.{ "python3", "tools/generate_operator_interface.py", "--profile", "zynum-v1", "--header" });
            operator_generate.addFileInput(b.path("tools/generate_operator_interface.py"));
            const operator_header = operator_generate.addOutputFileArg("operator_v1.h");
            operator_generate.addArg("--exports");
            const operator_exports = operator_generate.addOutputFileArg("operator_v1.zig");
            operator_generate.addArg("--manifest");
            const operator_manifest_output = operator_generate.addOutputFileArg("manifest.json");
            if (fft_interface == .fftw3) operator_generate.addArg("--with-fftw3");
            if (operator_phase_agreement) {
                operator_generate.addArg("--with-phase-agreement");
                operator_generate.addFileInput(b.path("src/numerics/synchronization.zig"));
            }
            const operator_binding_mod = b.createModule(.{ .root_source_file = b.path("src/interop/operator_binding.zig"), .target = target, .optimize = optimize });
            operator_binding_owner = operator_binding_mod;
            const operator_exports_mod = b.createModule(.{ .root_source_file = operator_exports, .target = target, .optimize = optimize });
            operator_exports_mod.addImport("zynum-core", zynum_core_mod);
            operator_exports_mod.addImport("checked-interface-owner", csp);
            operator_exports_mod.addImport("operator-interface-owner", operator_binding_mod);
            if (fft_interface == .fftw3) operator_exports_mod.addImport("fftw-interface-owner", fftw_binding_mod);
            const operator_combined_mod = b.createModule(.{ .root_source_file = b.path("src/abplas_operator_lp64.zig"), .target = target, .optimize = optimize, .link_libc = true });
            operator_combined_mod.addImport("zynum-core", zynum_core_mod);
            operator_combined_mod.addImport("checked-interface-owner", csp);
            operator_combined_mod.addImport("operator-interface-owner", operator_binding_mod);
            operator_combined_mod.addImport("zynum-abplas-base-exports", abplas_lp64_root);
            operator_combined_mod.addImport("zynum-operator-exports", operator_exports_mod);
            abplas_lp64_selected_root = operator_combined_mod;
            operator_manifest = operator_manifest_output;
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(operator_header, .header, "zynum/operator/operator_v1.h").step);
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(operator_manifest.?, .header, "zynum/operator/abi_manifest.json").step);
        }
        const abplas_lp64_pre_contour_root = abplas_lp64_selected_root;
        if (contour_interface == .zynum_v1 and operator_binding_owner != null and operator_phase_agreement) {
            const contour_generate = b.addSystemCommand(&.{ "python3", "-B", "tools/generate_contour_interface.py", "--with-entries", "--wire-header" });
            for ([_][]const u8{ "tools/generate_contour_interface.py", "tools/contour_wire_profile.json", "tools/interface_templates/contour_wire_types.zig.in", "tools/generate_operator_interface.py", "src/numerics/synchronization.zig" }) |input|
                contour_generate.addFileInput(b.path(input));
            const contour_wire_header = contour_generate.addOutputFileArg("contour_wire_v1.h");
            contour_generate.addArg("--types");
            const contour_types = contour_generate.addOutputFileArg("contour_v1_types.zig");
            contour_generate.addArg("--header");
            const contour_header = contour_generate.addOutputFileArg("contour_v1.h");
            contour_generate.addArg("--exports");
            const contour_exports = contour_generate.addOutputFileArg("contour_v1.zig");
            contour_generate.addArg("--manifest");
            contour_manifest = contour_generate.addOutputFileArg("manifest.json");
            const contour_wire_mod = b.createModule(.{ .root_source_file = contour_types, .target = target, .optimize = optimize });
            const contour_exports_mod = b.createModule(.{ .root_source_file = contour_exports, .target = target, .optimize = optimize });
            contour_exports_mod.addImport("zynum-core", zynum_core_mod);
            contour_exports_mod.addImport("checked-interface-owner", csp);
            contour_exports_mod.addImport("operator-interface-owner", operator_binding_owner.?);
            contour_exports_mod.addImport("zynum-contour-wire", contour_wire_mod);
            const contour_combined_mod = b.createModule(.{ .root_source_file = b.path("src/abplas_contour_lp64.zig"), .target = target, .optimize = optimize, .link_libc = true });
            contour_combined_mod.addImport("zynum-core", zynum_core_mod);
            contour_combined_mod.addImport("checked-interface-owner", csp);
            contour_combined_mod.addImport("operator-interface-owner", operator_binding_owner.?);
            contour_combined_mod.addImport("zynum-abplas-base-exports", abplas_lp64_pre_contour_root);
            contour_combined_mod.addImport("zynum-contour-exports", contour_exports_mod);
            abplas_lp64_selected_root = contour_combined_mod;
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(contour_header, .header, "zynum/operator/contour_v1.h").step);
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(contour_wire_header, .header, "zynum/operator/contour_wire_v1.h").step);
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(contour_manifest.?, .header, "zynum/operator/contour_abi_manifest.json").step);
        }
        if (sparse_solves) {
            const abplas_standard_sparse_exports = b.createModule(.{ .root_source_file = b.path("interface_subset/mkl2026.lp64/sparse-csr-csc-mv-mm-trsv-trsm-ordinary-lp64-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            abplas_standard_sparse_exports.addImport("sparse-interface-owner", mkl_sparse_binding_mod);
            const sparse_combined_mod = b.createModule(.{ .root_source_file = b.path("src/abplas_sparse_lp64.zig"), .target = target, .optimize = optimize, .link_libc = true });
            sparse_combined_mod.addImport("zynum-core", zynum_core_mod);
            sparse_combined_mod.addImport("sparse-interface-owner", mkl_sparse_binding_mod);
            sparse_combined_mod.addImport("zynum-abplas-base-exports", abplas_lp64_selected_root);
            sparse_combined_mod.addImport("zynum-standard-sparse-exports", abplas_standard_sparse_exports);
            abplas_lp64_selected_root = sparse_combined_mod;
            abplas_lp64_install.dependOn(&sparse_lp64_header_install.step);
            abplas_lp64_install.dependOn(&sparse_lp64_manifest_install.step);
        }
        const abplas_lp64_pre_blas_root = abplas_lp64_selected_root;
        if (blas_interface == .lp64) {
            const blas_combined_mod = b.createModule(.{ .root_source_file = b.path("src/abplas_blas_lp64.zig"), .target = target, .optimize = optimize, .link_libc = true });
            blas_combined_mod.addImport("zynum-core", zynum_core_mod);
            blas_combined_mod.addImport("zynum-blas-abi", blas_abi_mod);
            blas_combined_mod.addImport("zynum-abplas-base-exports", abplas_lp64_pre_blas_root);
            abplas_lp64_selected_root = blas_combined_mod;
            const abplas_blas_cblas_header = b.path("include/zynum/blas/cblas.h");
            const abplas_blas_fortran_header = b.path("include/zynum/blas/blas.h");
            const abplas_blas_fortran_module = b.path("include/zynum/blas/blas.f90");
            const abplas_blas_manifest = b.path("include/zynum/blas/abi_manifest.json");
            const abplas_blas_cblas_install = b.addInstallFileWithDir(abplas_blas_cblas_header, .header, "zynum/blas/cblas.h");
            const abplas_blas_fortran_install = b.addInstallFileWithDir(abplas_blas_fortran_header, .header, "zynum/blas/blas.h");
            const abplas_blas_module_install = b.addInstallFileWithDir(abplas_blas_fortran_module, .header, "zynum/blas/blas.f90");
            const abplas_blas_manifest_install = b.addInstallFileWithDir(abplas_blas_manifest, .header, "zynum/blas/abi_manifest.json");
            abplas_lp64_install.dependOn(&abplas_blas_cblas_install.step);
            abplas_lp64_install.dependOn(&abplas_blas_fortran_install.step);
            abplas_lp64_install.dependOn(&abplas_blas_module_install.step);
            abplas_lp64_install.dependOn(&abplas_blas_manifest_install.step);
        }
        const abplas_lp64_pre_feast_root = abplas_lp64_selected_root;
        var feast_manifest: ?std.Build.LazyPath = null;
        if (feast_interface == .feast4_lp64_abplas) {
            const feast_generate = b.addSystemCommand(&.{ "python3", "tools/generate_feast.py", "--profile", "tools/feast_lp64_profile.json", "--output" });
            feast_generate.addFileInput(b.path("tools/generate_feast.py"));
            feast_generate.addFileInput(b.path("tools/feast_lp64_profile.json"));
            feast_generate.addFileInput(b.path("tools/interface_templates/feast_config.cmake.in"));
            feast_generate.addFileInput(b.path("tools/interface_templates/feast_config_version.cmake.in"));
            const feast_directory = feast_generate.addOutputDirectoryArg("feast4-lp64");
            const feast_binding_mod = b.createModule(.{ .root_source_file = b.path("src/interop/feast_binding.zig"), .target = target, .optimize = optimize, .link_libc = true });
            feast_binding_mod.addImport("zynum-core", zynum_core_mod);
            const feast_exports_mod = b.createModule(.{ .root_source_file = feast_directory.path(b, "feast_exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            feast_exports_mod.addImport("feast-interface-owner", feast_binding_mod);
            const feast_combined_mod = b.createModule(.{ .root_source_file = b.path("src/abplas_feast_lp64.zig"), .target = target, .optimize = optimize, .link_libc = true });
            feast_combined_mod.addImport("zynum-core", zynum_core_mod);
            feast_combined_mod.addImport("feast-interface-owner", feast_binding_mod);
            feast_combined_mod.addImport("zynum-abplas-base-exports", abplas_lp64_pre_feast_root);
            feast_combined_mod.addImport("zynum-feast-exports", feast_exports_mod);
            abplas_lp64_selected_root = feast_combined_mod;
            feast_manifest = feast_directory.path(b, "manifest.json");
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(feast_directory.path(b, "feast_lp64.h"), .header, "zynum/feast/feast_lp64.h").step);
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(feast_manifest.?, .header, "zynum/feast/abi_manifest.json").step);
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(feast_directory.path(b, "FEASTConfig.cmake"), .prefix, "lib/cmake/FEAST/FEASTConfig.cmake").step);
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(feast_directory.path(b, "FEASTConfigVersion.cmake"), .prefix, "lib/cmake/FEAST/FEASTConfigVersion.cmake").step);
        }
        var fftw_f32_manifest: ?std.Build.LazyPath = null;
        if (fft_f32) {
            const fftw_f32_generate = b.addSystemCommand(&.{ "python3", "tools/generate_fftw_f32.py", "--profile", "fftw3.f32.abplas-v1", "--zig", b.graph.zig_exe, "--output" });
            fftw_f32_generate.addFileInput(b.path("tools/generate_fftw_f32.py"));
            fftw_f32_generate.addFileInput(b.path("tools/fftw_f32_profile.json"));
            fftw_f32_generate.addFileInput(b.path("tools/generate_fftw.py"));
            fftw_f32_generate.addFileInput(b.path("tools/fftw_profile.json"));
            fftw_f32_generate.addFileInput(b.path("tools/interface_templates/fftw_exports.zig.in"));
            const fftw_f32_directory = fftw_f32_generate.addOutputDirectoryArg("fftw3-f32");
            const fftw_f32_exports_mod = b.createModule(.{ .root_source_file = fftw_f32_directory.path(b, "exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            fftw_f32_exports_mod.addImport("fftw-interface-owner", fftw_binding_mod);
            const fftw_f32_composed_root = b.createModule(.{ .root_source_file = b.path("src/abplas_fftw_f32_lp64.zig"), .target = target, .optimize = optimize, .link_libc = true });
            fftw_f32_composed_root.addImport("zynum-core", zynum_core_mod);
            fftw_f32_composed_root.addImport("fftw-interface-owner", fftw_binding_mod);
            fftw_f32_composed_root.addImport("zynum-abplas-base-exports", abplas_lp64_selected_root);
            fftw_f32_composed_root.addImport("zynum-fftw-f32-exports", fftw_f32_exports_mod);
            abplas_lp64_selected_root = fftw_f32_composed_root;
            fftw_f32_manifest = fftw_f32_directory.path(b, "abi_manifest.json");
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(fftw_f32_directory.path(b, "fftw3f_subset.h"), .header, "zynum/fft/fftw3f_subset.h").step);
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(fftw_f32_manifest.?, .header, "zynum/fft/f32_abi_manifest.json").step);
        }
        const abplas_lp64_static = b.addLibrary(.{ .name = "zynum_abplas_lp64", .linkage = .static, .root_module = abplas_lp64_selected_root });
        abplas_lp64_static.bundle_compiler_rt = true;
        if (fft_interface == .fftw3) {
            const fftw_exports_mod = b.createModule(.{ .root_source_file = b.path("interface_subset/fftw3.f64/abplas-v1/exports.zig"), .target = target, .optimize = optimize, .link_libc = true });
            fftw_exports_mod.addImport("fftw-interface-owner", fftw_binding_mod);
            abplas_lp64_root.addImport("fftw-interface-owner", fftw_binding_mod);
            abplas_lp64_root.addImport("zynum-fftw-exports", fftw_exports_mod);
            const fftw_check = b.addSystemCommand(&.{ "python3", "tools/generate_fftw.py", "--profile", "fftw3.f64.abplas-v1", "--output", "interface_subset/fftw3.f64/abplas-v1", "--check" });
            abplas_lp64_static.step.dependOn(&fftw_check.step);
        }

        const abplas_lp64_check = b.addSystemCommand(&.{ "python3", "tools/generate_checked_abi.py", "--profile", "mkl2026.lp64", "--binding-set", lapacke_lp64_set, "--target-contract", "apple-arm64-default-c-v1", "--zig", b.graph.zig_exe, "--check" });
        const abplas_lp64_checked_check = b.addSystemCommand(&.{ "python3", "tools/generate_checked_abi.py", "--profile", "checked-v1", "--zig", b.graph.zig_exe, "--check" });
        abplas_lp64_static.step.dependOn(&abplas_lp64_check.step);
        if (lapacke_lp64_component_check) |check| abplas_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_syev_check) |check| abplas_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_sygv_check) |check| abplas_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_controls_check) |check| abplas_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_evd_check) |check| abplas_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_gvd_check) |check| abplas_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_gels_check) |check| abplas_lp64_static.step.dependOn(&check.step);
        if (lapacke_lp64_gelss_check) |check| abplas_lp64_static.step.dependOn(&check.step);
        abplas_lp64_static.step.dependOn(&abplas_lp64_checked_check.step);
        abplas_lp64_static.step.dependOn(&csp_check.step);
        if (checked_complex_cholesky) abplas_lp64_static.step.dependOn(&cholesky_check.step);
        const abplas_lp64_repack = b.addSystemCommand(&.{ "python3", "tools/repack_darwin_archive.py", b.graph.zig_exe });
        abplas_lp64_repack.addFileArg(abplas_lp64_static.getEmittedBin());
        const abplas_lp64_archive = abplas_lp64_repack.addOutputFileArg("libzynum_abplas_lp64.a");
        const abplas_lp64_shared = b.addSystemCommand(&.{ "python3", "tools/link_darwin_interface.py" });
        abplas_lp64_shared.addFileArg(abplas_lp64_archive);
        abplas_lp64_shared.addFileArg(b.path("include/zynum/checked/sparse_manifest.json"));
        const abplas_lp64_output = abplas_lp64_shared.addOutputFileArg("libzynum_abplas_lp64.dylib");
        abplas_lp64_shared.addArgs(&.{ "--macos-min", b.fmt("{d}.{d}.{d}", .{ csp_macos_min.major, csp_macos_min.minor, csp_macos_min.patch }), "--legacy-manifest" });
        abplas_lp64_shared.addFileArg(b.path("include/zynum/checked/abi_manifest.json"));
        if (checked_complex_cholesky) {
            abplas_lp64_shared.addArg("--checked-cholesky-manifest");
            abplas_lp64_shared.addFileArg(b.path("include/zynum/checked/cholesky_manifest.json"));
        }
        abplas_lp64_shared.addArg("--lapacke-manifest");
        abplas_lp64_shared.addFileArg(lapacke_lp64_manifest);

        if (sparse_solves) {
            const standard_sparse_check = b.addSystemCommand(&.{ "python3", "tools/generate_checked_abi.py", "--profile", "mkl2026.lp64", "--binding-set", sparse_lp64_set, "--target-contract", "apple-arm64-default-c-v1", "--zig", b.graph.zig_exe, "--check" });
            abplas_lp64_static.step.dependOn(&standard_sparse_check.step);
            abplas_lp64_shared.addArg("--sparse-manifest");
            abplas_lp64_shared.addFileArg(b.path("interface_subset/mkl2026.lp64/sparse-csr-csc-mv-mm-trsv-trsm-ordinary-lp64-v1/abi_manifest.json"));
        }
        if (feast_manifest) |manifest| {
            abplas_lp64_shared.addArg("--feast-manifest");
            abplas_lp64_shared.addFileArg(manifest);
        }
        if (operator_manifest) |manifest| {
            abplas_lp64_shared.addArg("--operator-manifest");
            abplas_lp64_shared.addFileArg(manifest);
        }
        if (contour_manifest) |manifest| {
            abplas_lp64_shared.addArg("--contour-manifest");
            abplas_lp64_shared.addFileArg(manifest);
        }
        if (blas_interface == .lp64) {
            abplas_lp64_shared.addArg("--blas-manifest");
            abplas_lp64_shared.addFileArg(b.path("include/zynum/blas/abi_manifest.json"));
        }

        if (fft_interface == .fftw3) {
            abplas_lp64_shared.addArg("--fft-manifest");
            abplas_lp64_shared.addFileArg(b.path("interface_subset/fftw3.f64/abplas-v1/abi_manifest.json"));
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(b.path("interface_subset/fftw3.f64/abplas-v1/fftw3_subset.h"), .header, "zynum/fft/fftw3_subset.h").step);
            abplas_lp64_install.dependOn(&b.addInstallFileWithDir(b.path("interface_subset/fftw3.f64/abplas-v1/abi_manifest.json"), .header, "zynum/fft/abi_manifest.json").step);
        }
        if (fftw_f32_manifest) |manifest| {
            abplas_lp64_shared.addArg("--fft-f32-manifest");
            abplas_lp64_shared.addFileArg(manifest);
        }
        abplas_lp64_install.dependOn(&b.addInstallFileWithDir(abplas_lp64_archive, .lib, "libzynum_abplas_lp64.a").step);
        abplas_lp64_install.dependOn(&b.addInstallFileWithDir(abplas_lp64_output, .lib, "libzynum_abplas_lp64.dylib").step);
        abplas_lp64_install.dependOn(&checked_header_install.step);
        abplas_lp64_install.dependOn(&csp_sparse_header_install.step);
        abplas_lp64_install.dependOn(&checked_complex_header_install.step);
        if (checked_complex_cholesky) {
            abplas_lp64_install.dependOn(&checked_cholesky_header_install.step);
            abplas_lp64_install.dependOn(&checked_cholesky_manifest_install.step);
        }
        abplas_lp64_install.dependOn(&checked_fortran_install.step);
        abplas_lp64_install.dependOn(&checked_manifest_install.step);
        abplas_lp64_install.dependOn(&csp_sparse_manifest_install.step);
        abplas_lp64_install.dependOn(&lapacke_lp64_header_install.step);
        abplas_lp64_install.dependOn(&lapacke_lp64_manifest_install.step);
        if (lapacke_controls) {
            abplas_lp64_install.dependOn(&lapacke_lp64_base_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_controls_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_controls_manifest_install.step);
        }
        if (lapacke_potri) {
            abplas_lp64_install.dependOn(&lapacke_lp64_base_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_potri_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_potri_manifest_install.step);
        }
        if (lapacke_syev) {
            abplas_lp64_install.dependOn(&lapacke_lp64_base_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_syev_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_syev_manifest_install.step);
        }
        if (lapacke_evd) {
            abplas_lp64_install.dependOn(&lapacke_lp64_base_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_evd_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_evd_manifest_install.step);
        }
        if (lapacke_gvd) {
            abplas_lp64_install.dependOn(&lapacke_lp64_base_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_gvd_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_gvd_manifest_install.step);
        }
        if (lapacke_gels) {
            abplas_lp64_install.dependOn(&lapacke_lp64_base_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_gels_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_gels_manifest_install.step);
        }
        if (lapacke_gelss) {
            abplas_lp64_install.dependOn(&lapacke_lp64_base_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_gelss_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_gelss_manifest_install.step);
        }
        if (lapacke_sygv) {
            abplas_lp64_install.dependOn(&lapacke_lp64_base_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_sygv_header_install.step);
            abplas_lp64_install.dependOn(&lapacke_lp64_sygv_manifest_install.step);
        }
        const abplas_lp64_files = b.addWriteFiles();
        const abplas_lp64_pc = abplas_lp64_files.add("zynum_abplas_lp64.pc", b.fmt("lapacke_gelss={s}\nlapacke_gels={s}\nlapacke_gvd={s}\nlapacke_evd={s}\nlapacke_controls={s}\n{s}{s}{s}{s}{s}{s}{s}operator_interface={s}\nfft_interface={s}\nfeast_interface={s}\nprefix=${{pcfiledir}}/../..\nlibdir=${{prefix}}/lib\nincludedir=${{prefix}}/include\nName: zynum_abplas_lp64\nDescription: Zynum {s}, one runtime owner\nVersion: 0.0.1-beta\nLibs: -L${{libdir}} -lzynum_abplas_lp64\nCflags: -I${{includedir}} -I${{includedir}}/zynum/lapacke/lp64{s}\n", .{ if (lapacke_gelss) "true" else "false", if (lapacke_gels) "true" else "false", if (lapacke_gvd) "true" else "false", if (lapacke_evd) "true" else "false", if (lapacke_controls) "true" else "false", if (blas_interface == .lp64) "blas_interface=lp64\n" else "", if (operator_phase_agreement) "operator_phase_agreement=true\n" else "", if (checked_complex_cholesky) "checked_complex_cholesky=true\n" else "checked_complex_cholesky=false\n", if (fft_f32) "fft_f32=true\n" else "", if (contour_interface == .zynum_v1) "contour_interface=zynum_v1\n" else "", if (lapacke_syev) "lapacke_syev=true\n" else "", if (lapacke_sygv) "lapacke_sygv=true\n" else "", @tagName(operator_interface), @tagName(fft_interface), @tagName(feast_interface), if (blas_interface == .lp64) "selected LP64 BLAS/LAPACKE and mixed-index checked Sparse" else "finite LP64 LAPACKE and mixed-index checked Sparse", if (blas_interface == .lp64) " -I${includedir}/zynum/blas" else "" }));
        abplas_lp64_install.dependOn(&b.addInstallFileWithDir(abplas_lp64_pc, .lib, "pkgconfig/zynum_abplas_lp64.pc").step);
        const test_lapacke_gels_installed = b.step("test-lapacke-gels-installed", "Check installed LP64 GELS numerical branches, boundaries and library ownership");
        if (lapacke_gels and lapacke_controls) {
            const lapacke_gels_installed_check = b.addSystemCommand(&.{ "python3", "tools/check_lapacke_gels.py", "--prefix" });
            lapacke_gels_installed_check.addFileInput(b.path("tools/check_lapacke_gels.py"));
            lapacke_gels_installed_check.addFileInput(b.path("test/headers/lapacke_gels_smoke.c"));
            lapacke_gels_installed_check.addDirectoryArg2(.{ .relative = .{ .base = .install_prefix } }, .{});
            lapacke_gels_installed_check.addArg("--output");
            _ = lapacke_gels_installed_check.addOutputDirectoryArg("lapacke-gels-validation");
            lapacke_gels_installed_check.step.dependOn(abplas_lp64_install);
            lapacke_gels_installed_check.step.dependOn(lapacke_lp64_install_step);
            test_lapacke_gels_installed.dependOn(&lapacke_gels_installed_check.step);
        } else {
            test_lapacke_gels_installed.dependOn(&b.addFail("installed GELS checks require lapacke-gels=true and lapacke-controls=true").step);
        }
        const test_lapacke_gelss_installed = b.step("test-lapacke-gelss-installed", "Check installed LP64 GELSS numerical branches, boundaries and library ownership");
        if (lapacke_gelss and lapacke_controls) {
            const lapacke_gelss_installed_check = b.addSystemCommand(&.{ "python3", "tools/check_lapacke_gelss.py", "--prefix" });
            lapacke_gelss_installed_check.addFileInput(b.path("tools/check_lapacke_gelss.py"));
            lapacke_gelss_installed_check.addFileInput(b.path("test/headers/lapacke_gelss_smoke.c"));
            lapacke_gelss_installed_check.addDirectoryArg2(.{ .relative = .{ .base = .install_prefix } }, .{});
            lapacke_gelss_installed_check.addArg("--output");
            _ = lapacke_gelss_installed_check.addOutputDirectoryArg("lapacke-gelss-validation");
            lapacke_gelss_installed_check.step.dependOn(abplas_lp64_install);
            lapacke_gelss_installed_check.step.dependOn(lapacke_lp64_install_step);
            test_lapacke_gelss_installed.dependOn(&lapacke_gelss_installed_check.step);
        } else {
            test_lapacke_gelss_installed.dependOn(&b.addFail("installed GELSS checks require lapacke-gelss=true and lapacke-controls=true").step);
        }
        const test_unified_blas = b.step("test-unified-blas", "Check installed unified BLAS numerical calls, boundaries and cross-module lifetimes");
        if (blas_interface == .lp64 and fft_interface == .fftw3) {
            const unified_blas_check = b.addSystemCommand(&.{ "python3", "tools/check_unified_blas.py", "--prefix" });
            unified_blas_check.addFileInput(b.path("tools/check_unified_blas.py"));
            unified_blas_check.addFileInput(b.path("test/headers/unified_blas_smoke.c"));
            unified_blas_check.addDirectoryArg2(.{ .relative = .{ .base = .install_prefix } }, .{});
            unified_blas_check.addArg("--output");
            _ = unified_blas_check.addOutputDirectoryArg("unified-blas-validation");
            unified_blas_check.step.dependOn(abplas_lp64_install);
            test_unified_blas.dependOn(&unified_blas_check.step);
        } else {
            test_unified_blas.dependOn(&b.addFail("test-unified-blas requires -Dblas-interface=lp64 -Dfft-interface=fftw3").step);
        }
    } else {
        abplas_lp64_install.dependOn(&b.addFail("Combined LP64 LAPACKE/Sparse requires the Apple arm64 declaration contract").step);
    }

    // BEGIN checked ABI integration candidate. Separate from BLAS artifacts.
    const install_checked_abi = b.option(bool, "checked-abi", "Include the experimental checked ABI in default installation") orelse false;
    const checked_nm = b.option([]const u8, "checked-nm", "nm executable for checked ABI artifact inspection (llvm-nm for cross objects)") orelse "nm";
    const checked_readobj = b.option([]const u8, "checked-readobj", "llvm-readobj executable for checked COFF export inspection") orelse "llvm-readobj";
    const checked_gfortran = b.option([]const u8, "checked-gfortran", "gfortran executable required by test-checked-fortran") orelse "gfortran";
    const checked_abi_mod = b.createModule(.{
        .root_source_file = b.path("src/checked.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    checked_abi_mod.addImport("zynum-numerics-resources", numerics_resources_mod);
    checked_abi_mod.addImport("zynum-core", zynum_core_mod);
    checked_abi_mod.addImport("zynum-execution", execution_mod);
    checked_abi_mod.addImport("zynum-build-options", zynum_build_options_mod);
    const checked_dynamic_lib = b.addLibrary(.{
        .name = "zynum_checked",
        .linkage = .dynamic,
        .root_module = checked_abi_mod,
        .win32_module_definition = if (target.result.ofmt == .coff) b.path("src/checked/exports.def") else null,
    });
    if (target.result.ofmt == .coff) checked_dynamic_lib.dll_export_fns = false;
    // Optional combined link product. Existing library names/install steps stay unchanged.
    const combined_interfaces_mod = b.createModule(.{
        .root_source_file = b.path("src/interfaces.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    combined_interfaces_mod.addImport("zynum-blas-exports", blas_compat_mod);
    combined_interfaces_mod.addImport("zynum-checked-exports", checked_abi_mod);
    const combined_interfaces_lib = b.addLibrary(.{
        .name = "zynum_interfaces",
        .linkage = .dynamic,
        .root_module = combined_interfaces_mod,
    });
    _ = combined_interfaces_lib.getEmittedBin();
    const build_combined_interfaces_step = b.step("build-combined-interfaces", "Build the optional combined existing interfaces without installation");
    build_combined_interfaces_step.dependOn(&combined_interfaces_lib.step);

    const checked_static_lib = b.addLibrary(.{
        .name = "zynum_checked",
        .linkage = .static,
        .root_module = checked_abi_mod,
    });
    const build_checked_abi_step = b.step("build-checked-abi", "Build separate checked shared/static libraries without installation");
    build_checked_abi_step.dependOn(&checked_dynamic_lib.step);
    build_checked_abi_step.dependOn(&checked_static_lib.step);
    const repack_checked_static_lib = b.addSystemCommand(&.{ "python3", "tools/repack_darwin_archive.py", b.graph.zig_exe });
    repack_checked_static_lib.addFileArg(checked_static_lib.getEmittedBin());
    const repacked_checked_static_archive = repack_checked_static_lib.addOutputFileArg("libzynum_checked.a");
    const checked_static_archive = if (target.result.ofmt == .macho) repacked_checked_static_archive else checked_static_lib.getEmittedBin();
    const checked_dynamic_install = b.addInstallArtifact(checked_dynamic_lib, .{});
    const checked_static_install: *std.Build.Step = if (target.result.ofmt == .macho)
        &b.addInstallFileWithDir(checked_static_archive, .lib, "libzynum_checked.a").step
    else
        &b.addInstallArtifact(checked_static_lib, if (target.result.os.tag == .windows)
            .{ .dest_dir = .{ .override = .{ .custom = "lib/static" } } }
        else
            .{}).step;
    const checked_pkgconfig_install = b.addInstallFileWithDir(b.path("pkgconfig/zynum_checked.pc"), .lib, "pkgconfig/zynum_checked.pc");
    const install_checked_abi_step = b.step("install-checked-abi", "Install only the separate checked libraries, header, Fortran source and manifest");
    install_checked_abi_step.dependOn(&checked_dynamic_install.step);
    install_checked_abi_step.dependOn(checked_static_install);
    install_checked_abi_step.dependOn(&checked_header_install.step);
    install_checked_abi_step.dependOn(&checked_fortran_install.step);
    install_checked_abi_step.dependOn(&checked_manifest_install.step);
    install_checked_abi_step.dependOn(&checked_pkgconfig_install.step);
    if (install_checked_abi) b.getInstallStep().dependOn(install_checked_abi_step);

    const generate_checked_abi = b.addSystemCommand(&.{ "python3", "tools/generate_checked_abi.py", "--zig", b.graph.zig_exe });
    generate_checked_abi.has_side_effects = true;
    const generate_checked_abi_step = b.step("generate-checked-abi", "Regenerate only the separate checked ABI declarations and manifest");
    generate_checked_abi_step.dependOn(&generate_checked_abi.step);
    const check_checked_abi_generated = b.addSystemCommand(&.{ "python3", "tools/generate_checked_abi.py", "--zig", b.graph.zig_exe, "--check" });
    const checked_abi_exports = b.addSystemCommand(&.{ "python3", "tools/check_checked_abi.py", "--manifest" });
    checked_abi_exports.addFileArg(b.path("include/zynum/checked/abi_manifest.json"));
    checked_abi_exports.addArgs(&.{ "--format", @tagName(target.result.ofmt), "--nm", checked_nm, "--readobj", checked_readobj, "--dynamic" });
    checked_abi_exports.addFileArg(checked_dynamic_lib.getEmittedBin());
    checked_abi_exports.addArg("--static");
    checked_abi_exports.addFileArg(checked_static_archive);
    const check_checked_abi_step = b.step("check-checked-abi", "Check generated checked declarations and independent dynamic/static export contract");
    check_checked_abi_step.dependOn(&check_checked_abi_generated.step);
    check_checked_abi_step.dependOn(&checked_abi_exports.step);

    const checked_abi_test_mod = b.createModule(.{
        .root_source_file = b.path("src/checked.zig"),
        .target = target,
        .optimize = test_optimize,
        .link_libc = true,
    });
    checked_abi_test_mod.addImport("zynum-numerics-resources", numerics_resources_mod);
    checked_abi_test_mod.addImport("zynum-core", zynum_core_test_mod);
    checked_abi_test_mod.addImport("zynum-execution", execution_test_mod);
    checked_abi_test_mod.addImport("zynum-build-options", zynum_build_options_mod);
    const checked_test_lib = b.addLibrary(.{
        .name = "zynum_checked",
        .linkage = .dynamic,
        .root_module = checked_abi_test_mod,
        .win32_module_definition = if (target.result.ofmt == .coff) b.path("src/checked/exports.def") else null,
    });
    if (target.result.ofmt == .coff) checked_test_lib.dll_export_fns = false;
    const checked_c_smoke = b.addExecutable(.{
        .name = "checked-c-smoke",
        .root_module = b.createModule(.{ .target = target, .optimize = test_optimize, .link_libc = true }),
    });
    checked_c_smoke.root_module.addCSourceFile(.{
        .file = b.path("test/headers/checked_c_smoke.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic" },
    });
    checked_c_smoke.root_module.addIncludePath(b.path("include"));
    checked_c_smoke.root_module.linkLibrary(checked_test_lib);
    const checked_cpp_smoke = b.addExecutable(.{
        .name = "checked-cpp-smoke",
        .root_module = b.createModule(.{ .target = target, .optimize = test_optimize, .link_libc = true }),
    });
    checked_cpp_smoke.root_module.addCSourceFile(.{
        .file = b.path("test/headers/checked_cpp_smoke.cpp"),
        .flags = &.{ "-std=c++20", "-Wall", "-Wextra", "-Werror", "-pedantic" },
    });
    checked_cpp_smoke.root_module.addIncludePath(b.path("include"));
    checked_cpp_smoke.root_module.linkLibrary(checked_test_lib);
    if (target.result.os.tag != .windows) {
        checked_c_smoke.root_module.addRPath(checked_test_lib.getEmittedBinDirectory());
        checked_cpp_smoke.root_module.addRPath(checked_test_lib.getEmittedBinDirectory());
    }
    // These two AArch64 flags select compiler heuristics, not instructions.
    // Installed Zig target definitions: aggressive_fma and store_pair_suppress.
    // Preserve every other required bit and all independent native-run guards.
    var checked_required_cpu_features = target.result.cpu.features;
    if (target.result.cpu.arch == .aarch64) {
        checked_required_cpu_features.removeFeature(@backingInt(std.Target.aarch64.Feature.aggressive_fma));
        checked_required_cpu_features.removeFeature(@backingInt(std.Target.aarch64.Feature.store_pair_suppress));
    }
    const checked_native_guard: *std.Build.Step = if (!native_feature_target_matches_host)
        &b.addFail("checked native consumers require target arch/os/abi/ofmt to match the host exactly").step
    else if (!b.graph.host.result.cpu.features.isSuperSetOf(checked_required_cpu_features))
        &b.addFail("checked native consumers require CPU features supported by the build host").step
    else
        &native_execution_check.step;
    const run_checked_c_smoke = b.addRunArtifact(checked_c_smoke);
    const run_checked_cpp_smoke = b.addRunArtifact(checked_cpp_smoke);
    run_checked_c_smoke.setCwd(b.path("."));
    run_checked_cpp_smoke.setCwd(b.path("."));
    run_checked_c_smoke.step.dependOn(checked_native_guard);
    run_checked_cpp_smoke.step.dependOn(checked_native_guard);
    // On Windows the loader searches the working directory for the DLL; Unix
    // binaries carry an explicit build-artifact rpath above.
    if (target.result.os.tag == .windows) {
        run_checked_c_smoke.setCwd(checked_test_lib.getEmittedBinDirectory());
        run_checked_cpp_smoke.setCwd(checked_test_lib.getEmittedBinDirectory());
    }
    const test_checked_abi_step = b.step("test-checked-abi", "Run supplemental native checked C/C++ consumers; not inventory observations");
    test_checked_abi_step.dependOn(check_checked_abi_step);
    test_checked_abi_step.dependOn(&run_checked_c_smoke.step);
    test_checked_abi_step.dependOn(&run_checked_cpp_smoke.step);

    const build_checked_fortran_smoke = b.addSystemCommand(&.{ checked_gfortran, "-std=f2018", "-Wall", "-Wextra", "-Werror", "-fcheck=all" });
    build_checked_fortran_smoke.step.dependOn(checked_native_guard);
    if (target.result.os.tag == .macos) {
        const checked_macos_min = target.result.os.version_range.semver.min;
        build_checked_fortran_smoke.addArg(b.fmt("-mmacosx-version-min={d}.{d}.{d}", .{ checked_macos_min.major, checked_macos_min.minor, checked_macos_min.patch }));
    }
    _ = build_checked_fortran_smoke.addOutputDirectoryArg2("checked-fortran-modules", .{ .prefix = "-J" });
    build_checked_fortran_smoke.addFileArg(b.path("include/zynum/checked/checked.f90"));
    build_checked_fortran_smoke.addFileArg(b.path("test/headers/checked_fortran_smoke.f90"));
    build_checked_fortran_smoke.addFileArg(if (target.result.os.tag == .windows) checked_test_lib.getEmittedImplib() else checked_test_lib.getEmittedBin());
    if (target.result.os.tag != .windows) {
        build_checked_fortran_smoke.addDirectoryArg2(checked_test_lib.getEmittedBinDirectory(), .{ .prefix = "-Wl,-rpath,", .make_absolute = true });
    }
    build_checked_fortran_smoke.addArg("-o");
    const checked_fortran_exe = build_checked_fortran_smoke.addOutputFileArg(if (target.result.os.tag == .windows) "checked-fortran-smoke.exe" else "checked-fortran-smoke");
    const run_checked_fortran_smoke = std.Build.Step.Run.create(b, "run checked Fortran consumer");
    run_checked_fortran_smoke.addFileArg(checked_fortran_exe);
    run_checked_fortran_smoke.has_side_effects = true;
    run_checked_fortran_smoke.step.dependOn(checked_native_guard);
    if (target.result.os.tag == .windows) run_checked_fortran_smoke.setCwd(checked_test_lib.getEmittedBinDirectory());
    const test_checked_fortran_step = b.step("test-checked-fortran", "Require gfortran and run the supplemental native ISO_C_BINDING consumer");
    test_checked_fortran_step.dependOn(&check_checked_abi_generated.step);
    test_checked_fortran_step.dependOn(&run_checked_fortran_smoke.step);
    // END checked ABI integration candidate.

    const native_feature_profile_guard: *std.Build.Step = if (!explicit_non_baseline_cpu_profile)
        &b.addFail("test-native-feature requires an explicit non-baseline CPU profile").step
    else if (!native_feature_target_matches_host)
        &b.addFail("test-native-feature requires target arch/os/abi/ofmt to match the build host exactly").step
    else if (!b.graph.host.result.cpu.features.isSuperSetOf(target.result.cpu.features))
        &b.addFail("test-native-feature requested CPU features are not supported by the build host").step
    else
        &test_inventory_structure_check.step;
    for (inventory_cases) |inventory_case| {
        const official_tests = inventory_case.logical_tests orelse continue;
        const run_native_feature_tests = b.addRunArtifact(official_tests);
        run_native_feature_tests.step.dependOn(native_feature_profile_guard);
        run_native_feature_tests.step.dependOn(&native_execution_check.step);
        test_native_feature_step.dependOn(&run_native_feature_tests.step);
    }

    run_blas_module_tests.step.dependOn(test_inventory_step);
    run_blas_public_surface_contract_tests.step.dependOn(test_inventory_step);
    run_cblas_tests.step.dependOn(test_inventory_step);
    run_execution_contract_tests.step.dependOn(test_inventory_step);
    run_fortran_tests.step.dependOn(test_inventory_step);
    run_gemm_registry_tests.step.dependOn(test_inventory_step);
    run_header_smoke_tests.step.dependOn(test_inventory_step);
    run_level1_registry_tests.step.dependOn(test_inventory_step);
    run_level2_compact_registry_tests.step.dependOn(test_inventory_step);
    run_level2_fused_registry_tests.step.dependOn(test_inventory_step);
    run_modern_tests.step.dependOn(test_inventory_step);
    run_numerics_contract_tests.step.dependOn(test_inventory_step);
    run_lapack_tests.step.dependOn(test_inventory_step);
    run_sparse_tests.step.dependOn(test_inventory_step);
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

    const multiversion_check = b.addSystemCommand(&.{ "python3", "tools/generate_multiversion.py", "--check" });
    const check_multiversion_step = b.step("check-multiversion", "Check generated private kernel dispatch protocol and adapters");
    check_multiversion_step.dependOn(&multiversion_check.step);
    const dynamic_tests = b.addTest(.{
        .name = "zynum-dynamic-dispatch-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/blas/dynamic_dispatch_test.zig"),
            .target = target,
            .optimize = test_optimize,
            .link_libc = true,
        }),
    });
    dynamic_tests.root_module.addImport("zynum-build-options", zynum_build_options_mod);
    dynamic_tests.root_module.addImport("zynum-core", zynum_core_test_mod);
    dynamic_tests.root_module.addImport("zynum-blas-abi", blas_abi_test_mod);
    const run_dynamic_tests = b.addRunArtifact(dynamic_tests);
    const run_dynamic_baseline_tests = b.addRunArtifact(dynamic_tests);
    run_dynamic_baseline_tests.setEnvironmentVariable("ZYNUM_MAX_ISA", "baseline");
    // Fresh processes make the cached capability ceiling independently testable.
    const test_dynamic_step = b.step("test-dynamic-dispatch", "Run native dynamic-dispatch correctness and forced-baseline checks; not inventory evidence");
    if (dynamic_dispatch and native_feature_target_matches_host) {
        run_dynamic_tests.step.dependOn(&native_execution_check.step);
        run_dynamic_baseline_tests.step.dependOn(&native_execution_check.step);
        test_dynamic_step.dependOn(&run_dynamic_tests.step);
        test_dynamic_step.dependOn(&run_dynamic_baseline_tests.step);
        test_dynamic_step.dependOn(check_multiversion_step);
    } else {
        test_dynamic_step.dependOn(&b.addFail("test-dynamic-dispatch requires a native dynamic build without external executors; omit -Dcpu").step);
    }

    // Reuse the canonical bodies in native dynamic builds. These runs verify
    // correctness only; they do not satisfy baseline inventory observations.
    const run_dynamic_lapack_tests = b.addRunArtifact(lapack_tests);
    const run_dynamic_sparse_tests = b.addRunArtifact(sparse_tests);
    const run_dynamic_numerics_contract_tests = b.addRunArtifact(numerics_contract_tests);
    const run_dynamic_execution_contract_tests = b.addRunArtifact(execution_contract_tests);
    run_dynamic_lapack_tests.step.dependOn(test_dynamic_step);
    run_dynamic_sparse_tests.step.dependOn(test_dynamic_step);
    run_dynamic_numerics_contract_tests.step.dependOn(test_dynamic_step);
    run_dynamic_execution_contract_tests.step.dependOn(test_dynamic_step);
    run_dynamic_lapack_tests.step.dependOn(&test_inventory_structure_check.step);
    run_dynamic_sparse_tests.step.dependOn(&test_inventory_structure_check.step);
    run_dynamic_numerics_contract_tests.step.dependOn(&test_inventory_structure_check.step);
    run_dynamic_execution_contract_tests.step.dependOn(&test_inventory_structure_check.step);
    run_dynamic_lapack_tests.step.dependOn(&native_execution_check.step);
    run_dynamic_sparse_tests.step.dependOn(&native_execution_check.step);
    run_dynamic_numerics_contract_tests.step.dependOn(&native_execution_check.step);
    run_dynamic_execution_contract_tests.step.dependOn(&native_execution_check.step);

    const test_step = b.step("test", "Run correctness tests");
    if (runtime_lifecycle_native_supported) test_step.dependOn(test_runtime_lifecycle_step);
    if (dynamic_dispatch) {
        test_step.dependOn(test_dynamic_step);
        test_step.dependOn(&run_dynamic_lapack_tests.step);
        test_step.dependOn(&run_dynamic_sparse_tests.step);
        test_step.dependOn(&run_dynamic_numerics_contract_tests.step);
        test_step.dependOn(&run_dynamic_execution_contract_tests.step);
        if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);
    } else {
        test_step.dependOn(&test_inventory_structure_check.step);
        test_step.dependOn(test_inventory_step);
        test_step.dependOn(&run_modern_tests.step);
        test_step.dependOn(&run_blas_module_tests.step);
        test_step.dependOn(&run_lapack_tests.step);
        test_step.dependOn(&run_sparse_tests.step);
        test_step.dependOn(&run_numerics_contract_tests.step);
        test_step.dependOn(&run_execution_contract_tests.step);
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
        if (host_tool_smoke) test_step.dependOn(host_tool_smoke_test_step);
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
    run_bench.addPassthruArgs();
    // The emitted library argument supplies the dependency; unrelated installed
    // probes and the static library are not prerequisites for a benchmark run.

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
    run_gemm_sweep.addFileArg2(b.graph.path(.install_prefix, "gemm_sweep.csv"), .{});
    addOptionalBenchLibrary(run_gemm_sweep, "--openblas", bench_openblas, if (target.result.os.tag == .macos) "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" else null);
    addOptionalBenchLibrary(run_gemm_sweep, "--accelerate", bench_accelerate, if (target.result.os.tag == .macos) "/System/Library/Frameworks/Accelerate.framework/Accelerate" else null);
    addOptionalBenchLibrary(run_gemm_sweep, "--mkl", bench_mkl, null);
    addOptionalBenchLibrary(run_gemm_sweep, "--aocl-blis", bench_aocl_blis, null);
    run_gemm_sweep.addPassthruArgs();
    // Installing only the shared library also creates the selected output prefix
    // before the sweep opens its default CSV on a clean checkout.
    run_gemm_sweep.step.dependOn(&install_dynamic_lib.step);

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
    run_gemm_sweep_isolated.addPassthruArgs();

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
    run_vector_matrix_sweep.addPassthruArgs();

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
    const build_rank_k_probe_step = b.step(
        "build-rank-k-probe",
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
    run_rank_k_report.addPassthruArgs();

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
    const build_symm_probe_step = b.step(
        "build-symm-probe",
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
    run_symm_report.addPassthruArgs();

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
    const build_triangular_matrix_probe_step = b.step(
        "build-triangular-matrix-probe",
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
    run_triangular_matrix_report.addPassthruArgs();

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
    const build_rotg_latency_probe_step = b.step(
        "build-rotg-latency-probe",
        "Build the opt-in Level 1 ROTG/ROTMG latency probe",
    );
    build_rotg_latency_probe_step.dependOn(&install_rotg_latency_probe.step);

    if (native_canonical_windows_python_tooling) {
        python_tooling_tests.addArg("--windows-zynum-blas-build-output");
        python_tooling_tests.addFileArg(lib.getEmittedBin());
        python_tooling_tests.addArg("--windows-zynum-blas-installed-output");
        python_tooling_tests.addFileArg2(b.graph.path(.install_bin, "zynum_blas.dll"), .{});
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
    run_rotg_latency_report.addPassthruArgs();

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
    return .{
        .core = zynum_core_mod,
        .execution = execution_mod,
        .resources = numerics_resources_mod,
        .options = zynum_build_options_mod,
        .lapacke_binding = lapacke_binding_mod,
        .mkl_sparse_binding = mkl_sparse_binding_mod,
    };
}

fn buildKernelTiers(
    b: *std.Build,
    baseline: std.Build.ResolvedTarget,
    optimize: std.lang.Optimize,
    apple_amx: bool,
    level1_sve: bool,
    level1_fixed: bool,
    level2_fixed: bool,
    level2_width: bool,
) []const *std.Build.Step.Compile {
    const names: []const []const u8 = switch (baseline.result.cpu.arch) {
        .aarch64 => &.{ "baseline", "aarch64_sve2", "aarch64_sme", "aarch64_sme2", "aarch64_sme2p1" },
        .x86_64 => &.{ "baseline", "x86_avx", "x86_avx2_fma", "x86_avx512" },
        else => &.{"baseline"},
    };
    const libraries = b.allocator.alloc(*std.Build.Step.Compile, names.len) catch @panic("out of memory");
    for (names, 0..) |name, index| {
        var query = baseline.query;
        if (index != 0) {
            query.cpu_model = .baseline;
            query.cpu_features_add = switch (baseline.result.cpu.arch) {
                .aarch64 => switch (index) {
                    1 => std.Target.aarch64.featureSet(&.{.sve2}),
                    2 => std.Target.aarch64.featureSet(&.{.sme}),
                    3 => std.Target.aarch64.featureSet(&.{ .sme2, .sme_f64f64 }),
                    4 => std.Target.aarch64.featureSet(&.{ .sme2p1, .sme_f64f64 }),
                    else => unreachable,
                },
                .x86_64 => switch (index) {
                    1 => std.Target.x86.featureSet(&.{.avx}),
                    2 => std.Target.x86.featureSet(&.{ .avx2, .fma }),
                    3 => std.Target.x86.featureSet(&.{ .avx512f, .avx512dq, .avx512bw, .avx512vl, .fma }),
                    else => unreachable,
                },
                else => unreachable,
            };
        }
        const options = b.addOptions();
        options.addOption(bool, "dynamic_dispatch", false);
        options.addOption(bool, "apple_amx", apple_amx);
        options.addOption(usize, "thread_limit", 0);
        options.addOption([]const u8, "kernel_entry", b.fmt("zynum_internal_kernel_{s}", .{name}));
        options.addOption(bool, "level1_sve_candidates", level1_sve);
        options.addOption(bool, "level1_fixed_candidates", level1_fixed);
        options.addOption(bool, "level2_fixed_candidates", level2_fixed);
        options.addOption(bool, "level2_width_candidates", level2_width);
        const module = b.createModule(.{
            .root_source_file = b.path("src/blas/kernel_object_root.zig"),
            .target = b.resolveTargetQuery(query),
            .optimize = optimize,
            .link_libc = true,
            .pic = true,
        });
        module.addOptions("zynum-build-options", options);
        libraries[index] = b.addObject(.{
            .name = b.fmt("zynum-kernel-{s}-{s}", .{ name, @tagName(optimize) }),
            .root_module = module,
            // Zig 0.17's self-hosted debug backend cannot assemble vmovntdq.
            .use_llvm = if (baseline.result.cpu.arch == .x86_64 and index == 2 and optimize == .debug) true else null,
        });
        // ISA objects must remain separate code-generation units. Never LTO
        // target-specific instructions into the baseline resolver or scheduler.
        libraries[index].lto = .none;
    }
    return libraries;
}
