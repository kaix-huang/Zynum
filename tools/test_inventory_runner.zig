// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Test-inventory runner: enumerate compiled tests without executing their bodies.

const std = @import("std");
const builtin = @import("builtin");

const Io = std.Io;
const runner_io: Io = Io.Threaded.global_single_threaded.io();
const maximum_inventory_bytes = 4 * 1024 * 1024;
const CURRENT_TEST_INVENTORY_SHA256: []const u8 = "a3d5e31a6c1b6cc0845f06cc15f54b0a8c3cf2b1cf2e5e926ee6aa94852700d4";
const NEXT_TEST_INVENTORY_SHA256: ?[]const u8 = null;

const Inventory = struct {
    schema_id: []const u8,
    schema_version: usize,
    build_inventory_schema_id: []const u8,
    optimize_modes: []const OptimizeMode,
    predicates: std.json.Value,
    environment_profiles: []const EnvironmentProfile,
    test_enumeration_classes: []const TestEnumerationClass,
    test_roots: std.json.Value,
    zig_test_files: std.json.Value,
    python_test_modules: std.json.Value,
    python_skip_contracts: std.json.Value,
    expected_test_sets: []const ExpectedTestSet,
    native_observation_bindings: []const NativeObservationBinding,
    test_mode_rows: []const TestModeRow,
    workflow_mode_bindings: std.json.Value,
    known_gaps: std.json.Value,
    matrix_row_contract: std.json.Value,
    strict_summary: std.json.Value,
};

const EnvironmentProfile = struct {
    id: []const u8,
    target: []const u8,
    architecture: []const u8,
    os: []const u8,
    libc: []const u8,
    cpu: []const u8,
    resolved_cpu_model: []const u8,
    cpu_feature_policy: []const u8,
    host_tool_smoke: bool,
};

const TestEnumerationClass = struct {
    id: []const u8,
    language: []const u8,
    architecture: ?[]const u8,
    os: ?[]const u8,
    libc: ?[]const u8,
    object_format: ?[]const u8,
    environment_ids: []const []const u8,
    enumeration_source: []const u8,
};

const OptimizeMode = struct {
    id: []const u8,
    zig_value: []const u8,
};

const ExpectedTestSet = struct {
    id: []const u8,
    root_id: []const u8,
    tests: []const ExpectedTest,
    count: usize,
    digest: []const u8,
    enumeration_source: []const u8,
};

const ExpectedTest = struct {
    id: []const u8,
    name: []const u8,
    ordinal: usize,
};

const NativeObservationBinding = struct {
    id: []const u8,
    row_id: []const u8,
    evidence_slot_id: []const u8,
    enumeration_class_id: []const u8,
    optimize_mode_id: []const u8,
    expected_test_set_id: []const u8,
    enumeration_source: []const u8,
    digest: []const u8,
};

const TestModeRow = struct {
    id: []const u8,
    environment_id: []const u8,
    root_id: []const u8,
    optimize_mode_id: []const u8,
    disposition: []const u8,
    predicate_id: []const u8,
    command_template: ?[]const u8,
    mode_effect: []const u8,
    expected_actual_module_optimize: ?[]const u8,
    evidence_slot_id: []const u8,
    enumeration_class_id: []const u8,
    expected_test_set_id: ?[]const u8,
    expectation_state: []const u8,
};

const Arguments = struct {
    inventory_path: []const u8,
    environment_id: []const u8,
    root_id: []const u8,
    mode: []const u8,
    class_id: []const u8,
};

const BindingFacts = struct {
    enumeration_class_id: []const u8,
    enumeration_source: []const u8,
    evidence_slot_id: []const u8,
    expected_test_set_id: []const u8,
    optimize_mode_id: []const u8,
    row_id: []const u8,
};

const InventoryValidation = enum {
    frozen,
    requires_native_enumeration,
};

pub fn main(init: std.process.Init.Minimal) void {
    @disableInstrumentation();
    run(init) catch std.process.exit(1);
}

fn run(init: std.process.Init.Minimal) !void {
    const allocator = std.heap.page_allocator;
    const args = try init.args.toSlice(allocator);
    defer allocator.free(args);
    const arguments = try parseArguments(args);

    const validation = try validateInventory(allocator, arguments);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = Io.File.stdout().writerStreaming(runner_io, &stdout_buffer);
    const writer = &stdout.interface;

    try emitProtocol(writer, arguments);
    try writer.flush();

    if (validation == .requires_native_enumeration) return error.NativeEnumerationRequired;
}

fn emitProtocol(writer: *Io.Writer, arguments: Arguments) !void {
    try writer.writeAll("ZYNUM-TEST-INVENTORY-V2\n");
    try writer.print("mode:{s}\n", .{@tagName(builtin.mode)});
    writeHex(writer, "root", arguments.root_id);
    writeHex(writer, "class", arguments.class_id);
    try writer.print("count:{d}\n", .{builtin.test_functions.len});
    for (builtin.test_functions, 0..) |test_function, ordinal| {
        try writer.print("test:{d}:{d}:", .{ ordinal, test_function.name.len });
        for (test_function.name) |byte| try writer.print("{x:0>2}", .{byte});
        try writer.writeByte('\n');
    }
}

fn parseArguments(args: []const []const u8) !Arguments {
    var inventory_path: ?[]const u8 = null;
    var environment_id: ?[]const u8 = null;
    var root_id: ?[]const u8 = null;
    var mode: ?[]const u8 = null;
    var class_id: ?[]const u8 = null;
    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--inventory-environment")) {
            if (environment_id != null or index + 1 >= args.len) return error.InvalidArguments;
            index += 1;
            environment_id = args[index];
        } else if (std.mem.eql(u8, arg, "--inventory-root")) {
            if (root_id != null or index + 1 >= args.len) return error.InvalidArguments;
            index += 1;
            root_id = args[index];
        } else if (std.mem.eql(u8, arg, "--inventory-mode")) {
            if (mode != null or index + 1 >= args.len) return error.InvalidArguments;
            index += 1;
            mode = args[index];
        } else if (std.mem.eql(u8, arg, "--inventory-class")) {
            if (class_id != null or index + 1 >= args.len) return error.InvalidArguments;
            index += 1;
            class_id = args[index];
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return error.InvalidArguments;
        } else {
            if (inventory_path != null) return error.InvalidArguments;
            inventory_path = arg;
        }
    }

    const path = inventory_path orelse return error.InvalidArguments;
    const environment = environment_id orelse return error.InvalidArguments;
    const root = root_id orelse return error.InvalidArguments;
    const requested_mode = mode orelse return error.InvalidArguments;
    const requested_class = class_id orelse return error.InvalidArguments;
    if (path.len == 0 or environment.len == 0 or root.len == 0 or requested_mode.len == 0 or requested_class.len == 0) {
        return error.InvalidArguments;
    }
    return .{
        .inventory_path = path,
        .environment_id = environment,
        .root_id = root,
        .mode = requested_mode,
        .class_id = requested_class,
    };
}

fn validateInventory(allocator: std.mem.Allocator, arguments: Arguments) !InventoryValidation {
    if (!std.mem.eql(u8, arguments.mode, @tagName(builtin.mode))) return error.OptimizeModeMismatch;

    const bytes = try readInventoryAlloc(allocator, arguments.inventory_path);
    defer allocator.free(bytes);
    try validateInventoryByteCount(bytes.len);
    try validateInventoryDigest(bytes);

    var parsed = try std.json.parseFromSlice(Inventory, allocator, bytes, .{});
    defer parsed.deinit();
    const inventory = parsed.value;
    if (!std.mem.eql(u8, inventory.schema_id, "zynum-test-inventory-v3") or inventory.schema_version != 3) {
        return error.InvalidInventorySchema;
    }

    var mode_matches: usize = 0;
    for (inventory.optimize_modes) |mode| {
        if (std.mem.eql(u8, mode.zig_value, arguments.mode)) mode_matches += 1;
    }
    if (mode_matches != 1) return error.InvalidOptimizeMode;

    var mode_row: ?*const TestModeRow = null;
    for (inventory.test_mode_rows) |*candidate| {
        if (std.mem.eql(u8, candidate.environment_id, arguments.environment_id) and
            std.mem.eql(u8, candidate.root_id, arguments.root_id) and
            modeIdMatches(candidate.optimize_mode_id, arguments.mode))
        {
            if (mode_row != null) return error.DuplicateTestModeRow;
            mode_row = candidate;
        }
    }
    const row = mode_row orelse return error.MissingTestModeRow;

    try validateRowIdentity(allocator, row, arguments);
    const environment = try findEnvironmentProfile(inventory.environment_profiles, arguments.environment_id);
    try validateEnvironmentProfile(environment);

    const actual_class_id = try actualEnumerationClassId();
    if (!std.mem.eql(u8, arguments.class_id, row.enumeration_class_id) or
        !std.mem.eql(u8, arguments.class_id, actual_class_id))
    {
        return error.EnumerationClassArgumentMismatch;
    }
    const class = try findEnumerationClass(inventory.test_enumeration_classes, arguments.class_id);
    try validateEnumerationClass(class);
    if (countMatches(class.environment_ids, row.environment_id) != 1) return error.InvalidRowEnvironment;

    if (!std.mem.eql(u8, row.disposition, "execute") or row.command_template == null) {
        return error.NotApplicableEnumerationExecuted;
    }
    if (!std.mem.eql(u8, row.mode_effect, "test-module-optimize") or
        row.expected_actual_module_optimize == null or
        !std.mem.eql(u8, row.expected_actual_module_optimize.?, arguments.mode))
    {
        return error.InvalidRowOptimizeMode;
    }

    var native_binding: ?*const NativeObservationBinding = null;
    for (inventory.native_observation_bindings) |*candidate| {
        if (std.mem.eql(u8, candidate.row_id, row.id)) {
            if (native_binding != null) return error.DuplicateNativeObservationBinding;
            native_binding = candidate;
        }
    }

    if (std.mem.eql(u8, row.expectation_state, "requires-native-enumeration")) {
        if (row.expected_test_set_id != null or native_binding != null) return error.InvalidPendingEnumeration;
        return .requires_native_enumeration;
    }
    if (std.mem.eql(u8, row.expectation_state, "not-applicable")) return error.NotApplicableEnumerationExecuted;
    if (!std.mem.eql(u8, row.expectation_state, "frozen-compiler-enumeration")) {
        return error.InvalidExpectationState;
    }

    const expected_set_id = row.expected_test_set_id orelse return error.MissingExpectedTestSetReference;
    const binding = native_binding orelse return error.MissingNativeObservationBinding;
    if (!std.mem.eql(u8, binding.evidence_slot_id, row.evidence_slot_id) or
        !std.mem.eql(u8, binding.enumeration_class_id, row.enumeration_class_id) or
        !std.mem.eql(u8, binding.optimize_mode_id, row.optimize_mode_id) or
        !std.mem.eql(u8, binding.expected_test_set_id, expected_set_id) or
        !std.mem.eql(u8, binding.enumeration_source, "zig-0.16-builtin-test-functions"))
    {
        return error.NativeObservationBindingMismatch;
    }
    try validateBindingIdentity(allocator, binding);
    var expected_set: ?*const ExpectedTestSet = null;
    for (inventory.expected_test_sets) |*candidate| {
        if (std.mem.eql(u8, candidate.id, expected_set_id)) {
            if (expected_set != null) return error.DuplicateExpectedTestSet;
            expected_set = candidate;
        }
    }
    const expected = expected_set orelse return error.MissingExpectedTestSet;
    if (!std.mem.eql(u8, expected.root_id, arguments.root_id) or
        !std.mem.eql(u8, expected.enumeration_source, class.enumeration_source))
    {
        return error.ExpectedTestSetIdentityMismatch;
    }
    if (expected.count != expected.tests.len or expected.count != builtin.test_functions.len) {
        return error.TestCountMismatch;
    }
    const root_slug = try slugAlloc(allocator, arguments.root_id);
    defer allocator.free(root_slug);
    for (expected.tests, builtin.test_functions, 0..) |expected_test, actual_test, ordinal| {
        const expected_test_id = try std.fmt.allocPrint(allocator, "test:{s}:{d}", .{ root_slug, ordinal });
        defer allocator.free(expected_test_id);
        if (!std.mem.eql(u8, expected_test.id, expected_test_id) or
            expected_test.ordinal != ordinal or
            !std.mem.eql(u8, expected_test.name, actual_test.name))
        {
            return error.TestIdentityMismatch;
        }
    }
    const digest = try canonicalDigest(allocator, expected.tests);
    if (!std.mem.eql(u8, expected.digest, &digest)) return error.ExpectedTestSetDigestMismatch;
    const canonical_set_id = try std.fmt.allocPrint(allocator, "set:{s}:{s}", .{ expected.root_id, digest });
    defer allocator.free(canonical_set_id);
    if (!std.mem.eql(u8, expected.id, canonical_set_id) or
        !std.mem.eql(u8, expected_set_id, canonical_set_id))
    {
        return error.ExpectedTestSetIdentityMismatch;
    }
    return .frozen;
}

fn readInventoryAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const directory_path = std.fs.path.dirname(path) orelse ".";
    const basename = std.fs.path.basename(path);
    if (basename.len == 0 or
        std.mem.eql(u8, basename, ".") or
        std.mem.eql(u8, basename, ".."))
    {
        return error.InvalidInventoryPath;
    }

    const directory = try Io.Dir.cwd().openDir(runner_io, directory_path, .{
        .follow_symlinks = false,
    });
    defer directory.close(runner_io);

    const file = try openInventoryFile(directory, basename);
    defer file.close(runner_io);

    const before = try file.stat(runner_io);
    if (before.kind != .file) return error.InvalidInventoryFileType;
    if (before.inode == 0) return error.InventoryIdentityUnavailable;
    if (before.size > maximum_inventory_bytes) return error.InventoryTooLarge;

    const frozen_size: usize = @intCast(before.size);
    const bytes = try allocator.alloc(u8, frozen_size);
    errdefer allocator.free(bytes);
    const read_count = try file.readPositionalAll(runner_io, bytes, 0);
    if (read_count != frozen_size) return error.InventoryTruncatedDuringRead;

    var growth_probe: [1]u8 = undefined;
    if (try file.readPositionalAll(runner_io, &growth_probe, before.size) != 0) {
        return error.InventoryGrewDuringRead;
    }

    const after = try file.stat(runner_io);
    if (!inventoryMetadataStable(before, after)) {
        return error.InventoryChangedDuringRead;
    }
    const admitted_path = try directory.statFile(runner_io, basename, .{
        .follow_symlinks = false,
    });
    if (!inventoryMetadataStable(before, admitted_path)) {
        return error.InventoryPathReboundDuringRead;
    }
    return bytes;
}

fn inventoryMetadataStable(before: Io.File.Stat, after: Io.File.Stat) bool {
    return after.kind == .file and
        after.inode == before.inode and
        after.nlink == before.nlink and
        after.size == before.size and
        after.permissions == before.permissions and
        after.mtime.nanoseconds == before.mtime.nanoseconds and
        after.ctime.nanoseconds == before.ctime.nanoseconds and
        after.block_size == before.block_size;
}

fn openInventoryFile(directory: Io.Dir, basename: []const u8) !Io.File {
    return switch (builtin.os.tag) {
        .windows => directory.openFile(runner_io, basename, .{
            .follow_symlinks = false,
        }),
        else => openInventoryFilePosix(directory, basename),
    };
}

fn openInventoryFilePosix(directory: Io.Dir, basename: []const u8) !Io.File {
    var flags: std.posix.O = .{
        .ACCMODE = .RDONLY,
        .NONBLOCK = true,
        .NOFOLLOW = true,
    };
    if (@hasField(std.posix.O, "CLOEXEC")) flags.CLOEXEC = true;
    if (@hasField(std.posix.O, "LARGEFILE")) flags.LARGEFILE = true;
    if (@hasField(std.posix.O, "NOCTTY")) flags.NOCTTY = true;

    return .{
        .handle = try std.posix.openat(directory.handle, basename, flags, 0),
        .flags = .{ .nonblocking = true },
    };
}

fn validateInventoryByteCount(byte_count: usize) !void {
    if (byte_count > maximum_inventory_bytes) return error.InventoryTooLarge;
}

fn validateInventoryDigest(bytes: []const u8) !void {
    var digest_bytes: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest_bytes, .{});
    const digest = std.fmt.bytesToHex(digest_bytes, .lower);
    if (std.mem.eql(u8, &digest, CURRENT_TEST_INVENTORY_SHA256)) return;
    if (NEXT_TEST_INVENTORY_SHA256) |next| {
        if (std.mem.eql(u8, &digest, next)) return;
    }
    return error.UnreviewedTestInventoryDigest;
}

fn validateRowIdentity(allocator: std.mem.Allocator, row: *const TestModeRow, arguments: Arguments) !void {
    const environment_slug = try slugAlloc(allocator, arguments.environment_id);
    defer allocator.free(environment_slug);
    const root_slug = try slugAlloc(allocator, arguments.root_id);
    defer allocator.free(root_slug);
    const row_id = try std.fmt.allocPrint(allocator, "row:{s}:{s}:{s}", .{
        environment_slug,
        root_slug,
        arguments.mode,
    });
    defer allocator.free(row_id);
    const evidence_slot_id = try std.fmt.allocPrint(allocator, "evidence-slot:{s}", .{row_id["row:".len..]});
    defer allocator.free(evidence_slot_id);
    if (!std.mem.eql(u8, row.id, row_id) or
        !std.mem.eql(u8, row.evidence_slot_id, evidence_slot_id))
    {
        return error.InvalidTestModeRowIdentity;
    }
}

fn findEnvironmentProfile(profiles: []const EnvironmentProfile, environment_id: []const u8) !*const EnvironmentProfile {
    var result: ?*const EnvironmentProfile = null;
    for (profiles) |*profile| {
        if (std.mem.eql(u8, profile.id, environment_id)) {
            if (result != null) return error.DuplicateEnvironmentProfile;
            result = profile;
        }
    }
    return result orelse error.MissingEnvironmentProfile;
}

fn validateEnvironmentProfile(profile: *const EnvironmentProfile) !void {
    if (!std.mem.eql(u8, profile.cpu, "baseline")) return error.InvalidEnvironmentCpu;
    const baseline_cpu = std.Target.Cpu.baseline(builtin.cpu.arch, builtin.os);
    if (builtin.cpu.model != baseline_cpu.model or
        !builtin.cpu.features.eql(baseline_cpu.features))
    {
        return error.NonCanonicalBaselineCpu;
    }

    const expected = if (builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos and builtin.target.abi == .none)
        EnvironmentProfile{
            .id = "env:aarch64-macos-baseline",
            .target = "aarch64-macos",
            .architecture = "aarch64",
            .os = "macos",
            .libc = "system",
            .cpu = "baseline",
            .resolved_cpu_model = "apple_m1",
            .cpu_feature_policy = "canonical-baseline-resolved-features",
            .host_tool_smoke = true,
        }
    else if (builtin.cpu.arch == .aarch64 and builtin.os.tag == .linux and builtin.target.abi == .gnu)
        EnvironmentProfile{
            .id = "env:aarch64-linux-gnu-baseline",
            .target = "aarch64-linux-gnu",
            .architecture = "aarch64",
            .os = "linux",
            .libc = "gnu",
            .cpu = "baseline",
            .resolved_cpu_model = "generic",
            .cpu_feature_policy = "canonical-baseline-resolved-features",
            .host_tool_smoke = true,
        }
    else if (builtin.cpu.arch == .x86_64 and builtin.os.tag == .linux and builtin.target.abi == .gnu)
        EnvironmentProfile{
            .id = "env:x86-64-linux-gnu-baseline",
            .target = "x86_64-linux-gnu",
            .architecture = "x86_64",
            .os = "linux",
            .libc = "gnu",
            .cpu = "baseline",
            .resolved_cpu_model = "x86_64",
            .cpu_feature_policy = "canonical-baseline-resolved-features",
            .host_tool_smoke = true,
        }
    else if (builtin.cpu.arch == .x86_64 and builtin.os.tag == .windows and builtin.target.abi == .gnu)
        EnvironmentProfile{
            .id = "env:x86-64-windows-gnu-baseline",
            .target = "x86_64-windows-gnu",
            .architecture = "x86_64",
            .os = "windows",
            .libc = "gnu",
            .cpu = "baseline",
            .resolved_cpu_model = "x86_64",
            .cpu_feature_policy = "canonical-baseline-resolved-features",
            .host_tool_smoke = false,
        }
    else
        return error.UnsupportedInventoryEnvironment;

    if (!environmentProfileEquals(profile, &expected)) return error.EnvironmentProfileMismatch;
}

fn environmentProfileEquals(actual: *const EnvironmentProfile, expected: *const EnvironmentProfile) bool {
    return std.mem.eql(u8, actual.id, expected.id) and
        std.mem.eql(u8, actual.target, expected.target) and
        std.mem.eql(u8, actual.architecture, expected.architecture) and
        std.mem.eql(u8, actual.os, expected.os) and
        std.mem.eql(u8, actual.libc, expected.libc) and
        std.mem.eql(u8, actual.cpu, expected.cpu) and
        std.mem.eql(u8, actual.resolved_cpu_model, expected.resolved_cpu_model) and
        std.mem.eql(u8, actual.cpu_feature_policy, expected.cpu_feature_policy) and
        actual.host_tool_smoke == expected.host_tool_smoke;
}

fn findEnumerationClass(classes: []const TestEnumerationClass, class_id: []const u8) !*const TestEnumerationClass {
    var result: ?*const TestEnumerationClass = null;
    for (classes) |*class| {
        if (std.mem.eql(u8, class.id, class_id)) {
            if (result != null) return error.DuplicateEnumerationClass;
            result = class;
        }
    }
    return result orelse error.MissingEnumerationClass;
}

fn validateBindingIdentity(allocator: std.mem.Allocator, binding: *const NativeObservationBinding) !void {
    const facts = BindingFacts{
        .enumeration_class_id = binding.enumeration_class_id,
        .enumeration_source = binding.enumeration_source,
        .evidence_slot_id = binding.evidence_slot_id,
        .expected_test_set_id = binding.expected_test_set_id,
        .optimize_mode_id = binding.optimize_mode_id,
        .row_id = binding.row_id,
    };
    const digest = try canonicalDigest(allocator, facts);
    if (!std.mem.eql(u8, binding.digest, &digest)) return error.NativeObservationBindingDigestMismatch;
    const binding_id = try std.fmt.allocPrint(allocator, "native-observation:{s}", .{digest});
    defer allocator.free(binding_id);
    if (!std.mem.eql(u8, binding.id, binding_id)) return error.NativeObservationBindingIdentityMismatch;
}

fn canonicalDigest(allocator: std.mem.Allocator, value: anytype) ![64]u8 {
    const canonical = try std.json.Stringify.valueAlloc(allocator, value, .{});
    defer allocator.free(canonical);
    var digest_bytes: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(canonical, &digest_bytes, .{});
    return std.fmt.bytesToHex(digest_bytes, .lower);
}

fn slugAlloc(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    if (!std.unicode.utf8ValidateSlice(value)) return error.InvalidUtf8Identity;
    var result: Io.Writer.Allocating = .init(allocator);
    defer result.deinit();
    var separator_pending = false;
    var index: usize = 0;
    while (index < value.len) {
        const byte = value[index];
        if (std.mem.startsWith(u8, value[index..], "\u{130}")) {
            if (separator_pending and result.written().len != 0) try result.writer.writeByte('-');
            try result.writer.writeByte('i');
            separator_pending = true;
            index += "\u{130}".len;
            continue;
        }
        if (std.mem.startsWith(u8, value[index..], "\u{212a}")) {
            if (separator_pending and result.written().len != 0) try result.writer.writeByte('-');
            try result.writer.writeByte('k');
            separator_pending = false;
            index += "\u{212a}".len;
            continue;
        }
        const lowered = std.ascii.toLower(byte);
        if ((lowered >= 'a' and lowered <= 'z') or std.ascii.isDigit(lowered)) {
            if (separator_pending and result.written().len != 0) try result.writer.writeByte('-');
            try result.writer.writeByte(lowered);
            separator_pending = false;
        } else {
            separator_pending = true;
        }
        index += 1;
    }
    return result.toOwnedSlice();
}

fn actualEnumerationClassId() ![]const u8 {
    if (builtin.cpu.arch == .aarch64 and builtin.os.tag == .macos and builtin.object_format == .macho) {
        return "enumeration-class:aarch64-macos-system-macho";
    }
    if (builtin.cpu.arch == .x86_64 and builtin.os.tag == .linux and
        builtin.target.abi == .gnu and builtin.object_format == .elf)
    {
        return "enumeration-class:x86-64-linux-gnu-elf";
    }
    if (builtin.cpu.arch == .aarch64 and builtin.os.tag == .linux and
        builtin.target.abi == .gnu and builtin.object_format == .elf)
    {
        return "enumeration-class:aarch64-linux-gnu-elf";
    }
    if (builtin.cpu.arch == .x86_64 and builtin.os.tag == .windows and
        builtin.target.abi == .gnu and builtin.object_format == .coff)
    {
        return "enumeration-class:x86-64-windows-gnu-coff";
    }
    return error.UnsupportedEnumerationClass;
}

fn validateEnumerationClass(class: *const TestEnumerationClass) !void {
    const architecture = switch (builtin.cpu.arch) {
        .aarch64 => "aarch64",
        .x86_64 => "x86_64",
        else => return error.UnsupportedEnumerationArchitecture,
    };
    const os = switch (builtin.os.tag) {
        .macos => "macos",
        .linux => "linux",
        .windows => "windows",
        else => return error.UnsupportedEnumerationOs,
    };
    const libc = if (builtin.os.tag == .macos) "system" else if (builtin.target.abi == .gnu) "gnu" else return error.UnsupportedEnumerationLibc;
    const object_format = switch (builtin.object_format) {
        .macho => "macho",
        .elf => "elf",
        .coff => "coff",
        else => return error.UnsupportedEnumerationObjectFormat,
    };

    if (!std.mem.eql(u8, class.language, "zig") or
        !optionalEquals(class.architecture, architecture) or
        !optionalEquals(class.os, os) or
        !optionalEquals(class.libc, libc) or
        !optionalEquals(class.object_format, object_format) or
        !std.mem.eql(u8, class.enumeration_source, "zig-0.16-builtin-test-functions") or
        class.environment_ids.len == 0)
    {
        return error.InvalidEnumerationClass;
    }
    for (class.environment_ids, 0..) |environment_id, ordinal| {
        if (environment_id.len == 0 or countMatches(class.environment_ids[ordinal..], environment_id) != 1) {
            return error.InvalidEnumerationClassEnvironments;
        }
    }
}

fn optionalEquals(actual: ?[]const u8, expected: []const u8) bool {
    return if (actual) |value| std.mem.eql(u8, value, expected) else false;
}

fn modeIdMatches(mode_id: []const u8, mode: []const u8) bool {
    return mode_id.len == "mode:".len + mode.len and
        std.mem.eql(u8, mode_id[0.."mode:".len], "mode:") and
        std.mem.eql(u8, mode_id["mode:".len..], mode);
}

fn countMatches(values: []const []const u8, needle: []const u8) usize {
    var count: usize = 0;
    for (values) |value| {
        if (std.mem.eql(u8, value, needle)) count += 1;
    }
    return count;
}

fn writeHex(writer: *Io.Writer, comptime tag: []const u8, bytes: []const u8) void {
    writer.print("{s}:{d}:", .{ tag, bytes.len }) catch @panic("inventory runner write failed");
    for (bytes) |byte| writer.print("{x:0>2}", .{byte}) catch @panic("inventory runner write failed");
    writer.writeByte('\n') catch @panic("inventory runner write failed");
}
