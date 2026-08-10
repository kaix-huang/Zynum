// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const std = @import("std");
const coverage = @import("kernel-coverage");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(allocator);
    const root = if (args.len > 1) args[1] else ".";
    const streaming_vector_bytes = if (args.len > 2)
        try std.fmt.parseInt(usize, args[2], 10)
    else
        64;

    const report = coverage.entries(streaming_vector_bytes);
    const summary = coverage.summarize(&report);
    const registry = coverage.registryEntries(streaming_vector_bytes);
    const registry_summary = coverage.summarizeRegistry(&registry);
    if (registry_summary.total != coverage.registry_entry_count or
        registry_summary.portable_fallback == 0 or
        registry_summary.production == 0 or
        registry_summary.experimental == 0 or
        registry_summary.rejected == 0 or
        registry_summary.fallback_edges == 0 or
        registry_summary.maximum_fallback_depth < 2 or
        !registry_summary.unique_ids or
        !registry_summary.legal_lifecycles or
        !registry_summary.fallback_chains_terminate or
        !coverage.matchesExecutorBindingBaseline(registry_summary))
    {
        return error.InvalidKernelRegistryBaseline;
    }
    const registry_digest = registryDigest(&registry);
    var output = std.Io.Writer.Allocating.init(allocator);
    const writer = &output.writer;

    try writer.writeAll("{\n  \"schema_version\": 3,\n  \"generator\": \"zig build generate-kernel-coverage\",\n");
    try writer.print("  \"streaming_vector_bytes\": {d},\n", .{streaming_vector_bytes});
    try writer.writeAll("  \"registry_baseline\": {");
    try writer.print(
        "\n    \"stable_projection_sha256\": \"{s}\",\n    \"total\": {d},\n    \"portable_fallback\": {d},\n    \"production\": {d},\n    \"experimental\": {d},\n    \"rejected\": {d},\n    \"fallback_edges\": {d},\n    \"maximum_fallback_depth\": {d},\n    \"unique_ids\": {s},\n    \"legal_lifecycles\": {s},\n    \"fallback_chains_terminate\": {s},\n    \"bound_default\": {d},\n    \"unbound_default_eligible\": {d},\n    \"unbound_experimental\": {d},\n    \"rejected_records\": {d},\n    \"executor_binding_legal\": {s},\n    \"overall_executor_complete\": {s},\n    \"real_gemm_executor_complete\": {s},\n    \"complex_gemm_executor_complete\": {s}\n  }},\n",
        .{
            registry_digest,
            registry_summary.total,
            registry_summary.portable_fallback,
            registry_summary.production,
            registry_summary.experimental,
            registry_summary.rejected,
            registry_summary.fallback_edges,
            registry_summary.maximum_fallback_depth,
            boolString(registry_summary.unique_ids),
            boolString(registry_summary.legal_lifecycles),
            boolString(registry_summary.fallback_chains_terminate),
            registry_summary.bound_default,
            registry_summary.unbound_default_eligible,
            registry_summary.unbound_experimental,
            registry_summary.rejected_records,
            boolString(registry_summary.executor_binding_legal),
            boolString(registry_summary.overall_executor_complete),
            boolString(registry_summary.real_gemm_executor_complete),
            boolString(registry_summary.complex_gemm_executor_complete),
        },
    );
    try writer.writeAll("  \"registry_entries\": [\n");
    for (registry, 0..) |entry, index| {
        try writer.writeAll("    {\"family\": ");
        try writeJsonString(writer, @tagName(entry.family));
        try writer.writeAll(", \"stable_id\": ");
        try writeJsonString(writer, entry.stable_id);
        try writer.writeAll(", \"lifecycle\": ");
        try writeJsonString(writer, @tagName(entry.lifecycle));
        try writer.writeAll(", \"fallback_id\": ");
        if (entry.fallback_id) |fallback_id| {
            try writeJsonString(writer, fallback_id);
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(", \"executor_binding\": ");
        try writeJsonString(writer, @tagName(entry.executor_binding));
        try writer.writeAll("}");
        try writer.writeAll(if (index + 1 == registry.len) "\n" else ",\n");
    }
    try writer.writeAll("  ],\n");
    try writer.writeAll("  \"summary\": {");
    try writer.print(
        "\n    \"total\": {d},\n    \"implemented\": {d},\n    \"experimental\": {d},\n    \"rejected\": {d},\n    \"missing\": {d},\n    \"unsupported\": {d},\n    \"build_tested\": {d},\n    \"native_correctness_tested\": {d},\n    \"native_performance_tested\": {d}\n  }},\n",
        .{
            summary.total,
            summary.implemented,
            summary.experimental,
            summary.rejected,
            summary.missing,
            summary.unsupported,
            summary.build_tested,
            summary.native_correctness_tested,
            summary.native_performance_tested,
        },
    );
    try writer.writeAll("  \"entries\": [\n");
    for (report, 0..) |entry, index| {
        try writer.writeAll("    {\n      \"level\": ");
        try writeJsonString(writer, @tagName(entry.level));
        try writer.writeAll(",\n      \"stable_id\": ");
        try writeJsonString(writer, entry.stable_id);
        try writer.writeAll(",\n      \"operation\": ");
        try writeJsonString(writer, entry.operation);
        try writer.writeAll(",\n      \"scalar\": ");
        try writeJsonString(writer, entry.scalar);
        try writer.writeAll(",\n      \"implementation\": ");
        try writeJsonString(writer, entry.implementation);
        try writer.writeAll(",\n      \"specialization\": ");
        try writeJsonString(writer, entry.specialization);
        try writer.writeAll(",\n      \"capability\": ");
        try writeJsonString(writer, @tagName(entry.capability));
        try writer.writeAll(",\n      \"availability\": ");
        try writeJsonString(writer, @tagName(entry.availability));
        try writer.writeAll(",\n      \"lifecycle\": ");
        try writeJsonString(writer, @tagName(entry.lifecycle));
        try writer.writeAll(",\n      \"state\": ");
        try writeJsonString(writer, @tagName(entry.state));
        try writer.print(
            ",\n      \"evidence\": {{\"build\": {s}, \"native_correctness\": {s}, \"native_performance\": {s}}},\n      \"evidence_note\": ",
            .{
                boolString(entry.evidence.build),
                boolString(entry.evidence.native_correctness),
                boolString(entry.evidence.native_performance),
            },
        );
        try writeJsonString(writer, entry.evidence_note);
        try writer.writeAll("\n    }");
        try writer.writeAll(if (index + 1 == report.len) "\n" else ",\n");
    }
    try writer.writeAll("  ]\n}\n");

    const output_path = try std.fs.path.join(allocator, &.{ root, "docs/kernel_coverage.json" });
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = output_path,
        .data = output.written(),
    });

    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);
    try stdout_writer.interface.print(
        "Generated {d} kernel coverage entries ({d} missing, {d} unsupported, {d} rejected, {d} experimental).\n",
        .{ summary.total, summary.missing, summary.unsupported, summary.rejected, summary.experimental },
    );
    try stdout_writer.flush();
}

fn registryDigest(registry: []const coverage.RegistryEntry) [64]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var length: [8]u8 = undefined;
    for (registry) |entry| {
        hash.update(@tagName(entry.family));
        std.mem.writeInt(u64, &length, entry.stable_id.len, .little);
        hash.update(&length);
        hash.update(entry.stable_id);
        hash.update(@tagName(entry.lifecycle));
        const fallback = entry.fallback_id orelse "";
        std.mem.writeInt(u64, &length, fallback.len, .little);
        hash.update(&length);
        hash.update(fallback);
        hash.update(@tagName(entry.executor_binding));
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

fn boolString(value: bool) []const u8 {
    return if (value) "true" else "false";
}

fn writeJsonString(writer: *std.Io.Writer, value: []const u8) !void {
    const hex = "0123456789abcdef";
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0...8, 11...12, 14...0x1f => try writer.writeAll(&.{ '\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 0x0f] }),
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeByte('"');
}
