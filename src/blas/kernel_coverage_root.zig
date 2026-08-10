// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

const coverage = @import("kernels/coverage.zig");

pub const Level = coverage.Level;
pub const Availability = coverage.Availability;
pub const Evidence = coverage.Evidence;
pub const Entry = coverage.Entry;
pub const Summary = coverage.Summary;
pub const RegistryEntry = coverage.RegistryEntry;
pub const RegistrySummary = coverage.RegistrySummary;
pub const ExecutorBinding = coverage.ExecutorBinding;
pub const entry_count = coverage.entry_count;
pub const registry_entry_count = coverage.registry_entry_count;
pub const entries = coverage.entries;
pub const registryEntries = coverage.registryEntries;
pub const summarize = coverage.summarize;
pub const summarizeRegistry = coverage.summarizeRegistry;
pub const matchesExecutorBindingBaseline = coverage.matchesExecutorBindingBaseline;
