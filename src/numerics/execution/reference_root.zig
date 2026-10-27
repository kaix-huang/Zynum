// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const options = @import("zynum-build-options");
comptime {
    if (@hasDecl(options, "poison_reference")) {
        if (options.poison_reference) @compileError("reference legacy module poison reached");
    }
}
pub const owner = @import("owner.zig");
pub const policy = @import("policy.zig");
pub const test_support = @import("test_support.zig");
pub const ReferenceContext = @import("legacy_adapter.zig").ReferenceContext;
pub const LegacyLifetime = @import("legacy_services.zig").LegacyLifetime;
pub const ReferenceCallResources = @import("reference_call_resources.zig").ReferenceCallResources;
pub const ContextScratch = @import("context.zig").ContextScratch;
pub const get = @import("default.zig").get;
pub const getContext = @import("default.zig").getContext;
pub const beginDefault = @import("default.zig").beginDefault;
pub const defaultCacheDomain = @import("default.zig").defaultCacheDomain;
