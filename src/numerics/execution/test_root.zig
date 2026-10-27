// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
test {
    _ = @import("provider_test.zig");
    _ = @import("call_resources.zig");
    if (@import("zynum-build-options").reference_compat) {
        _ = @import("context_test.zig");
        _ = @import("context_scratch_test.zig");
    }
    if (@import("zynum-build-options").actual_runtime) _ = @import("actual_provider_test.zig");
}
