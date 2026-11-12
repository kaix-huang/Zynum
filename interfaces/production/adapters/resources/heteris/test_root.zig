// SPDX-License-Identifier: LGPL-3.0-or-later
test {
    _ = @import("provider.zig");
    _ = @import("provider_test.zig");
    if (@import("zynum-build-options").actual_runtime) _ = @import("actual_provider_test.zig");
}
