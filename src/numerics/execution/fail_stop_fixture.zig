// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//! Uninstalled focused subprocess fixture; never imported by execution/root.
pub fn main() void {
    @import("provider_test.zig").failStopFixture(@import("zynum-build-options").failure_mode);
}
