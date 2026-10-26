// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const adapter = @import("checked-sparse-tests");

test "checked mixed complex MV MM preserve padding and adjoint semantics" {
    try adapter.checkAdapter0();
}

test "legacy checked real SpMV still rejects extended adjoint flag" {
    try adapter.checkAdapter1();
}

test "checked mixed pointer preflight precedes structure reads and protects output" {
    try adapter.checkAdapter2();
}
