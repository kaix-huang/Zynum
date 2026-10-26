// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const checks = @import("general-operator-tests").OperatorChainChecks;
test "chain raw four scalar ABI publishes tridiagonal coefficients and preserves suffixes" {
    try checks.success();
}
test "chain raw geometry ranges and mixed aliases reject before callbacks and preserve all outputs" {
    try checks.boundaries();
}
test "chain raw callback failures preserve outputs and same operation retry completes across close" {
    try checks.failureAndClose();
}
