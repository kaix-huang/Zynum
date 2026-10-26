// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const checks = @import("general-operator-tests").ChainChecks;
test "Lanczos coefficient chain four scalar phased tridiagonals retain exact coefficients and tails" {
    try checks.checkExactTridiagonal();
}
test "Lanczos coefficient chain agrees with independent finite eta diagonal spectral measure" {
    try checks.checkIndependentSpectralMeasure();
}
test "Lanczos coefficient chain uses strict absolute breakdown threshold and exact action depth" {
    try checks.checkStrictThresholdAndDepthOne();
}
test "Lanczos coefficient chain validates all ten mixed byte aliases and declared tails" {
    try checks.checkMixedAliasesAndTails();
}
test "Lanczos coefficient chain preflight rejects finite geometry capacity and overflow before action" {
    try checks.checkPreflight();
}
test "Lanczos coefficient chain late action failure preserves every coefficient output" {
    try checks.checkLateFailureTransaction();
}
test "Lanczos coefficient chain reduction and Hermitian failures preserve outputs" {
    try checks.checkReductionFailureAndHermitianThreshold();
}
test "Lanczos coefficient chain executes empty global shards without implicit dimension depth cap" {
    try checks.checkEmptyGlobalAndDepth();
}
test "Lanczos coefficient chain normalizes initial and continuing subnormal metrics in four scalar types" {
    try checks.checkSubnormalStartsAndContinuation();
}
test "Lanczos coefficient chain subnormal unit metric faults preserve outputs and release original pins" {
    try checks.checkSubnormalMetricFailures();
}
test "Lanczos coefficient chain five routes retain original admission across close and every scratch OOM" {
    try checks.checkAllRoutesCloseAndAllocationFailures();
}
