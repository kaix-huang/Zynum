// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
const checks = @import("general-operator-tests");

test "symmetric admitted closing caller and disabled default preserve oracle and padding" {
    try checks.SymmetricAdmittedChecks.checkClosingCaller();
}

test "symmetric admitted argument errors preserve all outputs and borrowed lifetime" {
    try checks.SymmetricAdmittedChecks.checkArgumentPreservation();
}

test "symmetric admitted controlled stops zero size and scalar keep the caller admission" {
    try checks.SymmetricAdmittedChecks.checkControlledStops();
}

test "syev adapter both precisions layouts triangles jobs and levels satisfy independent eigen oracle" {
    try checks.SymmetricAdapterChecks.checkFiniteOracle();
}

test "syev adapter queries scalar quick returns and scalar error precedence avoid unrelated reads" {
    try checks.SymmetricAdapterChecks.checkQueryAndScalar();
}

test "syev adapter high NaN order compile scan and allocation diagnostics use original stages" {
    try checks.SymmetricAdapterChecks.checkNanAndDiagnosticOrder();
}

test "syev adapter every scratch allocation failure releases before diagnostics" {
    try checks.SymmetricAdapterChecks.checkAllocationFailures();
}

test "syev adapter positive INFO counts actual native tridiagonal edges" {
    try checks.SymmetricAdapterChecks.checkPositiveInfo();
}

test "syev adapter raw query nonreads and managed closing callback preserve the one owner" {
    try checks.SymmetricAdapterChecks.checkRawAndClosingOwner();
}

test "syev adapter dense known spectrum survives huge tiny and common subnormal scales" {
    try checks.SymmetricAdapterChecks.checkExtremeScales();
}

test "syev component diagnostics preserve LAPACKE and Fortran names signs and argument positions" {
    try checks.SymmetricAdapterChecks.checkDiagnosticNames();
}

test "Contour binding returns complete-space eigenpairs with one borrowed operation" {
    try checks.ContourChecks.checkCompleteSpace();
}

test "Contour binding preserves outputs on format rejection and controlled stop" {
    try checks.ContourChecks.checkFormatAndControlledStop();
}

test "Contour binding unwinds callback failure and closing owner without publication" {
    try checks.ContourChecks.checkCallbackAndClosingOwner();
}

test "Contour binding final agreement failure preserves every public output" {
    try checks.ContourChecks.checkFinalAgreement();
}

test "Contour binding native-control rejection preserves public outputs" {
    try checks.ContourChecks.checkNativeControlRejections();
}

test "Operator phase agreement retains four-scalar local and remote arithmetic failures" {
    try checks.PhaseChecks.checkChebyshevLocalAndRemoteFailures();
}
test "Operator phase agreement releases successful leases after remote rejection and retains commit admission" {
    try checks.PhaseChecks.checkAllocationAdmissionAndCommit();
}
test "Operator phase agreement compares exact scalar bits and includes empty panel shards" {
    try checks.PhaseChecks.checkExactCoefficientsAndEmptyPanel();
}
test "Operator phase agreement releases both Lanczos leases and synchronizes final status" {
    try checks.PhaseChecks.checkLanczosSecondAllocationAndStopping();
}
test "Operator phase agreement synchronizes subnormal chain metrics and commits coefficients under the original pins" {
    try checks.PhaseChecks.checkChainSubnormalAndCommit();
}

test "Chebyshev four scalar families match independent cosine polynomial and preserve tails" {
    try checks.checkCase0();
}

test "Chebyshev step widens canceling intermediates and commits only finite result" {
    try checks.checkCase1();
}

test "Chebyshev callback failure and nonfinite intermediate preserve all outputs" {
    try checks.checkCase2();
}

test "Chebyshev preflight rejects geometry workspace aliases and nonfinite input before callbacks" {
    try checks.checkCase3();
}

test "Chebyshev degree zero and zero-local action maintain defined callback sequence" {
    try checks.checkCase4();
}

test "Chebyshev scratch uses existing pin and admitted operation across close request" {
    try checks.checkCase5();
}

test "Chebyshev actual mixed-index complex CSR matches independent polynomial of Hermitian action" {
    try checks.checkCase6();
}

test "Chebyshev scratch allocation failure cleans admission and permits retry" {
    try checks.checkCase7();
}

test "Chebyshev concurrent calls retain separate scratch and one shared Context close gate" {
    try checks.checkCase8();
}

test "Chebyshev panels use real mixed-index SpMM and matrix-free fallback preserving padding" {
    try checks.checkCase9();
}

test "Chebyshev dense complex Hermitian storage and CSR share the polynomial contract" {
    try checks.checkCase10();
}

test "Chebyshev panel boundaries and late column failure preserve entire destination" {
    try checks.checkCase11();
}

test "Lanczos four scalar families recover independent diagonal spectrum and preserve input and tails" {
    try checks.checkCase12();
}

test "Lanczos distinguishes invariant subspace breakdown exhaustion and finite budget" {
    try checks.checkCase13();
}

test "Lanczos rejects nonHermitian actions callback reduction faults and zero initial vectors" {
    try checks.checkCase14();
}

test "Lanczos borrowed global reductions execute on zero local shards with one admission" {
    try checks.checkCase15();
}

test "Lanczos stable norm handles large vectors and second scratch OOM cleans both pins" {
    try checks.checkCase16();
}

test "Lanczos actual complex mixed-index CSR resolves a nondiagonal Hermitian spectrum" {
    try checks.checkCase17();
}

test "Lanczos preflight validates borrowed workspace aliases dimension options and byte overflow" {
    try checks.checkCase18();
}

test "Lanczos retains two existing typed scratch pins across a close request" {
    try checks.checkCase19();
}

test "General algorithm callbacks reuse checked Sparse wire adapters without nested admission" {
    try checks.checkCase20();
}

test "Existing checked mixed Sparse wire semantics remain unchanged" {
    try checks.checkCase21();
}

test "Generated protocol borrows the injected operation through foreign callbacks and close" {
    try checks.checkCase22();
}

test "Generated incremental step uses mixed CSR with one admission and transactional boundaries" {
    try checks.checkCase23();
}

test "Managed incremental step releases OOM retry and close pins and calls zero-local actions" {
    try checks.checkCase24();
}

test "Lanczos subnormal initial norms preserve identity and independent spectra across four scalar families" {
    try checks.checkLanczosSubnormalStarts();
}

test "Lanczos subnormal unit metric failure and close retain one admission and release both scratch pins" {
    try checks.checkLanczosSubnormalMetricFailureAndClose();
}

test "Lanczos subnormal global normalization includes empty shards and rejects zero unit metric before action" {
    try checks.checkLanczosSubnormalEmptyGlobalMetric();
}
