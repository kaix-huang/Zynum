# Scientific branch validation

Code consolidated on 2026-10-26; final validation completed on 2026-10-27.
Host: native Apple arm64 macOS with Zig 0.17.0 and Python 3.14.7.
These are functional and packaging checks, not new performance
measurements or full vendor certification.

## Current numerical source

The full numerical runs used an immutable 1,746-file public-source snapshot.
Every public file under `src/` and `test/`, and `build.zig`, is byte-identical
to the consolidated tree. Later changes finalized documentation and removed one
trailing space from the operator renderer; its Python AST is unchanged. Its two
source-provenance pins were refreshed, and the current build inventory and
three contour/profile tests passed again. The final SDK was built from a
separate frozen snapshot containing that cleanup.

| Actual job | Result |
| --- | --- |
| canonical-debug | 147/147 steps succeeded; 1786/1790 tests passed (4 skipped) |
| canonical-safe | 147/147 steps succeeded; 1786/1790 tests passed (4 skipped) |
| canonical-fast | 147/147 steps succeeded; 1783/1790 tests passed (7 skipped) |
| host-tool-smoke | 12/12 steps succeeded |
| default-build | 37/37 steps succeeded |

Each canonical mode actually ran 26 Zig test executables, including the
supplemental POSV and complex Cholesky lifecycle roots. Each passed 758 LAPACK,
203 Sparse, 208 common numerical-contract and 23 execution-contract bodies.
Debug/Safe retain four existing x86-specific skips on this host. Fast also
skips three pre-existing checked-build tests (short storage and two alias
rejections) guarded by `builtin.mode == .fast` in `test/api/zynum_test.zig`.
No numerical run was accepted from a cached test result. Host-tool smoke passed
492 benchmark-tooling cases (nine original skips), 194 ABI-tooling cases (one
original skip) and four build-profile cases. Profile and tooling tests do not
add numerical or performance coverage.

The final formatting check, Python tool compilation, staged whitespace check,
strict current build inventory and current test-inventory structure check pass.
The interface entry point re-renders 256 accepted selections and exactly
matches all 1,037 committed generated files. Its direct-renderer equivalence,
invalid-profile rejection and drift rejection checks also pass. Inventory
checker algorithms remain unchanged; reviewed source pins, classifications,
provenance and digest constants were updated as data only. All migration slots
are closed. Seventy-two existing host enumeration bindings are retained;
222 foreign native rows remain pending.

The current positive test-inventory fixture and real package archive checks
both pass. Public archives contain source, tests, declarations and documentation;
SDK binaries, caches and raw local records remain outside the repository.

## Installed external calls

The final ReleaseFast SDK contains 70 files, including ten numerical libraries
and 28 headers. All 28 headers and five dynamic export sets match the preceding
qualified GESVD SDK. Each artifact contains one default Context; the six
LAPACKE-bearing artifacts each contain one existing configuration state.
Independent libraries retain independent owners.

| Dynamic artifact | Defined public exports |
| --- | ---: |
| `libzynum_abplas_lp64.dylib` | 846 |
| `libzynum_lapacke_ilp64.dylib` | 92 |
| `libzynum_lapacke_lp64.dylib` | 176 |
| `libzynum_sparse_ilp64.dylib` | 35 |
| `libzynum_sparse_lp64.dylib` | 35 |

Forty-eight freshly compiled and executed LP64 clients pass against the final
SDK: GESVD, complex GELSD, GELSS and GELS, each in twelve C11/C++20,
static/shared, standalone/unified and default/std::complex combinations.
They perform 249,984 numerical calls, 5,424 queries, 3,048 boundary checks and
336 symbol-owner checks. Exact loaded images are checked; the SDK payloads
remain unchanged. This does not qualify ILP64 GELS/GELSS or new ABPLaS adoption.

Focused probes on the integrated partial-GESVD candidate separately passed
48 original failure-payload cases and 23,040 job/layout/scalar/budget cases in
each of Debug, ReleaseSafe and ReleaseFast. Each broad run included 13,440
positive-INFO cases and 9,600 successful cases. Those internal budget probes
are separate from installed C calls; they do not inject a production C budget
or establish bitwise vendor partial-payload equivalence.

## Reproduction and open integration

Use the sanitized environment from the [contributor guide](contributors/README.md).
The canonical runs selected `test`, `test-lapacke-posv`,
`test-checked-cholesky-lifecycle` and the existing profile checks for POSV,
TRTRS, TRTRI, GETRI, POTRI, checked Sparse/Cholesky, FEAST, contour, SYEV,
SYGV, controls, EVD, GVD, GELS and GELSS. Controls/EVD/GVD/GELS/GELSS options
were enabled; CPU was baseline, host-tool smoke was disabled during those
three jobs, and both optimize options selected the same Debug/ReleaseSafe/
ReleaseFast mode. Host-tool smoke and the default build ran separately.
Local runs used `-j2` and isolated caches outside the checkout. SDK selection
commands are in the [interface guide](interop/README.md).

Earlier ABPLaS regressions and material runs retain their original source/SDK
identities. No fresh ABPLaS build or material acceptance is claimed for this
consolidation. The interrupted newer material and source-closeout jobs remain
unqualified; completed numerical and installed-client checks are recorded
separately above. Full material TimeEvolution, statistical convergence, broader
routine/profile coverage and foreign native environments remain open.
Historical benchmarks retain their [original measurement scope](performance/README.md).
