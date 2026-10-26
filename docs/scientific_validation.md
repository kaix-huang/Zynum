# Scientific branch validation

Development snapshot consolidated on 2026-10-26 on native Apple arm64 macOS,
Zig 0.17.0 and Python 3.14.7. This record separates completed checks from the
broader cold-build validation still running. It is not a release certification.

## Completed checks

| Check | Actual result |
| --- | --- |
| Full Debug aggregate plus POSV/lifecycle and profile checks | 147/147 steps; 1,786/1,790 Zig bodies passed, four original skips |
| LAPACK / Sparse / numerical contracts / execution contracts in that run | 758 / 203 / 208 / 23 bodies passed |
| Final ReleaseFast scientific SDK | 70 installed files, ten numerical libraries, 28 headers |
| Installed LP64 C11/C++20 clients | 48 passed across GESVD, complex GELSD, GELSS and GELS |
| Calls made by those clients | 249,984 numerical calls; 5,424 queries; 3,048 boundaries; 336 symbol-owner checks |
| Committed generated-interface drift check | 256 selections; 1,037 files matched |
| Current build inventory and test-inventory structure | Both passed; migration slots closed |
| Current positive inventory fixture and real source archive | Both passed |
| Formatting, Python compilation and staged whitespace | Passed |

All 28 installed headers and five dynamic export sets match the preceding
qualified GESVD SDK. Each of the ten artifacts contains one default Context;
the six LAPACKE-bearing artifacts each contain one existing configuration state.
Independent libraries retain independent owners. Dynamic export counts are
846 unified, 176 LAPACKE LP64, 92 LAPACKE ILP64 and 35 per Sparse width.
The 48 clients cover static/shared, standalone/unified and default/std::complex
variants, with exact loaded-image checks. They do not qualify ILP64 GELS/GELSS
or new ABPLaS adoption.

The numerical runs use an immutable 1,746-file source snapshot. Public `src/`
and `test/` files and `build.zig` match the consolidated numerical source.
Later documentation finalization and one operator-renderer trailing-space
cleanup do not change its Python AST. Its two source pins were refreshed;
the current build inventory and three contour/profile tests passed again.
The final SDK was rebuilt after that cleanup and its 48 clients reran.

Inventory checker algorithms are unchanged; source pins, classifications,
provenance and digest constants were reviewed as data. Seventy-two existing
host compiler-enumeration bindings are retained. The 222 foreign native rows
remain pending. Enumeration is separate from actual numerical execution.

Focused partial-GESVD probes on the integrated candidate separately passed
48 original failure-payload cases and 23,040 job/layout/scalar/budget cases in
each of Debug, ReleaseSafe and ReleaseFast. Each broad run included 13,440
positive-INFO and 9,600 successful cases. These internal budget probes do not
inject a production C budget or establish bitwise vendor payload equivalence.

## Checks still running at the initial code publication

The broader full ReleaseSafe aggregate is running. Full ReleaseFast,
standalone host-tool smoke and the default build follow in the same guarded
pipeline. They are not counted as passed in this snapshot. The completed
checks above are independent of those pending results. The record will be
updated after their actual native exit results are available.

## Reproduction and open integration

Use the sanitized environment and canonical commands in the
[contributor guide](contributors/README.md). Local runs use baseline CPU,
matching production/test optimize modes, `-j2` and isolated caches. Supplemental
checks select POSV, complex Cholesky lifecycle and the existing finite
interface/profile tests. Scientific SDK flags are in the
[interface guide](interop/README.md).

Earlier ABPLaS regressions and material runs retain their original source/SDK
identities. No fresh application or material acceptance is claimed for this
consolidation. The interrupted newer material and source-closeout jobs remain
unqualified. Full TimeEvolution, statistical convergence, wider routine/profile
coverage and foreign native environments remain open. Historical benchmarks
retain their [original measurement scope](performance/README.md).
