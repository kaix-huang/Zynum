# Cholesky C boundary source migration

This bounded B05 slice separates Cholesky C wire decoding, admission and exports
from the mixed checked-v1/Sparse source owner. It does not change the public C
ABI, scalar/layout schema, symbols, header bytes or logical installation paths.
Historical `zynum_checked_*` symbol names remain their published API identity.

The only Cholesky C exports now live in
`interfaces/production/public/c/cholesky/{real_exports,complex_exports}.zig`.
Their respective existing generators remain authoritative; `status.zig` owns
the additive public `INVALID_HERMITIAN_DIAGONAL = 15` contract. `exports.zig` is
an artifact composition module, not an additional public Zig API. Artifact
roots explicitly instantiate its real/complex export factories with the
existing wire owner. The complex product remains explicitly selected for its
reviewed Apple arm64 profile; native test execution does not grant another
platform product or ABI qualification.

`interfaces/production/adapters/c/cholesky/decode.zig` decodes triangle flags
and dense descriptors. It borrows exactly `Wire`, `densePreflight`, `readBuffer`
and `writeBuffer` from the composition-supplied wire owner. `factor_solve.zig`
opens/ends the legacy checked C per-call admission, maps result/status through
`statusCode`, and delegates to the same admitted LAPACK functions. Both real and
complex exports instantiate this one adapter with the caller's existing Core
and execution module identities. There is no runtime registration, allocation,
factor handle, second mathematical implementation or forwarding file at the old
Cholesky paths. Existing no-Context checked calls keep their old default
admission; integrated/native explicit-Context entry points are not redirected.

The descriptor/flag/pointer order is unchanged: triangle first; dense dimensions,
leading dimension and span before pointer conversion; checked pointer count,
byte extent, null/alignment/address range before constructing slices; factor A
before RHS B; both descriptor ranges before Core may read numeric memory.
Validated empty systems remain no-ops. Mathematical preflight, finite/triangle
inspection, mutation and floating-point operation order remain under the
unchanged single `src/lapack` Core.

`src/checked/implementation.zig` still owns the common ABI wire pointer/count/
dimension/span/status rules and Sparse adapters/callbacks. This is an explicit
remaining boundary, shared by operator/Contour candidate tests; this slice does
not move or duplicate it. A later shared-wire migration must close all consumers
before removing that owner. Sparse algorithms/protocols, GEMM, H resource ABI,
other checked numerics and complete H-free standalone distribution remain out
of scope. B05 as a whole is not complete.

Generator source drift guards now bind the actual separated Cholesky decode and
admission sources as well as the borrowed common wire owner. Current source
inventory records must reflect the new paths without rewriting historical native
namesets or execution evidence. Source metadata refresh grants no new native
qualification; the pre-existing pending ledger remains pending.

Required acceptance includes focused profile/lifecycle and error-order tests in
Debug/ReleaseSafe/ReleaseFast, same-toolchain baseline/candidate symbol regression,
system C/C++ consumers, independent review and full CPU integration with installed/
relocated consumers. The legacy native `test-checked-abi` profile has separately
observed baseline limitations under the selected global Zig 0.17.1 (strict ELF
extra-symbol rejection and native smoke environment/toolchain checks); failed
commands remain failures. Full integration/installed evidence and the final
source cut are recorded by the coordinating repository, not inferred here.
No material, SCF, cluster, GPU or performance qualification is claimed.
