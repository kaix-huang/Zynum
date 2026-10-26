# Integration source record

This integration derives its scalar arithmetic and portable GEMM reference from
<https://github.com/kaix-huang/Zynum>, remote branch `scientific`, commit
`c5cb123e55eaae683de09b0c8030837f5600f140`.

The platform now uses this complete repository as a submodule, based on
`scientific` commit `a68d742f9e442fd401e3d13b565beae2884a775f`. That base differs
from the reference revision above only in documentation. Development continues
on `scientific-platform`; the superproject pins commits rather than a moving branch.
The adapter imports these original files directly through
`src/scientific_platform_sources.zig` (paths below are relative to the repository):

| Used module | Original file |
| --- | --- |
| scalar types | `src/blas/types.zig` |
| scalar arithmetic | `src/blas/core/shared/scalar.zig` |
| recurrence arithmetic | `src/numerics/recurrence_arithmetic.zig` |
| `LICENSE` | `LICENSE` (GNU LGPL v3) |
| `COPYING` | `COPYING` (GNU GPL v3) |

There is no `src/upstream` copy. Existing sibling checkouts and their unrelated
local edits remain untouched. The repository is an actively developed component,
not a frozen vendor snapshot. Changes to generic mathematics belong in the
original modules; C ABI/provider/view admission belongs in this integration.
The byte-stride adapters, Heteris integration, periodic stencil, and build/test
scaffolding are new bridge implementation under LGPL-3.0-or-later.

All subsequent `src/` adapter paths are relative to `integrations/heteris/`.
`src/adapted/gemm.zig` is an explicitly adapted extraction of the scalar serial
loops in upstream `src/blas/core/matrix_matrix/gemm.zig`, functions
`gemmWithResources` and `gemmComplexPortable`, at the same revision. The exact
upstream Git blob is `198b22be0eb636361a7369c458a52275f9ee7923`. Copyright and
LGPL notices are retained. The extraction retains the zero-alpha/zero-k scaling
branch, ordered dot products, and beta-zero no-read branch. It replaces BLAS
column-major pointers/leading dimensions with a compile-time matrix accessor
that implements Heteris signed byte strides and transpose/conjugation. Its only
source dependency is the unchanged `scalar.zig` above, which itself depends on
the unchanged `types.zig`. Provider/descriptor admission resides in the new
`src/dense.zig`, and shared validation/mapping in `src/views.zig` delegates to
the authoritative Heteris C validator. No upstream executor, pool, TLS scratch,
packing, architecture-specific kernels, or optimized BLAS dispatch was imported.

The unchanged `recurrence_arithmetic.zig` above has SHA256
`16624a3b7fe2487ea48db2de2b809c42fa34aa0ef95a0c02d7360746e81cacdc` and depends
only on Zig's standard library. `src/spectral.zig` adapts the `panelAdmitted` and
`seriesArithmetic` algorithm from upstream `src/numerics/chebyshev.zig` at the
same revision. Explicit Heteris views, resource callbacks, operator callbacks and
caller workspace replace the upstream execution admission and operator graph.
Four recurrence panels, polynomial order, no endpoint halving, checked widened
arithmetic, and final output commit are preserved. This is a generic polynomial
application, without an implicit spectral interval estimator or DOS model.

`zynum_chebyshev_moments_f64` in `src/spectral.zig` is new bridge code: it
composes the same checked affine block recurrence with widened per-column dot
products and a caller-owned staged commit. The pinned upstream Chebyshev file
was inspected from its Git object; that revision has no Chebyshev moments API
to import. The new API returns raw `X_j^T T_k(A) X_j` values with no sampling,
spin, DOS reconstruction, normalization or spectral interval estimation policy.
It preserves the upstream notices and imports the unchanged arithmetic helper.

The upstream package declares minimum Zig **0.17.0**. The integration now uses
the global WSL **0.17.0** installation; the earlier manual 0.16 installation and
its PATH override have been removed. Only the scalar/recurrence subset and
adapted portable GEMM/polynomial references above have been integrated and
verified here. The upstream optimized BLAS dispatch, vector kernels, and thread
pool remain pending migration. Historical 0.16 evidence remains identified by
its recorded compiler version.
