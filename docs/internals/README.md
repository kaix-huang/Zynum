# Internal Design Guide

This layer is for maintainers and contributors who need implementation detail:
module boundaries, call paths, ABI layering, dispatch ownership, and file split
rules.

## Core Maps

| Area | Primary docs |
| --- | --- |
| Module boundaries and ownership | [`../architecture.md`](../architecture.md) |
| Public API checks and aliasing model | [`../development_and_usage.md`](../development_and_usage.md) |
| ABI and generated compatibility files | [`../fortran_compatibility.md`](../fortran_compatibility.md) |
| Zig `std.Io` threading notes | [`../common/zig_std_io_threading.md`](../common/zig_std_io_threading.md) |
| CPU affinity and heterogeneous scheduling | [`../common/cpu_affinity_and_heterogeneous_scheduling.md`](../common/cpu_affinity_and_heterogeneous_scheduling.md) |
| BLAS Level 1 implementation notes | [`../common/level1_optimization_notes.md`](../common/level1_optimization_notes.md) |
| BLAS Level 2 implementation notes | [`../common/level2_optimization_notes.md`](../common/level2_optimization_notes.md) |
| GEMM implementation rules | [`../common/gemm_optimization_notes.md`](../common/gemm_optimization_notes.md) |

## Facade Pattern

Public Zig import roots are thin aliases of one internal implementation owner:

- `src/facades/zynum.zig`
- `src/facades/blas.zig`
- `src/facades/lapack.zig`
- `src/facades/sparse.zig`

The package names `zynum`, `zynum-blas`, `zynum-lapack` and `zynum-sparse`
and their declarations remain stable; these physical roots replace the former
implementation-root bindings. The build binds all four to the same internal
owner at `src/zynum.zig` for each compatible artifact/configuration. Resources
and kernel objects attach to that implementation owner. Core/runtime consumers
share the exact named execution `Build.Module` and one options `Build.Module`
for the same target and optimization mode. Reusing a source path or options
Step does not provide module identity. Production and test modes can require
separate execution module objects.

Do not recreate implementations or runtime globals in facade modules. This
within-artifact identity does not unify separately linked DSOs or duplicate
package dependency instances. Host tools and the isolated lifecycle executable
retain their own applicable target/configuration boundaries.

Internal namespace boundaries also stay as facades:

- `src/zynum.zig`
- `src/blas.zig`
- `src/lapack.zig`
- `src/sparse.zig`
- `src/blas/api.zig`
- `src/blas/core.zig`
- `src/blas/core/unchecked.zig`
- `src/blas/core/vector.zig`
- `src/blas/core/matrix_vector.zig`
- `src/blas/core/matrix_matrix.zig`

Concrete implementations should live below those facades, grouped by operation
family or backend responsibility. Preserve facade exports when moving code so
typed APIs, ABI wrappers, and tests do not learn implementation paths. Keep ABI
wrappers on `src/blas/core/unchecked.zig`; public checked operations should continue
to use `src/blas/core.zig` or the public API facade.

## Execution Ownership

| Source | Responsibility |
| --- | --- |
| `src/numerics/execution/owner.zig` | Allocator-captured `Executor`, synchronous admission/submission/completion, two Io lifecycles and restartable `drainAndReset`. |
| `src/numerics/execution/policy.zig` | Shared thread override, hardware/environment caches and worker QoS/affinity policy. |
| `src/numerics/execution/default.zig` | One stable c_allocator-backed default executor in the shared execution module. |
| `src/numerics/execution/test_support.zig` | Regression/probe fixture policy and thread-safe real allocation-failure/accounting support. |
| `src/blas/runtime.zig` | Existing runtime controls forwarded to the common policy, plus the existing target-tier/host bridge boundary. |
| `src/blas/core/execution/thread_pool.zig` | BLAS cache-domain callback, typed task bridges and forwarding; no pool state. |

Common execution must not import BLAS. Callback code and cache-domain
metadata outlive workers; helper cleanup runs on the helper that owns the TLS.
Task zero and inline fallback do not run helper-cache release callbacks.
Keep live executors at stable addresses and borrow task payloads only through
the synchronous call. Release uses the allocator captured by that executor.

Ordinary and persistent execution retain separate `Io.Threaded` objects and
reverse initialization-order cleanup. `drainAndReset` waits for admitted work,
releases those objects and permits restart. Caller quiescence is still required
for destruction or unloading. Refused callers can run a serial fallback outside
admission; reset is not a global operation drain or terminal Context close.
Policy currently uses static function pointers and shared policy state, not
per-instance public Context configuration.

The canonical `test-execution-contract` root is `owner.zig`, with seven tests
using its fixture policy and no production options/default/resources/ISA
objects. The two source-local wrapper tests have an existing BLAS internal
root home. The BLAS module test verifies facade/default policy identity and
caller fallback; the modern consumer checks coexistence of all four facades.
Named dependencies must not be assumed to enumerate their tests. The isolated
six-case lifecycle driver also tests real zero/partial helper allocation and
signal restoration. Test contracts, linked-artifact identity, native execution
and official inventory evidence remain distinct; these descriptions are not
an integrated-release or compatibility certification.

See [Shared Task Runtime](../architecture.md#shared-task-runtime) and the
[scientific ownership map](../scientific.md) for current lifetime boundaries.
The shared Context, admitted operations, scratch leases and pin accounting are
implemented in `src/numerics`. BLAS drain/reset retains its documented restart
contract. Broader plan/cache budgets and cross-DSO sharing remain separate work.
