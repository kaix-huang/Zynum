# Native RuntimeProvider SDK (development)

The explicit-provider native FP64 NN path has bounded local CPU qualification.
Candidate 2 reference Core Debug, Safe and Fast each passed 1839 tests; its
host gate failed. After an independently reviewed Python fixture-only repair,
a new source generation passed host and CPU integration with installed C/C++
consumers, with full-source before/after identity independently checked. These
are combined evidence across generations, not one complete Core rerun.
Packed, SIMD, GEMV, FFT, Sparse and legacy resource paths need further migration
and qualification. No cluster, materials/SCF, GPU or performance qualification
is established. Earlier checked-bounds evidence retains its own source identity.

## Profiles and builds

The root builder offers `reference` and `integrated_native` execution profiles.
`reference` provides compatibility services and a reference default; the default
is not an integrated RuntimeProvider. `integrated_native` requires explicit
provider binding and does not create a local pool, allocator, Runtime or
implicit fallback. The profiles share the original Core, with one main
`libzynum` artifact; no second native Core archive is required.

Provide an **absolute header-only** Heteris ABI SDK directory containing
`heteris.h` and `heteris_runtime_provider.h`. No Rust, MPI or CUDA is needed to
consume those headers or to build Zynum against this SDK.
The caller supplies and links an actual Runtime implementation separately when
it needs actual-provider execution. Use the same normalized SDK path for the
CLI and inherited environment, especially for nested build probes:

```sh
SDK=/absolute/path/to/heteris/abi/include
OUT=/absolute/path/to/external-build
export ZYNUM_HETERIS_INCLUDE="$SDK"
zig build platform -Dexecution-profile=integrated_native -Doptimize=safe \
  -Dheteris-include="$SDK" --prefix "$OUT/install" --cache-dir "$OUT/cache"
```

`platform` installs the main artifact and headers to the prefix. Standalone
compatibility provider is separately opt-in with `-Dstandalone=true`; it is not
an implicit provider for native calls. To run the original reference baseline,
select it explicitly and use separate external caches:

```sh
SDK=/absolute/path/to/heteris/abi/include
OUT=/absolute/path/to/external-reference
export ZYNUM_HETERIS_INCLUDE="$SDK"
zig build test test-lapacke-posv test-checked-cholesky-lifecycle \
  -Dexecution-profile=reference -Dheteris-include="$SDK" -Dcpu=baseline \
  -Doptimize=safe -Dtest-optimize=safe --summary all \
  --prefix "$OUT/install" --cache-dir "$OUT/cache"
```

The full project qualification workflow must additionally isolate global and
child caches and record its inputs and results. CLI-only configuration works
for ordinary root builds; nested historical Python probes need the inherited
variable. Mismatched paths fail rather than selecting an adjacent SDK.

Actual-provider test targets require a **prebuilt** Runtime archive; without
it they are not registered, and the builder does not build or discover Rust:

```sh
SDK=/absolute/path/to/heteris/abi/include
RUNTIME=/absolute/path/to/libheteris_runtime.a
OUT=/absolute/path/to/external-provider-tests
export ZYNUM_HETERIS_INCLUDE="$SDK"
zig build test-native-provider-math test-native-runtime-c \
  -Dexecution-profile=integrated_native -Doptimize=safe \
  -Dheteris-include="$SDK" -Dheteris-runtime-archive="$RUNTIME" \
  --prefix "$OUT/install" --cache-dir "$OUT/cache"
```

A build or example command is not a claim that this worktree passed a gate.

## Context and calls

`zynum_native.h` declares the versioned C API. Query storage size and alignment,
then supply stable caller-owned storage to `zynum_native_context_init_v1`.
Keep storage, provider-owned state and the implementation image alive through
close and every returning API call. The provider and policy descriptors are
borrowed for initialization; their required contents are copied and the provider
is retained. Follow the header's ABI version, struct size and reserved-field
requirements. Report storage must not overlap Context, descriptors or mutable
mathematical data. No implicit Context or Runtime is created.

Operations are admitted through the Context. Requesting close prevents new
admissions; progress waits for admitted operations and their resource pins.
Integrated close releases its retained provider token and does **not** drain or
reset the shared Runtime. If release fails, the Context retains the token in
`ZYNUM_NATIVE_CLOSE_RELEASE_FAILED`; an explicit later progress call can retry.
Keep the provider and storage alive until close succeeds. Inspect each returned
status and report: resource stage, submission and output states, close state,
raw provider/execution/callback/cleanup statuses and callback counts. Output may
be unchanged, complete, or partial/invalid; do not assume success after failure.

The first native math slice is checked FP64 column-major NN GEMM using explicit
`ZYNUM_NATIVE_ALGORITHM_PORTABLE_GENERIC`. Buffers are borrowed and their backing
extents and leading dimensions are checked. Every nonempty output write goes
through Heteris, including one-task and alpha-zero or zero-k work. A provider
rejection, allocation failure or execution failure is reported; the operation
does not fall back locally or replay potentially partial writes. The numerical
seam uses the original Core implementation.

For reference-only Zig use, `ReferenceContext.initInPlace` creates a wrapper
whose internal references require a stable address: do not move it after
initialization. Its caller-owned storage and allocator state must outlive close
and returning calls. The new reference default is a reference service, not an
integrated provider. Consult the actual declarations and current inventory
checker before relying on a profile or test result.
