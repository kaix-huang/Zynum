# GEMM Optimization Notes

This document records cross-platform GEMM engineering rules for Zynum BLAS.
Architecture-specific details belong in the AArch64 and x86_64 documents.

## Layering

GEMM responsibilities are split across:

- `src/blas/core/level3.zig`: BLAS semantics and fast-path detection.
- `src/blas/gemm/dispatch.zig`: shape policy, task splitting, and threading.
- `src/blas/gemm/pool.zig`: optional worker-pool experiments.
- `src/blas/kernels/backend.zig`: backend selection by target features.
- `src/blas/kernels/gemm_task.zig`: shared task description.
- `src/blas/kernels/<arch>/`: architecture-specific kernels.

Keep these boundaries strict. Shape policy does not belong inside ABI wrappers
or public Zig API code.

## Dispatch Principles

- Dispatch on capabilities, not marketing CPU names.
- Keep exact-shape gates narrow and documented.
- Prefer conservative default behavior over broad gates with mixed data.
- Use environment variables for experiments before making them default.
- Keep comparator-library measurements isolated when worker state can persist.

## Dispatch Rule Records

Any shape gate or dispatch rule that remains enabled by default must have a
written record. Put cross-platform rules here and architecture-specific rules in
the matching AArch64 or x86_64 note.

Required fields:

- Target predicate: target triple, required features, and any compile-time CPU
  tier such as ASIMD/FMA, SME, AVX2/FMA, or AVX512F/FMA.
- Shape predicate: dtype, transpose flags, `m`, `n`, `k`, alpha/beta assumptions,
  and thread-count assumptions.
- Dispatch effect: selected backend, tile/packing path, split policy, and any
  environment-variable override.
- Evidence chain: correctness command, focused benchmark command, full-sweep CSV,
  isolation level, comparator libraries, runtime environment, and summary.
- Boundary notes: nearby shapes that were tested but excluded, known unstable
  points, and the condition for disabling or narrowing the rule.

A gate can be narrower than the evidence, but it should not be broader. If the
evidence is focused, in-process, or missing comparator isolation, keep the rule
opt-in or label it experimental.

## Shape Classes

At minimum, reason about:

- Small square matrices, where fixed cost dominates.
- Medium square matrices, where packing and kernel startup both matter.
- Large square matrices, where throughput dominates.
- Tall/narrow and short/wide matrices, where task splitting can repeat packing.
- High-K matrices, where B packing and cache behavior dominate.
- Complex GEMM, where real-kernel decomposition can multiply workspace and
  dispatch overhead.

A rule that improves one class can easily hurt another.

## Packing

Packing is useful only when the saved kernel work exceeds the packing cost.
Rules:

- Write packed buffers contiguously.
- Keep small-stack pack limits explicit.
- Avoid repeated B packing across row splits unless the shape justifies it.
- Measure tail handling separately from the main tile path.

## Threading

Thread count policy is part of dispatch, not kernel code. Good defaults must
avoid over-threading small problems and avoid persistent worker interference in
long sweeps.

Runtime switches relevant to GEMM evidence:

- `ZYNUM_BLAS_NUM_THREADS`: user thread-count override. Record the value for
  every benchmark, including single-thread runs.
- `ZYNUM_BLAS_GEMM_POOL`: worker-pool experiment. Keep it disabled for reportable
  comparator sweeps unless the experiment is explicitly about the pool and uses
  fresh-process isolation.
- `ZYNUM_BLAS_GEMM_IO`: experimental GEMM IO/splitting behavior. Non-default
  values must be listed in the gate record and should stay opt-in until full
  sweeps are stable.

Threading changes should include single-thread and pinned multi-thread evidence.
Do not infer comparator fairness from default thread settings.

## Alpha And Beta

The `alpha=1,beta=0` path is often worth a dedicated store-only fast path.
General alpha/beta handling must remain correct and should be benchmarked
separately because it changes write-back cost.

## Complex GEMM

The current complex paths can use real GEMM transformations. This is useful for
coverage, but it can repeat packing, allocate more workspace, and trigger
multiple real GEMM dispatches.

Long-term complex GEMM should use dedicated packing and micro-kernels where the
target architecture justifies the maintenance cost.

## Retained Policy

Retain an optimization when:

- It passes correctness tests for the relevant target.
- It improves the target shape class repeatedly in focused runs and does not
  broadly regress other shape classes in a full sweep.
- Comparator claims, if any, are backed by fresh-process data.
- It has commands, raw CSV paths, summaries, runtime environment, and isolation
  level recorded.
- The rule is expressed by capabilities and shapes, not a single machine name.
- The default gate is narrower than the measured evidence and has a rollback
  condition.

Reject or keep opt-in when:

- It only wins a single point.
- It relies on long-lived state that pollutes later comparator measurements.
- It makes nearby shapes unstable.
- It has only cross-compiled or emulator data for a throughput claim.
- It requires ABI or public API layers to know kernel details.

Do not describe an optimization as faster than a comparator unless the retained
record includes the comparator library, version or path, thread policy, and
fresh-process CSV.
