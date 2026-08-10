# CPU Affinity And Heterogeneous Scheduling

This document defines portable rules for interpreting CPU topology, affinity,
and heterogeneous task placement while tuning BLAS paths. Topology is evidence
about available capacity; it is not permission to encode one machine's processor
numbering into dispatch.

## General Rules

- Derive usable concurrency from the CPU allocation visible to the process.
- Distinguish physical topology, allowed cpuset, task placement, and observed
  performance; they are different facts.
- Treat processor IDs as ephemeral identifiers, not performance classes.
- Measure task bodies, submission, waiting, and merge time separately before
  changing a split.
- Keep all algorithms correct when fewer helpers run than requested or when every
  task lands on the same capacity class.
- Use `ZYNUM_MAXIMUM_THREADS` only as a diagnostic cap. Default gates leave it
  unset and record the runtime-observed capacity.

## macOS On Heterogeneous AArch64

Public macOS APIs do not provide a general supported contract for pinning a
thread to an exact CPU or performance/efficiency core. The following are not
equivalent to CPU affinity:

- `hw.perflevel*` topology information;
- quality-of-service classes;
- Mach affinity tags; or
- observing one scheduling pattern in a trace.

Topology queries can estimate capacity classes and cache groupings. QoS can
express scheduling intent. Affinity tags may encourage related placement. None
of these proves an exact CPU, stable core class, or persistent cluster assignment.

Do not describe a result as pinned unless a supported API and runtime trace prove
the claim. Record the mechanism exactly, along with the process's runtime CPU
capacity and actual task durations.

A heterogeneous split may use capacity-weighted task sizes only after repeated
traces show the intended imbalance and boundary controls show a complete-call
gain. The path must remain correct under arbitrary placement and should fall back
to uniform work when runtime evidence is unavailable.

## Linux Affinity

Linux affinity masks are enforceable, but only inside the cpuset assigned to the
process. In containers and batch-scheduled environments, that set may be smaller
or differently numbered than the physical system.

Before an affinity experiment, record:

- the inherited process affinity mask;
- package, core, SMT-sibling, cache, and NUMA topology within that mask;
- the runtime CPU capacity reported to Zynum;
- comparator thread and affinity settings; and
- whether the caller and helpers are all covered by the policy.

A helper-pinning policy must choose only CPUs in the inherited mask. It must not
assume contiguous numbering, select CPUs outside the allocation, or leave the
caller competing unexpectedly with a pinned helper.

Affinity can diagnose migration, SMT contention, or NUMA placement. It becomes a
production policy only when it improves representative complete calls across
valid allocations and does not harm fairness or integration with the embedding
application.

## Designing Task Splits

For each candidate split, state:

- independent work and output ownership;
- minimum useful grain size;
- expected bandwidth or arithmetic balance;
- private workspace and merge cost;
- maximum task count;
- behavior after partial submission; and
- serial fallback.

Prefer disjoint output ranges. Reductions use bounded private partials and one
explicit merge. Dependency-ordered operations remain serial across dependency
steps even if work inside one step can use helpers.

Helper identity must not become part of numerical semantics. If a measured path
uses asymmetric shard sizes, encode a capacity ratio or task class in tuning,
not a processor ID or helper offset.

## Measurement Protocol

1. Run correctness with the default scheduler and with explicit one-thread and
   low-thread caps.
2. Record available cpuset or topology hints without changing placement.
3. Collect per-task start, stop, processed work, and caller/helper identity.
4. Repeat in fresh processes and inspect placement and duration distributions.
5. Apply one affinity, QoS, or weighting change at a time.
6. Compare the complete call, not only the fastest task.
7. Re-run off-gate shapes and comparator controls.

An apparent win caused by a favorable placement in one process is noise until it
survives repeated balanced-order measurements. A slower complete call with more
uniform task durations is still a regression.

## Retention And Rollback

Retain a topology-aware policy only when:

- the platform mechanism is supported and accurately described;
- it stays inside the execution allocation;
- all submission and fallback paths remain complete;
- native traces confirm the intended mechanism; and
- fresh-process complete-call results improve beyond variance.

Rollback when the policy depends on processor numbering, harms restricted
allocations, creates a second worker lifecycle, improves only capped diagnostics,
or loses its gain after balanced process ordering.

## Primary References

- [Apple Quality of Service classes](https://developer.apple.com/documentation/dispatch/dispatchqos)
- [Linux `sched_setaffinity(2)`](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html)
- [Linux cpuset control](https://docs.kernel.org/admin-guide/cgroup-v2.html)
- [hwloc documentation](https://www.open-mpi.org/projects/hwloc/)
