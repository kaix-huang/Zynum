# CPU Affinity And Heterogeneous Scheduling Notes

This note records what Zynum can and cannot assume about CPU placement when
tuning BLAS task splits on heterogeneous machines.

## Apple Silicon / macOS

Public macOS APIs do not provide Linux-style "pin this thread to CPU N" control
for Apple Silicon. The Mach `THREAD_AFFINITY_POLICY` interface is an
experimental affinity-tag hint. Apple's public XNU header describes matching
tags as a request for threads to share an L2 cache if possible; it is not a CPU
ID, P-core, E-core, or cluster selector.

On the local Apple M5 machine used for the 2026 tuning pass, the public affinity
interface is not usable even as an L2 hint:

```text
thread_policy_set THREAD_AFFINITY_POLICY tag=1 kr=46
task_info TASK_AFFINITY_TAG_INFO kr=0 count=4 set_count=0 task_count=0 min=0 max=0
thread_policy_get THREAD_AFFINITY_POLICY kr=46 count=1 default=0 tag=0
```

`KERN_NOT_SUPPORTED` is value 46 in the macOS SDK, and
`TASK_AFFINITY_TAG_INFO` reports zero available affinity sets. A runtime path
must therefore treat Mach affinity tags as unavailable on this M5 unless a
fresh probe on the measured OS reports otherwise.

The available local topology information is still useful as a capacity hint:

```text
hw.ncpu: 10
hw.logicalcpu: 10
hw.physicalcpu: 10
hw.perflevel0.logicalcpu: 4
hw.perflevel1.logicalcpu: 6
hw.perflevel0.l2cachesize: 16777216
hw.cachelinesize: 128
```

Zynum may use `hw.perflevel0.logicalcpu`, `hw.perflevel1.logicalcpu`,
`hw.perflevel0.l2cachesize`, and `hw.cachelinesize` as shape-policy inputs, as
`src/blas/runtime.zig` already does. It must not infer that helper ordinal 0..3
will stay on the four performance cores.

The main public scheduling knob left on macOS is thread QoS. The current runtime
sets worker threads to `QOS_CLASS_USER_INITIATED` with
`pthread_set_qos_class_self_np`. QoS is a scheduler priority/quality request,
not an affinity contract. Raising workers to `QOS_CLASS_USER_INTERACTIVE` should
only be tried as an explicit experiment with benchmark evidence; it is not a
library default to promote without wider latency and system-impact checks.

That QoS A/B was tested for the remaining c64 GER512 gap and removed. Raising
Zynum worker threads to `QOS_CLASS_USER_INTERACTIVE` passed target correctness,
but the focused fresh-process report
`zig-out/perf-report/level2_qos_interactive_n512_probe_20260707.csv` measured
`zgeru/zgerc c64 n=512` at 74.235/73.693 Gops, below the same-day baseline
74.457/74.125 Gops in
`zig-out/perf-report/level2_baseline_n512_before_next_20260707.csv`.
Sampling (`/tmp/zynum_zgeru512_qos_interactive_sample.txt`) still put useful
time in `runComplexGerTaskC64 -> vector.operations.axpy`, with `runLowLatency`
and `__ulock_wait2` secondary. The result does not support promoting
`USER_INTERACTIVE` as a default scheduling hint on this M5; the runtime remains
at `QOS_CLASS_USER_INITIATED`.

SME/SM state does not change the affinity conclusion. SM/ZA/SIMD state belongs
to the executing thread and is preserved by the OS as required, but a BLAS task
split still cannot assume that a helper keeps the same core, cluster, or P/E
class across wakeups. When a candidate depends on streaming mode, its evidence
must distinguish kernel-body cost from `smstart`/`smstop`, ABI save/restore, and
wait/migration effects.

## Linux / x86_64

Linux provides real CPU affinity masks through `sched_setaffinity` and
`pthread_setaffinity_np`. The current Zynum runtime only pins persistent helper
threads, and only inside the CPU set already allowed by the scheduler. It does
not pin the caller thread and does not choose CPUs outside the job's cpuset.

For H3C and other Linux hosts, topology-aware experiments should first record:

- the process affinity mask from `sched_getaffinity`;
- per-CPU topology from `/sys/devices/system/cpu/cpu*/topology/`;
- package, core, SMT sibling, NUMA, and job scheduler cpuset boundaries;
- comparator thread-count and dynamic-threading settings for MKL/OpenBLAS.

Use `sched_getcpu` or equivalent tracing to validate where tasks actually ran
when a result depends on placement. Do not infer that worker ordinal maps to a
specific socket, core, or SMT sibling unless the affinity mask and trace both
show it.

## Split-Design Consequences

Apple M5 split design should treat the 4P/6E topology as a planning signal, not
as a binding mechanism. Static weighted task arrays such as "four larger P-core
tasks plus smaller E-core tasks" are brittle without usable affinity. They can
win in one run if the scheduler happens to place helpers favorably, then regress
when helper wake order or core migration changes.

Retain or reject heterogeneous split policies only with diagnostics that show
the mechanism:

- per-task start/end timing and participant identity;
- samples or traces separating task-body work from `runLowLatency` wait/wake
  overhead;
- migration or current-CPU observations when the platform exposes them;
- disassembly when the result depends on SM/ZA/SIMD state transitions or a
  specific load/store schedule;
- repeated fresh-process reports for both default-thread and relevant
  single-thread/thread-cap runs.

On macOS, useful next experiments are QoS A/B runs, per-task timing, and
Instruments/System Trace or `sample` captures around outliers. Mach affinity
tagging should stay behind a probe and is currently expected to be disabled on
this M5.

On Linux/x86_64, useful next experiments are cpuset-aware helper ordering,
NUMA/socket-aware grouping, and SMT-aware masks. Those are valid implementation
directions for the later H3C pass, but performance runs must follow the remote
job-submission rules and keep comparator thread counts pinned.

## Primary References

- Apple XNU `thread_policy.h`:
  https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/thread_policy.h
- Apple libpthread `pthread/qos.h`:
  https://github.com/apple-oss-distributions/libpthread/blob/main/include/pthread/qos.h
- Linux `sched_setaffinity(2)`:
  https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html
- Linux `sched_getcpu(3)`:
  https://man7.org/linux/man-pages/man3/sched_getcpu.3.html
- Linux CPU topology sysfs:
  https://docs.kernel.org/admin-guide/cputopology.html
