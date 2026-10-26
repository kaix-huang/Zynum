# Generated numerical interfaces

Zynum implements numerical algorithms, storage, memory, execution and handles
internally. External profiles define declarations, binary layouts, argument
translation, workspace/query rules and observable error behavior. Generated
entry points forward to the configured implementation owner.

The [scientific capability map](../scientific.md) describes current scope.
[Detailed contracts](contracts.md) describe arguments and lifetimes;
[historical checkpoints](checkpoints.md) retain dated acceptance records.
Finite selections do not claim complete vendor compatibility.

## Ownership

The configured Core, Execution, Resources and options modules are shared
within one artifact. Adapters borrow an admitted Operation and ContextScratch
rather than creating another allocator or thread pool. The unified
`zynum_abplas_lp64` artifact composes selected interfaces over that owner.
Independently linked libraries retain independent globals; opaque objects and
allocation registries cannot be exchanged between them.

| Layer | Owner | Responsibility |
| --- | --- | --- |
| Mathematics | `src/lapack/core`, `src/fft/core.zig`, `src/sparse/core`, `src/numerics` | Native operands, algorithms and numerical outcomes |
| Managed execution | `src/numerics/execution/context.zig`, `src/numerics/resources.zig`, `execution` | Admission, captured allocators, leases and workers |
| Semantic conversion | `src/interop`, `src/checked/implementation.zig` | Wire validation, layout/index conversion, INFO and callbacks |
| Profile and renderer | `tools/*profile*`, `standard_*_bindings.py`, `interface_templates` | Finite declarations, forwarding and manifests |
| Export composition | `src/abplas_*.zig`, `src/lapacke_*_composed.zig`, `build.zig` | Bind shims to the existing implementation graph |

## Compile-time generation

The standalone entry point collects existing renderers while retaining their
profile checks and ownership:

```sh
python3 -B tools/generate_interfaces.py list
python3 -B tools/generate_interfaces.py check --zig zig
python3 -B tools/generate_interfaces.py check --family lapacke --family sparse
python3 -B tools/generate_interfaces.py generate fftw -- \
  --profile fftw3.f64.abplas-v1 --output /absolute/output/fftw
python3 -B tools/generate_interfaces.py generate fftw-f32 -- \
  --profile fftw3.f32.abplas-v1 --output /absolute/output/fftw-f32
```

`check` recomputes every committed accepted selection and rejects drift without
writing files. Optional f32 FFTW, operator, contour and FEAST outputs are
created by the selected build, rather than copied into every committed
combination. `generate` accepts the renderer's original options; direct renderer
commands remain valid. Register another renderer to add an external interface
while retaining the same Core and resource contracts.

Modify profile/schema/template inputs and regenerate; never hand-edit outputs.
LP64 and ILP64 are link-time choices. Headers, integer width and library must
agree; there is no mutable runtime ABI switch.

## Installation

The default installation remains BLAS. A combined scientific LP64 selection is:

```sh
zig build install-abplas-lp64 -Dcpu=baseline -Doptimize=ReleaseFast \
  -Dblas-interface=lp64 -Dfft-interface=fftw3 -Dfft-f32=true \
  -Dlapacke-qr=true -Dlapacke-complex-gelsd=true -Dlapacke-svd=true \
  -Dlapacke-posv=true -Dlapacke-trtrs=true -Dlapacke-trtri=true \
  -Dlapacke-getri=true -Dlapacke-potri=true -Dsparse-solves=true \
  -Dlapacke-controls=true -Dlapacke-syev=true -Dlapacke-sygv=true \
  -Dlapacke-evd=true -Dlapacke-gvd=true \
  -Dlapacke-gels=true -Dlapacke-gelss=true \
  -Doperator-interface=zynum_v1 -Doperator-phase-agreement=true \
  -Dcontour-interface=zynum_v1 -Dfeast-interface=feast4_lp64_abplas \
  -Dchecked-complex-cholesky=true --prefix /absolute/empty/prefix
```

Use `pkg-config zynum_abplas_lp64` from that prefix. `-Dsparse-solves=true`
adds the finite standard Sparse create/export/product/solve surface to the
combined artifact, alongside checked mixed-index products. Standalone steps
are `install-lapacke-lp64`, `install-lapacke-ilp64`, `install-sparse-lp64`,
`install-sparse-ilp64`, `install-checked-abi` and `install-checked-sparse`.
Use a fresh prefix when changing selections. Ordinary finite vendor layouts
are currently admitted for `apple-arm64-default-c-v1`.

## Verification boundaries

Declaration generation, compiler enumeration, numerical execution, installed
foreign calls and application adoption are separate checks. Test original
foreign declarations as well as generated headers; verify exports, residuals,
boundaries, lifecycle and the loaded image. See the
[contributor checks](../contributors/README.md).

Checked interfaces retain their own finite-input/status contract. LAPACKE
high and `_work` calls retain distinct NaN checks, workspace queries and INFO
conventions. GESVD exhaustion publishes an upper bidiagonal partial
factorization and unfinished edge count; finite/range faults remain separate.
Bitwise vendor failure payloads and complete IEEE equivalence are unqualified.
See the [contract reference](contracts.md).
