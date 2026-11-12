# Interval recurrence source migration

This B05 first slice moves the ownerless FP64 interval-scaled `series` and
`rawMoments` engine from `src/numerics/checked_spectral.zig` to
`src/numerics/spectral/interval_recurrence.zig`. Both operations retain one
mathematical Core and the original `Phase`, `Progress`, `Result` and function
bodies. Caller validation, admission, storage ownership and C wire decoding
remain the callers' responsibility. The engine itself imports only `std`;
arithmetic, trusted panels and synchronous callbacks are generic arguments.

The internal Zig source alias is now `internal.numerics.interval_recurrence`.
The narrow production adapter facade is now
`interfaces/production/public/zig/scientific_sources.zig`, exposing the
`interval_recurrence` alias through the already configured `zynum-core` module.
The native and Heteris spectral callers select that same engine. The old source
paths and `CheckedSpectral`/`checked_spectral` aliases have been removed in this
source revision; internal Zig consumers must migrate. No compatibility forwarder
or duplicate implementation is retained. Public C symbols and headers are unchanged.

The source contract and negative boundary tests bind the std-only engine and
same-Core aliases. Existing spectral tests cover wide recurrence arithmetic,
callback/progress failure order, zero-width/empty moments and stage-before-commit.
Atomic build/test inventory source refresh preserves existing native expectation
states and historical identities; it grants no new native qualification.

Full CPU integration and installed/relocated consumers are required before this
public source/build migration is accepted as a milestone. This slice does not
complete B05, header-free standalone distribution, other checked responsibilities,
legacy/adapted GEMM convergence, scientific, cluster, GPU or performance qualification.
