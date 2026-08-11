# Contributor Guide

This layer is for people changing Zynum itself. It collects the public
development workflow without mixing in host-local runbooks or raw benchmark
journals.

## Required Checks

Repository validation fails closed when any inherited environment name starts
with `GIT_`. If a change intentionally updates generated files, stage those
expected outputs before running this block. The generator rerun below then
rejects only additional working-tree drift or new untracked outputs. Run the
required checks in this sanitized subprocess:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
  bash <<'ZYNUM_VALIDATION'
set -euo pipefail
zig fmt --check build.zig build.zig.zon src test bench examples tools
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile bench/tools/*.py tools/*.py
python3 -B tools/check_build_inventory.py --root .
python3 -B tools/check_test_inventory.py --root . --structure-only
zig build test-build-inventory --summary failures
zig build test-test-inventory --summary failures
zig build test-host-tool-smoke -Dcpu=baseline --summary failures
zig build -Dcpu=baseline -Dtest-optimize=Debug -Dhost-tool-smoke=false test --summary failures
zig build --release=safe -Dcpu=baseline -Dtest-optimize=ReleaseSafe -Dhost-tool-smoke=false test --summary failures
zig build --release=fast -Dcpu=baseline -Dtest-optimize=ReleaseFast -Dhost-tool-smoke=false test --summary failures
zig build generate-headers --summary failures
zig build generate-kernel-coverage --summary failures
if ! git --no-pager diff --exit-code -- include/zynum/blas docs/kernel_coverage.json; then
  echo "The generator produced additional unstaged changes."
  echo "Review them, then stage the intended generated outputs before retrying."
  exit 1
fi
untracked="$(git ls-files --others --exclude-standard -- include/zynum/blas docs/kernel_coverage.json)"
if [[ -n "$untracked" ]]; then
  echo "The generator produced untracked outputs. Review and stage them before retrying."
  printf '%s\n' "$untracked"
  exit 1
fi
zig build --summary failures
ZYNUM_VALIDATION
```

By default, `zig build test` runs the test-inventory gate, all applicable Zig
correctness tests, and the `test-host-tool-smoke` aggregate. Set
`-Dhost-tool-smoke=false` when that aggregate has already run; invoking
`zig build test-host-tool-smoke` explicitly always runs it. The aggregate
includes the inventory-declared Python tooling tests, ABI manifest, C/C++ header
smoke checks, the Fortran module smoke when `gfortran` is available, and ABI
baseline observer tests. It does not
include build-inventory security, test-inventory security, or ABI artifact parity.
Run the independent `test-build-inventory` and `test-test-inventory` steps
separately, as shown above.

The canonical Python tooling gate reports and rejects unexpected skips, expected
failures, and unexpected successes. An allowed skip is bound to exact
`skip_kind`, reviewed `predicate_id`, and `predicate_ast_sha256`, the lowercase
SHA-256 of canonical
`ast.dump(predicate_expr, annotate_fields=True, include_attributes=False,
show_empty=True)` (or the equivalent default on Python 3.12 and earlier), as
well as its test identity and literal reason. Changing `unittest.skipIf` to
`unittest.skipUnless`, changing the predicate, or substituting unconditional
`unittest.skip` fails closed.

When ABI exports or kernel registrations change, regenerate the corresponding
files and check that `include/zynum/blas/` and `docs/kernel_coverage.json` have
no unexpected drift.

## Change Paths

| Change type | Read first | Validate with |
| --- | --- | --- |
| Public Zig API | [`../development_and_usage.md`](../development_and_usage.md), [`../internals/README.md`](../internals/README.md) | API tests, docs, examples |
| BLAS ABI export | [`../fortran_compatibility.md`](../fortran_compatibility.md), [`../architecture.md`](../architecture.md) | ABI tests, generated headers |
| Core BLAS semantics | [`../architecture.md`](../architecture.md) | API and ABI tests |
| Kernel dispatch or tuning | [`../performance/README.md`](../performance/README.md), [`../common/benchmarking.md`](../common/benchmarking.md) | Correctness, focused probes, sweep evidence |
| Release prep | [`../open_source_release_checklist.md`](../open_source_release_checklist.md) | Full validation checklist |

## Repository Hygiene

- Keep public docs in English.
- Keep benchmark claims tied to commands, CSV paths, target details, comparator
  libraries, and thread policy.
- Keep host-local notes, profiler transcripts, raw CSVs, and private runbooks
  out of the repository.
- Keep local planning artifacts outside the repository; they are not package,
  CI, or release inputs.
- Do not add new `ZYNUM_*` environment variables without changing the documented
  runtime contract.
