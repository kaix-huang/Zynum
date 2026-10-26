# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compile-time interface entry point; renderers own declarations, Core owns math.

`check` verifies every committed selection without writing files. `generate`
forwards the selected renderer's original arguments, including its explicit
profile/target checks. Build-only components remain build-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
from generate_checked_abi import outputs as checked_outputs, subset_outputs
from checked_cholesky_profile import candidate as cholesky_candidate, PROFILE as CHOLESKY_PROFILE, TARGET as CHOLESKY_TARGET
from generate_checked_cholesky import render as cholesky_outputs
import checked_sparse_profile
from generate_checked_sparse import render_candidate as checked_sparse_outputs
import standard_abi_bindings
import standard_sparse_bindings
from generate_lapacke_component import outputs as component_outputs
from lapacke_components import COMPONENTS, PROFILES, TARGET
from generate_fftw import render as fftw_outputs

ROOT = Path(__file__).resolve().parents[1]
RENDERERS = {
    "lapacke": "generate_checked_abi.py",
    "lapacke-component": "generate_lapacke_component.py",
    "sparse": "generate_checked_abi.py",
    "checked": "generate_checked_abi.py",
    "checked-cholesky": "generate_checked_cholesky.py",
    "checked-sparse": "generate_checked_sparse.py",
    "fftw": "generate_fftw.py",
    "fftw-f32": "generate_fftw_f32.py",
    "operator": "generate_operator_interface.py",
    "contour": "generate_contour_interface.py",
    "feast": "generate_feast.py",
}


def committed_outputs(families, zig):
    """Reuse accepted selections; never infer callable support from filenames."""
    if "checked" in families:
        yield checked_outputs(zig)
        yield cholesky_outputs(cholesky_candidate(CHOLESKY_PROFILE, CHOLESKY_TARGET), zig)
        yield checked_sparse_outputs(checked_sparse_profile.candidate(
            checked_sparse_profile.PROFILE_ID, checked_sparse_profile.SET_ID,
            checked_sparse_profile.TARGET_ID), zig)
    for family, module in (("lapacke", standard_abi_bindings),
                           ("sparse", standard_sparse_bindings)):
        if family not in families:
            continue
        for selection in sorted(module.GENERATION_ACCEPTANCES):
            yield subset_outputs(*selection, zig=zig)
    if "lapacke" in families:
        for profile in PROFILES:
            for component in COMPONENTS:
                # Symmetric/driver components currently have an LP64 contract.
                if profile.endswith("ilp64") and component not in ("potri", "controls"):
                    continue
                folder = Path("interface_components") / profile / (component + "-v1")
                yield {str(folder / name): content for name, content in
                       component_outputs(profile, component, TARGET, zig).items()}
    if "fftw" in families:
        paths = [ROOT / "tools" / name for name in
                 ("generate_fftw.py", "fftw_profile.json", "interface_templates/fftw_exports.zig.in")]
        profile = json.loads(paths[1].read_text())
        generated = fftw_outputs(profile, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in paths}, paths[2].read_text())
        folder = Path("interface_subset/fftw3.f64/abplas-v1")
        yield {str(folder / name): content for name, content in generated.items()}


def check(families, zig):
    seen = {}
    selections = 0
    for generated in committed_outputs(families, zig):
        selections += 1
        for name, content in generated.items():
            if name in seen and seen[name] != content:
                raise ValueError("conflicting interface output: " + name)
            seen[name] = content
    drift = [name for name, content in seen.items()
             if not (ROOT / name).is_file() or (ROOT / name).read_text() != content]
    if drift:
        raise ValueError("generated interface drift: " + ", ".join(sorted(drift)))
    return {"selections": selections, "files": len(seen), "drift": 0,
            "runtime_qualification": "separate numerical, ABI and lifecycle checks"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list explicit renderer entry points")
    checking = sub.add_parser("check", help="check all committed generated interfaces")
    checking.add_argument("--family", choices=("lapacke", "sparse", "checked", "fftw"), action="append")
    checking.add_argument("--zig", default="zig")
    generation = sub.add_parser("generate", help="run one renderer with its original options")
    generation.add_argument("interface", choices=tuple(RENDERERS))
    generation.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(RENDERERS, indent=2))
    elif args.command == "check":
        try:
            print(json.dumps(check(set(args.family or ("lapacke", "sparse", "checked", "fftw")), args.zig)))
        except ValueError as error:
            parser.exit(1, str(error) + "\n")
    else:
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        renderer = ROOT / "tools" / RENDERERS[args.interface]
        previous = sys.argv
        try:
            sys.argv = [str(renderer), *arguments]
            runpy.run_path(str(renderer), run_name="__main__")
        finally:
            sys.argv = previous


if __name__ == "__main__":
    main()
