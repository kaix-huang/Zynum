# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate an optional single-precision FFTW companion over the shared Core.

The default f64 generator and its outputs remain independent. This component
uses that profile's shared FFTW integer/enum types and its reviewed forwarding
recipe. It generates no numerical implementation or resource owner.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import generate_fftw

PROFILE = "fftw3.f32.abplas-v1"
EXPECTED_FUNCTIONS_SHA256 = "134b42e31d98337940557aeb8677af15ea7f246970b3288368c78adf814d428b"
ABI = {
    "int_bytes": 4, "unsigned_bytes": 4, "pointer_bytes": 8,
    "size_t_bytes": 8, "float_bytes": 4, "complex_bytes": 8,
    "complex_alignment": 4, "plan_tag": "fftwf_plan_s",
    "kind_tag": "fftw_r2r_kind_do_not_use_me",
    "iodim_tag": "fftw_iodim_do_not_use_me",
    "iodim64_tag": "fftw_iodim64_do_not_use_me",
}


def _single(text):
    return re.sub(r"\bdouble\b", "float", text.replace("fftw_", "fftwf_"))


def render(profile, base, source_hashes, template, zig="zig"):
    if (profile["profile"] != PROFILE
            or profile["target"] != "apple-arm64-default-c-v1"
            or base["profile"] != generate_fftw.PROFILE
            or base["target"] != profile["target"]):
        raise ValueError("unsupported FFTW single-precision profile or target")
    if profile["requires_profiles"] != [generate_fftw.PROFILE] or profile["abi"] != ABI:
        raise ValueError("single-precision component must use the reviewed shared FFTW ABI")
    if tuple(row["symbol"] for row in base["functions"]) != generate_fftw.REVIEWED_SYMBOLS:
        raise ValueError("base function selection differs from the reviewed FFTW set")
    functions = profile["functions"]
    expected = [{key: _single(row[key]) for key in ("symbol", "returns", "parameters")}
                for row in base["functions"]]
    digest = hashlib.sha256(json.dumps(functions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if functions != expected or digest != EXPECTED_FUNCTIONS_SHA256:
        raise ValueError("single-precision signatures differ from original-header acceptance")
    if any(profile[key] != base[key] for key in ("r2r_kinds", "constants", "constant_types")):
        raise ValueError("FFTW transform kinds and uppercase constants are shared across precisions")
    binding = 'const B = @import("fftw-interface-owner");'
    if template.count(binding) != 1:
        raise ValueError("forwarding template must borrow the existing FFTW owner exactly once")
    exports = template.replace(binding, 'const B = @import("fftw-interface-owner").F32;')
    exports = re.sub(r"(?<=export fn )fftw_", "fftwf_", exports)
    symbols = [row["symbol"] for row in functions]
    if sorted(re.findall(r"export fn (fftwf_\w+)\(", exports)) != sorted(symbols):
        raise ValueError("forwarding symbols differ from the single-precision selection")
    formatted = subprocess.run([zig, "fmt", "--stdin"], input=exports.encode(), capture_output=True)
    if formatted.returncode != 0:
        raise ValueError("single-precision forward formatting failed before publication: " + formatted.stderr.decode())
    exports = formatted.stdout.decode()
    header = [
        "// Copyright (C) 2026 Zynum contributors",
        "// SPDX-License-Identifier: LGPL-3.0-or-later",
        "/* Generated finite single-precision FFTW companion declarations. */",
        "/* Include standalone, or AFTER the original fftw3.h if using both. */",
        "#ifndef ZYNUM_FFTW3_F32_ABPLAS_V1_H",
        "#define ZYNUM_FFTW3_F32_ABPLAS_V1_H",
        '#include "fftw3_subset.h"',
        "#ifndef FFTW3_H",
        "#if !defined(FFTW_NO_Complex) && defined(_Complex_I) && defined(complex) && defined(I)",
        "typedef float _Complex fftwf_complex;",
        "#else", "typedef float fftwf_complex[2];", "#endif",
        "typedef struct fftwf_plan_s *fftwf_plan;",
        "typedef struct fftw_iodim_do_not_use_me fftwf_iodim;",
        "typedef struct fftw_iodim64_do_not_use_me fftwf_iodim64;",
        "typedef enum fftw_r2r_kind_do_not_use_me fftwf_r2r_kind;",
        "#endif",
        "#ifdef __cplusplus",
        "#define ZYNUM_FFTWF_ASSERT static_assert",
        "#define ZYNUM_FFTWF_ALIGN alignof",
        "#else", "#define ZYNUM_FFTWF_ASSERT _Static_assert",
        "#define ZYNUM_FFTWF_ALIGN _Alignof", "#endif",
        'ZYNUM_FFTWF_ASSERT(sizeof(float) == 4 && sizeof(fftwf_complex) == 8 && ZYNUM_FFTWF_ALIGN(fftwf_complex) == 4, "FFTW float complex ABI");',
        'ZYNUM_FFTWF_ASSERT(sizeof(fftwf_plan) == 8, "FFTW float plan ABI");',
        'ZYNUM_FFTWF_ASSERT(sizeof(fftwf_r2r_kind) == 4 && ZYNUM_FFTWF_ALIGN(fftwf_r2r_kind) == 4, "FFTW shared kind ABI");',
        'ZYNUM_FFTWF_ASSERT(sizeof(fftwf_iodim) == 12 && ZYNUM_FFTWF_ALIGN(fftwf_iodim) == 4 && offsetof(fftwf_iodim, is) == 4 && offsetof(fftwf_iodim, os) == 8, "FFTW shared guru ABI");',
        'ZYNUM_FFTWF_ASSERT(sizeof(fftwf_iodim64) == 24 && ZYNUM_FFTWF_ALIGN(fftwf_iodim64) == 8 && offsetof(fftwf_iodim64, is) == 8 && offsetof(fftwf_iodim64, os) == 16, "FFTW shared guru64 ABI");',
        "#undef ZYNUM_FFTWF_ASSERT", "#undef ZYNUM_FFTWF_ALIGN",
        '#ifdef __cplusplus', 'extern "C" {', '#endif',
    ]
    header += [row["returns"] + " " + row["symbol"] + "(" + row["parameters"] + ");" for row in functions]
    header += ["#ifdef __cplusplus", "}", "#endif", "#endif"]
    manifest = {
        "schema": "zynum-external-interface-declarations-v1",
        "profile": PROFILE, "target": profile["target"],
        "requires_profiles": profile["requires_profiles"],
        "symbols": symbols, "source_hashes": source_hashes,
        "full_fftw_compatibility": False, "runtime_exported": False,
        "status": "declarations-only-external-runtime-pending",
        "header_include_order": "standalone companion or original fftw3.h first",
        "shared_base_header": "fftw3_subset.h",
        "internal_owner": "existing Zynum Core and its scalar FFTW binding; no owner generated",
    }
    return {
        "fftw3f_subset.h": "\n".join(header) + "\n",
        "exports.zig": exports,
        "exports.txt": "".join(name + "\n" for name in symbols),
        "abi_manifest.json": json.dumps(manifest, indent=2) + "\n",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=[PROFILE], default=PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--zig", default="zig")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve(), directory / "fftw_f32_profile.json",
             directory / "fftw_profile.json", directory / "generate_fftw.py",
             directory / "interface_templates/fftw_exports.zig.in"]
    profile = json.loads(paths[1].read_text())
    base = json.loads(paths[2].read_text())
    outputs = render(profile, base, {str(path.relative_to(directory.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, paths[4].read_text(), zig=args.zig)
    if args.check:
        stale = [name for name, content in outputs.items()
                 if not (args.output / name).is_file() or (args.output / name).read_text() != content]
        if stale:
            parser.exit(1, "generated interface drift: " + ", ".join(stale) + "\n")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            (args.output / name).write_text(content)


if __name__ == "__main__":
    main()
