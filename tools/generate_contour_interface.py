#!/usr/bin/env python3
"""Generate an optional neutral Contour extension over the existing operator owner."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def entry_symbols():
    return ["zynum_operator_contour_eigensolve" + ("_agreed" if agreed else "") + "_complex_" + precision
            for precision in ["f32", "f64"] for agreed in [False, True]]

def contour_phases():
    from generate_operator_interface import phase_stages
    phases = {name: value for name, value in phase_stages(maximum=56) if value > 49}
    if list(phases.values()) != list(range(50, 57)):
        raise ValueError("unsupported canonical contour phases")
    return phases

def validate_manifest(manifest):
    profile = json.loads((ROOT / "tools/contour_wire_profile.json").read_text())
    if (manifest.get("schema"), manifest.get("profile"), manifest.get("target"), manifest.get("pointer_bits")) != (
            "zynum-contour-interface-v1", "zynum-contour-v1", "apple-arm64-default-c-v1", 64):
        raise ValueError("unsupported contour interface contract")
    if manifest.get("layouts") != profile["layouts"] or manifest.get("phase_stages") != contour_phases():
        raise ValueError("invalid contour wire or phase layout")
    if manifest.get("symbols") != entry_symbols() or manifest.get("rational_result_bytes") != 72:
        raise ValueError("invalid contour public symbol or rational result contract")
    if manifest.get("base_operator_profile") != "zynum-v1" or manifest.get("phase_agreement") is not True:
        raise ValueError("contour requires the existing agreed operator-v1 profile")
    return manifest["symbols"]

def generate(output=None, with_entries=False, outputs=None):
    profile_path = ROOT / "tools/contour_wire_profile.json"
    template_path = ROOT / "tools/interface_templates/contour_wire_types.zig.in"
    profile = json.loads(profile_path.read_text())
    assert profile["profile"] == "zynum-contour-v1" and profile["abi_version"] == 1
    for key in ["options", "ritz", "summary"]:
        assert len({name for name, kind in profile[key]}) == len(profile[key])
    names = ["contour_wire_v1.h", "contour_v1_types.zig", "manifest.json"]
    if with_entries:
        names += ["contour_v1.h", "contour_v1.zig"]
    if outputs is None:
        output.mkdir(parents=True, exist_ok=True)
        outputs = {name: output / name for name in names}
    assert set(outputs) == set(names)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    zig_types = {"u32": "u32", "i32": "i32", "i64": "i64", "real": "R", "f64": "f64", "rational": "Rational", "ritz": "Ritz"}
    zig = template_path.read_text().replace("@ABI_VERSION@", str(profile["abi_version"]))
    for marker, key in [("OPTIONS", "options"), ("RITZ", "ritz"), ("SUMMARY", "summary")]:
        zig = zig.replace("@" + marker + "_FIELDS@", "\n".join(f"        {name}: {zig_types[kind]}," for name, kind in profile[key]))
    zig = zig.replace("@COUNT_CHECKS@", "\n".join(f"    if (o.{name} < 0) return error.InvalidDimension;" for name, kind in profile["options"] if kind == "i64"))
    assert "@OPTIONS_FIELDS@" not in zig and "@COUNT_CHECKS@" not in zig
    outputs["contour_v1_types.zig"].write_text(zig)
    header = ["/* Contour wire types; optional solver entry declarations are in contour_v1.h. */",
              "#ifndef ZYNUM_CONTOUR_WIRE_V1_H", "#define ZYNUM_CONTOUR_WIRE_V1_H", "#include <stdint.h>", "#include <stddef.h>",
              "#include <zynum/operator/operator_v1.h>",
              "#if defined(__cplusplus)", "#define ZOC_ASSERT(c,m) static_assert(c,m)", "#else", "#define ZOC_ASSERT(c,m) _Static_assert(c,m)", "#endif",
              "enum { ZYNUM_CONTOUR_WIRE_VERSION=1, ZYNUM_CONTOUR_RANK_RELATIVE_PRESENT=1, ZYNUM_CONTOUR_RITZ_RELATIVE_PRESENT=2, ZYNUM_CONTOUR_ALLOW_INEXACT_ITERATION_LIMIT=4, ZYNUM_CONTOUR_PUBLISH_FINITE_STOPS=8 };",
              "/* Optional tolerances are read only when their presence flag is set. */",
              "/* Numerical validation belongs to Core. Zero iteration budgets remain representable. */"]
    primitive = {"u32": "uint32_t", "i32": "int32_t", "i64": "int64_t", "f64": "double"}
    header.append("typedef struct { " + " ".join(primitive[k] + " " + n + ";" for n, k in profile["ritz"]) + " } zynum_operator_contour_ritz;")
    header.append('ZOC_ASSERT(sizeof(zynum_operator_contour_ritz)==32,"contour Ritz bytes");')
    for precision, real in [("f32", "float"), ("f64", "double")]:
        types = primitive | {"real": real, "rational": "zynum_operator_rational_complex_" + precision, "ritz": "zynum_operator_contour_ritz"}
        options = "zynum_operator_contour_options_" + precision
        summary = "zynum_operator_contour_summary_" + precision
        for name, key in [(options, "options"), (summary, "summary")]:
            header.append("typedef struct { " + " ".join(types[k] + " " + n + ";" for n, k in profile[key]) + " } " + name + ";")
        layout = profile["layouts"][precision]
        header += [f'ZOC_ASSERT(sizeof({options})=={layout["options_bytes"]},"contour options bytes");',
                   f'ZOC_ASSERT(sizeof({summary})=={layout["summary_bytes"]},"contour summary bytes");',
                   f'ZOC_ASSERT(offsetof({options},lower)=={layout["real_offset"]},"contour real offset");',
                   f'ZOC_ASSERT(offsetof({options},ritz_relative_tolerance)=={layout["ritz_relative_offset"]},"contour Ritz tolerance offset");',
                   f'ZOC_ASSERT(offsetof({summary},linear)=={layout["linear_offset"]},"contour linear offset");',
                   f'ZOC_ASSERT(offsetof({summary},ritz)=={layout["ritz_offset"]},"contour Ritz offset");']
    header += ["#undef ZOC_ASSERT", "#endif", ""]
    outputs["contour_wire_v1.h"].write_text("\n".join(header))
    if with_entries:
        interface = ["/* Generated neutral Contour extension using the existing operator-v1 owner. */",
                     "#ifndef ZYNUM_CONTOUR_V1_H", "#define ZYNUM_CONTOUR_V1_H", '#include "contour_wire_v1.h"',
                     "#define ZYNUM_OPERATOR_HAS_CONTOUR_EIGENSOLVER 1",
                     "/* All operator-v1 borrowing, callback and phase-agreement rules apply. */",
                     "/* Match the complete phase/index/control tuple after agreeing local failures. */",
                     "/* Numerical outputs are mutually disjoint; initial may overlap them. */",
                     "/* Summary must not overlap initial or outputs. Descriptors must not overlap outputs. */",
                     "/* Errors preserve vectors, eigenvalues, residuals and summary, including final agreement errors. */",
                     "/* OK commits summary: inspect summary.status and published. Controlled stops preserve numerical outputs */",
                     "/* unless PUBLISH_FINITE_STOPS explicitly permits finite candidates. Only published columns are written. */",
                     "/* Complete-space capacity is valid. Partial capacity and absent Ritz values do not prove completeness. */",
                     "/* Trace convergence alone does not certify true residuals. Padding and unused tails are retained. */"]
        for prefix, values in profile["enums"].items():
            interface.append("enum { " + ", ".join("ZYNUM_CONTOUR_" + prefix + "_" + name + "=" + str(value) for name, value in values.items()) + " };")
        interface.append("enum { " + ", ".join("ZYNUM_OPERATOR_PHASE_" + name.upper() + "=" + str(value) for name, value in contour_phases().items()) + " };")
        interface += ["#ifdef __cplusplus", 'extern "C" {', "#endif"]
        wrappers = ['// Generated forwarding only. Core owns algorithms, operation, scratch and runtime.',
                    'const Core=@import("zynum-core");', 'const Checked=@import("checked-interface-owner");',
                    'const B=@import("operator-interface-owner").Binding(Core,Checked);',
                    'const CW=@import("zynum-contour-wire");', 'const S=Core.internal.numerics.Scalar;', 'const W=Checked.Wire;']
        for precision, ztype in [("f32", "S.ComplexF32"), ("f64", "S.ComplexF64")]:
            for agreed in [False, True]:
                name = "zynum_operator_contour_eigensolve" + ("_agreed" if agreed else "") + "_complex_" + precision
                agreement_c = "zynum_operator_agreement agreement, " if agreed else ""
                interface.append(f"zynum_checked_result {name}(zynum_operator_action action, const zynum_operator_action* inverse_action, const zynum_operator_reduction_complex_{precision}* reduction, const zynum_operator_block_reduction* projection_reduction, {agreement_c}const zynum_operator_contour_options_{precision}* options, zynum_operator_const_panel initial, zynum_operator_panel vectors, zynum_checked_buffer eigenvalues, zynum_checked_buffer residuals, zynum_operator_contour_summary_{precision}* summary);")
                agreement_zig = "g:B.Agreement," if agreed else ""
                agreement_arg = "g," if agreed else ""
                method = "contourAgreed" if agreed else "contour"
                wrappers.append(f"pub export fn {name}(a:B.Action,p:?*const anyopaque,r:?*const anyopaque,b:?*const anyopaque,{agreement_zig}o:?*const anyopaque,x:B.ConstPanel,y:B.Panel,v:W.buffer,e:W.buffer,s:?*anyopaque) callconv(.c) W.result {{return B.{method}({ztype},CW,a,p,r,b,{agreement_arg}o,x,y,v,e,s);}}")
        interface += ["#ifdef __cplusplus", "}", "#endif", "#endif", ""]
        outputs["contour_v1.h"].write_text("\n".join(interface))
        outputs["contour_v1.zig"].write_text("\n".join(wrappers) + "\n")
    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    inputs = [profile_path, template_path, Path(__file__).resolve()]
    if with_entries:
        inputs += [ROOT / "tools/generate_operator_interface.py", ROOT / "src/numerics/synchronization.zig"]
    manifest = {"profile": profile["profile"], "status": "candidate_forwarding" if with_entries else "candidate_wire_only", "target": profile["target"],
                "layouts": profile["layouts"], "exported_solver_entries": entry_symbols() if with_entries else [], "native_binding_forwarders_generated": with_entries,
                "ABP_integrated": False, "generator_input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                "output_sha256": {name: sha(outputs[name]) for name in names if name != "manifest.json"}}
    if with_entries:
        manifest.update(schema="zynum-contour-interface-v1", pointer_bits=64, symbols=entry_symbols(), phase_stages=contour_phases(),
                        rational_result_bytes=72, base_operator_profile="zynum-v1", phase_agreement=True,
                        implementation_owner="Injected Core and existing checked/operator interface owner", installed_interface_qualified=False)
        validate_manifest(manifest)
    outputs["manifest.json"].write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-entries", action="store_true")
    for argument in ["wire-header", "types", "header", "exports", "manifest"]:
        parser.add_argument("--" + argument, type=Path)
    args = parser.parse_args()
    if args.output is not None:
        if any(getattr(args, name) is not None for name in ["wire_header", "types", "header", "exports", "manifest"]):
            parser.error("--output cannot be combined with explicit output paths")
        generate(args.output, args.with_entries)
    else:
        keys = ["wire_header", "types", "manifest"] + (["header", "exports"] if args.with_entries else [])
        if any(getattr(args, name) is None for name in keys):
            parser.error("all explicit output paths are required")
        names = ["contour_wire_v1.h", "contour_v1_types.zig", "manifest.json"] + (["contour_v1.h", "contour_v1.zig"] if args.with_entries else [])
        paths = dict(zip(names, [getattr(args, name) for name in keys]))
        generate(with_entries=args.with_entries, outputs=paths)
