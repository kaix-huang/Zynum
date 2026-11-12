#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conservative source/module boundary check; not numerical or release qualification."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
ROOTS = ("src/zynum.zig", "src/numerics/execution/root.zig", "src/numerics/resources.zig")
NAMED = {"std", "builtin", "root", "zynum-build-options", "zynum-core", "zynum-execution", "zynum-numerics-resources"}

def inspect(root: Path, overrides: dict[str, str] | None = None) -> dict:
    overrides = overrides or {}
    pending = list(ROOTS)
    seen = {}
    errors = []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = root / name
        source = overrides.get(name, path.read_text(encoding="utf-8"))
        seen[name] = hashlib.sha256(source.encode()).hexdigest()
        # Every literal source import is visited, including dormant reference
        # branches. Named Core owners are resolved once through ROOTS above.
        code = re.sub(r"(?m)//[^\n]*", "", source)
        if "@cImport" in code or "@cInclude" in code:
            errors.append(f"{name}: C translation in Core source closure")
        for imported in re.findall(r'@import\("([^"\n]+)"\)', code):
            if imported in NAMED:
                continue
            if not imported.endswith(".zig"):
                errors.append(f"{name}: unapproved named module {imported}")
                continue
            target = (path.parent / imported).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                errors.append(f"{name}: missing/outside source import {imported}")
                continue
            relative = target.relative_to(root).as_posix()
            if relative.startswith("interfaces/") or relative.startswith("integrations/"):
                errors.append(f"{name}: Core imports interface implementation {relative}")
            pending.append(relative)
    build = overrides.get("build.zig", (root / "build.zig").read_text(encoding="utf-8"))
    # This checks real import wiring, alongside the compiled no-SDK target. It
    # does not assert that arbitrary Zig code has a complete static resolver.
    neutral_owners = r"(?:core|execution|resources|execution_mod|execution_test_mod|numerics_resources_mod|host_numerics_resources_mod|runtime_lifecycle_execution_mod|runtime_lifecycle_resources_mod)"
    for owner, imported in re.findall(r'\b(' + neutral_owners + r')\.addImport\("([^"\n]+)"', build):
        if imported not in NAMED:
            errors.append(f"build.zig: {owner} imports non-Core module {imported}")
    return {"status": "failed" if errors else "passed", "source_closure": seen, "source_count": len(seen), "errors": errors,
            "limits": "conservative literal source closure plus explicit module edges; requires actual header-free compile evidence; full standalone distribution remains separate"}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = inspect(root)
    if args.self_test:
        probes = [("src/numerics/execution/context.zig", '\nconst leak = @import("heteris-provider-c");\n'),
                  ("src/numerics/resources.zig", '\nconst leak = @import("../../interfaces/production/adapters/resources/heteris/provider.zig");\n'),
                  ("build.zig", '\nexecution.addImport("heteris-provider-c", provider_c);\n')]
        result["negative_probes"] = []
        for name, addition in probes:
            observed = inspect(root, {name: (root/name).read_text() + addition})
            rejected = observed["status"] == "failed"
            result["negative_probes"].append({"source": name, "rejected": rejected})
            if not rejected:
                result["errors"].append("negative probe escaped: " + name)
        if result["errors"]:
            result["status"] = "failed"
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1
if __name__ == "__main__":
    raise SystemExit(main())
