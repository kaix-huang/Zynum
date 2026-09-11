#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Render a local comparison with the README's existing chart renderers.

The pure renderers are shared; the POSIX publication entry points are not
called. PNG rendering requires resvg-py. No missing/failed value becomes zero.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics

import plot_level1_report as level1
import plot_level2_report as level2
import plot_gemm_sweep as gemm

ORDER = ["Zynum", "MKL", "OpenBLAS", "BLIS"]
COLORS = {"Zynum": "#2563eb", "MKL": "#7c3aed", "OpenBLAS": "#16a34a", "BLIS": "#f59e0b"}


def validate_raw(records, raw_groups):
    """Reconstruct aggregates, including failure completeness, before plotting."""
    identities = {(r["family"], r["case"], r["library"]) for r in records}
    if set(raw_groups) != identities:
        raise ValueError("raw identities differ from declared records")
    for r in records:
        key = (r["family"], r["case"], r["library"])
        entries = sorted(raw_groups[key], key=lambda x: x["repeat"])
        if [e["repeat"] for e in entries] != list(range(1, r["process_repeats"] + 1)):
            raise ValueError(f"missing or duplicate raw repeat: {key}")
        values = []
        for e in entries:
            raw = e["row"]
            try:
                value = float(raw[r["metric"]])
                if (raw.get("status", "ok") == "ok"
                        and raw.get("check_status", raw.get("check")) in {"checked-ok", "sampled-ok"}
                        and math.isfinite(value) and value > 0):
                    values.append(value)
            except (KeyError, TypeError, ValueError):
                pass
        close = lambda a, b: math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
        if (r["successful_repeats"] != len(values) or len(r["samples"]) != len(values)
                or not all(close(a, b) for a, b in zip(values, r["samples"]))):
            raise ValueError(f"raw samples disagree with aggregate: {key}")
        if len(values) != r["process_repeats"]:
            if r["status"] != "failed" or any(r[k] is not None for k in ("min", "median", "max")):
                raise ValueError(f"invalid failure aggregate: {key}")
            continue
        if not eligible(r):
            raise ValueError(f"complete samples have invalid aggregate: {key}")
        expected = (min(values), statistics.median(values), max(values))
        if r["family"] == "gemm":
            times = [float(e["row"]["median_ns"]) for e in entries]
            if not all(math.isfinite(t) and t > 0 for t in times):
                raise ValueError(f"invalid GEMM latency: {key}")
            p = json.loads(r["parameters"])
            work = (8 if p["kind"][0] in "cz" else 2) * int(p["m"]) * int(p["n"]) * int(p["k"])
            if (times != r["process_median_ns_samples"]
                    or not close(statistics.median(times), r["median_ns"])
                    or not all(close(work / t, v) for t, v in zip(times, values))):
                raise ValueError(f"GEMM timing/rate mismatch: {key}")
            expected = (work / max(times), work / statistics.median(times), work / min(times))
        if not all(close(r[k], v) for k, v in zip(("min", "median", "max"), expected)):
            raise ValueError(f"aggregate statistics disagree with raw samples: {key}")


def eligible(row):
    return (row.get("status") == "ok"
            and row.get("successful_repeats") == row.get("process_repeats")
            and row.get("process_repeats", 0) >= 3
            and row.get("check_status") == "checked"
            and all(isinstance(row.get(k), (float, int))
                    and math.isfinite(row[k]) and row[k] > 0
                    for k in ("min", "median", "max")))


def annotate(svg, caption):
    text = svg.decode("utf-8")
    match = re.search(r'height="(\d+)" viewBox="0 0 (\d+) (\d+)"', text)
    if not match:
        raise ValueError("unexpected renderer SVG dimensions")
    height = int(match[1])
    text = text[:match.start()] + f'height="{height+40}" viewBox="0 0 {match[2]} {height+40}"' + text[match.end():]
    footer = f'<text x="40" y="{height+24}" style="font:13px Segoe UI,sans-serif;fill:#475569">{html.escape(caption)}</text>'
    return text.replace("</svg>", footer + "\n</svg>").encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--cpu", default="Core Ultra 9 290HX Plus")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--zynum-build", default="auto", help="Build/dispatch description for chart provenance")
    args = parser.parse_args(argv)
    data = args.input_dir
    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("profile") != "readme":
        raise ValueError("this plotter supports only the README profile; full data require family-specific plots")
    records = json.loads((data / "records.json").read_text(encoding="utf-8"))
    if any(r["family"] not in {"level1", "level2", "gemm"} for r in records):
        raise ValueError("unsupported family in README comparison")
    if not meta.get("run_completed") or not meta.get("coverage", {}).get("matches_plan"):
        raise ValueError("comparison did not complete its declared coverage")
    if not meta.get("artifacts_unchanged"):
        raise ValueError("comparison artifacts changed during timing")
    planned_bytes = (data / "planned_cases.json").read_bytes()
    if hashlib.sha256(planned_bytes).hexdigest() != meta["planned_cases_sha256"]:
        raise ValueError("case plan changed")
    planned = json.loads(planned_bytes)
    identity = lambda r: (r["family"], r["case"], r["library"])
    if {identity(r) for r in planned} != {identity(r) for r in records} or len(records) != len({identity(r) for r in records}):
        raise ValueError("record identities differ from the case plan")
    raw_groups = defaultdict(list)
    for line in (data / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        raw_groups[(r["family"], r["case"], r.get("library", r["row"].get("library")))].append(r)
    validate_raw(records, raw_groups)
    valid = [r for r in records if eligible(r)]
    expected_libraries = {r["library"] for r in planned}
    if {r["library"] for r in valid} != expected_libraries:
        raise ValueError("a requested library has no eligible data; inspect failures before plotting")
    libraries = [name for name in ORDER if name in expected_libraries]
    libraries += sorted(expected_libraries - set(libraries))
    for module in (level1, level2, gemm):
        module.LIB_ORDER = libraries
        module.COLORS.update(COLORS)
    plot_rows = defaultdict(list)
    shapes = {}
    for r in valid:
        samples = sorted(raw_groups[identity(r)], key=lambda x: x["repeat"])
        if [s["repeat"] for s in samples] != list(range(1, r["process_repeats"]+1)):
            raise ValueError(f"missing or duplicate repeat: {identity(r)}")
        if any(s["row"].get("status", "ok") != "ok" or s["row"].get("check_status", s["row"].get("check")) not in {"checked-ok", "sampled-ok"} for s in samples):
            raise ValueError(f"failed raw check in eligible aggregate: {identity(r)}")
        raw = samples[0]["row"]
        if r["family"] == "level1":
            plot_rows["level1"].append({
                "group": raw["group"], "op": raw["op"],
                "variant": raw.get("variant") or "default",
                "incx": int(raw.get("incx") or 1), "incy": int(raw.get("incy") or 1),
                "case": level1.row_case(raw), "label": level1.row_label(raw),
                "library": r["library"], "metric": raw["metric"], "value": r["median"],
                "n": int(raw["n"]), "copy_bytes": int(raw["copy_bytes"]) if raw.get("copy_bytes") else None,
                "seconds": int(raw["seconds"])})
        elif r["family"] == "level2":
            plot_rows["level2"].append({"case": raw["case"], "kind": raw["kind"],
                "library": r["library"], "n": int(raw["n"]), "rate_gops": r["median"]})
        elif r["family"] == "gemm":
            shape = (raw["label"], int(raw["m"]), int(raw["n"]), int(raw["k"]))
            index = shapes.setdefault(shape, len(shapes))
            plot_rows["gemm"].append({"kind": raw["kind"], "shape_index": index,
                "label": shape[0], "m": shape[1], "n": shape[2], "k": shape[3],
                "library": r["library"], "gflops": r["median"]})
    charts = data / "charts"
    charts.mkdir(exist_ok=True)
    repeats = meta["process_repeats"]
    threads = meta.get("threads", {}).get("MKL_NUM_THREADS", "24")
    platform_label = "WSL Linux" if "microsoft" in meta.get("platform", "").lower() else ("Linux" if meta.get("platform", "").startswith("Linux") else "Windows 11")
    caption = f"{platform_label} | {args.cpu} | {repeats} interleaved processes | Zynum {args.zynum_build}; comparator caps {threads} | Local checked comparison"
    renders = {"level1": level1.render_bars, "level2": level2.render_svg, "gemm": gemm.render_svg}
    images = []
    for family, render in renders.items():
        if not plot_rows[family]:
            continue
        svg = annotate(render(plot_rows[family], stat="median"), caption)
        (charts / f"{family}.svg").write_bytes(svg)
        if not args.no_png:
            import resvg_py
            (charts / f"{family}.png").write_bytes(resvg_py.svg_to_bytes(svg_string=svg.decode(), font_family="Segoe UI"))
        images.append(family)
    paired = defaultdict(dict)
    for r in valid:
        paired[r["family"], r["case"]][r["library"]] = r
    ratios = []
    for (family, case), group in paired.items():
        if "Zynum" not in group:
            continue
        for library in libraries:
            if library == "Zynum" or library not in group:
                continue
            ratios.append({"family": family, "case": case, "comparator": library,
                "zynum_over_comparator": group["Zynum"]["median"] / group[library]["median"]})
    summary = []
    for family in renders:
        for library in libraries[1:]:
            values = [r["zynum_over_comparator"] for r in ratios if r["family"] == family and r["comparator"] == library]
            if values:
                summary.append({"family": family, "comparator": library, "paired_cases": len(values),
                    "geomean_zynum_over_comparator": math.exp(sum(math.log(x) for x in values)/len(values)),
                    "zynum_faster_cases": sum(x > 1 for x in values)})
    (charts / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if ratios:
        with (charts / "ratios.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ratios[0]))
            writer.writeheader()
            writer.writerows(ratios)
    failed = [r for r in records if not eligible(r)]
    body = ["<!doctype html><meta charset='utf-8'><title>Zynum Windows BLAS comparison</title>",
        "<style>body{font:15px system-ui;max-width:1700px;margin:40px auto;padding:0 24px;color:#172033;background:#f8fafc}img{max-width:100%;background:white}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}a{color:#2563eb}</style>",
        f"<h1>Zynum vs MKL / OpenBLAS / BLIS — {platform_label}</h1>", f"<p>{html.escape(caption)}</p>",
        f"<p>{len(valid)} of {len(records)} library/case groups have valid checked performance. {len(failed)} failed or incomplete groups are omitted from bars, never replaced by zero.</p>",
        "<p>README chart renderers and measured shape scope. Higher is better. Rates use process medians; GEMM uses operation count divided by median process-median elapsed time. Raw samples retain dispersion. Library order is rotated across repeats.</p>",
        "<p>Local comparison with before/after artifact hashes; POSIX certification and report publication gates are separate.</p>"]
    body.append("<h2>Equal-case geometric mean ratios</h2><p>Zynum / comparator; above 1 means Zynum is faster. This is not an application workload score.</p><table><tr><th>Family</th><th>Comparator</th><th>Paired cases</th><th>Ratio</th><th>Zynum faster cases</th></tr>")
    for r in summary:
        body.append(f"<tr><td>{r['family']}</td><td>{r['comparator']}</td><td>{r['paired_cases']}</td><td>{r['geomean_zynum_over_comparator']:.3f}</td><td>{r['zynum_faster_cases']}</td></tr>")
    body.append("</table>")
    for family in images:
        body.append(f"<h2>{family.upper()}</h2><p><a href='{family}.svg'>SVG (zoomable)</a> · <a href='{family}.png'>PNG</a></p><img src='{family}.svg' alt='{family} comparison'>")
    if failed:
        body.append("<h2>Failed / missing groups</h2><pre>" + html.escape(json.dumps(failed, indent=2)) + "</pre>")
    body.append("<h2>Every measured group</h2><table><tr><th>Family</th><th>Case</th><th>Library</th><th>Median</th><th>Min–max</th><th>Unit</th><th>Status</th></tr>")
    for r in records:
        median = f"{r['median']:.6g}" if eligible(r) else "—"
        spread = f"{r['min']:.6g}–{r['max']:.6g}" if eligible(r) else "—"
        body.append("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in [r['family'], r['case'], r['library'], median, spread, r['unit'], r['status']]) + "</tr>")
    body.append("</table>")
    (charts / "index.html").write_text("\n".join(body), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(charts / "index.html")


if __name__ == "__main__":
    main()
