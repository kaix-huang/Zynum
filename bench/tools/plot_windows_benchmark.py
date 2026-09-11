#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Render local Windows diagnostic measurements (requires matplotlib).

Each point represents the median of one complete checked fresh-process group.
Failed groups are listed in the HTML table but never plotted as performance.
These local files do not use the POSIX publication or comparison protocol.
"""

import argparse
from collections import defaultdict
import html
import hashlib
import json
import math
from pathlib import Path


def checked(record):
    return (
        record.get("status") == "ok"
        and record.get("check_status") == "checked"
        and record.get("successful_repeats") == record.get("process_repeats")
        and record.get("process_repeats", 0) >= 3
        and all(isinstance(record.get(k), (int, float))
                and math.isfinite(record[k]) and record[k] > 0
                for k in ("median", "min", "max"))
        and record["min"] <= record["median"] <= record["max"]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    records = json.loads((args.input_dir / "records.json").read_text(encoding="utf-8"))
    metadata = json.loads((args.input_dir / "metadata.json").read_text(encoding="utf-8"))
    if (not metadata.get("run_completed") or not metadata.get("coverage", {}).get("matches_plan")
            or not metadata.get("finished_utc") or not metadata.get("artifacts_unchanged")):
        raise ValueError("plotting requires complete planned coverage and unchanged artifacts")
    repeats = metadata["process_repeats"]
    keys = [(r["family"], r["case"], r["metric"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate case identities")
    planned_bytes = (args.input_dir / "planned_cases.json").read_bytes()
    if hashlib.sha256(planned_bytes).hexdigest() != metadata["planned_cases_sha256"]:
        raise ValueError("planned case file changed")
    planned = json.loads(planned_bytes)
    if {(r["family"], r["case"]) for r in planned} != {(r["family"], r["case"]) for r in records}:
        raise ValueError("record identities do not match the planned coverage")
    output = args.output_dir or args.input_dir / "charts"
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.titleweight": "bold", "savefig.facecolor": "#f8fafc"})
    valid = [r for r in records if checked(r)]
    charts = []

    def save(fig, name, title):
        fig.savefig(output / (name + ".png"), dpi=160, bbox_inches="tight")
        fig.savefig(output / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)
        charts.append((name, title))

    gemm = [r for r in valid if r["family"] == "gemm"]
    if gemm:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
        fig.suptitle("Zynum | Windows native GEMM", fontsize=21)
        dimensions = sorted({int(json.loads(r["parameters"])["n"]) for r in gemm})
        colors = {"NN": "#2563eb", "NT": "#14b8a6", "TN": "#f59e0b", "TT": "#a855f7"}
        for ax, kind in zip(axes.flat, ("sgemm", "dgemm", "cgemm", "zgemm")):
            for trans, color in colors.items():
                points = []
                for r in gemm:
                    p = json.loads(r["parameters"])
                    if r["operation"] == kind and p["transa"] + p["transb"] == trans:
                        points.append((int(p["n"]), r["median"], r["min"], r["max"]))
                points.sort()
                if points:
                    x, y, low, high = zip(*points)
                    ax.plot(x, y, "o-", color=color, label=trans, linewidth=1.8, markersize=4)
                    ax.fill_between(x, low, high, color=color, alpha=0.10)
            ax.set_title(kind.upper())
            ax.set_xscale("log", base=2)
            ax.set_xticks(dimensions, [str(n) for n in dimensions])
            ax.set_xlabel("Square dimension M = N = K")
            ax.set_ylabel("GFLOP/s (higher is better)")
            ax.grid(alpha=0.18)
            ax.legend(ncol=4, fontsize=8)
        fig.supxlabel(f"{repeats} fresh processes per case; line = median; shading = min–max. "
            "Per-process median timing; checked outputs; no comparator.", fontsize=9)
        save(fig, "gemm", "GEMM by scalar type and transpose")

    groups = defaultdict(list)
    for r in valid:
        if r["family"] != "gemm":
            groups[(r["family"], r["unit"])].append(r)
    for (family, unit), rows in groups.items():
        by_op = defaultdict(list)
        for r in rows:
            # Older local runner schema used the dtype in the Level 2 operation field.
            op = r["case"].split("/")[0] if family == "level2" else r["operation"]
            by_op[op].append(r)
        ops = sorted(by_op)
        fig, ax = plt.subplots(figsize=(12.5, max(4, len(ops) * 0.28 + 2.4)))
        palette = ["#2563eb", "#14b8a6", "#f59e0b", "#a855f7"]
        variants = sorted({str(json.loads(r["parameters"]).get("n", "all")) for r in rows})
        color_by = {v: palette[i % len(palette)] for i, v in enumerate(variants)}
        labeled = set()
        for i, op in enumerate(ops):
            for j, r in enumerate(by_op[op]):
                size = str(json.loads(r["parameters"]).get("n", "all"))
                color = color_by[size]
                # Deterministic vertical displacement only separates overlapping cases.
                offset = 0 if len(by_op[op]) == 1 else ((j % 7) - 3) * 0.045
                label = ("n=" + size) if size != "all" else "checked cases"
                ax.plot([r["min"], r["max"]], [i + offset, i + offset], color=color, alpha=0.22)
                ax.scatter(r["median"], i + offset, color=color, s=15, alpha=0.75,
                    label=label if label not in labeled else None)
                labeled.add(label)
        ax.set_yticks(range(len(ops)), ops)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.grid(axis="x", alpha=0.18)
        direction = "lower is better" if unit == "ns/call" else "higher is better"
        ax.set_xlabel(f"{unit} ({direction}); logarithmic scale")
        ax.set_title(f"Zynum | {family.replace('_', ' ').upper()} | Windows native", fontsize=17, pad=18)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
        fig.text(0.02, 0.012,
            f"Each dot = one checked case, median of {repeats} fresh processes. Thin line = process min–max.\n"
            "Multiple dots per routine represent different sizes / parameters; see the table for exact cases.",
            fontsize=8, color="#475569")
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        name = family + ("-bandwidth" if unit == "GB/s" else "-latency" if unit == "ns/call" else "-throughput")
        save(fig, name, f"{family} — {unit}")

    esc = html.escape
    body = ["<!doctype html><meta charset='utf-8'><title>Zynum Windows benchmark</title>",
        "<style>body{font:15px system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#172033;background:#f8fafc}"
        "img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;border-bottom:1px solid #dbe3ef;text-align:left}"
        "th{position:sticky;top:0;background:#e9eff9}.failed{color:#b91c1c}a{color:#2563eb}</style>",
        "<h1>Zynum Windows native benchmark</h1>",
        f"<p>{len(valid)} / {len(records)} checked complete cases. {len(records)-len(valid)} incomplete or failed groups are excluded from charts.</p>",
        f"<p>{esc(metadata['platform'])}; {esc(str(metadata['cpu_count']))} logical CPUs; "
        f"{esc(metadata['profile'])}. Automatic Zynum threads / ISA. No comparator; this is local diagnostic evidence.</p>",
        f"<p>Points show {repeats}-process medians. GEMM throughput uses per-process median elapsed time. "
        "Other metrics use the existing family probe definitions. Parameters and all raw samples remain in the run directory.</p>"]
    for name, title in charts:
        body += [f"<h2>{esc(title)}</h2><p><a href='{name}.svg'>SVG</a> · <a href='{name}.png'>PNG</a></p>",
                 f"<img src='{name}.png' alt='{esc(title)}'>"]
    body += ["<h2>All cases</h2><table><thead><tr><th>Family</th><th>Case</th><th>Unit</th>"
             "<th>Median</th><th>Min</th><th>Max</th><th>Status</th></tr></thead><tbody>"]
    for r in records:
        values = [r["family"], r["case"], r["unit"],
            *[f"{r[k]:.5g}" if checked(r) else "—" for k in ("median", "min", "max")], r["status"]]
        body.append(("<tr>" if checked(r) else "<tr class='failed'>") +
            "".join(f"<td>{esc(str(v))}</td>" for v in values) + "</tr>")
    body.append("</tbody></table>")
    (output / "index.html").write_text("\n".join(body), encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
