#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import html
import math
from pathlib import Path

LIB_ORDER = ["Zynum", "Accelerate", "OpenBLAS", "MKL", "AOCL-BLIS"]
COLORS = {
    "Zynum": "#2563eb",
    "Accelerate": "#dc2626",
    "OpenBLAS": "#16a34a",
    "MKL": "#7c3aed",
    "AOCL-BLIS": "#ea580c",
}
CASE_ORDER = [
    "sgemv_n",
    "sgemv_t",
    "ssymv",
    "sger",
    "dgemv_n",
    "dgemv_t",
    "dsymv",
    "dger",
    "cgemv_n",
    "cgemv_t",
    "chemv",
    "cgeru",
    "cgerc",
    "zgemv_n",
    "zgemv_t",
    "zhemv",
    "zgeru",
    "zgerc",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Level 2 report CSV.")
    parser.add_argument("csv")
    parser.add_argument("--bars-svg", required=True)
    return parser.parse_args()


def nice_ticks(max_value, count=5):
    if max_value <= 0 or not math.isfinite(max_value):
        return [0, 1]
    raw = max_value / count
    exp = math.floor(math.log10(raw))
    base = raw / (10**exp)
    if base <= 1:
        step = 1
    elif base <= 2:
        step = 2
    elif base <= 5:
        step = 5
    else:
        step = 10
    step *= 10**exp
    top = math.ceil(max_value / step) * step
    ticks = []
    value = 0
    while value <= top + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def sy(value, top, height, max_value):
    if max_value <= 0:
        return top + height
    return top + height - value / max_value * height


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            if raw["status"] != "ok":
                continue
            rows.append(
                {
                    "case": raw["case"],
                    "kind": raw["kind"],
                    "library": raw["library"],
                    "n": int(raw["n"]),
                    "rate_gops": float(raw["rate_gops"]),
                }
            )
    return rows


def libraries_for(rows):
    seen = []
    for row in rows:
        if row["library"] not in seen:
            seen.append(row["library"])
    libs = [lib for lib in LIB_ORDER if lib in seen]
    libs.extend(lib for lib in seen if lib not in libs)
    return libs


def plot(rows, output_path):
    sizes = sorted({row["n"] for row in rows})
    libs = libraries_for(rows)
    values = {(row["n"], row["case"], row["library"]): row for row in rows}
    width = 2400
    left = 90
    right = 35
    panel_height = 270
    panel_gap = 102
    top0 = 118
    bottom = 110
    height = top0 + len(sizes) * panel_height + max(0, len(sizes) - 1) * panel_gap + bottom
    chart_width = width - left - right

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        """
<style>
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }
  .title { font-size: 26px; font-weight: 700; }
  .sub { font-size: 13px; fill: #4b5563; }
  .panel { font-size: 17px; font-weight: 700; }
  .tick { font-size: 11px; fill: #4b5563; }
  .op { font-size: 10px; fill: #374151; }
  .val { font-size: 8px; fill: #111827; }
  .legend { font-size: 14px; }
  .axis { stroke: #111827; stroke-width: 1; }
  .grid { stroke: #e5e7eb; stroke-width: 1; }
</style>
""",
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="38" y="42" class="title">Level 2 current performance - real and complex types</text>',
        '<text x="38" y="66" class="sub">Higher is better. Fresh process per library and size, metric = Gops. Panels use the CSV n values; bars follow the legend order.</text>',
    ]

    legend_spacing = 170
    legend_x = max(38, width - len(libs) * legend_spacing - 35)
    for index, lib in enumerate(libs):
        x = legend_x + index * legend_spacing
        color = COLORS.get(lib, "#6b7280")
        svg.append(f'<rect x="{x}" y="43" width="20" height="13" fill="{color}"/>')
        svg.append(f'<text x="{x + 28}" y="55" class="legend">{lib}</text>')

    for panel_index, n in enumerate(sizes):
        top = top0 + panel_index * (panel_height + panel_gap)
        panel_rows = [row for row in rows if row["n"] == n]
        max_value = max((row["rate_gops"] for row in panel_rows), default=1.0) * 1.12
        ticks = nice_ticks(max_value)
        max_value = ticks[-1]
        svg.append(f'<text x="{left}" y="{top - 23}" class="panel">n={n}</text>')
        for tick in ticks:
            y = sy(tick, top, panel_height, max_value)
            svg.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" class="grid"/>'
            )
            svg.append(
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick:g}</text>'
            )
        svg.append(
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" class="axis"/>'
        )
        svg.append(
            f'<line x1="{left}" y1="{top + panel_height}" x2="{left + chart_width}" y2="{top + panel_height}" class="axis"/>'
        )
        slot = chart_width / len(CASE_ORDER)
        bar_width = min(26, slot * 0.70 / max(1, len(libs)))
        for case_index, case in enumerate(CASE_ORDER):
            cx = left + slot * case_index + slot / 2
            start = cx - bar_width * len(libs) / 2
            for lib_index, lib in enumerate(libs):
                row = values.get((n, case, lib))
                if row is None:
                    continue
                value = row["rate_gops"]
                x = start + lib_index * bar_width
                y = sy(value, top, panel_height, max_value)
                h = top + panel_height - y
                color = COLORS.get(lib, "#6b7280")
                svg.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{h:.1f}" fill="{color}"><title>{html.escape(case)} {lib} n={n}: {value:.3f} Gops</title></rect>'
                )
                label_y = max(top + 9, y - 4)
                svg.append(
                    f'<text x="{x + (bar_width - 2) / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" class="val">{value:.1f}</text>'
                )
            svg.append(
                f'<text x="{cx:.1f}" y="{top + panel_height + 18}" text-anchor="end" transform="rotate(-28 {cx:.1f} {top + panel_height + 18})" class="op">{html.escape(case)}</text>'
            )

    svg.append("</svg>\n")
    Path(output_path).write_text("\n".join(svg))


def main():
    args = parse_args()
    plot(read_rows(args.csv), args.bars_svg)


if __name__ == "__main__":
    main()
