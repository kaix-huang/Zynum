#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import html
import math
import sys
from collections import defaultdict

COLORS = {
    "zynum-blas": "#2563eb",
    "Accelerate": "#dc2626",
    "OpenBLAS": "#16a34a",
    "MKL": "#7c3aed",
}

DISPLAY_NAMES = {
    "zynum-blas": "Zynum BLAS",
}


def display_name(value):
    return DISPLAY_NAMES.get(value, value)


def usage():
    print("usage: plot_gemm_sweep.py input.csv output.svg", file=sys.stderr)


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


def sx(index, shape_count, left, width):
    if shape_count <= 1:
        return left + width / 2
    return left + index * width / (shape_count - 1)


def sy(value, max_value, top, height):
    if max_value <= 0:
        return top + height
    return top + height - (value / max_value) * height


def polyline(points):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def plot_heading(kinds):
    kind_set = set(kinds)
    library_text = "Zynum BLAS vs Accelerate vs OpenBLAS"
    if kind_set == {"cgemm", "zgemm"}:
        return (
            f"Complex GEMM Performance Sweep: {library_text}",
            "Best-of-reps GF/s for CGEMM/ZGEMM across square, skinny, wide, and K-varied column-major shapes",
        )
    if kind_set == {"sgemm", "dgemm"}:
        return (
            f"Real GEMM Performance Sweep: {library_text}",
            "Best-of-reps GF/s for SGEMM/DGEMM across square, skinny, wide, and K-varied column-major shapes",
        )
    kind_text = "/".join(kind.upper() for kind in kinds)
    return (
        f"{kind_text} Performance Sweep: {library_text}",
        "Best-of-reps GF/s across square, skinny, wide, and K-varied column-major GEMM shapes",
    )


def draw_panel(
    kind, rows, labels, libs, panel_top, panel_height, chart_left, chart_width
):
    max_value = max((row["gflops"] for row in rows), default=1.0)
    ticks = nice_ticks(max_value)
    max_tick = ticks[-1] if ticks else max_value
    out = []
    out.append(
        f'<text x="{chart_left}" y="{panel_top - 18}" class="panel-title">{kind.upper()} GF/s</text>'
    )

    for tick in ticks:
        y = sy(tick, max_tick, panel_top, panel_height)
        out.append(
            f'<line x1="{chart_left}" y1="{y:.1f}" x2="{chart_left + chart_width}" y2="{y:.1f}" class="grid"/>'
        )
        out.append(
            f'<text x="{chart_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis-label">{tick:g}</text>'
        )

    out.append(
        f'<line x1="{chart_left}" y1="{panel_top}" x2="{chart_left}" y2="{panel_top + panel_height}" class="axis"/>'
    )
    out.append(
        f'<line x1="{chart_left}" y1="{panel_top + panel_height}" x2="{chart_left + chart_width}" y2="{panel_top + panel_height}" class="axis"/>'
    )

    by_lib = defaultdict(dict)
    for row in rows:
        by_lib[row["library"]][row["shape_index"]] = row["gflops"]

    for lib in libs:
        values = by_lib.get(lib, {})
        points = []
        for idx in range(len(labels)):
            if idx in values:
                points.append(
                    (
                        sx(idx, len(labels), chart_left, chart_width),
                        sy(values[idx], max_tick, panel_top, panel_height),
                    )
                )
        if len(points) >= 2:
            color = COLORS.get(lib, "#111827")
            out.append(
                f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>'
            )
            for x, y in points:
                out.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="{color}"/>'
                )
    return out


def main():
    if len(sys.argv) != 3:
        usage()
        return 2
    csv_path, svg_path = sys.argv[1:]
    rows = []
    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f):
            rows.append(
                {
                    "kind": raw["kind"],
                    "shape_index": int(raw["shape_index"]),
                    "label": raw["label"],
                    "m": int(raw["m"]),
                    "n": int(raw["n"]),
                    "k": int(raw["k"]),
                    "library": raw["library"],
                    "gflops": float(raw["gflops"]),
                }
            )

    labels_by_index = {}
    libs = []
    for row in rows:
        label = f"{row['label']}\\n{row['m']}x{row['n']}x{row['k']}"
        old_label = labels_by_index.get(row["shape_index"])
        if old_label is not None and old_label != label:
            raise ValueError(
                f"shape_index {row['shape_index']} maps to both {old_label!r} and {label!r}"
            )
        labels_by_index[row["shape_index"]] = label
        if row["library"] not in libs:
            libs.append(row["library"])
    sorted_indices = sorted(labels_by_index)
    expected_indices = list(range(len(sorted_indices)))
    if sorted_indices != expected_indices:
        raise ValueError(
            f"shape_index values must be contiguous from 0; got {sorted_indices}"
        )
    labels = [labels_by_index[i] for i in sorted_indices]

    kinds = []
    for row in rows:
        if row["kind"] not in kinds:
            kinds.append(row["kind"])

    width = 1800
    panel_height = 260
    panel_gap = 110
    top0 = 110
    height = top0 + len(kinds) * panel_height + max(0, len(kinds) - 1) * panel_gap + 160
    chart_left = 110
    chart_width = width - 170
    bottom_axis = (
        top0 + max(0, len(kinds) - 1) * (panel_height + panel_gap) + panel_height
    )

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append(
        """
<style>
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }
  .title { font-size: 24px; font-weight: 700; }
  .subtitle { font-size: 13px; fill: #4b5563; }
  .panel-title { font-size: 16px; font-weight: 700; }
  .axis-label { font-size: 11px; fill: #4b5563; }
  .shape-label { font-size: 10px; fill: #374151; }
  .axis { stroke: #111827; stroke-width: 1; }
  .grid { stroke: #e5e7eb; stroke-width: 1; }
  .legend { font-size: 13px; }
</style>
"""
    )
    title, subtitle = plot_heading(kinds)
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="40" y="42" class="title">{html.escape(title)}</text>')
    svg.append(f'<text x="40" y="66" class="subtitle">{html.escape(subtitle)}</text>')

    legend_x = 1180
    legend_y = 38
    for i, lib in enumerate(libs):
        x = legend_x + i * 145
        color = COLORS.get(lib, "#111827")
        svg.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 32}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{x + 40}" y="{legend_y + 4}" class="legend">{html.escape(display_name(str(lib)))}</text>'
        )

    for panel_index, kind in enumerate(kinds):
        top = top0 + panel_index * (panel_height + panel_gap)
        kind_rows = [row for row in rows if row["kind"] == kind]
        svg.extend(
            draw_panel(
                kind,
                kind_rows,
                labels,
                libs,
                top,
                panel_height,
                chart_left,
                chart_width,
            )
        )

    for idx, label in enumerate(labels):
        x = sx(idx, len(labels), chart_left, chart_width)
        first, second = label.split("\\n")
        svg.append(f'<g transform="translate({x:.1f},{bottom_axis + 18}) rotate(58)">')
        svg.append(f'<text class="shape-label">{html.escape(first)}</text>')
        svg.append(f'<text y="12" class="shape-label">{html.escape(second)}</text>')
        svg.append("</g>")

    svg.append("</svg>")
    with open(svg_path, "w") as f:
        f.write("\n".join(svg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
