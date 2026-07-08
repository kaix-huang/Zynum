#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import csv
import html
import math
import sys
from collections import defaultdict

CHECKED_STATUSES = {"sampled-ok", "checked-ok"}

COLORS = {
    "Zynum": "#2563eb",
    "zynum-blas": "#2563eb",
    "Accelerate": "#dc2626",
    "OpenBLAS": "#16a34a",
    "MKL": "#7c3aed",
    "AOCL-BLIS": "#ea580c",
}

DISPLAY_NAMES = {
    "zynum-blas": "Zynum",
}

LIB_ORDER = ["Zynum", "zynum-blas", "Accelerate", "OpenBLAS", "MKL", "AOCL-BLIS"]


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


def library_heading(libs):
    names = [display_name(str(lib)) for lib in libs]
    if not names:
        return "GEMM Libraries"
    if len(names) == 1:
        return names[0]
    return " vs ".join(names)


def plot_heading(kinds, libs):
    kind_set = set(kinds)
    library_text = library_heading(libs)
    if kind_set == {"cgemm", "zgemm"}:
        return (
            f"Complex GEMM Performance Sweep: {library_text}",
            "Best-of-reps GF/s for CGEMM/ZGEMM across square, remainder, skinny, wide, and K-varied column-major shapes",
        )
    if kind_set == {"sgemm", "dgemm"}:
        return (
            f"Real GEMM Performance Sweep: {library_text}",
            "Best-of-reps GF/s for SGEMM/DGEMM across square, remainder, skinny, wide, and K-varied column-major shapes",
        )
    kind_text = "/".join(kind.upper() for kind in kinds)
    return (
        f"{kind_text} Performance Sweep: {library_text}",
        "Best-of-reps GF/s across square, remainder, skinny, wide, and K-varied column-major GEMM shapes",
    )


def shape_work(shape):
    return shape["m"] * shape["n"] * shape["k"]


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
        by_lib[row["library"]][row["plot_index"]] = row["gflops"]

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
            if raw.get("check") not in CHECKED_STATUSES:
                raise ValueError(
                    f"GEMM plots require correctness checks; row has check={raw.get('check')!r}"
                )
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

    shapes_by_index = {}
    seen_libs = []
    for row in rows:
        label = f"{row['label']}\\n{row['m']}x{row['n']}x{row['k']}"
        shape = {
            "shape_index": row["shape_index"],
            "label": label,
            "m": row["m"],
            "n": row["n"],
            "k": row["k"],
        }
        old_shape = shapes_by_index.get(row["shape_index"])
        if old_shape is not None and old_shape != shape:
            raise ValueError(
                f"shape_index {row['shape_index']} maps to both {old_shape!r} and {shape!r}"
            )
        shapes_by_index[row["shape_index"]] = shape
        if row["library"] not in seen_libs:
            seen_libs.append(row["library"])
    libs = [lib for lib in LIB_ORDER if lib in seen_libs]
    libs.extend(lib for lib in seen_libs if lib not in libs)
    ordered_shapes = sorted(
        shapes_by_index.values(),
        key=lambda shape: (shape_work(shape), max(shape["m"], shape["n"], shape["k"]), shape["shape_index"]),
    )
    plot_index_by_shape_index = {
        shape["shape_index"]: index for index, shape in enumerate(ordered_shapes)
    }
    for row in rows:
        row["plot_index"] = plot_index_by_shape_index[row["shape_index"]]
    labels = [shape["label"] for shape in ordered_shapes]

    kinds = []
    for row in rows:
        if row["kind"] not in kinds:
            kinds.append(row["kind"])

    width = max(1800, 90 * len(labels) + 170)
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
    title, subtitle = plot_heading(kinds, libs)
    subtitle = "Higher is better. Shapes are ordered by m*n*k so smaller cases stay at the front. " + subtitle
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
