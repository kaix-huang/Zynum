#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path

COLORS = {
    "Zynum": "#2563eb",
    "Accelerate": "#dc2626",
    "OpenBLAS": "#16a34a",
    "MKL": "#7c3aed",
    "AOCL-BLIS": "#ea580c",
}

GROUP_TITLES = {
    "copy": "Copy bandwidth (GB/s)",
    "swap": "Swap bandwidth (GB/s)",
    "index": "Index reduction bandwidth (GB/s)",
    "real_f32": "Real f32 Level 1 (Gops)",
    "real_f64": "Real f64 Level 1 (Gops)",
    "mixed_dot": "Mixed-precision dot Level 1 (Gops)",
    "complex_f32": "Complex f32 Level 1 (Gops)",
    "complex_f64": "Complex f64 Level 1 (Gops)",
}

GROUP_ORDER = [
    "copy",
    "swap",
    "index",
    "real_f32",
    "real_f64",
    "mixed_dot",
    "complex_f32",
    "complex_f64",
]
LIB_ORDER = ["Zynum", "Accelerate", "OpenBLAS", "MKL", "AOCL-BLIS"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Level 1 report CSV as grouped bar SVGs.")
    parser.add_argument("csv")
    parser.add_argument("--bars-svg", required=True)
    parser.add_argument("--ratio-svg", required=True)
    return parser.parse_args()


def format_byte_size(value):
    units = (
        (1024 * 1024 * 1024, "GiB"),
        (1024 * 1024, "MiB"),
        (1024, "KiB"),
    )
    for factor, suffix in units:
        if value >= factor and value % factor == 0:
            return f"{value // factor}{suffix}"
    return f"{value}B"


def copy_case_key(raw):
    copy_bytes = int(raw.get("copy_bytes") or raw["n"])
    return f"{raw['op']}:{copy_bytes}"


def row_case(raw):
    if raw["group"] == "copy" and raw.get("copy_bytes"):
        return copy_case_key(raw)
    variant = raw.get("variant") or "default"
    incx = raw.get("incx") or "1"
    incy = raw.get("incy") or "1"
    return f"{raw['op']}:{variant}:{incx}:{incy}"


def row_label(raw):
    if raw["group"] == "copy" and raw.get("copy_bytes"):
        copy_bytes = int(raw.get("copy_bytes") or raw["n"])
        return f"{raw['op']} {format_byte_size(copy_bytes)}"
    variant = raw.get("variant") or "default"
    incx = raw.get("incx") or "1"
    incy = raw.get("incy") or "1"
    suffix = "" if variant == "default" else f" {variant}"
    if incx != "1" or incy != "1":
        suffix += f" ({incx},{incy})"
    return f"{raw['op']}{suffix}"


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            if raw["status"] != "ok":
                continue
            metric = raw["metric"]
            value_field = "bandwidth_gbps" if metric == "bandwidth_gbps" else "rate_gops"
            value = raw[value_field]
            if not value:
                continue
            rows.append(
                {
                    "group": raw["group"],
                    "op": raw["op"],
                    "variant": raw.get("variant") or "default",
                    "incx": int(raw.get("incx") or 1),
                    "incy": int(raw.get("incy") or 1),
                    "case": row_case(raw),
                    "label": row_label(raw),
                    "library": raw["library"],
                    "metric": metric,
                    "value": float(value),
                    "n": int(raw["n"]),
                    "copy_bytes": int(raw["copy_bytes"]) if raw.get("copy_bytes") else None,
                    "seconds": int(raw["seconds"]),
                }
            )
    return rows


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


def sy(value, max_value, top, height):
    if max_value <= 0:
        return top + height
    return top + height - value / max_value * height


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        """
<style>
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }
  .title { font-size: 25px; font-weight: 700; }
  .subtitle { font-size: 13px; fill: #4b5563; }
  .panel-title { font-size: 16px; font-weight: 700; }
  .axis-label { font-size: 11px; fill: #4b5563; }
  .op-label { font-size: 11px; fill: #374151; }
  .value-label { font-size: 10px; fill: #111827; }
  .legend { font-size: 13px; }
  .axis { stroke: #111827; stroke-width: 1; }
  .grid { stroke: #e5e7eb; stroke-width: 1; }
</style>
""",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]


def ordered_libraries(rows):
    seen = []
    for row in rows:
        if row["library"] not in seen:
            seen.append(row["library"])
    libs = [lib for lib in LIB_ORDER if lib in seen]
    libs.extend(lib for lib in seen if lib not in libs)
    return libs


def draw_legend(svg, x, y, libraries):
    for i, lib in enumerate(libraries):
        lx = x + i * 145
        color = COLORS.get(lib, "#6b7280")
        svg.append(f'<rect x="{lx}" y="{y - 12}" width="18" height="12" fill="{color}"/>')
        svg.append(f'<text x="{lx + 26}" y="{y - 2}" class="legend">{html.escape(lib)}</text>')


def grouped(rows):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    return by_group


def ops_for_group(rows):
    ops = []
    for row in rows:
        if row["case"] not in ops:
            ops.append(row["case"])
    return ops


def labels_for_group(rows):
    labels = {}
    for row in rows:
        labels.setdefault(row["case"], row["label"])
    return labels


def libraries_for_group(rows):
    libs = []
    for lib in LIB_ORDER:
        if any(row["library"] == lib for row in rows):
            libs.append(lib)
    for row in rows:
        if row["library"] not in libs:
            libs.append(row["library"])
    return libs


def value_map(rows):
    result = {}
    for row in rows:
        result[(row["case"], row["library"])] = row["value"]
    return result


def draw_group_panel(svg, rows, group, top, left, width, height):
    ops = ops_for_group(rows)
    labels = labels_for_group(rows)
    libs = libraries_for_group(rows)
    values = value_map(rows)
    max_value = max((row["value"] for row in rows), default=1.0) * 1.08
    ticks = nice_ticks(max_value)
    max_tick = ticks[-1]

    svg.append(f'<text x="{left}" y="{top - 20}" class="panel-title">{html.escape(GROUP_TITLES.get(group, group))}</text>')
    for tick in ticks:
        y = sy(tick, max_tick, top, height)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis-label">{tick:g}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis"/>')

    if not ops:
        return
    slot = width / len(ops)
    cluster_width = slot * 0.72
    bar_width = min(26, cluster_width / max(1, len(libs)))
    for op_index, op in enumerate(ops):
        cx = left + slot * op_index + slot / 2
        start = cx - bar_width * len(libs) / 2
        for lib_index, lib in enumerate(libs):
            value = values.get((op, lib))
            if value is None:
                continue
            x = start + lib_index * bar_width
            y = sy(value, max_tick, top, height)
            h = top + height - y
            color = COLORS.get(lib, "#6b7280")
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{h:.1f}" fill="{color}"/>')
        label = html.escape(labels.get(op, op))
        label_y = top + height + 18
        if group == "copy":
            svg.append(
                f'<text x="{cx:.1f}" y="{label_y}" text-anchor="start" class="op-label" '
                f'transform="rotate(45 {cx:.1f} {label_y})">{label}</text>'
            )
        else:
            svg.append(
                f'<text x="{cx:.1f}" y="{label_y}" text-anchor="middle" class="op-label">{label}</text>'
            )


def plot_bars(rows, output_path):
    by_group = grouped(rows)
    groups = [group for group in GROUP_ORDER if group in by_group]
    width = 1500
    panel_height = 240
    panel_gap = 95
    top0 = 120
    height = top0 + len(groups) * panel_height + max(0, len(groups) - 1) * panel_gap + 80
    left = 105
    chart_width = width - 155
    svg = svg_header(width, height)
    n_values = sorted({row["n"] for row in rows if row["group"] != "copy"})
    copy_sizes = sorted(
        {row["copy_bytes"] for row in rows if row["group"] == "copy" and row["copy_bytes"]}
    )
    seconds_values = sorted({row["seconds"] for row in rows})
    svg.append('<text x="40" y="42" class="title">Zynum Level 1 Performance Coverage</text>')
    n_part = ",".join(map(str, n_values)) if n_values else "copy-only"
    if len(copy_sizes) > 8:
        copy_part = f"{format_byte_size(copy_sizes[0])}..{format_byte_size(copy_sizes[-1])} ({len(copy_sizes)} sizes)"
    else:
        copy_part = ",".join(format_byte_size(size) for size in copy_sizes)
    svg.append(
        f'<text x="40" y="66" class="subtitle">Higher is better. Fresh process per library/op/size; n={n_part}; copy={copy_part}; seconds={",".join(map(str, seconds_values))}; grouped bars use operation names on the x-axis</text>'
    )
    legend_libraries = ordered_libraries(rows)
    legend_x = max(40, width - len(legend_libraries) * 145 - 35)
    draw_legend(svg, legend_x, 45, legend_libraries)
    for index, group in enumerate(groups):
        top = top0 + index * (panel_height + panel_gap)
        draw_group_panel(svg, by_group[group], group, top, left, chart_width, panel_height)
    svg.append("</svg>\n")
    Path(output_path).write_text("\n".join(svg))


def ratio_rows(rows):
    by_case = defaultdict(dict)
    group_by_case = {}
    metric_by_case = {}
    label_by_case = {}
    for row in rows:
        by_case[row["case"]][row["library"]] = row["value"]
        group_by_case[row["case"]] = row["group"]
        metric_by_case[row["case"]] = row["metric"]
        label_by_case[row["case"]] = row["label"]
    result = []
    ordered_cases = []
    for group in GROUP_ORDER:
        for row in rows:
            if row["group"] == group and row["case"] not in ordered_cases:
                ordered_cases.append(row["case"])
    for case in ordered_cases:
        values = by_case[case]
        zynum = values.get("Zynum")
        comparators = [value for lib, value in values.items() if lib != "Zynum"]
        if zynum is None or not comparators:
            continue
        best = max(comparators)
        result.append(
            {
                "op": label_by_case[case],
                "group": group_by_case[case],
                "metric": metric_by_case[case],
                "ratio": zynum / best if best > 0 else 0,
                "zynum": zynum,
                "best_comparator": best,
            }
        )
    return result


def plot_ratio(rows, output_path):
    ratios = ratio_rows(rows)
    width = max(1500, 54 * len(ratios) + 160)
    height = 520
    top = 110
    left = 95
    chart_width = width - 145
    chart_height = 270
    max_ratio = max([1.25] + [row["ratio"] for row in ratios]) * 1.05
    ticks = nice_ticks(max_ratio)
    max_tick = ticks[-1]
    svg = svg_header(width, height)
    svg.append('<text x="40" y="42" class="title">Zynum vs Fastest Comparator Ratio</text>')
    svg.append('<text x="40" y="66" class="subtitle">Higher is better. Ratio = Zynum metric / fastest non-Zynum comparator. Values below 1.0 indicate a measured slower operation.</text>')
    for tick in ticks:
        y = sy(tick, max_tick, top, chart_height)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis-label">{tick:g}</text>')
    one_y = sy(1.0, max_tick, top, chart_height)
    svg.append(f'<line x1="{left}" y1="{one_y:.1f}" x2="{left + chart_width}" y2="{one_y:.1f}" stroke="#111827" stroke-width="1.5" stroke-dasharray="5 4"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" class="axis"/>')

    if ratios:
        slot = chart_width / len(ratios)
        bar_width = min(30, slot * 0.62)
        for index, row in enumerate(ratios):
            cx = left + slot * index + slot / 2
            ratio = row["ratio"]
            y = sy(ratio, max_tick, top, chart_height)
            h = top + chart_height - y
            color = "#16a34a" if ratio >= 1.0 else "#dc2626"
            svg.append(f'<rect x="{cx - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{color}"/>')
            svg.append(
                f'<text x="{cx:.1f}" y="{top + chart_height + 18}" text-anchor="middle" class="op-label" transform="rotate(55 {cx:.1f} {top + chart_height + 18})">{html.escape(row["op"])}</text>'
            )
            svg.append(f'<text x="{cx:.1f}" y="{y - 4:.1f}" text-anchor="middle" class="value-label">{ratio:.2f}</text>')
    svg.append("</svg>\n")
    Path(output_path).write_text("\n".join(svg))


def main():
    args = parse_args()
    rows = read_rows(args.csv)
    plot_bars(rows, args.bars_svg)
    plot_ratio(rows, args.ratio_svg)


if __name__ == "__main__":
    main()
