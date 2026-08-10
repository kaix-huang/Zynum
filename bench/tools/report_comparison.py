#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Statistics and evidence validation shared by fresh-process reports."""

import math

_MAX_AXIS_INTERVALS = 100


def parse_positive_finite(value, field):
    """Parse one performance value and reject unusable numeric evidence."""

    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return parsed


def positive_finite_ratio(candidate, comparator):
    """Return a ratio only when both inputs and the result are usable."""

    numerator = parse_positive_finite(candidate, "candidate")
    denominator = parse_positive_finite(comparator, "comparator")
    return parse_positive_finite(numerator / denominator, "ratio")


def positive_finite_axis_ticks(max_value, padding=1.0, count=5):
    """Return bounded finite axis ticks covering positive finite evidence."""

    evidence = parse_positive_finite(max_value, "axis maximum")
    scale = parse_positive_finite(padding, "axis padding")
    requested_count = parse_positive_finite(count, "axis tick count")

    padded = evidence * scale
    if not math.isfinite(padded):
        return [0.0, evidence]
    target = max(evidence, padded)
    raw = target / requested_count
    if not math.isfinite(raw) or raw <= 0:
        return [0.0, evidence]

    exponent = math.floor(math.log10(raw))
    magnitude = 10.0**exponent
    if not math.isfinite(magnitude) or magnitude <= 0:
        return [0.0, evidence]
    base = raw / magnitude
    if base <= 1:
        factor = 1.0
    elif base <= 2:
        factor = 2.0
    elif base <= 5:
        factor = 5.0
    else:
        factor = 10.0
    step = factor * magnitude
    if not math.isfinite(step) or step <= 0:
        return [0.0, evidence]

    intervals = math.ceil(target / step)
    if not 1 <= intervals <= _MAX_AXIS_INTERVALS:
        return [0.0, evidence]
    top = intervals * step
    if not math.isfinite(top) or top < evidence:
        return [0.0, evidence]

    ticks = [0.0]
    value = 0.0
    for _ in range(intervals):
        value += step
        if not math.isfinite(value) or value <= ticks[-1]:
            return [0.0, evidence]
        ticks.append(value)
    ticks[-1] = max(ticks[-1], top)
    if ticks[-1] < evidence:
        return [0.0, evidence]
    return ticks


def positive_finite_median(values, field="median"):
    """Return a median without overflowing the even-sample average."""

    ordered = sorted(
        parse_positive_finite(value, f"{field} sample[{index}]")
        for index, value in enumerate(values)
    )
    if not ordered:
        raise ValueError(f"{field} requires at least one sample")
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        result = ordered[middle]
    else:
        lower = ordered[middle - 1]
        upper = ordered[middle]
        result = lower + (upper - lower) / 2
    return parse_positive_finite(result, field)


def nearest_rank_percentile(values, percentile, field):
    """Return a nearest-rank percentile from positive finite samples."""

    try:
        rank_percent = float(percentile)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} percentile must be numeric") from exc
    if not math.isfinite(rank_percent) or not 0 < rank_percent <= 100:
        raise ValueError(f"{field} percentile must satisfy 0 < percentile <= 100")
    ordered = sorted(
        parse_positive_finite(value, f"{field} sample[{index}]")
        for index, value in enumerate(values)
    )
    if not ordered:
        raise ValueError(f"{field} requires at least one sample")
    rank = math.ceil(len(ordered) * rank_percent / 100)
    return ordered[rank - 1]


def validate_performance_fields(row, required=(), optional=()):
    """Validate raw performance fields before a runner aggregates them.

    Required fields must be present. Optional fields may be absent, but when a
    producer supplies them they carry the same finite, strictly-positive
    evidence contract.
    """

    for field in required:
        value = row.get(field)
        if value in (None, ""):
            raise ValueError(f"{field} is required")
        parse_positive_finite(value, field)
    for field in optional:
        value = row.get(field)
        if value not in (None, ""):
            parse_positive_finite(value, field)


def best_higher_row(rows, field):
    """Select the highest metric with a library-label tie break."""

    return min(
        rows,
        key=lambda row: (
            -parse_positive_finite(row[field], field),
            str(row.get("library", "")),
        ),
    )


def best_lower_row(rows, field):
    """Select the lowest metric with a library-label tie break."""

    return min(
        rows,
        key=lambda row: (
            parse_positive_finite(row[field], field),
            str(row.get("library", "")),
        ),
    )


def metric_samples(row):
    raw = row.get("metric_samples")
    if raw in (None, ""):
        raise ValueError("paired-median requires metric_samples")
    samples = [
        parse_positive_finite(value, f"metric_samples[{index}]")
        for index, value in enumerate(raw.split(","))
    ]
    if not samples:
        raise ValueError("metric_samples must contain positive values")
    return samples


def validate_optional_metric_evidence(row):
    """Validate aggregate/sample columns whenever a report supplies them."""

    for field in ("metric_min", "metric_median", "metric_max"):
        if row.get(field) not in (None, ""):
            parse_positive_finite(row[field], field)
    if row.get("metric_samples") not in (None, ""):
        metric_samples(row)


def paired_median_ratio(candidate, comparator):
    candidate_samples = metric_samples(candidate)
    comparator_samples = metric_samples(comparator)
    if len(candidate_samples) != len(comparator_samples):
        raise ValueError("paired metric_samples lengths differ")
    ratios = [
        positive_finite_ratio(candidate_value, comparator_value)
        for candidate_value, comparator_value in zip(
            candidate_samples, comparator_samples
        )
    ]
    return positive_finite_median(ratios, "paired median ratio")
