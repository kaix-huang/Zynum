#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Fresh-process scheduling shared by structured Level 3 report runners."""

import re

SCHEDULE_CHOICES = ("library-major", "interleaved")


def normalized_library_label(value):
    """Return the full-report semantic label without changing valid input."""

    name = value.strip()
    folded = re.sub(r"[^a-z0-9]+", "", name.lower())
    if folded in {"zynum", "zynumblas", "libzynum", "libzynumblas"}:
        return "Zynum"
    return name


def validate_unique_library_labels(libraries):
    """Reject labels that would identify the same report library."""

    seen = {}
    for label, _path in libraries:
        semantic_label = normalized_library_label(label)
        previous = seen.get(semantic_label)
        if previous is not None:
            raise ValueError(
                f"duplicate semantic library label {label!r}: "
                f"collides with {previous!r} as {semantic_label!r}"
            )
        seen[semantic_label] = label


def validate_schedule(library_count, process_repeats, schedule):
    if schedule not in SCHEDULE_CHOICES:
        raise ValueError(f"unknown report schedule: {schedule}")
    if library_count < 1:
        raise ValueError("report schedule requires at least one library")
    if process_repeats < 1:
        raise ValueError("process repeats must be at least 1")
    if schedule == "interleaved" and process_repeats % library_count != 0:
        raise ValueError(
            "interleaved schedule requires --process-repeats to be a multiple "
            f"of the {library_count} selected libraries"
        )


def repeat_library_order(library_count, repeat_index, case_index=0):
    """Return one cyclic Latin rotation for a case/repeat pair."""

    first = (repeat_index + case_index) % library_count
    return tuple(
        (first + position) % library_count for position in range(library_count)
    )


def library_repeat_schedule(
    library_count,
    process_repeats,
    schedule,
    case_count=1,
):
    """Return the canonical library/case/repeat execution schedule."""

    validate_schedule(library_count, process_repeats, schedule)
    if case_count < 1:
        raise ValueError("report schedule requires at least one case")
    if schedule == "library-major":
        return [
            (library_index, case_index, repeat_index)
            for library_index in range(library_count)
            for case_index in range(case_count)
            for repeat_index in range(process_repeats)
        ]
    return [
        (library_index, case_index, repeat_index)
        for repeat_index in range(process_repeats)
        for case_index in range(case_count)
        for library_index in repeat_library_order(
            library_count, repeat_index, case_index
        )
    ]


def collect_repeats(
    selected,
    cases,
    process_repeats,
    schedule,
    run_one,
    announce,
):
    """Return samples in canonical library-major/case-major bucket order.

    ``interleaved`` rotates the first library for every case and process repeat.
    This balances warmup, frequency, and node drift while preserving one fresh
    process for every library/case/repeat sample.
    """

    library_count = len(selected)
    buckets = [[[] for _ in cases] for _ in selected]
    if not cases:
        validate_schedule(library_count, process_repeats, schedule)
        return buckets
    for library_index, case_index, repeat_index in library_repeat_schedule(
        library_count,
        process_repeats,
        schedule,
        case_count=len(cases),
    ):
        if schedule == "interleaved":
            announce(library_index, case_index, repeat_index)
        elif repeat_index == 0:
            announce(library_index, case_index, None)
        buckets[library_index][case_index].append(
            run_one(library_index, case_index, repeat_index)
        )
    return buckets
