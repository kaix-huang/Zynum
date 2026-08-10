#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from report_schedule import (  # noqa: E402
    collect_repeats,
    library_repeat_schedule,
    normalized_library_label,
    repeat_library_order,
    validate_schedule,
    validate_unique_library_labels,
)


class ReportScheduleTests(unittest.TestCase):
    def test_library_label_validation_matches_full_report_zynum_aliases(self):
        for alias in ("Zynum", "zynum-blas", "libzynum", "libzynum-blas"):
            with self.subTest(alias=alias):
                self.assertEqual(normalized_library_label(alias), "Zynum")

        collisions = [
            [("Reference", "a"), (" Reference ", "b")],
            [("Zynum", "a"), ("zynum-blas", "b")],
            [("Zynum", "a"), ("libzynum", "b")],
            [("Zynum", "a"), ("lib-zynum-blas", "b")],
        ]
        for libraries in collisions:
            with self.subTest(libraries=libraries):
                with self.assertRaisesRegex(
                    ValueError, "duplicate semantic library label"
                ):
                    validate_unique_library_labels(libraries)

    def test_library_label_validation_preserves_valid_labels_paths_and_order(self):
        libraries = [
            ("Zynum", "first"),
            ("Reference", "second"),
            ("reference", "third"),
        ]
        original = list(libraries)
        self.assertIsNone(validate_unique_library_labels(libraries))
        self.assertEqual(libraries, original)

    def test_shared_schedule_has_exact_orders_for_one_two_and_three_libraries(self):
        self.assertEqual(
            library_repeat_schedule(1, 2, "interleaved", case_count=2),
            [
                (0, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 1, 1),
            ],
        )
        self.assertEqual(
            library_repeat_schedule(2, 2, "library-major", case_count=2),
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ],
        )
        self.assertEqual(
            library_repeat_schedule(3, 3, "interleaved"),
            [
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (1, 0, 1),
                (2, 0, 1),
                (0, 0, 1),
                (2, 0, 2),
                (0, 0, 2),
                (1, 0, 2),
            ],
        )

    def test_library_major_preserves_legacy_order(self):
        observed = []
        buckets = collect_repeats(
            ["a", "b"],
            ["x", "y"],
            2,
            "library-major",
            lambda library, case, repeat: (
                observed.append((library, case, repeat)) or (library, case, repeat)
            ),
            lambda *_: None,
        )
        self.assertEqual(
            observed,
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ],
        )
        self.assertEqual(buckets[1][0], [(1, 0, 0), (1, 0, 1)])

    def test_interleaved_rotates_first_library_and_keeps_buckets(self):
        observed = []
        buckets = collect_repeats(
            ["a", "b", "c"],
            ["x", "y"],
            3,
            "interleaved",
            lambda library, case, repeat: (
                observed.append((library, case, repeat)) or (library, case, repeat)
            ),
            lambda *_: None,
        )
        self.assertEqual(
            observed,
            [
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (1, 1, 0),
                (2, 1, 0),
                (0, 1, 0),
                (1, 0, 1),
                (2, 0, 1),
                (0, 0, 1),
                (2, 1, 1),
                (0, 1, 1),
                (1, 1, 1),
                (2, 0, 2),
                (0, 0, 2),
                (1, 0, 2),
                (0, 1, 2),
                (1, 1, 2),
                (2, 1, 2),
            ],
        )
        self.assertEqual(buckets[0][1], [(0, 1, 0), (0, 1, 1), (0, 1, 2)])

    def test_interleaved_exposes_every_library_once_in_every_position(self):
        library_count = 5
        position_counts = [
            [0 for _ in range(library_count)] for _ in range(library_count)
        ]
        for repeat in range(library_count * 2):
            for position, library in enumerate(
                repeat_library_order(library_count, repeat)
            ):
                position_counts[library][position] += 1
        self.assertEqual(
            position_counts,
            [[2 for _ in range(library_count)] for _ in range(library_count)],
        )

    def test_shared_interleaving_balances_each_case_by_actual_library_count(self):
        library_count = 3
        case_count = 2
        position_counts = [
            [[0 for _ in range(library_count)] for _ in range(library_count)]
            for _ in range(case_count)
        ]
        schedule = library_repeat_schedule(
            library_count,
            library_count,
            "interleaved",
            case_count=case_count,
        )
        for offset in range(0, len(schedule), library_count):
            group = schedule[offset : offset + library_count]
            case_index = group[0][1]
            for position, (library_index, _, _) in enumerate(group):
                position_counts[case_index][library_index][position] += 1
        self.assertEqual(
            position_counts,
            [
                [[1 for _ in range(library_count)] for _ in range(library_count)]
                for _ in range(case_count)
            ],
        )

    def test_interleaved_rejects_repeat_count_without_complete_rotations(self):
        with self.assertRaisesRegex(ValueError, "multiple of the 3 selected"):
            validate_schedule(3, 2, "interleaved")
        with self.assertRaisesRegex(ValueError, "multiple of the 3 selected"):
            collect_repeats(
                ["a", "b", "c"],
                ["x"],
                4,
                "interleaved",
                lambda *_: None,
                lambda *_: None,
            )
        for library_count, process_repeats in ((2, 3), (3, 4)):
            with self.subTest(
                library_count=library_count,
                process_repeats=process_repeats,
            ):
                with self.assertRaisesRegex(
                    ValueError, f"multiple of the {library_count} selected"
                ):
                    library_repeat_schedule(
                        library_count,
                        process_repeats,
                        "interleaved",
                    )

    def test_shared_schedule_rejects_empty_actual_dimensions(self):
        with self.assertRaisesRegex(ValueError, "at least one library"):
            library_repeat_schedule(0, 1, "library-major")
        with self.assertRaisesRegex(ValueError, "at least one case"):
            library_repeat_schedule(1, 1, "library-major", case_count=0)

    def test_collect_repeats_preserves_empty_case_compatibility(self):
        self.assertEqual(
            collect_repeats(
                ["a", "b"],
                [],
                2,
                "library-major",
                lambda *_: self.fail("empty cases must not run"),
                lambda *_: self.fail("empty cases must not announce"),
            ),
            [[], []],
        )


if __name__ == "__main__":
    unittest.main()
