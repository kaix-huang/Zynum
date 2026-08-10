#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

import math
import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from report_comparison import (  # noqa: E402
    best_higher_row,
    best_lower_row,
    nearest_rank_percentile,
    paired_median_ratio,
    parse_positive_finite,
    positive_finite_axis_ticks,
    positive_finite_median,
    positive_finite_ratio,
    validate_performance_fields,
    validate_optional_metric_evidence,
)


class ReportComparisonTests(unittest.TestCase):
    def assert_finite_axis(self, ticks, evidence):
        self.assertLessEqual(len(ticks), 101)
        self.assertEqual(ticks[0], 0.0)
        self.assertGreater(ticks[-1], 0.0)
        self.assertGreaterEqual(ticks[-1], evidence)
        self.assertTrue(all(math.isfinite(tick) for tick in ticks))
        self.assertTrue(all(left < right for left, right in zip(ticks, ticks[1:])))

    def test_axis_ticks_preserve_ordinary_nice_scale(self):
        self.assertEqual(
            positive_finite_axis_ticks(10.25, padding=1.08),
            [0.0, 5.0, 10.0, 15.0],
        )

    def test_axis_ticks_bound_every_float_extreme(self):
        minimum_subnormal = math.ulp(0.0)
        maximum_finite = sys.float_info.max
        for label, evidence, padding in (
            ("padding overflow", maximum_finite, 2.0),
            ("rounded top overflow", maximum_finite, 1.0),
            ("raw count underflow", minimum_subnormal, 1.0),
        ):
            with self.subTest(label=label):
                ticks = positive_finite_axis_ticks(evidence, padding=padding)
                self.assert_finite_axis(ticks, evidence)
                self.assertEqual(ticks, [0.0, evidence])

    def test_axis_ticks_reject_invalid_maximum_padding_and_count(self):
        for arguments in (
            (0, 1, 5),
            (float("inf"), 1, 5),
            (1, 0, 5),
            (1, float("inf"), 5),
            (1, 1, 0),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    positive_finite_axis_ticks(*arguments)

    def test_nearest_rank_percentile_sorts_and_observes_rank_limits(self):
        values = [40, 10, 30, 20]
        self.assertEqual(nearest_rank_percentile(values, 0.001, "latency"), 10)
        self.assertEqual(nearest_rank_percentile(values, 25, "latency"), 10)
        self.assertEqual(nearest_rank_percentile(values, 50, "latency"), 20)
        self.assertEqual(nearest_rank_percentile(values, 50.001, "latency"), 30)
        self.assertEqual(nearest_rank_percentile(values, 100, "latency"), 40)

    def test_nearest_rank_percentile_requires_positive_finite_evidence(self):
        for value in (0, -1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "latency sample"):
                    nearest_rank_percentile([1, value], 95, "latency")
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            nearest_rank_percentile([], 95, "latency")

    def test_nearest_rank_percentile_requires_a_finite_bounded_percentile(self):
        for percentile in (0, -1, 100.001, "nan", "inf", "not-a-number"):
            with self.subTest(percentile=percentile):
                with self.assertRaisesRegex(ValueError, "latency percentile"):
                    nearest_rank_percentile([1], percentile, "latency")

    def test_positive_finite_parser_rejects_every_unusable_numeric_class(self):
        self.assertEqual(parse_positive_finite("1.25", "metric"), 1.25)
        for value in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    parse_positive_finite(value, "metric")

    def test_ratio_rejects_overflow_and_underflow(self):
        with self.assertRaisesRegex(ValueError, "ratio"):
            positive_finite_ratio(1e308, 1e-308)
        with self.assertRaisesRegex(ValueError, "ratio"):
            positive_finite_ratio(1e-308, 1e308)

    def test_median_is_safe_at_finite_float_extremes(self):
        for value in (1e308, 5e-324):
            with self.subTest(value=value):
                self.assertEqual(
                    positive_finite_median([value, value], "metric_median"),
                    value,
                )
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            positive_finite_median([], "metric_median")

    def test_paired_median_uses_repeat_ratios_instead_of_independent_medians(self):
        candidate = {"metric_samples": "1,10,1"}
        comparator = {"metric_samples": "2,5,2"}
        self.assertEqual(paired_median_ratio(candidate, comparator), 0.5)

    def test_paired_median_rejects_missing_mismatched_and_nonpositive_samples(self):
        with self.assertRaisesRegex(ValueError, "requires metric_samples"):
            paired_median_ratio({}, {"metric_samples": "1"})
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            paired_median_ratio({"metric_samples": "1,2"}, {"metric_samples": "1"})
        with self.assertRaisesRegex(ValueError, "positive"):
            paired_median_ratio({"metric_samples": "1,0"}, {"metric_samples": "1,2"})

    def test_paired_median_rejects_nonfinite_samples_ratios_and_final_median(self):
        for sample in ("nan", "inf", "-inf"):
            with self.subTest(sample=sample):
                with self.assertRaises(ValueError):
                    paired_median_ratio(
                        {"metric_samples": f"1,{sample}"},
                        {"metric_samples": "1,1"},
                    )
        with self.assertRaisesRegex(ValueError, "ratio"):
            paired_median_ratio(
                {"metric_samples": "1e308"},
                {"metric_samples": "1e-308"},
            )
        self.assertEqual(
            paired_median_ratio(
                {"metric_samples": "1e308,1e308"},
                {"metric_samples": "1,1"},
            ),
            1e308,
        )

    def test_optional_aggregate_and_samples_are_validated_when_present(self):
        validate_optional_metric_evidence(
            {
                "metric_min": "1",
                "metric_median": "2",
                "metric_max": "3",
                "metric_samples": "1,2,3",
            }
        )
        for field, value in (
            ("metric_min", "nan"),
            ("metric_median", "inf"),
            ("metric_max", "0"),
            ("metric_samples", "1,-1"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_optional_metric_evidence({field: value})

    def test_runner_fields_reject_missing_and_every_invalid_numeric_class(self):
        validate_performance_fields(
            {"required": "1", "optional": "2"},
            required=("required",),
            optional=("optional",),
        )
        with self.assertRaisesRegex(ValueError, "required is required"):
            validate_performance_fields({}, required=("required",))
        for value in ("0", "nan", "inf", "-inf", "1e999999", 10**10000):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_performance_fields(
                        {"required": value}, required=("required",)
                    )

    def test_best_row_ties_use_library_label_not_input_order(self):
        second = {"library": "Second", "metric": "1"}
        reference = {"library": "Reference", "metric": "1"}
        for rows in ((second, reference), (reference, second)):
            with self.subTest(rows=rows):
                self.assertIs(best_higher_row(rows, "metric"), reference)
                self.assertIs(best_lower_row(rows, "metric"), reference)


if __name__ == "__main__":
    unittest.main()
