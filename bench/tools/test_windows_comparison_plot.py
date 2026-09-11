# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from pathlib import Path
import tempfile
import unittest

import plot_windows_comparison as plot


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        row = dict(family="level1", case="sdot/default", library="Zynum", metric="rate_gops",
                   status="ok", process_repeats=4, successful_repeats=4,
                   samples=[1., 2., 4., 9.], min=1., median=3., max=9., check_status="checked")
        raw = {(row["family"], row["case"], row["library"]): [
            dict(repeat=i + 1, row=dict(rate_gops=v, check_status="sampled-ok"))
            for i, v in enumerate(row["samples"])]}
        return row, raw

    def test_valid_and_corrupt_median(self):
        row, raw = self.fixture()
        plot.validate_raw([row], raw)
        row["median"] = 4.
        with self.assertRaisesRegex(ValueError, "statistics"):
            plot.validate_raw([row], raw)

    def test_missing_repeat_in_failed_group(self):
        row, raw = self.fixture()
        row.update(status="failed", min=None, median=None, max=None)
        next(iter(raw.values())).pop()
        with self.assertRaisesRegex(ValueError, "repeat"):
            plot.validate_raw([row], raw)

    def test_nonfinite_raw_value(self):
        row, raw = self.fixture()
        next(iter(raw.values()))[0]["row"]["rate_gops"] = float("inf")
        with self.assertRaisesRegex(ValueError, "samples"):
            plot.validate_raw([row], raw)

    def test_unexpected_raw_identity(self):
        row, raw = self.fixture()
        raw[("level1", "unexpected", "Zynum")] = []
        with self.assertRaisesRegex(ValueError, "identities"):
            plot.validate_raw([row], raw)

    def test_gemm_even_repeat_latency_median(self):
        times = [1., 2., 4., 8.]
        values = [16 / t for t in times]
        row = dict(family="gemm", case="sgemm/NN/a", library="Zynum", metric="median_gflops",
                   status="ok", process_repeats=4, successful_repeats=4, samples=values,
                   min=2., median=16 / 3, max=16., check_status="checked", median_ns=3.,
                   process_median_ns_samples=times, parameters=json.dumps(dict(kind="sgemm", m=2, n=2, k=2)))
        raw = {("gemm", row["case"], "Zynum"): [dict(repeat=i + 1,
            row=dict(median_ns=t, median_gflops=v, check_status="checked-ok"))
            for i, (t, v) in enumerate(zip(times, values))]}
        plot.validate_raw([row], raw)
        row["median"] = 6.  # median of rates, which is deliberately different.
        with self.assertRaisesRegex(ValueError, "statistics"):
            plot.validate_raw([row], raw)

    def test_reject_full_profile_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "metadata.json").write_text('{"profile": "full"}')
            with self.assertRaisesRegex(ValueError, "README profile"):
                plot.main([directory, "--no-png"])


if __name__ == "__main__":
    unittest.main()
