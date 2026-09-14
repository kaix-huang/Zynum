# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
import csv
import io
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_windows_benchmark as bench


class LocalBenchmarkTests(unittest.TestCase):
    def test_failed_repeat_never_produces_aggregate(self):
        good = {"status": "ok", "check_status": "sampled-ok", "rate": 5}
        bad = {**good, "check_status": "failed", "rate": 999}
        result = bench.summarize("test", "x", "rate", "Gop/s", [good, good, bad], 3)
        self.assertEqual(result["successful_repeats"], 2)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["median"])
        self.assertEqual(result["samples"], [5, 5])

    def test_nonfinite_and_unchecked_are_rejected(self):
        for value, check in [(float("nan"), "checked-ok"), (float("inf"), "checked-ok"),
                             (0, "checked-ok"), (100, "unchecked")]:
            row = {"metric": value, "check": check}
            result = bench.summarize("test", "x", "metric", "ns", [row] * 3, 3)
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["max"])

    def test_median_and_missing_repeat(self):
        rows = [{"metric": x, "check": "checked-ok"} for x in [8, 2, 5]]
        result = bench.summarize("test", "x", "metric", "ns", rows, 3)
        self.assertEqual((result["min"], result["median"], result["max"]), (2, 5, 8))
        self.assertEqual(bench.summarize("test", "x", "metric", "ns", rows[:2], 3)["status"], "failed")

    def test_quick_matrix_profiles_have_all_routines_and_distinct_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            args = bench.parse_args(["--output", str(Path(temp) / "run"), "--quick"])
            runner = bench.Runner(args)
            try:
                for family, module, probe in [("rank_k", bench.rank, "rank-k-probe"),
                    ("symm", bench.symm, "symm-probe"),
                    ("triangular", bench.triangular, "triangular-matrix-probe")]:
                    captured = []
                    with patch.object(runner, "group", side_effect=lambda f, n, *rest: captured.append(n)):
                        runner.matrix(family, module, probe)
                    self.assertEqual(len(captured), len(set(captured)))
                    self.assertEqual(len(captured), len(module.ROUTINES))
            finally:
                runner.raw.close()
                runner.log.close()

    def test_full_plan_is_independent_of_probe_execution(self):
        args = bench.parse_args(["--output", "unused"])
        with patch.object(bench.subprocess, "run", side_effect=AssertionError("planning spawned process")):
            plan = bench.planned_case_identities(args)
        self.assertEqual(Counter(row["family"] for row in plan), {
            "level1": 156, "level2": 636, "gemm": 96, "rotg": 50,
            "rank_k": 96, "symm": 48, "triangular": 160})
        self.assertEqual(len(plan), 1242)
        self.assertFalse(bench.coverage_evidence(plan, plan[:-1])["matches_plan"])
        self.assertFalse(bench.coverage_evidence(plan, plan + plan[:1])["matches_plan"])
        self.assertTrue(bench.coverage_evidence(plan, plan)["matches_plan"])

    def test_level2_case_missing_in_every_worker_is_recorded_as_failed(self):
        row = {"case": "sgemv_n", "kind": "f32", "shape": "sq32", "m": "32", "n": "32",
               "metric": "gops", "rate_gops": "10", "time_ns": "100", "status": "ok",
               "check_status": "sampled-ok"}
        key = "/".join(bench.l2.process_group_key(row))
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=bench.l2.CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
        with tempfile.TemporaryDirectory() as temp:
            args = bench.parse_args(["--output", str(Path(temp) / "run"), "--quick"])
            runner = bench.Runner(args)
            try:
                with patch.object(bench, "expected_level2_keys", return_value=[key, "missing-all-rounds"]), \
                     patch.object(bench.l2, "run_one_process", return_value=SimpleNamespace(
                         returncode=0, stdout=output.getvalue(), stderr="")) as worker, redirect_stdout(io.StringIO()):
                    runner.level2()
                self.assertEqual(worker.call_count, 3)
                self.assertEqual(len(runner.records), 2)
                self.assertEqual(runner.records[0]["operation"], "sgemv_n")
                missing = runner.records[1]
                self.assertEqual(missing["status"], "failed")
                self.assertEqual(missing["successful_repeats"], 0)
                self.assertIsNone(missing["median"])
            finally:
                runner.raw.close()
                runner.log.close()

    def test_exception_does_not_mark_run_completed(self):
        modules = (bench.l1, bench.l2, bench.rank, bench.rotg, bench.symm, bench.triangular)
        original_bindings = [module.subprocess for module in modules]
        result = SimpleNamespace(returncode=0, stdout="ok", stderr="")
        with patch.object(bench.subprocess, "run", return_value=result) as process:
            captured = bench.run_probe_process(
                ["probe"], capture_output=True, text=True, env={"A": "B"}, timeout=9
            )
            self.assertIs(captured, result)
            self.assertEqual(process.call_args.kwargs["timeout"], 9)
            self.assertEqual(process.call_args.kwargs["env"], {"A": "B"})
            self.assertTrue(process.call_args.kwargs["capture_output"])
            self.assertTrue(process.call_args.kwargs["text"])

        def interrupt(*args, **kwargs):
            self.assertTrue(all(module.subprocess is not bench.subprocess for module in modules))
            raise RuntimeError("interrupted")

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            with patch.object(bench, "command_text", return_value={}), \
                 patch.dict(bench.os.environ), \
                 patch.object(bench.Runner, "group", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    bench.main(["--output", str(output), "--quick", "--families", "gemm",
                                "--bin-dir", temp, "--library", str(Path(temp) / "missing.dll")])
            self.assertEqual(original_bindings, [module.subprocess for module in modules])
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertFalse(metadata["run_completed"])
            self.assertEqual(metadata["completed_families"], [])
            self.assertFalse(metadata["coverage"]["matches_plan"])
            self.assertEqual(len(metadata["coverage"]["missing"]), 4)


if __name__ == "__main__":
    unittest.main()
