# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
import io
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_windows_comparison as bench


class ComparisonTests(unittest.TestCase):
    def test_probe_suffix_matches_execution_platform(self):
        for platform, suffix in (("win32", ".exe"), ("linux", "")):
            with tempfile.TemporaryDirectory() as temp, patch.object(bench.sys, "platform", platform):
                runner = bench.Comparison(self.args(temp, bin_dir=temp))
                try:
                    self.assertEqual(Path(runner.exe("level1-probe")).name, "level1-probe" + suffix)
                finally:
                    self.close(runner)

    def args(self, temp, **kwargs):
        values = dict(output=str(Path(temp) / "out"), profile="readme", quick=False,
                      families=["level1", "level2", "gemm"], process_repeats=4,
                      libraries={"Zynum": str(Path(temp) / "zynum.dll"), "Missing": str(Path(temp) / "missing.dll")})
        values.update(kwargs)
        return SimpleNamespace(**values)

    def close(self, runner):
        runner.raw.close(); runner.processes.close(); runner.schedule_log.close()

    def test_readme_and_full_counts_without_processes(self):
        with patch.object(bench.subprocess, "run", side_effect=AssertionError("spawned")):
            args = self.args("unused")
            jobs = bench.jobs_for(args)
            self.assertEqual(Counter(job["family"] for job in jobs for _ in job["cases"]),
                             {"level1": 46, "level2": 60, "gemm": 168})
            args.profile, args.families = "full", list(bench.base.FAMILIES)
            self.assertEqual(sum(len(job["cases"]) for job in bench.jobs_for(args)), 1242)

    def test_four_repeat_interleave_is_balanced(self):
        libraries = ["Zynum", "MKL", "OpenBLAS", "BLIS"]
        order = bench.interleaved_schedule(libraries, 4)
        self.assertEqual([name for _, name in order[:8]], libraries + libraries[1:] + libraries[:1])
        for position in range(4):
            self.assertEqual(set(name for _, name in order[position::4]), set(libraries))

    def test_missing_library_retains_every_case_and_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = bench.Comparison(self.args(temp, families=["gemm"], quick=True))
            try:
                job = bench.jobs_for(runner.args)[0]
                with redirect_stdout(io.StringIO()): runner.run_job(job, 0)
                self.assertEqual(len(runner.records), 2)
                self.assertTrue(all(row["status"] == "failed" for row in runner.records))
                self.assertTrue(all(row["median"] is None for row in runner.records))
                runner.raw.flush()
                self.assertEqual(len((runner.output / "raw.jsonl").read_text().splitlines()), 8)
                plan = bench.planned_cases([job], runner.args.libraries)
                self.assertTrue(bench.coverage(plan, runner.records)["matches_plan"])
                self.assertFalse(bench.coverage(plan, runner.records[:-1])["matches_plan"])
            finally: self.close(runner)

    def test_gemm_even_repeat_median_uses_latency(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = bench.Comparison(self.args(temp, families=["gemm"], quick=True, libraries={"Zynum": "fake"}))
            try:
                job = bench.jobs_for(runner.args)[0]
                work = 2 * 32**3
                def execute(job, name, repeat, index):
                    ns = [1, 2, 8, 10][repeat]
                    return {job["cases"][0]["case"]: {"check": "checked-ok", "median_ns": ns, "median_gflops": work / ns}}
                with patch.object(runner, "execute", side_effect=execute), redirect_stdout(io.StringIO()):
                    runner.run_job(job, 0)
                row = runner.records[0]
                self.assertEqual(row["median_ns"], 5)
                self.assertEqual(row["median"], work / 5)
                self.assertNotEqual(row["median"], (work / 2 + work / 8) / 2)
            finally: self.close(runner)

    def test_worker_omission_remains_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = bench.Comparison(self.args(temp, families=["level2"], quick=True, libraries={"Zynum": "fake"}))
            try:
                job = bench.jobs_for(runner.args)[0]
                with patch.object(runner, "execute", return_value={}), redirect_stdout(io.StringIO()):
                    runner.run_job(job, 0)
                self.assertEqual(len(runner.records), 20)
                self.assertTrue(all(row["status"] == "failed" for row in runner.records))
            finally: self.close(runner)

    def test_one_missing_repeat_blocks_aggregate(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = bench.Comparison(self.args(temp, families=["gemm"], quick=True, libraries={"Zynum": "fake"}))
            try:
                job = bench.jobs_for(runner.args)[0]
                def execute(job, name, repeat, index):
                    return {} if repeat == 2 else {job["cases"][0]["case"]:
                        {"check": "checked-ok", "median_ns": 2, "median_gflops": 10}}
                with patch.object(runner, "execute", side_effect=execute), redirect_stdout(io.StringIO()):
                    runner.run_job(job, 0)
                self.assertEqual(runner.records[0]["successful_repeats"], 3)
                self.assertEqual(runner.records[0]["status"], "failed")
                self.assertIsNone(runner.records[0]["median"])
            finally: self.close(runner)


if __name__ == "__main__": unittest.main()
