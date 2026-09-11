#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Local diagnostic benchmarks; does not publish a portable comparison report.

Uses existing checked probe/worker entry points, but deliberately does not use
the POSIX descriptor-snapshot report publisher. Paths and raw output are local.
Each measured repeat starts a fresh process. Partial/failing groups never have
an aggregate performance number. The default is broad routine coverage with
bounded sizes, not the Cartesian product of every BLAS parameter.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import run_level1_report as l1
import run_level2_report as l2
import run_rank_k_report as rank
import run_rotg_latency_report as rotg
import run_symm_report as symm
import run_triangular_matrix_report as triangular

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("level1", "level2", "gemm", "rotg", "rank_k", "symm", "triangular")
CHECKED = {"sampled-ok", "checked-ok"}
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def digest(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha}


def command_text(command):
    try:
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                           timeout=30, encoding="utf-8", errors="replace")
        return {"command": command, "returncode": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def summarize(family, case, metric, unit, rows, expected):
    values = []
    for row in rows:
        try:
            value = float(row[metric])
            valid = (row.get("status", "ok") == "ok"
                     and row.get("check_status", row.get("check")) in CHECKED
                     and math.isfinite(value) and value > 0)
            if valid:
                values.append(value)
        except (KeyError, TypeError, ValueError):
            pass
    complete = len(rows) == expected and len(values) == expected
    evidence = next((r for r in rows if r.get("status", "ok") == "ok"), {})
    parameters = {key: evidence[key] for key in ("shape", "label", "kind", "m", "n", "k",
        "side", "uplo", "trans", "transa", "transb", "diag", "incx", "incy", "alpha", "beta", "variant")
        if key in evidence}
    return {"family": family, "case": case, "library": "Zynum",
            "operation": (case.split("/")[0] if family == "level2" else
                evidence.get("routine", evidence.get("op", evidence.get("kind", case.split("/")[0])))),
            "parameters": json.dumps(parameters),
            "metric": metric, "unit": unit, "status": "ok" if complete else "failed",
            "process_repeats": expected, "successful_repeats": len(values),
            "median": statistics.median(values) if complete else None,
            "min": min(values) if complete else None,
            "max": max(values) if complete else None,
            "samples": values,
            "check_status": "checked" if complete else "see_raw"}


def expected_level2_keys(n):
    """Enumerate identities independently of worker output, without loading BLAS.

    emit is used only as a schema formatter. These synthetic rows never enter
    measurement records: discard all values except process_group_key.
    """
    shape = l2.Shape(f"sq{n}", n, n)
    operations = l2.expand_operations(["all", "compact"])
    bandwidth = min(7, n - 1)
    schemas = []
    for prefix, kind in (("s", "f32"), ("d", "f64"), ("c", "c32"), ("z", "c64")):
        suffixes = (("gemv_n", "gemv_t", "symv", "ger") if prefix in "sd" else
                    ("gemv_n", "gemv_t", "gemv_c", "hemv", "geru", "gerc"))
        for suffix in suffixes:
            l2.emit(schemas, prefix + suffix, kind, "Zynum", shape, 1, 1,
                    {"check_status": "sampled-ok"})
    cases = [*l2.triangular_cases(operations), *l2.rank_update_cases(operations),
             *l2.banded_cases(operations, bandwidth), *l2.packed_structured_mv_cases(operations),
             *l2.packed_triangular_cases(operations), *l2.packed_rank_cases(operations),
             *l2.triangular_banded_cases(operations, bandwidth)]
    for case in cases:
        l2.emit(schemas, case.case, case.kind, "Zynum", shape, 1, 1,
                {"check_status": "sampled-ok"}, parameters=case)
    keys = ["/".join(str(value) for value in l2.process_group_key(
        {key: str(value) for key, value in row.items()})) for row in schemas]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate planned Level 2 identity")
    return keys


def planned_case_identities(args):
    """Pure case enumeration using the same bounded profile, with no I/O/BLAS."""
    planner = object.__new__(Runner)
    planner.args = args
    planner.planning = True
    planner.library = "<planning-only>"
    planned = []
    planner.group = lambda family, case, *unused: planned.append({"family": family, "case": case})
    for family in args.families:
        planner.run_family(family)
    identities = [(row["family"], row["case"]) for row in planned]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate planned case identity")
    return planned


def coverage_evidence(planned, records):
    expected = {(row["family"], row["case"]) for row in planned}
    actual = Counter((row["family"], row["case"]) for row in records)
    missing = sorted(expected - actual.keys())
    unexpected = sorted(actual.keys() - expected)
    duplicates = sorted(key for key, count in actual.items() if count != 1)
    return {"matches_plan": not (missing or unexpected or duplicates),
            "missing": missing, "unexpected": unexpected, "duplicates": duplicates}


class Runner:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.records = []
        self.raw = (self.output / "raw.jsonl").open("w", encoding="utf-8")
        self.log = (self.output / "processes.jsonl").open("w", encoding="utf-8")
        self.real_run = subprocess.run
        self.library = str(Path(args.library).resolve())

    def capture_run(self, command, *pos, **kw):
        kw.setdefault("timeout", self.args.timeout)
        start = time.time()
        try:
            result = self.real_run(command, *pos, **kw)
            self.log.write(json.dumps({"command": list(map(str, command)),
                "started": start, "elapsed_seconds": time.time() - start,
                "returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr}, default=str) + "\n")
            self.log.flush()
            return result
        except Exception as exc:
            self.log.write(json.dumps({"command": list(map(str, command)),
                "started": start, "error": str(exc)}) + "\n")
            self.log.flush()
            raise

    def exe(self, name):
        return str(Path(self.args.bin_dir).resolve() / (name + ".exe"))

    def run_family(self, family):
        if family in {"rank_k", "symm", "triangular"}:
            module, probe = {"rank_k": (rank, "rank-k-probe"), "symm": (symm, "symm-probe"),
                             "triangular": (triangular, "triangular-matrix-probe")}[family]
            self.matrix(family, module, probe)
        else:
            getattr(self, family)()

    def group(self, family, name, metric, unit, execute):
        rows = []
        for repeat in range(self.args.process_repeats):
            try:
                row = execute(repeat)
            except Exception as exc:
                row = {"status": "error", "check_status": "error", "error": str(exc)}
            rows.append(row)
            self.raw.write(json.dumps({"family": family, "case": name,
                "repeat": repeat + 1, "row": row}, default=str) + "\n")
            self.raw.flush()
        record = summarize(family, name, metric, unit, rows, self.args.process_repeats)
        self.records.append(record)
        print(f"[{len(self.records)}] {family} {name}: {record['status']}", flush=True)
        self.save()

    def save(self):
        with (self.output / "records.csv").open("w", newline="", encoding="utf-8") as f:
            if self.records:
                writer = csv.DictWriter(f, fieldnames=list(self.records[0]))
                writer.writeheader()
                writer.writerows({**r, "samples": json.dumps(r["samples"])} for r in self.records)
        (self.output / "records.json").write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def level1(self):
        a = l1.parse_args(["--csv", "unused", "--process-repeats", "1"])
        a.level1_probe, a.copy_probe = self.exe("level1-probe"), self.exe("dcopy-probe")
        a.seconds = a.copy_seconds = 1
        for n in ([4096] if self.args.quick else [4096, 262144, 1048576]):
            a.n = n
            for group, op, variant, ix, iy in l1.level1_cases([(1, 1)]):
                metric = "bandwidth_gbps" if op in l1.LEVEL1_BANDWIDTH_OPS else "rate_gops"
                self.group("level1", f"{op}/{variant}/n{n}", metric,
                    "GB/s" if metric == "bandwidth_gbps" else "Gop/s",
                    lambda _, g=group, o=op, v=variant, x=ix, y=iy:
                    l1.run_level1_op(a, "Zynum", self.library, g, o, v, x, y))
            for kind, (op, _, width) in l1.COPY_KIND_SPECS.items():
                case = {"group": "copy", "op": op, "kind": kind,
                        "copy_bytes": n * width, "copy_elements": n}
                self.group("level1", f"{op}/n{n}", "bandwidth_gbps", "GB/s",
                    lambda _, c=case: l1.run_copy_op(a, "Zynum", self.library, c))

    def matrix(self, family, module, probe):
        flags = ["--csv", "unused", "--reps", "5", "--alpha", "1"]
        if family != "triangular":
            flags += ["--beta", "0"]
        # Each routine, both triangles/sides and legal transposes in full mode.
        if self.args.quick:
            flags += ["--uplo", "U"]
            if family != "symm": flags += ["--trans", "N"]
            if family != "rank_k": flags += ["--side", "L"]
        if family == "triangular": flags += ["--diag", "N"]
        for n in ([32] if self.args.quick else [64, 256]):
            flags += ["--shape", f"sq{n}:{n}:{n}"]
        a = module.parse_args(flags)
        a.probe = self.exe(probe)
        for case in module.requested_cases(a):
            name = "/".join(str(x) for x in (
                case.routine.name, case.shape.name,
                *[getattr(case, attr) for attr in ("side", "uplo", "trans", "diag") if hasattr(case, attr)]))
            self.group(family, name, "median_gflops", "GFLOP/s",
                lambda _, c=case: module.run_one_process(a, "Zynum", self.library, c,
                    probe_path=a.probe, public_library_path=self.library,
                    redact_private_paths=lambda value: value))

    def rotg(self):
        a = rotg.parse_args(["--csv", "unused", "--samples", "9",
                            "--calls-per-sample", "100000"])
        a.probe = self.exe("rotg-latency-probe")
        cases = rotg.requested_cases(a)
        if self.args.quick:
            seen = set()
            cases = [c for c in cases if c.routine not in seen and not seen.add(c.routine)]
        for case in cases:
            self.group("rotg", f"{case.routine}/{case.input_case}",
                "median_ns_per_call", "ns/call",
                lambda _, c=case: rotg.run_one_process(a, "Zynum", self.library, c))

    def level2(self):
        operations = ["all", "compact"]
        for n in ([32] if self.args.quick else [64, 256]):
            expected = expected_level2_keys(n)
            if getattr(self, "planning", False):
                for key in expected:
                    self.group("level2", key, "rate_gops", "Gop/s", None)
                continue
            repeats = []
            for repeat in range(self.args.process_repeats):
                try:
                    result = l2.run_one_process(Path(l2.__file__), "Zynum", self.library,
                        l2.Shape(f"sq{n}", n, n), 10, operations, bandwidth=min(7, n - 1))
                    if result.returncode != 0:
                        raise RuntimeError(f"exit={result.returncode}: {result.stdout} {result.stderr}")
                    rows = list(csv.DictReader(io.StringIO(result.stdout)))
                    if not rows: raise ValueError("Level 2 worker produced no rows")
                    mapped = {}
                    for row in rows:
                        key = "/".join(str(x) for x in l2.process_group_key(row))
                        if key in mapped: raise ValueError(f"duplicate Level 2 row: {key}")
                        mapped[key] = row
                    repeats.append(mapped)
                except Exception as exc:
                    repeats.append({f"worker/n{n}": {"status": "error", "error": str(exc)}})
            # Include independently planned keys even if EVERY worker omits one.
            keys = dict.fromkeys([*expected, *(key for rows in repeats for key in rows)])
            for key in keys:
                self.group("level2", key, "rate_gops", "Gop/s",
                    lambda repeat, k=key: repeats[repeat].get(k, {"status": "missing"}))

    def gemm(self):
        sizes = [32] if self.args.quick else [32, 64, 128, 256, 512, 1024]
        for kind in ("sgemm", "dgemm", "cgemm", "zgemm"):
            for trans in (["NN"] if self.args.quick else ["NN", "NT", "TN", "TT"]):
                for n in sizes:
                    name = f"{kind}/{trans}/sq{n}"
                    def execute(repeat):
                        output = self.output / f"gemm-{kind}-{trans}-{n}-{repeat}.csv"
                        command = [self.exe("gemm-sweep"), "--zynum-blas", self.library,
                            "--check", "--kind", kind, "--trans", trans,
                            "--shape", f"sq{n}:{n}:{n}:{n}", "--reps",
                            "9" if self.args.quick else "30", "--csv", str(output)]
                        p = subprocess.run(command, capture_output=True, text=True)
                        if p.returncode: raise RuntimeError(f"exit={p.returncode}: {p.stdout} {p.stderr}")
                        with output.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
                        if len(rows) != 1: raise ValueError(f"GEMM returned {len(rows)} rows")
                        row = rows[0]
                        for field, expected in {"kind": kind, "transa": trans[0], "transb": trans[1],
                                                "m": str(n), "n": str(n), "k": str(n)}.items():
                            if row.get(field) != expected: raise ValueError(f"GEMM {field} mismatch")
                        # gflops in the probe is based on best_ns; use median latency.
                        ns = float(row["median_ns"])
                        row["median_gflops"] = (8 if kind[0] in "cz" else 2) * n ** 3 / ns
                        return row
                    self.group("gemm", name, "median_gflops", "GFLOP/s", execute)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="new local output directory")
    p.add_argument("--bin-dir", default=str(ROOT / "zig-out/bin"))
    p.add_argument("--library", default=str(ROOT / "zig-out/bin/zynum_blas.dll"))
    p.add_argument("--quick", "--smoke", action="store_true", help="small cases in all families")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--process-repeats", type=int, default=3)
    p.add_argument("--timeout", type=int, default=180, help="seconds per child process")
    p.add_argument("--build-log", action="append", default=[])
    p.add_argument("--build-command", default="not supplied")
    p.add_argument("--zig", default="zig")
    a = p.parse_args(argv)
    a.families = a.families.split(",")
    if not a.families or set(a.families) - set(FAMILIES): p.error("unknown family")
    if len(set(a.families)) != len(a.families): p.error("duplicate families")
    if a.process_repeats < 3: p.error("at least 3 fresh-process repeats are required")
    if a.timeout < 1: p.error("timeout must be positive")
    return a


def main(argv=None):
    args = parse_args(argv)
    for key in THREAD_VARS: os.environ[key] = str(os.cpu_count() or 1)
    for key in ("ZYNUM_MAXIMUM_THREADS", "ZYNUM_MAX_ISA"):
        os.environ.pop(key, None)
    os.environ["OPENBLAS_DYNAMIC"] = "0"
    runner = Runner(args)
    planned = planned_case_identities(args)
    planned_bytes = (json.dumps(planned, indent=2) + "\n").encode("utf-8")
    (runner.output / "planned_cases.json").write_bytes(planned_bytes)
    sources = command_text(["git", "ls-files", "--cached", "--others", "--exclude-standard"])
    source_files = [ROOT / p for p in sources.get("stdout", "").splitlines()
                    if Path(p).suffix in {".zig", ".zon", ".py"}]
    artifact_paths = [Path(args.library), *Path(args.bin_dir).glob("*.exe")]
    metadata = {"schema": "zynum-local-diagnostic-v1", "scope": "local diagnostic; no comparator",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(), "python": sys.version,
        "argv": sys.argv, "profile": "quick" if args.quick else "full-routine-bounded-size",
        "families": args.families, "process_repeats": args.process_repeats,
        "run_completed": False, "completed_families": [],
        "planned_counts": dict(Counter(row["family"] for row in planned)),
        "planned_cases_sha256": hashlib.sha256(planned_bytes).hexdigest(),
        "threads": {key: os.environ.get(key) for key in (*THREAD_VARS, "OPENBLAS_DYNAMIC", "ZYNUM_MAXIMUM_THREADS", "ZYNUM_MAX_ISA")},
        "thread_note": "Zynum automatic threads/ISA (caps unset); comparator environment uses logical CPU count. Actual worker counts are not instrumented.",
        "build_command": args.build_command, "build_logs": [digest(p) for p in args.build_log],
        "zig_version": command_text([args.zig, "version"]),
        "revision": command_text(["git", "rev-parse", "HEAD"]),
        "git_status": command_text(["git", "status", "--short"]),
        "sources": [digest(p) for p in source_files], "artifacts_before": [digest(p) for p in artifact_paths],
        "correctness": "Per-case existing probe checks; sampled checks are not exhaustive proofs.",
        "isolation": "Fresh processes; before/after hashes, not immutable POSIX snapshots."}
    meta_path = runner.output / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        # Existing helpers share the subprocess module. Instrument calls only for
        # this sequential diagnostic run; restore it even on interruption.
        subprocess.run = runner.capture_run
        for family in args.families:
            runner.run_family(family)
            metadata["completed_families"].append(family)
        metadata["run_completed"] = True
    finally:
        subprocess.run = runner.real_run
        runner.raw.close()
        runner.log.close()
        runner.save()
        metadata["artifacts_after"] = [digest(p) for p in artifact_paths]
        metadata["artifacts_unchanged"] = metadata["artifacts_before"] == metadata["artifacts_after"]
        metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metadata["record_count"] = len(runner.records)
        metadata["failed_records"] = sum(r["status"] != "ok" for r in runner.records)
        metadata["coverage"] = coverage_evidence(planned, runner.records)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0 if (metadata["run_completed"] and metadata["coverage"]["matches_plan"]
                 and runner.records and not metadata["failed_records"]
                 and metadata["artifacts_unchanged"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
