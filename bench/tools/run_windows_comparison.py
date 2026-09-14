#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Local Windows/WSL comparison with checked, interleaved processes.

The readme profile reproduces the documented 46 Level 1, 60 Level 2 and
168 GEMM cases per library. This diagnostic runner does not use or weaken the
POSIX immutable-snapshot publication machinery. Missing libraries remain in the
plan and produce failures; no comparator or case is silently discarded.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import run_windows_benchmark as base
import run_gemm_sweep_isolated as gemm


def interleaved_schedule(libraries, repeats):
    return [(repeat, libraries[(repeat + offset) % len(libraries)])
            for repeat in range(repeats) for offset in range(len(libraries))]


def spec(name, metric, unit, parameters):
    return {"case": name, "metric": metric, "unit": unit, "parameters": parameters}


def jobs_for(args):
    """Enumerate the complete case manifest without spawning or loading BLAS."""
    jobs = []
    if "level1" in args.families:
        sizes = [4096] if args.quick else ([1048576] if args.profile == "readme" else [4096, 262144, 1048576])
        for n in sizes:
            for group, op, variant, ix, iy in base.l1.level1_cases([(1, 1)]):
                if args.profile == "readme" and op.endswith("axpby"):
                    continue
                metric = "bandwidth_gbps" if op in base.l1.LEVEL1_BANDWIDTH_OPS else "rate_gops"
                params = {"group": group, "op": op, "variant": variant, "n": n, "incx": ix, "incy": iy, "seconds": 1}
                jobs.append({"family": "level1", "type": "level1", "payload": (group, op, variant, ix, iy, n),
                    "cases": [spec(f"{op}/{variant}/n{n}", metric,
                                   "GB/s" if metric == "bandwidth_gbps" else "Gop/s", params)]})
            copies = (base.l1.copy_cases([8192, 8388608]) if args.profile == "readme" else
                [{"group": "copy", "op": op, "kind": kind, "copy_bytes": n * width, "copy_elements": n}
                 for kind, (op, _, width) in base.l1.COPY_KIND_SPECS.items()])
            for case in copies:
                if args.quick:
                    case = {**case, "copy_bytes": 4096 * base.l1.COPY_KIND_SPECS[case["kind"]][2],
                            "copy_elements": 4096}
                params = {**case, "n": case["copy_elements"], "seconds": 1, "variant": "default", "incx": 1, "incy": 1}
                jobs.append({"family": "level1", "type": "copy", "payload": case,
                    "cases": [spec(f"{case['op']}/bytes{case['copy_bytes']}", "bandwidth_gbps", "GB/s", params)]})
    if "level2" in args.families:
        sizes = [32] if args.quick else ([128, 256, 512] if args.profile == "readme" else [64, 256])
        for n in sizes:
            keys = base.expected_level2_keys(n)
            if args.profile == "readme": keys = keys[:20]
            jobs.append({"family": "level2", "type": "level2", "payload": n,
                         "cases": [spec(key, "rate_gops", "Gop/s", {"n": n, "m": n, "shape": f"sq{n}"}) for key in keys]})
    if "gemm" in args.families:
        shapes = (["sq32:32:32:32"] if args.quick else
                  gemm.DEFAULT_SHAPES if args.profile == "readme" else
                  [f"sq{n}:{n}:{n}:{n}" for n in (32, 64, 128, 256, 512, 1024)])
        transposes = ["NN"] if args.quick or args.profile == "readme" else ["NN", "NT", "TN", "TT"]
        for kind in ("sgemm", "dgemm", "cgemm", "zgemm"):
            for trans in transposes:
                for shape in shapes:
                    label, m, n, k = shape.split(":")
                    params = {"kind": kind, "shape": label, "m": int(m), "n": int(n), "k": int(k),
                              "transa": trans[0], "transb": trans[1]}
                    jobs.append({"family": "gemm", "type": "gemm", "payload": (kind, trans, shape),
                                 "cases": [spec(f"{kind}/{trans}/{label}", "median_gflops", "GFLOP/s", params)]})
    if args.profile == "full":
        for family, module, probe in (("rank_k", base.rank, "rank-k-probe"),
            ("symm", base.symm, "symm-probe"), ("triangular", base.triangular, "triangular-matrix-probe")):
            if family not in args.families: continue
            flags = ["--csv", "unused", "--reps", "5", "--alpha", "1"]
            if family != "triangular": flags += ["--beta", "0"]
            if family == "triangular": flags += ["--diag", "N"]
            if args.quick:
                flags += ["--uplo", "U"]
                if family != "symm": flags += ["--trans", "N"]
                if family != "rank_k": flags += ["--side", "L"]
            for n in ([32] if args.quick else [64, 256]): flags += ["--shape", f"sq{n}:{n}:{n}"]
            parsed = module.parse_args(flags)
            for case in module.requested_cases(parsed):
                params = {key: getattr(case, key) for key in ("side", "uplo", "trans", "diag", "alpha", "beta") if hasattr(case, key)}
                params.update({"routine": case.routine.name, "shape": case.shape.name})
                for key in ("m", "n", "k"):
                    if hasattr(case.shape, key): params[key] = getattr(case.shape, key)
                name = "/".join(str(x) for x in (case.routine.name, case.shape.name,
                    *[getattr(case, key) for key in ("side", "uplo", "trans", "diag") if hasattr(case, key)]))
                jobs.append({"family": family, "type": "matrix", "payload": (module, parsed, case, probe),
                             "cases": [spec(name, "median_gflops", "GFLOP/s", params)]})
        if "rotg" in args.families:
            parsed = base.rotg.parse_args(["--csv", "unused"])
            cases = base.rotg.requested_cases(parsed)
            if args.quick:
                seen = set()
                cases = [case for case in cases if case.routine not in seen and not seen.add(case.routine)]
            for case in cases:
                jobs.append({"family": "rotg", "type": "rotg", "payload": (parsed, case),
                    "cases": [spec(f"{case.routine}/{case.input_case}", "median_ns_per_call", "ns/call",
                                   {"routine": case.routine, "input_case": case.input_case})]})
    return jobs


def planned_cases(jobs, libraries):
    return [{"family": job["family"], **case, "library": library}
            for job in jobs for case in job["cases"] for library in libraries]


def coverage(planned, records):
    identity = lambda row: (row["family"], row["case"], row["library"])
    expected = Counter(map(identity, planned))
    actual = Counter(map(identity, records))
    return {"matches_plan": expected == actual,
            "missing": list((expected - actual).elements()),
            "unexpected_or_duplicate": list((actual - expected).elements())}


def assignments(values, label):
    result = {}
    for value in values:
        if "=" not in value: raise ValueError(f"{label} must be NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path or name in result: raise ValueError(f"invalid/duplicate {label}: {name}")
        result[name] = str(Path(path).resolve())
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--library", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--dependency-dir", action="append", default=[], metavar="NAME=PATH",
                   help="repeatable extra DLL search directories for a named library")
    p.add_argument("--profile", choices=("readme", "full"), default="readme")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--families", help="comma-separated subset; default profile families")
    p.add_argument("--process-repeats", type=int, default=3)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--bin-dir", default=str(base.ROOT / "zig-out/bin"))
    p.add_argument("--build-log", action="append", default=[])
    p.add_argument("--build-command", default="not supplied")
    p.add_argument("--zig", default="zig")
    args = p.parse_args(argv)
    try:
        args.libraries = assignments(args.library, "library")
        if "Zynum" not in args.libraries: raise ValueError("--library Zynum=PATH is required")
        if not Path(args.libraries["Zynum"]).is_file(): raise ValueError("Zynum DLL must exist")
        args.dependency_dirs = {name: [str(Path(path).parent)] for name, path in args.libraries.items()}
        for value in args.dependency_dir:
            if "=" not in value: raise ValueError("dependency-dir must be NAME=PATH")
            name, path = value.split("=", 1)
            if name not in args.libraries: raise ValueError(f"unknown dependency library: {name}")
            args.dependency_dirs[name].append(str(Path(path).resolve()))
        default = ["level1", "level2", "gemm"] if args.profile == "readme" else list(base.FAMILIES)
        args.families = args.families.split(",") if args.families else default
        if set(args.families) - set(default) or len(set(args.families)) != len(args.families):
            raise ValueError("invalid or duplicate families for profile")
        if args.process_repeats < 3: raise ValueError("at least 3 fresh process repeats are required")
        if args.timeout < 1: raise ValueError("timeout must be positive")
    except ValueError as exc: p.error(str(exc))
    return args


class Comparison:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.records = []
        self.raw = (self.output / "raw.jsonl").open("w", encoding="utf-8")
        self.processes = (self.output / "processes.jsonl").open("w", encoding="utf-8")
        self.schedule_log = (self.output / "schedule.jsonl").open("w", encoding="utf-8")
        self.active_library = None

    def exe(self, name):
        return str(Path(self.args.bin_dir).resolve() / (name + (".exe" if sys.platform == "win32" else "")))

    def capture_run(self, command, **kw):
        command = list(map(str, command))
        dirs = self.args.dependency_dirs[self.active_library]
        env = dict(kw.pop("env", os.environ))
        env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
        if sys.platform.startswith("linux"):
            env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [env.get("LD_LIBRARY_PATH", "")])
        if Path(command[0]).resolve() == Path(sys.executable).resolve():
            # ctypes on Windows uses safe DLL search rules rather than PATH.
            # Keep add_dll_directory handles alive through the worker execution.
            bootstrap = ("import os,sys,runpy; "
                f"dirs={dirs!r}; "
                "handles=[os.add_dll_directory(d) for d in dirs if os.path.isdir(d)] if hasattr(os,'add_dll_directory') else []; "
                "sys.argv=sys.argv[1:]; sys.path.insert(0,os.path.dirname(os.path.abspath(sys.argv[0]))); "
                "runpy.run_path(sys.argv[0],run_name='__main__')")
            command = [command[0], "-c", bootstrap, *command[1:]]
        kw.update(env=env)
        kw.setdefault("timeout", self.args.timeout)
        start = time.time()
        event = {"library": self.active_library, "command": command, "started": start,
                 "dependency_dirs": dirs}
        try:
            result = base.run_probe_process(command, **kw)
            event.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            return result
        except Exception as exc:
            event["error"] = str(exc)
            raise
        finally:
            event["elapsed_seconds"] = time.time() - start
            self.processes.write(json.dumps(event, default=str) + "\n")
            self.processes.flush()

    def execute(self, job, library, repeat, job_index):
        path = self.args.libraries[library]
        if not Path(path).is_file(): raise FileNotFoundError(f"library unavailable: {library}={path}")
        kind = job["type"]
        if kind in ("level1", "copy"):
            args = base.l1.parse_args(["--csv", "unused", "--process-repeats", "1"])
            args.level1_probe, args.copy_probe = self.exe("level1-probe"), self.exe("dcopy-probe")
            args.seconds = args.copy_seconds = 1
            if kind == "copy":
                row = base.l1.run_copy_op(args, library, path, job["payload"])
            else:
                group, op, variant, ix, iy, args.n = job["payload"]
                row = base.l1.run_level1_op(args, library, path, group, op, variant, ix, iy)
        elif kind == "level2":
            n = job["payload"]
            ops = ["legacy"] if self.args.profile == "readme" else ["all", "compact"]
            result = base.l2.run_one_process(base.l2.__file__, library, path,
                base.l2.Shape(f"sq{n}", n, n), 100 if n < 512 else 30, ops,
                bandwidth=None if self.args.profile == "readme" else min(7, n - 1))
            if result.returncode: raise RuntimeError(f"exit={result.returncode}: {result.stdout} {result.stderr}")
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            mapped = {}
            for row in rows:
                key = "/".join(str(value) for value in base.l2.process_group_key(row))
                if key in mapped: raise ValueError(f"duplicate Level 2 row {key}")
                if row.get("library") != library: raise ValueError("Level 2 library identity mismatch")
                mapped[key] = row
            expected = {case["case"] for case in job["cases"]}
            if mapped.keys() - expected: raise ValueError("unexpected Level 2 case identity")
            return mapped
        elif kind == "gemm":
            routine, trans, shape = job["payload"]
            outfile = self.output / f"gemm-{job_index}-{list(self.args.libraries).index(library)}-{repeat}.csv"
            result = self.capture_run([self.exe("gemm-sweep"), "--zynum-blas", path,
                "--kind", routine, "--trans", trans, "--shape", shape, "--check", "--reps",
                "9" if self.args.quick else "30", "--csv", str(outfile)], capture_output=True, text=True)
            if result.returncode: raise RuntimeError(f"exit={result.returncode}: {result.stdout} {result.stderr}")
            with outfile.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
            if len(rows) != 1: raise ValueError(f"GEMM returned {len(rows)} rows")
            row = rows[0]
            params = job["cases"][0]["parameters"]
            for key in ("kind", "m", "n", "k", "transa", "transb"):
                if row.get(key) != str(params[key]): raise ValueError(f"GEMM {key} mismatch")
            gemm.validate_gemm_evidence(row)
            row["median_gflops"] = ((8 if routine[0] in "cz" else 2) * params["m"] * params["n"] * params["k"] / float(row["median_ns"]))
            # gemm-sweep names its single --zynum-blas slot Zynum even for a
            # comparator. Preserve the emitted label separately and use the
            # independently scheduled library identity for this local report.
            row["probe_library_label"] = row.get("library")
            row["library"] = library
        elif kind == "matrix":
            module, args, case, probe = job["payload"]
            row = module.run_one_process(args, library, path, case,
                probe_path=self.exe(probe), public_library_path=path, redact_private_paths=lambda value: value)
        elif kind == "rotg":
            args, case = job["payload"]
            row = base.rotg.run_one_process(args, library, path, case, probe_path=self.exe("rotg-latency-probe"))
        else: raise ValueError(f"unknown job type {kind}")
        return {job["cases"][0]["case"]: row}

    def run_job(self, job, index):
        collected = {(case["case"], name): [] for case in job["cases"] for name in self.args.libraries}
        for repeat, name in interleaved_schedule(list(self.args.libraries), self.args.process_repeats):
            self.active_library = name
            self.schedule_log.write(json.dumps({"job": index, "family": job["family"],
                "cases": [case["case"] for case in job["cases"]], "library": name, "repeat": repeat + 1}) + "\n")
            self.schedule_log.flush()
            try:
                rows = self.execute(job, name, repeat, index)
            except Exception as exc:
                rows = {case["case"]: {"status": "error", "check_status": "error", "error": str(exc)} for case in job["cases"]}
            for case in job["cases"]:
                row = rows.get(case["case"], {"status": "missing", "check_status": "missing", "error": "worker omitted planned case"})
                collected[case["case"], name].append(row)
                self.raw.write(json.dumps({"family": job["family"], "case": case["case"],
                    "library": name, "repeat": repeat + 1, "row": row}, default=str) + "\n")
            self.raw.flush()
        for case in job["cases"]:
            for name in self.args.libraries:
                row = base.summarize(job["family"], case["case"], case["metric"], case["unit"],
                                     collected[case["case"], name], self.args.process_repeats)
                row["library"] = name
                row["parameters"] = json.dumps({**case["parameters"], **json.loads(row["parameters"])})
                row["operation"] = case["case"].split("/")[0]
                row["median_ns"] = None
                row["process_median_ns_samples"] = []
                if job["family"] == "gemm" and row["status"] == "ok":
                    latency = [float(value["median_ns"]) for value in collected[case["case"], name]]
                    row["median_ns"] = statistics.median(latency)
                    row["process_median_ns_samples"] = latency
                    params = case["parameters"]
                    work = (8 if params["kind"][0] in "cz" else 2) * params["m"] * params["n"] * params["k"]
                    row["median"] = work / row["median_ns"]
                    row["min"], row["max"] = work / max(latency), work / min(latency)
                self.records.append(row)
        self.save()
        print(f"[{index + 1}] {job['family']} {job['cases'][0]['case']} ({len(job['cases'])} cases)", flush=True)

    def save(self):
        (self.output / "records.json").write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        with (self.output / "records.csv").open("w", newline="", encoding="utf-8") as handle:
            if self.records:
                writer = csv.DictWriter(handle, fieldnames=list(self.records[0]))
                writer.writeheader()
                writer.writerows({**row, "samples": json.dumps(row["samples"])} for row in self.records)


def main(argv=None):
    args = parse_args(argv)
    for key in base.THREAD_VARS: os.environ[key] = "24"
    os.environ.update(OPENBLAS_DYNAMIC="0", MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE", MKL_INTERFACE_LAYER="LP64")
    for key in ("ZYNUM_MAXIMUM_THREADS", "ZYNUM_MAX_ISA"): os.environ.pop(key, None)
    runner = Comparison(args)
    jobs = jobs_for(args)
    plan = planned_cases(jobs, args.libraries)
    plan_bytes = (json.dumps(plan, indent=2) + "\n").encode("utf-8")
    (runner.output / "planned_cases.json").write_bytes(plan_bytes)
    artifacts = set(map(Path, args.libraries.values()))
    artifacts.update(p for p in Path(args.bin_dir).iterdir() if p.is_file() and
                     (p.suffix == ".exe" if sys.platform == "win32" else os.access(p, os.X_OK)))
    for dirs in args.dependency_dirs.values():
        for directory in dirs:
            artifacts.update(Path(directory).glob("*.dll" if sys.platform == "win32" else "*.so*"))
    artifacts = sorted(artifacts, key=str)
    sources = base.command_text(["git", "ls-files", "--cached", "--others", "--exclude-standard"])
    metadata = {"schema": "zynum-local-comparison-v1", "profile": args.profile, "quick": args.quick,
        "libraries": args.libraries, "dependency_dirs": args.dependency_dirs,
        "isolation": "fresh processes; Level 2 runs one shape batch per library/repeat; before/after hashes, not immutable snapshots",
        "schedule": "job-major, cyclic library rotation by repeat; schedule.jsonl contains exact order",
        "schedule_position_balanced": args.process_repeats % len(args.libraries) == 0,
        "thread_note": "comparators requested 24 threads/dynamic false; Zynum automatic threads and ISA; actual worker counts not instrumented",
        "threads": {key: os.environ.get(key) for key in (*base.THREAD_VARS, "OPENBLAS_DYNAMIC", "MKL_DYNAMIC", "OMP_DYNAMIC", "MKL_INTERFACE_LAYER", "ZYNUM_MAXIMUM_THREADS", "ZYNUM_MAX_ISA")},
        "platform": platform.platform(), "cpu_count": os.cpu_count(), "python": sys.version,
        "process_repeats": args.process_repeats, "run_completed": False, "completed_jobs": 0,
        "gemm_timing": "30 timed batches per process; calibrated batch >=100 us; fractional ns per call; batch_calls retained in raw rows",
        "planned_counts": dict(Counter(row["family"] for row in plan)),
        "planned_cases_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_command": args.build_command, "build_logs": [base.digest(path) for path in args.build_log],
        "revision": base.command_text(["git", "rev-parse", "HEAD"]),
        "git_status": base.command_text(["git", "status", "--short"]),
        "zig_version": base.command_text([args.zig, "version"]),
        "sources": [base.digest(base.ROOT / path) for path in sources.get("stdout", "").splitlines()
                    if Path(path).suffix in {".py", ".zig", ".zon"}],
        "artifacts_before": [base.digest(path) for path in artifacts],
        "correctness": "existing per-case numerical checks; sampled checks are not exhaustive; only all-repeat positive finite checked groups receive aggregates"}
    meta_path = runner.output / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        with base.capture_probe_processes(runner.capture_run):
            for index, job in enumerate(jobs):
                runner.run_job(job, index)
                metadata["completed_jobs"] = index + 1
            metadata["run_completed"] = True
    finally:
        runner.raw.close(); runner.processes.close(); runner.schedule_log.close()
        runner.save()
        metadata["artifacts_after"] = [base.digest(path) for path in artifacts]
        metadata["artifacts_unchanged"] = metadata["artifacts_before"] == metadata["artifacts_after"]
        metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metadata["coverage"] = coverage(plan, runner.records)
        metadata["failed_records"] = sum(row["status"] != "ok" for row in runner.records)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0 if (metadata["run_completed"] and metadata["coverage"]["matches_plan"] and
                 metadata["artifacts_unchanged"] and not metadata["failed_records"] and runner.records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
