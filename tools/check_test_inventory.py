#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fail-closed validation for the public test inventory schema v3."""

from __future__ import annotations

import argparse
import ast
import contextlib
import ctypes
import fnmatch
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
import posixpath
import re
import secrets
import shlex
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, NamedTuple
from unittest import mock

SCHEMA_ID = "zynum-test-inventory-v3"
SCHEMA_VERSION = 3
BUILD_SCHEMA_ID = "zynum-build-inventory-v3"
INVENTORY_PATH = "tools/test_inventory.json"
BUILD_INVENTORY_PATH = "tools/build_inventory.json"
SELF_TEST_PATH = "test/build/test_test_inventory.py"
AGGREGATE_STEP_ID = "step:build.zig:build:test"
HOST_TOOL_SMOKE_STEP_ID = "step:build.zig:build:test-host-tool-smoke"
MODES = ("Debug", "ReleaseSafe", "ReleaseFast")
ZIG_ENUMERATION_SOURCE = "zig-0.16-builtin-test-functions"
PYTHON_ENUMERATION_SOURCE = "static"
FACTORY_ROLE = "test-inventory-enumerator-factory"
FACTORY_EXPANSION_RELATION = "one-per-applicable-logical-zig-root"
FACTORY_ROOT_MODULE_RELATION = "same-pointer"
FACTORY_COMPILE_ID = "compile:build.zig:build:inventory_tests"
FACTORY_LAUNCH_ID = "launch:build.zig:build:run_inventory_tests"
PYTHON_TOOLING_ROOT_ID = "python-root:benchmark-tools-discovery"
PYTHON_TOOLING_LAUNCH_ID = "launch:build.zig:build:python_tooling_tests"
PYTHON_TOOLING_STEP_ID = "step:build.zig:build:test-python-tooling"
HOST_TOOL_SMOKE_DIRECT_DEPENDENCIES = (
    {"id": PYTHON_TOOLING_STEP_ID, "condition": "always"},
    {"id": "launch:build.zig:build:abi_manifest_smoke_test", "condition": "always"},
    {"id": "launch:build.zig:build:c_header_smoke_test", "condition": "always"},
    {"id": "launch:build.zig:build:cpp_header_smoke_test", "condition": "always"},
    {
        "id": "launch:build.zig:build:fortran_module_smoke_test",
        "condition": "always",
    },
    {
        "id": "step:build.zig:build:test-abi-baseline-observer",
        "condition": "always",
    },
)
LEGACY_WORKFLOW_MODE_COMMANDS = {
    "workflow-launch:.github/workflows/ci.yml:target-tests:test-debug-target": "zig build test ${{ matrix.target_args }} -Dtest-optimize=Debug -Dhost-tool-smoke=${{ matrix.host_tool_smoke }} --summary failures",
    "workflow-launch:.github/workflows/ci.yml:target-tests:test-releasesafe-target": "zig build --release=safe test ${{ matrix.target_args }} -Dtest-optimize=ReleaseSafe -Dhost-tool-smoke=${{ matrix.host_tool_smoke }} --summary failures",
    "workflow-launch:.github/workflows/ci.yml:target-tests:test-releasefast-target": "zig build --release=fast test ${{ matrix.target_args }} -Dtest-optimize=ReleaseFast -Dhost-tool-smoke=${{ matrix.host_tool_smoke }} --summary failures",
    "workflow-launch:.github/workflows/release.yml:artifacts:test": "zig build test ${{ matrix.target_args }} -Dtest-optimize=ReleaseSafe --summary failures",
}
_PYTHON_TOOLING_REVIEWED_SOURCE_SHA256 = (
    (
        "bench/tools/test_benchmark_artifact_snapshot.py",
        "eef6fb1aaf709f4720b2c16bb7bb9d932cccf95d93970d544c105c06113c8cc9",
    ),
    (
        "bench/tools/test_benchmark_metadata.py",
        "175c6e9a97ee195830355799d274ae85b904aa4edffd8d69defc97674ca6dbc8",
    ),
    (
        "bench/tools/test_full_benchmark_report.py",
        "066421c364f32bc25f44687c55c3086902d2a30e2661af8bf1862dc8d7c34726",
    ),
    (
        "bench/tools/test_gemm_sweep_tools.py",
        "044a873001af17fe2100e381feba0ad4488e93cbfc9f55970bf81d5d5061d320",
    ),
    (
        "bench/tools/test_level1_report.py",
        "00f94923e2bae67768c3d21dc619fd26c307175b14ad5fbfbeddef15283851f8",
    ),
    (
        "bench/tools/test_level2_report.py",
        "2ce852761cc7e9e6d74a9c324419c56d2db6abc5a349e9d524281b3ddf3697ee",
    ),
    (
        "bench/tools/test_rank_k_report.py",
        "834ffd3fd3d1bb578d44fbca5f8d4b3d497c23b590a083edee9f81d70774f0af",
    ),
    (
        "bench/tools/test_report_comparison.py",
        "29514c5ced81ec8ca8e5665ccbead524c84098d3f04ec045a41d182a59bc7250",
    ),
    (
        "bench/tools/test_report_plotters.py",
        "362b6339cd9c76f45b4729cbaa9d42499f01b0e9833a6890d7dff7fce90b4e4f",
    ),
    (
        "bench/tools/test_report_publication.py",
        "521d5237d9c5f99d87f07e7305becd09b98d43b4a94d6f5457c1b4483de4a01c",
    ),
    (
        "bench/tools/test_report_schedule.py",
        "4e008b60e2272bc8f09fd02676a164f5d25aa919dfd560a03f66530829ae2ae3",
    ),
    (
        "bench/tools/test_rotg_latency_report.py",
        "db967016f6201a8bd93861566301f093ac46bd74fae8f59b5526715ac8f5119d",
    ),
    (
        "bench/tools/test_symm_report.py",
        "0f6a981a87909ed07e75e5feba88134702a0577f5c6afd054921650fb8587f14",
    ),
    (
        "bench/tools/test_triangular_matrix_report.py",
        "50f818b9aa7ccef69c8881459c720a7e0206b9f779364e76b316cad540fb7f87",
    ),
)
_PYTHON_TOOLING_EXECUTION_SOURCE_SHA256 = (
    (
        "bench/tools/test_benchmark_artifact_snapshot.py",
        "eef6fb1aaf709f4720b2c16bb7bb9d932cccf95d93970d544c105c06113c8cc9",
    ),
    (
        "bench/tools/test_benchmark_metadata.py",
        "175c6e9a97ee195830355799d274ae85b904aa4edffd8d69defc97674ca6dbc8",
    ),
    (
        "bench/tools/test_full_benchmark_report.py",
        "066421c364f32bc25f44687c55c3086902d2a30e2661af8bf1862dc8d7c34726",
    ),
    (
        "bench/tools/test_gemm_sweep_tools.py",
        "044a873001af17fe2100e381feba0ad4488e93cbfc9f55970bf81d5d5061d320",
    ),
    (
        "bench/tools/test_level1_report.py",
        "00f94923e2bae67768c3d21dc619fd26c307175b14ad5fbfbeddef15283851f8",
    ),
    (
        "bench/tools/test_level2_report.py",
        "2ce852761cc7e9e6d74a9c324419c56d2db6abc5a349e9d524281b3ddf3697ee",
    ),
    (
        "bench/tools/test_rank_k_report.py",
        "834ffd3fd3d1bb578d44fbca5f8d4b3d497c23b590a083edee9f81d70774f0af",
    ),
    (
        "bench/tools/test_report_comparison.py",
        "29514c5ced81ec8ca8e5665ccbead524c84098d3f04ec045a41d182a59bc7250",
    ),
    (
        "bench/tools/test_report_plotters.py",
        "362b6339cd9c76f45b4729cbaa9d42499f01b0e9833a6890d7dff7fce90b4e4f",
    ),
    (
        "bench/tools/test_report_publication.py",
        "521d5237d9c5f99d87f07e7305becd09b98d43b4a94d6f5457c1b4483de4a01c",
    ),
    (
        "bench/tools/test_report_schedule.py",
        "4e008b60e2272bc8f09fd02676a164f5d25aa919dfd560a03f66530829ae2ae3",
    ),
    (
        "bench/tools/test_rotg_latency_report.py",
        "db967016f6201a8bd93861566301f093ac46bd74fae8f59b5526715ac8f5119d",
    ),
    (
        "bench/tools/test_symm_report.py",
        "0f6a981a87909ed07e75e5feba88134702a0577f5c6afd054921650fb8587f14",
    ),
    (
        "bench/tools/test_triangular_matrix_report.py",
        "50f818b9aa7ccef69c8881459c720a7e0206b9f779364e76b316cad540fb7f87",
    ),
    (
        "bench/tools/benchmark_artifacts.py",
        "9d0cdbd974c97564e549582ddf540d07968f66ddf9abe34273daa618426d287b",
    ),
    (
        "bench/tools/benchmark_metadata.py",
        "28eeac857246f31554f766c4a8e1e492bbdefb95d3ab4c421f9c5a2ab6cf222f",
    ),
    (
        "bench/tools/check_gemm_sweep.py",
        "14a53b9f0e3938e93bdcb37bbd819ea244474d74d274c4cfecb65227fdd5cbcd",
    ),
    (
        "bench/tools/check_level1_report.py",
        "81fc7446e099071a2278a772d64b5aa2c24bebb161c70f0b4fa8e7af10c60fb1",
    ),
    (
        "bench/tools/check_level2_report.py",
        "1e3a8c84edf2e605bff838118659f72aa5ba2a3118a572a5d63145f3bdc6702f",
    ),
    (
        "bench/tools/check_rank_k_report.py",
        "464f4b4542f7476e7a001b9069b866667a88d8a4a616a84505fa056d82ca1fae",
    ),
    (
        "bench/tools/check_rotg_latency_report.py",
        "c4713814b42839476ed380120f4fdd8e6c05d065837380848cf8215090e92a96",
    ),
    (
        "bench/tools/check_symm_report.py",
        "9a483b0ea55aa722ba52f898ac7fda3c8dc8385c742f9e4f947bb3a7f8c958d4",
    ),
    (
        "bench/tools/check_triangular_matrix_report.py",
        "934b0bc68602eb7a83de98b8faa4e7419aeeddba2f867fafbb27127ba292981d",
    ),
    (
        "bench/tools/plot_gemm_sweep.py",
        "f61aaed27ced6fe8eeb21b6b69ec20f9338c8f9489bcb3e80cd492991a4c1aee",
    ),
    (
        "bench/tools/plot_level1_report.py",
        "99d3765cc52c59d8a607b6d034bf469656336adb5e0a6468a9df7bf3c7531a4b",
    ),
    (
        "bench/tools/plot_level2_report.py",
        "9436916d466b2950ef923b110a384650e5032eecd50d6e254e7085bdd95a9c3d",
    ),
    (
        "bench/tools/render_full_benchmark_report.py",
        "4619d481895aa0bd6dae66f77279a9f8a6520c3ae94d76ebdab3edc1af928bac",
    ),
    (
        "bench/tools/report_comparison.py",
        "c2b658e116e079faa3064c83a2699b9983e7a260f5b9fee55afef35cff98baf7",
    ),
    (
        "bench/tools/report_publication.py",
        "602427b0a55522432ccbd7812b00fe9cf2eab9f8c868758500230a1477f5dd53",
    ),
    (
        "bench/tools/report_schedule.py",
        "29b78ce69ce08068e73913f53a04fa95dfbf7793eaffc3d5bc17f1ca503e0477",
    ),
    (
        "bench/tools/run_gemm_sweep_isolated.py",
        "cad1e4c83cced9493ca67fbd7e05b7c934088749413b2abc4f3e77a3087713b0",
    ),
    (
        "bench/tools/run_level1_report.py",
        "58b3ae68feef3b73e487f6d7d354d989cda50bfa7f5041c755fcce6c6253fcae",
    ),
    (
        "bench/tools/run_level2_report.py",
        "3a7d01a7a7eb1e457b01fa3cf69395e20c95252c33450a325bc6480d18ac35f3",
    ),
    (
        "bench/tools/run_rank_k_report.py",
        "e81385fc61863de059d47f81f7620487566853290a44e07f1831e4b59e5ce63a",
    ),
    (
        "bench/tools/run_rotg_latency_report.py",
        "18af91edceb521e787ef63a439c56c9fbc3a54f0b5f2384cbc64bd989a0f9b5f",
    ),
    (
        "bench/tools/run_symm_report.py",
        "5836bacbf83ec82dad45224c38d4390f0e044fe53f28d1dc8144a3cde8fbd019",
    ),
    (
        "bench/tools/run_triangular_matrix_report.py",
        "2f101f8cd50b035fecb215815b709d90d1956d3ee57eb4d7261acfc5c996a138",
    ),
    (
        "tools/repository_git.py",
        "0335896512a2ae707b1321a44f2e749fe015b04b9de6809816cc8ddd6a078cec",
    ),
    (
        "tools/repository_snapshot.py",
        "5ce200930baaa090170b65c393f04eab786e8c831eb72d5120c9344b465aff35",
    ),
)
_PYTHON_TOOLING_EXECUTION_MODULES = (
    (
        "test_benchmark_artifact_snapshot",
        "bench/tools/test_benchmark_artifact_snapshot.py",
    ),
    ("test_benchmark_metadata", "bench/tools/test_benchmark_metadata.py"),
    ("test_full_benchmark_report", "bench/tools/test_full_benchmark_report.py"),
    ("test_gemm_sweep_tools", "bench/tools/test_gemm_sweep_tools.py"),
    ("test_level1_report", "bench/tools/test_level1_report.py"),
    ("test_level2_report", "bench/tools/test_level2_report.py"),
    ("test_rank_k_report", "bench/tools/test_rank_k_report.py"),
    ("test_report_comparison", "bench/tools/test_report_comparison.py"),
    ("test_report_plotters", "bench/tools/test_report_plotters.py"),
    ("test_report_publication", "bench/tools/test_report_publication.py"),
    ("test_report_schedule", "bench/tools/test_report_schedule.py"),
    ("test_rotg_latency_report", "bench/tools/test_rotg_latency_report.py"),
    ("test_symm_report", "bench/tools/test_symm_report.py"),
    (
        "test_triangular_matrix_report",
        "bench/tools/test_triangular_matrix_report.py",
    ),
    ("benchmark_artifacts", "bench/tools/benchmark_artifacts.py"),
    ("benchmark_metadata", "bench/tools/benchmark_metadata.py"),
    ("check_gemm_sweep", "bench/tools/check_gemm_sweep.py"),
    ("check_level1_report", "bench/tools/check_level1_report.py"),
    ("check_level2_report", "bench/tools/check_level2_report.py"),
    ("check_rank_k_report", "bench/tools/check_rank_k_report.py"),
    ("check_rotg_latency_report", "bench/tools/check_rotg_latency_report.py"),
    ("check_symm_report", "bench/tools/check_symm_report.py"),
    ("check_triangular_matrix_report", "bench/tools/check_triangular_matrix_report.py"),
    ("plot_gemm_sweep", "bench/tools/plot_gemm_sweep.py"),
    ("plot_level1_report", "bench/tools/plot_level1_report.py"),
    ("plot_level2_report", "bench/tools/plot_level2_report.py"),
    (
        "render_full_benchmark_report",
        "bench/tools/render_full_benchmark_report.py",
    ),
    ("report_comparison", "bench/tools/report_comparison.py"),
    ("report_publication", "bench/tools/report_publication.py"),
    ("report_schedule", "bench/tools/report_schedule.py"),
    ("run_gemm_sweep_isolated", "bench/tools/run_gemm_sweep_isolated.py"),
    ("run_level1_report", "bench/tools/run_level1_report.py"),
    ("run_level2_report", "bench/tools/run_level2_report.py"),
    ("run_rank_k_report", "bench/tools/run_rank_k_report.py"),
    ("run_rotg_latency_report", "bench/tools/run_rotg_latency_report.py"),
    ("run_symm_report", "bench/tools/run_symm_report.py"),
    ("run_triangular_matrix_report", "bench/tools/run_triangular_matrix_report.py"),
    ("_zynum_repository_git", "tools/repository_git.py"),
    (
        "_zynum_benchmark_artifact_repository_snapshot",
        "tools/repository_snapshot.py",
    ),
    ("_zynum_benchmark_repository_snapshot", "tools/repository_snapshot.py"),
    ("_zynum_report_repository_snapshot", "tools/repository_snapshot.py"),
)
_PYTHON_TOOLING_EXECUTION_MANIFEST_SHA256 = (
    "21c39d3d89a73953a9d25ec9db95635c88a4a6ffb6e2a866a93a40b8ad887f04"
)
_PYTHON_TOOLING_RUNTIME_ORDER_SHA256 = (
    "fe47ceff1b1520d52339b694168560d4eafa6754b71349365a355ecaf1d6f5a6"
)
_PYTHON_TOOLING_CAPSULE_MAGIC = b"ZYNUM-PYTHON-CAPSULE-V1\0"
_PYTHON_TOOLING_CAPSULE_MAX_BYTES = 16 * 1024 * 1024
_PYTHON_TOOLING_TRUSTED_SUBPROCESS_RUN = subprocess.run
_PYTHON_TOOLING_TRUSTED_COMPILE = compile
_PYTHON_TOOLING_TRUSTED_EXEC = exec
_PYTHON_TOOLING_TRUSTED_SPEC_FROM_FILE_LOCATION = importlib.util.spec_from_file_location
_PYTHON_TOOLING_TRUSTED_BUILTIN_FIND_SPEC = (
    importlib.machinery.BuiltinImporter.find_spec
)
_PYTHON_TOOLING_TRUSTED_FROZEN_FIND_SPEC = importlib.machinery.FrozenImporter.find_spec
_PYTHON_TOOLING_TRUSTED_PATH_FIND_SPEC = importlib.machinery.PathFinder.find_spec
_PYTHON_TOOLING_BOOTSTRAP_SOURCE = (
    "import hashlib,importlib.util,os,stat,subprocess,sys,tempfile,types\n"
    "limit=16777216\n"
    "bootstrap_source=sys.orig_argv[sys.orig_argv.index('-c')+1]\n"
    "if hashlib.sha256(bootstrap_source.encode()).hexdigest()!=sys.argv[3]: raise SystemExit(120)\n"
    "descriptor=int(sys.argv[1])\n"
    "if os.name=='nt':\n"
    " import msvcrt\n"
    " descriptor=msvcrt.open_osfhandle(descriptor,os.O_RDONLY)\n"
    "with os.fdopen(descriptor,'rb',closefd=True) as stream:\n"
    " data=stream.read(limit+1)\n"
    "magic=b'ZYNUM-PYTHON-CAPSULE-V1\\0'\n"
    "if len(data)>limit or len(data)<len(magic)+32: raise SystemExit(121)\n"
    "if not data.startswith(magic): raise SystemExit(122)\n"
    "if hashlib.sha256(data[:-32]).digest()!=data[-32:]: raise SystemExit(123)\n"
    "if hashlib.sha256(data).hexdigest()!=sys.argv[2]: raise SystemExit(124)\n"
    "body=memoryview(data[:-32]);pos=len(magic)\n"
    "def frame():\n"
    " global pos\n"
    " if pos+8>len(body): raise SystemExit(125)\n"
    " size=int.from_bytes(body[pos:pos+8],'big');pos+=8\n"
    " if size>limit or pos+size>len(body): raise SystemExit(125)\n"
    " value=bytes(body[pos:pos+size]);pos+=size;return value\n"
    "root=frame().decode('utf-8')\n"
    "if not os.path.isabs(root): raise SystemExit(126)\n"
    "source_count=int.from_bytes(body[pos:pos+8],'big');pos+=8\n"
    "if source_count!=39: raise SystemExit(127)\n"
    "sources={}\n"
    "for _ in range(source_count):\n"
    " path=frame().decode('utf-8');digest=frame().decode('ascii');payload=frame()\n"
    " if path.startswith('/') or '..' in path.split('/') or path in sources: raise SystemExit(128)\n"
    " if hashlib.sha256(payload).hexdigest()!=digest: raise SystemExit(129)\n"
    " sources[path]=(digest,payload)\n"
    "module_count=int.from_bytes(body[pos:pos+8],'big');pos+=8\n"
    "if module_count!=41: raise SystemExit(130)\n"
    "modules={}\n"
    "for _ in range(module_count):\n"
    " name=frame().decode('utf-8');path=frame().decode('utf-8')\n"
    " if not name or name in modules or path not in sources: raise SystemExit(131)\n"
    " modules[name]=path\n"
    "if pos!=len(body): raise SystemExit(132)\n"
    "trusted_spec=importlib.util.spec_from_file_location\n"
    "def canonical(path): return os.path.join(root,*path.split('/'))\n"
    "def in_root(path):\n"
    " try: return os.path.commonpath((root,os.path.abspath(path)))==root or os.path.commonpath((root,os.path.realpath(path)))==root\n"
    " except (OSError,ValueError): return True\n"
    "class Loader:\n"
    " def __init__(self,name,path): self.name=name;self.path=path\n"
    " def create_module(self,spec): return None\n"
    " def exec_module(self,module):\n"
    "  filename=canonical(self.path);module.__file__=filename\n"
    "  exec(compile(sources[self.path][1],filename,'exec'),module.__dict__)\n"
    "  if self.name=='benchmark_artifacts': module._set_frozen_source_resolver(frozen_source)\n"
    "def frozen_spec(name):\n"
    " path=modules[name];return trusted_spec(name,canonical(path),loader=Loader(name,path))\n"
    "class Finder:\n"
    " def find_spec(self,name,path=None,target=None):\n"
    "  if name in modules:return frozen_spec(name)\n"
    "  return None\n"
    "def reviewed_spec(name,location,*args,**kwargs):\n"
    " lexical=os.path.abspath(os.fspath(location));resolved=os.path.realpath(lexical)\n"
    " if name in modules:\n"
    "  expected=canonical(modules[name])\n"
    "  if lexical!=expected or resolved!=os.path.realpath(expected): raise ImportError('reviewed module path changed')\n"
    "  return frozen_spec(name)\n"
    " if in_root(lexical) or in_root(resolved): raise ImportError('unreviewed repository import')\n"
    " return trusted_spec(name,location,*args,**kwargs)\n"
    "importlib.util.spec_from_file_location=reviewed_spec\n"
    "sys.path[:]=[item for item in sys.path if item and not in_root(item)]\n"
    "sys.meta_path.insert(0,Finder())\n"
    "trusted_run=subprocess.run\n"
    "allowed_targets={'run_level1_report','run_level2_report'}\n"
    "names_by_path={}\n"
    "for module_name,module_path in modules.items(): names_by_path.setdefault(module_path,[]).append(module_name)\n"
    "def target_name(script,child_cwd):\n"
    " script=os.fspath(script);absolute=os.path.abspath(script if os.path.isabs(script) else os.path.join(child_cwd,script));matched=None\n"
    " for candidate in sources:\n"
    "  if canonical(candidate)==absolute: matched=candidate;break\n"
    " if matched is None:\n"
    "  flags=os.O_RDONLY|getattr(os,'O_CLOEXEC',0)|getattr(os,'O_NOFOLLOW',0);handle=os.open(absolute,flags)\n"
    "  try:\n"
    "   before=os.fstat(handle)\n"
    "   if not stat.S_ISREG(before.st_mode) or before.st_size>limit: raise OSError('invalid controller')\n"
    "   chunks=[];remaining=limit+1\n"
    "   while remaining:\n"
    "    chunk=os.read(handle,min(65536,remaining))\n"
    "    if not chunk: break\n"
    "    chunks.append(chunk);remaining-=len(chunk)\n"
    "   payload=b''.join(chunks);after=os.fstat(handle)\n"
    "  finally: os.close(handle)\n"
    "  before_id=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_mode)\n"
    "  after_id=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_mode)\n"
    "  if before_id!=after_id or len(payload)!=after.st_size or len(payload)>limit: raise OSError('controller changed')\n"
    "  matches=[path for path,(digest,data_bytes) in sources.items() if data_bytes==payload and hashlib.sha256(payload).hexdigest()==digest]\n"
    "  if len(matches)!=1: raise OSError('controller is not uniquely frozen')\n"
    "  matched=matches[0]\n"
    " admitted=[name for name in names_by_path[matched] if name in allowed_targets]\n"
    " if len(admitted)!=1: raise OSError('controller target is not admitted')\n"
    " return admitted[0]\n"
    "def capsule_run(args,*positional,**kwargs):\n"
    " if positional or kwargs.get('shell') or not isinstance(args,(list,tuple)) or len(args)<2: raise OSError('noncanonical subprocess invocation')\n"
    " if 'executable' in kwargs: raise OSError('capsule transport is checker-owned')\n"
    " original=args;argv=[os.fspath(item) for item in args]\n"
    " if os.path.realpath(argv[0])!=os.path.realpath(sys.executable): return trusted_run(args,**kwargs)\n"
    " if argv[1].startswith('-'): raise OSError('raw interpreter mode is forbidden')\n"
    " owned={'close_fds','creationflags','pass_fds','preexec_fn','startupinfo'}\n"
    " if owned.intersection(kwargs): raise OSError('capsule transport is checker-owned')\n"
    " child_cwd=os.path.abspath(os.fspath(kwargs.get('cwd',os.getcwd())));child_target=target_name(argv[1],child_cwd);check=bool(kwargs.pop('check',False))\n"
    " with tempfile.TemporaryFile(mode='w+b') as transport:\n"
    "  transport.write(data);transport.flush();transport.seek(0)\n"
    "  child_descriptor=transport.fileno();startup={}\n"
    "  if os.name=='nt':\n"
    "   child_descriptor=msvcrt.get_osfhandle(transport.fileno());os.set_handle_inheritable(child_descriptor,True)\n"
    "   info=subprocess.STARTUPINFO();info.lpAttributeList={'handle_list':[child_descriptor]};startup={'close_fds':True,'startupinfo':info}\n"
    "  else: startup={'pass_fds':(transport.fileno(),)}\n"
    "  command=[sys.executable,'-I','-S','-B','-c',bootstrap_source,str(child_descriptor),hashlib.sha256(data).hexdigest(),hashlib.sha256(bootstrap_source.encode()).hexdigest(),'run',child_target,*argv[2:]]\n"
    "  try: result=trusted_run(command,check=False,**startup,**kwargs)\n"
    "  except subprocess.TimeoutExpired as error: error.cmd=original;raise\n"
    "  finally:\n"
    "   if os.name=='nt': os.set_handle_inheritable(child_descriptor,False)\n"
    " result.args=original\n"
    " if check: result.check_returncode()\n"
    " return result\n"
    "subprocess.run=capsule_run\n"
    "def frozen_source(public_path):\n"
    " absolute=os.path.abspath(os.fspath(public_path))\n"
    " matches=[path for path in sources if canonical(path)==absolute]\n"
    " if len(matches)!=1: raise OSError('frozen source path is not admitted')\n"
    " path=matches[0];return canonical(path),sources[path][1],sources[path][0]\n"
    "mode=sys.argv[4];target=sys.argv[5]\n"
    "if mode not in ('nested-probe','probe','run') or target not in modules: raise SystemExit(133)\n"
    "if mode=='nested-probe':\n"
    " private_descriptor,private_path=tempfile.mkstemp(suffix='.py')\n"
    " try:\n"
    "  payload=sources[modules[target]][1];written=0\n"
    "  while written<len(payload): written+=os.write(private_descriptor,payload[written:])\n"
    "  os.close(private_descriptor);private_descriptor=-1\n"
    "  original=[sys.executable,private_path,'--help'];nested=capsule_run(original,capture_output=True,text=True,check=False)\n"
    "  if nested.returncode or nested.args is not original or not nested.stdout.startswith('usage:'): raise SystemExit(136)\n"
    "  sys.stdout.write('zynum-capsule-nested-ok\\n');raise SystemExit(0)\n"
    " finally:\n"
    "  if private_descriptor!=-1: os.close(private_descriptor)\n"
    "  try: os.unlink(private_path)\n"
    "  except FileNotFoundError: pass\n"
    "target_path=modules[target];filename=canonical(target_path);target_args=sys.argv[6:]\n"
    "sys.argv=[filename,*target_args];sys.orig_argv=[sys.executable,filename,*target_args]\n"
    "main=types.ModuleType('__main__');main.__file__=filename;main.__loader__=Loader('__main__',target_path);main.__package__=None;main.__spec__=None\n"
    "sys.modules['__main__']=main\n"
    "try: exec(compile(sources[target_path][1],filename,'exec'),main.__dict__)\n"
    "finally:\n"
    " if subprocess.run is not capsule_run: raise SystemExit(134)\n"
    " artifact=sys.modules.get('benchmark_artifacts')\n"
    " if artifact is not None and getattr(artifact,'_FROZEN_SOURCE_RESOLVER',None) is not frozen_source: raise SystemExit(135)\n"
    "if mode=='probe': sys.stdout.write('zynum-capsule-target-ok|'+main.__name__+'|'+main.__file__+'|'+repr(sys.argv)+'|'+repr(sys.orig_argv)+'\\n')\n"
)
_PYTHON_TOOLING_BOOTSTRAP_SHA256 = (
    "83b807444228d772b60a7b1c4b140d356d8b580fb7842fc01cf99490573b033e"
)


def _python_tooling_posix_capsule_probe(
    capsule: bytes, *, nested: bool = False
) -> subprocess.CompletedProcess[str]:
    if os.name != "posix":
        raise InventoryError(
            "Python tooling capsule transport requires explicit Windows handles"
        )
    if (
        hashlib.sha256(_PYTHON_TOOLING_BOOTSTRAP_SOURCE.encode()).hexdigest()
        != _PYTHON_TOOLING_BOOTSTRAP_SHA256
    ):
        raise InventoryError("Python tooling reviewed bootstrap changed")
    with tempfile.TemporaryFile(mode="w+b") as transport:
        transport.write(capsule)
        transport.flush()
        transport.seek(0)
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _PYTHON_TOOLING_BOOTSTRAP_SOURCE,
                str(transport.fileno()),
                hashlib.sha256(capsule).hexdigest(),
                _PYTHON_TOOLING_BOOTSTRAP_SHA256,
                "nested-probe" if nested else "probe",
                "run_level2_report" if nested else "report_schedule",
                *(() if nested else ("semantic-argument",)),
            ],
            check=False,
            capture_output=True,
            text=True,
            pass_fds=(transport.fileno(),),
        )


_PYTHON_TOOLING_CAPSULE_LAUNCH_TARGETS = frozenset(
    {"run_level1_report", "run_level2_report"}
)


def _python_tooling_windows_handle_runtime() -> Any:
    import msvcrt

    return msvcrt


class _PythonToolingCapsuleLauncher:
    def __init__(self, closure: "_PythonExecutionClosure", capsule: bytes) -> None:
        closure.verify(require_complete=True)
        self._closure = closure
        self._capsule = capsule
        self._capsule_sha256 = hashlib.sha256(capsule).hexdigest()
        self._run_wrapper = self._run
        self._resolver_wrapper = self._resolve_frozen_source
        self._benchmark_module: types.ModuleType | None = None
        self._source_by_path = {
            str(source.source_path): source for source in closure.sources.values()
        }
        self._names_by_path: dict[str, list[str]] = {}
        for name, path in closure.module_paths.items():
            self._names_by_path.setdefault(path, []).append(name)

    @property
    def run(self) -> Any:
        return self._run_wrapper

    @property
    def frozen_source_resolver(self) -> Any:
        return self._resolver_wrapper

    def _resolve_frozen_source(self, public_path: str) -> tuple[str, bytes, str]:
        source = self._source_by_path.get(os.path.abspath(public_path))
        if source is None:
            raise InventoryError("Python tooling frozen source path is not admitted")
        return str(source.source_path), source.source_bytes, source.source_sha256

    def _target_module(self, script: str, child_cwd: str) -> str:
        absolute = os.path.abspath(
            script if os.path.isabs(script) else os.path.join(child_cwd, script)
        )
        source = self._source_by_path.get(absolute)
        if source is None:
            snapshot = _read_regular_stable_snapshot(
                absolute,
                MAX_INVENTORY_BYTES,
                "Python tooling private controller",
            )
            matches = [
                candidate
                for candidate in self._closure.sources.values()
                if candidate.source_sha256 == snapshot.sha256
                and candidate.source_bytes == snapshot.bytes
            ]
            if len(matches) != 1:
                raise InventoryError(
                    "Python tooling private controller has no unique frozen source"
                )
            source = matches[0]
        names = self._names_by_path.get(source.inventory_path, [])
        admitted = [
            name for name in names if name in _PYTHON_TOOLING_CAPSULE_LAUNCH_TARGETS
        ]
        if len(admitted) != 1:
            raise InventoryError("Python tooling subprocess target is not admitted")
        return admitted[0]

    def _run(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        if positional or kwargs.get("shell"):
            raise InventoryError("Python tooling subprocess invocation is noncanonical")
        if "executable" in kwargs:
            raise InventoryError("Python tooling capsule transport is checker-owned")
        if (
            not isinstance(args, (list, tuple))
            or len(args) < 2
            or not all(isinstance(item, (str, os.PathLike)) for item in args)
        ):
            raise InventoryError("Python tooling subprocess argv is noncanonical")
        original_args = args
        argv = [os.fspath(item) for item in args]
        if os.path.realpath(argv[0]) != os.path.realpath(sys.executable):
            return _PYTHON_TOOLING_TRUSTED_SUBPROCESS_RUN(args, **kwargs)
        if argv[1] in {"-c", "-m"} or argv[1].startswith("-"):
            raise InventoryError("Python tooling raw interpreter mode is forbidden")
        transport_owned = {
            "close_fds",
            "creationflags",
            "pass_fds",
            "preexec_fn",
            "startupinfo",
        }
        if transport_owned.intersection(kwargs):
            raise InventoryError("Python tooling capsule transport is checker-owned")
        child_cwd = os.path.abspath(os.fspath(kwargs.get("cwd", os.getcwd())))
        target = self._target_module(argv[1], child_cwd)
        check = bool(kwargs.pop("check", False))
        with tempfile.TemporaryFile(mode="w+b") as transport:
            transport.write(self._capsule)
            transport.flush()
            transport.seek(0)
            command = [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _PYTHON_TOOLING_BOOTSTRAP_SOURCE,
                str(transport.fileno()),
                self._capsule_sha256,
                _PYTHON_TOOLING_BOOTSTRAP_SHA256,
                "run",
                target,
                *argv[2:],
            ]
            try:
                if os.name == "posix":
                    result = _PYTHON_TOOLING_TRUSTED_SUBPROCESS_RUN(
                        command,
                        check=False,
                        pass_fds=(transport.fileno(),),
                        **kwargs,
                    )
                elif os.name == "nt":
                    msvcrt = _python_tooling_windows_handle_runtime()
                    inherited_handle = msvcrt.get_osfhandle(transport.fileno())
                    os.set_handle_inheritable(inherited_handle, True)
                    startup_info = subprocess.STARTUPINFO()
                    startup_info.lpAttributeList = {"handle_list": [inherited_handle]}
                    command[6] = str(inherited_handle)
                    try:
                        result = _PYTHON_TOOLING_TRUSTED_SUBPROCESS_RUN(
                            command,
                            check=False,
                            close_fds=True,
                            startupinfo=startup_info,
                            **kwargs,
                        )
                    finally:
                        os.set_handle_inheritable(inherited_handle, False)
                else:
                    raise InventoryError(
                        "Python tooling capsule transport is unsupported"
                    )
            except subprocess.TimeoutExpired as exc:
                exc.cmd = original_args
                raise
        result.args = original_args
        if check:
            result.check_returncode()
        return result

    def verify(self) -> None:
        if subprocess.run is not self._run_wrapper:
            raise InventoryError("Python tooling subprocess launcher changed")
        if (
            self._benchmark_module is None
            or getattr(self._benchmark_module, "_FROZEN_SOURCE_RESOLVER", None)
            is not self._resolver_wrapper
        ):
            raise InventoryError("Python tooling artifact resolver changed")
        self._closure.verify(require_complete=True)


@contextlib.contextmanager
def _python_tooling_capsule_runtime(
    closure: "_PythonExecutionClosure", capsule: bytes
) -> Iterator[_PythonToolingCapsuleLauncher]:
    launcher = _PythonToolingCapsuleLauncher(closure, capsule)
    benchmark = closure.instances.get("benchmark_artifacts")
    if benchmark is None:
        raise InventoryError("Python tooling artifact authority is missing")
    setter = getattr(benchmark.module, "_set_frozen_source_resolver", None)
    if not callable(setter):
        raise InventoryError("Python tooling artifact resolver seam is missing")
    previous_run = subprocess.run
    previous_resolver = getattr(benchmark.module, "_FROZEN_SOURCE_RESOLVER", None)
    if previous_resolver is not None:
        raise InventoryError("Python tooling artifact resolver was already installed")
    launcher._benchmark_module = benchmark.module
    try:
        setter(launcher.frozen_source_resolver)
    except BaseException as exc:
        try:
            setter(previous_resolver)
        except BaseException as restore_exc:
            raise InventoryError(
                "Python tooling artifact resolver install rollback failed"
            ) from restore_exc
        raise InventoryError("Python tooling artifact resolver install failed") from exc
    setattr(subprocess, "run", launcher.run)
    try:
        launcher.verify()
        yield launcher
        launcher.verify()
    finally:
        setattr(subprocess, "run", previous_run)
        try:
            setter(previous_resolver)
        except BaseException as exc:
            if subprocess.run is not previous_run:
                raise InventoryError(
                    "Python tooling subprocess launcher restore failed"
                ) from exc
            raise InventoryError(
                "Python tooling artifact resolver restore failed"
            ) from exc


def _python_tooling_capsule_frame(payload: bytes) -> bytes:
    if type(payload) is not bytes or len(payload) > _PYTHON_TOOLING_CAPSULE_MAX_BYTES:
        raise InventoryError("Python tooling capsule field is noncanonical")
    return len(payload).to_bytes(8, "big") + payload


def _python_tooling_execution_capsule(
    closure: "_PythonExecutionClosure",
) -> bytes:
    closure.verify()
    chunks = [_PYTHON_TOOLING_CAPSULE_MAGIC]
    chunks.append(_python_tooling_capsule_frame(os.fspath(closure.root).encode()))
    chunks.append(len(_PYTHON_TOOLING_EXECUTION_SOURCE_SHA256).to_bytes(8, "big"))
    for path, digest in _PYTHON_TOOLING_EXECUTION_SOURCE_SHA256:
        source = closure.sources[path]
        chunks.extend(
            (
                _python_tooling_capsule_frame(path.encode()),
                _python_tooling_capsule_frame(digest.encode("ascii")),
                _python_tooling_capsule_frame(source.source_bytes),
            )
        )
    chunks.append(len(_PYTHON_TOOLING_EXECUTION_MODULES).to_bytes(8, "big"))
    for name, path in _PYTHON_TOOLING_EXECUTION_MODULES:
        chunks.extend(
            (
                _python_tooling_capsule_frame(name.encode()),
                _python_tooling_capsule_frame(path.encode()),
            )
        )
    body = b"".join(chunks)
    capsule = body + hashlib.sha256(body).digest()
    if len(capsule) > _PYTHON_TOOLING_CAPSULE_MAX_BYTES:
        raise InventoryError("Python tooling execution capsule exceeds its bound")
    return capsule


def _decode_python_tooling_execution_capsule(
    capsule: bytes,
) -> tuple[str, tuple[tuple[str, str, bytes], ...], tuple[tuple[str, str], ...]]:
    if (
        type(capsule) is not bytes
        or len(capsule) > _PYTHON_TOOLING_CAPSULE_MAX_BYTES
        or len(capsule) < len(_PYTHON_TOOLING_CAPSULE_MAGIC) + 32
        or not capsule.startswith(_PYTHON_TOOLING_CAPSULE_MAGIC)
        or hashlib.sha256(capsule[:-32]).digest() != capsule[-32:]
    ):
        raise InventoryError("Python tooling execution capsule is invalid")
    body = memoryview(capsule[:-32])
    cursor = len(_PYTHON_TOOLING_CAPSULE_MAGIC)

    def frame() -> bytes:
        nonlocal cursor
        if cursor + 8 > len(body):
            raise InventoryError("Python tooling execution capsule is truncated")
        length = int.from_bytes(body[cursor : cursor + 8], "big")
        cursor += 8
        if length > _PYTHON_TOOLING_CAPSULE_MAX_BYTES or cursor + length > len(body):
            raise InventoryError("Python tooling execution capsule is truncated")
        value = bytes(body[cursor : cursor + length])
        cursor += length
        return value

    try:
        root = frame().decode()
        source_count = int.from_bytes(body[cursor : cursor + 8], "big")
        cursor += 8
        sources = tuple(
            (frame().decode(), frame().decode("ascii"), frame())
            for _ in range(source_count)
        )
        module_count = int.from_bytes(body[cursor : cursor + 8], "big")
        cursor += 8
        modules = tuple(
            (frame().decode(), frame().decode()) for _ in range(module_count)
        )
    except (UnicodeError, ValueError) as exc:
        raise InventoryError("Python tooling execution capsule is invalid") from exc
    if (
        cursor != len(body)
        or source_count != 39
        or module_count != 41
        or tuple((path, digest) for path, digest, _ in sources)
        != _PYTHON_TOOLING_EXECUTION_SOURCE_SHA256
        or modules != _PYTHON_TOOLING_EXECUTION_MODULES
        or any(
            hashlib.sha256(data).hexdigest() != digest for _, digest, data in sources
        )
        or len({path for path, _, _ in sources}) != 39
        or len({name for name, _ in modules}) != 41
        or any(
            PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
            for path, _, _ in sources
        )
    ):
        raise InventoryError("Python tooling execution capsule is noncanonical")
    return root, sources, modules


WINDOWS_PYTHON_TOOLING_FIXTURE_PATHS = (
    "zig-out/bin/rank-k-probe.exe",
    "zig-out/bin/rotg-latency-probe.exe",
    "zig-out/bin/symm-probe.exe",
    "zig-out/bin/triangular-matrix-probe.exe",
)
WINDOWS_PYTHON_TOOLING_BLAS_PATH = "zig-out/bin/zynum_blas.dll"
WINDOWS_PYTHON_TOOLING_BLAS_WINMODE = 0x00000900
WINDOWS_PYTHON_TOOLING_BLAS_REQUIRED_SYMBOLS = (
    "sgemv_",
    "dgemv_",
    "cgemv_",
    "zgemv_",
    "sger_",
    "dger_",
    "cgeru_",
    "cgerc_",
    "zgeru_",
    "zgerc_",
    "strmv_",
    "dtrmv_",
    "ctrmv_",
    "ztrmv_",
    "strsv_",
    "dtrsv_",
    "ctrsv_",
    "ztrsv_",
    "ssyr_",
    "dsyr_",
    "cher_",
    "zher_",
    "ssyr2_",
    "dsyr2_",
    "cher2_",
    "zher2_",
    "sgbmv_",
    "dgbmv_",
    "cgbmv_",
    "zgbmv_",
    "ssbmv_",
    "dsbmv_",
    "chbmv_",
    "zhbmv_",
    "sspmv_",
    "dspmv_",
    "chpmv_",
    "zhpmv_",
    "stpmv_",
    "dtpmv_",
    "ctpmv_",
    "ztpmv_",
    "stpsv_",
    "dtpsv_",
    "ctpsv_",
    "ztpsv_",
    "sspr_",
    "dspr_",
    "chpr_",
    "zhpr_",
    "sspr2_",
    "dspr2_",
    "chpr2_",
    "zhpr2_",
    "stbmv_",
    "dtbmv_",
    "ctbmv_",
    "ztbmv_",
    "stbsv_",
    "dtbsv_",
    "ctbsv_",
    "ztbsv_",
)
WINDOWS_PYTHON_TOOLING_EXPECTED_NON_PLATFORM_SKIPS = 5
WINDOWS_PYTHON_TOOLING_EXPECTED_TOTAL_SKIPS = 98
MAX_WINDOWS_PYTHON_TOOLING_DLL_BYTES = 512 * 1024 * 1024
FROZEN_STATE = "frozen-compiler-enumeration"
PENDING_STATE = "requires-native-enumeration"
MAX_PROTOCOL_BYTES = 4 * 1024 * 1024
MAX_INVENTORY_BYTES = 4 * 1024 * 1024
MAX_PROTOCOL_BLOCKS = 64
MAX_PROTOCOL_TESTS_PER_BLOCK = 4096
MAX_PROTOCOL_TOTAL_TESTS = MAX_PROTOCOL_BLOCKS * MAX_PROTOCOL_TESTS_PER_BLOCK
MAX_PROTOCOL_LINE_BYTES = 64 * 1024
MAX_PROTOCOL_VALUE_BYTES = 16 * 1024
MAX_JSON_DEPTH = 128
MAX_JSON_NODES = 262_144
NATIVE_PROJECTION_SCHEMA_ID = "zynum-reviewed-native-test-projection-v1"
NATIVE_PROJECTION_SCHEMA_VERSION = 1
CURRENT_TEST_INVENTORY_SHA256 = (
    "95873919040e797481897d9711b4dfb11a2277d88e8cc6499de50f65323a7470"
)
NEXT_TEST_INVENTORY_SHA256: str | None = None
CURRENT_NATIVE_PROJECTION_SHA256 = (
    "26d980fc23a6ee4b45e1f3a2fe11cdf9fbecf7a974ec04e683c8a89bdd677bd2"
)
NEXT_NATIVE_PROJECTION_SHA256: str | None = None
TOP_LEVEL_KEYS = {
    "schema_id",
    "schema_version",
    "build_inventory_schema_id",
    "optimize_modes",
    "predicates",
    "environment_profiles",
    "test_enumeration_classes",
    "test_roots",
    "zig_test_files",
    "python_test_modules",
    "python_skip_contracts",
    "expected_test_sets",
    "native_observation_bindings",
    "test_mode_rows",
    "workflow_mode_bindings",
    "known_gaps",
    "matrix_row_contract",
    "strict_summary",
}
PYTHON_SKIP_PREDICATE_IDS = frozenset(
    {
        "python-skip-predicate:accelerate-unavailable",
        "python-skip-predicate:drop-in-blas-unavailable",
        "python-skip-predicate:file-backed-blas-unavailable",
        "python-skip-predicate:rank-k-artifacts-unavailable",
        "python-skip-predicate:rotg-latency-artifacts-unavailable",
        "python-skip-predicate:symm-artifacts-unavailable",
        "python-skip-predicate:triangular-matrix-artifacts-unavailable",
        "python-skip-predicate:no-alternate-supplementary-group",
        "python-skip-predicate:no-setgid-inheritance",
        "python-skip-predicate:not-darwin",
        "python-skip-predicate:no-automatic-provenance-xattr",
        "python-skip-predicate:case-distinct-names",
        "python-skip-predicate:normalization-distinct-names",
        "python-skip-predicate:case-aliasing-filesystem",
        "python-skip-predicate:artifact-snapshot-platform-unavailable",
        "python-skip-predicate:report-publication-platform-unavailable",
    }
)
REPORT_PUBLICATION_DYNAMIC_SKIP_PREDICATE_IDS = (
    "python-skip-predicate:no-alternate-supplementary-group",
    "python-skip-predicate:no-setgid-inheritance",
    "python-skip-predicate:no-automatic-provenance-xattr",
    "python-skip-predicate:case-distinct-names",
    "python-skip-predicate:normalization-distinct-names",
    "python-skip-predicate:case-aliasing-filesystem",
)
REPORT_PUBLICATION_SUBORDINATE_SKIP_PREDICATE_IDS = (
    *REPORT_PUBLICATION_DYNAMIC_SKIP_PREDICATE_IDS,
    "python-skip-predicate:not-darwin",
)
PYTHON_DECORATOR_SKIP_PREDICATE_IDS = frozenset(
    {
        "python-skip-predicate:accelerate-unavailable",
        "python-skip-predicate:drop-in-blas-unavailable",
        "python-skip-predicate:file-backed-blas-unavailable",
        "python-skip-predicate:rank-k-artifacts-unavailable",
        "python-skip-predicate:rotg-latency-artifacts-unavailable",
        "python-skip-predicate:symm-artifacts-unavailable",
        "python-skip-predicate:triangular-matrix-artifacts-unavailable",
        "python-skip-predicate:not-darwin",
    }
)
PYTHON_INVENTORY_PLATFORM_SKIP_KIND = "inventory-platform-applicability"
PYTHON_INVENTORY_PLATFORM_PREDICATE_IDS = frozenset(
    {
        "python-skip-predicate:artifact-snapshot-platform-unavailable",
        "python-skip-predicate:report-publication-platform-unavailable",
    }
)
PYTHON_INVENTORY_PLATFORM_COUNTS = {
    "python-skip-predicate:artifact-snapshot-platform-unavailable": 33,
    "python-skip-predicate:report-publication-platform-unavailable": 60,
}
PYTHON_INVENTORY_PLATFORM_REASONS = {
    "python-skip-predicate:artifact-snapshot-platform-unavailable": (
        "POSIX artifact snapshot APIs are unavailable"
    ),
    "python-skip-predicate:report-publication-platform-unavailable": (
        "POSIX report publication APIs are unavailable"
    ),
}
PYTHON_SKIP_PREDICATE_SOURCE_BINDINGS = {
    "python-skip-predicate:accelerate-unavailable": (
        "unittest.skipUnless",
        "7db4bcf8bef7d89e9fe37c689c8f3a6d522cd4f32e2cc4a5a35b3013a57b1479",
    ),
    "python-skip-predicate:drop-in-blas-unavailable": (
        "unittest.skipUnless",
        "b991f8b98b69767c9ee85ab52863e2cd2a8c47fcbebe38b1bff4d9885e9a0f44",
    ),
    "python-skip-predicate:file-backed-blas-unavailable": (
        "unittest.skipUnless",
        "03e7e4adc6c9162c7b90d7fed2a193e06eac27419e7fd40d855154bb4428c6b3",
    ),
    "python-skip-predicate:rank-k-artifacts-unavailable": (
        "unittest.skipUnless",
        "9c5a795bb6b9b408dc6989c6f7648c45fd6cef1a23fd834264ab2cb0a174e70e",
    ),
    "python-skip-predicate:rotg-latency-artifacts-unavailable": (
        "unittest.skipUnless",
        "0528009144f89189929391cf810b02fb1286dd1a057daf5be6f276e3c44711e1",
    ),
    "python-skip-predicate:symm-artifacts-unavailable": (
        "unittest.skipUnless",
        "36810485020e7b5f68dc738e3f48d0be9a35045838d5ee05160be90f2a91f4f2",
    ),
    "python-skip-predicate:triangular-matrix-artifacts-unavailable": (
        "unittest.skipUnless",
        "b238b284b4f18337c82c42cc05309dd82aece3dd39df78527fe4064079252567",
    ),
    "python-skip-predicate:no-alternate-supplementary-group": (
        "self.skipTest-if",
        "153c54176bb2b7a2fab46a16b1317f9eeb7491ab0d80b8a00a1611f40ea762f5",
    ),
    "python-skip-predicate:no-setgid-inheritance": (
        "self.skipTest-if",
        "25a61701139091a44f2496b4bf51ec893b886618d42fdca04b637bb3f0174917",
    ),
    "python-skip-predicate:not-darwin": (
        "unittest.skipUnless",
        "dafa642a93068ffbb7fb0cb3c682a8bd75f121e9716afa4dc17b4bbce9819633",
    ),
    "python-skip-predicate:no-automatic-provenance-xattr": (
        "self.skipTest-if",
        "15454dd5e64b75d2a60a095b1576ddee26238a629cc35255aa5adeef9040d352",
    ),
    "python-skip-predicate:case-distinct-names": (
        "self.skipTest-if",
        "571abac73b8f84ec91cdfae5ff80c6047a97a757ca22d779369482258343ba67",
    ),
    "python-skip-predicate:normalization-distinct-names": (
        "self.skipTest-if",
        "b85a4f3d06b5696b07a8519be4e641b0004de13bf798f00d52a5aa37ba0f4012",
    ),
    "python-skip-predicate:case-aliasing-filesystem": (
        "self.skipTest-if",
        "b68aa26809c517630acb54d79fa5a46e4cfbb048fd703c498a8f5228fcdb0b69",
    ),
    "python-skip-predicate:artifact-snapshot-platform-unavailable": (
        PYTHON_INVENTORY_PLATFORM_SKIP_KIND,
        "11e53ac6b1342da5810467bcbb4074266c130f91893fbfc5bf1f95c4ca2478fc",
    ),
    "python-skip-predicate:report-publication-platform-unavailable": (
        PYTHON_INVENTORY_PLATFORM_SKIP_KIND,
        "bbe4c69d96a7c7fe9367432e1cba875b97dffee7bc7ae2fcac10dc10def0a1ce",
    ),
}
if set(PYTHON_SKIP_PREDICATE_SOURCE_BINDINGS) != set(PYTHON_SKIP_PREDICATE_IDS):
    raise RuntimeError("Python skip predicate source bindings are incomplete")

ROOT_VARIANTS = {
    "blas_public_surface_contract_tests": "blas-module",
    "zynum_public_surface_contract_tests": "top-level",
}
ROOT_ENTRY_PATHS = {
    "modern_tests": ("src/zynum.zig",),
    "blas_module_tests": ("src/blas.zig",),
    "zynum_public_surface_contract_tests": ("src/zynum.zig",),
    "blas_public_surface_contract_tests": ("src/blas.zig",),
    "fortran_tests": ("src/blas/compat_fortran.zig", "src/blas.zig"),
    "cblas_tests": ("src/blas/compat_cblas.zig",),
}
ROOT_MODULE_SYMBOLS = {
    "modern_tests": "zynum_test_mod",
    "blas_module_tests": "zynum_blas_test_mod",
    "fortran_tests": "fortran_compat_test_mod",
    "cblas_tests": "cblas_compat_test_mod",
    "structured_object_tests": "structured_object_test_mod",
    "triangular_packed_unit_tests": "triangular_packed_unit_test_mod",
    "triangular_band_solve_tests": "triangular_band_solve_test_mod",
    "vector_stride2_parallel_tests": "vector_stride2_parallel_test_mod",
}
PYTHON_ROOTS = (
    {
        "id": "python-root:abi-baseline-discovery",
        "kind": "discovery",
        "module_paths": (
            "test/abi/baseline/test_abi_artifact_parity.py",
            "test/abi/baseline/test_observe_abi_baseline.py",
            "test/abi/baseline/test_package_archive.py",
        ),
        "launch_ids": ("launch:build.zig:build:abi_baseline_observer_tests",),
        "aggregate": True,
        "matrix": True,
        "discovery_start": "test/abi/baseline",
        "discovery_pattern": "test_*.py",
    },
    {
        "id": "python-root:abi-artifact-parity-direct",
        "kind": "direct",
        "module_paths": ("test/abi/baseline/test_abi_artifact_parity.py",),
        "launch_ids": ("launch:build.zig:build:abi_artifact_parity_verifier_tests",),
        "aggregate": False,
        "matrix": False,
    },
    {
        "id": "python-root:build-inventory-direct",
        "kind": "direct",
        "module_paths": ("test/build/test_build_inventory.py",),
        "launch_ids": ("launch:build.zig:build:build_inventory_tests",),
        "aggregate": False,
        "matrix": True,
    },
    {
        "id": PYTHON_TOOLING_ROOT_ID,
        "kind": "discovery",
        "module_paths": (),
        "launch_ids": (PYTHON_TOOLING_LAUNCH_ID,),
        "aggregate": True,
        "matrix": False,
        "discovery_start": "bench/tools",
        "discovery_pattern": "test_*.py",
    },
    {
        "id": "python-root:test-inventory-direct",
        "kind": "direct",
        "module_paths": (SELF_TEST_PATH,),
        "launch_ids": (),
        "aggregate": False,
        "matrix": False,
    },
)
ENVIRONMENTS = (
    {
        "id": "env:aarch64-macos-baseline",
        "target": "aarch64-macos",
        "architecture": "aarch64",
        "os": "macos",
        "libc": "system",
        "cpu": "baseline",
        "resolved_cpu_model": "apple_m1",
        "cpu_feature_policy": "canonical-baseline-resolved-features",
        "host_tool_smoke": True,
    },
    {
        "id": "env:x86-64-linux-gnu-baseline",
        "target": "x86_64-linux-gnu",
        "architecture": "x86_64",
        "os": "linux",
        "libc": "gnu",
        "cpu": "baseline",
        "resolved_cpu_model": "x86_64",
        "cpu_feature_policy": "canonical-baseline-resolved-features",
        "host_tool_smoke": True,
    },
    {
        "id": "env:aarch64-linux-gnu-baseline",
        "target": "aarch64-linux-gnu",
        "architecture": "aarch64",
        "os": "linux",
        "libc": "gnu",
        "cpu": "baseline",
        "resolved_cpu_model": "generic",
        "cpu_feature_policy": "canonical-baseline-resolved-features",
        "host_tool_smoke": True,
    },
    {
        "id": "env:x86-64-windows-gnu-baseline",
        "target": "x86_64-windows-gnu",
        "architecture": "x86_64",
        "os": "windows",
        "libc": "gnu",
        "cpu": "baseline",
        "resolved_cpu_model": "x86_64",
        "cpu_feature_policy": "canonical-baseline-resolved-features",
        "host_tool_smoke": False,
    },
)


class InventoryError(Exception):
    """Deterministic validation failure."""


class InventoryPublicationIndeterminate(InventoryError):
    """The candidate was installed, but directory durability is uncertain."""


class FrozenInventorySnapshot(NamedTuple):
    """One bounded inventory image and its descriptor-derived identity."""

    bytes: bytes
    identity: tuple[int, int, int, int, int]
    sha256: str
    mode: int


class RefreshedInventoryCandidate(NamedTuple):
    """Validated refresh result ready for compare-and-replace publication."""

    inventory: dict[str, Any]
    bytes: bytes
    expected_snapshot: FrozenInventorySnapshot
    incomplete_count: int


class _ProtocolTotals:
    __slots__ = ("bytes", "blocks", "tests")

    def __init__(self) -> None:
        self.bytes = 0
        self.blocks = 0
        self.tests = 0


class _PythonToolingSuiteContract(NamedTuple):
    discovered_count: int
    required_decorator_skips: frozenset[tuple[str, str]]
    permitted_dynamic_skips: frozenset[tuple[str, str]]
    dynamic_skip_authorizations: tuple["_PythonDynamicSkipAuthorization", ...]
    platform_skips: frozenset[tuple[str, str]]
    platform_skip_authorizations: tuple["_PythonPlatformSkipAuthorization", ...]
    discovered_test_bindings: tuple["_PythonTestBinding", ...]
    runtime_integrity_callback: Any
    runtime_order: tuple[str, ...]


_PythonToolingSuiteContract.__new__.__defaults__ = (None, ())


class _PythonSkipSourceFact(NamedTuple):
    runtime_id: str
    reason: str
    skip_kind: str
    predicate_ast_sha256: str


class _PythonDynamicSkipSite(NamedTuple):
    runtime_id: str
    reason: str
    source_path: Path
    source_sha256: str
    line: int


class _PythonDynamicSkipAuthorization(NamedTuple):
    test: unittest.TestCase
    runtime_id: str
    reason: str
    code: Any
    line: int


class _PythonPlatformSkipAuthorization(NamedTuple):
    test: unittest.TestCase
    runtime_id: str
    reason: str
    predicate_id: str


class _PythonReviewedSourceModule(NamedTuple):
    inventory_path: str
    module_name: str
    source_path: Path
    source_bytes: bytes
    source_sha256: str


class _PythonFrozenSource(NamedTuple):
    inventory_path: str
    source_path: Path
    source_bytes: bytes
    source_sha256: str
    source_identity: tuple[int, int, int, int, int]
    source_mode: int


class _PythonModuleInstance(NamedTuple):
    name: str
    source: _PythonFrozenSource
    module: types.ModuleType
    spec: Any
    loader: Any


class _PythonExecutionClosure:
    __slots__ = (
        "context",
        "lexical_root",
        "root",
        "sources",
        "module_paths",
        "instances",
        "executed",
        "execution_sys_path",
        "repo_local_names",
    )

    def __init__(
        self,
        root: Path,
        context: Any,
        sources: tuple[_PythonFrozenSource, ...],
    ) -> None:
        self.lexical_root = Path(context.public_files.supplied_root)
        self.root = root
        self.context = context
        self.sources = {source.inventory_path: source for source in sources}
        self.module_paths = dict(_PYTHON_TOOLING_EXECUTION_MODULES)
        self.instances: dict[str, _PythonModuleInstance] = {}
        self.executed: set[str] = set()
        self.execution_sys_path: tuple[str, ...] | None = None
        self.repo_local_names = frozenset(
            PurePosixPath(path).stem
            for path in context.public_files.paths
            if path.endswith(".py")
            and (path.startswith("bench/tools/") or path.startswith("tools/"))
        )

    def verify(self, *, require_complete: bool = False) -> None:
        if self.execution_sys_path is not None and (
            type(sys.path) is not list or tuple(sys.path) != self.execution_sys_path
        ):
            raise InventoryError("Python tooling execution sys.path identity changed")
        if require_complete and set(self.instances) != set(self.module_paths):
            raise InventoryError(
                "Python tooling execution closure is incomplete: missing="
                f"{sorted(set(self.module_paths) - set(self.instances))!r}"
            )
        for name, instance in self.instances.items():
            checks = (
                type(instance) is _PythonModuleInstance,
                instance.name == name,
                self.module_paths.get(name) == instance.source.inventory_path,
                sys.modules.get(name) is instance.module,
                instance.module.__spec__ is instance.spec,
                instance.module.__loader__ is instance.loader,
                instance.module.__name__ == name,
                instance.module.__file__ == str(instance.source.source_path),
                instance.spec.loader is instance.loader,
                instance.spec.origin == str(instance.source.source_path),
            )
            if not all(checks):
                raise InventoryError(
                    "Python tooling frozen module identity changed: "
                    f"{name}:{[index for index, ok in enumerate(checks) if not ok]!r}"
                )

    def live_recheck(self) -> None:
        for source in self.sources.values():
            try:
                resolved = source.source_path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise InventoryError(
                    "Python tooling closure source physical path changed"
                ) from exc
            if resolved != source.source_path:
                raise InventoryError(
                    "Python tooling closure source physical path changed"
                )
            observed = _read_regular_stable_snapshot(
                source.source_path,
                MAX_INVENTORY_BYTES,
                f"Python tooling closure source {source.inventory_path}",
            )
            if (
                observed.sha256 != source.source_sha256
                or observed.identity != source.source_identity
                or observed.mode != source.source_mode
            ):
                raise InventoryError(
                    "Python tooling closure source changed after freeze"
                )


class _PythonSourceModuleBinding(NamedTuple):
    reviewed: _PythonReviewedSourceModule
    module: types.ModuleType
    namespace: dict[str, Any]
    spec: Any
    loader: Any
    name: str
    file: str


class _PythonTestBinding(NamedTuple):
    test: unittest.TestCase
    runtime_id: str
    test_class: type[unittest.TestCase]
    method_name: str
    method_descriptor: Any
    bound_method: Any
    code: Any
    fixtures: tuple["_PythonFixtureBinding", ...]
    source_module: _PythonSourceModuleBinding | None = None
    descriptor_name: str | None = None
    descriptor_qualname: str | None = None
    descriptor_module: str | None = None
    descriptor_wrapped_present: bool | None = None
    descriptor_wrapped: Any = None


_PythonTestBinding.__new__.__defaults__ = (None, None, None, None, None, None)


class _PythonFixtureBinding(NamedTuple):
    kind: str
    owner: Any
    name: str
    present: bool
    descriptor: Any
    bound_callable: Any
    code: Any


class _PythonUnittestRuntimePrimitives(NamedTuple):
    getframe: Any
    token_hex: Any
    skip_exception: type[BaseException]
    test_case_type: type[unittest.TestCase]
    test_case_skip_test: Any
    test_case_run: Any
    test_suite_type: type[unittest.TestSuite]
    test_suite_call: Any
    test_suite_run: Any
    loader_type: type[unittest.TestLoader]
    loader_discover: Any
    loader_find_tests: Any
    loader_find_test_path: Any
    loader_load_tests_from_module: Any
    loader_get_test_case_names: Any
    runner_type: type[unittest.TextTestRunner]
    runner_init: Any
    runner_run: Any
    runner_make_result: Any
    result_type: type[unittest.TextTestResult]
    result_add_skip: Any
    result_add_success: Any
    result_add_error: Any
    result_add_failure: Any
    result_was_successful: Any
    base_test_suite_type: type[unittest.BaseTestSuite]
    test_result_type: type[unittest.TestResult]
    integrity_attributes: tuple[tuple[Any, str, Any], ...]
    loader_suite_class: type[unittest.TestSuite]


_PYTHON_TEST_CASE_EXECUTION_HOOKS = (
    "_callSetUp",
    "_callTestMethod",
    "_callTearDown",
    "doCleanups",
    "_callCleanup",
    "_addDuration",
    "_addExpectedFailure",
    "_addUnexpectedSuccess",
)


class ZigToken(NamedTuple):
    kind: str
    value: str
    offset: int


def _load_build_checker() -> Any:
    path = Path(__file__).with_name("check_build_inventory.py")
    spec = importlib.util.spec_from_file_location("_zynum_build_inventory", path)
    if spec is None or spec.loader is None:
        raise InventoryError("cannot load the build inventory policy")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD_CHECKER = _load_build_checker()
REPOSITORY_SNAPSHOT = BUILD_CHECKER.repository_snapshot


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    return json.loads(
        text, object_pairs_hook=unique_object, parse_constant=reject_constant
    )


def _canonical_inventory_bytes(inventory: dict[str, Any]) -> bytes:
    payload = (json.dumps(inventory, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    if len(payload) > MAX_INVENTORY_BYTES:
        raise InventoryError(f"candidate inventory exceeds {MAX_INVENTORY_BYTES} bytes")
    return payload


def _json_structure_error(value: Any) -> str | None:
    """Return a bounded iterative JSON-structure error before recursive checks."""
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            return f"test inventory JSON exceeds {MAX_JSON_NODES} nodes"
        if isinstance(current, dict):
            if depth > MAX_JSON_DEPTH:
                return f"test inventory JSON exceeds maximum depth {MAX_JSON_DEPTH}"
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            if depth > MAX_JSON_DEPTH:
                return f"test inventory JSON exceeds maximum depth {MAX_JSON_DEPTH}"
            stack.extend((child, depth + 1) for child in current)
    return None


def _fact_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _native_projection(inventory: dict[str, Any]) -> dict[str, Any]:
    """Return the exact native facts covered by the reviewed digest slots."""
    section_names = (
        "environment_profiles",
        "test_enumeration_classes",
        "test_roots",
        "test_mode_rows",
        "expected_test_sets",
        "native_observation_bindings",
    )
    sections: dict[str, list[dict[str, Any]]] = {}
    for section_name in section_names:
        section = inventory.get(section_name)
        if not isinstance(section, list) or not all(
            isinstance(row, dict) for row in section
        ):
            raise InventoryError(
                f"native projection source {section_name} must be an array of objects"
            )
        sections[section_name] = section

    def exact_fields(
        row: dict[str, Any], fields: tuple[str, ...], subject: str
    ) -> dict[str, Any]:
        try:
            return {field: row[field] for field in fields}
        except KeyError as exc:
            raise InventoryError(
                f"native projection source {subject} is missing {exc.args[0]!r}"
            ) from exc

    zig_root_ids = {
        row.get("id")
        for row in sections["test_roots"]
        if row.get("language") == "zig" and isinstance(row.get("id"), str)
    }
    source_rows = [
        row
        for row in sections["test_mode_rows"]
        if row.get("root_id") in zig_root_ids and row.get("disposition") == "execute"
    ]
    row_fields = (
        "id",
        "environment_id",
        "root_id",
        "optimize_mode_id",
        "enumeration_class_id",
        "evidence_slot_id",
        "expectation_state",
        "expected_test_set_id",
    )
    native_rows = sorted(
        (exact_fields(row, row_fields, "native execution row") for row in source_rows),
        key=lambda row: row["id"],
    )
    environment_ids = {row["environment_id"] for row in native_rows}
    class_ids = {row["enumeration_class_id"] for row in native_rows}
    frozen_rows = [
        row for row in native_rows if row["expectation_state"] == FROZEN_STATE
    ]
    frozen_row_ids = {row["id"] for row in frozen_rows}
    frozen_set_ids = {
        row["expected_test_set_id"]
        for row in frozen_rows
        if isinstance(row["expected_test_set_id"], str)
    }

    environment_fields = (
        "id",
        "target",
        "architecture",
        "os",
        "libc",
        "cpu",
        "resolved_cpu_model",
        "cpu_feature_policy",
    )
    environments = sorted(
        (
            exact_fields(row, environment_fields, "environment profile")
            for row in sections["environment_profiles"]
            if row.get("id") in environment_ids
        ),
        key=lambda row: row["id"],
    )
    class_fields = (
        "id",
        "language",
        "architecture",
        "os",
        "libc",
        "object_format",
        "environment_ids",
        "enumeration_source",
    )
    classes = sorted(
        (
            exact_fields(row, class_fields, "test enumeration class")
            for row in sections["test_enumeration_classes"]
            if row.get("id") in class_ids
        ),
        key=lambda row: row["id"],
    )
    set_fields = (
        "id",
        "root_id",
        "tests",
        "count",
        "digest",
        "enumeration_source",
    )
    expected_sets = sorted(
        (
            exact_fields(row, set_fields, "native expected set")
            for row in sections["expected_test_sets"]
            if row.get("id") in frozen_set_ids
        ),
        key=lambda row: row["id"],
    )
    binding_fields = (
        "id",
        "row_id",
        "evidence_slot_id",
        "enumeration_class_id",
        "optimize_mode_id",
        "expected_test_set_id",
        "enumeration_source",
        "digest",
    )
    bindings = sorted(
        (
            exact_fields(row, binding_fields, "native observation binding")
            for row in sections["native_observation_bindings"]
            if row.get("row_id") in frozen_row_ids
        ),
        key=lambda row: row["id"],
    )
    return {
        "schema_id": NATIVE_PROJECTION_SCHEMA_ID,
        "schema_version": NATIVE_PROJECTION_SCHEMA_VERSION,
        "environment_profiles": environments,
        "test_enumeration_classes": classes,
        "native_execution_rows": native_rows,
        "expected_test_sets": expected_sets,
        "native_observation_bindings": bindings,
    }


def _native_projection_digest(inventory: dict[str, Any]) -> str:
    return _fact_digest(_native_projection(inventory))


def _digest_slots_error(
    current: object, next_digest: object, subject: str
) -> str | None:
    digest_pattern = r"[0-9a-f]{64}"
    if (
        not isinstance(current, str)
        or re.fullmatch(digest_pattern, current) is None
        or (
            next_digest is not None
            and (
                not isinstance(next_digest, str)
                or re.fullmatch(digest_pattern, next_digest) is None
                or next_digest == current
            )
        )
    ):
        return f"reviewed {subject} policy constants are invalid"
    return None


def _inventory_digest_policy_constants_error() -> str | None:
    return _digest_slots_error(
        CURRENT_TEST_INVENTORY_SHA256,
        NEXT_TEST_INVENTORY_SHA256,
        "whole-file test inventory",
    )


def _reviewed_inventory_bytes_error(
    inventory_bytes: bytes, *, require_current_only: bool = False
) -> str | None:
    constants_error = _inventory_digest_policy_constants_error()
    if constants_error is not None:
        return constants_error
    observed = hashlib.sha256(inventory_bytes).hexdigest()
    reviewed = {CURRENT_TEST_INVENTORY_SHA256}
    if not require_current_only and NEXT_TEST_INVENTORY_SHA256 is not None:
        reviewed.add(NEXT_TEST_INVENTORY_SHA256)
    if observed in reviewed:
        return None
    next_digest = (
        NEXT_TEST_INVENTORY_SHA256 if NEXT_TEST_INVENTORY_SHA256 is not None else "none"
    )
    return (
        "reviewed whole-file test inventory mismatch: "
        f"observed sha256={observed}; "
        f"current sha256={CURRENT_TEST_INVENTORY_SHA256}; "
        f"next sha256={next_digest}"
    )


def _native_projection_policy_constants_error() -> str | None:
    if (
        NATIVE_PROJECTION_SCHEMA_ID != "zynum-reviewed-native-test-projection-v1"
        or type(NATIVE_PROJECTION_SCHEMA_VERSION) is not int
        or NATIVE_PROJECTION_SCHEMA_VERSION != 1
    ):
        return "reviewed native projection policy constants are invalid"
    return _digest_slots_error(
        CURRENT_NATIVE_PROJECTION_SHA256,
        NEXT_NATIVE_PROJECTION_SHA256,
        "native projection",
    )


def _reviewed_native_projection_error(
    inventory: dict[str, Any], *, require_current_only: bool = False
) -> str | None:
    constants_error = _native_projection_policy_constants_error()
    if constants_error is not None:
        return constants_error
    try:
        projection = _native_projection(inventory)
    except (InventoryError, KeyError, TypeError, ValueError) as exc:
        return f"cannot compute reviewed native projection: {exc}"
    observed = _fact_digest(projection)
    reviewed = {CURRENT_NATIVE_PROJECTION_SHA256}
    if not require_current_only and NEXT_NATIVE_PROJECTION_SHA256 is not None:
        reviewed.add(NEXT_NATIVE_PROJECTION_SHA256)
    if observed in reviewed:
        return None
    count_fields = (
        "environment_profiles",
        "test_enumeration_classes",
        "native_execution_rows",
        "expected_test_sets",
        "native_observation_bindings",
    )
    counts = ",".join(f"{field}={len(projection[field])}" for field in count_fields)
    next_digest = (
        NEXT_NATIVE_PROJECTION_SHA256
        if NEXT_NATIVE_PROJECTION_SHA256 is not None
        else "none"
    )
    return (
        "reviewed native projection mismatch: "
        f"observed sha256={observed}; "
        f"current sha256={CURRENT_NATIVE_PROJECTION_SHA256}; "
        f"next sha256={next_digest}; counts={counts}"
    )


def _current_only_slots_error() -> str | None:
    inventory_error = _inventory_digest_policy_constants_error()
    if inventory_error is not None:
        return inventory_error
    native_error = _native_projection_policy_constants_error()
    if native_error is not None:
        return native_error
    if (
        NEXT_TEST_INVENTORY_SHA256 is not None
        or NEXT_NATIVE_PROJECTION_SHA256 is not None
    ):
        return (
            "current-only policy requires both reviewed NEXT digest slots to be empty"
        )
    return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _zig_root_id(symbol: str) -> str:
    return f"zig-root:{_slug(symbol)}"


def _set_id(root_id: str) -> str:
    raise InventoryError("content-bound expected-set IDs require canonical tests")


def _content_set_id(root_id: str, tests: list[dict[str, Any]]) -> str:
    return f"set:{root_id}:{_fact_digest(tests)}"


def _enumeration_class_id(language: str, environment_id: str | None = None) -> str:
    if language == "python":
        return "enumeration-class:python-static"
    zig_classes = {
        "env:aarch64-macos-baseline": "enumeration-class:aarch64-macos-system-macho",
        "env:x86-64-linux-gnu-baseline": "enumeration-class:x86-64-linux-gnu-elf",
        "env:aarch64-linux-gnu-baseline": "enumeration-class:aarch64-linux-gnu-elf",
        "env:x86-64-windows-gnu-baseline": "enumeration-class:x86-64-windows-gnu-coff",
    }
    if language == "zig" and environment_id in zig_classes:
        return zig_classes[environment_id]
    raise InventoryError("enumeration class requires a supported language/environment")


def _evidence_slot_id(row_id: str) -> str:
    """Return the stable local-evidence join key owned by one matrix row."""
    return f"evidence-slot:{row_id.removeprefix('row:')}"


def _native_observation_binding_facts(
    row: dict[str, Any], expected_test_set_id: str
) -> dict[str, str]:
    """Return the exact native observation identity bound to one matrix row."""
    return {
        "row_id": row["id"],
        "evidence_slot_id": row["evidence_slot_id"],
        "enumeration_class_id": row["enumeration_class_id"],
        "optimize_mode_id": row["optimize_mode_id"],
        "expected_test_set_id": expected_test_set_id,
        "enumeration_source": ZIG_ENUMERATION_SOURCE,
    }


def _native_observation_binding(
    row: dict[str, Any], expected_test_set_id: str
) -> dict[str, str]:
    facts = _native_observation_binding_facts(row, expected_test_set_id)
    digest = _fact_digest(facts)
    return {"id": f"native-observation:{digest}", **facts, "digest": digest}


def _decode_zig_string(raw: str, path: str, offset: int) -> str:
    output: list[str] = []
    index = 0
    escapes = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "\\": "\\",
        '"': '"',
        "'": "'",
        "0": "\0",
    }
    while index < len(raw):
        char = raw[index]
        if char != "\\":
            output.append(char)
            index += 1
            continue
        index += 1
        if index >= len(raw):
            raise InventoryError(f"{path}:{offset}: incomplete Zig string escape")
        escape = raw[index]
        index += 1
        if escape in escapes:
            output.append(escapes[escape])
        elif escape == "x":
            digits = raw[index : index + 2]
            if not re.fullmatch(r"[0-9a-fA-F]{2}", digits):
                raise InventoryError(f"{path}:{offset}: invalid Zig hex escape")
            output.append(chr(int(digits, 16)))
            index += 2
        elif escape == "u" and index < len(raw) and raw[index] == "{":
            end = raw.find("}", index + 1)
            digits = raw[index + 1 : end] if end >= 0 else ""
            if not digits or not re.fullmatch(r"[0-9a-fA-F]+", digits):
                raise InventoryError(f"{path}:{offset}: invalid Zig Unicode escape")
            output.append(chr(int(digits, 16)))
            index = end + 1
        else:
            raise InventoryError(f"{path}:{offset}: unsupported Zig string escape")
    return "".join(output)


def _zig_tokens(text: str, path: str) -> list[ZigToken]:
    tokens: list[ZigToken] = []
    index = 0
    block_depth = 0
    while index < len(text):
        if block_depth:
            if text.startswith("/*", index):
                block_depth += 1
                index += 2
            elif text.startswith("*/", index):
                block_depth -= 1
                index += 2
            else:
                index += 1
            continue
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            index = len(text) if end < 0 else end + 1
            continue
        if text.startswith("/*", index):
            block_depth = 1
            index += 2
            continue
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char == '"':
            start = index
            index += 1
            raw: list[str] = []
            while index < len(text):
                if text[index] == '"':
                    index += 1
                    break
                if text[index] == "\n":
                    raise InventoryError(f"{path}:{start}: newline in Zig string")
                if text[index] == "\\":
                    raw.append(text[index])
                    index += 1
                    if index >= len(text):
                        raise InventoryError(f"{path}:{start}: incomplete Zig string")
                raw.append(text[index])
                index += 1
            else:
                raise InventoryError(f"{path}:{start}: unterminated Zig string")
            tokens.append(
                ZigToken("string", _decode_zig_string("".join(raw), path, start), start)
            )
            continue
        if char == "'":
            start = index
            index += 1
            escaped = False
            while index < len(text):
                current = text[index]
                index += 1
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == "'":
                    break
            else:
                raise InventoryError(f"{path}:{start}: unterminated Zig character")
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < len(text) and (text[index].isalnum() or text[index] == "_"):
                index += 1
            tokens.append(ZigToken("identifier", text[start:index], start))
            continue
        tokens.append(ZigToken("symbol", char, index))
        index += 1
    if block_depth:
        raise InventoryError(f"{path}: unterminated Zig block comment")
    return tokens


def _zig_declarations(text: str, path: str) -> list[dict[str, Any]]:
    tokens = _zig_tokens(text, path)
    declarations: list[dict[str, Any]] = []
    unnamed_ordinal = 0
    for index, token in enumerate(tokens):
        if token.kind != "identifier" or token.value != "test":
            continue
        if (
            index + 1 < len(tokens)
            and tokens[index + 1].kind == "symbol"
            and tokens[index + 1].value == "{"
        ):
            name = f"test_{unnamed_ordinal}"
            kind = "unnamed"
            unnamed_ordinal += 1
        elif index + 1 < len(tokens) and tokens[index + 1].kind == "string":
            name = tokens[index + 1].value
            kind = "named"
        else:
            continue
        ordinal = len(declarations)
        declarations.append(
            {
                "id": f"zig-decl:{path}:{ordinal}",
                "name": name,
                "ordinal": ordinal,
                "kind": kind,
            }
        )
    return declarations


def _zig_imports(text: str, path: str) -> list[str]:
    tokens = _zig_tokens(text, path)
    imports: list[str] = []
    for index in range(len(tokens) - 4):
        window = tokens[index : index + 5]
        if (
            window[0].kind == "symbol"
            and window[0].value == "@"
            and window[1] == ZigToken("identifier", "import", window[1].offset)
            and window[2].kind == "symbol"
            and window[2].value == "("
            and window[3].kind == "string"
            and window[4].kind == "symbol"
            and window[4].value == ")"
        ):
            imports.append(window[3].value)
    return imports


def _cached_text(context: Any, path: str) -> str:
    node = context.public_files.node(path)
    if node.kind != "regular" or node.bytes is None:
        raise InventoryError(f"public source bytes were not frozen: {path}")
    try:
        return node.bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InventoryError(f"public source is not UTF-8: {path}") from exc


def _load_build_inventory(context: Any) -> dict[str, Any]:
    data = _strict_json_loads(_cached_text(context, BUILD_INVENTORY_PATH))
    if not isinstance(data, dict):
        raise InventoryError("build inventory root must be an object")
    if data.get("schema_id") != BUILD_SCHEMA_ID or data.get("schema_version") != 3:
        raise InventoryError("build inventory schema identity mismatch")
    observations = data.get("build_observations")
    if not isinstance(observations, list) or not all(
        isinstance(item, dict) for item in observations
    ):
        raise InventoryError("build inventory observations are malformed")
    ids = [item.get("id") for item in observations]
    if any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise InventoryError("build inventory observation IDs are not unique strings")
    return data


def _expected_test_rows(root_id: str, names: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {"id": f"test:{_slug(root_id)}:{ordinal}", "name": name, "ordinal": ordinal}
        for ordinal, name in enumerate(names)
    ]


def _python_tests(text: str, path: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        raise InventoryError(f"Python test discovery failed for {path}: {exc}") from exc
    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_tests"
        for node in ast.walk(tree)
    ):
        raise InventoryError(f"{path}: dynamic unittest discovery is unknown")
    tests: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        is_case = any(
            (isinstance(base, ast.Name) and base.id.endswith("TestCase"))
            or (isinstance(base, ast.Attribute) and base.attr.endswith("TestCase"))
            for base in node.bases
        )
        if not is_case:
            continue
        for member in node.body:
            if isinstance(
                member, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and member.name.startswith("test_"):
                ordinal = len(tests)
                name = f"{node.name}.{member.name}"
                tests.append(
                    {
                        "id": f"python-decl:{path}:{ordinal}",
                        "name": name,
                        "ordinal": ordinal,
                    }
                )
            elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                targets = (
                    member.targets
                    if isinstance(member, ast.Assign)
                    else [member.target]
                )
                if any(
                    isinstance(target, ast.Name) and target.id.startswith("test_")
                    for target in targets
                ):
                    raise InventoryError(
                        f"{path}: dynamic unittest discovery is unknown"
                    )
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
        and node.args[1].value.startswith("test_")
        for node in ast.walk(tree)
    ):
        raise InventoryError(f"{path}: dynamic unittest discovery is unknown")
    return tests


def _workflow_commands(context: Any) -> dict[str, str]:
    commands: dict[str, str] = {}
    for path in (".github/workflows/ci.yml", ".github/workflows/release.yml"):
        lines = _cached_text(context, path).splitlines()
        job = ""
        step_name = ""
        in_jobs = False
        job_indent = -1
        for index, line in enumerate(lines):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped == "jobs:":
                in_jobs = True
                continue
            if in_jobs and indent == 2 and re.fullmatch(r"[A-Za-z0-9_-]+:", stripped):
                job = stripped[:-1]
                job_indent = indent
                continue
            if job and indent > job_indent and stripped.startswith("- name:"):
                step_name = stripped.split(":", 1)[1].strip().strip("'\"")
                continue
            if step_name and stripped.startswith("run:"):
                value = stripped.split(":", 1)[1].strip()
                if value in {"|", ">"}:
                    body: list[str] = []
                    body_indent = None
                    for following in lines[index + 1 :]:
                        next_indent = len(following) - len(following.lstrip())
                        if following.strip() and body_indent is None:
                            body_indent = next_indent
                        if (
                            following.strip()
                            and body_indent is not None
                            and next_indent < body_indent
                        ):
                            break
                        if body_indent is not None:
                            body.append(following[body_indent:])
                    value = "\n".join(body).strip()
                identifier = f"workflow-launch:{path}:{job}:{_slug(step_name)}"
                commands[identifier] = value
                step_name = ""
    return commands


def _command_for(environment: dict[str, Any], mode: str) -> str:
    release = {
        "Debug": "",
        "ReleaseSafe": " --release=safe",
        "ReleaseFast": " --release=fast",
    }[mode]
    smoke = str(environment["host_tool_smoke"]).lower()
    return (
        f"zig build{release} test -Dtarget={environment['target']} "
        f"-Dcpu={environment['cpu']} -Dtest-optimize={mode} "
        f"-Dhost-tool-smoke={smoke} --summary failures"
    )


def _python_command(root: dict[str, Any]) -> str:
    kind = root["root_kind"]
    paths = root["module_paths"]
    if kind == "direct":
        if len(paths) != 1:
            raise InventoryError(f"{root['id']}: direct Python root must have one path")
        path = paths[0]
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", path):
            raise InventoryError(f"{root['id']}: direct path is not shell-portable")
        return f"python3 -B {path}"
    if kind == "discovery":
        start = root.get("discovery_start")
        pattern = root.get("discovery_pattern")
        if not isinstance(start, str) or not isinstance(pattern, str):
            raise InventoryError(f"{root['id']}: discovery command metadata is missing")
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", start) or not re.fullmatch(
            r"[A-Za-z0-9_.*?-]+", pattern
        ):
            raise InventoryError(f"{root['id']}: discovery arguments are not portable")
        return f"python3 -B -m unittest discover -s {start} -p {json.dumps(pattern)}"
    raise InventoryError(f"{root['id']}: unsupported Python root kind {kind!r}")


def _section_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    sections = (
        "test_enumeration_classes",
        "test_roots",
        "zig_test_files",
        "python_test_modules",
        "expected_test_sets",
        "native_observation_bindings",
        "test_mode_rows",
        "workflow_mode_bindings",
        "known_gaps",
    )
    return {
        section: {
            "count": len(inventory[section]),
            "digest": _fact_digest(inventory[section]),
        }
        for section in sections
    }


def discover(
    root: Path, inventory_path: Path, *, _context: Any | None = None
) -> dict[str, Any]:
    context = (
        BUILD_CHECKER._make_discovery_context(root, inventory_path)
        if _context is None
        else _context
    )
    build_inventory = _load_build_inventory(context)
    observations = build_inventory["build_observations"]
    workflow_observations = build_inventory.get("workflow_launches", [])
    if not isinstance(workflow_observations, list) or not all(
        isinstance(item, dict) for item in workflow_observations
    ):
        raise InventoryError("build inventory workflow observations are malformed")
    by_id = {item["id"]: item for item in [*observations, *workflow_observations]}
    compile_tests = sorted(
        (
            item
            for item in observations
            if item.get("call") == "b.addTest"
            and item.get("id") != FACTORY_COMPILE_ID
            and item.get("artifact_role") != FACTORY_ROLE
        ),
        key=lambda item: item["id"],
    )
    if len(compile_tests) != 21:
        raise InventoryError(
            f"expected exactly 21 b.addTest logical roots, found {len(compile_tests)}"
        )

    public_paths = context.public_files.path_set
    zig_text: dict[str, str] = {}
    zig_declarations: dict[str, list[dict[str, Any]]] = {}
    for path in context.public_files.paths:
        if PurePosixPath(path).suffix != ".zig":
            continue
        text = _cached_text(context, path)
        zig_text[path] = text
        declarations = _zig_declarations(text, path)
        if declarations:
            zig_declarations[path] = declarations

    launch_by_compile: dict[str, dict[str, Any]] = {}
    for observation in observations:
        source_artifact = observation.get("source_artifact")
        if isinstance(source_artifact, str):
            if source_artifact in launch_by_compile:
                raise InventoryError(f"duplicate launch for {source_artifact}")
            launch_by_compile[source_artifact] = observation
    aggregate = by_id.get(AGGREGATE_STEP_ID)
    if not isinstance(aggregate, dict):
        raise InventoryError("canonical test aggregate observation is missing")
    aggregate_dependencies = {
        item.get("id"): item.get("condition")
        for item in aggregate.get("direct_dependencies", [])
        if isinstance(item, dict)
    }
    focused_by_launch: dict[str, list[str]] = defaultdict(list)
    for observation in observations:
        if (
            observation.get("category") != "step"
            or observation.get("step_role") != "focused-validation"
        ):
            continue
        for dependency in observation.get("direct_dependencies", []):
            if isinstance(dependency, dict) and isinstance(dependency.get("id"), str):
                focused_by_launch[dependency["id"]].append(observation["id"])

    python_tooling_launch = by_id.get(PYTHON_TOOLING_LAUNCH_ID)
    if (
        not isinstance(python_tooling_launch, dict)
        or python_tooling_launch.get("category") != "launch"
        or python_tooling_launch.get("inventory_root_id") != PYTHON_TOOLING_ROOT_ID
        or python_tooling_launch.get("launch_role")
        != "inventory-declared-python-tooling-tests"
    ):
        raise InventoryError(
            "Python tooling launch observation is missing or incorrect"
        )
    python_tooling_step = by_id.get(PYTHON_TOOLING_STEP_ID)
    expected_python_tooling_dependencies = [
        {"id": PYTHON_TOOLING_LAUNCH_ID, "condition": "always"}
    ]
    expected_python_tooling_closure = {
        "launch_observation_id": PYTHON_TOOLING_LAUNCH_ID,
        "launch_count": 1,
        "relation": "only-direct-dependency",
    }
    if (
        not isinstance(python_tooling_step, dict)
        or python_tooling_step.get("category") != "step"
        or python_tooling_step.get("step_role") != "focused-validation"
        or python_tooling_step.get("aggregate_test_membership")
        != "host-tool-smoke-member"
        or python_tooling_step.get("aggregate_condition")
        != "always within test-host-tool-smoke"
        or python_tooling_step.get("direct_dependencies")
        != expected_python_tooling_dependencies
        or python_tooling_step.get("closure_contract")
        != expected_python_tooling_closure
    ):
        raise InventoryError("Python tooling focused-step closure drifted")
    host_tool_step = by_id.get(HOST_TOOL_SMOKE_STEP_ID)
    if (
        not isinstance(host_tool_step, dict)
        or host_tool_step.get("category") != "step"
        or host_tool_step.get("step_role") != "aggregate-validation"
        or host_tool_step.get("aggregate_test_membership") != "conditional-member"
        or host_tool_step.get("aggregate_condition") != "host-tool-smoke is true"
        or host_tool_step.get("direct_dependencies")
        != list(HOST_TOOL_SMOKE_DIRECT_DEPENDENCIES)
        or host_tool_step.get("closure_contract")
        != {
            "direct_dependency_count": len(HOST_TOOL_SMOKE_DIRECT_DEPENDENCIES),
            "relation": "exact-six-direct-host-tool-dependencies",
        }
    ):
        raise InventoryError("host-tool smoke aggregate closure drifted")
    if (
        sum(
            isinstance(edge, dict)
            and edge.get("id") == HOST_TOOL_SMOKE_STEP_ID
            and edge.get("condition") == "host-tool-smoke is true"
            for edge in aggregate.get("direct_dependencies", [])
        )
        != 1
    ):
        raise InventoryError("host-tool smoke canonical aggregate edge is missing")
    forbidden_aggregate_ids = {
        PYTHON_TOOLING_STEP_ID,
        PYTHON_TOOLING_LAUNCH_ID,
        "step:build.zig:build:test-build-inventory",
        "launch:build.zig:build:build_inventory_tests",
        *(dependency["id"] for dependency in HOST_TOOL_SMOKE_DIRECT_DEPENDENCIES[1:]),
    }
    if any(
        isinstance(edge, dict) and edge.get("id") in forbidden_aggregate_ids
        for edge in aggregate.get("direct_dependencies", [])
    ):
        raise InventoryError(
            "canonical test aggregate bypasses the unique host-tool smoke path"
        )

    root_rows: list[dict[str, Any]] = []
    root_reach: dict[str, list[str]] = defaultdict(list)
    root_expected_names: dict[str, list[str]] = {}
    for compile_observation in compile_tests:
        symbol = compile_observation["anchor"]["symbol"]
        root_id = _zig_root_id(symbol)
        paths = compile_observation.get("root_source")
        if (
            not isinstance(paths, list)
            or len(paths) != 1
            or not isinstance(paths[0], str)
        ):
            raise InventoryError(
                f"{compile_observation['id']}: b.addTest must have one root"
            )
        physical_path = paths[0]
        launch = launch_by_compile.get(compile_observation["id"])
        if launch is None:
            raise InventoryError(f"{compile_observation['id']}: test launch is missing")
        if launch["id"] not in aggregate_dependencies:
            raise InventoryError(
                f"{compile_observation['id']}: aggregate edge is missing"
            )
        entry_paths = [physical_path, *ROOT_ENTRY_PATHS.get(symbol, ())]
        if any(path not in public_paths for path in entry_paths):
            raise InventoryError(
                f"{compile_observation['id']}: source entry is missing"
            )
        reached: set[str] = set()
        pending = list(entry_paths)
        while pending:
            path = pending.pop()
            if path in reached:
                continue
            reached.add(path)
            if path not in zig_text:
                continue
            parent = PurePosixPath(path).parent
            for imported in _zig_imports(zig_text[path], path):
                candidate = posixpath.normpath((parent / imported).as_posix())
                if candidate in public_paths and candidate not in reached:
                    pending.append(candidate)
        reaching_files = sorted(path for path in reached if path in zig_declarations)
        for path in reaching_files:
            root_reach[path].append(root_id)
        names = [
            f"{PurePosixPath(path).stem}.test.{declaration['name']}"
            for path in reaching_files
            for declaration in zig_declarations[path]
        ]
        if not names or len(names) != len(set(names)):
            raise InventoryError(
                f"{root_id}: expected test names must be nonempty and unique"
            )
        root_expected_names[root_id] = names

        module_symbol = ROOT_MODULE_SYMBOLS.get(symbol)
        required_objects: list[dict[str, str]] = []
        if module_symbol is not None:
            links = sorted(
                (
                    item
                    for item in observations
                    if item.get("category") == "link"
                    and item.get("anchor", {}).get("symbol") == module_symbol
                    and item["id"].endswith("isolated_test_library")
                ),
                key=lambda item: item["id"],
            )
            for link in links:
                producer_symbol = link["id"].rsplit("<-", 1)[1]
                compile_id = f"compile:build.zig:build:{producer_symbol}"
                producer = by_id.get(compile_id)
                if not isinstance(producer, dict):
                    raise InventoryError(
                        f"{link['id']}: isolated compile observation missing"
                    )
                if producer.get("optimize_source") != "test-optimize":
                    raise InventoryError(
                        f"{compile_id}: isolated object uses production optimize"
                    )
                if "x86_64" not in str(
                    producer.get("condition")
                ) or "x86_64" not in str(link.get("condition")):
                    raise InventoryError(
                        f"{link['id']}: isolated object guard is incorrect"
                    )
                required_objects.append(
                    {
                        "compile_observation_id": compile_id,
                        "link_observation_id": link["id"],
                        "predicate_id": "predicate:arch-x86-64",
                    }
                )
        root_rows.append(
            {
                "id": root_id,
                "language": "zig",
                "physical_path": physical_path,
                "variant_id": ROOT_VARIANTS.get(symbol),
                "compile_observation_id": compile_observation["id"],
                "launch_observation_id": launch["id"],
                "selector_step_observation_ids": sorted(
                    focused_by_launch[launch["id"]]
                ),
                "aggregate_step_observation_id": AGGREGATE_STEP_ID,
                "aggregate_predicate_id": (
                    "predicate:arch-x86-64"
                    if "x86_64" in str(aggregate_dependencies[launch["id"]])
                    else "predicate:always"
                ),
                "source_entry_paths": entry_paths,
                "required_isolated_objects": required_objects,
            }
        )

    zig_file_rows: list[dict[str, Any]] = []
    zig_gap_paths: list[str] = []
    root_physical_paths = {row["physical_path"] for row in root_rows}
    for path, declarations in sorted(zig_declarations.items()):
        reaches = sorted(root_reach.get(path, []))
        if path in root_physical_paths:
            classification = "logical-root"
            gap_id = None
        elif reaches:
            classification = "permitted-source-local"
            gap_id = None
        else:
            classification = "current-gap"
            gap_id = f"gap:zig-unreached:{_slug(path)}"
            zig_gap_paths.append(path)
        zig_file_rows.append(
            {
                "id": f"zig-file:{path}",
                "path": path,
                "classification": classification,
                "declarations": declarations,
                "reaching_root_ids": reaches,
                "gap_id": gap_id,
            }
        )

    python_paths = sorted(
        path
        for path in context.public_files.paths
        if PurePosixPath(path).suffix == ".py"
        and PurePosixPath(path).name.startswith("test_")
    )
    if len(python_paths) != 19:
        raise InventoryError(
            f"expected exactly 19 Python test candidates, found {len(python_paths)}"
        )
    benchmark_paths = tuple(
        path for path in python_paths if path.startswith("bench/tools/")
    )
    if len(benchmark_paths) != 14:
        raise InventoryError(
            f"expected exactly 14 benchmark Python test candidates, found {len(benchmark_paths)}"
        )
    python_root_specs: list[dict[str, Any]] = []
    host_tool_dependency_ids = {
        dependency["id"] for dependency in HOST_TOOL_SMOKE_DIRECT_DEPENDENCIES
    }
    for spec in PYTHON_ROOTS:
        row = dict(spec)
        if spec["id"] == PYTHON_TOOLING_ROOT_ID:
            row["module_paths"] = benchmark_paths
        for launch_id in spec["launch_ids"]:
            launch = by_id.get(launch_id)
            if not isinstance(launch, dict) or launch.get("category") != "launch":
                raise InventoryError(
                    f"{spec['id']}: Python launch observation is missing: {launch_id}"
                )
            if spec["aggregate"] and not (
                launch_id in aggregate_dependencies
                or any(
                    step_id in aggregate_dependencies
                    or step_id in host_tool_dependency_ids
                    for step_id in focused_by_launch[launch_id]
                )
            ):
                raise InventoryError(
                    f"{spec['id']}: Python launch is outside the canonical aggregate"
                )
        python_root_specs.append(row)
    memberships: dict[str, list[str]] = defaultdict(list)
    for spec in python_root_specs:
        for path in spec["module_paths"]:
            memberships[path].append(spec["id"])
    if set(memberships) != set(python_paths):
        raise InventoryError("Python suite memberships do not cover candidates exactly")
    python_file_tests: dict[str, list[dict[str, Any]]] = {}
    python_module_rows: list[dict[str, Any]] = []
    for path in python_paths:
        tests = _python_tests(_cached_text(context, path), path)
        if not tests:
            raise InventoryError(f"{path}: unexpected zero-test Python candidate")
        python_file_tests[path] = tests
        root_ids = sorted(memberships[path])
        launch_ids = sorted(
            {
                launch_id
                for spec in python_root_specs
                if spec["id"] in root_ids
                for launch_id in spec["launch_ids"]
            }
        )
        python_module_rows.append(
            {
                "id": f"python-module:{path}",
                "path": path,
                "declarations": tests,
                "root_ids": root_ids,
                "launch_observation_ids": launch_ids,
                "gap_ids": [],
            }
        )

    # Zig expected sets are compiler observations, never source-parser projections.
    # The static names above remain useful only for declaration/reachability sanity.
    expected_sets: list[dict[str, Any]] = []
    for spec in python_root_specs:
        names = [
            f"{path}::{declaration['name']}"
            for path in spec["module_paths"]
            for declaration in python_file_tests[path]
        ]
        tests = _expected_test_rows(spec["id"], names)
        set_id = _content_set_id(spec["id"], tests)
        expected_sets.append(
            {
                "id": set_id,
                "root_id": spec["id"],
                "tests": tests,
                "count": len(tests),
                "digest": _fact_digest(tests),
                "enumeration_source": PYTHON_ENUMERATION_SOURCE,
            }
        )
        root_row = {
            "id": spec["id"],
            "language": "python",
            "root_kind": spec["kind"],
            "module_paths": list(spec["module_paths"]),
            "launch_observation_ids": list(spec["launch_ids"]),
            "aggregate_step_observation_id": AGGREGATE_STEP_ID
            if spec["aggregate"]
            else None,
            "matrix_applicable": spec["matrix"],
        }
        if spec["kind"] == "discovery":
            root_row["discovery_start"] = spec["discovery_start"]
            root_row["discovery_pattern"] = spec["discovery_pattern"]
        root_rows.append(root_row)

    mode_rows: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        for root_row in root_rows:
            for mode in MODES:
                row_id = (
                    f"row:{_slug(environment['id'])}:{_slug(root_row['id'])}:{mode}"
                )
                if root_row["language"] == "zig":
                    structured = (
                        root_row["aggregate_predicate_id"] == "predicate:arch-x86-64"
                    )
                    skip = structured and environment["architecture"] != "x86_64"
                    disposition = "structured-skip" if skip else "execute"
                    mode_rows.append(
                        {
                            "id": row_id,
                            "environment_id": environment["id"],
                            "root_id": root_row["id"],
                            "optimize_mode_id": f"mode:{mode}",
                            "disposition": disposition,
                            "predicate_id": (
                                "predicate:arch-not-x86-64"
                                if skip
                                else root_row["aggregate_predicate_id"]
                            ),
                            "command_template": None
                            if skip
                            else _command_for(environment, mode),
                            "mode_effect": "test-module-optimize",
                            "expected_actual_module_optimize": None if skip else mode,
                            "evidence_slot_id": _evidence_slot_id(row_id),
                            "enumeration_class_id": _enumeration_class_id(
                                "zig", environment["id"]
                            ),
                            "expected_test_set_id": None,
                            "expectation_state": (
                                "not-applicable" if skip else PENDING_STATE
                            ),
                        }
                    )
                else:
                    smoke_skip = (
                        root_row["matrix_applicable"]
                        and not environment["host_tool_smoke"]
                    )
                    disposition = "inapplicable" if smoke_skip else "execute"
                    mode_rows.append(
                        {
                            "id": row_id,
                            "environment_id": environment["id"],
                            "root_id": root_row["id"],
                            "optimize_mode_id": f"mode:{mode}",
                            "disposition": disposition,
                            "predicate_id": (
                                "predicate:host-tool-smoke-disabled"
                                if smoke_skip
                                else "predicate:host-tool-smoke"
                                if root_row["matrix_applicable"]
                                else "predicate:always"
                            ),
                            "command_template": (
                                None
                                if disposition != "execute"
                                else _python_command(root_row)
                            ),
                            "mode_effect": "not-applicable",
                            "expected_actual_module_optimize": None,
                            "evidence_slot_id": _evidence_slot_id(row_id),
                            "enumeration_class_id": _enumeration_class_id("python"),
                            "expected_test_set_id": (
                                None
                                if disposition != "execute"
                                else next(
                                    item["id"]
                                    for item in expected_sets
                                    if item["root_id"] == root_row["id"]
                                )
                            ),
                            "expectation_state": (
                                "not-applicable"
                                if disposition != "execute"
                                else FROZEN_STATE
                            ),
                        }
                    )

    workflow_commands = _workflow_commands(context)
    workflow_ids = (
        "workflow-launch:.github/workflows/ci.yml:target-tests:test-debug-target",
        "workflow-launch:.github/workflows/ci.yml:target-tests:test-releasesafe-target",
        "workflow-launch:.github/workflows/ci.yml:target-tests:test-releasefast-target",
        "workflow-launch:.github/workflows/release.yml:artifacts:test",
    )
    workflow_modes = ("Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSafe")
    workflow_bindings = []
    for identifier, mode in zip(workflow_ids, workflow_modes, strict=True):
        if identifier not in by_id or identifier not in workflow_commands:
            raise InventoryError(
                f"workflow mode binding source is missing: {identifier}"
            )
        command = workflow_commands[identifier]
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise InventoryError(
                f"{identifier}: workflow command is malformed"
            ) from exc
        optimize_flags = [
            token for token in tokens if token.startswith("-Dtest-optimize=")
        ]
        if optimize_flags != [f"-Dtest-optimize={mode}"]:
            raise InventoryError(
                f"{identifier}: workflow command must contain one explicit test optimize mode"
            )
        host_tool_flags = [
            token for token in tokens if token.startswith("-Dhost-tool-smoke=")
        ]
        if host_tool_flags != ["-Dhost-tool-smoke=false"]:
            raise InventoryError(
                f"{identifier}: workflow correctness command must disable the separately executed host-tool aggregate exactly"
            )
        workflow_bindings.append(
            {
                "id": f"binding:{_slug(identifier)}",
                "workflow_observation_id": identifier,
                "optimize_mode_id": f"mode:{mode}",
                "command_template": command,
            }
        )

    known_gaps = [
        {
            "id": f"gap:zig-unreached:{_slug(path)}",
            "kind": "zig-test-file-without-reaching-root",
            "subject_ids": [f"zig-file:{path}"],
        }
        for path in zig_gap_paths
    ]
    for environment in ENVIRONMENTS:
        pending = [
            row["id"]
            for row in mode_rows
            if row["environment_id"] == environment["id"]
            and row["expectation_state"] == PENDING_STATE
        ]
        if pending:
            known_gaps.append(
                {
                    "id": f"gap:native-test-enumeration:{_slug(environment['id'])}",
                    "kind": "native-test-enumeration-required",
                    "subject_ids": pending,
                }
            )
    object_formats = {"macos": "macho", "linux": "elf", "windows": "coff"}
    enumeration_classes = [
        {
            "id": _enumeration_class_id("zig", environment["id"]),
            "language": "zig",
            "architecture": environment["architecture"],
            "os": environment["os"],
            "libc": environment["libc"],
            "object_format": object_formats[environment["os"]],
            "environment_ids": [environment["id"]],
            "enumeration_source": ZIG_ENUMERATION_SOURCE,
        }
        for environment in ENVIRONMENTS
    ]
    enumeration_classes.append(
        {
            "id": _enumeration_class_id("python"),
            "language": "python",
            "architecture": None,
            "os": None,
            "libc": None,
            "object_format": None,
            "environment_ids": [environment["id"] for environment in ENVIRONMENTS],
            "enumeration_source": PYTHON_ENUMERATION_SOURCE,
        }
    )
    result: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "build_inventory_schema_id": BUILD_SCHEMA_ID,
        "optimize_modes": [{"id": f"mode:{mode}", "zig_value": mode} for mode in MODES],
        "predicates": [
            {"id": "predicate:always", "expression": "true"},
            {"id": "predicate:arch-x86-64", "expression": "target.arch == x86_64"},
            {"id": "predicate:arch-not-x86-64", "expression": "target.arch != x86_64"},
            {
                "id": "predicate:host-tool-smoke",
                "expression": "host_tool_smoke == true",
            },
            {
                "id": "predicate:host-tool-smoke-disabled",
                "expression": "host_tool_smoke == false",
            },
        ],
        "environment_profiles": [dict(item) for item in ENVIRONMENTS],
        "test_enumeration_classes": enumeration_classes,
        "test_roots": sorted(root_rows, key=lambda item: item["id"]),
        "zig_test_files": zig_file_rows,
        "python_test_modules": python_module_rows,
        "python_skip_contracts": [],
        "expected_test_sets": sorted(expected_sets, key=lambda item: item["id"]),
        "native_observation_bindings": [],
        "test_mode_rows": sorted(mode_rows, key=lambda item: item["id"]),
        "workflow_mode_bindings": workflow_bindings,
        "known_gaps": sorted(known_gaps, key=lambda item: item["id"]),
        "matrix_row_contract": {},
        "strict_summary": {},
    }
    row_ids = [row["id"] for row in result["test_mode_rows"]]
    result["matrix_row_contract"] = {
        "relation": "exact-row-id-superset",
        "required_row_ids": row_ids,
        "count": len(row_ids),
        "digest": _fact_digest(row_ids),
    }
    result["strict_summary"] = _section_summary(result)
    return result


def _decode_protocol_value(line: str, tag: str) -> str:
    parts = line.split(":")
    if len(parts) != 3 or parts[0] != tag or not parts[1].isdigit():
        raise InventoryError(f"malformed {tag} protocol line")
    if len(parts[1]) > 10:
        raise InventoryError(f"{tag} protocol payload exceeds byte limit")
    declared_length = int(parts[1])
    if declared_length > MAX_PROTOCOL_VALUE_BYTES:
        raise InventoryError(f"{tag} protocol payload exceeds byte limit")
    payload = parts[2]
    if len(payload) != declared_length * 2 or not re.fullmatch(r"[0-9a-f]*", payload):
        raise InventoryError(f"malformed {tag} protocol payload")
    try:
        decoded = bytes.fromhex(payload).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise InventoryError(f"invalid UTF-8 {tag} protocol payload") from exc
    if len(decoded.encode("utf-8")) != declared_length:
        raise InventoryError(f"incorrect {tag} protocol byte length")
    return decoded


def _read_regular_stable_snapshot(
    path: Path | str,
    maximum_bytes: int,
    subject: str = "compiler enumeration protocol",
    *,
    directory_fd: int | None = None,
) -> FrozenInventorySnapshot:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise InventoryError(f"cannot read {subject}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise InventoryError(f"{subject} is not a regular file: {path}")
        if before.st_size > maximum_bytes:
            raise InventoryError(f"{subject} exceeds {maximum_bytes} bytes: {path}")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > maximum_bytes:
            raise InventoryError(f"{subject} exceeds {maximum_bytes} bytes: {path}")
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            before_identity != after_identity
            or before.st_mode != after.st_mode
            or len(data) != after.st_size
        ):
            raise InventoryError(f"{subject} changed while reading: {path}")
        return FrozenInventorySnapshot(
            bytes=data,
            identity=after_identity,
            sha256=hashlib.sha256(data).hexdigest(),
            mode=after.st_mode,
        )
    except OSError as exc:
        raise InventoryError(f"cannot read {subject}: {path}") from exc
    finally:
        os.close(descriptor)


def _read_regular_stable_bytes(
    path: Path, maximum_bytes: int, subject: str = "compiler enumeration protocol"
) -> bytes:
    return _read_regular_stable_snapshot(path, maximum_bytes, subject).bytes


def _parse_protocol_log(
    path: Path, totals: _ProtocolTotals | None = None
) -> list[dict[str, Any]]:
    protocol_totals = _ProtocolTotals() if totals is None else totals
    try:
        snapshot = _read_regular_stable_snapshot(path, MAX_PROTOCOL_BYTES)
        if protocol_totals.bytes + len(snapshot.bytes) > MAX_PROTOCOL_BYTES:
            raise InventoryError(
                "compiler enumeration protocol logs exceed cumulative byte limit"
            )
        protocol_totals.bytes += len(snapshot.bytes)
        lines = snapshot.bytes.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise InventoryError(
            f"compiler enumeration protocol is not UTF-8: {path}"
        ) from exc
    if any(len(line.encode("utf-8")) > MAX_PROTOCOL_LINE_BYTES for line in lines):
        raise InventoryError(
            f"compiler enumeration protocol line exceeds limit: {path}"
        )
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    local_tests = 0
    index = 0
    while index < len(lines):
        if lines[index] == "ZYNUM-TEST-INVENTORY-V1":
            raise InventoryError(
                f"unsupported compiler enumeration protocol V1: {path}"
            )
        if lines[index] != "ZYNUM-TEST-INVENTORY-V2":
            index += 1
            continue
        if index + 3 >= len(lines):
            raise InventoryError(f"truncated compiler enumeration protocol: {path}")
        marker = lines[index]
        mode_line = lines[index + 1]
        if not mode_line.startswith("mode:"):
            raise InventoryError(f"missing protocol mode after {marker}: {path}")
        mode = mode_line.removeprefix("mode:")
        if mode not in MODES:
            raise InventoryError(f"unsupported protocol optimize mode {mode!r}")
        root_id = _decode_protocol_value(lines[index + 2], "root")
        cursor = index + 3
        if not lines[cursor].startswith("class:"):
            raise InventoryError(f"missing protocol class after {marker}: {path}")
        class_id = _decode_protocol_value(lines[cursor], "class")
        cursor += 1
        if cursor >= len(lines):
            raise InventoryError(f"truncated compiler enumeration protocol: {path}")
        count_line = lines[cursor]
        if not re.fullmatch(r"count:[0-9]+", count_line):
            raise InventoryError(f"malformed protocol count for {root_id}")
        count_text = count_line.removeprefix("count:")
        if len(count_text) > 10:
            raise InventoryError(f"protocol test count exceeds limit for {root_id}")
        count = int(count_text)
        if count > MAX_PROTOCOL_TESTS_PER_BLOCK:
            raise InventoryError(f"protocol test count exceeds limit for {root_id}")
        local_tests += count
        if local_tests > MAX_PROTOCOL_TOTAL_TESTS:
            raise InventoryError(f"protocol total test count exceeds limit: {path}")
        if protocol_totals.tests + count > MAX_PROTOCOL_TOTAL_TESTS:
            raise InventoryError("protocol logs exceed cumulative test count limit")
        test_lines = lines[cursor + 1 : cursor + 1 + count]
        if len(test_lines) != count:
            raise InventoryError(f"truncated protocol test list for {root_id}")
        names: list[str] = []
        for ordinal, line in enumerate(test_lines):
            parts = line.split(":")
            if len(parts) != 4 or parts[0] != "test" or parts[1] != str(ordinal):
                raise InventoryError(
                    f"noncanonical protocol test ordinal for {root_id}"
                )
            names.append(_decode_protocol_value(":".join(("test", *parts[2:])), "test"))
        if not names or len(names) != len(set(names)):
            raise InventoryError(
                f"protocol test names must be nonempty and unique: {root_id}"
            )
        identity = (root_id, mode)
        if identity in seen:
            raise InventoryError(
                f"duplicate compiler enumeration protocol block {identity}"
            )
        seen.add(identity)
        if protocol_totals.blocks + 1 > MAX_PROTOCOL_BLOCKS:
            raise InventoryError("protocol cumulative block count exceeds limit")
        protocol_totals.blocks += 1
        protocol_totals.tests += count
        observations.append(
            {
                "protocol": marker,
                "root_id": root_id,
                "mode": mode,
                "class_id": class_id,
                "names": names,
            }
        )
        index = cursor + 1 + count
    if not observations:
        raise InventoryError(f"no compiler enumeration protocol blocks found: {path}")
    return observations


def _refresh_native_gaps(inventory: dict[str, Any]) -> None:
    static_gaps = [
        gap
        for gap in inventory["known_gaps"]
        if gap["kind"] == "zig-test-file-without-reaching-root"
    ]
    native_gaps: list[dict[str, Any]] = []
    for environment in inventory["environment_profiles"]:
        pending = [
            row["id"]
            for row in inventory["test_mode_rows"]
            if row["environment_id"] == environment["id"]
            and row["expectation_state"] == PENDING_STATE
        ]
        if pending:
            native_gaps.append(
                {
                    "id": f"gap:native-test-enumeration:{_slug(environment['id'])}",
                    "kind": "native-test-enumeration-required",
                    "subject_ids": pending,
                }
            )
    inventory["known_gaps"] = sorted(
        [*static_gaps, *native_gaps], key=lambda row: row["id"]
    )


def refresh_from_protocol(
    root: Path, inventory_path: Path, bindings: list[tuple[str, Path]]
) -> RefreshedInventoryCandidate:
    """Return a fully validated v3 inventory refreshed by native protocols."""
    try:
        existing_snapshot = _read_regular_stable_snapshot(
            inventory_path, MAX_INVENTORY_BYTES, "inventory for refresh"
        )
        digest_error = _reviewed_inventory_bytes_error(existing_snapshot.bytes)
        if digest_error is not None:
            raise InventoryError(digest_error)
        existing = _strict_json_loads(existing_snapshot.bytes.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise InventoryError(f"cannot read inventory for refresh: {exc}") from exc
    if not isinstance(existing, dict):
        raise InventoryError("existing inventory is invalid: root must be an object")
    existing_errors = _validate_inventory_data(
        root,
        inventory_path,
        existing,
        structure_only=True,
        inventory_bytes=existing_snapshot.bytes,
        filesystem_snapshot=existing_snapshot,
        _refresh_base=True,
    )
    if existing_errors:
        raise InventoryError(
            "existing inventory is invalid: " + "; ".join(existing_errors)
        )
    refreshed = discover(root, inventory_path)
    refreshed["python_skip_contracts"] = existing["python_skip_contracts"]

    sets_by_id = {row["id"]: row for row in refreshed["expected_test_sets"]}
    rows_by_id = {row["id"]: row for row in refreshed["test_mode_rows"]}
    old_sets = {row["id"]: row for row in existing["expected_test_sets"]}
    old_bindings = {
        row["row_id"]: row for row in existing["native_observation_bindings"]
    }
    native_bindings_by_row: dict[str, dict[str, str]] = {}
    for old_row in existing["test_mode_rows"]:
        if old_row["expectation_state"] != FROZEN_STATE or not old_row[
            "root_id"
        ].startswith("zig-root:"):
            continue
        row = rows_by_id[old_row["id"]]
        set_id = old_row["expected_test_set_id"]
        binding = old_bindings[old_row["id"]]
        if binding != _native_observation_binding(row, set_id):
            raise InventoryError(
                f"existing native observation binding changed: {old_row['id']}"
            )
        row["expected_test_set_id"] = set_id
        row["expectation_state"] = FROZEN_STATE
        sets_by_id[set_id] = old_sets[set_id]
        native_bindings_by_row[row["id"]] = binding

    known_environments = {row["id"] for row in refreshed["environment_profiles"]}
    seen_observations: set[tuple[str, str, str]] = set()
    protocol_totals = _ProtocolTotals()
    for environment_id, path in bindings:
        if environment_id not in known_environments:
            raise InventoryError(f"unknown protocol environment {environment_id!r}")
        binding_class_id = _enumeration_class_id("zig", environment_id)
        for observation in _parse_protocol_log(path, protocol_totals):
            root_id = observation["root_id"]
            mode = observation["mode"]
            identity = (environment_id, root_id, mode)
            if identity in seen_observations:
                raise InventoryError(
                    f"duplicate compiler enumeration protocol observation {identity}"
                )
            seen_observations.add(identity)
            row_id = f"row:{_slug(environment_id)}:{_slug(root_id)}:{mode}"
            row = rows_by_id.get(row_id)
            if row is None or not root_id.startswith("zig-root:"):
                raise InventoryError(f"protocol block has no matrix row: {row_id}")
            if row["disposition"] != "execute":
                raise InventoryError(
                    f"protocol block targets inapplicable row: {row_id}"
                )
            class_id = observation["class_id"]
            if (
                class_id != binding_class_id
                or row["enumeration_class_id"] != binding_class_id
            ):
                raise InventoryError(f"protocol enumeration class mismatch: {row_id}")
            tests = _expected_test_rows(root_id, observation["names"])
            digest = _fact_digest(tests)
            set_id = _content_set_id(root_id, tests)
            expected_set = {
                "id": set_id,
                "root_id": root_id,
                "tests": tests,
                "count": len(tests),
                "digest": digest,
                "enumeration_source": ZIG_ENUMERATION_SOURCE,
            }
            prior = sets_by_id.get(set_id)
            if prior is not None and prior != expected_set:
                raise InventoryError(f"expected-set digest collision: {set_id}")
            sets_by_id[set_id] = expected_set
            row["expected_test_set_id"] = set_id
            row["expectation_state"] = FROZEN_STATE
            native_bindings_by_row[row_id] = _native_observation_binding(row, set_id)

    referenced = {
        row["expected_test_set_id"]
        for row in refreshed["test_mode_rows"]
        if row["expected_test_set_id"] is not None
    }
    refreshed["expected_test_sets"] = sorted(
        (sets_by_id[set_id] for set_id in referenced), key=lambda row: row["id"]
    )
    refreshed["native_observation_bindings"] = sorted(
        native_bindings_by_row.values(), key=lambda row: row["id"]
    )
    _refresh_native_gaps(refreshed)
    refreshed["strict_summary"] = _section_summary(refreshed)
    candidate_bytes = _canonical_inventory_bytes(refreshed)
    candidate_errors = _validate_inventory_data(
        root,
        inventory_path,
        refreshed,
        structure_only=True,
        inventory_bytes=candidate_bytes,
        filesystem_snapshot=existing_snapshot,
    )
    if candidate_errors:
        raise InventoryError(
            "refreshed inventory is invalid: " + "; ".join(candidate_errors)
        )
    return RefreshedInventoryCandidate(
        inventory=refreshed,
        bytes=candidate_bytes,
        expected_snapshot=existing_snapshot,
        incomplete_count=_matrix_incomplete_count(refreshed),
    )


def _privacy_errors(value: Any, path: str = "inventory") -> list[str]:
    errors: list[str] = []
    forbidden_keys = {
        "resolved_command",
        "result",
        "exit",
        "exit_code",
        "hostname",
        "runner",
        "timestamp",
        "run_id",
        "log",
        "private_identity",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden_keys:
                errors.append(f"public privacy contract forbids key {path}.{key}")
            errors.extend(_privacy_errors(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_privacy_errors(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if ".local-docs" in lowered or re.search(r"(?:^|\s)/(?:users|home)/", lowered):
            errors.append(f"public privacy contract forbids private path at {path}")
        if re.search(
            r"\b(?:timestamp|run[ _-]?id|hostname|resolved[ _-]?command)\b", lowered
        ):
            errors.append(f"public privacy contract forbids private evidence at {path}")
    return errors


def _validate_runner_source(context: Any) -> list[str]:
    path = "tools/test_inventory_runner.zig"
    text = _cached_text(context, path)
    requirements = (
        '"zynum-test-inventory-v3"',
        "builtin.test_functions",
        "@tagName(builtin.mode)",
        "TestEnumerationClass",
        "TestModeRow",
        "enumeration_class_id",
        "expected_test_set_id",
        "NativeObservationBinding",
        "native_observation_bindings",
        "expectation_state",
        ZIG_ENUMERATION_SOURCE,
        "--inventory-environment",
        "environment_id",
        "resolved_cpu_model",
        "cpu_feature_policy",
        "canonical-baseline-resolved-features",
        "std.Target.Cpu.baseline(builtin.cpu.arch, builtin.os)",
        "builtin.cpu.features.eql(baseline_cpu.features)",
        "const frozen_size: usize = @intCast(before.size)",
        "file.readPositionalAll(runner_io, bytes, 0)",
        "file.readPositionalAll(runner_io, &growth_probe, before.size)",
        "inventoryMetadataStable(before, after)",
        "after.inode == before.inode",
        "after.mtime.nanoseconds == before.mtime.nanoseconds",
        "after.ctime.nanoseconds == before.ctime.nanoseconds",
    )
    errors = [
        f"{path}: protocol identity changed"
        for item in requirements
        if item not in text
    ]
    if re.search(r"\.func\s*\(", text):
        errors.append(f"{path}: inventory runner must not execute test bodies")
    validation = text.find("const validation = try validateInventory")
    emission = text.find("try emitProtocol")
    if validation < 0 or emission < 0 or validation >= emission:
        errors.append(f"{path}: inventory validation must precede protocol emission")
    if "ZYNUM-TEST-INVENTORY-V1" in text or "ZYNUM-TEST-INVENTORY-V2" not in text:
        errors.append(f"{path}: protocol marker must be strict V2")
    row_selection = text.find(
        "std.mem.eql(u8, candidate.environment_id, arguments.environment_id)"
    )
    row_validation = text.find("try validateRowIdentity")
    environment_validation = text.find("try validateEnvironmentProfile")
    if (
        row_selection < 0
        or row_validation < 0
        or environment_validation < 0
        or row_selection >= row_validation
        or row_validation >= environment_validation
    ):
        errors.append(
            f"{path}: runner must bind environment/root/mode before CPU attestation"
        )
    if "builtin.cpu.model.features" in text:
        errors.append(f"{path}: CPU attestation must use resolved baseline features")
    current_matches = re.findall(
        r'^const CURRENT_TEST_INVENTORY_SHA256: \[\]const u8 = "([0-9a-f]{64})";$',
        text,
        flags=re.MULTILINE,
    )
    next_matches = re.findall(
        r"^const NEXT_TEST_INVENTORY_SHA256: \?\[\]const u8 = "
        r'(null|"[0-9a-f]{64}");$',
        text,
        flags=re.MULTILINE,
    )
    if len(current_matches) != 1 or len(next_matches) != 1:
        errors.append(
            f"{path}: runner whole-file digest constants must use the unique strict format"
        )
    else:
        runner_next = None if next_matches[0] == "null" else next_matches[0].strip('"')
        if (
            current_matches[0] != CURRENT_TEST_INVENTORY_SHA256
            or runner_next != NEXT_TEST_INVENTORY_SHA256
        ):
            errors.append(
                f"{path}: runner whole-file digest constants must match the checker"
            )
    digest_validation = text.find("try validateInventoryDigest(bytes)")
    json_parse = text.find("std.json.parseFromSlice")
    if digest_validation < 0 or json_parse < 0 or digest_validation >= json_parse:
        errors.append(f"{path}: whole-file digest validation must precede JSON parsing")
    if "std.crypto.hash.sha2.Sha256.hash(bytes" not in text:
        errors.append(f"{path}: runner must hash the exact bounded inventory bytes")
    if ".follow_symlinks = false" not in text or ".NOFOLLOW = true" not in text:
        errors.append(
            f"{path}: runner must reject symlink and reparse-point inventory paths"
        )
    anchored_path_requirements = (
        "const directory = try Io.Dir.cwd().openDir",
        "const file = try openInventoryFile(directory, basename)",
        "const admitted_path = try directory.statFile",
        "inventoryMetadataStable(before, admitted_path)",
        "std.posix.openat(directory.handle, basename",
    )
    if any(requirement not in text for requirement in anchored_path_requirements):
        errors.append(
            f"{path}: runner must retain the containing directory and reject "
            "post-read pathname rebinding"
        )
    if "allocRemaining" in text:
        errors.append(f"{path}: runner must read exactly the frozen descriptor size")
    return errors


def _validate_build_cpu_source(context: Any) -> list[str]:
    path = "build.zig"
    text = _cached_text(context, path)
    requirements = (
        "standardTargetOptionsQueryOnly",
        "target_query.cpu_model == .baseline",
        "target_query.cpu_features_add.isEmpty()",
        "target_query.cpu_features_sub.isEmpty()",
        "std.Target.Cpu.baseline(",
        "--inventory-environment",
    )
    errors = [
        f"{path}: canonical baseline CPU query provenance changed"
        for item in requirements
        if item not in text
    ]
    if "target.result.cpu.model.features" in text:
        errors.append(
            f"{path}: resolved CPU gate must compare against std.Target.Cpu.baseline"
        )
    return errors


def _validate_factory_projection(
    build_inventory: dict[str, Any], discovered: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    observations = build_inventory["build_observations"]
    factories = [
        row for row in observations if row.get("artifact_role") == FACTORY_ROLE
    ]
    if len(factories) != 1:
        return [
            f"build inventory must contain exactly one {FACTORY_ROLE!r} observation"
        ]
    factory = factories[0]
    if factory.get("id") != FACTORY_COMPILE_ID:
        errors.append("test inventory factory identity is incorrect")
    if factory.get("expansion_relation") != FACTORY_EXPANSION_RELATION:
        errors.append("test inventory factory expansion relation is incorrect")
    if factory.get("root_module_relation") != FACTORY_ROOT_MODULE_RELATION:
        errors.append("test inventory factory root-module relation is incorrect")
    if factory.get("test_runner") != {
        "path": "tools/test_inventory_runner.zig",
        "mode": "simple",
    }:
        errors.append("test inventory factory runner mapping is incorrect")
    expected_cases = sorted(
        (
            {
                "root_id": root["id"],
                "logical_compile_observation_id": root["compile_observation_id"],
                "predicate_id": root["aggregate_predicate_id"],
            }
            for root in discovered["test_roots"]
            if root["language"] == "zig"
        ),
        key=lambda row: row["root_id"],
    )
    cases = factory.get("expansion_cases")
    if cases != expected_cases:
        errors.append(
            "test inventory factory expansion cases do not map Zig roots exactly"
        )
    if factory.get("expansion_case_count") != len(expected_cases):
        errors.append("test inventory factory expansion count is incorrect")
    if factory.get("expansion_cases_digest") != _fact_digest(expected_cases):
        errors.append("test inventory factory expansion digest is incorrect")
    zig_classes = [
        row
        for row in discovered["test_enumeration_classes"]
        if row["language"] == "zig"
    ]
    expected_projection = {
        "kind": "optional-exact-baseline-target-cpu-profile-mapping",
        "query_source": "standardTargetOptionsQueryOnly",
        "resolution_source": "resolveTargetQuery(target_query)",
        "requested_cpu_gate": {
            "cpu_model": "baseline",
            "features_add": "empty",
            "features_sub": "empty",
        },
        "resolved_cpu_gate": {
            "model": "canonical-baseline-resolved-model",
            "features": "canonical-baseline-resolved-features",
        },
        "mappings": [
            {
                "architecture": row["architecture"],
                "os": row["os"],
                "abi": "none" if row["libc"] == "system" else row["libc"],
                "object_format": row["object_format"],
                "cpu_model": next(
                    environment["resolved_cpu_model"]
                    for environment in ENVIRONMENTS
                    if environment["id"] == row["environment_ids"][0]
                ),
                "environment_id": row["environment_ids"][0],
                "enumeration_class_id": row["id"],
            }
            for row in zig_classes
        ],
        "fallback": None,
        "known_branch": FACTORY_ROLE,
        "unknown_branch_dependency": (
            BUILD_CHECKER.TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID
        ),
    }
    if factory.get("enumeration_class_projection") != expected_projection:
        errors.append(
            "test inventory factory optional enumeration projection is incorrect"
        )

    launches = [
        row
        for row in observations
        if row.get("launch_role") == "test-inventory-enumerator-factory-run"
    ]
    if len(launches) != 1:
        errors.append(
            "build inventory must contain exactly one enumerator factory launch"
        )
    else:
        launch = launches[0]
        if launch.get("source_factory") != factory.get("id"):
            errors.append("test inventory factory launch foreign key is incorrect")
        if launch.get("id") != FACTORY_LAUNCH_ID:
            errors.append("test inventory factory launch identity is incorrect")
        expected_argument_contract = {
            "cardinality": "one-launch-per-applicable-expansion-case",
            "ordered_argv": [
                {"kind": "repository-path", "value": "tools/test_inventory.json"},
                {"kind": "literal", "value": "--inventory-root"},
                {"kind": "expansion-field", "value": "root_id"},
                {"kind": "literal", "value": "--inventory-mode"},
                {"kind": "build-option-tag", "value": "test-optimize"},
                {"kind": "literal", "value": "--inventory-environment"},
                {"kind": "target-derived", "value": "environment-id"},
                {"kind": "literal", "value": "--inventory-class"},
                {"kind": "target-derived", "value": "enumeration-class-id"},
            ],
        }
        if launch.get("argument_contract") != expected_argument_contract:
            errors.append(
                "test inventory factory launch argument contract is incorrect"
            )

    by_id = {row.get("id"): row for row in observations}
    expected_step_edges = {
        "step:build.zig:build:test-inventory-link": [
            {
                "id": FACTORY_COMPILE_ID,
                "condition": "per applicable expansion case for an exact baseline target CPU profile",
            },
            {
                "id": BUILD_CHECKER.TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID,
                "condition": "unknown or nonbaseline target CPU profile",
            },
        ],
        "step:build.zig:build:test-inventory": [
            {
                "id": FACTORY_LAUNCH_ID,
                "condition": "per applicable expansion case for an exact baseline target CPU profile",
            },
            {
                "id": BUILD_CHECKER.TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID,
                "condition": "unknown or nonbaseline target CPU profile",
            },
        ],
    }
    for step_id, dependencies in expected_step_edges.items():
        if by_id.get(step_id, {}).get("direct_dependencies") != dependencies:
            errors.append(f"{step_id}: test inventory factory step wiring is incorrect")
    aggregate_dependencies = by_id.get(AGGREGATE_STEP_ID, {}).get(
        "direct_dependencies", []
    )
    if (
        sum(
            isinstance(edge, dict)
            and edge.get("id") == "step:build.zig:build:test-inventory"
            and edge.get("condition") == "always"
            for edge in aggregate_dependencies
        )
        != 1
    ):
        errors.append("canonical test aggregate factory foreign key is incorrect")
    return errors


def _validate_python_skip_contracts(
    root: Path, inventory: dict[str, Any], discovered: dict[str, Any]
) -> list[str]:
    contracts = inventory.get("python_skip_contracts")
    if not isinstance(contracts, list) or len(contracts) != 1:
        return ["python_skip_contracts must contain exactly one root contract"]
    contract = contracts[0]
    if not isinstance(contract, dict) or set(contract) != {
        "root_id",
        "entries",
        "count",
        "digest",
    }:
        return ["python_skip_contracts root contract has unknown or missing keys"]
    if contract.get("root_id") != PYTHON_TOOLING_ROOT_ID:
        return ["python_skip_contracts root foreign key is noncanonical"]
    entries = contract.get("entries")
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        return ["python_skip_contracts entries must be an array of objects"]
    errors: list[str] = []
    exact_entry_keys = {
        "test",
        "reason",
        "predicate_id",
        "skip_kind",
        "predicate_ast_sha256",
    }
    if entries != sorted(
        entries,
        key=lambda entry: (
            entry.get("test", ""),
            entry.get("reason", ""),
            entry.get("predicate_id", ""),
            entry.get("skip_kind", ""),
            entry.get("predicate_ast_sha256", ""),
        ),
    ):
        errors.append("python_skip_contracts entries must be strictly sorted")
    entry_facts: list[tuple[str, str, str, str, str]] = []
    applicability_by_test: dict[str, str] = {}
    applicability_counts: Counter[str] = Counter()
    for entry in entries:
        if set(entry) != exact_entry_keys:
            errors.append("python_skip_contracts entry has unknown or missing keys")
            continue
        test_name = entry.get("test")
        reason = entry.get("reason")
        predicate_id = entry.get("predicate_id")
        skip_kind = entry.get("skip_kind")
        predicate_ast_sha256 = entry.get("predicate_ast_sha256")
        if (
            not isinstance(test_name, str)
            or not test_name
            or not isinstance(reason, str)
            or not reason
            or predicate_id not in PYTHON_SKIP_PREDICATE_IDS
            or not isinstance(skip_kind, str)
            or not isinstance(predicate_ast_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", predicate_ast_sha256)
        ):
            errors.append("python_skip_contracts entry is noncanonical")
            continue
        if PYTHON_SKIP_PREDICATE_SOURCE_BINDINGS[predicate_id] != (
            skip_kind,
            predicate_ast_sha256,
        ):
            errors.append(
                "python_skip_contracts predicate_id has a noncanonical "
                "skip kind or predicate AST digest"
            )
        entry_facts.append(
            (test_name, reason, predicate_id, skip_kind, predicate_ast_sha256)
        )
        if skip_kind == PYTHON_INVENTORY_PLATFORM_SKIP_KIND:
            if predicate_id not in PYTHON_INVENTORY_PLATFORM_PREDICATE_IDS:
                errors.append(
                    "python_skip_contracts applicability predicate is noncanonical"
                )
            previous = applicability_by_test.setdefault(test_name, predicate_id)
            if previous != predicate_id or applicability_counts[test_name] != 0:
                errors.append(
                    "python_skip_contracts contain multiple platform applicability "
                    f"entries for {test_name}"
                )
            applicability_counts[test_name] += 1
            if reason != PYTHON_INVENTORY_PLATFORM_REASONS.get(predicate_id):
                errors.append(
                    "python_skip_contracts platform applicability reason is noncanonical"
                )
    if len(entry_facts) != len(set(entry_facts)):
        errors.append("python_skip_contracts entries contain duplicates")
    if contract.get("count") != len(entries) or contract.get("digest") != _fact_digest(
        entries
    ):
        errors.append("python_skip_contracts count/digest mismatch")

    tooling_root = next(
        (
            row
            for row in discovered["test_roots"]
            if row["id"] == PYTHON_TOOLING_ROOT_ID
        ),
        None,
    )
    tooling_set = next(
        (
            row
            for row in discovered["expected_test_sets"]
            if row["root_id"] == PYTHON_TOOLING_ROOT_ID
        ),
        None,
    )
    if tooling_root is None or tooling_set is None:
        return [*errors, "Python tooling root skip-contract target is missing"]
    expected_names = {test["name"] for test in tooling_set["tests"]}
    contract_names = {test_name for test_name, _, _, _, _ in entry_facts}
    unknown_names = contract_names - expected_names
    if unknown_names:
        errors.append(
            "python_skip_contracts contain tests outside the expected set: "
            + ", ".join(sorted(unknown_names))
        )
    observed_applicability_counts = Counter(applicability_by_test.values())
    if observed_applicability_counts != Counter(PYTHON_INVENTORY_PLATFORM_COUNTS):
        errors.append(
            "python_skip_contracts platform applicability must contain exact "
            "artifact/publication identity counts"
        )
    try:
        source_decorators, source_dynamic = _python_tooling_source_skip_contract(
            root,
            tooling_root["module_paths"],
            tooling_root["discovery_start"],
            tooling_root["discovery_pattern"],
        )
    except InventoryError as exc:
        errors.append(f"python_skip_contracts source declaration failed: {exc}")
    else:
        source_facts = source_decorators | source_dynamic
        inventory_facts = frozenset(
            _PythonSkipSourceFact(
                _unittest_runtime_id(test_name),
                reason,
                skip_kind,
                predicate_ast_sha256,
            )
            for test_name, reason, predicate_id, skip_kind, predicate_ast_sha256 in entry_facts
            if predicate_id not in PYTHON_INVENTORY_PLATFORM_PREDICATE_IDS
        )
        if inventory_facts != source_facts:
            errors.append(
                "python_skip_contracts must cover every exact source-declared "
                "skip identity/reason/kind/predicate"
            )
    return errors


def _validate_runtime_bound_sections(
    root: Path, inventory: dict[str, Any], discovered: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    errors.extend(_validate_python_skip_contracts(root, inventory, discovered))
    roots = {row["id"]: row for row in discovered["test_roots"]}
    if inventory.get("test_enumeration_classes") != discovered.get(
        "test_enumeration_classes"
    ):
        errors.append(
            "test_enumeration_classes must match the environment/language matrix"
        )
    classes = {
        row["id"]: row
        for row in inventory.get("test_enumeration_classes", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    recorded_sets = inventory.get("expected_test_sets")
    if not isinstance(recorded_sets, list) or not all(
        isinstance(row, dict) for row in recorded_sets
    ):
        return ["expected_test_sets must be an array of objects"]
    sets_by_id: dict[str, dict[str, Any]] = {}
    sets_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_set_keys = {
        "id",
        "root_id",
        "tests",
        "count",
        "digest",
        "enumeration_source",
    }
    for row in recorded_sets:
        if set(row) != exact_set_keys:
            errors.append("expected_test_sets: unknown key or missing required key")
            continue
        root_id = row.get("root_id")
        set_id = row.get("id")
        if not isinstance(root_id, str) or root_id not in roots:
            errors.append("expected_test_sets: noncanonical root foreign key")
            continue
        if not isinstance(set_id, str) or set_id in sets_by_id:
            errors.append("expected_test_sets: duplicate or noncanonical set ID")
            continue
        sets_by_id[set_id] = row
        sets_by_root[root_id].append(row)

    discovered_python_sets = {
        row["id"]: row
        for row in discovered["expected_test_sets"]
        if row["root_id"].startswith("python-root:")
    }
    for set_id, expected_set in discovered_python_sets.items():
        if sets_by_id.get(set_id) != expected_set:
            errors.append(f"{expected_set['root_id']}: Python expected set changed")

    recorded_rows = inventory.get("test_mode_rows")
    if not isinstance(recorded_rows, list) or not all(
        isinstance(row, dict) for row in recorded_rows
    ):
        return [*errors, "test_mode_rows must be an array of objects"]
    discovered_rows = {row["id"]: row for row in discovered["test_mode_rows"]}
    rows_by_id = {row.get("id"): row for row in recorded_rows}
    if set(rows_by_id) != set(discovered_rows):
        errors.append("test_mode_rows must cover the matrix exactly")
    recorded_bindings = inventory.get("native_observation_bindings")
    if not isinstance(recorded_bindings, list) or not all(
        isinstance(binding, dict) for binding in recorded_bindings
    ):
        return [*errors, "native_observation_bindings must be an array of objects"]
    exact_binding_keys = {
        "id",
        "row_id",
        "evidence_slot_id",
        "enumeration_class_id",
        "optimize_mode_id",
        "expected_test_set_id",
        "enumeration_source",
        "digest",
    }
    if recorded_bindings != sorted(
        recorded_bindings, key=lambda binding: binding.get("id", "")
    ):
        errors.append("native_observation_bindings must be strictly sorted by ID")
    bindings_by_row: dict[str, dict[str, Any]] = {}
    binding_ids: set[str] = set()
    for binding in recorded_bindings:
        if set(binding) != exact_binding_keys:
            errors.append(
                "native_observation_bindings: unknown key or missing required key"
            )
            continue
        row_id = binding.get("row_id")
        binding_id = binding.get("id")
        if not isinstance(row_id, str) or row_id in bindings_by_row:
            errors.append(
                "native_observation_bindings: duplicate or invalid row foreign key"
            )
            continue
        if not isinstance(binding_id, str) or binding_id in binding_ids:
            errors.append(
                "native_observation_bindings: duplicate or invalid binding ID"
            )
            continue
        binding_ids.add(binding_id)
        bindings_by_row[row_id] = binding
        facts = {
            key: binding.get(key)
            for key in (
                "row_id",
                "evidence_slot_id",
                "enumeration_class_id",
                "optimize_mode_id",
                "expected_test_set_id",
                "enumeration_source",
            )
        }
        if not all(isinstance(value, str) for value in facts.values()):
            errors.append(f"{binding_id}: native observation facts are malformed")
            continue
        digest = _fact_digest(facts)
        if (
            binding.get("digest") != digest
            or binding_id != f"native-observation:{digest}"
        ):
            errors.append(
                f"{binding_id}: native observation identity is not content-bound"
            )
    runtime_fields = {"expected_test_set_id", "expectation_state"}
    referenced_set_ids: set[str] = set()
    pending_row_ids: list[str] = []
    for row_id, expected_row in discovered_rows.items():
        row = rows_by_id.get(row_id)
        if not isinstance(row, dict):
            continue
        if {key: value for key, value in row.items() if key not in runtime_fields} != {
            key: value
            for key, value in expected_row.items()
            if key not in runtime_fields
        }:
            errors.append(f"{row_id}: immutable matrix fields changed")
        root = roots[expected_row["root_id"]]
        class_row = classes.get(row.get("enumeration_class_id"))
        if class_row is None:
            errors.append(f"{row_id}: enumeration class foreign key is invalid")
        elif (
            class_row.get("language") != root["language"]
            or (
                root["language"] == "zig"
                and class_row.get("environment_ids") != [row["environment_id"]]
            )
            or (
                root["language"] == "python"
                and row["environment_id"] not in class_row.get("environment_ids", [])
            )
        ):
            errors.append(f"{row_id}: enumeration class mapping is incorrect")
        state = row.get("expectation_state")
        set_id = row.get("expected_test_set_id")
        binding = bindings_by_row.get(row_id)
        if state == "not-applicable":
            if row["disposition"] == "execute" or set_id is not None:
                errors.append(
                    f"{row_id}: not-applicable row has an executable expectation"
                )
            if binding is not None:
                errors.append(f"{row_id}: not-applicable row has native evidence")
        elif state == PENDING_STATE:
            if (
                row["disposition"] != "execute"
                or root["language"] != "zig"
                or set_id is not None
            ):
                errors.append(f"{row_id}: requires-native state is inconsistent")
            if binding is not None:
                errors.append(f"{row_id}: pending row has native evidence")
            pending_row_ids.append(row_id)
        elif state == FROZEN_STATE:
            if row["disposition"] != "execute" or not isinstance(set_id, str):
                errors.append(f"{row_id}: frozen row lacks an expected-set foreign key")
            elif set_id not in sets_by_id:
                errors.append(f"{row_id}: expected-set foreign key is invalid")
            elif sets_by_id[set_id]["root_id"] != row["root_id"]:
                errors.append(f"{row_id}: expected-set root foreign key is incorrect")
            else:
                referenced_set_ids.add(set_id)
            if root["language"] == "zig":
                if not isinstance(
                    set_id, str
                ) or binding != _native_observation_binding(row, set_id):
                    errors.append(
                        f"{row_id}: frozen Zig row lacks its exact native observation binding"
                    )
            elif binding is not None:
                errors.append(f"{row_id}: Python row must not have native evidence")
        else:
            errors.append(f"{row_id}: invalid expectation state")

    unknown_binding_rows = set(bindings_by_row) - set(discovered_rows)
    if unknown_binding_rows:
        errors.append(
            "native_observation_bindings contain unknown rows: "
            + ", ".join(sorted(unknown_binding_rows))
        )

    orphaned = set(sets_by_id) - referenced_set_ids
    if orphaned:
        errors.append(
            "expected_test_sets contain orphan variants: " + ", ".join(sorted(orphaned))
        )

    declaration_names: dict[str, set[str]] = {}
    static_reach: dict[str, set[str]] = {}
    for row in discovered["zig_test_files"]:
        path = row["path"]
        declaration_names[path] = {
            (
                f".test.{declaration['name']}"
                if declaration["kind"] == "named"
                else f".{declaration['name']}"
            )
            for declaration in row["declarations"]
        }
        static_reach[path] = set(row["reaching_root_ids"])
    names_by_root: dict[str, set[str]] = defaultdict(set)
    for root_id, root_sets in sets_by_root.items():
        for row in root_sets:
            tests = row.get("tests")
            if not isinstance(tests, list) or not tests:
                errors.append(f"{root_id}: expected set must be nonempty")
                continue
            names = [item.get("name") for item in tests if isinstance(item, dict)]
            if len(names) != len(tests) or any(
                not isinstance(name, str) for name in names
            ):
                errors.append(f"{root_id}: expected test rows are malformed")
                continue
            if len(names) != len(set(names)):
                errors.append(f"{root_id}: expected test names are not unique")
            canonical_tests = _expected_test_rows(root_id, names)
            if tests != canonical_tests:
                errors.append(f"{root_id}: expected test IDs/order are noncanonical")
            digest = _fact_digest(tests)
            if row.get("count") != len(tests) or row.get("digest") != digest:
                errors.append(f"{root_id}: expected set count/digest mismatch")
            if row.get("id") != _content_set_id(root_id, tests):
                errors.append(f"{root_id}: expected set identity is not content-bound")
            source = row.get("enumeration_source")
            required_source = (
                ZIG_ENUMERATION_SOURCE
                if root_id.startswith("zig-root:")
                else PYTHON_ENUMERATION_SOURCE
            )
            if source != required_source:
                errors.append(f"{root_id}: enumeration source is incorrect")
            names_by_root[root_id].update(names)
            if root_id.startswith("zig-root:"):
                for name in names:
                    if not any(
                        any(name.endswith(suffix) for suffix in declarations)
                        and root_id in static_reach[path]
                        for path, declarations in declaration_names.items()
                    ):
                        errors.append(
                            f"{root_id}: compiler name has no reaching declaration: {name}"
                        )

    recorded_zig_files = inventory.get("zig_test_files")
    if not isinstance(recorded_zig_files, list) or not all(
        isinstance(row, dict) for row in recorded_zig_files
    ):
        return [*errors, "zig_test_files must be an array of objects"]
    if recorded_zig_files != discovered["zig_test_files"]:
        errors.append(
            "zig_test_files must match static declaration/reachability sanity"
        )

    expected_gaps = [
        gap
        for gap in discovered["known_gaps"]
        if gap["kind"] == "zig-test-file-without-reaching-root"
    ]
    for environment in discovered["environment_profiles"]:
        subject_ids = [
            row_id
            for row_id in pending_row_ids
            if rows_by_id[row_id]["environment_id"] == environment["id"]
        ]
        if subject_ids:
            expected_gaps.append(
                {
                    "id": f"gap:native-test-enumeration:{_slug(environment['id'])}",
                    "kind": "native-test-enumeration-required",
                    "subject_ids": subject_ids,
                }
            )
    if inventory.get("known_gaps") != sorted(expected_gaps, key=lambda row: row["id"]):
        errors.append("known_gaps must match exact current gaps")
    if inventory.get("strict_summary") != _section_summary(inventory):
        errors.append("strict_summary count/digest mismatch")
    return errors


def _matrix_incomplete_count(inventory: dict[str, Any]) -> int:
    rows = inventory.get("test_mode_rows")
    if not isinstance(rows, list):
        return 0
    return sum(
        isinstance(row, dict) and row.get("expectation_state") == PENDING_STATE
        for row in rows
    )


def _refresh_base_compatible_discovery(
    inventory: dict[str, Any], discovered: dict[str, Any]
) -> dict[str, Any]:
    """Reconstruct the only source-derived facts allowed to be stale at refresh."""
    recorded_roots = inventory.get("test_roots")
    current_roots = discovered["test_roots"]
    if (
        not isinstance(recorded_roots, list)
        or not all(isinstance(root, dict) for root in recorded_roots)
        or len(recorded_roots) != len(current_roots)
    ):
        raise InventoryError(
            "refresh base Python root identities must match current discovery"
        )
    for recorded, current in zip(recorded_roots, current_roots, strict=True):
        if recorded == current:
            continue
        legacy = dict(current)
        legacy["launch_observation_ids"] = []
        tooling_migration = (
            current.get("id") != PYTHON_TOOLING_ROOT_ID
            or recorded != legacy
            or current.get("launch_observation_ids") != [PYTHON_TOOLING_LAUNCH_ID]
            or current.get("aggregate_step_observation_id") != AGGREGATE_STEP_ID
            or current.get("matrix_applicable") is not False
        )
        build_inventory_legacy = dict(current)
        build_inventory_legacy["matrix_applicable"] = False
        build_inventory_older_legacy = dict(current)
        build_inventory_older_legacy["aggregate_step_observation_id"] = (
            AGGREGATE_STEP_ID
        )
        build_inventory_migration = (
            current.get("id") != "python-root:build-inventory-direct"
            or recorded not in (build_inventory_legacy, build_inventory_older_legacy)
            or current.get("aggregate_step_observation_id") is not None
            or current.get("matrix_applicable") is not True
        )
        if tooling_migration and build_inventory_migration:
            raise InventoryError("refresh base Python root immutable facts changed")

    recorded_modules = inventory.get("python_test_modules")
    current_modules = discovered["python_test_modules"]
    if not isinstance(recorded_modules, list) or not all(
        isinstance(module, dict) for module in recorded_modules
    ):
        raise InventoryError(
            "refresh base python_test_modules must be an array of objects"
        )
    if len(recorded_modules) != len(current_modules):
        raise InventoryError(
            "refresh base Python module identities must match current discovery"
        )

    module_keys = {
        "id",
        "path",
        "declarations",
        "root_ids",
        "launch_observation_ids",
        "gap_ids",
    }
    declaration_keys = {"id", "name", "ordinal"}
    modules_by_path: dict[str, dict[str, Any]] = {}
    for recorded, current in zip(recorded_modules, current_modules, strict=True):
        if set(recorded) != module_keys:
            raise InventoryError(
                "refresh base Python module has unknown or missing keys"
            )
        immutable = {
            key: value for key, value in recorded.items() if key != "declarations"
        }
        current_immutable = {
            key: value for key, value in current.items() if key != "declarations"
        }
        if immutable != current_immutable:
            legacy_immutable = dict(current_immutable)
            legacy_immutable["launch_observation_ids"] = []
            if (
                recorded.get("root_ids") != [PYTHON_TOOLING_ROOT_ID]
                or recorded.get("launch_observation_ids") != []
                or current.get("launch_observation_ids") != [PYTHON_TOOLING_LAUNCH_ID]
                or immutable != legacy_immutable
            ):
                raise InventoryError(
                    "refresh base Python module immutable facts changed"
                )
        path = recorded.get("path")
        module_id = recorded.get("id")
        if (
            not isinstance(path, str)
            or not path
            or module_id != f"python-module:{path}"
            or path in modules_by_path
        ):
            raise InventoryError("refresh base Python module identity is noncanonical")
        declarations = recorded.get("declarations")
        if not isinstance(declarations, list) or not declarations:
            raise InventoryError(
                f"refresh base Python declarations must be a nonempty array: {path}"
            )
        names: set[str] = set()
        for ordinal, declaration in enumerate(declarations):
            if (
                not isinstance(declaration, dict)
                or set(declaration) != declaration_keys
            ):
                raise InventoryError(
                    f"refresh base Python declaration has unknown or missing keys: {path}"
                )
            name = declaration.get("name")
            recorded_ordinal = declaration.get("ordinal")
            identifier = declaration.get("id")
            name_parts = name.split(".") if isinstance(name, str) else []
            if (
                type(recorded_ordinal) is not int
                or recorded_ordinal != ordinal
                or identifier != f"python-decl:{path}:{ordinal}"
                or len(name_parts) != 2
                or not all(part.isidentifier() for part in name_parts)
                or not name_parts[1].startswith("test_")
                or name in names
            ):
                raise InventoryError(
                    f"refresh base Python declaration identity/name/order is noncanonical: {path}"
                )
            names.add(name)
        modules_by_path[path] = recorded

    python_roots = [
        root for root in discovered["test_roots"] if root["language"] == "python"
    ]
    derived_sets: list[dict[str, Any]] = []
    derived_set_ids: dict[str, str] = {}
    for root in python_roots:
        root_id = root["id"]
        try:
            names = [
                f"{path}::{declaration['name']}"
                for path in root["module_paths"]
                for declaration in modules_by_path[path]["declarations"]
            ]
        except KeyError as exc:
            raise InventoryError(
                f"refresh base Python root module membership is incomplete: {root_id}"
            ) from exc
        tests = _expected_test_rows(root_id, names)
        set_id = _content_set_id(root_id, tests)
        derived_set_ids[root_id] = set_id
        derived_sets.append(
            {
                "id": set_id,
                "root_id": root_id,
                "tests": tests,
                "count": len(tests),
                "digest": _fact_digest(tests),
                "enumeration_source": PYTHON_ENUMERATION_SOURCE,
            }
        )
    derived_sets.sort(key=lambda row: row["id"])

    recorded_sets = inventory.get("expected_test_sets")
    if not isinstance(recorded_sets, list) or not all(
        isinstance(expected_set, dict) for expected_set in recorded_sets
    ):
        raise InventoryError(
            "refresh base expected_test_sets must be an array of objects"
        )
    recorded_python_sets = sorted(
        (
            expected_set
            for expected_set in recorded_sets
            if isinstance(expected_set.get("root_id"), str)
            and expected_set["root_id"].startswith("python-root:")
        ),
        key=lambda row: row.get("id", ""),
    )
    if recorded_python_sets != derived_sets:
        raise InventoryError(
            "refresh base Python expected sets do not match recorded declarations"
        )

    recorded_rows = inventory.get("test_mode_rows")
    current_rows = {row["id"]: row for row in discovered["test_mode_rows"]}
    if not isinstance(recorded_rows, list) or not all(
        isinstance(row, dict) for row in recorded_rows
    ):
        raise InventoryError("refresh base test_mode_rows must be an array of objects")
    for row in recorded_rows:
        root_id = row.get("root_id")
        if root_id not in derived_set_ids:
            continue
        current = current_rows.get(row.get("id"))
        if current is None or current.get("root_id") != root_id:
            raise InventoryError("refresh base Python row identity is noncanonical")
        if root_id == "python-root:build-inventory-direct":
            continue
        expected_set_id = (
            derived_set_ids[root_id] if current["disposition"] == "execute" else None
        )
        if row.get("expected_test_set_id") != expected_set_id:
            raise InventoryError(
                f"refresh base Python row expected-set foreign key is inconsistent: {row['id']}"
            )

    compatible = dict(discovered)
    compatible["test_roots"] = recorded_roots
    compatible["python_test_modules"] = recorded_modules
    compatible["expected_test_sets"] = derived_sets
    compatible["test_mode_rows"] = inventory["test_mode_rows"]
    recorded_workflow_bindings = inventory.get("workflow_mode_bindings")
    current_workflow_bindings = discovered["workflow_mode_bindings"]
    if not isinstance(recorded_workflow_bindings, list) or len(
        recorded_workflow_bindings
    ) != len(current_workflow_bindings):
        raise InventoryError(
            "refresh base workflow mode bindings must match current identities"
        )
    workflow_keys = {
        "id",
        "workflow_observation_id",
        "optimize_mode_id",
        "command_template",
    }
    for recorded, current in zip(
        recorded_workflow_bindings, current_workflow_bindings, strict=True
    ):
        if recorded == current:
            continue
        if (
            not isinstance(recorded, dict)
            or set(recorded) != workflow_keys
            or {key: recorded[key] for key in workflow_keys - {"command_template"}}
            != {key: current[key] for key in workflow_keys - {"command_template"}}
            or recorded["command_template"]
            != LEGACY_WORKFLOW_MODE_COMMANDS.get(recorded["workflow_observation_id"])
        ):
            raise InventoryError(
                "refresh base workflow mode binding identities or reviewed legacy command changed"
            )
    compatible["workflow_mode_bindings"] = recorded_workflow_bindings
    return compatible


def _validate_inventory_data(
    root: Path,
    inventory_path: Path,
    inventory: dict[str, Any],
    *,
    structure_only: bool,
    inventory_bytes: bytes | None = None,
    filesystem_snapshot: FrozenInventorySnapshot | None = None,
    _refresh_base: bool = False,
    require_current_only: bool = False,
    _context: Any | None = None,
) -> list[str]:
    if _refresh_base:
        try:
            return _validate_impl(
                root,
                inventory_path,
                structure_only=structure_only,
                _inventory=inventory,
                _inventory_bytes=inventory_bytes,
                _filesystem_snapshot=filesystem_snapshot,
                _refresh_base=True,
                require_current_only=require_current_only,
                _context=_context,
            )
        except RecursionError:
            return ["test inventory validation exceeded the recursion limit"]
    return validate(
        root,
        inventory_path,
        structure_only=structure_only,
        _inventory=inventory,
        _inventory_bytes=inventory_bytes,
        _filesystem_snapshot=filesystem_snapshot,
        require_current_only=require_current_only,
        _context=_context,
    )


def _validate_impl(
    root: Path,
    inventory_path: Path,
    *,
    structure_only: bool = False,
    _inventory: dict[str, Any] | None = None,
    _inventory_bytes: bytes | None = None,
    _filesystem_snapshot: FrozenInventorySnapshot | None = None,
    _refresh_base: bool = False,
    require_current_only: bool = False,
    _context: Any | None = None,
) -> list[str]:
    try:
        filesystem_snapshot = (
            _read_regular_stable_snapshot(
                inventory_path, MAX_INVENTORY_BYTES, "test inventory"
            )
            if _filesystem_snapshot is None
            else _filesystem_snapshot
        )
        if _inventory_bytes is not None and len(_inventory_bytes) > MAX_INVENTORY_BYTES:
            raise InventoryError(
                f"candidate inventory exceeds {MAX_INVENTORY_BYTES} bytes"
            )
        reviewed_bytes = (
            filesystem_snapshot.bytes if _inventory_bytes is None else _inventory_bytes
        )
        if require_current_only:
            slots_error = _current_only_slots_error()
            if slots_error is not None:
                raise InventoryError(slots_error)
        digest_error = _reviewed_inventory_bytes_error(
            reviewed_bytes, require_current_only=require_current_only
        )
        if digest_error is not None:
            raise InventoryError(digest_error)
        context = (
            BUILD_CHECKER._make_discovery_context(root, inventory_path)
            if _context is None
            else BUILD_CHECKER._context_for(root, _context)
        )
        node = context.inventory_node
        if node is None or node.bytes is None or node.sha256 is None:
            raise InventoryError("test inventory bytes were not frozen")
        node_identity = (
            node.identity.device,
            node.identity.inode,
            node.identity.size,
            node.identity.mtime_ns,
            node.identity.ctime_ns,
        )
        if (
            node.bytes != filesystem_snapshot.bytes
            or node.sha256 != filesystem_snapshot.sha256
            or node.identity.mode != filesystem_snapshot.mode
            or node_identity != filesystem_snapshot.identity
        ):
            raise InventoryError("test inventory changed after its bounded snapshot")
        if _inventory is None:
            inventory = _strict_json_loads(filesystem_snapshot.bytes.decode("utf-8"))
        else:
            inventory = _inventory
    except (
        InventoryError,
        BUILD_CHECKER.InventoryError,
        OSError,
        UnicodeError,
        ValueError,
        RecursionError,
    ) as exc:
        return [f"cannot read test inventory: {exc}"]
    structure_error = _json_structure_error(inventory)
    if structure_error is not None:
        return [structure_error]
    if not isinstance(inventory, dict):
        return ["test inventory root must be an object"]
    errors = _privacy_errors(inventory)
    if set(inventory) != TOP_LEVEL_KEYS:
        errors.append("test inventory top-level keys must match the schema exactly")
    if inventory.get("schema_id") != SCHEMA_ID:
        errors.append(f"schema_id must be {SCHEMA_ID}")
    if inventory.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if inventory.get("build_inventory_schema_id") != BUILD_SCHEMA_ID:
        errors.append("build inventory schema foreign key is incorrect")
    native_projection_error = _reviewed_native_projection_error(
        inventory, require_current_only=require_current_only
    )
    if native_projection_error is not None:
        errors.append(native_projection_error)
    try:
        expected = discover(root, inventory_path, _context=context)
        if _refresh_base:
            expected = _refresh_base_compatible_discovery(inventory, expected)
    except (
        InventoryError,
        BUILD_CHECKER.InventoryError,
        OSError,
        SyntaxError,
        ValueError,
        RecursionError,
    ) as exc:
        errors.append(f"test inventory discovery failed closed: {exc}")
        return errors
    for section in TOP_LEVEL_KEYS - {
        "schema_id",
        "schema_version",
        "build_inventory_schema_id",
        "expected_test_sets",
        "native_observation_bindings",
        "python_skip_contracts",
        "test_mode_rows",
        "zig_test_files",
        "known_gaps",
        "strict_summary",
    }:
        if inventory.get(section) != expected[section]:
            errors.append(
                f"{section} must exactly match independently discovered facts"
            )
    errors.extend(_validate_runtime_bound_sections(root, inventory, expected))
    try:
        build_inventory = _load_build_inventory(context)
    except (
        InventoryError,
        BUILD_CHECKER.InventoryError,
        OSError,
        ValueError,
        RecursionError,
    ) as exc:
        errors.append(f"cannot validate test enumerator factory: {exc}")
    else:
        errors.extend(_validate_factory_projection(build_inventory, expected))
    for section in (
        "test_roots",
        "zig_test_files",
        "python_test_modules",
        "expected_test_sets",
        "native_observation_bindings",
        "test_mode_rows",
    ):
        value = inventory.get(section)
        if isinstance(value, list):
            ids = [item.get("id") for item in value if isinstance(item, dict)]
            for identifier, count in Counter(ids).items():
                if count > 1:
                    errors.append(f"{section}: duplicate id {identifier!r}")
            if any(
                not isinstance(identifier, str) or not identifier for identifier in ids
            ):
                errors.append(f"{section}: noncanonical ID")
    sets = inventory.get("expected_test_sets")
    if isinstance(sets, list):
        for expected_set in sets:
            if not isinstance(expected_set, dict):
                continue
            tests = expected_set.get("tests")
            if not isinstance(tests, list) or not tests:
                errors.append(
                    f"{expected_set.get('id')}: expected set must be nonempty"
                )
                continue
            names = [item.get("name") for item in tests if isinstance(item, dict)]
            if len(names) != len(set(names)):
                errors.append(f"{expected_set.get('id')}: duplicate test name")
            if expected_set.get("count") != len(tests) or expected_set.get(
                "digest"
            ) != _fact_digest(tests):
                errors.append(f"{expected_set.get('id')}: count/digest mismatch")
    contract = inventory.get("matrix_row_contract")
    if isinstance(contract, dict):
        row_ids = contract.get("required_row_ids")
        if not isinstance(row_ids, list) or len(row_ids) != len(set(row_ids)):
            errors.append("matrix_row_contract contains missing or duplicate row IDs")
    mode_rows = inventory.get("test_mode_rows")
    if isinstance(mode_rows, list):
        evidence_slot_ids = [
            row.get("evidence_slot_id") for row in mode_rows if isinstance(row, dict)
        ]
        if len(evidence_slot_ids) != len(set(evidence_slot_ids)):
            errors.append("test_mode_rows contain duplicate evidence slot IDs")
        for row in mode_rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            if row.get("evidence_slot_id") != _evidence_slot_id(row["id"]):
                errors.append(
                    f"{row['id']}: evidence slot ID must be the stable row join key"
                )
    errors.extend(_validate_runner_source(context))
    errors.extend(_validate_build_cpu_source(context))
    incomplete = _matrix_incomplete_count(inventory)
    if incomplete and not structure_only:
        errors.append(
            f"matrix incomplete: {incomplete} rows require native compiler enumeration"
        )
    return errors


def validate(
    root: Path,
    inventory_path: Path,
    *,
    structure_only: bool = False,
    _inventory: dict[str, Any] | None = None,
    _inventory_bytes: bytes | None = None,
    _filesystem_snapshot: FrozenInventorySnapshot | None = None,
    require_current_only: bool = False,
    _context: Any | None = None,
) -> list[str]:
    try:
        return _validate_impl(
            root,
            inventory_path,
            structure_only=structure_only,
            _inventory=_inventory,
            _inventory_bytes=_inventory_bytes,
            _filesystem_snapshot=_filesystem_snapshot,
            require_current_only=require_current_only,
            _context=_context,
        )
    except RecursionError:
        return ["test inventory validation exceeded the recursion limit"]


def _publication_capability_error() -> str | None:
    if os.name != "posix":
        return "test inventory refresh publication requires POSIX filesystem semantics"
    dir_fd_functions = os.supports_dir_fd
    for function in (os.mkdir, os.open, os.rename, os.rmdir, os.stat, os.unlink):
        if function not in dir_fd_functions:
            return (
                "test inventory refresh publication requires anchored dir_fd support for "
                f"{function.__name__}"
            )
    follow_functions = os.supports_follow_symlinks
    if os.stat not in follow_functions:
        return "test inventory refresh publication requires no-follow anchored stat support"
    return None


def _claim_and_remove_inventory_temporary(
    directory_descriptor: int,
    temporary_name: str,
    expected_identity: tuple[int, int],
    expected_bytes: bytes,
    *,
    directory_path: Path | None = None,
) -> None:
    """Claim and remove one temporary through the shared cleanup state machine."""
    if directory_path is None:
        directory_path = Path(os.path.realpath(f"/dev/fd/{directory_descriptor}"))

    def verify_claimed(descriptor: int, path_metadata: os.stat_result) -> Any:
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or (path_metadata.st_dev, path_metadata.st_ino) != expected_identity
        ):
            return REPOSITORY_SNAPSHOT.ClaimVerification.FOREIGN
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino) != expected_identity
                or before.st_size < 0
                or before.st_size > MAX_INVENTORY_BYTES
            ):
                return REPOSITORY_SNAPSHOT.ClaimVerification.FOREIGN
            chunks: list[bytes] = []
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor,
                    min(64 * 1024, before.st_size - offset),
                    offset,
                )
                if not chunk:
                    return REPOSITORY_SNAPSHOT.ClaimVerification.UNKNOWN
                chunks.append(chunk)
                offset += len(chunk)
            if os.pread(descriptor, 1, before.st_size):
                return REPOSITORY_SNAPSHOT.ClaimVerification.UNKNOWN
            after = os.fstat(descriptor)
        except OSError:
            return REPOSITORY_SNAPSHOT.ClaimVerification.UNKNOWN
        stable_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if stable_identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            return REPOSITORY_SNAPSHOT.ClaimVerification.UNKNOWN
        observed = b"".join(chunks)
        if (
            observed != expected_bytes
            or hashlib.sha256(observed).digest()
            != hashlib.sha256(expected_bytes).digest()
        ):
            return REPOSITORY_SNAPSHOT.ClaimVerification.FOREIGN
        return REPOSITORY_SNAPSHOT.ClaimVerification.MATCH

    anchor = REPOSITORY_SNAPSHOT.DirectoryAnchor(
        directory_descriptor,
        directory_path,
    )
    try:
        outcome = REPOSITORY_SNAPSHOT.claim_and_remove(
            anchor,
            temporary_name,
            verify_claimed,
            quarantine_prefix=f"{temporary_name}.",
            quarantine_suffix=".cleanup-quarantine",
            token_bytes=12,
            claimed_name="claimed",
            expect_public_absent=True,
        )
    except REPOSITORY_SNAPSHOT.CleanupFailure as exc:
        outcome = exc.outcome
    if outcome.disposition is REPOSITORY_SNAPSHOT.CleanupDisposition.REMOVED:
        return
    recovery_paths = (
        ", ".join(os.fspath(path) for path in outcome.recovery_paths) or "none"
    )
    candidate_paths = (
        ", ".join(os.fspath(path) for path in outcome.candidate_paths) or "none"
    )
    issue_codes = ", ".join(issue.code for issue in outcome.issues) or "cleanup_absent"
    raise InventoryError(
        "inventory temporary cleanup failed closed: "
        f"disposition={outcome.disposition.name.lower()}; "
        f"arena_binding={outcome.arena_binding.name.lower()}; "
        f"public_candidate={outcome.public_candidate.name.lower()}; "
        f"exact recovery paths: {recovery_paths}; "
        f"candidate paths: {candidate_paths}; issues: {issue_codes}"
    )


def _publish_inventory_atomic(
    inventory_path: Path,
    candidate_bytes: bytes,
    expected_snapshot: FrozenInventorySnapshot,
) -> None:
    """Optimistically reverify and atomically install already-validated bytes.

    The final descriptor-derived comparison rejects observed lost updates.  POSIX
    has no portable pathname compare-and-swap primitive, so this is deliberately
    an optimistic reverify immediately before a same-directory replacement.
    """
    if len(candidate_bytes) > MAX_INVENTORY_BYTES:
        raise InventoryError(f"candidate inventory exceeds {MAX_INVENTORY_BYTES} bytes")
    digest_error = _reviewed_inventory_bytes_error(candidate_bytes)
    if digest_error is not None:
        raise InventoryError(digest_error)
    try:
        candidate = _strict_json_loads(candidate_bytes.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InventoryError(
            f"publication candidate is not strict inventory JSON: {exc}"
        ) from exc
    structure_error = _json_structure_error(candidate)
    if structure_error is not None:
        raise InventoryError(structure_error)
    if not isinstance(candidate, dict):
        raise InventoryError("publication candidate inventory root must be an object")
    native_projection_error = _reviewed_native_projection_error(candidate)
    if native_projection_error is not None:
        raise InventoryError(native_projection_error)
    capability_error = _publication_capability_error()
    if capability_error is not None:
        raise InventoryError(capability_error)

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW

    directory_descriptor = -1
    descriptor = -1
    temporary_name: str | None = None
    temporary_identity: tuple[int, int] | None = None
    temporary_identity_unknown = False
    temporary_candidate_path: Path | None = None
    temporary_descriptor_close_error: OSError | None = None
    committed = False
    try:
        directory_descriptor = os.open(inventory_path.parent, directory_flags)
        for _ in range(128):
            candidate_name = f".{inventory_path.name}.{secrets.token_hex(12)}.tmp"
            temporary_flags = (
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            try:
                descriptor = os.open(
                    candidate_name,
                    temporary_flags,
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate_name
            temporary_identity_unknown = True
            temporary_candidate_path = inventory_path.with_name(candidate_name)
            try:
                metadata = os.fstat(descriptor)
            except OSError as exc:
                raise InventoryError(
                    "inventory publication temporary identity could not be "
                    "established; public temporary candidate path retained as "
                    f"{temporary_candidate_path}"
                ) from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise InventoryError(
                    "inventory publication temporary is not a regular file; "
                    "public temporary candidate path retained as "
                    f"{temporary_candidate_path}"
                )
            temporary_identity = (metadata.st_dev, metadata.st_ino)
            temporary_identity_unknown = False
            break
        else:
            raise OSError("cannot allocate a unique inventory temporary file")

        offset = 0
        while offset < len(candidate_bytes):
            written = os.write(descriptor, candidate_bytes[offset:])
            if written <= 0:
                raise OSError("short write while publishing inventory")
            offset += written
        os.fchmod(descriptor, stat.S_IMODE(expected_snapshot.mode))
        os.fsync(descriptor)
        closing_descriptor = descriptor
        descriptor = -1
        try:
            os.close(closing_descriptor)
        except OSError as exc:
            temporary_descriptor_close_error = exc
            raise

        if temporary_name is None or temporary_identity is None:
            raise InventoryError("inventory publication temporary identity is missing")
        temporary_snapshot = _read_regular_stable_snapshot(
            temporary_name,
            MAX_INVENTORY_BYTES,
            "inventory publication temporary",
            directory_fd=directory_descriptor,
        )
        if (
            temporary_snapshot.bytes != candidate_bytes
            or temporary_snapshot.sha256 != hashlib.sha256(candidate_bytes).hexdigest()
            or temporary_snapshot.identity[:2] != temporary_identity
            or temporary_snapshot.identity[2] != len(candidate_bytes)
            or temporary_snapshot.mode != expected_snapshot.mode
        ):
            raise InventoryError(
                "inventory publication temporary changed before replace"
            )

        observed_snapshot = _read_regular_stable_snapshot(
            inventory_path.name,
            MAX_INVENTORY_BYTES,
            "inventory publication target",
            directory_fd=directory_descriptor,
        )
        if observed_snapshot != expected_snapshot:
            raise InventoryError(
                "inventory changed since refresh; refusing lost-update publication"
            )
        os.replace(
            temporary_name,
            inventory_path.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        committed = True
        temporary_name = None
        temporary_identity = None
        temporary_identity_unknown = False
        temporary_candidate_path = None
        os.fsync(directory_descriptor)
        closing_directory = directory_descriptor
        directory_descriptor = -1
        os.close(closing_directory)
    except InventoryPublicationIndeterminate:
        raise
    except InventoryError:
        raise
    except OSError as exc:
        if committed:
            raise InventoryPublicationIndeterminate(
                f"candidate installed but durability uncertain: {exc}"
            ) from exc
        raise InventoryError(f"cannot publish refreshed inventory: {exc}") from exc
    finally:
        publication_error = sys.exc_info()[1]
        cleanup_error: InventoryError | None = None
        if descriptor >= 0:
            closing_descriptor = descriptor
            descriptor = -1
            try:
                os.close(closing_descriptor)
            except OSError as exc:
                if committed:
                    raise InventoryPublicationIndeterminate(
                        f"candidate installed but durability uncertain: {exc}"
                    ) from exc
                temporary_descriptor_close_error = exc
        if (
            temporary_descriptor_close_error is None
            and temporary_name is not None
            and temporary_identity is not None
            and not temporary_identity_unknown
            and directory_descriptor >= 0
        ):
            try:
                _claim_and_remove_inventory_temporary(
                    directory_descriptor,
                    temporary_name,
                    temporary_identity,
                    candidate_bytes,
                    directory_path=inventory_path.parent,
                )
            except InventoryError as exc:
                cleanup_error = exc
        if directory_descriptor >= 0:
            closing_directory = directory_descriptor
            directory_descriptor = -1
            try:
                os.close(closing_directory)
            except OSError as exc:
                if committed:
                    raise InventoryPublicationIndeterminate(
                        f"candidate installed but durability uncertain: {exc}"
                    ) from exc
        if temporary_descriptor_close_error is not None:
            close_error = (
                "inventory publication temporary descriptor close failed: "
                f"{temporary_descriptor_close_error}"
            )
            if temporary_candidate_path is not None:
                close_error += (
                    "; public temporary candidate path retained as "
                    f"{temporary_candidate_path}"
                )
            if publication_error is not None:
                close_error = f"{publication_error}; {close_error}"
            if cleanup_error is not None:
                close_error += f"; inventory temporary cleanup failed: {cleanup_error}"
            raise InventoryError(close_error) from temporary_descriptor_close_error
        if cleanup_error is not None:
            if publication_error is not None:
                raise InventoryError(
                    f"{publication_error}; inventory temporary cleanup failed: "
                    f"{cleanup_error}"
                ) from cleanup_error
            raise cleanup_error


def _unittest_runtime_id(inventory_name: str) -> str:
    path, separator, declaration = inventory_name.partition("::")
    if (
        not separator
        or not path
        or not declaration
        or PurePosixPath(path).suffix != ".py"
    ):
        raise InventoryError(
            f"Python tooling test identity is noncanonical: {inventory_name!r}"
        )
    return f"{PurePosixPath(path).stem}.{declaration}"


def _predicate_ast_sha256(predicate: ast.expr) -> str:
    if "show_empty" in inspect.signature(ast.dump).parameters:
        canonical = ast.dump(
            predicate,
            annotate_fields=True,
            include_attributes=False,
            show_empty=True,
        )
    else:
        canonical = ast.dump(predicate, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_PYTHON_UNITTEST_SKIP_BINDINGS = frozenset(
    {"unittest.skip", "unittest.skipIf", "unittest.skipUnless"}
)
_PYTHON_ALLOWED_TEST_DECORATOR_BINDINGS = frozenset({"unittest.mock.patch.object"})


def _python_static_binding(
    expression: ast.expr, bindings: dict[str, str]
) -> str | None:
    if isinstance(expression, ast.Name):
        return bindings.get(expression.id)
    if isinstance(expression, ast.Attribute):
        base = _python_static_binding(expression.value, bindings)
        return f"{base}.{expression.attr}" if base is not None else None
    if isinstance(expression, ast.Call):
        called = _python_static_binding(expression.func, bindings)
        if called in _PYTHON_UNITTEST_SKIP_BINDINGS:
            return f"{called}()"
    return None


def _python_bound_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {
            name for element in target.elts for name in _python_bound_names(element)
        }
    return set()


def _update_python_static_bindings(
    statement: ast.stmt, bindings: dict[str, str]
) -> None:
    if isinstance(statement, ast.Import):
        for imported in statement.names:
            local_name = imported.asname or imported.name.partition(".")[0]
            if imported.name == "unittest":
                bindings[local_name] = "unittest"
            elif imported.name == "unittest.mock":
                if imported.asname is None:
                    bindings["unittest"] = "unittest"
                else:
                    bindings[imported.asname] = "unittest.mock"
            else:
                bindings.pop(local_name, None)
        return
    if isinstance(statement, ast.ImportFrom):
        for imported in statement.names:
            if imported.name == "*":
                continue
            local_name = imported.asname or imported.name
            if statement.level == 0 and statement.module in {
                "unittest",
                "unittest.mock",
            }:
                bindings[local_name] = f"{statement.module}.{imported.name}"
            else:
                bindings.pop(local_name, None)
        return

    targets: list[ast.expr] = []
    value: ast.expr | None = None
    if isinstance(statement, ast.Assign):
        targets = statement.targets
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
        value = statement.value
    elif isinstance(statement, ast.AugAssign):
        targets = [statement.target]
    elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bindings.pop(statement.name, None)
        return
    elif isinstance(statement, ast.Delete):
        for target in statement.targets:
            for name in _python_bound_names(target):
                bindings.pop(name, None)
        return
    else:
        return

    resolved = _python_static_binding(value, bindings) if value is not None else None
    for target in targets:
        names = _python_bound_names(target)
        if len(names) == 1 and isinstance(target, ast.Name) and resolved is not None:
            bindings[target.id] = resolved
        else:
            for name in names:
                bindings.pop(name, None)


def _literal_unittest_skip_decorator(
    decorator: ast.expr, path: str, bindings: dict[str, str]
) -> tuple[str, str, str] | None:
    target = (
        _python_static_binding(decorator.func, bindings)
        if isinstance(decorator, ast.Call)
        else _python_static_binding(decorator, bindings)
    )
    if target is None or not any(
        target == skip_binding or target == f"{skip_binding}()"
        for skip_binding in _PYTHON_UNITTEST_SKIP_BINDINGS
    ):
        return None
    if not isinstance(decorator, ast.Call):
        raise InventoryError(
            f"{path}: unittest skip aliases are forbidden on Python tooling tests"
        )
    function = decorator.func
    if not (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "unittest"
        and bindings.get("unittest") == "unittest"
        and function.attr in {"skip", "skipIf", "skipUnless"}
    ):
        raise InventoryError(
            f"{path}: unittest skip aliases are forbidden on Python tooling tests"
        )
    if function.attr == "skip":
        raise InventoryError(
            f"{path}: unconditional unittest.skip is forbidden by the finite "
            "skip predicate contract"
        )
    if (
        len(decorator.args) != 2
        or decorator.keywords
        or not isinstance(decorator.args[1], ast.Constant)
        or not isinstance(decorator.args[1].value, str)
        or not decorator.args[1].value
    ):
        raise InventoryError(
            f"{path}: conditional unittest skip decorator must have one "
            "predicate and one exact literal reason"
        )
    return (
        f"unittest.{function.attr}",
        decorator.args[1].value,
        _predicate_ast_sha256(decorator.args[0]),
    )


def _reviewed_test_decorators(
    decorators: list[ast.expr], path: str, bindings: dict[str, str]
) -> set[tuple[str, str, str]]:
    facts: set[tuple[str, str, str]] = set()
    for decorator in decorators:
        fact = _literal_unittest_skip_decorator(decorator, path, bindings)
        if fact is not None:
            facts.add(fact)
            continue
        target = (
            _python_static_binding(decorator.func, bindings)
            if isinstance(decorator, ast.Call)
            else _python_static_binding(decorator, bindings)
        )
        if target in _PYTHON_ALLOWED_TEST_DECORATOR_BINDINGS:
            continue
        raise InventoryError(f"{path}: Python tooling test has an unreviewed decorator")
    return facts


def _reviewed_conditional_skips(
    member: ast.FunctionDef | ast.AsyncFunctionDef,
    runtime_id: str,
    source_path: Path,
    source_sha256: str,
) -> tuple[
    set[tuple[str, str, str]],
    set[int],
    set[_PythonDynamicSkipSite],
]:
    parents = {
        child: parent
        for parent in ast.walk(member)
        for child in ast.iter_child_nodes(parent)
    }
    facts: set[tuple[str, str, str]] = set()
    allowed_attributes: set[int] = set()
    sites: set[_PythonDynamicSkipSite] = set()
    for node in ast.walk(member):
        if not isinstance(node, ast.Attribute) or node.attr != "skipTest":
            continue
        call = parents.get(node)
        statement = parents.get(call) if isinstance(call, ast.Call) else None
        if not (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(call, ast.Call)
            and call.func is node
            and isinstance(statement, ast.Expr)
            and statement.value is call
        ):
            raise InventoryError(
                "Python tooling test has a noncanonical skipTest capability access"
            )
        if (
            len(call.args) != 1
            or call.keywords
            or not isinstance(call.args[0], ast.Constant)
            or not isinstance(call.args[0].value, str)
            or not call.args[0].value
        ):
            raise InventoryError(
                "Python tooling skipTest must have one exact literal reason"
            )
        ancestor = parents.get(statement)
        conditional: ast.If | None = None
        nested_callable = False
        while ancestor is not None and ancestor is not member:
            if isinstance(
                ancestor,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                nested_callable = True
                break
            if isinstance(ancestor, ast.If) and conditional is None:
                conditional = ancestor
            ancestor = parents.get(ancestor)
        if nested_callable or not conditional:
            raise InventoryError(
                "Python tooling skipTest must be a direct conditional capability "
                "declaration in the test method"
            )
        if (
            parents.get(statement) is not conditional
            or statement not in conditional.body
        ):
            raise InventoryError(
                "Python tooling skipTest must be in the positive body of its "
                "direct conditional capability declaration"
            )
        facts.add(
            (
                "self.skipTest-if",
                call.args[0].value,
                _predicate_ast_sha256(conditional.test),
            )
        )
        allowed_attributes.add(id(node))
        sites.add(
            _PythonDynamicSkipSite(
                runtime_id,
                call.args[0].value,
                source_path,
                source_sha256,
                call.lineno,
            )
        )
    return facts, allowed_attributes, sites


def _python_static_text(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _python_static_text(expression.left)
        right = _python_static_text(expression.right)
        return None if left is None or right is None else left + right
    if isinstance(expression, ast.JoinedStr):
        pieces: list[str] = []
        for value in expression.values:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                return None
            pieces.append(value.value)
        return "".join(pieces)
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        pieces = [_python_static_text(item) for item in expression.elts]
        return None if any(piece is None for piece in pieces) else "".join(pieces)  # type: ignore[arg-type]
    if isinstance(expression, ast.Subscript):
        if (
            not isinstance(expression.slice, ast.Constant)
            or type(expression.slice.value) is not int
        ):
            return None
        if not isinstance(expression.value, (ast.List, ast.Tuple)):
            return None
        index = expression.slice.value
        try:
            return _python_static_text(expression.value.elts[index])
        except IndexError:
            return None
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "join"
        and not expression.keywords
        and len(expression.args) == 1
    ):
        separator = _python_static_text(expression.func.value)
        iterable = expression.args[0]
        if separator is None:
            return None
        if isinstance(iterable, (ast.List, ast.Tuple, ast.Set)):
            pieces = [_python_static_text(item) for item in iterable.elts]
        else:
            pieces = [
                child.value
                for child in ast.walk(iterable)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            ]
        if not pieces or any(piece is None for piece in pieces):
            return None
        return separator.join(pieces)  # type: ignore[arg-type]
    return None


def _python_skip_capability_audit(
    tree: ast.Module,
    allowed_attributes: set[int],
) -> None:
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    assignments: list[tuple[set[str], ast.expr, ast.AST]] = []
    assignment_targets: list[tuple[ast.expr, ast.expr, ast.AST, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            assignment_targets.extend(
                (target, node.value, node, False) for target in node.targets
            )
            assignments.append(
                (
                    {
                        name
                        for target in node.targets
                        for name in _python_bound_names(target)
                    },
                    node.value,
                    node,
                )
            )
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assignment_targets.append((node.target, node.value, node, False))
            assignments.append((_python_bound_names(node.target), node.value, node))
        elif isinstance(node, ast.NamedExpr):
            assignment_targets.append((node.target, node.value, node, False))
            assignments.append((_python_bound_names(node.target), node.value, node))
        elif isinstance(node, ast.AugAssign):
            assignment_targets.append((node.target, node.value, node, True))
            assignments.append((_python_bound_names(node.target), node.value, node))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            assignment_targets.append((node.target, node.iter, node, False))
            assignments.append((_python_bound_names(node.target), node.iter, node))

    scope_types = (
        ast.Module,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
    )

    def enclosing_scope(node: ast.AST) -> ast.AST:
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, scope_types):
                return parent
            parent = parents.get(parent)
        return tree

    def enclosing_function(node: ast.AST) -> ast.AST | None:
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return parent
            parent = parents.get(parent)
        return None

    def direct_references_names(expression: ast.expr, names: set[str]) -> bool:
        return any(
            isinstance(child, ast.Name)
            and child.id in names
            or isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and name_resolves_to_unittest_module(
                child.value.id, enclosing_scope(child.value)
            )
            and child.attr == "TestCase"
            for child in ast.walk(expression)
        )

    function_nodes: list[ast.AST] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
    ]
    nonlocal_assignment_names: dict[ast.AST, set[str]] = {
        function: set() for function in function_nodes
    }
    global_assignment_names: dict[ast.AST, set[str]] = {
        function: set() for function in function_nodes
    }
    enclosing_assignment_names: dict[ast.AST, set[str]] = {
        function: set() for function in function_nodes
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            owner = enclosing_function(node)
            if owner is not None:
                nonlocal_assignment_names[owner].update(node.names)
                if isinstance(node, ast.Global):
                    global_assignment_names[owner].update(node.names)
                else:
                    enclosing_assignment_names[owner].update(node.names)

    def pure_local_assignment_target(target: ast.expr) -> bool:
        if isinstance(target, ast.Name):
            return True
        if isinstance(target, (ast.Tuple, ast.List)):
            return all(pure_local_assignment_target(item) for item in target.elts)
        return False

    class_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    scope_bound_names: dict[ast.AST, set[str]] = {
        scope: set() for scope in (tree, *class_nodes, *function_nodes)
    }
    scope_binding_counts: dict[ast.AST, dict[str, int]] = {
        scope: {} for scope in scope_bound_names
    }

    def record_scope_binding(scope: ast.AST, name: str) -> None:
        scope_bound_names[scope].add(name)
        scope_binding_counts[scope][name] = scope_binding_counts[scope].get(name, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            record_scope_binding(enclosing_scope(node), node.id)
        elif isinstance(node, ast.arg):
            record_scope_binding(enclosing_scope(node), node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            record_scope_binding(enclosing_scope(node), node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            scope = enclosing_scope(node)
            for alias in node.names:
                record_scope_binding(scope, alias.asname or alias.name.split(".", 1)[0])

    def parent_lookup_scope(scope: ast.AST) -> ast.AST | None:
        parent = enclosing_scope(scope)
        if isinstance(
            scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ) and isinstance(parent, ast.ClassDef):
            return enclosing_scope(parent)
        return None if parent is scope else parent

    callable_bindings: dict[ast.AST, dict[str, set[ast.AST]]] = {
        scope: {} for scope in scope_bound_names
    }
    class_bindings: dict[ast.AST, dict[str, set[ast.ClassDef]]] = {
        scope: {} for scope in scope_bound_names
    }

    def bind_callable(scope: ast.AST, name: str, function: ast.AST) -> bool:
        candidates = callable_bindings[scope].setdefault(name, set())
        before = len(candidates)
        candidates.add(function)
        return len(candidates) != before

    for node in function_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bind_callable(enclosing_scope(node), node.name, node)
    for node in class_nodes:
        class_bindings[enclosing_scope(node)].setdefault(node.name, set()).add(node)
    for names, value, owner in assignments:
        if isinstance(value, ast.Lambda):
            for name in names:
                bind_callable(enclosing_scope(owner), name, value)

    def resolve_scoped_bindings(
        bindings: dict[ast.AST, dict[str, set[ast.AST]]],
        name: str,
        scope: ast.AST,
    ) -> set[ast.AST]:
        cursor: ast.AST | None = scope
        while cursor is not None:
            if name in scope_bound_names[cursor]:
                return set(bindings[cursor].get(name, set()))
            cursor = parent_lookup_scope(cursor)
        return set()

    aliases_changed = True
    while aliases_changed:
        aliases_changed = False
        for names, value, owner in assignments:
            if not isinstance(value, ast.Name):
                continue
            scope = enclosing_scope(owner)
            candidates = resolve_scoped_bindings(callable_bindings, value.id, scope)
            for name in names:
                for candidate in candidates:
                    aliases_changed = (
                        bind_callable(scope, name, candidate) or aliases_changed
                    )

    root_aliases: dict[ast.AST, set[str]] = {
        scope: set() for scope in scope_bound_names
    }
    unittest_module_aliases: dict[ast.AST, set[str]] = {
        scope: set() for scope in scope_bound_names
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "unittest":
            root_aliases[enclosing_scope(node)].update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "TestCase"
            )
        elif isinstance(node, ast.Import):
            unittest_module_aliases[enclosing_scope(node)].update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "unittest"
            )

    def name_resolves_to_unittest_module(name: str, scope: ast.AST) -> bool:
        cursor: ast.AST | None = scope
        while cursor is not None:
            if name in scope_bound_names[cursor]:
                return name in unittest_module_aliases[cursor]
            cursor = parent_lookup_scope(cursor)
        return False

    module_aliases_changed = True
    while module_aliases_changed:
        module_aliases_changed = False
        for names, value, owner in assignments:
            if not isinstance(value, ast.Name):
                continue
            scope = enclosing_scope(owner)
            if not name_resolves_to_unittest_module(value.id, scope):
                continue
            before = len(unittest_module_aliases[scope])
            unittest_module_aliases[scope].update(names)
            module_aliases_changed = (
                len(unittest_module_aliases[scope]) != before or module_aliases_changed
            )

    def name_resolves_to_root(name: str, scope: ast.AST) -> bool:
        cursor: ast.AST | None = scope
        while cursor is not None:
            if name in scope_bound_names[cursor]:
                return name in root_aliases[cursor]
            cursor = parent_lookup_scope(cursor)
        return False

    root_aliases_changed = True
    while root_aliases_changed:
        root_aliases_changed = False
        for names, value, owner in assignments:
            scope = enclosing_scope(owner)
            value_is_root = (
                isinstance(value, ast.Name)
                and name_resolves_to_root(value.id, scope)
                or isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and name_resolves_to_unittest_module(value.value.id, scope)
                and value.attr == "TestCase"
            )
            if value_is_root:
                before = len(root_aliases[scope])
                root_aliases[scope].update(names)
                root_aliases_changed = (
                    len(root_aliases[scope]) != before or root_aliases_changed
                )

    root_class_nodes: set[ast.ClassDef] = set()
    root_classes_changed = True
    while root_classes_changed:
        root_classes_changed = False
        for class_node in class_nodes:
            for base in class_node.bases:
                base_is_root = (
                    isinstance(base, ast.Name)
                    and name_resolves_to_root(base.id, enclosing_scope(base))
                    or isinstance(base, ast.Attribute)
                    and isinstance(base.value, ast.Name)
                    and name_resolves_to_unittest_module(
                        base.value.id, enclosing_scope(base.value)
                    )
                    and base.attr == "TestCase"
                    or isinstance(base, ast.Name)
                    and any(
                        candidate in root_class_nodes
                        for candidate in resolve_scoped_bindings(
                            class_bindings, base.id, enclosing_scope(base)
                        )
                    )
                )
                if base_is_root and class_node not in root_class_nodes:
                    root_class_nodes.add(class_node)
                    root_classes_changed = True

    def function_has_root_receiver(function: ast.AST) -> bool:
        cursor: ast.AST | None = function
        while cursor is not None:
            parameters, _, _, _ = function_parameters(cursor)
            if "self" in parameters:
                return enclosing_scope(cursor) in root_class_nodes
            cursor = enclosing_function(cursor)
        return False

    decorator_bindings: dict[ast.AST, dict[str, set[str]]] = {
        scope: {} for scope in scope_bound_names
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "builtins":
            scope = enclosing_scope(node)
            for alias in node.names:
                if alias.name in {"staticmethod", "classmethod"}:
                    decorator_bindings[scope].setdefault(
                        alias.asname or alias.name, set()
                    ).add(alias.name)

    def resolved_decorator_kind(name: str, scope: ast.AST) -> str | None:
        cursor: ast.AST | None = scope
        while cursor is not None:
            if name in scope_bound_names[cursor]:
                kinds = decorator_bindings[cursor].get(name, set())
                if scope_binding_counts[cursor].get(name) == 1 and len(kinds) == 1:
                    return next(iter(kinds))
                return None
            cursor = parent_lookup_scope(cursor)
        return name if name in {"staticmethod", "classmethod"} else None

    decorator_aliases_changed = True
    while decorator_aliases_changed:
        decorator_aliases_changed = False
        for names, value, owner in assignments:
            if not isinstance(value, ast.Name):
                continue
            scope = enclosing_scope(owner)
            kind = resolved_decorator_kind(value.id, scope)
            if kind is None:
                continue
            for name in names:
                if scope_binding_counts[scope].get(name) != 1:
                    continue
                kinds = decorator_bindings[scope].setdefault(name, set())
                before = len(kinds)
                kinds.add(kind)
                if len(kinds) != before:
                    decorator_aliases_changed = True

    def method_kind(function: ast.AST) -> str:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "function"
        if function.name == "__new__":
            return "class"
        if not function.decorator_list:
            return "instance"
        if len(function.decorator_list) != 1 or not isinstance(
            function.decorator_list[0], ast.Name
        ):
            return "unknown"
        decorator = function.decorator_list[0]
        kind = resolved_decorator_kind(decorator.id, enclosing_scope(function))
        if kind == "staticmethod":
            return "static"
        if kind == "classmethod":
            return "class"
        return "unknown"

    root_name_candidates = {
        *(name for names in root_aliases.values() for name in names),
        *(class_node.name for class_node in root_class_nodes),
    }

    def visible_root_names(scope: ast.AST) -> set[str]:
        return {
            name
            for name in root_name_candidates
            if name_resolves_to_root(name, scope)
            or any(
                candidate in root_class_nodes
                for candidate in resolve_scoped_bindings(class_bindings, name, scope)
            )
        }

    module_root_names = visible_root_names(tree)

    function_returns_root = {node: False for node in function_nodes}
    function_returns_capability = {node: False for node in function_nodes}
    function_returned_callables: dict[ast.AST, set[ast.AST]] = {
        node: set() for node in function_nodes
    }
    function_parameter_roots: dict[ast.AST, set[str]] = {
        node: set() for node in function_nodes
    }
    function_parameter_capability_roots: dict[ast.AST, set[str]] = {
        node: set() for node in function_nodes
    }
    function_roots: dict[ast.AST, set[str]] = {}
    function_capability_roots: dict[ast.AST, set[str]] = {}

    methods_by_name: dict[str, set[ast.AST]] = {}
    for class_node in class_nodes:
        for name, candidates in callable_bindings[class_node].items():
            methods_by_name.setdefault(name, set()).update(candidates)

    def resolved_classes(expression: ast.expr) -> set[ast.ClassDef]:
        if isinstance(expression, ast.Name):
            return {
                candidate
                for candidate in resolve_scoped_bindings(
                    class_bindings, expression.id, enclosing_scope(expression)
                )
                if isinstance(candidate, ast.ClassDef)
            }
        if isinstance(expression, ast.Call):
            return resolved_classes(expression.func)
        return set()

    def callee_candidates(expression: ast.expr) -> set[ast.AST]:
        if isinstance(expression, ast.Name):
            return resolve_scoped_bindings(
                callable_bindings, expression.id, enclosing_scope(expression)
            )
        if isinstance(expression, ast.Attribute):
            classes = resolved_classes(expression.value)
            if classes:
                return {
                    candidate
                    for class_node in classes
                    for candidate in callable_bindings[class_node].get(
                        expression.attr, set()
                    )
                }
            return set(methods_by_name.get(expression.attr, set()))
        if isinstance(expression, ast.Call):
            return {
                returned
                for producer in callee_candidates(expression.func)
                for returned in function_returned_callables[producer]
            }
        return set()

    def references_names(expression: ast.expr, names: set[str]) -> bool:
        if direct_references_names(expression, names):
            return True
        for child in ast.walk(expression):
            if not isinstance(child, ast.Call):
                continue
            candidates = callee_candidates(child.func)
            if any(function_returns_root[candidate] for candidate in candidates):
                return True
            if not candidates and any(
                references_names(argument, names)
                for argument in (
                    *child.args,
                    *(keyword.value for keyword in child.keywords),
                )
            ):
                return True
        return False

    def explicit_argument_references_root(
        expression: ast.expr, names: set[str]
    ) -> bool:
        pending = [expression]
        while pending:
            child = pending.pop()
            if child is not expression and isinstance(child, ast.Lambda):
                continue
            if (
                isinstance(child, ast.Name)
                and child.id in names
                and not (
                    isinstance(parents.get(child), ast.Attribute)
                    and parents[child].value is child
                )
            ):
                return True
            if isinstance(child, ast.Call) and any(
                function_returns_capability[candidate]
                for candidate in callee_candidates(child.func)
            ):
                return True
            pending.extend(ast.iter_child_nodes(child))
        return False

    def explicit_argument_contains_name(expression: ast.expr, name: str) -> bool:
        pending = [expression]
        while pending:
            child = pending.pop()
            if child is not expression and isinstance(child, ast.Lambda):
                continue
            if (
                isinstance(child, ast.Name)
                and child.id == name
                and not (
                    isinstance(parents.get(child), ast.Attribute)
                    and parents[child].value is child
                )
            ):
                return True
            pending.extend(ast.iter_child_nodes(child))
        return False

    def function_parameters(
        function: ast.AST,
    ) -> tuple[list[str], list[str], str | None, str | None]:
        arguments = function.args
        return (
            [
                *(argument.arg for argument in arguments.posonlyargs),
                *(argument.arg for argument in arguments.args),
            ],
            [argument.arg for argument in arguments.kwonlyargs],
            arguments.vararg.arg if arguments.vararg is not None else None,
            arguments.kwarg.arg if arguments.kwarg is not None else None,
        )

    def reviewed_unittest_class_construction(call: ast.Call) -> bool:
        def unshadowed_builtin_type(expression: ast.expr) -> bool:
            if not isinstance(expression, ast.Name) or expression.id != "type":
                return False
            cursor: ast.AST | None = enclosing_scope(expression)
            while cursor is not None:
                if "type" in scope_bound_names[cursor]:
                    return False
                cursor = parent_lookup_scope(cursor)
            return True

        def canonical_type_call(expression: ast.expr) -> bool:
            return (
                isinstance(expression, ast.Call)
                and unshadowed_builtin_type(expression.func)
                and len(expression.args) == 3
                and not expression.keywords
            )

        if canonical_type_call(call):
            return not isinstance(parents.get(call), ast.Expr)
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "loadTestsFromTestCase"
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "defaultTestLoader"
            and isinstance(call.func.value.value, ast.Name)
            and name_resolves_to_unittest_module(
                call.func.value.value.id, enclosing_scope(call.func.value.value)
            )
        ):
            return True
        return (
            isinstance(call.func, ast.Call)
            and isinstance(call.func.func, ast.Attribute)
            and call.func.func.attr == "skipUnless"
            and isinstance(call.func.func.value, ast.Name)
            and name_resolves_to_unittest_module(
                call.func.func.value.id, enclosing_scope(call.func.func.value)
            )
            and len(call.args) == 1
            and canonical_type_call(call.args[0])
        )

    reviewed_constructed_test_classes: dict[ast.AST, set[str]] = {
        scope: set() for scope in scope_bound_names
    }
    for names, value, owner_node in assignments:
        if isinstance(value, ast.Call) and reviewed_unittest_class_construction(value):
            reviewed_constructed_test_classes[enclosing_scope(owner_node)].update(names)

    def reviewed_skip_metadata_read(node: ast.Attribute) -> bool:
        return (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id
            in reviewed_constructed_test_classes[enclosing_scope(node.value)]
        )

    reviewed_constructed_suites: dict[ast.AST, set[str]] = {
        scope: set() for scope in scope_bound_names
    }
    for names, value, owner_node in assignments:
        scope = enclosing_scope(owner_node)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "loadTestsFromTestCase"
            and any(
                isinstance(argument, ast.Name)
                and argument.id in reviewed_constructed_test_classes[scope]
                for argument in value.args
            )
        ):
            reviewed_constructed_suites[scope].update(names)

    def reviewed_suite_run(node: ast.Attribute) -> bool:
        return (
            node.attr == "run"
            and isinstance(node.value, ast.Name)
            and node.value.id
            in reviewed_constructed_suites[enclosing_scope(node.value)]
        )

    unsafe_root_assignment_escape = False
    unsafe_unresolved_root_call = False
    unsafe_unknown_decorator_root_call = False
    interprocedural_changed = True
    while interprocedural_changed:
        interprocedural_changed = False
        for function in function_nodes:
            derived = {
                *function_parameter_roots[function],
            }
            if function_has_root_receiver(function):
                derived.add("self")
            derived.update(visible_root_names(function))
            local_assignments = [
                (names, value)
                for names, value, owner in assignments
                if enclosing_function(owner) is function
            ]
            local_changed = True
            while local_changed:
                local_changed = False
                for names, value in local_assignments:
                    if explicit_argument_references_root(value, derived):
                        prior = len(derived)
                        derived.update(names)
                        local_changed = len(derived) != prior
            if function_roots.get(function) != derived:
                function_roots[function] = derived
                interprocedural_changed = True

            capability_roots = {
                *function_parameter_capability_roots[function],
                *visible_root_names(function),
            }
            if function_has_root_receiver(function):
                capability_roots.add("self")
            local_changed = True
            while local_changed:
                local_changed = False
                for names, value in local_assignments:
                    if any(
                        explicit_argument_contains_name(value, name)
                        for name in capability_roots
                    ):
                        prior = len(capability_roots)
                        capability_roots.update(names)
                        local_changed = len(capability_roots) != prior
            if function_capability_roots.get(function) != capability_roots:
                function_capability_roots[function] = capability_roots
                interprocedural_changed = True

        for function in function_nodes:
            returns = [
                node.value
                for node in ast.walk(function)
                if isinstance(node, ast.Return)
                and node.value is not None
                and enclosing_function(node.value) is function
            ]
            returns_root = any(
                references_names(value, function_roots[function]) for value in returns
            )
            if function_returns_root[function] != returns_root:
                function_returns_root[function] = returns_root
                interprocedural_changed = True
            returns_capability = any(
                explicit_argument_references_root(
                    value, function_capability_roots[function]
                )
                for value in returns
            )
            if function_returns_capability[function] != returns_capability:
                function_returns_capability[function] = returns_capability
                interprocedural_changed = True
            returned_callables = {
                callee for value in returns for callee in callee_candidates(value)
            }
            if function_returned_callables[function] != returned_callables:
                function_returned_callables[function] = returned_callables
                interprocedural_changed = True

        for names, value, owner in assignments:
            if not isinstance(value, ast.Call):
                continue
            scope = enclosing_scope(owner)
            for name in names:
                for candidate in callee_candidates(value):
                    interprocedural_changed = (
                        bind_callable(scope, name, candidate) or interprocedural_changed
                    )

        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            candidates = callee_candidates(call.func)
            caller = enclosing_function(call)
            caller_roots = function_roots.get(caller, module_root_names)
            caller_capability_roots = function_capability_roots.get(
                caller, module_root_names
            )
            explicit_root = any(
                explicit_argument_references_root(argument, caller_capability_roots)
                for argument in (
                    *call.args,
                    *(keyword.value for keyword in call.keywords),
                )
            )
            direct_self_root = (
                caller is not None
                and function_has_root_receiver(caller)
                and any(
                    explicit_argument_contains_name(argument, "self")
                    for argument in (
                        *call.args,
                        *(keyword.value for keyword in call.keywords),
                    )
                )
            )
            if not candidates:
                unsafe_unresolved_root_call = (
                    (direct_self_root or explicit_root)
                    and not reviewed_unittest_class_construction(call)
                    or unsafe_unresolved_root_call
                )
                continue
            for callee in candidates:
                positional, keyword_only, vararg, kwarg = function_parameters(callee)
                before = len(function_parameter_roots[callee])
                capability_before = len(function_parameter_capability_roots[callee])
                definition_scope = enclosing_scope(callee)
                class_qualified = (
                    isinstance(call.func, ast.Attribute)
                    and bool(resolved_classes(call.func.value))
                    and not isinstance(call.func.value, ast.Call)
                )
                kind = method_kind(callee)
                if (
                    kind == "unknown"
                    and isinstance(definition_scope, ast.ClassDef)
                    and (
                        explicit_root
                        or isinstance(call.func, ast.Attribute)
                        and references_names(call.func.value, caller_roots)
                    )
                ):
                    unsafe_unknown_decorator_root_call = True
                bound_method = (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(definition_scope, ast.ClassDef)
                    and (kind == "class" or kind == "instance" and not class_qualified)
                )
                positional_offset = 1 if bound_method else 0
                if (
                    bound_method
                    and positional
                    and references_names(call.func.value, caller_roots)
                ):
                    function_parameter_roots[callee].add(positional[0])
                if (
                    bound_method
                    and positional
                    and references_names(call.func.value, caller_capability_roots)
                ):
                    function_parameter_capability_roots[callee].add(positional[0])
                for index, argument in enumerate(call.args):
                    parameter_index = index + positional_offset
                    argument_is_root = references_names(argument, caller_roots)
                    if isinstance(argument, ast.Starred) and argument_is_root:
                        function_parameter_roots[callee].update(
                            positional[parameter_index:]
                        )
                        if vararg is not None:
                            function_parameter_roots[callee].add(vararg)
                    elif argument_is_root:
                        if parameter_index < len(positional):
                            function_parameter_roots[callee].add(
                                positional[parameter_index]
                            )
                        elif vararg is not None:
                            function_parameter_roots[callee].add(vararg)
                    argument_is_capability_root = explicit_argument_references_root(
                        argument, caller_capability_roots
                    )
                    if (
                        isinstance(argument, ast.Starred)
                        and argument_is_capability_root
                    ):
                        function_parameter_capability_roots[callee].update(
                            positional[parameter_index:]
                        )
                        if vararg is not None:
                            function_parameter_capability_roots[callee].add(vararg)
                    elif argument_is_capability_root:
                        if parameter_index < len(positional):
                            function_parameter_capability_roots[callee].add(
                                positional[parameter_index]
                            )
                        elif vararg is not None:
                            function_parameter_capability_roots[callee].add(vararg)
                keyword_parameters = {*positional, *keyword_only}
                for keyword in call.keywords:
                    if not references_names(keyword.value, caller_roots):
                        pass
                    elif keyword.arg is None:
                        function_parameter_roots[callee].update(keyword_parameters)
                        if kwarg is not None:
                            function_parameter_roots[callee].add(kwarg)
                    elif keyword.arg in keyword_parameters:
                        function_parameter_roots[callee].add(keyword.arg)
                    elif kwarg is not None:
                        function_parameter_roots[callee].add(kwarg)
                    if explicit_argument_references_root(
                        keyword.value, caller_capability_roots
                    ):
                        if keyword.arg is None:
                            function_parameter_capability_roots[callee].update(
                                keyword_parameters
                            )
                            if kwarg is not None:
                                function_parameter_capability_roots[callee].add(kwarg)
                        elif keyword.arg in keyword_parameters:
                            function_parameter_capability_roots[callee].add(keyword.arg)
                        elif kwarg is not None:
                            function_parameter_capability_roots[callee].add(kwarg)
                if len(function_parameter_roots[callee]) != before:
                    interprocedural_changed = True
                if (
                    len(function_parameter_capability_roots[callee])
                    != capability_before
                ):
                    interprocedural_changed = True

        for target, value, owner_node, include_target_value in assignment_targets:
            owner = enclosing_function(owner_node)
            names = function_roots.get(owner, module_root_names)
            target_names = _python_bound_names(target)
            rooted_assignment_value = explicit_argument_references_root(value, names)
            rooted_augmented_target = (
                include_target_value
                and explicit_argument_references_root(target, names)
            )
            if include_target_value and isinstance(target, ast.Name):
                if (
                    owner is not None
                    and target.id in global_assignment_names[owner]
                    and target.id in module_root_names
                ):
                    rooted_augmented_target = True
                elif (
                    owner is not None and target.id in enclosing_assignment_names[owner]
                ):
                    outer = enclosing_function(owner)
                    while outer is not None:
                        if target.id in function_roots.get(outer, set()):
                            rooted_augmented_target = True
                            break
                        outer = enclosing_function(outer)
            if (rooted_assignment_value or rooted_augmented_target) and (
                not pure_local_assignment_target(target)
                or owner is not None
                and bool(target_names & nonlocal_assignment_names[owner])
            ):
                unsafe_root_assignment_escape = True

    if unsafe_root_assignment_escape:
        raise InventoryError(
            "Python tooling module has a noncanonical skipTest capability access: "
            "rooted assignment escape"
        )
    if unsafe_unresolved_root_call:
        raise InventoryError(
            "Python tooling module has a noncanonical skipTest capability access: "
            "unresolved rooted call"
        )
    if unsafe_unknown_decorator_root_call:
        raise InventoryError(
            "Python tooling module has a noncanonical skipTest capability access: "
            "unknown decorated rooted call"
        )

    def references_root(expression: ast.expr) -> bool:
        owner = enclosing_function(expression)
        names = function_roots.get(owner, module_root_names)
        return references_names(expression, names)

    def callable_chain_calls(capability: ast.expr) -> list[ast.Call]:
        calls: list[ast.Call] = []
        cursor: ast.AST = capability
        while True:
            parent = parents.get(cursor)
            if isinstance(parent, ast.Attribute) and parent.value is cursor:
                cursor = parent
                continue
            if isinstance(parent, ast.Call) and parent.func is cursor:
                calls.append(parent)
                cursor = parent
                continue
            return calls

    def reflective_access_is_unsafe(
        capability: ast.expr,
        receiver: ast.expr | None,
        require_direct_use: bool,
    ) -> bool:
        calls = callable_chain_calls(capability)
        if receiver is not None and references_root(receiver):
            return True
        if require_direct_use and not calls:
            return True
        return any(
            references_root(argument)
            for call in calls
            for argument in (
                *call.args,
                *(keyword.value for keyword in call.keywords),
            )
        )

    dangerous_identifiers = {
        "skipTest",
        "SkipTest",
        "_outcome",
        *_PYTHON_RESULT_CALLBACK_NAMES[1:],
    }
    frame_authority_attributes = {
        "_getframe",
        "_current_frames",
        "currentframe",
        "stack",
        "f_back",
        "f_locals",
        "f_globals",
        "tb_frame",
        "__closure__",
        "get_referrers",
        "get_objects",
        "__subclasses__",
    }
    execution_hooks = {
        "__call__",
        "run",
        "__getattribute__",
        "__setattr__",
        *_PYTHON_TEST_CASE_EXECUTION_HOOKS,
    }
    reflective_names = {"getattr", "hasattr", "setattr", "vars", "dir", "type"}
    reflective_attributes = {
        "__class__",
        "__dict__",
        "__mro__",
        "mro",
        "__bases__",
        "__getattribute__",
    }
    for node in ast.walk(tree):
        identifier: str | None = None
        if isinstance(node, ast.Name):
            identifier = node.id
        elif isinstance(node, ast.arg):
            identifier = node.arg
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifier = node.name
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = parents.get(node)
            if (
                node.name in execution_hooks
                and isinstance(parent, ast.ClassDef)
                and parent in root_class_nodes
            ):
                raise InventoryError(
                    "Python tooling TestCase defines a noncanonical execution hook"
                )
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            parents.get(node), ast.ClassDef
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            class_node = parents[node]
            if class_node in root_class_nodes and any(
                name in execution_hooks
                for target in targets
                for name in _python_bound_names(target)
            ):
                raise InventoryError(
                    "Python tooling TestCase mutates a noncanonical execution hook"
                )
        if identifier in dangerous_identifiers or (
            identifier is not None and identifier.startswith("__unittest_skip")
        ):
            raise InventoryError(
                "Python tooling module has a noncanonical skipTest capability access "
                f"through {identifier} at line {getattr(node, 'lineno', 0)}"
            )
        if isinstance(node, ast.alias) and (
            node.name in dangerous_identifiers
            or (node.asname or "") in dangerous_identifiers
            or node.name.startswith("__unittest_skip")
            or (node.asname or "").startswith("__unittest_skip")
        ):
            raise InventoryError(
                "Python tooling module has a noncanonical skipTest capability access"
            )
        if isinstance(node, ast.Attribute):
            if node.attr in frame_authority_attributes:
                raise InventoryError(
                    "Python tooling module accesses noncanonical runtime authority"
                )
            if (
                node.attr in execution_hooks
                and references_root(node.value)
                and not reviewed_suite_run(node)
            ):
                raise InventoryError(
                    "Python tooling TestCase accesses a noncanonical execution hook"
                )
            if node.attr == "skipTest" and id(node) not in allowed_attributes:
                raise InventoryError(
                    "Python tooling module has a noncanonical skipTest capability "
                    f"access through {node.attr} at line {node.lineno}"
                )
            if (
                node.attr in dangerous_identifiers
                and not (node.attr == "skipTest" and id(node) in allowed_attributes)
                or node.attr.startswith("__unittest_skip")
                and not reviewed_skip_metadata_read(node)
            ):
                raise InventoryError(
                    "Python tooling module has a noncanonical skipTest capability access "
                    f"through {node.attr} at line {node.lineno}"
                )
            if node.attr in reflective_attributes | reflective_names:
                if reflective_access_is_unsafe(
                    node,
                    node.value,
                    node.attr in reflective_names | {"__getattribute__", "mro"},
                ):
                    raise InventoryError(
                        "Python tooling module has a noncanonical skipTest "
                        "capability access"
                    )
        if isinstance(node, ast.expr):
            static_text = _python_static_text(node)
            if static_text in dangerous_identifiers or (
                static_text is not None and static_text.startswith("__unittest_skip")
            ):
                raise InventoryError(
                    "Python tooling module has a noncanonical skipTest capability access"
                )
        if isinstance(node, ast.Name) and node.id in reflective_names:
            reviewed_type_name = (
                node.id == "type"
                and isinstance(parents.get(node), ast.Call)
                and parents[node].func is node
                and reviewed_unittest_class_construction(parents[node])
            )
            if not reviewed_type_name and reflective_access_is_unsafe(node, None, True):
                raise InventoryError(
                    "Python tooling module has a noncanonical skipTest capability "
                    f"access through {node.id} at line {node.lineno}"
                )
        if isinstance(node, ast.Attribute) and node.attr == "attrgetter":
            raise InventoryError(
                "Python tooling module has a noncanonical skipTest capability access"
            )


def _reviewed_python_source_module(
    root: Path,
    path: str,
    discovery_start: str,
    discovery_pattern: str,
) -> _PythonReviewedSourceModule:
    relative_path = PurePosixPath(path)
    start_path = PurePosixPath(discovery_start)
    if (
        relative_path.is_absolute()
        or start_path.is_absolute()
        or ".." in relative_path.parts
        or ".." in start_path.parts
        or relative_path.suffix != ".py"
    ):
        raise InventoryError("Python tooling reviewed module path is noncanonical")
    try:
        discovery_relative = relative_path.relative_to(start_path)
    except ValueError as exc:
        raise InventoryError(
            f"Python tooling module is outside its discovery start: {path}"
        ) from exc
    if len(discovery_relative.parts) != 1 or not fnmatch.fnmatchcase(
        discovery_relative.name, discovery_pattern
    ):
        raise InventoryError(
            f"Python tooling module does not match exact discovery metadata: {path}"
        )
    module_name = discovery_relative.stem
    if not module_name.isidentifier():
        raise InventoryError(f"Python tooling module name is noncanonical: {path}")
    source_path = root.joinpath(*relative_path.parts)
    snapshot = _read_regular_stable_snapshot(
        source_path,
        MAX_INVENTORY_BYTES,
        f"Python tooling source {path}",
    )
    return _PythonReviewedSourceModule(
        path,
        module_name,
        source_path,
        snapshot.bytes,
        snapshot.sha256,
    )


def _reviewed_python_tooling_source_modules(
    root: Path,
    module_paths: list[str],
    discovery_start: str,
    discovery_pattern: str,
) -> tuple[_PythonReviewedSourceModule, ...]:
    admission = _PYTHON_TOOLING_REVIEWED_SOURCE_SHA256
    if type(admission) is not tuple or len(admission) != 14:
        raise InventoryError("Python tooling source admission registry is noncanonical")
    expected_paths: list[str] = []
    expected_digests: list[str] = []
    for entry in admission:
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", entry[1]) is None
        ):
            raise InventoryError(
                "Python tooling source admission registry is noncanonical"
            )
        expected_paths.append(entry[0])
        expected_digests.append(entry[1])
    if len(expected_paths) != len(set(expected_paths)):
        raise InventoryError("Python tooling source admission registry is duplicated")
    if type(module_paths) is not list or tuple(module_paths) != tuple(expected_paths):
        raise InventoryError(
            "Python tooling inventory module order differs from source admission"
        )

    reviewed_modules = tuple(
        _reviewed_python_source_module(root, path, discovery_start, discovery_pattern)
        for path in expected_paths
    )
    for reviewed, path, expected_digest in zip(
        reviewed_modules, expected_paths, expected_digests, strict=True
    ):
        if (
            reviewed.inventory_path != path
            or reviewed.source_sha256 != expected_digest
            or hashlib.sha256(reviewed.source_bytes).hexdigest() != expected_digest
        ):
            raise InventoryError(
                f"Python tooling source is not admitted by exact bytes: {path}"
            )
    return reviewed_modules


def _freeze_python_tooling_execution_closure(
    root: Path,
    context: Any,
    module_paths: list[str],
) -> _PythonExecutionClosure:
    source_manifest = _PYTHON_TOOLING_EXECUTION_SOURCE_SHA256
    instance_manifest = _PYTHON_TOOLING_EXECUTION_MODULES
    if (
        type(source_manifest) is not tuple
        or len(source_manifest) != 39
        or type(instance_manifest) is not tuple
        or len(instance_manifest) != 41
        or tuple(module_paths)
        != tuple(path for path, _ in _PYTHON_TOOLING_REVIEWED_SOURCE_SHA256)
    ):
        raise InventoryError(
            "Python tooling execution closure manifest is noncanonical"
        )
    physical_root = context.root
    if (
        not isinstance(physical_root, Path)
        or context.public_files.root != physical_root
        or context.public_files.supplied_root != Path(os.path.abspath(root))
    ):
        raise InventoryError("Python tooling execution physical root is noncanonical")
    try:
        resolved_root = physical_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InventoryError(
            "Python tooling execution physical root is noncanonical"
        ) from exc
    if resolved_root != physical_root:
        raise InventoryError("Python tooling execution physical root is noncanonical")
    try:
        manifest_bytes = json.dumps(
            (source_manifest, instance_manifest),
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    except (TypeError, UnicodeError, ValueError) as exc:
        raise InventoryError(
            "Python tooling execution closure manifest is noncanonical"
        ) from exc
    if (
        hashlib.sha256(manifest_bytes).hexdigest()
        != _PYTHON_TOOLING_EXECUTION_MANIFEST_SHA256
    ):
        raise InventoryError(
            "Python tooling execution closure manifest is noncanonical"
        )
    source_paths: list[str] = []
    frozen: list[_PythonFrozenSource] = []
    for entry in source_manifest:
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", entry[1]) is None
        ):
            raise InventoryError(
                "Python tooling execution source manifest is noncanonical"
            )
        path, expected_digest = entry
        source_paths.append(path)
        node = context.public_files.node(path)
        if node.kind != "regular" or node.bytes is None or node.sha256 is None:
            raise InventoryError(f"Python tooling closure source is not frozen: {path}")
        if (
            node.sha256 != expected_digest
            or hashlib.sha256(node.bytes).hexdigest() != expected_digest
        ):
            raise InventoryError(
                f"Python tooling closure source is not admitted: {path}"
            )
        source_path = physical_root / PurePosixPath(path)
        try:
            resolved_source_path = source_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InventoryError(
                f"Python tooling closure source physical path is invalid: {path}"
            ) from exc
        if resolved_source_path != source_path or node.path != path:
            raise InventoryError(
                f"Python tooling closure source physical path is invalid: {path}"
            )
        observed = _read_regular_stable_snapshot(
            source_path,
            MAX_INVENTORY_BYTES,
            f"Python tooling closure source {path}",
        )
        node_identity = (
            node.identity.device,
            node.identity.inode,
            node.identity.size,
            node.identity.mtime_ns,
            node.identity.ctime_ns,
        )
        if (
            observed.bytes != node.bytes
            or observed.sha256 != node.sha256
            or observed.identity != node_identity
            or observed.mode != node.identity.mode
        ):
            raise InventoryError(
                f"Python tooling closure source snapshot identity changed: {path}"
            )
        frozen.append(
            _PythonFrozenSource(
                path,
                source_path,
                node.bytes,
                expected_digest,
                node_identity,
                node.identity.mode,
            )
        )
    if len(source_paths) != len(set(source_paths)):
        raise InventoryError("Python tooling execution source manifest is duplicated")
    instance_names: list[str] = []
    instance_paths: list[str] = []
    for entry in instance_manifest:
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or not entry[0]
            or type(entry[1]) is not str
        ):
            raise InventoryError(
                "Python tooling execution instance manifest is noncanonical"
            )
        instance_names.append(entry[0])
        instance_paths.append(entry[1])
    if len(instance_names) != len(set(instance_names)) or set(instance_paths) != set(
        source_paths
    ):
        raise InventoryError(
            "Python tooling execution instance manifest is inconsistent"
        )
    return _PythonExecutionClosure(physical_root, context, tuple(frozen))


class _PythonFrozenLoader(importlib.abc.Loader):
    def __init__(self, closure: _PythonExecutionClosure, name: str) -> None:
        self.closure = closure
        self.name = name
        self.source = closure.sources[closure.module_paths[name]]

    def create_module(self, spec: Any) -> types.ModuleType:
        existing = self.closure.instances.get(self.name)
        if existing is not None:
            if self.name not in self.closure.executed:
                raise InventoryError(
                    "Python tooling frozen module creation is noncanonical"
                )
            return existing.module
        if spec.name != self.name:
            raise InventoryError(
                "Python tooling frozen module creation is noncanonical"
            )
        module = types.ModuleType(self.name)
        self.closure.instances[self.name] = _PythonModuleInstance(
            self.name, self.source, module, spec, self
        )
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        instance = self.closure.instances.get(self.name)
        if (
            instance is not None
            and instance.module is module
            and self.name in self.closure.executed
        ):
            return
        if (
            instance is None
            or instance.module is not module
            or self.name in self.closure.executed
        ):
            raise InventoryError(
                "Python tooling frozen module execution is noncanonical"
            )
        code = _PYTHON_TOOLING_TRUSTED_COMPILE(
            self.source.source_bytes,
            str(self.source.source_path),
            "exec",
            dont_inherit=True,
        )
        if type(code) is not types.CodeType:
            raise InventoryError("Python tooling frozen compiled code changed")
        _PYTHON_TOOLING_TRUSTED_EXEC(code, vars(module))
        self.closure.executed.add(self.name)
        self.closure.verify()


def _load_python_tooling_closure_dependencies(
    closure: _PythonExecutionClosure, root_names: set[str]
) -> None:
    for name, path in closure.module_paths.items():
        if name in root_names or path.startswith("tools/"):
            continue
        importlib.import_module(name)


def _python_path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _python_tooling_path_identity(
    value: str | os.PathLike[str],
) -> tuple[Path, Path]:
    lexical = Path(os.path.abspath(os.fspath(value) or os.getcwd()))
    return lexical, Path(os.path.realpath(lexical))


def _python_tooling_path_touches_repository(
    closure: _PythonExecutionClosure, lexical: Path, resolved: Path
) -> bool:
    return any(
        _python_path_is_within(candidate, root)
        for candidate in (lexical, resolved)
        for root in (closure.lexical_root, closure.root)
    )


def _python_tooling_live_repo_import_candidate(
    closure: _PythonExecutionClosure, fullname: str
) -> bool:
    top_level = fullname.partition(".")[0]
    if not top_level or os.sep in top_level or (os.altsep and os.altsep in top_level):
        return True
    search_roots = (
        closure.lexical_root,
        closure.lexical_root / "bench/tools",
        closure.root,
        closure.root / "bench/tools",
    )
    suffixes = (".py", *importlib.machinery.EXTENSION_SUFFIXES)
    for base in search_roots:
        if os.path.lexists(base / top_level):
            return True
        if any(os.path.lexists(base / f"{top_level}{suffix}") for suffix in suffixes):
            return True
    return False


class _PythonFrozenFinder(importlib.abc.MetaPathFinder):
    def __init__(self, closure: _PythonExecutionClosure) -> None:
        self.closure = closure

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname in self.closure.module_paths:
            return _python_frozen_module_spec(self.closure, fullname)
        if (
            fullname in self.closure.repo_local_names
            or _python_tooling_live_repo_import_candidate(self.closure, fullname)
        ):
            raise InventoryError(
                "Python tooling repo-local import is outside the frozen manifest"
            )

        try:
            builtin = _PYTHON_TOOLING_TRUSTED_BUILTIN_FIND_SPEC(fullname, path, target)
            if builtin is not None:
                return builtin
            frozen = _PYTHON_TOOLING_TRUSTED_FROZEN_FIND_SPEC(fullname, path, target)
            if frozen is not None:
                return frozen
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            raise InventoryError(
                "Python tooling external import cannot be resolved canonically"
            ) from exc

        search_path = sys.path if path is None else path
        if not isinstance(search_path, (list, tuple)) and not hasattr(
            search_path, "__iter__"
        ):
            raise InventoryError(
                "Python tooling external import search path is noncanonical"
            )
        filtered: list[str] = []
        try:
            for entry in search_path:
                if not isinstance(entry, (str, os.PathLike)):
                    continue
                lexical, resolved = _python_tooling_path_identity(entry)
                if _python_tooling_path_touches_repository(
                    self.closure, lexical, resolved
                ):
                    continue
                filtered.append(os.fspath(entry))
            spec = _PYTHON_TOOLING_TRUSTED_PATH_FIND_SPEC(fullname, filtered, target)
        except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
            raise InventoryError(
                "Python tooling external import cannot be resolved canonically"
            ) from exc
        if spec is None:
            raise ModuleNotFoundError(
                f"Python tooling external import is unavailable: {fullname}",
                name=fullname,
            )
        if spec.loader is None or not isinstance(spec.origin, str):
            raise InventoryError(
                "Python tooling external import cannot be resolved canonically"
            )
        if spec.origin not in {"built-in", "frozen"}:
            lexical, resolved = _python_tooling_path_identity(spec.origin)
            if _python_tooling_path_touches_repository(self.closure, lexical, resolved):
                raise InventoryError(
                    "Python tooling external import resolved inside the repository"
                )
        locations = spec.submodule_search_locations
        if locations is not None:
            for location in locations:
                lexical, resolved = _python_tooling_path_identity(location)
                if _python_tooling_path_touches_repository(
                    self.closure, lexical, resolved
                ):
                    raise InventoryError(
                        "Python tooling external package resolved inside the repository"
                    )
        return spec


def _python_frozen_module_spec(closure: _PythonExecutionClosure, name: str) -> Any:
    if name not in closure.module_paths:
        raise InventoryError(
            f"Python tooling repo-local module is not admitted: {name}"
        )
    existing = closure.instances.get(name)
    if existing is not None and name in closure.executed:
        return existing.spec
    loader = _PythonFrozenLoader(closure, name)
    source = closure.sources[closure.module_paths[name]]
    spec = _PYTHON_TOOLING_TRUSTED_SPEC_FROM_FILE_LOCATION(
        name, source.source_path, loader=loader
    )
    if spec is None or spec.loader is not loader:
        raise InventoryError("Python tooling frozen module spec is noncanonical")
    return spec


@contextlib.contextmanager
def _python_tooling_execution_imports(
    closure: _PythonExecutionClosure,
) -> Any:
    manifest_names = set(closure.module_paths)
    if manifest_names.intersection(sys.modules):
        raise InventoryError("Python tooling execution module is already loaded")
    finder = _PythonFrozenFinder(closure)
    original_meta_path = tuple(sys.meta_path)
    tools_path = str(closure.root / "bench/tools")
    execution_sys_path = tuple(
        [tools_path, *sys.path] if tools_path not in sys.path else sys.path
    )
    closure.execution_sys_path = execution_sys_path

    def frozen_spec_from_file_location(
        name: str, location: Any, *args: Any, **kwargs: Any
    ) -> Any:
        try:
            lexical, resolved = _python_tooling_path_identity(location)
        except (OSError, TypeError, ValueError) as exc:
            raise InventoryError(
                "Python tooling module spec path is noncanonical"
            ) from exc
        expected = closure.module_paths.get(name)
        if expected is not None:
            source = closure.sources[expected]
            if (
                lexical != source.source_path
                or resolved != source.source_path
                or args
                or kwargs
            ):
                raise InventoryError(
                    "Python tooling frozen module spec request is outside the frozen manifest"
                )
            return _python_frozen_module_spec(closure, name)
        if not _python_tooling_path_touches_repository(closure, lexical, resolved):
            return _PYTHON_TOOLING_TRUSTED_SPEC_FROM_FILE_LOCATION(
                name, location, *args, **kwargs
            )
        raise InventoryError(
            "Python tooling repo-local spec request is outside the frozen manifest"
        )

    try:
        with (
            mock.patch.object(sys, "meta_path", [finder, *original_meta_path]),
            mock.patch.object(sys, "path", list(execution_sys_path)),
            mock.patch.object(
                importlib.util,
                "spec_from_file_location",
                frozen_spec_from_file_location,
            ),
        ):
            try:
                yield closure
                closure.verify(require_complete=True)
            finally:
                closure.verify()
                closure.live_recheck()
    finally:
        closure.execution_sys_path = None
        for name in manifest_names:
            sys.modules.pop(name, None)


_PYTHON_TOOLING_PYTHON_SUBPROCESS_SITES = frozenset(
    {
        *(
            ("bench/tools/test_level2_report.py", line, "subprocess.run")
            for line in (1566, 1632, 1696, 1808, 1859, 2352, 2402, 2451, 2504)
        ),
        ("bench/tools/run_level1_report.py", 894, "subprocess.run"),
        ("bench/tools/run_level2_report.py", 2881, "subprocess.run"),
    }
)
_PYTHON_TOOLING_NONPYTHON_SUBPROCESS_SITES = frozenset(
    {
        (
            "bench/tools/test_benchmark_artifact_snapshot.py",
            145,
            "os.posix_spawn",
        ),
        ("bench/tools/test_benchmark_metadata.py", 29, "subprocess.run"),
        ("bench/tools/test_rank_k_report.py", 1341, "subprocess.run"),
        ("bench/tools/test_rotg_latency_report.py", 1295, "subprocess.run"),
        ("bench/tools/test_symm_report.py", 1218, "subprocess.run"),
        ("bench/tools/test_triangular_matrix_report.py", 1229, "subprocess.run"),
        ("bench/tools/benchmark_metadata.py", 287, "subprocess.run"),
        ("bench/tools/run_gemm_sweep_isolated.py", 258, "subprocess.run"),
        ("bench/tools/run_gemm_sweep_isolated.py", 502, "subprocess.run"),
        ("bench/tools/run_level1_report.py", 445, "subprocess.run"),
        ("bench/tools/run_level1_report.py", 1429, "subprocess.run"),
        ("bench/tools/run_level2_report.py", 836, "subprocess.run"),
        ("bench/tools/run_rank_k_report.py", 612, "subprocess.run"),
        ("bench/tools/run_rank_k_report.py", 766, "subprocess.run"),
        ("bench/tools/run_rotg_latency_report.py", 412, "subprocess.run"),
        ("bench/tools/run_rotg_latency_report.py", 554, "subprocess.run"),
        ("bench/tools/run_symm_report.py", 610, "subprocess.run"),
        ("bench/tools/run_symm_report.py", 764, "subprocess.run"),
        ("bench/tools/run_triangular_matrix_report.py", 621, "subprocess.run"),
        ("bench/tools/run_triangular_matrix_report.py", 774, "subprocess.run"),
        ("tools/repository_git.py", 560, "subprocess.run"),
    }
)


def _python_tooling_subprocess_source_audit(
    source: _PythonFrozenSource,
) -> None:
    try:
        tree = ast.parse(
            source.source_bytes.decode("utf-8"), filename=source.inventory_path
        )
    except (SyntaxError, UnicodeError, ValueError) as exc:
        raise InventoryError(
            "Python tooling subprocess source cannot be parsed"
        ) from exc
    observed: set[tuple[str, int, str]] = set()
    raw_names = {"run", "Popen", "call", "check_call", "check_output"}
    raw_os_names = {
        "fork",
        "forkpty",
        "popen",
        "posix_spawn",
        "posix_spawnp",
        "startfile",
        "system",
    }
    subprocess_modules = {"subprocess"}
    os_modules = {"os"}
    raw_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    subprocess_modules.add(alias.asname or alias.name)
                elif alias.name == "os":
                    os_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                bound = alias.asname or alias.name
                if node.module == "subprocess" and alias.name in raw_names:
                    raw_aliases[bound] = f"subprocess.{alias.name}"
                elif node.module == "os" and (
                    alias.name in raw_os_names
                    or alias.name.startswith(("spawn", "exec"))
                ):
                    raw_aliases[bound] = f"os.{alias.name}"
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            value = node.value
            if value is None:
                continue
            canonical = None
            if isinstance(value, ast.Name):
                canonical = raw_aliases.get(value.id)
            elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                if value.value.id in subprocess_modules and value.attr in raw_names:
                    canonical = f"subprocess.{value.attr}"
                elif value.value.id in os_modules and (
                    value.attr in raw_os_names
                    or value.attr.startswith(("spawn", "exec"))
                ):
                    canonical = f"os.{value.attr}"
            if canonical is None:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and raw_aliases.get(target.id) != canonical
                ):
                    raw_aliases[target.id] = canonical
                    changed = True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = None
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in subprocess_modules
            and node.func.attr in raw_names
        ):
            call_name = f"subprocess.{node.func.attr}"
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in os_modules
            and (
                node.func.attr in raw_os_names
                or node.func.attr.startswith(("spawn", "exec"))
            )
        ):
            call_name = f"os.{node.func.attr}"
        elif isinstance(node.func, ast.Name) and node.func.id in raw_aliases:
            call_name = raw_aliases[node.func.id]
        if call_name is None:
            continue
        site = (source.inventory_path, node.lineno, call_name)
        observed.add(site)
        allowed = (
            _PYTHON_TOOLING_PYTHON_SUBPROCESS_SITES
            | _PYTHON_TOOLING_NONPYTHON_SUBPROCESS_SITES
        )
        if site not in allowed:
            raise InventoryError("Python tooling raw subprocess site is not admitted")
        if any(
            keyword.arg == "shell"
            and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            )
            for keyword in node.keywords
        ):
            raise InventoryError(
                "Python tooling subprocess shell dispatch is forbidden"
            )
        if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            literals = {
                element.value
                for element in node.args[0].elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
            if literals & {"-m", "-c"}:
                raise InventoryError(
                    "Python tooling subprocess interpreter mode is forbidden"
                )
    admitted_for_source = {
        site
        for site in (
            _PYTHON_TOOLING_PYTHON_SUBPROCESS_SITES
            | _PYTHON_TOOLING_NONPYTHON_SUBPROCESS_SITES
        )
        if site[0] == source.inventory_path
    }
    if observed != admitted_for_source:
        raise InventoryError("Python tooling subprocess source sites changed")


def _python_windows_blas_source_audit(
    tree: ast.Module, reviewed: _PythonReviewedSourceModule
) -> None:
    canonical_globals = {
        "TEST_BLAS",
        "TEST_FILE_BLAS",
        "_TEST_BLAS_LIBRARY",
        "WINDOWS_TEST_BLAS_WINMODE",
        "WINDOWS_TEST_BLAS_REQUIRED_SYMBOLS",
    }
    protected_globals = {"TEST_BLAS", "TEST_FILE_BLAS", "_TEST_BLAS_LIBRARY"}
    loader_attributes = {"CDLL", "WinDLL", "windll"}
    allowed_assignments: dict[str, ast.AST] = {}

    def assigned_names(target: ast.AST) -> tuple[str, ...]:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(
                name for element in target.elts for name in assigned_names(element)
            )
        return ()

    if reviewed.module_name == "test_level2_report":
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            names = assigned_names(statement.targets[0])
            if not names or not set(names) <= canonical_globals:
                continue
            for name in names:
                if name in allowed_assignments:
                    raise InventoryError(
                        "Python Windows BLAS protected global is assigned more than once"
                    )
                allowed_assignments[name] = statement
        if set(allowed_assignments) != canonical_globals:
            raise InventoryError(
                "Python Windows BLAS protected globals lack canonical initialization"
            )
        winmode_assignment = allowed_assignments["WINDOWS_TEST_BLAS_WINMODE"]
        if (
            not isinstance(winmode_assignment, ast.Assign)
            or not isinstance(winmode_assignment.value, ast.Constant)
            or winmode_assignment.value.value != WINDOWS_PYTHON_TOOLING_BLAS_WINMODE
        ):
            raise InventoryError("Python Windows BLAS winmode is noncanonical")
        symbols_assignment = allowed_assignments["WINDOWS_TEST_BLAS_REQUIRED_SYMBOLS"]
        if not isinstance(symbols_assignment, ast.Assign):
            raise InventoryError("Python Windows BLAS symbol inventory is noncanonical")
        try:
            source_symbols = ast.literal_eval(symbols_assignment.value)
        except (ValueError, TypeError, RecursionError) as exc:
            raise InventoryError(
                "Python Windows BLAS symbol inventory is noncanonical"
            ) from exc
        if source_symbols != WINDOWS_PYTHON_TOOLING_BLAS_REQUIRED_SYMBOLS:
            raise InventoryError("Python Windows BLAS symbol inventory is noncanonical")
        library_assignment = allowed_assignments["TEST_BLAS"]
        if (
            library_assignment is not allowed_assignments["_TEST_BLAS_LIBRARY"]
            or not isinstance(library_assignment, ast.Assign)
            or not isinstance(library_assignment.targets[0], ast.Tuple)
            or assigned_names(library_assignment.targets[0])
            != ("TEST_BLAS", "_TEST_BLAS_LIBRARY")
            or not isinstance(library_assignment.value, ast.Call)
            or not isinstance(library_assignment.value.func, ast.Name)
            or library_assignment.value.func.id != "find_test_blas"
            or library_assignment.value.args
            or library_assignment.value.keywords
        ):
            raise InventoryError(
                "Python Windows BLAS loader result initialization is noncanonical"
            )

    aliases: dict[str, str] = {
        "ctypes": "ctypes",
        "test_level2_report": "test_level2_report",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                if alias.name == "ctypes":
                    aliases[local_name] = "ctypes"
                elif alias.name.rsplit(".", 1)[-1] == "test_level2_report":
                    aliases[local_name] = "test_level2_report"
                elif alias.name == "sys":
                    aliases[local_name] = "sys"
                elif alias.name == "importlib":
                    aliases[local_name] = "importlib"
                elif alias.name == "unittest.mock":
                    aliases[local_name] = "unittest.mock"
                elif alias.name in {
                    "builtins",
                    "functools",
                    "gc",
                    "importlib",
                    "inspect",
                    "marshal",
                    "operator",
                    "pickle",
                    "traceback",
                    "types",
                }:
                    aliases[local_name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                raise InventoryError(
                    "Python Windows BLAS authority wildcard import is forbidden"
                )
            module_name = node.module or ""
            for alias in node.names:
                local_name = alias.asname or alias.name
                if (
                    module_name.rsplit(".", 1)[-1] == "test_level2_report"
                    and alias.name in protected_globals
                ):
                    aliases[local_name] = f"test_level2_report.{alias.name}"
                elif module_name == "ctypes" and alias.name in loader_attributes:
                    aliases[local_name] = f"ctypes.{alias.name}"
                elif module_name == "unittest" and alias.name == "mock":
                    aliases[local_name] = "unittest.mock"
                elif module_name == "unittest.mock" and alias.name == "patch":
                    aliases[local_name] = "unittest.mock.patch"
                elif module_name in {
                    "builtins",
                    "functools",
                    "gc",
                    "importlib",
                    "inspect",
                    "marshal",
                    "operator",
                    "pickle",
                    "traceback",
                    "types",
                }:
                    aliases[local_name] = f"{module_name}.{alias.name}"
                elif module_name == "ctypes" and alias.name in {
                    "PyDLL",
                    "pythonapi",
                }:
                    aliases[local_name] = f"ctypes.{alias.name}"
                elif module_name == "sys" and alias.name in {
                    "_current_frames",
                    "_getframe",
                    "modules",
                }:
                    aliases[local_name] = f"sys.{alias.name}"

    def literal_text(node: ast.AST) -> str | None:
        return (
            node.value
            if isinstance(node, ast.Constant) and type(node.value) is str
            else None
        )

    def call_identifier(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            owner = call_identifier(node.value)
            return f"{owner}.{node.attr}" if owner is not None else None
        return None

    def expression_path(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            if node.id in aliases:
                return aliases[node.id]
            if (
                reviewed.module_name == "test_level2_report"
                and node.id in canonical_globals
            ):
                return f"test_level2_report.{node.id}"
            return None
        if isinstance(node, ast.Attribute):
            owner = expression_path(node.value)
            return f"{owner}.{node.attr}" if owner is not None else None
        if isinstance(node, ast.Call):
            call_name = call_identifier(node.func)
            if (
                call_name
                in {
                    "builtins.getattr",
                    "getattr",
                    "object.__getattribute__",
                }
                and len(node.args) >= 2
            ):
                owner = expression_path(node.args[0])
                attribute = literal_text(node.args[1])
                if owner is not None and attribute is not None:
                    return f"{owner}.{attribute}"
            if call_name in {"builtins.vars", "vars"} and len(node.args) == 1:
                owner = expression_path(node.args[0])
                return f"{owner}.__dict__" if owner is not None else None
            if (
                call_name in {"builtins.globals", "globals"}
                and not node.args
                and reviewed.module_name == "test_level2_report"
            ):
                return "test_level2_report.__dict__"
            if (
                call_name
                in {
                    "builtins.__import__",
                    "importlib.import_module",
                    "__import__",
                }
                and node.args
            ):
                module_name = literal_text(node.args[0])
                if (
                    module_name is not None
                    and module_name.rsplit(".", 1)[-1] == "test_level2_report"
                ):
                    return "test_level2_report"
            return None
        if isinstance(node, ast.Subscript):
            key = literal_text(node.slice)
            owner = expression_path(node.value)
            if (
                owner == "sys.modules"
                and key is not None
                and key.rsplit(".", 1)[-1] == "test_level2_report"
            ):
                return "test_level2_report"
            if owner is not None and key is not None:
                if owner.endswith(".__dict__"):
                    return f"{owner[:-9]}.{key}"
                return f"{owner}.{key}"
        return None

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            path = expression_path(value) if value is not None else None
            if path is None or not (
                path == "ctypes"
                or path.startswith("ctypes.")
                or path == "test_level2_report"
                or path.startswith("test_level2_report.")
            ):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and aliases.get(target.id) != path:
                    aliases[target.id] = path
                    changed = True

    def protected_path(path: str | None) -> bool:
        if path is None:
            return False
        parts = path.split(".")
        return (
            len(parts) >= 2
            and parts[0] == "test_level2_report"
            and parts[1] in protected_globals
            and (
                len(parts) == 2
                or parts[1] == "_TEST_BLAS_LIBRARY"
                and parts[2:] == ["_handle"]
            )
        ) or (
            len(parts) == 2 and parts[0] == "ctypes" and parts[1] in loader_attributes
        )

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    high_risk_calls = {
        "compile",
        "eval",
        "exec",
        "marshal.load",
        "marshal.loads",
        "pickle.load",
        "pickle.loads",
        "types.CodeType",
        "types.CellType",
        "types.FrameType",
        "types.FunctionType",
        "types.TracebackType",
    }
    frame_or_gc_calls = {
        "gc.get_objects",
        "gc.get_referrers",
        "inspect.currentframe",
        "inspect.getclosurevars",
        "inspect.getcoroutinelocals",
        "inspect.getgeneratorlocals",
        "inspect.stack",
        "sys._current_frames",
        "sys._getframe",
        "sys.exc_info",
        "traceback.walk_stack",
        "traceback.walk_tb",
    }
    meta_execution_identifiers = (
        high_risk_calls
        | frame_or_gc_calls
        | {
            "__import__",
            "builtins.__import__",
            "builtins.compile",
            "builtins.eval",
            "builtins.exec",
            "builtins.getattr",
            "builtins.globals",
            "builtins.locals",
            "builtins.vars",
            "builtins.__dict__",
            "builtins",
            "compile",
            "ctypes.PyDLL",
            "ctypes.pythonapi",
            "eval",
            "exec",
            "getattr",
            "globals",
            "importlib.import_module",
            "locals",
            "object.__getattribute__",
            "sys.modules",
            "vars",
            "__builtins__",
        }
    )
    runtime_authority_attributes = {
        "__annotations__",
        "__base__",
        "__bases__",
        "__builtins__",
        "__class__",
        "__code__",
        "__closure__",
        "__defaults__",
        "__func__",
        "__getattribute__",
        "__globals__",
        "__mro__",
        "__kwdefaults__",
        "__self__",
        "__subclasses__",
        "__traceback__",
        "__wrapped__",
        "CodeType",
        "FunctionType",
        "PyDLL",
        "_current_frames",
        "_getframe",
        "ag_await",
        "ag_code",
        "ag_frame",
        "cell_contents",
        "compile",
        "cr_await",
        "cr_code",
        "cr_frame",
        "currentframe",
        "eval",
        "exc_info",
        "exec",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "f_trace",
        "func_code",
        "get_objects",
        "get_referrers",
        "getattr",
        "gi_code",
        "gi_frame",
        "gi_yieldfrom",
        "globals",
        "import_module",
        "locals",
        "pythonapi",
        "stack",
        "tb_frame",
        "tb_next",
        "vars",
        "walk_stack",
        "walk_tb",
    }
    runtime_authority_literal_names = {
        identifier.rsplit(".", 1)[-1] for identifier in meta_execution_identifiers
    } | runtime_authority_attributes

    def enclosing_function_name(node: ast.AST) -> str | None:
        current = parents.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current.name
            current = parents.get(current)
        return None

    allowed_import_shapes: set[tuple[Any, ...]] = {
        ("import", (("argparse", None),)),
        ("import", (("benchmark_artifacts", None),)),
        ("import", (("benchmark_metadata", None),)),
        ("import", (("csv", None),)),
        ("import", (("ctypes", None),)),
        ("import", (("ctypes.util", None),)),
        ("import", (("dataclasses", None),)),
        ("import", (("errno", None),)),
        ("import", (("fcntl", None),)),
        ("import", (("hashlib", None),)),
        ("import", (("importlib.util", None),)),
        ("import", (("io", None),)),
        ("import", (("itertools", None),)),
        ("import", (("json", None),)),
        ("import", (("math", None),)),
        ("import", (("os", None),)),
        ("import", (("plot_gemm_sweep", None),)),
        ("import", (("plot_level1_report", None),)),
        ("import", (("plot_level2_report", None),)),
        ("import", (("report_publication", "publication"),)),
        ("import", (("report_publication", None),)),
        ("import", (("shutil", None),)),
        ("import", (("stat", None),)),
        ("import", (("subprocess", None),)),
        ("import", (("sys", None),)),
        ("import", (("tempfile", None),)),
        ("import", (("unittest", None),)),
        ("from", "__future__", 0, (("annotations", None),)),
        (
            "from",
            "contextlib",
            0,
            (("redirect_stderr", None), ("redirect_stdout", None)),
        ),
        ("from", "contextlib", 0, (("redirect_stdout", None),)),
        ("from", "pathlib", 0, (("Path", None),)),
        (
            "from",
            "report_comparison",
            0,
            (
                ("best_higher_row", None),
                ("best_lower_row", None),
                ("nearest_rank_percentile", None),
                ("paired_median_ratio", None),
                ("parse_positive_finite", None),
                ("positive_finite_axis_ticks", None),
                ("positive_finite_median", None),
                ("positive_finite_ratio", None),
                ("validate_performance_fields", None),
                ("validate_optional_metric_evidence", None),
            ),
        ),
        (
            "from",
            "report_publication",
            0,
            (
                ("ReportOutput", None),
                ("RollbackIndeterminateError", None),
                ("TransactionCompleteCleanupError", None),
                ("publish_outputs", None),
            ),
        ),
        (
            "from",
            "report_schedule",
            0,
            (
                ("collect_repeats", None),
                ("library_repeat_schedule", None),
                ("normalized_library_label", None),
                ("repeat_library_order", None),
                ("validate_schedule", None),
                ("validate_unique_library_labels", None),
            ),
        ),
        ("from", "unittest", 0, (("mock", None),)),
    }
    allowed_sys_attributes = {
        "argv",
        "executable",
        "float_info",
        "path",
        "platform",
        "stderr",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            shape = (
                "import",
                tuple((alias.name, alias.asname) for alias in node.names),
            )
            if shape not in allowed_import_shapes:
                raise InventoryError(
                    "Python Windows BLAS noncanonical import shape is forbidden"
                )
        elif isinstance(node, ast.ImportFrom):
            shape = (
                "from",
                node.module,
                node.level,
                tuple((alias.name, alias.asname) for alias in node.names),
            )
            if shape not in allowed_import_shapes:
                raise InventoryError(
                    "Python Windows BLAS noncanonical import shape is forbidden"
                )
        if not isinstance(node, ast.Attribute):
            continue
        identifier = call_identifier(node)
        if identifier is None:
            continue
        if identifier == "importlib.util":
            continue
        if identifier.startswith("importlib.util."):
            if (
                identifier
                in {
                    "importlib.util.module_from_spec",
                    "importlib.util.spec_from_file_location",
                }
                and enclosing_function_name(node) == "load_tool"
            ):
                continue
            raise InventoryError(
                "Python Windows BLAS noncanonical importlib authority use is forbidden"
            )
        if identifier == "sys.modules":
            parent = parents.get(node)
            if (
                isinstance(parent, ast.Subscript)
                and isinstance(parent.ctx, ast.Store)
                and enclosing_function_name(parent) == "load_tool"
            ):
                continue
            raise InventoryError(
                "Python Windows BLAS noncanonical sys.modules use is forbidden"
            )
        if identifier.startswith("sys."):
            attribute = identifier.split(".", 2)[1]
            if attribute not in allowed_sys_attributes:
                raise InventoryError(
                    "Python Windows BLAS noncanonical sys authority use is forbidden"
                )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            identifier = call_identifier(node.func)
            if identifier in high_risk_calls | frame_or_gc_calls:
                raise InventoryError(
                    "Python Windows BLAS meta-execution capability is forbidden"
                )
            if identifier in {
                "builtins.globals",
                "builtins.locals",
                "globals",
                "locals",
            } or (identifier in {"builtins.vars", "vars"} and not node.args):
                raise InventoryError(
                    "Python Windows BLAS dynamic namespace recovery is forbidden"
                )
            if identifier in {
                "builtins.getattr",
                "getattr",
                "object.__getattribute__",
            } and (len(node.args) < 2 or literal_text(node.args[1]) is None):
                raise InventoryError(
                    "Python Windows BLAS dynamic attribute recovery is forbidden"
                )
            if (
                identifier
                in {
                    "builtins.getattr",
                    "getattr",
                    "object.__getattribute__",
                }
                and len(node.args) >= 2
                and literal_text(node.args[1]) in runtime_authority_literal_names
            ):
                raise InventoryError(
                    "Python Windows BLAS runtime authority recovery is forbidden"
                )
            if (
                identifier
                in {
                    "object.__setattr__",
                    "setattr",
                }
                and len(node.args) >= 2
                and literal_text(node.args[1])
                in {
                    "__code__",
                    "func_code",
                }
            ):
                raise InventoryError(
                    "Python Windows BLAS code replacement is forbidden"
                )
            if identifier in {"__import__", "importlib.import_module"} and (
                not node.args or literal_text(node.args[0]) is None
            ):
                raise InventoryError(
                    "Python Windows BLAS dynamic import recovery is forbidden"
                )
        if isinstance(node, ast.Attribute):
            path = expression_path(node)
            if path in {"ctypes.PyDLL", "ctypes.pythonapi"}:
                raise InventoryError(
                    "Python Windows BLAS Python-runtime loader capability is forbidden"
                )
            if isinstance(node.ctx, (ast.Store, ast.Del)) and node.attr in {
                "__code__",
                "func_code",
            }:
                raise InventoryError(
                    "Python Windows BLAS code replacement is forbidden"
                )
        if isinstance(node, ast.Name) and aliases.get(node.id) in {
            "ctypes.PyDLL",
            "ctypes.pythonapi",
        }:
            raise InventoryError(
                "Python Windows BLAS Python-runtime loader capability is forbidden"
            )
        if (
            isinstance(node, ast.Subscript)
            and expression_path(node.value) == "sys.modules"
            and literal_text(node.slice) is None
        ):
            enclosing_function: ast.AST | None = parents.get(node)
            while enclosing_function is not None and not isinstance(
                enclosing_function, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                enclosing_function = parents.get(enclosing_function)
            if not (
                isinstance(enclosing_function, (ast.FunctionDef, ast.AsyncFunctionDef))
                and enclosing_function.name == "load_tool"
                and isinstance(node.ctx, ast.Store)
            ):
                raise InventoryError(
                    "Python Windows BLAS dynamic module recovery is forbidden"
                )

    mutation_capability = 0x01
    protected_target = 0x02
    meta_execution_capability = 0x04
    function_nodes = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    lambda_nodes = tuple(
        node for node in ast.walk(tree) if isinstance(node, ast.Lambda)
    )
    callable_nodes: tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda, ...] = (
        *function_nodes,
        *lambda_nodes,
    )
    class_nodes = tuple(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    )
    ScopeOwner = (
        ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef | None
    )

    def scoped_nodes(
        owner: ScopeOwner,
    ) -> tuple[ast.AST, ...]:
        if owner is None:
            roots: Iterable[ast.AST] = tree.body
        elif isinstance(owner, ast.Lambda):
            roots = (owner.body,)
        else:
            roots = owner.body
        pending = list(reversed(roots))
        nodes: list[ast.AST] = []
        while pending:
            node = pending.pop()
            nodes.append(node)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                continue
            pending.extend(reversed(tuple(ast.iter_child_nodes(node))))
        return tuple(nodes)

    scope_owners: tuple[ScopeOwner, ...] = (
        None,
        *class_nodes,
        *function_nodes,
        *lambda_nodes,
    )
    nodes_by_scope = {owner: scoped_nodes(owner) for owner in scope_owners}
    provenance_by_scope = {owner: {} for owner in scope_owners}
    provenance_environment_owners = {
        id(environment): owner for owner, environment in provenance_by_scope.items()
    }
    return_provenance = {function: 0 for function in callable_nodes}
    returned_callables = {function: set() for function in callable_nodes}
    returned_class_symbols = {function: set() for function in callable_nodes}
    returned_instances = {function: set() for function in callable_nodes}
    callable_bindings: dict[ScopeOwner, dict[str, set[ast.AST]]] = {
        owner: {} for owner in scope_owners
    }
    class_symbol_bindings: dict[ScopeOwner, dict[str, set[ast.ClassDef]]] = {
        owner: {} for owner in scope_owners
    }
    instance_bindings: dict[ScopeOwner, dict[str, set[ast.ClassDef]]] = {
        owner: {} for owner in scope_owners
    }
    global_declarations: dict[ScopeOwner, set[str]] = {
        owner: set() for owner in scope_owners
    }
    nonlocal_declarations: dict[ScopeOwner, set[str]] = {
        owner: set() for owner in scope_owners
    }
    trusted_provenance_receivers = {
        "Path",
        "bool",
        "bytes",
        "ctypes.CDLL",
        "id",
        "isinstance",
        "len",
        "repr",
        "str",
        "subprocess.run",
        "type",
        "unittest.skipUnless",
    }

    def mutation_capability_expression(node: ast.AST) -> bool:
        identifier = call_identifier(node)
        if identifier in {"setattr", "delattr", "object.__setattr__"}:
            return True
        if identifier is not None and (
            identifier in {"mock.patch", "unittest.mock.patch"}
            or identifier.startswith("mock.patch.")
            or identifier.startswith("unittest.mock.patch.")
        ):
            return True
        if isinstance(node, ast.Attribute) and node.attr in {
            "update",
            "setdefault",
            "pop",
            "clear",
            "__setitem__",
            "__delitem__",
        }:
            return expression_path(node.value) in {
                "test_level2_report.__dict__",
                "ctypes.__dict__",
            }
        return False

    def meta_execution_capability_expression(node: ast.AST) -> bool:
        identifier = call_identifier(node)
        path = expression_path(node)
        return (
            identifier in meta_execution_identifiers
            or path in meta_execution_identifiers
            or isinstance(node, ast.Attribute)
            and node.attr in runtime_authority_attributes
        )

    def target_expression(node: ast.AST) -> bool:
        path = expression_path(node)
        literal_path = literal_text(node)
        return (
            protected_path(literal_path)
            or path
            in {
                "ctypes",
                "ctypes.__dict__",
                "test_level2_report",
                "test_level2_report.__dict__",
            }
            or protected_path(path)
        )

    def enclosing_class(
        function: ScopeOwner,
    ) -> ast.ClassDef | None:
        current: ast.AST | None = function
        while current is not None:
            current = parents.get(current)
            if isinstance(current, ast.ClassDef):
                return current
        return None

    def containing_scope(node: ast.AST) -> ScopeOwner:
        parent = parents.get(node)
        while parent is not None and not isinstance(
            parent,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
                ast.ClassDef,
                ast.Module,
            ),
        ):
            parent = parents.get(parent)
        return None if isinstance(parent, ast.Module) else parent

    def parent_lookup_scope(owner: ScopeOwner) -> ScopeOwner:
        if owner is None:
            return None
        parent = containing_scope(owner)
        if isinstance(
            owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ) and isinstance(parent, ast.ClassDef):
            return containing_scope(parent)
        return parent

    for declaration in ast.walk(tree):
        if isinstance(declaration, ast.Global):
            global_declarations[containing_scope(declaration)].update(declaration.names)
        elif isinstance(declaration, ast.Nonlocal):
            nonlocal_declarations[containing_scope(declaration)].update(
                declaration.names
            )

    def assignment_destination(name: str, owner: ScopeOwner) -> ScopeOwner:
        if name in global_declarations[owner]:
            return None
        if name not in nonlocal_declarations[owner]:
            return owner
        destination = parent_lookup_scope(owner)
        while isinstance(destination, ast.ClassDef):
            destination = parent_lookup_scope(destination)
        return destination

    for import_node in ast.walk(tree):
        if isinstance(import_node, ast.Import):
            imported_names = tuple(
                alias.asname or alias.name.split(".", 1)[0]
                for alias in import_node.names
            )
        elif isinstance(import_node, ast.ImportFrom):
            imported_names = tuple(
                alias.asname or alias.name for alias in import_node.names
            )
        else:
            continue
        import_owner = containing_scope(import_node)
        if any(
            assignment_destination(name, import_owner) is not import_owner
            for name in imported_names
        ):
            raise InventoryError(
                "Python Windows BLAS import escaped through a global or nonlocal binding"
            )

    def resolve_scoped_bindings(
        bindings: dict[ScopeOwner, dict[str, set[Any]]],
        name: str,
        owner: ScopeOwner,
    ) -> set[Any]:
        current = owner
        while True:
            candidates = bindings[current].get(name)
            if candidates:
                return set(candidates)
            if current is None:
                return set()
            current = parent_lookup_scope(current)

    def bind_callable(owner: ScopeOwner, name: str, candidate: ast.AST) -> bool:
        destination = assignment_destination(name, owner)
        if destination is not owner:
            raise InventoryError(
                "Python Windows BLAS function identity escaped through a global or "
                "nonlocal definition binding"
            )
        candidates = callable_bindings[destination].setdefault(name, set())
        prior = len(candidates)
        candidates.add(candidate)
        return len(candidates) != prior

    def bind_class(owner: ScopeOwner, name: str, candidate: ast.ClassDef) -> bool:
        destination = assignment_destination(name, owner)
        if destination is not owner:
            raise InventoryError(
                "Python Windows BLAS class identity escaped through a global or "
                "nonlocal definition binding"
            )
        candidates = class_symbol_bindings[destination].setdefault(name, set())
        prior = len(candidates)
        candidates.add(candidate)
        return len(candidates) != prior

    for function in function_nodes:
        bind_callable(containing_scope(function), function.name, function)
    for class_node in class_nodes:
        bind_class(containing_scope(class_node), class_node.name, class_node)

    decorator_aliases = {
        "staticmethod": "staticmethod",
        "builtins.staticmethod": "staticmethod",
        "classmethod": "classmethod",
        "builtins.classmethod": "classmethod",
        "property": "property",
        "builtins.property": "property",
    }
    decorator_aliases_changed = True
    while decorator_aliases_changed:
        decorator_aliases_changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            value_identifier = (
                decorator_aliases.get(node.value.id)
                if isinstance(node.value, ast.Name)
                else decorator_aliases.get(call_identifier(node.value) or "")
            )
            if value_identifier is None:
                continue
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if decorator_aliases.get(target.id) != value_identifier:
                    decorator_aliases[target.id] = value_identifier
                    decorator_aliases_changed = True

    def decorator_identifier(decorator: ast.AST) -> str | None:
        subject = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(subject, ast.Name) and subject.id in decorator_aliases:
            return decorator_aliases[subject.id]
        return decorator_aliases.get(call_identifier(subject) or "") or call_identifier(
            subject
        )

    trusted_decorator_identifiers = {
        "mock.patch",
        "mock.patch.dict",
        "mock.patch.object",
        "unittest.mock.patch",
        "unittest.mock.patch.dict",
        "unittest.mock.patch.object",
        "unittest.skipIf",
        "unittest.skipUnless",
    }

    def method_binding_kind(function: ast.AST) -> str:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "function"
        decorator_names = {
            decorator_identifier(decorator) for decorator in function.decorator_list
        }
        if "staticmethod" in decorator_names:
            return "static"
        if "classmethod" in decorator_names:
            return "class"
        if "property" in decorator_names:
            return "property"
        return "instance"

    def has_unknown_decorator(function: ast.AST) -> bool:
        return isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            decorator_identifier(decorator)
            not in {
                "staticmethod",
                "classmethod",
                "property",
                *trusted_decorator_identifiers,
            }
            for decorator in function.decorator_list
        )

    for function in function_nodes:
        owner_class = containing_scope(function)
        if not isinstance(owner_class, ast.ClassDef):
            continue
        parameters = (*function.args.posonlyargs, *function.args.args)
        if not parameters:
            continue
        receiver = parameters[0]
        binding_kind = method_binding_kind(function)
        bindings = (
            class_symbol_bindings
            if binding_kind == "class"
            else (
                instance_bindings if binding_kind in {"instance", "property"} else None
            )
        )
        if bindings is not None:
            bindings[function].setdefault(receiver.arg, set()).add(owner_class)

    trusted_external_bases = {"dict", "object", "unittest.TestCase"}

    def resolve_class_reference(
        node: ast.AST, owner: ScopeOwner
    ) -> set[ast.ClassDef] | None:
        if isinstance(node, ast.Name):
            candidates = {
                candidate
                for candidate in resolve_scoped_bindings(
                    class_symbol_bindings, node.id, owner
                )
                if isinstance(candidate, ast.ClassDef)
            }
            if candidates:
                return candidates
            return set() if call_identifier(node) in trusted_external_bases else None
        if isinstance(node, ast.Attribute):
            if call_identifier(node) in trusted_external_bases:
                return set()
            owners = resolve_class_reference(node.value, owner)
            if owners is None or not owners:
                return None
            candidates = {
                candidate
                for owner_class in owners
                for candidate in class_symbol_bindings[owner_class].get(
                    node.attr, set()
                )
                if isinstance(candidate, ast.ClassDef)
            }
            return candidates or None
        return None

    class_bases: dict[ast.ClassDef, set[ast.ClassDef]] = {
        class_node: set() for class_node in class_nodes
    }
    for class_node in class_nodes:
        if class_node.keywords:
            raise InventoryError(
                "Python Windows BLAS dynamic or parameterized metaclass is forbidden"
            )
    bases_changed = True
    while bases_changed:
        bases_changed = False
        for class_node in class_nodes:
            for base in class_node.bases:
                candidates = resolve_class_reference(base, containing_scope(class_node))
                if candidates is None:
                    raise InventoryError(
                        "Python Windows BLAS class MRO cannot be proven statically"
                    )
                prior = len(class_bases[class_node])
                class_bases[class_node].update(candidates)
                if len(class_bases[class_node]) != prior:
                    bases_changed = True

    def class_lineage(class_node: ast.ClassDef) -> set[ast.ClassDef]:
        lineage = {class_node}
        pending = [class_node]
        while pending:
            current = pending.pop()
            for base in class_bases[current]:
                if base not in lineage:
                    lineage.add(base)
                    pending.append(base)
        return lineage

    identity_sanitizing_calls = {
        "mock.patch",
        "mock.patch.dict",
        "mock.patch.object",
        "unittest.mock.patch",
        "unittest.mock.patch.dict",
        "unittest.mock.patch.object",
    }
    container_lookup_methods = {
        "__getitem__",
        "get",
        "items",
        "keys",
        "pop",
        "popitem",
        "popleft",
        "setdefault",
        "values",
    }
    container_mutator_methods = {
        "__setitem__",
        "add",
        "append",
        "extend",
        "insert",
        "setdefault",
        "update",
    }
    binary_protocol_methods = {
        "__add__",
        "__and__",
        "__floordiv__",
        "__lshift__",
        "__matmul__",
        "__mod__",
        "__mul__",
        "__or__",
        "__pow__",
        "__radd__",
        "__rand__",
        "__rfloordiv__",
        "__rlshift__",
        "__rmatmul__",
        "__rmod__",
        "__rmul__",
        "__ror__",
        "__rpow__",
        "__rrshift__",
        "__rshift__",
        "__rsub__",
        "__rtruediv__",
        "__rxor__",
        "__sub__",
        "__truediv__",
        "__xor__",
    }
    comparison_protocol_methods = {
        "__eq__",
        "__ge__",
        "__gt__",
        "__le__",
        "__lt__",
        "__ne__",
    }

    def class_method_candidates(
        classes: set[ast.ClassDef], method_names: set[str]
    ) -> set[ast.AST]:
        return {
            candidate
            for class_node in classes
            for owner_class in class_lineage(class_node)
            for method_name in method_names
            for candidate in callable_bindings[owner_class].get(method_name, set())
        }

    def protocol_returned_callables(
        classes: set[ast.ClassDef], method_names: set[str]
    ) -> set[ast.AST]:
        return {
            returned
            for method in class_method_candidates(classes, method_names)
            for returned in returned_callables.get(method, set())
        }

    def protocol_returned_instances(
        classes: set[ast.ClassDef], method_names: set[str]
    ) -> set[ast.ClassDef]:
        return {
            returned
            for method in class_method_candidates(classes, method_names)
            for returned in returned_instances.get(method, set())
        }

    def protocol_returned_class_symbols(
        classes: set[ast.ClassDef], method_names: set[str]
    ) -> set[ast.ClassDef]:
        return {
            returned
            for method in class_method_candidates(classes, method_names)
            for returned in returned_class_symbols.get(method, set())
        }

    def protocol_returned_provenance(
        classes: set[ast.ClassDef], method_names: set[str]
    ) -> int:
        provenance = 0
        for method in class_method_candidates(classes, method_names):
            provenance |= return_provenance.get(method, 0)
        return provenance

    def protocol_chain_identities(
        classes: set[ast.ClassDef], first: set[str], second: set[str]
    ) -> tuple[int, set[ast.AST], set[ast.ClassDef], set[ast.ClassDef]]:
        intermediate_instances = protocol_returned_instances(classes, first)
        provenance = protocol_returned_provenance(
            classes, first
        ) | protocol_returned_provenance(intermediate_instances, second)
        callables = protocol_returned_callables(
            classes, first
        ) | protocol_returned_callables(intermediate_instances, second)
        instances = intermediate_instances | protocol_returned_instances(
            intermediate_instances, second
        )
        symbols = protocol_returned_class_symbols(
            classes, first
        ) | protocol_returned_class_symbols(intermediate_instances, second)
        if first == {"__iter__"} and second == {"__next__"}:
            provenance |= protocol_returned_provenance(classes, {"__getitem__"})
            callables.update(protocol_returned_callables(classes, {"__getitem__"}))
            instances.update(protocol_returned_instances(classes, {"__getitem__"}))
            symbols.update(protocol_returned_class_symbols(classes, {"__getitem__"}))
        return provenance, callables, instances, symbols

    def expression_class_symbols(node: ast.AST, owner: ScopeOwner) -> set[ast.ClassDef]:
        return expression_class_identities(node, owner, symbols=True)

    def expression_classes(node: ast.AST, owner: ScopeOwner) -> set[ast.ClassDef]:
        return expression_class_identities(node, owner, symbols=False)

    def expression_class_identities(
        node: ast.AST, owner: ScopeOwner, *, symbols: bool
    ) -> set[ast.ClassDef]:
        bindings = class_symbol_bindings if symbols else instance_bindings
        returned = returned_class_symbols if symbols else returned_instances
        resolver = expression_class_symbols if symbols else expression_classes
        protocol_resolver = (
            protocol_returned_class_symbols if symbols else protocol_returned_instances
        )
        if isinstance(node, ast.Name):
            return {
                candidate
                for candidate in resolve_scoped_bindings(bindings, node.id, owner)
                if isinstance(candidate, ast.ClassDef)
            }
        if isinstance(node, ast.Attribute):
            owner_instances = expression_classes(node.value, owner)
            owner_classes = owner_instances | expression_class_symbols(
                node.value, owner
            )
            descriptor_classes = {
                descriptor
                for owner_class in owner_classes
                for defining_class in class_lineage(owner_class)
                for descriptor in instance_bindings[defining_class].get(
                    node.attr, set()
                )
            }
            property_getters = {
                candidate
                for owner_class in owner_instances
                for candidate in class_method_candidates({owner_class}, {node.attr})
                if method_binding_kind(candidate) == "property"
            }
            return protocol_resolver(descriptor_classes, {"__get__"}) | {
                identity
                for getter in property_getters
                for identity in returned.get(getter, set())
            }
        if isinstance(node, ast.Call):
            call_name = call_identifier(node.func)
            if call_name == "iter" and len(node.args) >= 2:
                producers = expression_callables(node.args[0], owner)
                producers.update(
                    class_method_candidates(
                        expression_classes(node.args[0], owner), {"__call__"}
                    )
                )
                return {
                    identity
                    for producer in producers
                    for identity in returned.get(producer, set())
                } | {
                    identity
                    for argument in node.args
                    for identity in resolver(argument, owner)
                }
            adapter_methods = {
                "anext": {"__anext__"},
                "iter": {"__getitem__", "__iter__"},
                "next": {"__next__"},
            }.get(call_name)
            if adapter_methods is not None and node.args:
                adapter_classes = expression_classes(node.args[0], owner)
                return protocol_resolver(adapter_classes, adapter_methods) | {
                    identity
                    for argument in node.args
                    for identity in resolver(argument, owner)
                }
            constructors = expression_class_symbols(node.func, owner)
            callable_instances = expression_classes(node.func, owner)
            producers = expression_callables(node.func, owner)
            producers.update(class_method_candidates(callable_instances, {"__call__"}))
            if constructors:
                producers.update(
                    class_method_candidates(constructors, {"__init__", "__new__"})
                )
            produced = {
                identity
                for producer in producers
                for identity in returned.get(producer, set())
            }
            if not symbols:
                produced.update(constructors)
            if constructors or producers:
                return produced
            if call_identifier(node.func) in identity_sanitizing_calls:
                return set()
            derived = {
                candidate
                for argument in (
                    *node.args,
                    *(keyword.value for keyword in node.keywords),
                )
                for candidate in resolver(argument, owner)
            }
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in container_lookup_methods
            ):
                derived.update(resolver(node.func.value, owner))
            return derived
        if isinstance(node, ast.IfExp):
            return resolver(node.body, owner) | resolver(node.orelse, owner)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return {
                candidate
                for element in node.elts
                for candidate in resolver(element, owner)
            }
        if isinstance(node, ast.Dict):
            return {
                candidate
                for element in (*node.keys, *node.values)
                if element is not None
                for candidate in resolver(element, owner)
            }
        if isinstance(node, ast.Subscript):
            receiver_classes = expression_classes(node.value, owner)
            return (
                resolver(node.value, owner)
                | resolver(node.slice, owner)
                | protocol_resolver(receiver_classes, {"__getitem__"})
            )
        if isinstance(node, ast.Starred):
            return resolver(node.value, owner)
        if isinstance(node, ast.NamedExpr):
            return resolver(node.value, owner)
        if isinstance(node, ast.Await):
            receiver_classes = expression_classes(node.value, owner)
            _, _, chained_instances, chained_symbols = protocol_chain_identities(
                receiver_classes, {"__await__"}, {"__next__"}
            )
            return resolver(node.value, owner) | (
                chained_symbols if symbols else chained_instances
            )
        if isinstance(node, ast.UnaryOp):
            receiver_classes = expression_classes(node.operand, owner)
            methods = {
                ast.Invert: "__invert__",
                ast.UAdd: "__pos__",
                ast.USub: "__neg__",
            }
            method = methods.get(type(node.op))
            return resolver(node.operand, owner) | (
                protocol_resolver(receiver_classes, {method})
                if method is not None
                else set()
            )
        if isinstance(node, ast.BinOp):
            operand_classes = expression_classes(node.left, owner) | expression_classes(
                node.right, owner
            )
            return (
                resolver(node.left, owner)
                | resolver(node.right, owner)
                | protocol_resolver(operand_classes, binary_protocol_methods)
            )
        if isinstance(node, ast.BoolOp):
            return {
                candidate
                for value in node.values
                for candidate in resolver(value, owner)
            }
        if isinstance(node, ast.Compare):
            values = (node.left, *node.comparators)
            operand_classes = {
                candidate
                for value in values
                for candidate in expression_classes(value, owner)
            }
            return {
                candidate for value in values for candidate in resolver(value, owner)
            } | protocol_resolver(operand_classes, comparison_protocol_methods)
        if isinstance(node, ast.FormattedValue):
            receiver_classes = expression_classes(node.value, owner)
            return resolver(node.value, owner) | protocol_resolver(
                receiver_classes, {"__format__"}
            )
        if isinstance(node, ast.JoinedStr):
            return {
                candidate
                for value in node.values
                if isinstance(value, ast.FormattedValue)
                for candidate in resolver(value, owner)
            }
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return resolver(node.value, owner) if node.value is not None else set()
        if isinstance(
            node, (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp)
        ):
            values: tuple[ast.expr, ...]
            if isinstance(node, ast.DictComp):
                values = (node.key, node.value)
            else:
                values = (node.elt,)
            values += tuple(
                expression
                for generator in node.generators
                for expression in (generator.iter, *generator.ifs)
            )
            return {
                candidate for value in values for candidate in resolver(value, owner)
            }
        return set()

    def expression_callables(node: ast.AST, owner: ScopeOwner) -> set[ast.AST]:
        if isinstance(node, ast.Lambda):
            return {node}
        if isinstance(node, ast.Name):
            return resolve_scoped_bindings(callable_bindings, node.id, owner)
        if isinstance(node, ast.Attribute):
            explicit_callables = (
                expression_callables(node.value, owner)
                if node.attr == "__call__"
                else set()
            )
            owner_instances = expression_classes(node.value, owner)
            owner_classes = owner_instances | expression_class_symbols(
                node.value, owner
            )
            if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
                owner_class = enclosing_class(owner)
                if owner_class is not None:
                    owner_classes.add(owner_class)
            direct_methods = {
                candidate
                for owner_class in owner_classes
                for defining_class in class_lineage(owner_class)
                for candidate in callable_bindings[defining_class].get(node.attr, set())
                if method_binding_kind(candidate) != "property"
            }
            descriptor_classes = {
                descriptor
                for owner_class in owner_classes
                for defining_class in class_lineage(owner_class)
                for descriptor in instance_bindings[defining_class].get(
                    node.attr, set()
                )
            }
            property_getters = {
                candidate
                for owner_class in owner_instances
                for candidate in class_method_candidates({owner_class}, {node.attr})
                if method_binding_kind(candidate) == "property"
            }
            return (
                explicit_callables
                | direct_methods
                | protocol_returned_callables(descriptor_classes, {"__get__"})
                | {
                    returned
                    for getter in property_getters
                    for returned in returned_callables.get(getter, set())
                }
            )
        if isinstance(node, ast.Call):
            call_name = call_identifier(node.func)
            if call_name == "iter" and len(node.args) >= 2:
                producers = expression_callables(node.args[0], owner)
                producers.update(
                    class_method_candidates(
                        expression_classes(node.args[0], owner), {"__call__"}
                    )
                )
                return {
                    returned
                    for producer in producers
                    for returned in returned_callables.get(producer, set())
                } | {
                    callback
                    for argument in node.args
                    for callback in expression_callables(argument, owner)
                }
            adapter_methods = {
                "anext": {"__anext__"},
                "iter": {"__getitem__", "__iter__"},
                "next": {"__next__"},
            }.get(call_name)
            if adapter_methods is not None and node.args:
                return protocol_returned_callables(
                    expression_classes(node.args[0], owner), adapter_methods
                ) | {
                    callback
                    for argument in node.args
                    for callback in expression_callables(argument, owner)
                }
            constructors = expression_class_symbols(node.func, owner)
            if constructors:
                return protocol_returned_callables(
                    constructors, {"__init__", "__new__"}
                )
            producers = expression_callables(node.func, owner)
            producers.update(
                class_method_candidates(
                    expression_classes(node.func, owner), {"__call__"}
                )
            )
            produced = {
                returned
                for producer in producers
                for returned in returned_callables.get(producer, set())
            }
            if producers:
                return produced
            if call_identifier(node.func) in identity_sanitizing_calls:
                return set()
            derived = {
                candidate
                for argument in (
                    *node.args,
                    *(keyword.value for keyword in node.keywords),
                )
                for candidate in expression_callables(argument, owner)
            }
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in container_lookup_methods
            ):
                derived.update(expression_callables(node.func.value, owner))
            return derived
        if isinstance(node, ast.IfExp):
            return expression_callables(node.body, owner) | expression_callables(
                node.orelse, owner
            )
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return {
                candidate
                for element in node.elts
                for candidate in expression_callables(element, owner)
            }
        if isinstance(node, ast.Dict):
            return {
                candidate
                for element in (*node.keys, *node.values)
                if element is not None
                for candidate in expression_callables(element, owner)
            }
        if isinstance(node, ast.Subscript):
            receiver_classes = expression_classes(node.value, owner)
            return (
                expression_callables(node.value, owner)
                | expression_callables(node.slice, owner)
                | protocol_returned_callables(receiver_classes, {"__getitem__"})
            )
        if isinstance(node, ast.Starred):
            return expression_callables(node.value, owner)
        if isinstance(node, ast.NamedExpr):
            return expression_callables(node.value, owner)
        if isinstance(node, ast.UnaryOp):
            receiver_classes = expression_classes(node.operand, owner)
            methods = {
                ast.Invert: "__invert__",
                ast.UAdd: "__pos__",
                ast.USub: "__neg__",
            }
            method = methods.get(type(node.op))
            return expression_callables(node.operand, owner) | (
                protocol_returned_callables(receiver_classes, {method})
                if method is not None
                else set()
            )
        if isinstance(node, ast.Await):
            receiver_classes = expression_classes(node.value, owner)
            _, chained_callables, _, _ = protocol_chain_identities(
                receiver_classes, {"__await__"}, {"__next__"}
            )
            return expression_callables(node.value, owner) | chained_callables
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return (
                expression_callables(node.value, owner)
                if node.value is not None
                else set()
            )
        if isinstance(node, ast.BinOp):
            operand_classes = expression_classes(node.left, owner) | expression_classes(
                node.right, owner
            )
            return {
                candidate
                for child in (node.left, node.right)
                for candidate in expression_callables(child, owner)
            } | protocol_returned_callables(operand_classes, binary_protocol_methods)
        if isinstance(node, ast.BoolOp):
            return {
                candidate
                for child in node.values
                for candidate in expression_callables(child, owner)
            }
        if isinstance(node, ast.Compare):
            values = (node.left, *node.comparators)
            operand_classes = {
                candidate
                for value in values
                for candidate in expression_classes(value, owner)
            }
            return {
                candidate
                for value in values
                for candidate in expression_callables(value, owner)
            } | protocol_returned_callables(
                operand_classes, comparison_protocol_methods
            )
        if isinstance(node, ast.FormattedValue):
            receiver_classes = expression_classes(node.value, owner)
            return expression_callables(
                node.value, owner
            ) | protocol_returned_callables(receiver_classes, {"__format__"})
        if isinstance(node, ast.JoinedStr):
            return {
                candidate
                for value in node.values
                if isinstance(value, ast.FormattedValue)
                for candidate in expression_callables(value, owner)
            }
        if isinstance(
            node, (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp)
        ):
            values: tuple[ast.expr, ...]
            if isinstance(node, ast.DictComp):
                values = (node.key, node.value)
            else:
                values = (node.elt,)
            values += tuple(
                expression
                for generator in node.generators
                for expression in (generator.iter, *generator.ifs)
            )
            return {
                candidate
                for value in values
                for candidate in expression_callables(value, owner)
            }
        return set()

    def resolve_user_functions(call: ast.Call, owner: ScopeOwner) -> set[ast.AST]:
        functions = expression_callables(call.func, owner)
        functions.update(
            class_method_candidates(expression_classes(call.func, owner), {"__call__"})
        )
        functions.update(
            class_method_candidates(
                expression_class_symbols(call.func, owner), {"__init__", "__new__"}
            )
        )
        return functions

    def call_has_implicit_receiver(
        call: ast.Call, function: ast.AST, owner: ScopeOwner
    ) -> bool:
        owner_class = containing_scope(function)
        if not isinstance(owner_class, ast.ClassDef):
            return False
        binding_kind = method_binding_kind(function)
        if binding_kind == "static":
            return False
        constructors = expression_class_symbols(call.func, owner)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            function.name in {"__init__", "__new__"} and owner_class in constructors
        ):
            return True
        if binding_kind == "class":
            return isinstance(call.func, ast.Attribute) or bool(constructors)
        callable_instances = expression_classes(call.func, owner)
        if owner_class in callable_instances:
            return True
        if isinstance(call.func, ast.Attribute):
            receiver_instances = expression_classes(call.func.value, owner)
            receiver_symbols = expression_class_symbols(call.func.value, owner)
            if receiver_instances and receiver_symbols:
                raise InventoryError(
                    "Python Windows BLAS method receiver binding is ambiguous"
                )
            if owner_class in receiver_instances:
                return True
            if owner_class in receiver_symbols:
                return False
            if call.func.attr == "__call__" and function in expression_callables(
                call.func.value, owner
            ):
                return True
        if isinstance(call.func, ast.Name):
            raise InventoryError(
                "Python Windows BLAS aliased method receiver binding is ambiguous"
            )
        return False

    def reflection_receiver_has_runtime_authority(
        node: ast.AST, owner: ScopeOwner
    ) -> bool:
        return (
            call_identifier(node)
            in {
                "builtins",
                "gc",
                "inspect",
                "object",
                "sys",
                "traceback",
                "type",
                "types",
                "__builtins__",
            }
            or bool(expression_callables(node, owner))
            or bool(expression_classes(node, owner))
            or meta_execution_capability_expression(node)
        )

    def binding_provenance(
        name: str,
        owner: ScopeOwner,
    ) -> int:
        current = owner
        while True:
            environment = provenance_by_scope[current]
            if name in environment:
                return environment[name]
            if current is None:
                return 0
            current = parent_lookup_scope(current)

    def expression_provenance(
        node: ast.AST,
        owner: ScopeOwner,
    ) -> int:
        provenance = 0
        if meta_execution_capability_expression(node):
            provenance |= meta_execution_capability
        if mutation_capability_expression(node):
            provenance |= mutation_capability
            if isinstance(node, ast.Attribute) and expression_path(node.value) in {
                "test_level2_report.__dict__",
                "ctypes.__dict__",
            }:
                provenance |= protected_target
        if target_expression(node):
            provenance |= protected_target
        if isinstance(node, ast.Name):
            provenance |= binding_provenance(node.id, owner)
        elif isinstance(node, ast.Constant):
            pass
        elif isinstance(node, ast.Attribute):
            path = expression_path(node)
            if not (
                path is not None
                and path.startswith("ctypes.")
                and path
                not in {
                    "ctypes.CDLL",
                    "ctypes.PyDLL",
                    "ctypes.WinDLL",
                    "ctypes.__dict__",
                    "ctypes.pythonapi",
                    "ctypes.windll",
                }
            ):
                provenance |= expression_provenance(node.value, owner)
            owner_classes = expression_classes(node.value, owner)
            descriptor_classes = {
                descriptor
                for owner_class in owner_classes
                for defining_class in class_lineage(owner_class)
                for descriptor in instance_bindings[defining_class].get(
                    node.attr, set()
                )
            }
            provenance |= protocol_returned_provenance(descriptor_classes, {"__get__"})
            property_getters = {
                candidate
                for owner_class in owner_classes
                for candidate in class_method_candidates({owner_class}, {node.attr})
                if method_binding_kind(candidate) == "property"
            }
            for getter in property_getters:
                provenance |= return_provenance.get(getter, 0)
        elif isinstance(node, ast.Call):
            call_name = call_identifier(node.func)
            if call_name == "iter" and len(node.args) >= 2:
                producers = expression_callables(node.args[0], owner)
                producers.update(
                    class_method_candidates(
                        expression_classes(node.args[0], owner), {"__call__"}
                    )
                )
                for producer in producers:
                    provenance |= return_provenance.get(producer, 0)
                for argument in node.args:
                    provenance |= expression_provenance(argument, owner)
                return provenance
            adapter_methods = {
                "anext": {"__anext__"},
                "iter": {"__getitem__", "__iter__"},
                "next": {"__next__"},
            }.get(call_name)
            if adapter_methods is not None and node.args:
                for argument in node.args:
                    provenance |= expression_provenance(argument, owner)
                provenance |= protocol_returned_provenance(
                    expression_classes(node.args[0], owner), adapter_methods
                )
                return provenance
            if call_name in trusted_provenance_receivers:
                return provenance
            if (
                call_name
                in {
                    "builtins.getattr",
                    "getattr",
                    "object.__getattribute__",
                }
                and len(node.args) >= 2
                and literal_text(node.args[1]) is not None
                and literal_text(node.args[1]) not in runtime_authority_literal_names
                and not reflection_receiver_has_runtime_authority(node.args[0], owner)
                and not (
                    expression_provenance(node.args[0], owner)
                    & (protected_target | meta_execution_capability)
                )
            ):
                return provenance & ~meta_execution_capability
            if (
                call_name in {"builtins.vars", "vars"}
                and len(node.args) == 1
                and not reflection_receiver_has_runtime_authority(node.args[0], owner)
                and not (
                    expression_provenance(node.args[0], owner)
                    & (protected_target | meta_execution_capability)
                )
            ):
                return provenance & ~meta_execution_capability
            called_functions = resolve_user_functions(node, owner)
            callee_provenance = expression_provenance(node.func, owner)
            if mutation_capability_expression(node.func):
                callee_provenance &= ~mutation_capability
            provenance |= callee_provenance
            if called_functions:
                for called_function in called_functions:
                    provenance |= return_provenance[called_function]
                if isinstance(node.func, ast.Attribute):
                    provenance |= expression_provenance(node.func.value, owner)
            else:
                for argument in node.args:
                    provenance |= expression_provenance(argument, owner)
                for keyword in node.keywords:
                    provenance |= expression_provenance(keyword.value, owner)
        elif isinstance(node, ast.IfExp):
            provenance |= expression_provenance(node.body, owner)
            provenance |= expression_provenance(node.orelse, owner)
        elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for element in node.elts:
                provenance |= expression_provenance(element, owner)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if key is not None:
                    provenance |= expression_provenance(key, owner)
                provenance |= expression_provenance(value, owner)
        elif isinstance(node, ast.Subscript):
            receiver_classes = expression_classes(node.value, owner)
            provenance |= expression_provenance(node.value, owner)
            provenance |= expression_provenance(node.slice, owner)
            provenance |= protocol_returned_provenance(
                receiver_classes, {"__getitem__"}
            )
        elif isinstance(node, ast.Starred):
            provenance |= expression_provenance(node.value, owner)
        elif isinstance(node, ast.NamedExpr):
            provenance |= expression_provenance(node.value, owner)
        elif isinstance(node, ast.Lambda):
            for default in node.args.defaults:
                provenance |= expression_provenance(default, owner)
            for default in node.args.kw_defaults:
                if default is not None:
                    provenance |= expression_provenance(default, owner)
        elif isinstance(node, ast.Await):
            receiver_classes = expression_classes(node.value, owner)
            chained_provenance, _, _, _ = protocol_chain_identities(
                receiver_classes, {"__await__"}, {"__next__"}
            )
            provenance |= expression_provenance(node.value, owner)
            provenance |= chained_provenance
        elif isinstance(node, ast.UnaryOp):
            receiver_classes = expression_classes(node.operand, owner)
            methods = {
                ast.Invert: "__invert__",
                ast.UAdd: "__pos__",
                ast.USub: "__neg__",
            }
            method = methods.get(type(node.op))
            provenance |= expression_provenance(node.operand, owner)
            if method is not None:
                provenance |= protocol_returned_provenance(receiver_classes, {method})
        elif isinstance(node, ast.BinOp):
            operand_classes = expression_classes(node.left, owner) | expression_classes(
                node.right, owner
            )
            provenance |= expression_provenance(node.left, owner)
            provenance |= expression_provenance(node.right, owner)
            provenance |= protocol_returned_provenance(
                operand_classes, binary_protocol_methods
            )
        elif isinstance(node, ast.Compare):
            values = (node.left, *node.comparators)
            operand_classes = {
                candidate
                for value in values
                for candidate in expression_classes(value, owner)
            }
            for value in values:
                provenance |= expression_provenance(value, owner)
            provenance |= protocol_returned_provenance(
                operand_classes, comparison_protocol_methods
            )
        elif isinstance(node, ast.FormattedValue):
            receiver_classes = expression_classes(node.value, owner)
            provenance |= expression_provenance(node.value, owner)
            if node.format_spec is not None:
                provenance |= expression_provenance(node.format_spec, owner)
            provenance |= protocol_returned_provenance(receiver_classes, {"__format__"})
        elif isinstance(
            node,
            (
                ast.BoolOp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.JoinedStr,
                ast.ListComp,
                ast.SetComp,
                ast.Slice,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    provenance |= expression_provenance(child, owner)
            if isinstance(
                node, (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp)
            ):
                for generator in node.generators:
                    provenance |= expression_provenance(generator.iter, owner)
                    for condition in generator.ifs:
                        provenance |= expression_provenance(condition, owner)
        else:
            child_provenance = 0
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    child_provenance |= expression_provenance(child, owner)
            if child_provenance:
                raise InventoryError(
                    "Python Windows BLAS unclassified expression carries mutation "
                    f"provenance: {type(node).__name__}"
                )
        return provenance

    def assign_provenance(
        target: ast.AST, value: int, environment: dict[str, int]
    ) -> bool:
        changed_binding = False
        if isinstance(target, (ast.Name, ast.arg)):
            name = target.id if isinstance(target, ast.Name) else target.arg
            owner = provenance_environment_owners[id(environment)]
            destination = assignment_destination(name, owner)
            destination_environment = provenance_by_scope[destination]
            combined = destination_environment.get(name, 0) | value
            if combined != destination_environment.get(name, 0):
                destination_environment[name] = combined
                changed_binding = True
            if destination is not owner and value:
                raise InventoryError(
                    "Python Windows BLAS rooted provenance escaped through a "
                    "global or nonlocal binding"
                )
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                changed_binding |= assign_provenance(element, value, environment)
        elif isinstance(target, ast.Starred):
            changed_binding |= assign_provenance(target.value, value, environment)
        elif isinstance(target, ast.Attribute):
            changed_binding |= assign_provenance(target.value, value, environment)
        elif isinstance(target, ast.Subscript):
            changed_binding |= assign_provenance(target.value, value, environment)
        return changed_binding

    def assign_identity(
        target: ast.AST,
        candidates: set[Any],
        bindings: dict[ScopeOwner, dict[str, set[Any]]],
        owner: ScopeOwner,
    ) -> bool:
        changed_binding = False
        if isinstance(target, (ast.Name, ast.arg)):
            name = target.id if isinstance(target, ast.Name) else target.arg
            destination = assignment_destination(name, owner)
            bound = bindings[destination].setdefault(name, set())
            prior = len(bound)
            bound.update(candidates)
            changed_binding = len(bound) != prior
            if destination is not owner and candidates:
                raise InventoryError(
                    "Python Windows BLAS identity escaped through a global or "
                    "nonlocal binding"
                )
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                changed_binding |= assign_identity(element, candidates, bindings, owner)
        elif isinstance(target, ast.Starred):
            changed_binding |= assign_identity(
                target.value, candidates, bindings, owner
            )
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            callable_escape = bool(candidates) and bindings is callable_bindings
            dangerous_class_escape = (
                bindings is class_symbol_bindings or bindings is instance_bindings
            ) and any(
                return_provenance.get(method, 0)
                or returned_callables.get(method, set())
                or returned_class_symbols.get(method, set())
                or returned_instances.get(method, set())
                for candidate in candidates
                if isinstance(candidate, ast.ClassDef)
                for methods in callable_bindings[candidate].values()
                for method in methods
            )
            if callable_escape or dangerous_class_escape:
                raise InventoryError(
                    "Python Windows BLAS callable or class identity escaped through an "
                    "indirect assignment target: "
                    f"{reviewed.inventory_path}:{getattr(target, 'lineno', 0)}"
                )
        return changed_binding

    def assign_class_identities(
        target: ast.AST,
        expression: ast.AST,
        source_owner: ScopeOwner,
        target_owner: ScopeOwner,
    ) -> bool:
        return assign_identity(
            target,
            expression_class_symbols(expression, source_owner),
            class_symbol_bindings,
            target_owner,
        ) | assign_identity(
            target,
            expression_classes(expression, source_owner),
            instance_bindings,
            target_owner,
        )

    def function_parameters(
        function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ) -> tuple[ast.arg, ...]:
        return (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )

    def pattern_bound_names(pattern: ast.pattern) -> tuple[str, ...]:
        names: list[str] = []
        for node in ast.walk(pattern):
            if isinstance(node, ast.MatchAs) and node.name is not None:
                names.append(node.name)
            elif isinstance(node, ast.MatchStar) and node.name is not None:
                names.append(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest is not None:
                names.append(node.rest)
        return tuple(names)

    provenance_changed = True
    while provenance_changed:
        provenance_changed = False
        for function in callable_nodes:
            environment = provenance_by_scope[function]
            definition_owner = containing_scope(function)
            positional = (*function.args.posonlyargs, *function.args.args)
            default_offset = len(positional) - len(function.args.defaults)
            for index, default in enumerate(
                function.args.defaults, start=default_offset
            ):
                provenance_changed |= assign_provenance(
                    positional[index],
                    expression_provenance(default, definition_owner),
                    environment,
                )
                provenance_changed |= assign_identity(
                    positional[index],
                    expression_callables(default, definition_owner),
                    callable_bindings,
                    function,
                )
                provenance_changed |= assign_class_identities(
                    positional[index], default, definition_owner, function
                )
            for parameter, default in zip(
                function.args.kwonlyargs,
                function.args.kw_defaults,
                strict=True,
            ):
                if default is None:
                    continue
                provenance_changed |= assign_provenance(
                    parameter,
                    expression_provenance(default, definition_owner),
                    environment,
                )
                provenance_changed |= assign_identity(
                    parameter,
                    expression_callables(default, definition_owner),
                    callable_bindings,
                    function,
                )
                provenance_changed |= assign_class_identities(
                    parameter, default, definition_owner, function
                )
        for owner in scope_owners:
            environment = provenance_by_scope[owner]
            for node in nodes_by_scope[owner]:
                if isinstance(node, ast.Assign):
                    value = expression_provenance(node.value, owner)
                    for target in node.targets:
                        provenance_changed |= assign_provenance(
                            target, value, environment
                        )
                        provenance_changed |= assign_identity(
                            target,
                            expression_callables(node.value, owner),
                            callable_bindings,
                            owner,
                        )
                        provenance_changed |= assign_class_identities(
                            target, node.value, owner, owner
                        )
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    provenance_changed |= assign_provenance(
                        node.target,
                        expression_provenance(node.value, owner),
                        environment,
                    )
                    provenance_changed |= assign_identity(
                        node.target,
                        expression_callables(node.value, owner),
                        callable_bindings,
                        owner,
                    )
                    provenance_changed |= assign_class_identities(
                        node.target, node.value, owner, owner
                    )
                elif isinstance(node, ast.AugAssign):
                    combined_provenance = expression_provenance(
                        node.target, owner
                    ) | expression_provenance(node.value, owner)
                    provenance_changed |= assign_provenance(
                        node.target, combined_provenance, environment
                    )
                    provenance_changed |= assign_identity(
                        node.target,
                        expression_callables(node.target, owner)
                        | expression_callables(node.value, owner),
                        callable_bindings,
                        owner,
                    )
                    for expression in (node.target, node.value):
                        provenance_changed |= assign_class_identities(
                            node.target, expression, owner, owner
                        )
                elif isinstance(node, ast.NamedExpr):
                    provenance_changed |= assign_provenance(
                        node.target,
                        expression_provenance(node.value, owner),
                        environment,
                    )
                    provenance_changed |= assign_identity(
                        node.target,
                        expression_callables(node.value, owner),
                        callable_bindings,
                        owner,
                    )
                    provenance_changed |= assign_class_identities(
                        node.target, node.value, owner, owner
                    )
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    iteration_classes = expression_classes(node.iter, owner)
                    first_methods = (
                        {"__aiter__"}
                        if isinstance(node, ast.AsyncFor)
                        else {"__iter__"}
                    )
                    second_methods = (
                        {"__anext__"}
                        if isinstance(node, ast.AsyncFor)
                        else {"__next__"}
                    )
                    (
                        iteration_provenance,
                        iteration_callables,
                        iteration_instances,
                        iteration_symbols,
                    ) = protocol_chain_identities(
                        iteration_classes, first_methods, second_methods
                    )
                    provenance_changed |= assign_provenance(
                        node.target,
                        expression_provenance(node.iter, owner) | iteration_provenance,
                        environment,
                    )
                    provenance_changed |= assign_identity(
                        node.target,
                        expression_callables(node.iter, owner) | iteration_callables,
                        callable_bindings,
                        owner,
                    )
                    provenance_changed |= assign_identity(
                        node.target,
                        iteration_instances | iteration_classes,
                        instance_bindings,
                        owner,
                    )
                    provenance_changed |= assign_identity(
                        node.target,
                        iteration_symbols | expression_class_symbols(node.iter, owner),
                        class_symbol_bindings,
                        owner,
                    )
                elif isinstance(node, ast.comprehension):
                    iteration_classes = expression_classes(node.iter, owner)
                    first_methods = {"__aiter__"} if node.is_async else {"__iter__"}
                    second_methods = {"__anext__"} if node.is_async else {"__next__"}
                    (
                        iteration_provenance,
                        iteration_callables,
                        iteration_instances,
                        iteration_symbols,
                    ) = protocol_chain_identities(
                        iteration_classes, first_methods, second_methods
                    )
                    provenance_changed |= assign_provenance(
                        node.target,
                        expression_provenance(node.iter, owner) | iteration_provenance,
                        environment,
                    )
                    provenance_changed |= assign_identity(
                        node.target,
                        expression_callables(node.iter, owner) | iteration_callables,
                        callable_bindings,
                        owner,
                    )
                    provenance_changed |= assign_identity(
                        node.target,
                        iteration_instances | iteration_classes,
                        instance_bindings,
                        owner,
                    )
                    provenance_changed |= assign_identity(
                        node.target,
                        iteration_symbols | expression_class_symbols(node.iter, owner),
                        class_symbol_bindings,
                        owner,
                    )
                elif isinstance(node, (ast.With, ast.AsyncWith)):
                    for item in node.items:
                        if item.optional_vars is not None:
                            context_classes = expression_classes(
                                item.context_expr, owner
                            )
                            provenance_changed |= assign_provenance(
                                item.optional_vars,
                                expression_provenance(item.context_expr, owner)
                                | protocol_returned_provenance(
                                    context_classes,
                                    {"__aenter__", "__enter__"},
                                ),
                                environment,
                            )
                            provenance_changed |= assign_identity(
                                item.optional_vars,
                                expression_callables(item.context_expr, owner)
                                | protocol_returned_callables(
                                    context_classes, {"__aenter__", "__enter__"}
                                ),
                                callable_bindings,
                                owner,
                            )
                            provenance_changed |= assign_identity(
                                item.optional_vars,
                                context_classes
                                | protocol_returned_instances(
                                    context_classes, {"__aenter__", "__enter__"}
                                ),
                                instance_bindings,
                                owner,
                            )
                            provenance_changed |= assign_identity(
                                item.optional_vars,
                                expression_class_symbols(item.context_expr, owner)
                                | protocol_returned_class_symbols(
                                    context_classes, {"__aenter__", "__enter__"}
                                ),
                                class_symbol_bindings,
                                owner,
                            )
                elif isinstance(node, ast.Match):
                    subject_provenance = expression_provenance(node.subject, owner)
                    subject_callables = expression_callables(node.subject, owner)
                    subject_classes = expression_classes(node.subject, owner)
                    subject_class_symbols = expression_class_symbols(
                        node.subject, owner
                    )
                    for case in node.cases:
                        for name in pattern_bound_names(case.pattern):
                            target = ast.Name(id=name)
                            provenance_changed |= assign_provenance(
                                target, subject_provenance, environment
                            )
                            provenance_changed |= assign_identity(
                                target,
                                subject_callables,
                                callable_bindings,
                                owner,
                            )
                            provenance_changed |= assign_identity(
                                target,
                                subject_classes,
                                instance_bindings,
                                owner,
                            )
                            provenance_changed |= assign_identity(
                                target,
                                subject_class_symbols,
                                class_symbol_bindings,
                                owner,
                            )
                elif (
                    isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom))
                    and node.value is not None
                    and owner in return_provenance
                ):
                    combined = return_provenance[owner] | expression_provenance(
                        node.value, owner
                    )
                    if combined != return_provenance[owner]:
                        return_provenance[owner] = combined
                        provenance_changed = True
                    callable_result = returned_callables[owner] | expression_callables(
                        node.value, owner
                    )
                    if callable_result != returned_callables[owner]:
                        returned_callables[owner] = callable_result
                        provenance_changed = True
                    instance_result = returned_instances[owner] | expression_classes(
                        node.value, owner
                    )
                    if instance_result != returned_instances[owner]:
                        returned_instances[owner] = instance_result
                        provenance_changed = True
                    symbol_result = returned_class_symbols[
                        owner
                    ] | expression_class_symbols(node.value, owner)
                    if symbol_result != returned_class_symbols[owner]:
                        returned_class_symbols[owner] = symbol_result
                        provenance_changed = True
                elif isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr in container_mutator_methods
                    ):
                        receiver = node.func.value
                        arguments = (
                            *node.args,
                            *(keyword.value for keyword in node.keywords),
                        )
                        argument_provenance = 0
                        argument_callables: set[ast.AST] = set()
                        argument_instances: set[ast.ClassDef] = set()
                        argument_symbols: set[ast.ClassDef] = set()
                        for argument in arguments:
                            argument_provenance |= expression_provenance(
                                argument, owner
                            )
                            argument_callables.update(
                                expression_callables(argument, owner)
                            )
                            argument_instances.update(
                                expression_classes(argument, owner)
                            )
                            argument_symbols.update(
                                expression_class_symbols(argument, owner)
                            )
                        if (
                            argument_provenance
                            or argument_callables
                            or argument_instances
                            or argument_symbols
                        ) and not isinstance(
                            receiver, (ast.Name, ast.Attribute, ast.Subscript)
                        ):
                            raise InventoryError(
                                "Python Windows BLAS dangerous container mutation has "
                                "no provable persistent receiver"
                            )
                        provenance_changed |= assign_provenance(
                            receiver, argument_provenance, environment
                        )
                        provenance_changed |= assign_identity(
                            receiver,
                            argument_callables,
                            callable_bindings,
                            owner,
                        )
                        for argument in arguments:
                            provenance_changed |= assign_class_identities(
                                receiver, argument, owner, owner
                            )
                    called_functions = resolve_user_functions(node, owner)
                    if not called_functions:
                        continue
                    for called_function in called_functions:
                        parameters = function_parameters(called_function)
                        implicit_receiver = bool(
                            parameters
                            and call_has_implicit_receiver(node, called_function, owner)
                        )
                        parameter_offset = 1 if implicit_receiver else 0
                        called_environment = provenance_by_scope[called_function]

                        def propagate_argument(
                            parameter: ast.arg, argument: ast.AST
                        ) -> bool:
                            return (
                                assign_provenance(
                                    parameter,
                                    expression_provenance(argument, owner),
                                    called_environment,
                                )
                                | assign_identity(
                                    parameter,
                                    expression_callables(argument, owner),
                                    callable_bindings,
                                    called_function,
                                )
                                | assign_class_identities(
                                    parameter, argument, owner, called_function
                                )
                            )

                        if implicit_receiver and isinstance(node.func, ast.Attribute):
                            provenance_changed |= propagate_argument(
                                parameters[0], node.func.value
                            )
                        positional_parameters = (
                            *called_function.args.posonlyargs,
                            *called_function.args.args,
                        )[parameter_offset:]
                        vararg = called_function.args.vararg
                        positional_index = 0
                        expanded_positional = False
                        for argument in node.args:
                            argument_value = (
                                argument.value
                                if isinstance(argument, ast.Starred)
                                else argument
                            )
                            if isinstance(argument, ast.Starred):
                                expanded_positional = True
                                for parameter in positional_parameters[
                                    positional_index:
                                ]:
                                    provenance_changed |= propagate_argument(
                                        parameter, argument_value
                                    )
                                if vararg is not None:
                                    provenance_changed |= propagate_argument(
                                        vararg, argument_value
                                    )
                                continue
                            if expanded_positional:
                                for parameter in positional_parameters[
                                    positional_index:
                                ]:
                                    provenance_changed |= propagate_argument(
                                        parameter, argument_value
                                    )
                                if vararg is not None:
                                    provenance_changed |= propagate_argument(
                                        vararg, argument_value
                                    )
                            elif positional_index < len(positional_parameters):
                                provenance_changed |= propagate_argument(
                                    positional_parameters[positional_index],
                                    argument_value,
                                )
                                positional_index += 1
                            elif vararg is not None:
                                provenance_changed |= propagate_argument(
                                    vararg, argument_value
                                )
                        parameters_by_name = {
                            parameter.arg: parameter
                            for parameter in (
                                *called_function.args.args[
                                    1 if implicit_receiver else 0 :
                                ],
                                *called_function.args.kwonlyargs,
                            )
                        }
                        for keyword in node.keywords:
                            if keyword.arg is None:
                                for parameter in parameters_by_name.values():
                                    provenance_changed |= propagate_argument(
                                        parameter, keyword.value
                                    )
                                if called_function.args.kwarg is not None:
                                    provenance_changed |= propagate_argument(
                                        called_function.args.kwarg, keyword.value
                                    )
                                continue
                            parameter = parameters_by_name.get(keyword.arg)
                            if parameter is not None:
                                provenance_changed |= propagate_argument(
                                    parameter, keyword.value
                                )
                            elif called_function.args.kwarg is not None:
                                provenance_changed |= propagate_argument(
                                    called_function.args.kwarg, keyword.value
                                )

        for lambda_node in lambda_nodes:
            combined = return_provenance[lambda_node] | expression_provenance(
                lambda_node.body, lambda_node
            )
            if combined != return_provenance[lambda_node]:
                return_provenance[lambda_node] = combined
                provenance_changed = True
            callable_result = returned_callables[lambda_node] | expression_callables(
                lambda_node.body, lambda_node
            )
            if callable_result != returned_callables[lambda_node]:
                returned_callables[lambda_node] = callable_result
                provenance_changed = True
            instance_result = returned_instances[lambda_node] | expression_classes(
                lambda_node.body, lambda_node
            )
            if instance_result != returned_instances[lambda_node]:
                returned_instances[lambda_node] = instance_result
                provenance_changed = True
            symbol_result = returned_class_symbols[
                lambda_node
            ] | expression_class_symbols(lambda_node.body, lambda_node)
            if symbol_result != returned_class_symbols[lambda_node]:
                returned_class_symbols[lambda_node] = symbol_result
                provenance_changed = True

    for function in function_nodes:
        if not has_unknown_decorator(function):
            continue
        parameters = (*function.args.posonlyargs, *function.args.args)
        receiver_provenance = (
            binding_provenance(parameters[0].arg, function) if parameters else 0
        )
        if (
            return_provenance[function]
            or returned_callables[function]
            or returned_instances[function]
            or returned_class_symbols[function]
            or receiver_provenance
        ):
            raise InventoryError(
                "Python Windows BLAS unknown decorator obscures a dangerous method "
                "binding"
            )

    known_pattern_types = {
        ast.MatchAs,
        ast.MatchClass,
        ast.MatchMapping,
        ast.MatchOr,
        ast.MatchSequence,
        ast.MatchSingleton,
        ast.MatchStar,
        ast.MatchValue,
    }

    def safe_reflection_call(
        call: ast.Call,
        owner: ScopeOwner,
    ) -> bool:
        identifier = call_identifier(call.func)
        if identifier in {
            "builtins.getattr",
            "getattr",
            "object.__getattribute__",
        }:
            return (
                len(call.args) >= 2
                and literal_text(call.args[1]) is not None
                and literal_text(call.args[1]) not in runtime_authority_literal_names
                and not reflection_receiver_has_runtime_authority(call.args[0], owner)
                and not (
                    expression_provenance(call.args[0], owner)
                    & (protected_target | meta_execution_capability)
                )
            )
        return (
            identifier in {"builtins.vars", "vars"}
            and len(call.args) == 1
            and not reflection_receiver_has_runtime_authority(call.args[0], owner)
            and not (
                expression_provenance(call.args[0], owner)
                & (protected_target | meta_execution_capability)
            )
        )

    for owner in scope_owners:
        for node in nodes_by_scope[owner]:
            if (
                isinstance(node, ast.Subscript)
                and literal_text(node.slice) is None
                and expression_provenance(node.value, owner) & meta_execution_capability
                and not (
                    expression_path(node.value) == "sys.modules"
                    and owner is not None
                    and getattr(owner, "name", None) == "load_tool"
                    and isinstance(node.ctx, ast.Store)
                )
            ):
                raise InventoryError(
                    "Python Windows BLAS dynamic runtime namespace lookup is forbidden: "
                    f"{reviewed.inventory_path}:{node.lineno}"
                )
            escaped_expression: ast.AST | None = None
            if isinstance(node, ast.Assign):
                escaped_expression = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                escaped_expression = node.value
            elif isinstance(node, ast.AugAssign):
                if (
                    expression_provenance(node.target, owner)
                    | expression_provenance(node.value, owner)
                ) & meta_execution_capability:
                    raise InventoryError(
                        "Python Windows BLAS meta-execution capability escaped "
                        "through augmented assignment"
                    )
                escaped_expression = node.value
            elif isinstance(node, ast.TypeAlias):
                escaped_expression = node.value
            elif isinstance(node, ast.NamedExpr):
                escaped_expression = node.value
            elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
                escaped_expression = node.value
            if (
                escaped_expression is not None
                and expression_provenance(escaped_expression, owner)
                & meta_execution_capability
            ):
                raise InventoryError(
                    "Python Windows BLAS meta-execution capability escaped static review"
                )
    for function in callable_nodes:
        definition_owner = containing_scope(function)
        for default in (
            *function.args.defaults,
            *(default for default in function.args.kw_defaults if default is not None),
        ):
            if (
                expression_provenance(default, definition_owner)
                & meta_execution_capability
            ):
                raise InventoryError(
                    "Python Windows BLAS meta-execution capability escaped in a default"
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and expression_provenance(
            node.annotation, containing_scope(node)
        ):
            raise InventoryError(
                "Python Windows BLAS executable annotation carries mutation provenance"
            )
        if isinstance(node, ast.TypeAlias):
            type_alias_expressions = [node.value]
            for type_parameter in node.type_params:
                type_alias_expressions.extend(
                    child
                    for child in ast.walk(type_parameter)
                    if isinstance(child, ast.expr)
                )
            if any(
                expression_provenance(expression, containing_scope(node))
                for expression in type_alias_expressions
            ):
                raise InventoryError(
                    "Python Windows BLAS executable type alias carries mutation "
                    "provenance"
                )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definition_owner = containing_scope(node)
            for decorator in node.decorator_list:
                if expression_provenance(decorator, definition_owner):
                    raise InventoryError(
                        "Python Windows BLAS decorator carries mutation provenance"
                    )
            definition_expressions: list[ast.expr] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definition_expressions.extend(
                    parameter.annotation
                    for parameter in (
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    )
                    if parameter.annotation is not None
                )
                if (
                    node.args.vararg is not None
                    and node.args.vararg.annotation is not None
                ):
                    definition_expressions.append(node.args.vararg.annotation)
                if (
                    node.args.kwarg is not None
                    and node.args.kwarg.annotation is not None
                ):
                    definition_expressions.append(node.args.kwarg.annotation)
                if node.returns is not None:
                    definition_expressions.append(node.returns)
            else:
                definition_expressions.extend(node.bases)
                definition_expressions.extend(
                    keyword.value for keyword in node.keywords
                )
            for type_parameter in getattr(node, "type_params", ()):
                definition_expressions.extend(
                    child
                    for child in ast.walk(type_parameter)
                    if isinstance(child, ast.expr)
                )
            if any(
                expression_provenance(expression, definition_owner)
                for expression in definition_expressions
            ):
                raise InventoryError(
                    "Python Windows BLAS executable definition surface carries "
                    "mutation provenance"
                )
        if isinstance(node, ast.pattern) and type(node) not in known_pattern_types:
            match_node: ast.AST | None = parents.get(node)
            while match_node is not None and not isinstance(match_node, ast.Match):
                match_node = parents.get(match_node)
            if isinstance(match_node, ast.Match) and expression_provenance(
                match_node.subject, containing_scope(match_node)
            ):
                raise InventoryError(
                    "Python Windows BLAS unclassified pattern carries mutation provenance"
                )
        if isinstance(node, ast.MatchClass):
            match_authority_attributes = {
                *runtime_authority_attributes,
                "__dict__",
                *protected_globals,
                *loader_attributes,
            }
            if any(
                attribute in match_authority_attributes for attribute in node.kwd_attrs
            ):
                raise InventoryError(
                    "Python Windows BLAS match pattern reads runtime authority"
                )
            if node.patterns:
                raise InventoryError(
                    "Python Windows BLAS positional class pattern reads runtime "
                    "authority"
                )

    for owner in scope_owners:
        for node in nodes_by_scope[owner]:
            if not isinstance(node, ast.Call):
                continue
            called_functions = resolve_user_functions(node, owner)
            if called_functions:
                continue
            argument_provenance = 0
            for argument in node.args:
                argument_provenance |= expression_provenance(argument, owner)
            for keyword in node.keywords:
                argument_provenance |= expression_provenance(keyword.value, owner)
            call_name = call_identifier(node.func)
            callee_provenance = expression_provenance(node.func, owner)
            if call_name in trusted_provenance_receivers:
                continue
            if safe_reflection_call(node, owner):
                continue
            if argument_provenance & meta_execution_capability or (
                callee_provenance & meta_execution_capability
            ):
                raise InventoryError(
                    "Python Windows BLAS meta-execution capability was invoked or escaped"
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {
                    "append",
                    "extend",
                }
                and not (callee_provenance & mutation_capability)
            ):
                continue
            if argument_provenance & (mutation_capability | protected_target) or (
                callee_provenance & (mutation_capability | protected_target)
                and not (
                    isinstance(node.func, ast.Attribute)
                    and mutation_capability_expression(node.func)
                )
            ):
                raise InventoryError(
                    "Python Windows BLAS mutation capability escaped static review: "
                    f"{reviewed.inventory_path}:{node.lineno}"
                )

    allowed_nodes = {id(statement) for statement in allowed_assignments.values()}
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Global, ast.Nonlocal)
        ) and canonical_globals.intersection(node.names):
            raise InventoryError(
                "Python Windows BLAS protected global mutation is forbidden"
            )
        mutation_targets: tuple[ast.AST, ...] = ()
        if isinstance(node, ast.Assign):
            mutation_targets = tuple(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            mutation_targets = (node.target,)
        elif isinstance(node, ast.Delete):
            mutation_targets = tuple(node.targets)
        for target in mutation_targets:
            dynamic_owner = (
                target.value
                if isinstance(target, (ast.Attribute, ast.Subscript))
                else None
            )
            node_owner = containing_scope(node)
            if id(node) not in allowed_nodes and (
                protected_path(expression_path(target))
                or dynamic_owner is not None
                and expression_provenance(dynamic_owner, node_owner) & protected_target
            ):
                raise InventoryError(
                    "Python Windows BLAS loader or protected global mutation is forbidden"
                )
        if not isinstance(node, ast.Call):
            continue
        call_name = call_identifier(node.func)
        if (
            call_name in {"setattr", "delattr", "object.__setattr__"}
            and len(node.args) >= 2
        ):
            owner = expression_path(node.args[0])
            attribute = literal_text(node.args[1])
            if (
                owner is not None
                and attribute is not None
                and protected_path(f"{owner}.{attribute}")
            ):
                raise InventoryError(
                    "Python Windows BLAS loader or protected global mutation is forbidden"
                )
        if call_name in {"mock.patch", "unittest.mock.patch"} and node.args:
            target = node.args[0]
            if protected_path(literal_text(target)):
                raise InventoryError("Python Windows BLAS loader mutation is forbidden")
        if (
            call_name in {"mock.patch.object", "unittest.mock.patch.object"}
            and len(node.args) >= 2
        ):
            owner = expression_path(node.args[0])
            attribute = literal_text(node.args[1])
            if (
                owner is not None
                and attribute is not None
                and protected_path(f"{owner}.{attribute}")
            ):
                raise InventoryError("Python Windows BLAS loader mutation is forbidden")
        if (
            call_name in {"mock.patch.dict", "unittest.mock.patch.dict"}
            and len(node.args) >= 2
        ):
            owner = expression_path(node.args[0]) or literal_text(node.args[0])
            if owner in {"test_level2_report.__dict__", "ctypes.__dict__"}:
                clear_requested = any(
                    keyword.arg == "clear"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                )
                dictionary = node.args[1]
                keys = (
                    [
                        key
                        for item in dictionary.keys
                        if item is not None and (key := literal_text(item)) is not None
                    ]
                    if isinstance(dictionary, ast.Dict)
                    else []
                )
                if clear_requested or any(
                    protected_path(f"{owner[:-9]}.{key}") for key in keys
                ):
                    raise InventoryError(
                        "Python Windows BLAS loader or protected global mutation is forbidden"
                    )
        if isinstance(node.func, ast.Attribute):
            owner = expression_path(node.func.value)
            method = node.func.attr
            if owner in {"test_level2_report.__dict__", "ctypes.__dict__"}:
                if method == "clear":
                    raise InventoryError(
                        "Python Windows BLAS loader or protected global mutation is forbidden"
                    )
                candidate_keys: list[str] = []
                if node.args and isinstance(node.args[0], ast.Dict):
                    candidate_keys.extend(
                        key
                        for item in node.args[0].keys
                        if item is not None and (key := literal_text(item)) is not None
                    )
                elif node.args:
                    key = literal_text(node.args[0])
                    if key is not None:
                        candidate_keys.append(key)
                candidate_keys.extend(
                    keyword.arg for keyword in node.keywords if keyword.arg
                )
                if method in {
                    "update",
                    "setdefault",
                    "pop",
                    "__setitem__",
                    "__delitem__",
                } and any(
                    protected_path(f"{owner[:-9]}.{key}") for key in candidate_keys
                ):
                    raise InventoryError(
                        "Python Windows BLAS loader or protected global mutation is forbidden"
                    )


def _python_tooling_source_skip_review(
    root: Path,
    module_paths: list[str],
    discovery_start: str,
    discovery_pattern: str,
    *,
    _closure: _PythonExecutionClosure | None = None,
) -> tuple[
    frozenset[_PythonSkipSourceFact],
    frozenset[_PythonSkipSourceFact],
    tuple[_PythonDynamicSkipSite, ...],
    tuple[_PythonReviewedSourceModule, ...],
]:
    decorator_skips: set[_PythonSkipSourceFact] = set()
    dynamic_skips: set[_PythonSkipSourceFact] = set()
    dynamic_sites: set[_PythonDynamicSkipSite] = set()
    if _closure is None:
        reviewed_modules = _reviewed_python_tooling_source_modules(
            root, module_paths, discovery_start, discovery_pattern
        )
    else:
        if tuple(module_paths) != tuple(
            path for path, _ in _PYTHON_TOOLING_REVIEWED_SOURCE_SHA256
        ):
            raise InventoryError("Python tooling root source order is noncanonical")
        reviewed_modules = tuple(
            _PythonReviewedSourceModule(
                path,
                PurePosixPath(path).stem,
                _closure.sources[path].source_path,
                _closure.sources[path].source_bytes,
                _closure.sources[path].source_sha256,
            )
            for path in module_paths
        )
        for path, _ in _PYTHON_TOOLING_EXECUTION_SOURCE_SHA256:
            _python_tooling_subprocess_source_audit(_closure.sources[path])
    reviewed_names: set[str] = set()
    for reviewed in reviewed_modules:
        path = reviewed.inventory_path
        if reviewed.module_name in reviewed_names:
            raise InventoryError("Python tooling reviewed module names are duplicated")
        reviewed_names.add(reviewed.module_name)
        try:
            tree = ast.parse(reviewed.source_bytes.decode("utf-8"), filename=path)
        except (SyntaxError, UnicodeError) as exc:
            raise InventoryError(
                f"Python tooling skip contract cannot parse {path}: {exc}"
            ) from exc
        _python_windows_blas_source_audit(tree, reviewed)
        module_bindings: dict[str, str] = {}
        allowed_attributes: set[int] = set()
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                _update_python_static_bindings(node, module_bindings)
                continue
            has_test_methods = any(
                isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name.startswith("test_")
                for member in node.body
            )
            class_facts = (
                _reviewed_test_decorators(node.decorator_list, path, module_bindings)
                if has_test_methods
                else set()
            )
            class_bindings = dict(module_bindings)
            for member in node.body:
                if not (
                    isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and member.name.startswith("test_")
                ):
                    _update_python_static_bindings(member, class_bindings)
                    continue
                runtime_id = f"{PurePosixPath(path).stem}.{node.name}.{member.name}"
                method_facts = _reviewed_test_decorators(
                    member.decorator_list, path, class_bindings
                )
                decorator_skips.update(
                    _PythonSkipSourceFact(runtime_id, reason, kind, digest)
                    for kind, reason, digest in class_facts | method_facts
                )
                method_facts, method_attributes, method_sites = (
                    _reviewed_conditional_skips(
                        member,
                        runtime_id,
                        reviewed.source_path,
                        reviewed.source_sha256,
                    )
                )
                dynamic_skips.update(
                    _PythonSkipSourceFact(runtime_id, reason, kind, digest)
                    for kind, reason, digest in method_facts
                )
                allowed_attributes.update(method_attributes)
                dynamic_sites.update(method_sites)
                _update_python_static_bindings(member, class_bindings)
            _update_python_static_bindings(node, module_bindings)
        _python_skip_capability_audit(tree, allowed_attributes)
    return (
        frozenset(decorator_skips),
        frozenset(dynamic_skips),
        tuple(sorted(dynamic_sites)),
        reviewed_modules,
    )


def _python_tooling_source_skip_contract(
    root: Path,
    module_paths: list[str],
    discovery_start: str,
    discovery_pattern: str,
) -> tuple[frozenset[_PythonSkipSourceFact], frozenset[_PythonSkipSourceFact]]:
    decorator_skips, dynamic_skips, _, _ = _python_tooling_source_skip_review(
        root, module_paths, discovery_start, discovery_pattern
    )
    return decorator_skips, dynamic_skips


def _verify_python_reviewed_source_module(
    reviewed: _PythonReviewedSourceModule,
) -> None:
    if (
        type(reviewed) is not _PythonReviewedSourceModule
        or type(reviewed.inventory_path) is not str
        or type(reviewed.module_name) is not str
        or not isinstance(reviewed.source_path, Path)
        or type(reviewed.source_bytes) is not bytes
        or len(reviewed.source_bytes) > MAX_INVENTORY_BYTES
        or type(reviewed.source_sha256) is not str
        or hashlib.sha256(reviewed.source_bytes).hexdigest() != reviewed.source_sha256
    ):
        raise InventoryError("Python tooling reviewed source module changed")


def _verify_python_source_module_binding(
    binding: _PythonSourceModuleBinding,
) -> None:
    if type(binding) is not _PythonSourceModuleBinding:
        raise InventoryError("Python tooling source module binding changed")
    reviewed = binding.reviewed
    if type(reviewed) is not _PythonReviewedSourceModule:
        raise InventoryError("Python tooling source module binding changed")
    module = binding.module
    namespace = binding.namespace
    spec = binding.spec
    loader = binding.loader
    if (
        type(binding.name) is not str
        or type(binding.file) is not str
        or type(reviewed.inventory_path) is not str
        or type(reviewed.module_name) is not str
        or not isinstance(reviewed.source_path, Path)
        or type(reviewed.source_bytes) is not bytes
        or len(reviewed.source_bytes) > MAX_INVENTORY_BYTES
        or type(reviewed.source_sha256) is not str
        or type(module) is not types.ModuleType
        or type(namespace) is not dict
        or sys.modules.get(binding.name) is not module
        or vars(module) is not namespace
        or module.__spec__ is not spec
        or module.__loader__ is not loader
        or module.__name__ != binding.name
        or module.__file__ != binding.file
        or binding.name != reviewed.module_name
        or binding.file != str(reviewed.source_path)
        or spec is None
        or spec.loader is not loader
        or spec.name != binding.name
        or spec.origin != binding.file
        or namespace.get("__name__") != binding.name
        or namespace.get("__file__") != binding.file
        or namespace.get("__spec__") is not spec
        or namespace.get("__loader__") is not loader
    ):
        raise InventoryError("Python tooling source module binding changed")


def _verify_python_source_module_registry(
    registry: tuple[_PythonSourceModuleBinding, ...],
    reviewed_modules: tuple[_PythonReviewedSourceModule, ...] | None = None,
) -> None:
    if type(registry) is not tuple:
        raise InventoryError("Python tooling source module registry changed")
    for binding in registry:
        _verify_python_source_module_binding(binding)
    if reviewed_modules is not None:
        if (
            type(reviewed_modules) is not tuple
            or len(registry) != len(reviewed_modules)
            or any(
                binding.reviewed is not reviewed
                for binding, reviewed in zip(registry, reviewed_modules)
            )
        ):
            raise InventoryError(
                "Python tooling source module reviewed input identity changed"
            )
    names = [binding.name for binding in registry]
    paths = [binding.reviewed.inventory_path for binding in registry]
    if len(names) != len(set(names)) or len(paths) != len(set(paths)):
        raise InventoryError("Python tooling source module registry is duplicated")


@contextlib.contextmanager
def _registered_python_tooling_modules(
    reviewed_modules: tuple[_PythonReviewedSourceModule, ...],
) -> Any:
    if type(reviewed_modules) is not tuple:
        raise InventoryError("Python tooling reviewed source registry changed")
    bindings: list[_PythonSourceModuleBinding] = []
    for reviewed in reviewed_modules:
        _verify_python_reviewed_source_module(reviewed)
        spec = importlib.util.spec_from_file_location(
            reviewed.module_name, reviewed.source_path
        )
        if spec is None or spec.loader is None:
            raise InventoryError(
                f"cannot create Python tooling module spec: {reviewed.inventory_path}"
            )
        module = importlib.util.module_from_spec(spec)
        bindings.append(
            _PythonSourceModuleBinding(
                reviewed,
                module,
                vars(module),
                spec,
                spec.loader,
                reviewed.module_name,
                str(reviewed.source_path),
            )
        )
    registry = tuple(bindings)
    replacements = {binding.name: binding.module for binding in registry}
    frozen_registry = bool(registry) and all(
        isinstance(binding.loader, _PythonFrozenLoader) for binding in registry
    )
    if frozen_registry:
        if set(replacements).intersection(sys.modules):
            raise InventoryError("Python tooling frozen root module is already loaded")
        sys.modules.update(replacements)
    modules_context = (
        contextlib.nullcontext()
        if frozen_registry
        else mock.patch.dict("sys.modules", replacements, clear=False)
    )
    with modules_context:
        registry = tuple(bindings)
        _verify_python_source_module_registry(registry, reviewed_modules)
        for binding in registry:
            if isinstance(binding.loader, _PythonFrozenLoader):
                binding.loader.exec_module(binding.module)
            else:
                code = compile(
                    binding.reviewed.source_bytes,
                    binding.file,
                    "exec",
                    dont_inherit=True,
                )
                if type(code) is not types.CodeType or code.co_filename != binding.file:
                    raise InventoryError("Python tooling compiled code binding changed")
                exec(code, binding.namespace)
            _verify_python_source_module_registry(registry, reviewed_modules)
            observed = _read_regular_stable_snapshot(
                binding.reviewed.source_path,
                MAX_INVENTORY_BYTES,
                f"Python tooling runtime source {binding.reviewed.inventory_path}",
            )
            if observed.sha256 != binding.reviewed.source_sha256:
                raise InventoryError(
                    "Python tooling source changed between review and module execution"
                )
        if frozen_registry:
            bindings[0].loader.closure.live_recheck()
        yield registry
        _verify_python_source_module_registry(registry, reviewed_modules)


def _resolved_python_class_attribute(owner: type[Any], name: str) -> Any:
    for base in type.__getattribute__(owner, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    raise InventoryError(f"Python unittest class attribute is missing: {name}")


def _ordinary_synchronous_code(callable_object: Any) -> Any:
    code = getattr(callable_object, "__code__", None)
    if (
        not inspect.isfunction(callable_object)
        or code is None
        or code.co_flags
        & (inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR | inspect.CO_GENERATOR)
    ):
        raise InventoryError(
            "Python tooling discovered a noncanonical synchronous callable"
        )
    return code


def _canonical_python_test_id(
    test_class: type[unittest.TestCase], method_name: str
) -> str:
    module = type.__getattribute__(test_class, "__module__")
    qualname = type.__getattribute__(test_class, "__qualname__")
    if (
        not isinstance(module, str)
        or not module
        or not isinstance(qualname, str)
        or not qualname
    ):
        raise InventoryError("Python tooling test class identity is noncanonical")
    return f"{module}.{qualname}.{method_name}"


def _freeze_python_fixture_bindings(
    test: unittest.TestCase,
    test_class: type[unittest.TestCase],
    source_module: _PythonSourceModuleBinding | None,
) -> tuple[_PythonFixtureBinding, ...]:
    if source_module is not None:
        _verify_python_source_module_binding(source_module)
    fixtures: list[_PythonFixtureBinding] = []
    namespace = object.__getattribute__(test, "__dict__")
    for name in ("setUp", "tearDown"):
        if name in namespace:
            raise InventoryError("Python tooling discovered a shadowed test fixture")
        descriptor = _resolved_python_class_attribute(test_class, name)
        bound = object.__getattribute__(test, name)
        code = _ordinary_synchronous_code(descriptor)
        if (
            not inspect.ismethod(bound)
            or getattr(bound, "__self__", None) is not test
            or getattr(bound, "__func__", None) is not descriptor
        ):
            raise InventoryError(
                "Python tooling discovered a noncanonical test fixture"
            )
        fixtures.append(
            _PythonFixtureBinding("instance", test, name, True, descriptor, bound, code)
        )
    for name in ("setUpClass", "tearDownClass"):
        descriptor = _resolved_python_class_attribute(test_class, name)
        if not isinstance(descriptor, classmethod):
            raise InventoryError(
                "Python tooling discovered a noncanonical class fixture"
            )
        function = descriptor.__func__
        code = _ordinary_synchronous_code(function)
        bound = getattr(test_class, name)
        if (
            not inspect.ismethod(bound)
            or getattr(bound, "__self__", None) is not test_class
            or getattr(bound, "__func__", None) is not function
        ):
            raise InventoryError(
                "Python tooling discovered a noncanonical class fixture"
            )
        fixtures.append(
            _PythonFixtureBinding(
                "class", test_class, name, True, descriptor, bound, code
            )
        )
    module_name = type.__getattribute__(test_class, "__module__")
    if source_module is None:
        return tuple(fixtures)
    module_globals = source_module.namespace
    if source_module is not None and source_module.name != module_name:
        raise InventoryError("Python tooling test module registry binding is incorrect")
    for name in ("setUpModule", "tearDownModule"):
        if name not in module_globals:
            fixtures.append(
                _PythonFixtureBinding(
                    "module", source_module, name, False, None, None, None
                )
            )
            continue
        function = module_globals[name]
        code = None if function is None else _ordinary_synchronous_code(function)
        fixtures.append(
            _PythonFixtureBinding(
                "module", source_module, name, True, function, function, code
            )
        )
    return tuple(fixtures)


def _verify_python_fixture_binding(binding: _PythonFixtureBinding) -> None:
    if binding.kind == "instance":
        if not binding.present:
            raise InventoryError("Python tooling test fixture binding changed")
        namespace = object.__getattribute__(binding.owner, "__dict__")
        if binding.name in namespace:
            raise InventoryError("Python tooling test fixture binding changed")
        descriptor = _resolved_python_class_attribute(type(binding.owner), binding.name)
        bound = object.__getattribute__(binding.owner, binding.name)
        callable_object = descriptor
    elif binding.kind == "class":
        if not binding.present:
            raise InventoryError("Python tooling test fixture binding changed")
        descriptor = _resolved_python_class_attribute(binding.owner, binding.name)
        bound = getattr(binding.owner, binding.name)
        callable_object = (
            descriptor.__func__ if isinstance(descriptor, classmethod) else None
        )
    elif binding.kind == "module":
        if type(binding.owner) is not _PythonSourceModuleBinding:
            raise InventoryError("Python tooling module fixture owner is noncanonical")
        _verify_python_source_module_binding(binding.owner)
        namespace = binding.owner.namespace
        present = binding.name in namespace
        if present != binding.present:
            raise InventoryError("Python tooling test fixture binding changed")
        if not present:
            if any(
                value is not None
                for value in (
                    binding.descriptor,
                    binding.bound_callable,
                    binding.code,
                )
            ):
                raise InventoryError("Python tooling test fixture binding changed")
            return
        descriptor = namespace[binding.name]
        bound = descriptor
        callable_object = descriptor
        if descriptor is None:
            if (
                binding.descriptor is not None
                or binding.bound_callable is not None
                or binding.code is not None
            ):
                raise InventoryError("Python tooling test fixture binding changed")
            return
    else:
        raise InventoryError("Python tooling fixture binding kind is noncanonical")
    if (
        descriptor is not binding.descriptor
        or bound != binding.bound_callable
        or callable_object is None
        or getattr(callable_object, "__code__", None) is not binding.code
    ):
        raise InventoryError("Python tooling test fixture binding changed")


def _verify_python_module_fixture_transition(
    previous_fixtures: tuple[_PythonFixtureBinding, ...],
    current_fixtures: tuple[_PythonFixtureBinding, ...],
    invoke_teardown: Any,
) -> None:
    def verify_both_sides() -> None:
        for fixture in (*previous_fixtures, *current_fixtures):
            if fixture.kind == "module":
                _verify_python_fixture_binding(fixture)

    verify_both_sides()
    try:
        invoke_teardown()
    finally:
        verify_both_sides()


def _invoke_python_fixture_helper(invoke: Any, verify: Any) -> None:
    verify()
    try:
        invoke()
    finally:
        verify()


def _verify_python_test_case_dispatch(
    test: unittest.TestCase,
    trusted: _PythonUnittestRuntimePrimitives,
    method_name: str,
    expected_hooks: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    namespace = object.__getattribute__(test, "__dict__")
    dispatch = {
        "__call__": _resolved_python_class_attribute(
            trusted.test_case_type, "__call__"
        ),
        "run": trusted.test_case_run,
        "id": _resolved_python_class_attribute(trusted.test_case_type, "id"),
        "__getattribute__": object.__getattribute__,
        "__setattr__": object.__setattr__,
        **{
            name: _resolved_python_class_attribute(trusted.test_case_type, name)
            for name in _PYTHON_TEST_CASE_EXECUTION_HOOKS
            if hasattr(trusted.test_case_type, name)
        },
    }
    if expected_hooks is not None:
        dispatch.update(expected_hooks)
    if any(name in namespace for name in (*dispatch, method_name)) or any(
        _resolved_python_class_attribute(type(test), name) is not required
        for name, required in dispatch.items()
    ):
        raise InventoryError(
            "Python tooling discovered a test with noncanonical dispatch"
        )
    descriptor = _resolved_python_class_attribute(type(test), method_name)
    bound_method = object.__getattribute__(test, method_name)
    code = getattr(descriptor, "__code__", None)
    if (
        not inspect.isfunction(descriptor)
        or not inspect.ismethod(bound_method)
        or code is None
        or code.co_flags
        & (inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR | inspect.CO_GENERATOR)
        or getattr(bound_method, "__self__", None) is not test
        or getattr(bound_method, "__func__", None) is not descriptor
        or getattr(bound_method, "__code__", None) is not code
    ):
        raise InventoryError(
            "Python tooling discovered a test with a noncanonical test method"
        )
    return descriptor, bound_method, code


def _verify_python_test_bindings(
    bindings: tuple[_PythonTestBinding, ...],
    trusted: _PythonUnittestRuntimePrimitives,
    expected_hooks: dict[str, Any] | None = None,
) -> None:
    for binding in bindings:
        if type(binding.test) is not binding.test_class:
            raise InventoryError("Python tooling test class identity changed")
        if binding.source_module is not None:
            _verify_python_source_module_binding(binding.source_module)
            class_name = type.__getattribute__(binding.test_class, "__name__")
            class_qualname = type.__getattribute__(binding.test_class, "__qualname__")
            class_module = type.__getattribute__(binding.test_class, "__module__")
            if (
                not isinstance(class_name, str)
                or class_qualname != class_name
                or class_module != binding.source_module.name
                or binding.source_module.namespace.get(class_name)
                is not binding.test_class
            ):
                raise InventoryError(
                    "Python tooling test class registry binding changed"
                )
        method_name = object.__getattribute__(binding.test, "_testMethodName")
        if method_name != binding.method_name:
            raise InventoryError("Python tooling test method identity changed")
        descriptor, bound_method, code = _verify_python_test_case_dispatch(
            binding.test, trusted, binding.method_name, expected_hooks
        )
        if (
            descriptor is not binding.method_descriptor
            or code is not binding.code
            or bound_method != binding.bound_method
            or (
                binding.source_module is not None
                and (
                    getattr(descriptor, "__name__", None) != binding.descriptor_name
                    or getattr(descriptor, "__qualname__", None)
                    != binding.descriptor_qualname
                    or getattr(descriptor, "__module__", None)
                    != binding.descriptor_module
                    or hasattr(descriptor, "__wrapped__")
                    is not binding.descriptor_wrapped_present
                    or (
                        binding.descriptor_wrapped_present
                        and getattr(descriptor, "__wrapped__", None)
                        is not binding.descriptor_wrapped
                    )
                )
            )
        ):
            raise InventoryError("Python tooling test method binding changed")
        if (
            _canonical_python_test_id(binding.test_class, binding.method_name)
            != binding.runtime_id
        ):
            raise InventoryError("Python tooling canonical test identity changed")
        for fixture in binding.fixtures:
            _verify_python_fixture_binding(fixture)


def _verify_python_test_suite_dispatch(
    suite: unittest.TestSuite,
    trusted: _PythonUnittestRuntimePrimitives,
    expected_run: Any,
) -> None:
    trusted_suite_iter = _resolved_python_class_attribute(
        trusted.base_test_suite_type, "__iter__"
    )
    if type(suite) is not trusted.loader_suite_class or type(suite) is not (
        trusted.test_suite_type
    ):
        raise InventoryError("Python tooling discovery returned a noncanonical suite")
    namespace = object.__getattribute__(suite, "__dict__")
    if (
        any(name in namespace for name in ("__call__", "run", "__iter__"))
        or _resolved_python_class_attribute(type(suite), "__call__")
        is not trusted.test_suite_call
        or _resolved_python_class_attribute(type(suite), "run") is not expected_run
        or _resolved_python_class_attribute(type(suite), "__iter__")
        is not trusted_suite_iter
    ):
        raise InventoryError(
            "Python tooling discovered a suite with noncanonical dispatch"
        )


def _flatten_unittest_suite(
    suite: unittest.TestSuite,
    trusted: _PythonUnittestRuntimePrimitives,
) -> tuple[unittest.TestCase, ...]:
    tests: list[unittest.TestCase] = []

    def visit(item: unittest.TestSuite | unittest.TestCase) -> None:
        if isinstance(item, trusted.test_case_type):
            method_name = object.__getattribute__(item, "_testMethodName")
            if not isinstance(method_name, str) or not method_name:
                raise InventoryError("Python tooling test method name is malformed")
            _verify_python_test_case_dispatch(item, trusted, method_name)
            tests.append(item)
            return
        _verify_python_test_suite_dispatch(item, trusted, trusted.test_suite_run)
        for child in item:
            visit(child)

    visit(suite)
    return tuple(tests)


def _filesystem_names_alias(first: str, second: str, probe_root: Path) -> bool:
    with tempfile.TemporaryDirectory(
        prefix="test-inventory-skip-capability-", dir=probe_root
    ) as raw:
        directory = Path(raw)
        first_path = directory / first
        second_path = directory / second
        first_path.write_bytes(b"first")
        try:
            descriptor = os.open(
                second_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            return True
        else:
            os.close(descriptor)
            return False


def _darwin_descriptor_xattr_names(path: Path) -> frozenset[bytes]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise InventoryError(
            "Darwin provenance probe descriptor cannot be opened"
        ) from exc
    try:
        try:
            library = ctypes.CDLL(None, use_errno=True)
            flistxattr = library.flistxattr
            flistxattr.restype = ctypes.c_ssize_t
            flistxattr.argtypes = [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            size = flistxattr(descriptor, None, 0, 0)
            if size < 0:
                error = ctypes.get_errno()
                raise OSError(error or 5, os.strerror(error or 5))
            if size == 0:
                return frozenset()
            if size > 1024 * 1024:
                raise InventoryError(
                    "Darwin provenance probe xattr name list is too large"
                )
            names_buffer = ctypes.create_string_buffer(size)
            observed_size = flistxattr(descriptor, names_buffer, size, 0)
            if observed_size != size:
                raise InventoryError(
                    "Darwin provenance probe xattrs changed while inspected"
                )
            raw_names = names_buffer.raw[:observed_size]
            names = tuple(filter(None, raw_names.split(b"\0")))
            if not names or sum(len(name) + 1 for name in names) != observed_size:
                raise InventoryError(
                    "Darwin provenance probe xattr name list is malformed"
                )
            return frozenset(names)
        except (
            AttributeError,
            OSError,
            TypeError,
            ValueError,
            ctypes.ArgumentError,
        ) as exc:
            raise InventoryError(
                "Darwin provenance probe cannot inspect descriptor xattrs"
            ) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise InventoryError(
                "Darwin provenance probe descriptor cannot be closed"
            ) from exc


def _report_publication_platform_unavailable() -> bool:
    if os.name != "posix" or sys.platform not in {"darwin", "linux"}:
        return True
    if not hasattr(os, "O_NOFOLLOW") or os.O_NOFOLLOW is None:
        return True
    if not hasattr(os, "supports_dir_fd") or os.supports_dir_fd is None:
        return True
    if (
        not hasattr(os, "supports_follow_symlinks")
        or os.supports_follow_symlinks is None
    ):
        return True
    if not hasattr(os, "open") or os.open not in os.supports_dir_fd:
        return True
    if not hasattr(os, "stat") or os.stat not in os.supports_dir_fd:
        return True
    if not hasattr(os, "mkdir") or os.mkdir not in os.supports_dir_fd:
        return True
    if not hasattr(os, "unlink") or os.unlink not in os.supports_dir_fd:
        return True
    if not hasattr(os, "rmdir") or os.rmdir not in os.supports_dir_fd:
        return True
    if not hasattr(os, "rename") or os.rename not in os.supports_dir_fd:
        return True
    if not hasattr(os, "stat") or os.stat not in os.supports_follow_symlinks:
        return True
    return False


def _artifact_snapshot_platform_unavailable() -> bool:
    probe_name = "artifact-probe.exe" if sys.platform == "win32" else "artifact-probe"
    if (
        not hasattr(os, "O_CLOEXEC")
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        return True
    if (
        not hasattr(os, "fchmod")
        or not hasattr(os, "fstat")
        or not hasattr(os, "geteuid")
        or not hasattr(os, "open")
        or not hasattr(os, "unlink")
    ):
        return True
    try:
        with tempfile.TemporaryDirectory(
            prefix="test-inventory-artifact-capability-"
        ) as raw:
            probe = Path(raw) / probe_name
            probe.write_bytes(b"probe")
            descriptor = os.open(probe, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    return True
            finally:
                os.close(descriptor)
    except (NotImplementedError, OSError):
        return True
    return os.name != "posix" or sys.platform not in {"darwin", "linux"}


def _apply_report_publication_platform_predicate(
    predicates: dict[str, bool], platform_unavailable: bool
) -> None:
    predicates["python-skip-predicate:report-publication-platform-unavailable"] = (
        platform_unavailable
    )
    if not platform_unavailable:
        return
    for predicate_id in REPORT_PUBLICATION_SUBORDINATE_SKIP_PREDICATE_IDS:
        predicates[predicate_id] = False


def _dynamic_python_skip_predicates(probe_root: Path) -> dict[str, bool]:
    alternate_groups: list[int] = []
    if hasattr(os, "getgroups") and hasattr(os, "getegid"):
        for group in os.getgroups():
            if group != os.getegid():
                alternate_groups.append(group)
    no_setgid_inheritance = False
    if alternate_groups and hasattr(os, "chown"):
        try:
            with tempfile.TemporaryDirectory(
                prefix="test-inventory-setgid-capability-", dir=probe_root
            ) as raw:
                parent = Path(raw) / "setgid-parent"
                parent.mkdir()
                os.chown(parent, -1, alternate_groups[0])
                parent.chmod(0o2770)
                probe = parent / "inheritance-probe"
                probe.write_bytes(b"probe")
                no_setgid_inheritance = probe.stat().st_gid != alternate_groups[0]
        except OSError:
            no_setgid_inheritance = False

    no_automatic_provenance = False
    if sys.platform == "darwin":
        try:
            with tempfile.TemporaryDirectory(
                prefix="test-inventory-xattr-capability-", dir=probe_root
            ) as raw:
                probe = Path(raw) / "probe"
                probe.write_bytes(b"probe")
                no_automatic_provenance = b"com.apple.provenance" not in (
                    _darwin_descriptor_xattr_names(probe)
                )
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise InventoryError(
                "Darwin provenance probe filesystem operation failed"
            ) from exc

    aliases_case = _filesystem_names_alias("Report.svg", "report.svg", probe_root)
    aliases_normalization = _filesystem_names_alias(
        "\N{LATIN SMALL LETTER E WITH ACUTE}.svg",
        "e\N{COMBINING ACUTE ACCENT}.svg",
        probe_root,
    )
    return {
        "python-skip-predicate:no-alternate-supplementary-group": not alternate_groups,
        "python-skip-predicate:no-setgid-inheritance": no_setgid_inheritance,
        "python-skip-predicate:no-automatic-provenance-xattr": no_automatic_provenance,
        "python-skip-predicate:case-distinct-names": not aliases_case,
        "python-skip-predicate:normalization-distinct-names": not aliases_normalization,
        "python-skip-predicate:case-aliasing-filesystem": aliases_case,
    }


def _required_python_tooling_module(
    modules: dict[str, _PythonSourceModuleBinding], name: str
) -> dict[str, Any]:
    module = modules.get(name)
    if module is None:
        raise InventoryError(
            f"Python skip predicate capability module is unavailable: {name}"
        )
    _verify_python_source_module_binding(module)
    return module.namespace


def _python_skip_predicates(
    module_registry: tuple[_PythonSourceModuleBinding, ...],
) -> dict[str, bool]:
    _verify_python_source_module_registry(module_registry)
    modules = {
        PurePosixPath(binding.reviewed.inventory_path).name: binding
        for binding in module_registry
    }
    if len(modules) != len(module_registry):
        raise InventoryError("Python tooling source module basenames are duplicated")

    platform_unavailable = _report_publication_platform_unavailable()
    if platform_unavailable:
        predicates = {
            predicate_id: False
            for predicate_id in REPORT_PUBLICATION_DYNAMIC_SKIP_PREDICATE_IDS
        }
    else:
        publication = modules.get("test_report_publication.py")
        if publication is None:
            raise InventoryError(
                "Python skip predicate capability module is unavailable: "
                "test_report_publication.py"
            )
        _verify_python_source_module_binding(publication)
        predicates = _dynamic_python_skip_predicates(
            publication.reviewed.source_path.parent
        )

    level1 = _required_python_tooling_module(modules, "test_level1_report.py")
    level1_runner = level1.get("runner")
    if level1_runner is None:
        raise InventoryError("Level 1 skip predicate runner is unavailable")
    predicates["python-skip-predicate:accelerate-unavailable"] = not bool(
        level1_runner.library_available(level1_runner.DEFAULT_ACCELERATE)
    )

    level2 = _required_python_tooling_module(modules, "test_level2_report.py")
    predicates["python-skip-predicate:drop-in-blas-unavailable"] = not bool(
        level2.get("TEST_BLAS")
    )
    predicates["python-skip-predicate:file-backed-blas-unavailable"] = not bool(
        level2.get("TEST_FILE_BLAS")
    )

    artifact_specs = (
        (
            "python-skip-predicate:rank-k-artifacts-unavailable",
            "test_rank_k_report.py",
            "rank-k-probe",
        ),
        (
            "python-skip-predicate:rotg-latency-artifacts-unavailable",
            "test_rotg_latency_report.py",
            "rotg-latency-probe",
        ),
        (
            "python-skip-predicate:symm-artifacts-unavailable",
            "test_symm_report.py",
            "symm-probe",
        ),
    )
    for predicate_id, module_name, probe_name in artifact_specs:
        module = _required_python_tooling_module(modules, module_name)
        repository_root = module.get("REPO_ROOT")
        runner = module.get("runner")
        if not isinstance(repository_root, Path) or runner is None:
            raise InventoryError(
                f"Python skip predicate capability inputs are unavailable: {module_name}"
            )
        predicates[predicate_id] = not (
            (repository_root / "zig-out/bin" / probe_name).is_file()
            and (repository_root / runner.default_zynum_blas()).is_file()
        )

    triangular = _required_python_tooling_module(
        modules, "test_triangular_matrix_report.py"
    )
    triangular_root = triangular.get("REPO_ROOT")
    triangular_blas = triangular.get("integration_blas")
    if not isinstance(triangular_root, Path) or not callable(triangular_blas):
        raise InventoryError(
            "triangular-matrix skip predicate capability inputs are unavailable"
        )
    predicates["python-skip-predicate:triangular-matrix-artifacts-unavailable"] = not (
        (triangular_root / "zig-out/bin/triangular-matrix-probe").is_file()
        and (sys.platform == "darwin" or triangular_blas().is_file())
    )
    predicates["python-skip-predicate:not-darwin"] = sys.platform != "darwin"
    predicates["python-skip-predicate:artifact-snapshot-platform-unavailable"] = (
        _artifact_snapshot_platform_unavailable()
    )
    _apply_report_publication_platform_predicate(predicates, platform_unavailable)

    if set(predicates) != set(PYTHON_SKIP_PREDICATE_IDS) or not all(
        isinstance(value, bool) for value in predicates.values()
    ):
        raise InventoryError("Python skip predicate capability set is incomplete")
    return predicates


def _capture_python_unittest_runtime_primitives() -> _PythonUnittestRuntimePrimitives:
    return _PythonUnittestRuntimePrimitives(
        sys._getframe,
        secrets.token_hex,
        unittest.SkipTest,
        unittest.TestCase,
        unittest.TestCase.skipTest,
        unittest.TestCase.run,
        unittest.TestSuite,
        unittest.TestSuite.__call__,
        unittest.TestSuite.run,
        unittest.TestLoader,
        unittest.TestLoader.discover,
        unittest.TestLoader._find_tests,
        unittest.TestLoader._find_test_path,
        unittest.TestLoader.loadTestsFromModule,
        unittest.TestLoader.getTestCaseNames,
        unittest.TextTestRunner,
        unittest.TextTestRunner.__init__,
        unittest.TextTestRunner.run,
        unittest.TextTestRunner._makeResult,
        unittest.TextTestResult,
        unittest.TextTestResult.addSkip,
        unittest.TextTestResult.addSuccess,
        unittest.TextTestResult.addError,
        unittest.TextTestResult.addFailure,
        unittest.TestResult.wasSuccessful,
        unittest.BaseTestSuite,
        unittest.TestResult,
        tuple(
            (owner, name, getattr(owner, name))
            for owner, names in (
                (
                    unittest.TestCase,
                    (
                        "__call__",
                        "id",
                        "run",
                        *(
                            name
                            for name in _PYTHON_TEST_CASE_EXECUTION_HOOKS
                            if hasattr(unittest.TestCase, name)
                        ),
                    ),
                ),
                (object, ("__getattribute__", "__setattr__")),
                (
                    unittest.BaseTestSuite,
                    ("__call__", "__iter__", "run", "_removeTestAtIndex"),
                ),
                (
                    unittest.TestSuite,
                    (
                        "run",
                        "_tearDownPreviousClass",
                        "_handleModuleFixture",
                        "_handleClassSetUp",
                        "_handleModuleTearDown",
                    ),
                ),
                (
                    unittest.TestLoader,
                    (
                        "discover",
                        "_find_tests",
                        "_find_test_path",
                        "loadTestsFromModule",
                        "getTestCaseNames",
                        "suiteClass",
                    ),
                ),
                (
                    unittest.TextTestRunner,
                    ("__init__", "run", "_makeResult"),
                ),
                (
                    unittest.TestResult,
                    (
                        "startTest",
                        "startTestRun",
                        "stopTest",
                        "stopTestRun",
                        "addSuccess",
                        "addError",
                        "addFailure",
                        "addSkip",
                        "addExpectedFailure",
                        "addUnexpectedSuccess",
                        "addSubTest",
                        "wasSuccessful",
                    ),
                ),
                (
                    unittest.TextTestResult,
                    (
                        "__init__",
                        "startTestRun",
                        "stopTestRun",
                        "startTest",
                        "stopTest",
                        "addSuccess",
                        "addError",
                        "addFailure",
                        "addSkip",
                        "addExpectedFailure",
                        "addUnexpectedSuccess",
                        "addSubTest",
                        "wasSuccessful",
                    ),
                ),
            )
            for name in names
        ),
        unittest.TestLoader.suiteClass,
    )


def _verify_python_unittest_runtime_primitives(
    trusted: _PythonUnittestRuntimePrimitives,
    expected_skip_test: Any,
    expected_suite_run: Any | None = None,
    expected_case_hooks: dict[str, Any] | None = None,
    expected_suite_helpers: dict[str, Any] | None = None,
) -> None:
    if expected_suite_run is None:
        expected_suite_run = trusted.test_suite_run
    if expected_case_hooks is None:
        expected_case_hooks = {}
    if expected_suite_helpers is None:
        expected_suite_helpers = {}
    try:
        observed = (
            sys._getframe,
            secrets.token_hex,
            unittest.SkipTest,
            unittest.TestCase,
            unittest.TestCase.skipTest,
            unittest.TestCase.run,
            unittest.TestSuite,
            unittest.TestSuite.__call__,
            unittest.TestSuite.run,
            unittest.TestLoader,
            unittest.TestLoader.discover,
            unittest.TestLoader._find_tests,
            unittest.TestLoader._find_test_path,
            unittest.TestLoader.loadTestsFromModule,
            unittest.TestLoader.getTestCaseNames,
            unittest.TextTestRunner,
            unittest.TextTestRunner.__init__,
            unittest.TextTestRunner.run,
            unittest.TextTestRunner._makeResult,
            unittest.TextTestResult,
            unittest.TextTestResult.addSkip,
            unittest.TextTestResult.addSuccess,
            unittest.TextTestResult.addError,
            unittest.TextTestResult.addFailure,
            unittest.TestResult.wasSuccessful,
            unittest.BaseTestSuite,
            unittest.TestResult,
            unittest.TestLoader.suiteClass,
        )
    except AttributeError as exc:
        raise InventoryError(
            "Python tooling mutated a trusted unittest runtime primitive"
        ) from exc
    expected = (
        trusted.getframe,
        trusted.token_hex,
        trusted.skip_exception,
        trusted.test_case_type,
        expected_skip_test,
        trusted.test_case_run,
        trusted.test_suite_type,
        trusted.test_suite_call,
        expected_suite_run,
        trusted.loader_type,
        trusted.loader_discover,
        trusted.loader_find_tests,
        trusted.loader_find_test_path,
        trusted.loader_load_tests_from_module,
        trusted.loader_get_test_case_names,
        trusted.runner_type,
        trusted.runner_init,
        trusted.runner_run,
        trusted.runner_make_result,
        trusted.result_type,
        trusted.result_add_skip,
        trusted.result_add_success,
        trusted.result_add_error,
        trusted.result_add_failure,
        trusted.result_was_successful,
        trusted.base_test_suite_type,
        trusted.test_result_type,
        trusted.loader_suite_class,
    )
    if len(observed) != len(expected) or any(
        actual is not required for actual, required in zip(observed, expected)
    ):
        raise InventoryError(
            "Python tooling mutated a trusted unittest runtime primitive"
        )
    try:
        attributes_intact = all(
            getattr(owner, name)
            is (
                expected_suite_run
                if owner is trusted.test_suite_type and name == "run"
                else expected_suite_helpers[name]
                if owner is trusted.test_suite_type and name in expected_suite_helpers
                else expected_case_hooks[name]
                if owner is trusted.test_case_type and name in expected_case_hooks
                else required
            )
            for owner, name, required in trusted.integrity_attributes
        )
    except AttributeError as exc:
        raise InventoryError(
            "Python tooling mutated a trusted unittest runtime primitive"
        ) from exc
    if not attributes_intact:
        raise InventoryError(
            "Python tooling mutated a trusted unittest runtime primitive"
        )


def _verify_python_test_loader_instance(
    loader: unittest.TestLoader,
    trusted: _PythonUnittestRuntimePrimitives,
) -> None:
    namespace = object.__getattribute__(loader, "__dict__")
    if (
        type(loader) is not trusted.loader_type
        or "suiteClass" in namespace
        or _resolved_python_class_attribute(type(loader), "suiteClass")
        is not trusted.loader_suite_class
    ):
        raise InventoryError("Python tooling test loader suiteClass is noncanonical")


def _python_tooling_runtime_order_digest(runtime_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(len(runtime_ids).to_bytes(8, "big"))
    for runtime_id in runtime_ids:
        payload = runtime_id.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _python_tooling_runtime_order_projection(
    reviewed_modules: tuple[_PythonReviewedSourceModule, ...],
) -> tuple[str, ...]:
    expected_paths = tuple(path for path, _ in _PYTHON_TOOLING_REVIEWED_SOURCE_SHA256)
    if (
        type(reviewed_modules) is not tuple
        or tuple(reviewed.inventory_path for reviewed in reviewed_modules)
        != expected_paths
    ):
        raise InventoryError("Python tooling runtime-order source registry changed")
    runtime_ids: list[str] = []
    for reviewed in reviewed_modules:
        _verify_python_reviewed_source_module(reviewed)
        try:
            tree = ast.parse(
                reviewed.source_bytes.decode("utf-8"),
                filename=reviewed.inventory_path,
            )
        except (SyntaxError, UnicodeError, ValueError) as exc:
            raise InventoryError(
                "Python tooling runtime-order source cannot be parsed"
            ) from exc
        class_nodes = sorted(
            (node for node in tree.body if isinstance(node, ast.ClassDef)),
            key=lambda node: node.name,
        )
        for class_node in class_nodes:
            method_nodes = sorted(
                (
                    node
                    for node in class_node.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test")
                ),
                key=lambda node: node.name,
            )
            for method_node in method_nodes:
                runtime_ids.append(
                    f"{reviewed.module_name}.{class_node.name}.{method_node.name}"
                )
    projection = tuple(runtime_ids)
    if (
        len(projection) != 465
        or len(set(projection)) != 465
        or _python_tooling_runtime_order_digest(projection)
        != _PYTHON_TOOLING_RUNTIME_ORDER_SHA256
    ):
        raise InventoryError("Python tooling runtime-order projection changed")
    return projection


def _python_tooling_suite_contract(
    expected_tests: list[dict[str, Any]],
    skip_contract: dict[str, Any],
    suite: unittest.TestSuite,
    dynamic_sites: tuple[_PythonDynamicSkipSite, ...],
    module_registry: tuple[_PythonSourceModuleBinding, ...],
    trusted: _PythonUnittestRuntimePrimitives,
    runtime_integrity_callback: Any = None,
) -> _PythonToolingSuiteContract:
    expected_ids = []
    for expected in expected_tests:
        if not isinstance(expected, dict) or not isinstance(expected.get("name"), str):
            raise InventoryError("Python tooling expected test identity is malformed")
        expected_ids.append(_unittest_runtime_id(expected["name"]))
    if len(expected_ids) != len(set(expected_ids)):
        raise InventoryError(
            "Python tooling expected runtime identities are not unique"
        )
    declared: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in skip_contract["entries"]:
        pair = (_unittest_runtime_id(entry["test"]), entry["reason"])
        if pair in declared:
            raise InventoryError(
                "Python tooling skip contract repeats an identity/reason"
            )
        declared[pair] = entry

    tests = _flatten_unittest_suite(suite, trusted)
    discovered_ids = [
        _canonical_python_test_id(
            type(test), object.__getattribute__(test, "_testMethodName")
        )
        for test in tests
    ]
    runtime_order = _python_tooling_runtime_order_projection(
        tuple(binding.reviewed for binding in module_registry)
    )
    if (
        len(discovered_ids) != len(set(discovered_ids))
        or set(discovered_ids) != set(expected_ids)
        or tuple(discovered_ids) != runtime_order
    ):
        raise InventoryError(
            "Python tooling discovered test identities differ from the "
            "inventory contract"
        )
    discovered_test_bindings_list: list[_PythonTestBinding] = []
    modules_by_name = {binding.name: binding for binding in module_registry}
    if len(modules_by_name) != len(module_registry):
        raise InventoryError("Python tooling source module registry is duplicated")
    _verify_python_source_module_registry(module_registry)
    for test, runtime_id in zip(tests, discovered_ids):
        method_name = object.__getattribute__(test, "_testMethodName")
        descriptor, bound_method, code = _verify_python_test_case_dispatch(
            test, trusted, method_name
        )
        test_class = type(test)
        class_name = type.__getattribute__(test_class, "__name__")
        class_qualname = type.__getattribute__(test_class, "__qualname__")
        class_module = type.__getattribute__(test_class, "__module__")
        source_module = modules_by_name.get(class_module)
        if (
            source_module is None
            or not isinstance(class_name, str)
            or class_qualname != class_name
            or source_module.namespace.get(class_name) is not test_class
            or runtime_id != f"{source_module.name}.{class_name}.{method_name}"
        ):
            raise InventoryError(
                "Python tooling discovered test class is outside the source module registry"
            )
        descriptor_name = getattr(descriptor, "__name__", None)
        descriptor_qualname = getattr(descriptor, "__qualname__", None)
        descriptor_module = getattr(descriptor, "__module__", None)
        if (
            descriptor_name != method_name
            or descriptor_qualname != f"{class_name}.{method_name}"
            or descriptor_module != source_module.name
        ):
            raise InventoryError(
                "Python tooling discovered test method metadata is noncanonical"
            )
        wrapped_present = hasattr(descriptor, "__wrapped__")
        wrapped = getattr(descriptor, "__wrapped__", None)
        if wrapped_present and (
            not inspect.isfunction(wrapped)
            or wrapped is descriptor
            or getattr(wrapped, "__name__", None) != descriptor_name
            or getattr(wrapped, "__qualname__", None) != descriptor_qualname
            or getattr(wrapped, "__module__", None) != descriptor_module
            or getattr(wrapped, "__globals__", None) is not source_module.namespace
        ):
            raise InventoryError(
                "Python tooling discovered test method __wrapped__ binding is noncanonical"
            )
        discovered_test_bindings_list.append(
            _PythonTestBinding(
                test=test,
                runtime_id=runtime_id,
                test_class=test_class,
                method_name=method_name,
                method_descriptor=descriptor,
                bound_method=bound_method,
                code=code,
                fixtures=_freeze_python_fixture_bindings(
                    test, test_class, source_module
                ),
                source_module=source_module,
                descriptor_name=descriptor_name,
                descriptor_qualname=descriptor_qualname,
                descriptor_module=descriptor_module,
                descriptor_wrapped_present=wrapped_present,
                descriptor_wrapped=wrapped,
            )
        )
    discovered_test_bindings = tuple(discovered_test_bindings_list)
    bindings_by_object = {
        id(binding.test): binding for binding in discovered_test_bindings
    }
    if len(bindings_by_object) != len(discovered_test_bindings):
        raise InventoryError("Python tooling discovered duplicate test objects")
    predicates = _python_skip_predicates(module_registry)
    active_platform_entries = {
        pair: entry
        for pair, entry in declared.items()
        if entry["predicate_id"] in PYTHON_INVENTORY_PLATFORM_PREDICATE_IDS
        and predicates[entry["predicate_id"]]
    }
    active_platform_ids = {runtime_id for runtime_id, _ in active_platform_entries}
    required_decorator_skips: set[tuple[str, str]] = set()
    for binding in discovered_test_bindings:
        test = binding.test
        method_name = getattr(test, "_testMethodName", None)
        if not isinstance(method_name, str) or not method_name:
            raise InventoryError("Python tooling discovered test method is malformed")
        method = getattr(test, method_name)
        test_class = test.__class__
        if not (
            getattr(test_class, "__unittest_skip__", False)
            or getattr(method, "__unittest_skip__", False)
        ):
            continue
        reason = getattr(test_class, "__unittest_skip_why__", "") or getattr(
            method, "__unittest_skip_why__", ""
        )
        pair = (binding.runtime_id, reason)
        if (
            not isinstance(reason, str)
            or pair not in declared
            or declared[pair]["predicate_id"] not in PYTHON_DECORATOR_SKIP_PREDICATE_IDS
        ):
            raise InventoryError(
                "Python tooling discovered an undeclared unittest skip decorator: "
                f"{pair!r}"
            )
        if binding.runtime_id in active_platform_ids:
            continue
        required_decorator_skips.add(pair)
    expected_decorator_skips = frozenset(
        pair
        for pair, entry in declared.items()
        if entry["predicate_id"] in PYTHON_DECORATOR_SKIP_PREDICATE_IDS
        and predicates[entry["predicate_id"]]
        and pair[0] not in active_platform_ids
    )
    if frozenset(required_decorator_skips) != expected_decorator_skips:
        raise InventoryError(
            "Python tooling decorator skips differ from independently evaluated "
            "finite predicates"
        )
    permitted_dynamic_skips = frozenset(
        pair
        for pair, entry in declared.items()
        if entry["predicate_id"] not in PYTHON_DECORATOR_SKIP_PREDICATE_IDS
        and entry["predicate_id"] not in PYTHON_INVENTORY_PLATFORM_PREDICATE_IDS
        and predicates[entry["predicate_id"]]
        and pair[0] not in active_platform_ids
    )
    bindings_by_id = {
        binding.runtime_id: binding for binding in discovered_test_bindings
    }
    authorizations: list[_PythonDynamicSkipAuthorization] = []
    reviewed_pairs: set[tuple[str, str]] = set()
    verified_sources: set[tuple[Path, str]] = set()
    for site in dynamic_sites:
        pair = (site.runtime_id, site.reason)
        if pair in reviewed_pairs or pair not in declared:
            raise InventoryError(
                "Python tooling dynamic skip source sites are noncanonical"
            )
        reviewed_pairs.add(pair)
        binding = bindings_by_id.get(site.runtime_id)
        if binding is None:
            raise InventoryError(
                "Python tooling dynamic skip source site has no discovered test"
            )
        test = binding.test
        code = (
            _ordinary_synchronous_code(binding.descriptor_wrapped)
            if binding.descriptor_wrapped_present
            else binding.code
        )
        if (
            code is None
            or Path(code.co_filename).resolve() != site.source_path.resolve()
            or not (code.co_firstlineno <= site.line)
        ):
            raise InventoryError(
                "Python tooling dynamic skip source site does not bind to the "
                "loaded test code"
            )
        source_identity = (site.source_path, site.source_sha256)
        if source_identity not in verified_sources:
            observed = _read_regular_stable_snapshot(
                site.source_path,
                MAX_INVENTORY_BYTES,
                f"Python tooling runtime source {site.source_path}",
            )
            if observed.sha256 != site.source_sha256:
                raise InventoryError(
                    "Python tooling source changed between review and execution"
                )
            verified_sources.add(source_identity)
        if pair in permitted_dynamic_skips:
            authorizations.append(
                _PythonDynamicSkipAuthorization(
                    test,
                    site.runtime_id,
                    site.reason,
                    code,
                    site.line,
                )
            )
    if {
        (authorization.runtime_id, authorization.reason)
        for authorization in authorizations
    } != permitted_dynamic_skips:
        raise InventoryError(
            "Python tooling active dynamic skips do not have exact runtime sites"
        )
    platform_authorizations: list[_PythonPlatformSkipAuthorization] = []
    for pair, entry in active_platform_entries.items():
        runtime_id, reason = pair
        binding = bindings_by_id.get(runtime_id)
        if binding is None:
            raise InventoryError(
                "Python tooling platform applicability has no discovered test binding"
            )
        for fixture in binding.fixtures:
            if (
                fixture.kind == "module"
                and fixture.present
                and fixture.descriptor is not None
            ):
                raise InventoryError(
                    "Python tooling platform applicability has an unsafe module fixture"
                )
            if fixture.kind == "class":
                trusted_descriptor = _resolved_python_class_attribute(
                    trusted.test_case_type, fixture.name
                )
                if fixture.descriptor is not trusted_descriptor:
                    raise InventoryError(
                        "Python tooling platform applicability has an unsafe class fixture"
                    )
        platform_authorizations.append(
            _PythonPlatformSkipAuthorization(
                binding.test,
                runtime_id,
                reason,
                entry["predicate_id"],
            )
        )
    platform_skips = frozenset(active_platform_entries)
    return _PythonToolingSuiteContract(
        len(tests),
        expected_decorator_skips,
        permitted_dynamic_skips,
        tuple(authorizations),
        platform_skips,
        tuple(platform_authorizations),
        discovered_test_bindings,
        runtime_integrity_callback,
        runtime_order,
    )


_PYTHON_RESULT_CALLBACK_NAMES = (
    "__init__",
    "startTestRun",
    "stopTestRun",
    "startTest",
    "stopTest",
    "addSuccess",
    "addFailure",
    "addError",
    "addSubTest",
    "addSkip",
    "addExpectedFailure",
    "addUnexpectedSuccess",
    "wasSuccessful",
)
_PYTHON_RESULT_CONTAINER_NAMES = (
    "failures",
    "errors",
    "skipped",
    "expectedFailures",
    "unexpectedSuccesses",
)


class _PythonToolingOutcome(NamedTuple):
    executed: int
    successful: bool
    skips: frozenset[tuple[str, str]]
    failures: int
    errors: int
    expected_failures: int
    unexpected_successes: int


class _PythonToolingRootSummary(NamedTuple):
    expected: int
    discovered: int
    outcome: _PythonToolingOutcome
    dynamic_skips: int
    artifact_platform_skips: int
    publication_platform_skips: int
    platform_skips: int


def _python_inventory_error_from_exc_info(error: Any) -> InventoryError | None:
    if (
        isinstance(error, tuple)
        and len(error) == 3
        and isinstance(error[1], InventoryError)
    ):
        return error[1]
    return None


class _PythonOutcomeLedger:
    def __init__(self, contract: _PythonToolingSuiteContract) -> None:
        self._bindings = {
            id(binding.test): binding for binding in contract.discovered_test_bindings
        }
        if len(self._bindings) != len(contract.discovered_test_bindings):
            raise InventoryError("Python tooling outcome bindings are duplicated")
        self._result: unittest.TestResult | None = None
        self._container_objects: dict[str, list[Any]] = {}
        self._container_snapshots: dict[str, tuple[Any, ...]] = {}
        self._tests_run = 0
        self._phase = "new"
        self._active: _PythonTestBinding | None = None
        self._started: set[int] = set()
        self._stopped: set[int] = set()
        self._events: dict[int, list[str]] = {}
        self._skips: set[tuple[str, str]] = set()
        self._failures = 0
        self._errors = 0
        self._expected_failures = 0
        self._unexpected_successes = 0

    def binding(self, test: unittest.TestCase) -> _PythonTestBinding | None:
        binding = self._bindings.get(id(test))
        return binding if binding is not None and binding.test is test else None

    def initialize(self, result: unittest.TestResult) -> None:
        if self._phase != "new" or self._result is not None:
            raise InventoryError("Python tooling result initialized more than once")
        self._result = result
        for name in _PYTHON_RESULT_CONTAINER_NAMES:
            value = getattr(result, name, None)
            if not isinstance(value, list):
                raise InventoryError("Python tooling result container is noncanonical")
            self._container_objects[name] = value
            self._container_snapshots[name] = tuple(value)
        tests_run = getattr(result, "testsRun", None)
        if (
            not isinstance(tests_run, int)
            or isinstance(tests_run, bool)
            or tests_run != 0
        ):
            raise InventoryError("Python tooling result testsRun is noncanonical")
        self._tests_run = tests_run
        self._phase = "initialized"

    def _require_result(self, result: unittest.TestResult) -> None:
        if self._result is not result:
            raise InventoryError(
                "Python tooling callback used an unknown result object"
            )

    def verify_containers(self, result: unittest.TestResult) -> None:
        self._require_result(result)
        for name in _PYTHON_RESULT_CONTAINER_NAMES:
            value = getattr(result, name, None)
            if (
                value is not self._container_objects[name]
                or tuple(value) != self._container_snapshots[name]
            ):
                raise InventoryError("Python tooling result outcome container changed")
        tests_run = getattr(result, "testsRun", None)
        if (
            not isinstance(tests_run, int)
            or isinstance(tests_run, bool)
            or tests_run != self._tests_run
        ):
            raise InventoryError("Python tooling result testsRun changed")

    def invoke_trusted(
        self,
        result: unittest.TestResult,
        callback: Any,
        *args: Any,
        allowed_delta: str | None = None,
        tests_run_delta: int = 0,
    ) -> Any:
        self.verify_containers(result)
        before = dict(self._container_snapshots)
        returned = callback(result, *args)
        tests_run = getattr(result, "testsRun", None)
        if (
            not isinstance(tests_run, int)
            or isinstance(tests_run, bool)
            or tests_run != self._tests_run + tests_run_delta
        ):
            raise InventoryError("Python tooling result testsRun delta is noncanonical")
        self._tests_run = tests_run
        changed: list[str] = []
        for name in _PYTHON_RESULT_CONTAINER_NAMES:
            value = getattr(result, name, None)
            if value is not self._container_objects[name] or not isinstance(
                value, list
            ):
                raise InventoryError("Python tooling result outcome container changed")
            current = tuple(value)
            prior = before[name]
            if current != prior:
                if len(current) != len(prior) + 1 or current[:-1] != prior:
                    raise InventoryError(
                        "Python tooling result outcome container changed"
                    )
                changed.append(name)
            self._container_snapshots[name] = current
        expected = [] if allowed_delta is None else [allowed_delta]
        if changed != expected:
            raise InventoryError("Python tooling result outcome delta is noncanonical")
        return returned

    def start_run(self, result: unittest.TestResult, callback: Any) -> None:
        if self._phase != "initialized":
            raise InventoryError("Python tooling test run phase is noncanonical")
        self.invoke_trusted(result, callback)
        self._phase = "running"

    def stop_run(self, result: unittest.TestResult, callback: Any) -> None:
        if self._phase != "running":
            raise InventoryError("Python tooling test run phase is noncanonical")
        self.invoke_trusted(result, callback)
        self._phase = "stopped"

    def start_test(
        self, result: unittest.TestResult, test: unittest.TestCase, callback: Any
    ) -> None:
        binding = self.binding(test)
        if (
            self._phase != "running"
            or self._active is not None
            or binding is None
            or id(test) in self._started
        ):
            raise InventoryError("Python tooling startTest callback is noncanonical")
        self.invoke_trusted(result, callback, test, tests_run_delta=1)
        self._started.add(id(test))
        self._active = binding
        self._events[id(test)] = []

    def _require_active(self, test: unittest.TestCase) -> _PythonTestBinding:
        binding = self.binding(test)
        if binding is None or self._active is not binding:
            raise InventoryError("Python tooling result callback test is noncanonical")
        return binding

    def record(
        self,
        result: unittest.TestResult,
        test: unittest.TestCase,
        event: str,
        callback: Any,
        *args: Any,
        allowed_delta: str | None = None,
        counter: str | None = None,
    ) -> None:
        self._require_active(test)
        events = self._events[id(test)]
        if event == "success" and any(prior != "subtest-success" for prior in events):
            raise InventoryError("Python tooling test outcome is duplicated")
        if event in {"success", "skip", "expected-failure", "unexpected-success"}:
            if any(
                prior in {"success", "skip", "expected-failure", "unexpected-success"}
                for prior in events
            ):
                raise InventoryError("Python tooling test outcome is duplicated")
        self.invoke_trusted(result, callback, test, *args, allowed_delta=allowed_delta)
        events.append(event)
        if counter is not None:
            current = object.__getattribute__(self, counter)
            object.__setattr__(self, counter, current + 1)

    def record_fixture_error(
        self,
        result: unittest.TestResult,
        test: Any,
        error: Any,
        callback: Any,
    ) -> None:
        if self._phase != "running" or self._active is not None:
            raise InventoryError("Python tooling fixture error phase is noncanonical")
        self.invoke_trusted(result, callback, test, error, allowed_delta="errors")
        self._errors += 1

    def record_inventory_error(self, test: unittest.TestCase) -> None:
        self._require_active(test)
        self._events[id(test)].append("inventory-error")

    def stop_test(
        self, result: unittest.TestResult, test: unittest.TestCase, callback: Any
    ) -> None:
        self._require_active(test)
        if not self._events[id(test)]:
            raise InventoryError("Python tooling test has no canonical outcome")
        self.invoke_trusted(result, callback, test)
        self._stopped.add(id(test))
        self._active = None

    def record_skip(self, binding: _PythonTestBinding, reason: str) -> None:
        pair = (binding.runtime_id, reason)
        if pair in self._skips:
            raise InventoryError("Python tooling skipped outcome is duplicated")
        self._skips.add(pair)

    def successful(self, result: unittest.TestResult) -> bool:
        if self._phase != "stopped":
            raise InventoryError("Python tooling wasSuccessful phase is noncanonical")
        self.verify_containers(result)
        return not any(
            (
                self._failures,
                self._errors,
                self._expected_failures,
                self._unexpected_successes,
            )
        )

    def outcome(self, result: unittest.TestResult) -> _PythonToolingOutcome:
        if self._phase != "stopped" or self._active is not None:
            raise InventoryError("Python tooling result phase is incomplete")
        self.verify_containers(result)
        if self._started != self._stopped:
            raise InventoryError("Python tooling test lifecycle is incomplete")
        if self._tests_run != len(self._stopped):
            raise InventoryError("Python tooling executed count is noncanonical")
        return _PythonToolingOutcome(
            executed=len(self._stopped),
            successful=not any(
                (
                    self._failures,
                    self._errors,
                    self._expected_failures,
                    self._unexpected_successes,
                )
            ),
            skips=frozenset(self._skips),
            failures=self._failures,
            errors=self._errors,
            expected_failures=self._expected_failures,
            unexpected_successes=self._unexpected_successes,
        )


class _PythonSkipRuntimeAuthorizer:
    _TICKET_PREFIX = "zynum-reviewed-skip-ticket:"

    def __init__(
        self,
        contract: _PythonToolingSuiteContract,
        trusted: _PythonUnittestRuntimePrimitives,
    ) -> None:
        self._trusted = trusted
        self._decorator_skips = contract.required_decorator_skips
        self._bindings = {
            id(binding.test): binding for binding in contract.discovered_test_bindings
        }
        if len(self._bindings) != len(contract.discovered_test_bindings):
            raise InventoryError("Python tooling runtime test bindings are duplicated")
        self._authorizations = {
            (id(authorization.test), authorization.reason): authorization
            for authorization in contract.dynamic_skip_authorizations
        }
        if len(self._authorizations) != len(contract.dynamic_skip_authorizations):
            raise InventoryError("Python tooling runtime skip sites are duplicated")
        self._platform_authorizations = {
            id(authorization.test): authorization
            for authorization in contract.platform_skip_authorizations
        }
        if len(self._platform_authorizations) != len(
            contract.platform_skip_authorizations
        ):
            raise InventoryError(
                "Python tooling runtime platform skip bindings are duplicated"
            )
        self._tickets: dict[
            str, _PythonDynamicSkipAuthorization | _PythonPlatformSkipAuthorization
        ] = {}
        self.outcome_ledger = _PythonOutcomeLedger(contract)

    def _binding(self, test: unittest.TestCase) -> _PythonTestBinding:
        binding = self._bindings.get(id(test))
        if binding is None or binding.test is not test:
            raise InventoryError("Python tooling rejected an unknown test object")
        return binding

    def skip_test(
        self, test: unittest.TestCase, reason: str, caller_frame: Any
    ) -> None:
        if not isinstance(reason, str):
            raise InventoryError("Python tooling skip reason is noncanonical")
        binding = self._binding(test)
        authorization = self._authorizations.get((id(test), reason))
        if (
            authorization is None
            or authorization.test is not test
            or authorization.runtime_id != binding.runtime_id
            or caller_frame.f_code is not authorization.code
            or caller_frame.f_lineno != authorization.line
        ):
            raise InventoryError("Python tooling rejected an unauthorized dynamic skip")
        self._issue_ticket(authorization)

    def skip_platform_applicability(self, test: unittest.TestCase) -> None:
        binding = self._binding(test)
        authorization = self._platform_authorizations.get(id(test))
        if (
            authorization is None
            or authorization.test is not test
            or authorization.runtime_id != binding.runtime_id
            or authorization.predicate_id not in PYTHON_INVENTORY_PLATFORM_PREDICATE_IDS
        ):
            raise InventoryError(
                "Python tooling rejected an unauthorized platform applicability skip"
            )
        self._issue_ticket(authorization)

    def has_platform_applicability_skip(self, test: unittest.TestCase) -> bool:
        authorization = self._platform_authorizations.get(id(test))
        return authorization is not None and authorization.test is test

    def _issue_ticket(
        self,
        authorization: _PythonDynamicSkipAuthorization
        | _PythonPlatformSkipAuthorization,
    ) -> None:
        for _ in range(128):
            ticket = self._trusted.token_hex(32)
            if ticket not in self._tickets:
                self._tickets[ticket] = authorization
                raise self._trusted.skip_exception(f"{self._TICKET_PREFIX}{ticket}")
        raise InventoryError("Python tooling could not allocate a skip ticket")

    def result_class(self) -> type[unittest.TextTestResult]:
        authorizer = self
        ledger = self.outcome_ledger
        result_base = self._trusted.result_type
        trusted_callbacks = {
            name: _trusted_python_unittest_attribute(self._trusted, result_base, name)
            for name in _PYTHON_RESULT_CALLBACK_NAMES
        }

        class AuthorizedTextTestResult(result_base):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                trusted_callbacks["__init__"](self, *args, **kwargs)
                ledger.initialize(self)

            def startTestRun(self) -> None:
                ledger.start_run(self, trusted_callbacks["startTestRun"])

            def stopTestRun(self) -> None:
                ledger.stop_run(self, trusted_callbacks["stopTestRun"])

            def startTest(self, test: unittest.TestCase) -> None:
                ledger.start_test(self, test, trusted_callbacks["startTest"])

            def stopTest(self, test: unittest.TestCase) -> None:
                ledger.stop_test(self, test, trusted_callbacks["stopTest"])

            def addSuccess(self, test: unittest.TestCase) -> None:
                ledger.record(
                    self,
                    test,
                    "success",
                    trusted_callbacks["addSuccess"],
                )

            def addError(self, test: unittest.TestCase, error: Any) -> None:
                inventory_error = _python_inventory_error_from_exc_info(error)
                if inventory_error is not None:
                    if ledger.binding(test) is not None:
                        ledger.record_inventory_error(test)
                    raise inventory_error
                if ledger.binding(test) is None:
                    ledger.record_fixture_error(
                        self, test, error, trusted_callbacks["addError"]
                    )
                    return
                ledger.record(
                    self,
                    test,
                    "error",
                    trusted_callbacks["addError"],
                    error,
                    allowed_delta="errors",
                    counter="_errors",
                )

            def addFailure(self, test: unittest.TestCase, error: Any) -> None:
                inventory_error = _python_inventory_error_from_exc_info(error)
                if inventory_error is not None:
                    ledger.record_inventory_error(test)
                    raise inventory_error
                ledger.record(
                    self,
                    test,
                    "failure",
                    trusted_callbacks["addFailure"],
                    error,
                    allowed_delta="failures",
                    counter="_failures",
                )

            def addSubTest(
                self, test: unittest.TestCase, subtest: Any, error: Any
            ) -> None:
                inventory_error = _python_inventory_error_from_exc_info(error)
                if inventory_error is not None:
                    ledger.record_inventory_error(test)
                    raise inventory_error
                allowed_delta = None
                counter = None
                if error is not None:
                    failure_exception = type.__getattribute__(
                        type(test), "failureException"
                    )
                    allowed_delta = (
                        "failures"
                        if issubclass(error[0], failure_exception)
                        else "errors"
                    )
                    counter = "_failures" if allowed_delta == "failures" else "_errors"
                ledger.record(
                    self,
                    test,
                    "subtest-failure" if error is not None else "subtest-success",
                    trusted_callbacks["addSubTest"],
                    subtest,
                    error,
                    allowed_delta=allowed_delta,
                    counter=counter,
                )

            def addSkip(self, test: unittest.TestCase, reason: str) -> None:
                binding = authorizer._binding(test)
                pair = (binding.runtime_id, reason)
                if pair in authorizer._decorator_skips:
                    canonical_reason = reason
                else:
                    if not isinstance(reason, str) or not reason.startswith(
                        authorizer._TICKET_PREFIX
                    ):
                        raise InventoryError(
                            "Python tooling rejected an unauthorized skipped outcome"
                        )
                    ticket = reason.removeprefix(authorizer._TICKET_PREFIX)
                    if not re.fullmatch(r"[0-9a-f]{64}", ticket):
                        raise InventoryError(
                            "Python tooling rejected a malformed skip ticket"
                        )
                    authorization = authorizer._tickets.get(ticket)
                    if (
                        authorization is None
                        or authorization.test is not test
                        or authorization.runtime_id != binding.runtime_id
                    ):
                        raise InventoryError(
                            "Python tooling rejected an unauthorized skip ticket"
                        )
                    del authorizer._tickets[ticket]
                    canonical_reason = authorization.reason
                ledger.record(
                    self,
                    test,
                    "skip",
                    trusted_callbacks["addSkip"],
                    canonical_reason,
                    allowed_delta="skipped",
                )
                ledger.record_skip(binding, canonical_reason)

            def addExpectedFailure(self, test: unittest.TestCase, error: Any) -> None:
                inventory_error = _python_inventory_error_from_exc_info(error)
                if inventory_error is not None:
                    ledger.record_inventory_error(test)
                    raise inventory_error
                ledger.record(
                    self,
                    test,
                    "expected-failure",
                    trusted_callbacks["addExpectedFailure"],
                    error,
                    allowed_delta="expectedFailures",
                    counter="_expected_failures",
                )

            def addUnexpectedSuccess(self, test: unittest.TestCase) -> None:
                ledger.record(
                    self,
                    test,
                    "unexpected-success",
                    trusted_callbacks["addUnexpectedSuccess"],
                    allowed_delta="unexpectedSuccesses",
                    counter="_unexpected_successes",
                )

            def wasSuccessful(self) -> bool:
                return ledger.successful(self)

        return AuthorizedTextTestResult

    def require_all_tickets_consumed(self) -> None:
        if self._tickets:
            raise InventoryError(
                "Python tooling has unconsumed dynamic skip tickets: "
                f"count={len(self._tickets)}"
            )


def _verify_python_tooling_result_integrity(
    result: unittest.TestResult,
    result_class: type[unittest.TextTestResult],
    authorized_methods: dict[str, Any],
    outcome_ledger: _PythonOutcomeLedger,
) -> None:
    if (
        type(result) is not result_class
        or set(authorized_methods) != set(_PYTHON_RESULT_CALLBACK_NAMES)
        or any(
            getattr(result_class, name, None) is not method
            for name, method in authorized_methods.items()
        )
        or any(name in vars(result) for name in authorized_methods)
    ):
        raise InventoryError("Python tooling runner returned a tampered result object")
    outcome_ledger.verify_containers(result)


def _trusted_python_unittest_attribute(
    trusted: _PythonUnittestRuntimePrimitives,
    owner: Any,
    name: str,
) -> Any:
    for candidate_owner, candidate_name, required in trusted.integrity_attributes:
        if candidate_owner is owner and candidate_name == name:
            return required
    raise InventoryError(f"trusted Python unittest attribute is missing: {name}")


def _verify_python_tooling_runtime_integrity(callback: Any) -> None:
    if callback is None:
        return
    if not callable(callback):
        raise InventoryError("Python tooling runtime integrity callback changed")
    callback()


def _run_verified_python_test_suite(
    suite: unittest.TestSuite,
    result: unittest.TestResult,
    bindings_by_object: dict[int, _PythonTestBinding],
    trusted: _PythonUnittestRuntimePrimitives,
    expected_suite_run: Any,
    remaining_binding_ids: set[int],
    expected_case_hooks: dict[str, Any],
    fixture_context: list[_PythonTestBinding | None],
    expected_module_setup_failed: list[bool],
    verified_module_teardown: Any,
    runtime_integrity_callback: Any,
    runtime_order: tuple[str, ...],
    runtime_cursor: list[int],
    debug: bool = False,
) -> unittest.TestResult:
    if debug:
        raise InventoryError("Python tooling verified suite rejects debug dispatch")
    _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
    _verify_python_test_suite_dispatch(suite, trusted, expected_suite_run)
    expected_previous_class = (
        None if fixture_context[0] is None else fixture_context[0].test_class
    )
    if (
        getattr(result, "shouldStop", None) is not False
        or not isinstance(getattr(result, "_testRunEntered", None), bool)
        or getattr(result, "_previousTestClass", None) is not expected_previous_class
        or getattr(result, "_moduleSetUpFailed", None)
        is not expected_module_setup_failed[0]
    ):
        raise InventoryError("Python tooling suite result state is noncanonical")
    top_level = False
    if getattr(result, "_testRunEntered", False) is False:
        result._testRunEntered = top_level = True
    tear_down_previous_class = _trusted_python_unittest_attribute(
        trusted, trusted.test_suite_type, "_tearDownPreviousClass"
    )
    handle_module_fixture = _trusted_python_unittest_attribute(
        trusted, trusted.test_suite_type, "_handleModuleFixture"
    )
    handle_class_setup = _trusted_python_unittest_attribute(
        trusted, trusted.test_suite_type, "_handleClassSetUp"
    )
    remove_test = _trusted_python_unittest_attribute(
        trusted, trusted.base_test_suite_type, "_removeTestAtIndex"
    )
    for index, test in enumerate(suite):
        expected_previous_class = (
            None if fixture_context[0] is None else fixture_context[0].test_class
        )
        if (
            getattr(result, "shouldStop", None) is not False
            or getattr(result, "_testRunEntered", None) is not True
            or getattr(result, "_previousTestClass", None)
            is not expected_previous_class
            or getattr(result, "_moduleSetUpFailed", None)
            is not expected_module_setup_failed[0]
        ):
            raise InventoryError("Python tooling suite result state is noncanonical")
        if isinstance(test, trusted.test_case_type):
            binding = bindings_by_object.get(id(test))
            if (
                binding is None
                or binding.test is not test
                or id(test) not in remaining_binding_ids
            ):
                raise InventoryError(
                    "Python tooling verified suite found a duplicate or unknown test object"
                )
            cursor = runtime_cursor[0]
            if (
                cursor >= len(runtime_order)
                or runtime_order[cursor] != binding.runtime_id
            ):
                raise InventoryError(
                    "Python tooling verified suite runtime order changed"
                )
            runtime_cursor[0] = cursor + 1
            remaining_binding_ids.remove(id(test))
            previous_class = getattr(result, "_previousTestClass", None)
            previous_bindings = tuple(
                candidate
                for candidate in bindings_by_object.values()
                if previous_class is candidate.test_class
            )
            if previous_bindings:
                _verify_python_test_bindings(
                    previous_bindings, trusted, expected_case_hooks
                )

            def verify_previous_bindings() -> None:
                _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
                if previous_bindings:
                    _verify_python_test_bindings(
                        previous_bindings, trusted, expected_case_hooks
                    )

            for fixture in binding.fixtures:
                if fixture.kind in {"class", "module"}:
                    _verify_python_fixture_binding(fixture)
            _invoke_python_fixture_helper(
                lambda: tear_down_previous_class(suite, test, result),
                verify_previous_bindings,
            )
            fixture_context[0] = binding

            def verify_current_class_and_module_fixtures() -> None:
                _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
                for fixture in binding.fixtures:
                    if fixture.kind in {"class", "module"}:
                        _verify_python_fixture_binding(fixture)

            _invoke_python_fixture_helper(
                lambda: handle_module_fixture(suite, test, result),
                verify_current_class_and_module_fixtures,
            )
            observed_module_setup_failed = getattr(result, "_moduleSetUpFailed", None)
            if not isinstance(observed_module_setup_failed, bool):
                raise InventoryError(
                    "Python tooling suite result state is noncanonical"
                )
            expected_module_setup_failed[0] = observed_module_setup_failed

            def verify_current_test_binding() -> None:
                _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
                _verify_python_test_bindings((binding,), trusted, expected_case_hooks)

            _invoke_python_fixture_helper(
                lambda: handle_class_setup(suite, test, result),
                verify_current_test_binding,
            )
            result._previousTestClass = type(test)
            if getattr(type(test), "_classSetupFailed", False) or getattr(
                result, "_moduleSetUpFailed", False
            ):
                continue
            try:
                _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
                trusted.test_case_run(test, result)
            finally:
                _verify_python_test_bindings((binding,), trusted, expected_case_hooks)
                _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
                if (
                    getattr(result, "shouldStop", None) is not False
                    or getattr(result, "_testRunEntered", None) is not True
                    or getattr(result, "_previousTestClass", None) is not type(test)
                    or getattr(result, "_moduleSetUpFailed", None)
                    is not expected_module_setup_failed[0]
                ):
                    raise InventoryError(
                        "Python tooling suite result state is noncanonical"
                    )
        else:
            _run_verified_python_test_suite(
                test,
                result,
                bindings_by_object,
                trusted,
                expected_suite_run,
                remaining_binding_ids,
                expected_case_hooks,
                fixture_context,
                expected_module_setup_failed,
                verified_module_teardown,
                runtime_integrity_callback,
                runtime_order,
                runtime_cursor,
            )
        if object.__getattribute__(suite, "_cleanup"):
            remove_test(suite, index)
    if top_level:
        previous_class = getattr(result, "_previousTestClass", None)
        previous_bindings = tuple(
            candidate
            for candidate in bindings_by_object.values()
            if previous_class is candidate.test_class
        )
        if previous_bindings:
            _verify_python_test_bindings(
                previous_bindings, trusted, expected_case_hooks
            )

        def verify_final_previous_bindings() -> None:
            _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
            if previous_bindings:
                _verify_python_test_bindings(
                    previous_bindings, trusted, expected_case_hooks
                )

        _invoke_python_fixture_helper(
            lambda: tear_down_previous_class(suite, None, result),
            verify_final_previous_bindings,
        )
        fixture_context[0] = None
        _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
        try:
            verified_module_teardown(suite, result)
        finally:
            _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
        result._testRunEntered = False
        if remaining_binding_ids:
            raise InventoryError(
                "Python tooling verified suite did not consume every frozen test binding"
            )
    return result


def _execute_python_tooling_suite(
    suite: unittest.TestSuite,
    suite_contract: _PythonToolingSuiteContract,
    trusted: _PythonUnittestRuntimePrimitives,
) -> _PythonToolingOutcome:
    authorizer = _PythonSkipRuntimeAuthorizer(suite_contract, trusted)
    runtime_integrity_callback = suite_contract.runtime_integrity_callback
    _verify_python_tooling_runtime_integrity(runtime_integrity_callback)

    def authorized_skip(test: unittest.TestCase, reason: str) -> None:
        authorizer.skip_test(test, reason, trusted.getframe(1))

    result_class = authorizer.result_class()
    authorized_result_methods = {
        name: getattr(result_class, name) for name in _PYTHON_RESULT_CALLBACK_NAMES
    }
    original_skip_test = trusted.test_case_skip_test
    original_suite_run = trusted.test_suite_run
    original_call_setup = _trusted_python_unittest_attribute(
        trusted, trusted.test_case_type, "_callSetUp"
    )
    original_call_test_method = _trusted_python_unittest_attribute(
        trusted, trusted.test_case_type, "_callTestMethod"
    )
    original_call_teardown = _trusted_python_unittest_attribute(
        trusted, trusted.test_case_type, "_callTearDown"
    )
    original_call_cleanup = (
        _trusted_python_unittest_attribute(
            trusted, trusted.test_case_type, "_callCleanup"
        )
        if hasattr(trusted.test_case_type, "_callCleanup")
        else None
    )
    original_module_teardown = _trusted_python_unittest_attribute(
        trusted, trusted.test_suite_type, "_handleModuleTearDown"
    )
    bindings_by_object = {
        id(binding.test): binding for binding in suite_contract.discovered_test_bindings
    }
    remaining_binding_ids = set(bindings_by_object)
    runtime_order = suite_contract.runtime_order
    if runtime_order == ():
        runtime_order = tuple(
            binding.runtime_id for binding in suite_contract.discovered_test_bindings
        )
    if type(runtime_order) is not tuple or len(runtime_order) != len(
        bindings_by_object
    ):
        raise InventoryError("Python tooling runtime-order contract changed")
    runtime_cursor = [0]
    fixture_context: list[_PythonTestBinding | None] = [None]
    expected_module_setup_failed = [False]

    def required_binding(test: unittest.TestCase) -> _PythonTestBinding:
        binding = bindings_by_object.get(id(test))
        if binding is None or binding.test is not test:
            raise InventoryError("Python tooling rejected an unknown test object")
        return binding

    expected_case_hooks: dict[str, Any] = {}

    def verified_call_setup(test: unittest.TestCase) -> None:
        binding = required_binding(test)
        _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
        _verify_python_test_bindings((binding,), trusted, expected_case_hooks)
        try:
            if authorizer.has_platform_applicability_skip(test):
                authorizer.skip_platform_applicability(test)
            else:
                original_call_setup(test)
        finally:
            _verify_python_test_bindings((binding,), trusted, expected_case_hooks)
            _verify_python_tooling_runtime_integrity(runtime_integrity_callback)

    def verified_call_test_method(test: unittest.TestCase, method: Any) -> None:
        binding = required_binding(test)
        _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
        _verify_python_test_bindings((binding,), trusted, expected_case_hooks)
        if method != binding.bound_method:
            raise InventoryError("Python tooling test dispatch method changed")
        try:
            original_call_test_method(test, method)
        finally:
            _verify_python_test_bindings((binding,), trusted, expected_case_hooks)
            _verify_python_tooling_runtime_integrity(runtime_integrity_callback)

    def verified_call_teardown(test: unittest.TestCase) -> None:
        binding = required_binding(test)
        _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
        _verify_python_test_bindings((binding,), trusted, expected_case_hooks)
        try:
            original_call_teardown(test)
        finally:
            _verify_python_test_bindings((binding,), trusted, expected_case_hooks)
            _verify_python_tooling_runtime_integrity(runtime_integrity_callback)

    def verified_call_cleanup(
        test: unittest.TestCase, function: Any, *args: Any, **kwargs: Any
    ) -> None:
        if original_call_cleanup is None:
            raise InventoryError("Python tooling cleanup hook is unavailable")
        binding = required_binding(test)
        _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
        _verify_python_test_bindings((binding,), trusted, expected_case_hooks)
        try:
            original_call_cleanup(test, function, *args, **kwargs)
        finally:
            _verify_python_test_bindings((binding,), trusted, expected_case_hooks)
            _verify_python_tooling_runtime_integrity(runtime_integrity_callback)

    expected_case_hooks.update(
        {
            "_callSetUp": verified_call_setup,
            "_callTestMethod": verified_call_test_method,
            "_callTearDown": verified_call_teardown,
        }
    )
    if original_call_cleanup is not None:
        expected_case_hooks["_callCleanup"] = verified_call_cleanup
    expected_suite_helpers: dict[str, Any] = {}

    def verified_module_teardown(
        verified_suite: unittest.TestSuite, result: unittest.TestResult
    ) -> None:
        previous_class = getattr(result, "_previousTestClass", None)
        previous_fixtures: tuple[_PythonFixtureBinding, ...] = ()
        for candidate in bindings_by_object.values():
            if previous_class is candidate.test_class:
                previous_fixtures = candidate.fixtures
                break
        current = fixture_context[0]
        current_fixtures = () if current is None else current.fixtures
        _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
        try:
            _verify_python_module_fixture_transition(
                previous_fixtures,
                current_fixtures,
                lambda: original_module_teardown(verified_suite, result),
            )
        finally:
            _verify_python_tooling_runtime_integrity(runtime_integrity_callback)

    expected_suite_helpers["_handleModuleTearDown"] = verified_module_teardown

    def verified_suite_run(
        verified_suite: unittest.TestSuite,
        result: unittest.TestResult,
        debug: bool = False,
    ) -> unittest.TestResult:
        return _run_verified_python_test_suite(
            verified_suite,
            result,
            bindings_by_object,
            trusted,
            verified_suite_run,
            remaining_binding_ids,
            expected_case_hooks,
            fixture_context,
            expected_module_setup_failed,
            verified_module_teardown,
            runtime_integrity_callback,
            runtime_order,
            runtime_cursor,
            debug,
        )

    platform_decorator_patches = contextlib.ExitStack()
    try:
        _verify_python_unittest_runtime_primitives(
            trusted, original_skip_test, original_suite_run
        )
        _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
        _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
        _verify_python_test_bindings(suite_contract.discovered_test_bindings, trusted)
        for authorization in suite_contract.platform_skip_authorizations:
            binding = required_binding(authorization.test)
            for owner in (binding.test_class, binding.method_descriptor):
                if getattr(owner, "__unittest_skip__", False):
                    platform_decorator_patches.enter_context(
                        mock.patch.object(owner, "__unittest_skip__", False)
                    )
        trusted.test_case_type.skipTest = authorized_skip
        trusted.test_suite_type.run = verified_suite_run
        trusted.test_case_type._callSetUp = verified_call_setup
        trusted.test_case_type._callTestMethod = verified_call_test_method
        trusted.test_case_type._callTearDown = verified_call_teardown
        if original_call_cleanup is not None:
            trusted.test_case_type._callCleanup = verified_call_cleanup
        trusted.test_suite_type._handleModuleTearDown = verified_module_teardown
        _verify_python_unittest_runtime_primitives(
            trusted,
            authorized_skip,
            verified_suite_run,
            expected_case_hooks,
            expected_suite_helpers,
        )
        runner = trusted.runner_type(
            stream=sys.stderr,
            verbosity=1,
            resultclass=result_class,
        )
        result = trusted.runner_run(runner, suite)
        if runtime_cursor != [len(runtime_order)]:
            raise InventoryError(
                "Python tooling verified suite runtime order is incomplete"
            )
        _verify_python_tooling_runtime_integrity(runtime_integrity_callback)
        _verify_python_unittest_runtime_primitives(
            trusted,
            authorized_skip,
            verified_suite_run,
            expected_case_hooks,
            expected_suite_helpers,
        )
        _verify_python_test_bindings(
            suite_contract.discovered_test_bindings, trusted, expected_case_hooks
        )
        _verify_python_tooling_result_integrity(
            result,
            result_class,
            authorized_result_methods,
            authorizer.outcome_ledger,
        )
        authorizer.require_all_tickets_consumed()
        return authorizer.outcome_ledger.outcome(result)
    except InventoryError:
        raise
    except Exception as exc:
        raise InventoryError("Python tooling execution adapter failed") from exc
    finally:
        trusted.test_case_type.skipTest = original_skip_test
        trusted.test_suite_type.run = original_suite_run
        trusted.test_case_type._callSetUp = original_call_setup
        trusted.test_case_type._callTestMethod = original_call_test_method
        trusted.test_case_type._callTearDown = original_call_teardown
        if original_call_cleanup is not None:
            trusted.test_case_type._callCleanup = original_call_cleanup
        trusted.test_suite_type._handleModuleTearDown = original_module_teardown
        platform_decorator_patches.close()
        _verify_python_unittest_runtime_primitives(
            trusted, original_skip_test, original_suite_run
        )


def _require_windows_python_tooling_fixtures(root: Path) -> None:
    for relative_path in WINDOWS_PYTHON_TOOLING_FIXTURE_PATHS:
        fixture_path = root / relative_path
        subject = f"Windows Python tooling executable fixture {relative_path}"
        try:
            path_before = os.stat(fixture_path, follow_symlinks=False)
        except OSError as exc:
            raise InventoryError(f"cannot read {subject}: {fixture_path}") from exc
        if not stat.S_ISREG(path_before.st_mode):
            raise InventoryError(f"{subject} is not a regular file: {fixture_path}")
        fixture = _read_regular_stable_snapshot(
            fixture_path,
            MAX_INVENTORY_BYTES,
            subject,
        )
        if not fixture.bytes:
            raise InventoryError(
                f"Windows Python tooling executable fixture is empty: {relative_path}"
            )
        try:
            path_after = os.stat(fixture_path, follow_symlinks=False)
        except OSError as exc:
            raise InventoryError(
                f"{subject} changed while reading: {fixture_path}"
            ) from exc
        path_before_identity = (
            path_before.st_dev,
            path_before.st_ino,
            path_before.st_size,
            path_before.st_mtime_ns,
            path_before.st_ctime_ns,
        )
        path_after_identity = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if (
            path_before_identity != path_after_identity
            or path_after_identity != fixture.identity
            or path_before.st_mode != path_after.st_mode
            or path_after.st_mode != fixture.mode
        ):
            raise InventoryError(f"{subject} changed while reading: {fixture_path}")


class _WindowsFileTime(ctypes.Structure):
    _fields_ = (("low", ctypes.c_uint32), ("high", ctypes.c_uint32))


class _WindowsByHandleFileInformation(ctypes.Structure):
    _fields_ = (
        ("file_attributes", ctypes.c_uint32),
        ("creation_time", _WindowsFileTime),
        ("last_access_time", _WindowsFileTime),
        ("last_write_time", _WindowsFileTime),
        ("volume_serial_number", ctypes.c_uint32),
        ("file_size_high", ctypes.c_uint32),
        ("file_size_low", ctypes.c_uint32),
        ("number_of_links", ctypes.c_uint32),
        ("file_index_high", ctypes.c_uint32),
        ("file_index_low", ctypes.c_uint32),
    )


class _WindowsFileId128(ctypes.Structure):
    _fields_ = (("identifier", ctypes.c_ubyte * 16),)


class _WindowsFileIdInfo(ctypes.Structure):
    _fields_ = (
        ("volume_serial_number", ctypes.c_uint64),
        ("file_id", _WindowsFileId128),
    )


class _WindowsKernelFunctions(NamedTuple):
    library: Any
    create_file: Any
    close_handle: Any
    get_file_information: Any
    get_file_information_ex: Any
    get_file_size: Any
    set_file_pointer: Any
    read_file: Any
    get_module_filename: Any
    get_proc_address: Any
    get_module_handle_ex: Any


class _WindowsHeldDllSnapshot(NamedTuple):
    file_id: tuple[int, bytes]
    size: int
    digest: str
    number_of_links: int
    file_attributes: int
    file_index: tuple[int, int]


def _configure_windows_function(
    function: Any, argtypes: list[Any], restype: Any
) -> Any:
    try:
        function.argtypes = argtypes
        function.restype = restype
    except (AttributeError, TypeError) as exc:
        raise InventoryError("Windows kernel API binding is noncanonical") from exc
    return function


def _windows_kernel_functions() -> _WindowsKernelFunctions:
    win_dll = getattr(ctypes, "WinDLL", None)
    if not isinstance(win_dll, type):
        raise InventoryError("Windows kernel loader is unavailable")
    try:
        kernel32 = win_dll("kernel32", use_last_error=True)
        create_file = _configure_windows_function(
            kernel32.CreateFileW,
            [
                ctypes.c_wchar_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
            ],
            ctypes.c_void_p,
        )
        close_handle = _configure_windows_function(
            kernel32.CloseHandle, [ctypes.c_void_p], ctypes.c_int
        )
        get_file_information = _configure_windows_function(
            kernel32.GetFileInformationByHandle,
            [ctypes.c_void_p, ctypes.POINTER(_WindowsByHandleFileInformation)],
            ctypes.c_int,
        )
        get_file_information_ex = _configure_windows_function(
            kernel32.GetFileInformationByHandleEx,
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32],
            ctypes.c_int,
        )
        get_file_size = _configure_windows_function(
            kernel32.GetFileSizeEx,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)],
            ctypes.c_int,
        )
        set_file_pointer = _configure_windows_function(
            kernel32.SetFilePointerEx,
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_uint32,
            ],
            ctypes.c_int,
        )
        read_file = _configure_windows_function(
            kernel32.ReadFile,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_void_p,
            ],
            ctypes.c_int,
        )
        get_module_filename = _configure_windows_function(
            kernel32.GetModuleFileNameW,
            [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32],
            ctypes.c_uint32,
        )
        get_proc_address = _configure_windows_function(
            kernel32.GetProcAddress,
            [ctypes.c_void_p, ctypes.c_char_p],
            ctypes.c_void_p,
        )
        get_module_handle_ex = _configure_windows_function(
            kernel32.GetModuleHandleExW,
            [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
            ctypes.c_int,
        )
    except AttributeError as exc:
        raise InventoryError("Windows kernel API is incomplete") from exc
    return _WindowsKernelFunctions(
        kernel32,
        create_file,
        close_handle,
        get_file_information,
        get_file_information_ex,
        get_file_size,
        set_file_pointer,
        read_file,
        get_module_filename,
        get_proc_address,
        get_module_handle_ex,
    )


def _windows_handle_value(value: Any, subject: str) -> int:
    if isinstance(value, ctypes.c_void_p):
        value = value.value
    invalid = ctypes.c_void_p(-1).value
    if type(value) is not int or value <= 0 or value == invalid:
        raise InventoryError(f"{subject} is invalid")
    return value


def _windows_validate_dll_path(path: Path, subject: str) -> None:
    try:
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise InventoryError(f"{subject} path is noncanonical")
        is_junction = getattr(path, "is_junction", None)
        if path.is_symlink() or (is_junction is not None and is_junction()):
            raise InventoryError(f"{subject} path is a filesystem alias")
        path_stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise InventoryError(f"{subject} is unavailable") from exc
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_size <= 0
        or path_stat.st_nlink != 1
        or getattr(path_stat, "st_file_attributes", 0) & 0x00000400
    ):
        raise InventoryError(
            f"{subject} must be a nonempty, unique-link, non-reparse regular file"
        )


def _windows_open_dll_handle(
    functions: _WindowsKernelFunctions, path: Path, subject: str
) -> int:
    _windows_validate_dll_path(path, subject)
    try:
        raw_handle = functions.create_file(
            str(path),
            0x80000000,
            0x00000001,
            None,
            3,
            0x00200080,
            None,
        )
    except Exception as exc:
        raise InventoryError(f"cannot open held {subject}") from exc
    return _windows_handle_value(raw_handle, f"held {subject} handle")


def _windows_file_metadata(
    functions: _WindowsKernelFunctions, handle: int, subject: str
) -> _WindowsHeldDllSnapshot:
    information = _WindowsByHandleFileInformation()
    file_id_information = _WindowsFileIdInfo()
    size_value = ctypes.c_int64()
    try:
        information_ok = functions.get_file_information(
            handle, ctypes.byref(information)
        )
        file_id_ok = functions.get_file_information_ex(
            handle,
            18,
            ctypes.byref(file_id_information),
            ctypes.sizeof(file_id_information),
        )
        size_ok = functions.get_file_size(handle, ctypes.byref(size_value))
    except Exception as exc:
        raise InventoryError(f"cannot inspect held {subject}") from exc
    size_from_information = (int(information.file_size_high) << 32) | int(
        information.file_size_low
    )
    file_index = (int(information.file_index_high), int(information.file_index_low))
    file_id = (
        int(file_id_information.volume_serial_number),
        bytes(file_id_information.file_id.identifier),
    )
    if (
        not information_ok
        or not file_id_ok
        or not size_ok
        or size_value.value <= 0
        or size_value.value > MAX_WINDOWS_PYTHON_TOOLING_DLL_BYTES
        or size_from_information != size_value.value
        or int(information.number_of_links) != 1
        or int(information.file_attributes) & 0x00000400
        or file_id[0] == 0
        or not any(file_id[1])
        or file_index == (0, 0)
    ):
        raise InventoryError(f"held {subject} identity is noncanonical")
    return _WindowsHeldDllSnapshot(
        file_id,
        size_value.value,
        "",
        int(information.number_of_links),
        int(information.file_attributes),
        file_index,
    )


def _windows_file_digest(
    functions: _WindowsKernelFunctions, handle: int, size: int, subject: str
) -> str:
    try:
        if not functions.set_file_pointer(handle, ctypes.c_int64(0), None, 0):
            raise InventoryError(f"cannot rewind held {subject}")
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            requested = min(1024 * 1024, remaining)
            buffer = ctypes.create_string_buffer(requested)
            received = ctypes.c_uint32()
            if not functions.read_file(
                handle, buffer, requested, ctypes.byref(received), None
            ):
                raise InventoryError(f"cannot read held {subject}")
            if received.value <= 0 or received.value > requested:
                raise InventoryError(f"held {subject} ended unexpectedly")
            digest.update(buffer.raw[: received.value])
            remaining -= received.value
        probe = ctypes.create_string_buffer(1)
        probe_received = ctypes.c_uint32()
        if not functions.read_file(
            handle, probe, 1, ctypes.byref(probe_received), None
        ):
            raise InventoryError(f"cannot probe held {subject}")
        if probe_received.value != 0:
            raise InventoryError(f"held {subject} grew while reading")
    except InventoryError:
        raise
    except Exception as exc:
        raise InventoryError(f"cannot digest held {subject}") from exc
    return digest.hexdigest()


def _windows_snapshot_dll_handle(
    functions: _WindowsKernelFunctions, handle: int, subject: str
) -> _WindowsHeldDllSnapshot:
    before = _windows_file_metadata(functions, handle, subject)
    digest = _windows_file_digest(functions, handle, before.size, subject)
    after = _windows_file_metadata(functions, handle, subject)
    if before != after:
        raise InventoryError(f"held {subject} changed while reading")
    return before._replace(digest=digest)


class _WindowsHeldDll:
    def __init__(
        self, functions: _WindowsKernelFunctions, path: Path, subject: str
    ) -> None:
        self.functions = functions
        self.path = path
        self.subject = subject
        self.handle = _windows_open_dll_handle(functions, path, subject)
        try:
            self.snapshot = _windows_snapshot_dll_handle(
                functions, self.handle, subject
            )
        except Exception:
            functions.close_handle(self.handle)
            raise
        self.closed = False

    def verify(self) -> None:
        if self.closed:
            raise InventoryError(f"held {self.subject} closed early")
        _windows_validate_dll_path(self.path, self.subject)
        if (
            _windows_snapshot_dll_handle(self.functions, self.handle, self.subject)
            != self.snapshot
        ):
            raise InventoryError(f"held {self.subject} identity changed")
        probe = _windows_open_dll_handle(self.functions, self.path, self.subject)
        try:
            path_snapshot = _windows_snapshot_dll_handle(
                self.functions, probe, self.subject
            )
        finally:
            if not self.functions.close_handle(probe):
                raise InventoryError(f"cannot close {self.subject} path probe")
        if path_snapshot != self.snapshot:
            raise InventoryError(f"{self.subject} pathname was rebound")

    def close(self) -> None:
        if self.closed:
            raise InventoryError(f"held {self.subject} closed more than once")
        self.closed = True
        if not self.functions.close_handle(self.handle):
            raise InventoryError(f"cannot close held {self.subject}")


class _WindowsPythonToolingDllFiles:
    def __init__(self, installed: _WindowsHeldDll, emitted: _WindowsHeldDll) -> None:
        self.installed = installed
        self.emitted = emitted
        self.verify()

    def verify(self) -> None:
        self.installed.verify()
        self.emitted.verify()
        if (
            self.installed.snapshot.file_id == self.emitted.snapshot.file_id
            or self.installed.snapshot.size != self.emitted.snapshot.size
            or self.installed.snapshot.digest != self.emitted.snapshot.digest
        ):
            raise InventoryError(
                "Windows emitted and installed BLAS DLL identities are noncanonical"
            )


@contextlib.contextmanager
def _held_windows_python_tooling_dlls(
    root: Path,
    emitted_path: Path | None,
    installed_path: Path | None,
) -> Any:
    if emitted_path is None or installed_path is None:
        raise InventoryError(
            "Windows Python tooling requires explicit build-emitted and installed DLLs"
        )
    expected_root = root.resolve(strict=True)
    expected_installed_path = expected_root.joinpath(
        *PurePosixPath(WINDOWS_PYTHON_TOOLING_BLAS_PATH).parts
    )
    installed_path = Path(os.path.abspath(installed_path))
    if installed_path != expected_installed_path:
        raise InventoryError("Windows installed BLAS DLL path is noncanonical")
    installed_path = installed_path.resolve(strict=True)
    if installed_path != expected_installed_path:
        raise InventoryError("Windows installed BLAS DLL path is a filesystem alias")
    emitted_path = Path(os.path.abspath(emitted_path)).resolve(strict=True)
    functions = _windows_kernel_functions()
    emitted = _WindowsHeldDll(functions, emitted_path, "build-emitted BLAS DLL")
    installed: _WindowsHeldDll | None = None
    try:
        installed = _WindowsHeldDll(
            functions, installed_path, "installed canonical BLAS DLL"
        )
        files = _WindowsPythonToolingDllFiles(installed, emitted)
        yield files
        files.verify()
    finally:
        close_error: Exception | None = None
        if installed is not None:
            try:
                installed.close()
            except Exception as exc:
                close_error = exc
        try:
            emitted.close()
        except Exception as exc:
            if close_error is None:
                close_error = exc
        if close_error is not None:
            raise close_error


def _windows_module_path(
    functions: _WindowsKernelFunctions, module_handle: int
) -> Path:
    buffer = ctypes.create_unicode_buffer(32768)
    try:
        length = functions.get_module_filename(module_handle, buffer, len(buffer))
    except Exception as exc:
        raise InventoryError("cannot resolve Windows BLAS module path") from exc
    if type(length) is not int or length <= 0 or length >= len(buffer) - 1:
        raise InventoryError("Windows BLAS module path is noncanonical")
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    try:
        return Path(os.path.abspath(value)).resolve(strict=True)
    except OSError as exc:
        raise InventoryError("Windows BLAS module path is unavailable") from exc


def _windows_verify_module_backing_file(
    functions: _WindowsKernelFunctions,
    module_handle: int,
    installed: _WindowsHeldDll,
) -> Path:
    module_path = _windows_module_path(functions, module_handle)
    if module_path != installed.path:
        raise InventoryError("Windows Python tooling BLAS module path differs")
    probe = _windows_open_dll_handle(
        functions, module_path, "loaded Windows BLAS module file"
    )
    try:
        module_snapshot = _windows_snapshot_dll_handle(
            functions, probe, "loaded Windows BLAS module file"
        )
    finally:
        if not functions.close_handle(probe):
            raise InventoryError("cannot close loaded Windows BLAS module file probe")
    if module_snapshot != installed.snapshot:
        raise InventoryError("Windows BLAS module file identity differs")
    return module_path


def _windows_module_exports(
    functions: _WindowsKernelFunctions, module_handle: int
) -> tuple[tuple[str, int], ...]:
    exports: list[tuple[str, int]] = []
    for symbol in WINDOWS_PYTHON_TOOLING_BLAS_REQUIRED_SYMBOLS:
        try:
            raw_address = functions.get_proc_address(
                module_handle, symbol.encode("ascii")
            )
        except Exception as exc:
            raise InventoryError(
                f"cannot resolve Windows BLAS symbol {symbol}"
            ) from exc
        address = _windows_handle_value(
            raw_address, f"Windows BLAS symbol address {symbol}"
        )
        owner = ctypes.c_void_p()
        try:
            owner_ok = functions.get_module_handle_ex(
                0x00000006,
                ctypes.c_void_p(address),
                ctypes.byref(owner),
            )
        except Exception as exc:
            raise InventoryError(
                f"cannot resolve Windows BLAS symbol owner {symbol}"
            ) from exc
        if (
            not owner_ok
            or _windows_handle_value(owner, f"Windows BLAS symbol owner {symbol}")
            != module_handle
        ):
            raise InventoryError(f"Windows BLAS symbol has foreign owner: {symbol}")
        exports.append((symbol, address))
    return tuple(exports)


class _WindowsCheckerDllProof:
    def __init__(self, files: _WindowsPythonToolingDllFiles) -> None:
        files.verify()
        cdll_type = ctypes.CDLL
        if not isinstance(cdll_type, type):
            raise InventoryError("Windows canonical BLAS loader type is noncanonical")
        try:
            checker_library = cdll_type(
                str(files.installed.path),
                winmode=WINDOWS_PYTHON_TOOLING_BLAS_WINMODE,
            )
        except OSError as exc:
            raise InventoryError("Windows canonical BLAS DLL cannot be loaded") from exc
        if type(checker_library) is not cdll_type:
            raise InventoryError("Windows canonical BLAS loader returned a proxy")
        module_handle = _windows_handle_value(
            getattr(checker_library, "_handle", None),
            "checker-owned Windows BLAS module handle",
        )
        _windows_verify_module_backing_file(
            files.installed.functions, module_handle, files.installed
        )
        exports = _windows_module_exports(files.installed.functions, module_handle)
        self.files = files
        self.cdll_type = cdll_type
        self.checker_library = checker_library
        self.module_handle = module_handle
        self.exports = exports

    def verify(self) -> None:
        self.files.verify()
        if (
            ctypes.CDLL is not self.cdll_type
            or type(self.checker_library) is not self.cdll_type
            or _windows_handle_value(
                getattr(self.checker_library, "_handle", None),
                "checker-owned Windows BLAS module handle",
            )
            != self.module_handle
            or _windows_verify_module_backing_file(
                self.files.installed.functions,
                self.module_handle,
                self.files.installed,
            )
            != self.files.installed.path
            or _windows_module_exports(
                self.files.installed.functions, self.module_handle
            )
            != self.exports
        ):
            raise InventoryError("checker-owned Windows BLAS proof changed")


class _WindowsPythonToolingBlasGuard:
    def __init__(
        self,
        root: Path,
        module_registry: tuple[_PythonSourceModuleBinding, ...],
        files: _WindowsPythonToolingDllFiles,
        checker_proof: _WindowsCheckerDllProof,
    ) -> None:
        _verify_python_source_module_registry(module_registry)
        level2_bindings = [
            binding
            for binding in module_registry
            if PurePosixPath(binding.reviewed.inventory_path).name
            == "test_level2_report.py"
        ]
        if len(level2_bindings) != 1:
            raise InventoryError(
                "Windows Python tooling BLAS module binding is noncanonical"
            )
        binding = level2_bindings[0]
        _verify_python_source_module_binding(binding)
        namespace = binding.namespace
        repository_root = namespace.get("REPO_ROOT")
        runner = namespace.get("runner")
        test_blas = namespace.get("TEST_BLAS")
        test_file_blas = namespace.get("TEST_FILE_BLAS")
        module_library = namespace.get("_TEST_BLAS_LIBRARY")
        try:
            expected_root = root.resolve(strict=True)
            runner_path = runner.default_zynum_blas()
        except (AttributeError, OSError) as exc:
            raise InventoryError(
                "Windows Python tooling BLAS binding is noncanonical"
            ) from exc
        expected_path = expected_root.joinpath(
            *PurePosixPath(WINDOWS_PYTHON_TOOLING_BLAS_PATH).parts
        )
        if (
            checker_proof.files is not files
            or not isinstance(repository_root, Path)
            or repository_root != expected_root
            or type(runner_path) is not str
            or runner_path != WINDOWS_PYTHON_TOOLING_BLAS_PATH
            or type(test_blas) is not str
            or type(test_file_blas) is not str
            or test_blas != str(expected_path)
            or test_file_blas != str(expected_path)
            or namespace.get("WINDOWS_TEST_BLAS_WINMODE")
            != WINDOWS_PYTHON_TOOLING_BLAS_WINMODE
            or namespace.get("WINDOWS_TEST_BLAS_REQUIRED_SYMBOLS")
            != WINDOWS_PYTHON_TOOLING_BLAS_REQUIRED_SYMBOLS
            or type(module_library) is not checker_proof.cdll_type
        ):
            raise InventoryError("Windows Python tooling BLAS binding is noncanonical")
        checker_proof.verify()
        module_handle = _windows_handle_value(
            getattr(module_library, "_handle", None),
            "Python tooling BLAS module handle",
        )
        if module_handle != checker_proof.module_handle:
            raise InventoryError("Windows Python tooling BLAS module handle differs")
        self.binding = binding
        self.module_registry = module_registry
        self.files = files
        self.namespace = namespace
        self.repository_root = repository_root
        self.runner = runner
        self.test_blas = test_blas
        self.test_file_blas = test_file_blas
        self.module_library = module_library
        self.checker_proof = checker_proof
        self.cdll_type = checker_proof.cdll_type
        self.module_handle = module_handle
        self.find_test_blas = namespace.get("find_test_blas")
        self.windows_identity = namespace.get("_windows_test_blas_identity")
        self.windows_snapshot = namespace.get("_windows_file_snapshot")
        self.ctypes_module = namespace.get("ctypes")

    def verify(self) -> None:
        _verify_python_source_module_registry(self.module_registry)
        _verify_python_source_module_binding(self.binding)
        self.checker_proof.verify()
        namespace = self.namespace
        if (
            namespace.get("REPO_ROOT") is not self.repository_root
            or namespace.get("runner") is not self.runner
            or namespace.get("TEST_BLAS") != self.test_blas
            or namespace.get("TEST_FILE_BLAS") != self.test_file_blas
            or namespace.get("_TEST_BLAS_LIBRARY") is not self.module_library
            or namespace.get("WINDOWS_TEST_BLAS_WINMODE")
            != WINDOWS_PYTHON_TOOLING_BLAS_WINMODE
            or namespace.get("WINDOWS_TEST_BLAS_REQUIRED_SYMBOLS")
            != WINDOWS_PYTHON_TOOLING_BLAS_REQUIRED_SYMBOLS
            or namespace.get("find_test_blas") is not self.find_test_blas
            or namespace.get("_windows_test_blas_identity") is not self.windows_identity
            or namespace.get("_windows_file_snapshot") is not self.windows_snapshot
            or namespace.get("ctypes") is not self.ctypes_module
            or self.ctypes_module is not ctypes
            or ctypes.CDLL is not self.cdll_type
            or type(self.module_library) is not self.cdll_type
            or _windows_handle_value(
                getattr(self.module_library, "_handle", None),
                "Python tooling BLAS module handle",
            )
            != self.module_handle
            or self.module_handle != self.checker_proof.module_handle
        ):
            raise InventoryError(
                "Windows Python tooling BLAS binding changed during the suite"
            )


def _windows_python_tooling_blas_identity(
    root: Path,
    module_registry: tuple[_PythonSourceModuleBinding, ...],
    files: _WindowsPythonToolingDllFiles,
    checker_proof: _WindowsCheckerDllProof,
) -> _WindowsPythonToolingBlasGuard:
    return _WindowsPythonToolingBlasGuard(root, module_registry, files, checker_proof)


def _run_python_tooling_root(
    root: Path,
    inventory_path: Path,
    root_id: str,
    windows_zynum_blas_build_output: Path | None = None,
    windows_zynum_blas_installed_output: Path | None = None,
) -> _PythonToolingRootSummary:
    if not root_id:
        raise InventoryError("Python tooling root ID must be nonempty")
    snapshot = _read_regular_stable_snapshot(
        inventory_path, MAX_INVENTORY_BYTES, "test inventory"
    )
    digest_error = _reviewed_inventory_bytes_error(snapshot.bytes)
    if digest_error is not None:
        raise InventoryError(digest_error)
    try:
        inventory = _strict_json_loads(snapshot.bytes.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InventoryError(f"cannot parse strict test inventory JSON: {exc}") from exc
    if not isinstance(inventory, dict):
        raise InventoryError("test inventory root must be an object")
    roots = inventory.get("test_roots")
    if not isinstance(roots, list):
        raise InventoryError("test_roots must be an array")
    matching_roots = [
        row for row in roots if isinstance(row, dict) and row.get("id") == root_id
    ]
    if len(matching_roots) != 1 or not isinstance(
        matching_roots[0].get("module_paths"), list
    ):
        raise InventoryError(f"Python tooling root must resolve uniquely: {root_id!r}")
    tooling_root = matching_roots[0]
    discovery_context = BUILD_CHECKER._make_discovery_context(root, inventory_path)
    execution_closure = _freeze_python_tooling_execution_closure(
        root, discovery_context, tooling_root["module_paths"]
    )
    errors = _validate_inventory_data(
        root,
        inventory_path,
        inventory,
        structure_only=True,
        inventory_bytes=snapshot.bytes,
        filesystem_snapshot=snapshot,
        _context=discovery_context,
    )
    if errors:
        raise InventoryError("test inventory is invalid: " + "; ".join(errors))

    if sys.platform == "win32":
        _require_windows_python_tooling_fixtures(root)

    if (
        tooling_root.get("language") != "python"
        or tooling_root.get("root_kind") != "discovery"
    ):
        raise InventoryError(
            f"Python tooling root must be a Python discovery root: {root_id!r}"
        )
    if (
        tooling_root.get("aggregate_step_observation_id") != AGGREGATE_STEP_ID
        or not isinstance(tooling_root.get("launch_observation_ids"), list)
        or not tooling_root["launch_observation_ids"]
    ):
        raise InventoryError(
            f"Python tooling root must be bound to the canonical aggregate: {root_id!r}"
        )
    if tooling_root.get("matrix_applicable") is not False:
        raise InventoryError(
            f"Python tooling root must not be matrix-applicable: {root_id!r}"
        )

    expected_sets = inventory.get("expected_test_sets")
    if not isinstance(expected_sets, list):
        raise InventoryError("expected_test_sets must be an array")
    matching_sets = [
        row
        for row in expected_sets
        if isinstance(row, dict) and row.get("root_id") == root_id
    ]
    if len(matching_sets) != 1:
        raise InventoryError(
            f"Python tooling expected set must resolve uniquely: {root_id!r}"
        )
    expected_count = matching_sets[0].get("count")
    expected_tests = matching_sets[0].get("tests")
    if type(expected_count) is not int or expected_count <= 0:
        raise InventoryError(
            f"Python tooling expected count must be positive: {root_id!r}"
        )
    if not isinstance(expected_tests, list):
        raise InventoryError(
            f"Python tooling expected tests must be an array: {root_id!r}"
        )
    skip_contracts = inventory.get("python_skip_contracts")
    matching_skip_contracts = (
        [
            contract
            for contract in skip_contracts
            if isinstance(contract, dict) and contract.get("root_id") == root_id
        ]
        if isinstance(skip_contracts, list)
        else []
    )
    if len(matching_skip_contracts) != 1:
        raise InventoryError(
            f"Python tooling skip contract must resolve uniquely: {root_id!r}"
        )
    skip_entries = matching_skip_contracts[0].get("entries")
    if not isinstance(skip_entries, list):
        raise InventoryError(
            f"Python tooling skip contract entries must be an array: {root_id!r}"
        )
    platform_contract_entries = {
        (_unittest_runtime_id(entry["test"]), entry["reason"]): entry["predicate_id"]
        for entry in skip_entries
        if isinstance(entry, dict)
        and entry.get("predicate_id") in PYTHON_INVENTORY_PLATFORM_PREDICATE_IDS
    }

    discovery_start = tooling_root.get("discovery_start")
    discovery_pattern = tooling_root.get("discovery_pattern")
    if not isinstance(discovery_start, str) or not isinstance(discovery_pattern, str):
        raise InventoryError(
            f"Python tooling discovery metadata is missing: {root_id!r}"
        )
    relative_start = PurePosixPath(discovery_start)
    if relative_start.is_absolute() or ".." in relative_start.parts:
        raise InventoryError(
            f"Python tooling discovery start escapes the repository: {root_id!r}"
        )
    working_directory = Path(os.path.abspath(os.getcwd()))
    if working_directory != root:
        raise InventoryError(
            "Python tooling execution requires the repository root as cwd: "
            f"expected={root}; observed={working_directory}"
        )
    trusted = _capture_python_unittest_runtime_primitives()
    execute_suite = _execute_python_tooling_suite
    _verify_python_unittest_runtime_primitives(trusted, trusted.test_case_skip_test)
    execution_imports = _python_tooling_execution_imports(execution_closure)
    execution_imports.__enter__()
    try:
        _, _, dynamic_sites, reviewed_modules = _python_tooling_source_skip_review(
            root,
            tooling_root["module_paths"],
            discovery_start,
            discovery_pattern,
            _closure=execution_closure,
        )
        windows_files_context = (
            _held_windows_python_tooling_dlls(
                root,
                windows_zynum_blas_build_output,
                windows_zynum_blas_installed_output,
            )
            if sys.platform == "win32"
            else contextlib.nullcontext(None)
        )
        with windows_files_context as windows_files:
            windows_checker_proof = (
                _WindowsCheckerDllProof(windows_files)
                if sys.platform == "win32"
                else None
            )
            _load_python_tooling_closure_dependencies(
                execution_closure,
                {reviewed.module_name for reviewed in reviewed_modules},
            )
            with _registered_python_tooling_modules(
                reviewed_modules
            ) as module_registry:
                windows_blas_guard = (
                    _windows_python_tooling_blas_identity(
                        root,
                        module_registry,
                        windows_files,
                        windows_checker_proof,
                    )
                    if sys.platform == "win32"
                    else None
                )
                loader = trusted.loader_type()
                _verify_python_test_loader_instance(loader, trusted)
                loader_suite = trusted.loader_suite_class(
                    tuple(
                        trusted.loader_load_tests_from_module(loader, binding.module)
                        for binding in module_registry
                    )
                )
                loaded_tests = _flatten_unittest_suite(loader_suite, trusted)
                loaded_runtime_order = tuple(
                    _canonical_python_test_id(
                        type(test), object.__getattribute__(test, "_testMethodName")
                    )
                    for test in loaded_tests
                )
                runtime_order = _python_tooling_runtime_order_projection(
                    reviewed_modules
                )
                if loaded_runtime_order != runtime_order:
                    raise InventoryError(
                        "Python tooling loader output differs from runtime-order projection"
                    )
                suite = loader_suite
                execution_closure.verify(require_complete=True)
                execution_closure.live_recheck()
                execution_capsule = _python_tooling_execution_capsule(execution_closure)
                capsule_runtime = _python_tooling_capsule_runtime(
                    execution_closure, execution_capsule
                )
                capsule_launcher = capsule_runtime.__enter__()

                def runtime_integrity() -> None:
                    execution_closure.verify(require_complete=True)
                    execution_closure.live_recheck()
                    capsule_launcher.verify()
                    if windows_blas_guard is not None:
                        windows_blas_guard.verify()

                try:
                    if windows_blas_guard is not None:
                        windows_blas_guard.verify()
                    _verify_python_source_module_registry(module_registry)
                    _verify_python_unittest_runtime_primitives(
                        trusted, trusted.test_case_skip_test
                    )
                    _verify_python_test_loader_instance(loader, trusted)
                    suite_contract = _python_tooling_suite_contract(
                        expected_tests,
                        matching_skip_contracts[0],
                        suite,
                        dynamic_sites,
                        module_registry,
                        trusted,
                        runtime_integrity,
                    )
                    discovered_count = suite_contract.discovered_count
                    if windows_blas_guard is not None:
                        windows_blas_guard.verify()
                    try:
                        outcome = execute_suite(suite, suite_contract, trusted)
                    finally:
                        if windows_blas_guard is not None:
                            windows_blas_guard.verify()
                finally:
                    capsule_runtime.__exit__(*sys.exc_info())
    except InventoryError:
        raise
    except Exception as exc:
        raise InventoryError(f"Python tooling discovery failed: {root_id!r}") from exc
    finally:
        execution_imports.__exit__(*sys.exc_info())
    expected_skips = (
        suite_contract.required_decorator_skips
        | suite_contract.permitted_dynamic_skips
        | suite_contract.platform_skips
    )
    if (
        type(outcome) is not _PythonToolingOutcome
        or type(discovered_count) is not int
        or type(outcome.executed) is not int
        or type(outcome.successful) is not bool
        or type(outcome.skips) is not frozenset
        or any(
            type(count) is not int or count < 0
            for count in (
                outcome.failures,
                outcome.errors,
                outcome.expected_failures,
                outcome.unexpected_successes,
            )
        )
    ):
        raise InventoryError("Python tooling execution outcome is noncanonical")
    unexpected_skips = outcome.skips - expected_skips
    missing_decorator_skips = suite_contract.required_decorator_skips - outcome.skips
    missing_dynamic_skips = suite_contract.permitted_dynamic_skips - outcome.skips
    missing_platform_skips = suite_contract.platform_skips - outcome.skips
    platform_contract_skips = frozenset(platform_contract_entries)
    expected_platform_skips = expected_skips & platform_contract_skips
    observed_platform_skips = outcome.skips & platform_contract_skips
    if (
        discovered_count != expected_count
        or outcome.executed != expected_count
        or not outcome.successful
        or outcome.failures != 0
        or outcome.errors != 0
        or outcome.expected_failures != 0
        or outcome.unexpected_successes != 0
        or unexpected_skips
        or missing_decorator_skips
        or missing_dynamic_skips
        or missing_platform_skips
        or observed_platform_skips != expected_platform_skips
    ):
        raise InventoryError(
            "Python tooling test contract failed: "
            f"root={root_id!r}; expected={expected_count}; "
            f"discovered={discovered_count!r}; executed={outcome.executed!r}; "
            f"successful={outcome.successful!r}; failures={outcome.failures!r}; "
            f"errors={outcome.errors!r}; expected_failures="
            f"{outcome.expected_failures!r}; unexpected_successes="
            f"{outcome.unexpected_successes!r}; unexpected_skips="
            f"{sorted(unexpected_skips)!r}; missing_decorator_skips="
            f"{sorted(missing_decorator_skips)!r}; missing_dynamic_skips="
            f"{sorted(missing_dynamic_skips)!r}; missing_platform_skips="
            f"{sorted(missing_platform_skips)!r}; expected_platform_skips="
            f"{sorted(expected_platform_skips)!r}; observed_platform_skips="
            f"{sorted(observed_platform_skips)!r}"
        )
    artifact_platform_skips = sum(
        platform_contract_entries[pair]
        == "python-skip-predicate:artifact-snapshot-platform-unavailable"
        for pair in observed_platform_skips
    )
    publication_platform_skips = sum(
        platform_contract_entries[pair]
        == "python-skip-predicate:report-publication-platform-unavailable"
        for pair in observed_platform_skips
    )
    non_platform_skips = len(outcome.skips - observed_platform_skips)
    if sys.platform == "win32" and (
        artifact_platform_skips
        != PYTHON_INVENTORY_PLATFORM_COUNTS[
            "python-skip-predicate:artifact-snapshot-platform-unavailable"
        ]
        or publication_platform_skips
        != PYTHON_INVENTORY_PLATFORM_COUNTS[
            "python-skip-predicate:report-publication-platform-unavailable"
        ]
        or len(observed_platform_skips)
        != sum(PYTHON_INVENTORY_PLATFORM_COUNTS.values())
        or non_platform_skips != WINDOWS_PYTHON_TOOLING_EXPECTED_NON_PLATFORM_SKIPS
        or len(outcome.skips) != WINDOWS_PYTHON_TOOLING_EXPECTED_TOTAL_SKIPS
    ):
        raise InventoryError(
            "Windows Python tooling skip summary differs from the exact contract: "
            f"non_platform_skips={non_platform_skips}; "
            f"artifact_platform_skips={artifact_platform_skips}; "
            f"publication_platform_skips={publication_platform_skips}; "
            f"platform_skips={len(observed_platform_skips)}; "
            f"skips={len(outcome.skips)}"
        )
    return _PythonToolingRootSummary(
        expected=expected_count,
        discovered=discovered_count,
        outcome=outcome,
        dynamic_skips=non_platform_skips,
        artifact_platform_skips=artifact_platform_skips,
        publication_platform_skips=publication_platform_skips,
        platform_skips=len(observed_platform_skips),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--inventory", type=Path)
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="validate relational closure while allowing requires-native rows",
    )
    parser.add_argument(
        "--require-current-only",
        action="store_true",
        help="require both reviewed NEXT digest slots to be empty",
    )
    parser.add_argument(
        "--refresh-from-protocol",
        action="store_true",
        help="refresh compiler-enumerated expected sets from protocol logs",
    )
    parser.add_argument(
        "--protocol-log",
        action="append",
        default=[],
        metavar="ENVIRONMENT_ID=PATH",
        help="bind one native protocol log to an environment; repeat per mode",
    )
    parser.add_argument(
        "--run-python-tooling-root",
        metavar="ROOT_ID",
        help="run one pinned inventory-declared Python discovery root",
    )
    parser.add_argument(
        "--windows-zynum-blas-build-output",
        type=Path,
        metavar="PATH",
        help="bind the Windows build-emitted DLL to its installed canonical copy",
    )
    parser.add_argument(
        "--windows-zynum-blas-installed-output",
        type=Path,
        metavar="PATH",
        help="bind the explicit installed canonical Windows DLL",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(os.path.abspath(args.root))
    inventory = (
        Path(os.path.abspath(args.inventory))
        if args.inventory
        else root / INVENTORY_PATH
    )
    windows_build_output = args.windows_zynum_blas_build_output
    windows_installed_output = args.windows_zynum_blas_installed_output
    if (
        windows_build_output is not None or windows_installed_output is not None
    ) and args.run_python_tooling_root is None:
        print(
            "test inventory error: Windows BLAS output options require "
            "--run-python-tooling-root",
            file=sys.stderr,
        )
        return 2
    if args.run_python_tooling_root is not None:
        incompatible = [
            option
            for enabled, option in (
                (args.structure_only, "--structure-only"),
                (args.require_current_only, "--require-current-only"),
                (args.refresh_from_protocol, "--refresh-from-protocol"),
                (bool(args.protocol_log), "--protocol-log"),
            )
            if enabled
        ]
        if incompatible:
            print(
                "test inventory error: --run-python-tooling-root is incompatible "
                "with " + ", ".join(incompatible),
                file=sys.stderr,
            )
            return 2
        if (
            windows_build_output is not None or windows_installed_output is not None
        ) and sys.platform != "win32":
            print(
                "test inventory error: Windows BLAS output options are Windows-only",
                file=sys.stderr,
            )
            return 2
        if (
            sys.platform == "win32"
            and args.run_python_tooling_root == PYTHON_TOOLING_ROOT_ID
        ):
            if windows_build_output is None or windows_installed_output is None:
                print(
                    "test inventory error: Windows Python tooling requires paired "
                    "--windows-zynum-blas-build-output and "
                    "--windows-zynum-blas-installed-output",
                    file=sys.stderr,
                )
                return 2
        elif windows_build_output is not None or windows_installed_output is not None:
            print(
                "test inventory error: Windows BLAS output options require the "
                "canonical Python tooling root",
                file=sys.stderr,
            )
            return 2
        try:
            summary = _run_python_tooling_root(
                root,
                inventory,
                args.run_python_tooling_root,
                (
                    Path(os.path.abspath(windows_build_output))
                    if windows_build_output is not None
                    else None
                ),
                (
                    Path(os.path.abspath(windows_installed_output))
                    if windows_installed_output is not None
                    else None
                ),
            )
        except (
            InventoryError,
            BUILD_CHECKER.InventoryError,
            OSError,
            UnicodeError,
            ValueError,
            RecursionError,
        ) as exc:
            print(
                f"test inventory error: Python tooling root failed: {exc}",
                file=sys.stderr,
            )
            return 1
        print(
            "Python tooling root passed: "
            f"{args.run_python_tooling_root}; expected={summary.expected}; "
            f"discovered={summary.discovered}; executed={summary.outcome.executed}; "
            f"skips={len(summary.outcome.skips)}; "
            f"artifact_platform_skips={summary.artifact_platform_skips}; "
            f"publication_platform_skips={summary.publication_platform_skips}; "
            f"platform_skips={summary.platform_skips}; "
            f"failures={summary.outcome.failures}; errors={summary.outcome.errors}; "
            f"expected_failures={summary.outcome.expected_failures}; "
            f"unexpected_successes={summary.outcome.unexpected_successes}"
        )
        return 0
    if args.protocol_log and not args.refresh_from_protocol:
        print(
            "test inventory error: --protocol-log requires --refresh-from-protocol",
            file=sys.stderr,
        )
        return 2
    validated_inventory: dict[str, Any] | None = None
    incomplete = 0
    if args.refresh_from_protocol:
        if not args.protocol_log:
            print(
                "test inventory error: refresh requires at least one --protocol-log",
                file=sys.stderr,
            )
            return 2
        bindings: list[tuple[str, Path]] = []
        for value in args.protocol_log:
            environment_id, separator, raw_path = value.partition("=")
            if not separator or not environment_id or not raw_path:
                print(
                    "test inventory error: protocol binding must be ENVIRONMENT_ID=PATH",
                    file=sys.stderr,
                )
                return 2
            bindings.append((environment_id, Path(os.path.abspath(raw_path))))
        try:
            candidate = refresh_from_protocol(root, inventory, bindings)
            if args.require_current_only:
                slots_error = _current_only_slots_error()
                if slots_error is not None:
                    raise InventoryError(slots_error)
                digest_error = _reviewed_inventory_bytes_error(
                    candidate.bytes, require_current_only=True
                )
                if digest_error is not None:
                    raise InventoryError(digest_error)
                native_error = _reviewed_native_projection_error(
                    candidate.inventory, require_current_only=True
                )
                if native_error is not None:
                    raise InventoryError(native_error)
            _publish_inventory_atomic(
                inventory, candidate.bytes, candidate.expected_snapshot
            )
            validated_inventory = candidate.inventory
            incomplete = candidate.incomplete_count
        except InventoryPublicationIndeterminate as exc:
            print(
                f"test inventory error: {exc}",
                file=sys.stderr,
            )
            return 3
        except (
            InventoryError,
            BUILD_CHECKER.InventoryError,
            OSError,
            UnicodeError,
            ValueError,
            RecursionError,
        ) as exc:
            print(f"test inventory error: refresh failed: {exc}", file=sys.stderr)
            return 1
    if validated_inventory is None:
        try:
            snapshot = _read_regular_stable_snapshot(
                inventory, MAX_INVENTORY_BYTES, "test inventory"
            )
            if args.require_current_only:
                slots_error = _current_only_slots_error()
                if slots_error is not None:
                    raise InventoryError(slots_error)
            digest_error = _reviewed_inventory_bytes_error(
                snapshot.bytes, require_current_only=args.require_current_only
            )
            if digest_error is not None:
                raise InventoryError(digest_error)
            parsed = _strict_json_loads(snapshot.bytes.decode("utf-8"))
        except (
            InventoryError,
            OSError,
            UnicodeError,
            ValueError,
            RecursionError,
        ) as exc:
            errors = [f"cannot read test inventory: {exc}"]
        else:
            errors = validate(
                root,
                inventory,
                structure_only=args.structure_only,
                _inventory=parsed if isinstance(parsed, dict) else None,
                _inventory_bytes=snapshot.bytes,
                _filesystem_snapshot=snapshot,
                require_current_only=args.require_current_only,
            )
            if isinstance(parsed, dict):
                validated_inventory = parsed
                incomplete = _matrix_incomplete_count(parsed)
    else:
        errors = []
        if incomplete and not args.structure_only:
            errors.append(
                f"matrix incomplete: {incomplete} rows require native compiler enumeration"
            )
    if errors:
        for error in errors:
            print(f"test inventory error: {error}", file=sys.stderr)
        return 1
    if args.structure_only:
        print(
            f"test inventory structure valid: {inventory}; "
            f"matrix incomplete: {incomplete} rows"
        )
    else:
        print(f"test inventory valid: {inventory}; matrix incomplete: 0 rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
