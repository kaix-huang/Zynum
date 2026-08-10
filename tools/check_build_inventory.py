#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fail-closed validation for the public build inventory schema v3."""

from __future__ import annotations

import argparse
import ast
import bisect
import copy
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shlex
import stat
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import date
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterator, NamedTuple

AST_TYPE_ALIAS_NODE = getattr(ast, "TypeAlias", None)
AST_TYPE_ALIAS_TYPES = (
    (AST_TYPE_ALIAS_NODE,) if isinstance(AST_TYPE_ALIAS_NODE, type) else ()
)
AST_TRY_STAR_NODE = getattr(ast, "TryStar", None)
AST_TRY_STAR_TYPES = (AST_TRY_STAR_NODE,) if isinstance(AST_TRY_STAR_NODE, type) else ()


def _is_ast_type_alias(node: ast.AST) -> bool:
    return bool(AST_TYPE_ALIAS_TYPES) and isinstance(node, AST_TYPE_ALIAS_TYPES)


_REPOSITORY_GIT_MODULE_NAME = "_zynum_inventory_repository_git"
_repository_git_spec = importlib.util.spec_from_file_location(
    _REPOSITORY_GIT_MODULE_NAME, Path(__file__).with_name("repository_git.py")
)
if _repository_git_spec is None or _repository_git_spec.loader is None:
    raise RuntimeError("cannot load the repository Git policy module")
repository_git = importlib.util.module_from_spec(_repository_git_spec)
sys.modules[_REPOSITORY_GIT_MODULE_NAME] = repository_git
_repository_git_spec.loader.exec_module(repository_git)

_REPOSITORY_SNAPSHOT_MODULE_NAME = "_zynum_inventory_repository_snapshot"
_repository_snapshot_spec = importlib.util.spec_from_file_location(
    _REPOSITORY_SNAPSHOT_MODULE_NAME, Path(__file__).with_name("repository_snapshot.py")
)
if _repository_snapshot_spec is None or _repository_snapshot_spec.loader is None:
    raise RuntimeError("cannot load the repository snapshot module")
repository_snapshot = importlib.util.module_from_spec(_repository_snapshot_spec)
sys.modules[_REPOSITORY_SNAPSHOT_MODULE_NAME] = repository_snapshot
_repository_snapshot_spec.loader.exec_module(repository_snapshot)

SCHEMA_VERSION = 3
INVENTORY_SCOPE = (
    "Current repository build composition, independently discovered build roots, "
    "normalized Python launch occurrences, reviewed public-file provenance, and "
    "host/requested-target payload identities. Detailed subprocess process-bound "
    "semantics are outside this inventory's declared scope."
)
INVENTORY_TOP_LEVEL_KEYS = {
    "schema_version",
    "schema_id",
    "scope",
    "owner_vocabulary",
    "build_roots",
    "build_manifests",
    "build_root_digests",
    "option_surfaces",
    "build_observations",
    "conditional_link_guard_digests",
    "python_launches",
    "workflow_launches",
    "workflow_source_digests",
    "generator_targets",
    "repository_file_classifications",
    "repository_file_classifications_digest",
    "derived_candidates",
    "current_gaps",
}
BUILD_CALLS = {
    "b.option": "option",
    "b.step": "step",
    "b.addFail": "step",
    "b.addLibrary": "compile",
    "b.addExecutable": "compile",
    "b.addTest": "compile",
    "b.addRunArtifact": "launch",
    "b.addSystemCommand": "launch",
    "linkLibrary": "link",
    "b.installArtifact": "install",
    "b.addInstallArtifact": "install",
    "b.installFile": "install",
}
PYTHON_PROCESS_CALLS = {
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
    "os.system",
    "os.popen",
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.startfile",
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    "asyncio.loop.subprocess_exec",
    "asyncio.loop.subprocess_shell",
    "pty.fork",
    "pty.spawn",
    "multiprocessing.process.start",
    "multiprocessing.Manager",
    "multiprocessing.Pool",
    "multiprocessing.pool.Pool",
    "multiprocessing.context.Pool",
    "multiprocessing.manager.start",
    "concurrent.futures.process_pool.submit",
    "concurrent.futures.process_pool.map",
}
PYTHON_PROCESS_CALL_ALIASES = {
    "asyncio.subprocess.create_subprocess_exec": "asyncio.create_subprocess_exec",
    "asyncio.subprocess.create_subprocess_shell": "asyncio.create_subprocess_shell",
    "posix.system": "os.system",
    "posix.fork": "os.fork",
    "posix.forkpty": "os.forkpty",
    "posix.posix_spawn": "os.posix_spawn",
    "posix.posix_spawnp": "os.posix_spawnp",
    "posix.execv": "os.execv",
    "posix.execve": "os.execve",
}
PYTHON_PROCESS_MODULES = {
    "subprocess",
    "os",
    "asyncio",
    "asyncio.subprocess",
    "pty",
    "posix",
    "multiprocessing",
    "multiprocessing.context",
    "multiprocessing.pool",
    "multiprocessing.managers",
    "concurrent",
    "concurrent.futures",
    "concurrent.futures.process",
}
PYTHON_PROCESS_FACTORIES = {
    "asyncio.get_event_loop": "asyncio.loop",
    "asyncio.get_running_loop": "asyncio.loop",
    "asyncio.new_event_loop": "asyncio.loop",
    "asyncio.get_event_loop_policy": "asyncio.policy",
    "asyncio.policy.get_event_loop": "asyncio.loop",
    "asyncio.Runner": "asyncio.runner",
    "asyncio.runner.get_loop": "asyncio.loop",
    "asyncio.SelectorEventLoop": "asyncio.loop",
    "asyncio.ProactorEventLoop": "asyncio.loop",
    "asyncio.EventLoop": "asyncio.loop",
    "multiprocessing.Process": "multiprocessing.process",
    "multiprocessing.context.Process": "multiprocessing.process",
    "multiprocessing.context.SpawnProcess": "multiprocessing.process",
    "multiprocessing.context.ForkProcess": "multiprocessing.process",
    "multiprocessing.context.ForkServerProcess": "multiprocessing.process",
    "multiprocessing.context.SpawnContext": "multiprocessing.context",
    "multiprocessing.context.ForkContext": "multiprocessing.context",
    "multiprocessing.context.ForkServerContext": "multiprocessing.context",
    "multiprocessing.get_context": "multiprocessing.context",
    "multiprocessing.managers.BaseManager": "multiprocessing.manager",
    "multiprocessing.managers.SyncManager": "multiprocessing.manager",
    "multiprocessing.managers.SharedMemoryManager": "multiprocessing.manager",
    "concurrent.futures.ProcessPoolExecutor": "concurrent.futures.process_pool",
    "concurrent.futures.process.ProcessPoolExecutor": "concurrent.futures.process_pool",
}
PYTHON_PROCESS_FACTORY_RESULTS = set(PYTHON_PROCESS_FACTORIES.values())
PYTHON_SHADOWED_ALIAS = "<shadowed>"
PYTHON_IMPORT_HELPERS = {
    "importlib.import_module",
    "importlib.__import__",
    "__import__",
    "builtins.__import__",
    "__builtins__.__import__",
}
PYTHON_DYNAMIC_CODE_CALLS = {
    "eval",
    "exec",
    "builtins.eval",
    "builtins.exec",
    "__builtins__.eval",
    "__builtins__.exec",
}
PYTHON_IMPORT_NAMESPACES = {"importlib", "builtins", "sys"}
PYTHON_NAMESPACE_PRODUCERS = {
    "globals",
    "locals",
    "vars",
    "builtins.globals",
    "builtins.locals",
    "builtins.vars",
    "__builtins__.globals",
    "__builtins__.locals",
    "__builtins__.vars",
}
PYTHON_GLOBALS_NAMESPACE = "<globals-namespace>"
PYTHON_SYS_MODULES_NAMESPACE = "sys.modules"
PYTHON_REVIEWED_ADAPTER_MODULE = "reviewed:tools.observe_abi_baseline"
PYTHON_REVIEWED_ADAPTER_CALL = f"{PYTHON_REVIEWED_ADAPTER_MODULE}.run_command"
PYTHON_NAMESPACE_ACCESSORS = {
    *(
        f"{namespace}.__dict__.get"
        for namespace in (PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES)
    ),
    *(
        f"{namespace}.__getattribute__"
        for namespace in (PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES)
    ),
    *(
        f"{namespace}.__dict__.__getitem__"
        for namespace in (PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES)
    ),
    f"{PYTHON_GLOBALS_NAMESPACE}.get",
    f"{PYTHON_GLOBALS_NAMESPACE}.__getitem__",
    f"{PYTHON_GLOBALS_NAMESPACE}.__getattribute__",
    "__builtins__.__dict__.get",
    "__builtins__.__dict__.__getitem__",
    "__builtins__.get",
    "__builtins__.__getattribute__",
    f"{PYTHON_SYS_MODULES_NAMESPACE}.get",
    f"{PYTHON_SYS_MODULES_NAMESPACE}.__getitem__",
}
PYTHON_NAMESPACE_MUTATIONS = {
    f"{namespace}.__dict__.{operation}"
    for namespace in (PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES)
    for operation in {
        "clear",
        "pop",
        "popitem",
        "setdefault",
        "update",
        "__delitem__",
        "__setitem__",
    }
}
PYTHON_NAMESPACE_MUTATIONS.update(
    f"{PYTHON_GLOBALS_NAMESPACE}.{operation}"
    for operation in {
        "clear",
        "pop",
        "popitem",
        "setdefault",
        "update",
        "__delitem__",
        "__setitem__",
    }
)
PYTHON_NAMESPACE_MUTATIONS.update(
    f"{PYTHON_SYS_MODULES_NAMESPACE}.{operation}"
    for operation in {
        "clear",
        "pop",
        "popitem",
        "setdefault",
        "update",
        "__delitem__",
        "__setitem__",
    }
)
PYTHON_TRACKED_ALIAS_VALUES = (
    PYTHON_PROCESS_CALLS
    | set(PYTHON_PROCESS_CALL_ALIASES)
    | PYTHON_PROCESS_MODULES
    | set(PYTHON_PROCESS_FACTORIES)
    | PYTHON_PROCESS_FACTORY_RESULTS
    | PYTHON_IMPORT_HELPERS
    | PYTHON_DYNAMIC_CODE_CALLS
    | PYTHON_IMPORT_NAMESPACES
    | PYTHON_NAMESPACE_PRODUCERS
    | {
        "__builtins__",
        "__builtins__.__dict__",
        "builtins.__dict__",
        PYTHON_GLOBALS_NAMESPACE,
        PYTHON_SYS_MODULES_NAMESPACE,
        PYTHON_REVIEWED_ADAPTER_MODULE,
        PYTHON_REVIEWED_ADAPTER_CALL,
    }
    | PYTHON_NAMESPACE_ACCESSORS
    | PYTHON_NAMESPACE_MUTATIONS
)
OWNER_VOCABULARY = {
    "abi-compatibility",
    "benchmark-maintainers",
    "build-composition",
    "documentation-maintainers",
    "kernel-coverage",
    "package-metadata",
    "release-validation",
    "test-infrastructure",
    "library-source",
    "public-api",
    "example-source",
    "workflow-maintainers",
    "developer-tooling",
    "project-governance",
}
DERIVED_CLASSES = {
    "designated-reproducible-contract-reference",
    "curated-documentation-asset",
    "non-generated-source",
}
PLATFORMS = {"host", "requested-target", "not-executable"}
BUILD_ROOT_EXCLUDED_PARTS = {
    ".git",
    ".zig-cache",
    "zig-cache",
    "zig-out",
    ".local-docs",
    "__pycache__",
}
PYTHON_SEMANTICS_VERSION = "python-call-semantics-v2"
PYTHON_CARRIER_SEMANTICS_VERSION = "python-carrier-semantics-v3"
EMBEDDED_PYTHON_MAX_DEPTH = 4
EMBEDDED_PYTHON_MAX_BYTES = 32 * 1024
EMBEDDED_PYTHON_MAX_NODES = 8_000
EMBEDDED_PYTHON_MAX_UNITS = 256
STATIC_PYTHON_ARGV_MAX_DEPTH = 32
STATIC_PYTHON_ARGV_MAX_ITEMS = 4_096
CARRIER_FREEZE_MAX_NODES = EMBEDDED_PYTHON_MAX_NODES
CARRIER_FREEZE_MAX_DEPTH = STATIC_PYTHON_ARGV_MAX_DEPTH
CARRIER_FREEZE_MAX_ITEMS = STATIC_PYTHON_ARGV_MAX_ITEMS
CARRIER_FREEZE_MAX_BYTES = 2 * EMBEDDED_PYTHON_MAX_BYTES
SNAPSHOT_MAX_REGULAR_FILE_BYTES = 256 * 1024 * 1024
SNAPSHOT_MAX_REGULAR_ROUND_BYTES = 2 * 1024 * 1024 * 1024
SNAPSHOT_MAX_CACHED_FILE_BYTES = 8 * 1024 * 1024
SNAPSHOT_MAX_CACHED_TOTAL_BYTES = 64 * 1024 * 1024
SNAPSHOT_MAX_DIRECTORIES = 100_000
SNAPSHOT_MAX_ENTRIES = 1_000_000
SNAPSHOT_MAX_DEPTH = 128
SNAPSHOT_MAX_TOTAL_NAME_BYTES = 64 * 1024 * 1024
SNAPSHOT_MAX_TOTAL_STRUCTURE_BYTES = 256 * 1024 * 1024
SNAPSHOT_GENERATOR_PATHS = frozenset(
    {
        "tools/generate_compat_headers.zig",
        "tools/generate_kernel_coverage.zig",
    }
)
DEPENDABOT_CONFIG_PATH = ".github/dependabot.yml"
REVIEWED_DEPENDABOT_CONFIG_SHA256 = (
    "8cc45e0a71d5c86f69270791a53e239d2e72b59d46bad459cb57bfefdc5a2255"
)
SOURCE_REFRESH_MAX_BYTES = SNAPSHOT_MAX_CACHED_FILE_BYTES
SOURCE_REFRESH_MAX_JSON_DEPTH = 128
SOURCE_REFRESH_MAX_JSON_NODES = 262_144
SOURCE_REFRESH_MAX_COLLECTION_ITEMS = 262_144
SOURCE_PROJECTION_SCHEMA_ID = "zynum-reviewed-build-source-projection-v1"
SOURCE_PROJECTION_SCHEMA_VERSION = 1
SOURCE_PROJECTION_FIELDS = (
    "build_roots",
    "build_manifests",
    "build_observations",
    "python_launches",
    "workflow_launches",
    "generator_targets",
    "build_root_digests",
    "conditional_link_guard_digests",
    "workflow_source_digests",
)
CURRENT_SOURCE_PROJECTION_SHA256 = (
    "b844b6c75c0f83feeb714bdb883655a60ed21b7d4596fc086e7737d2537816d7"
)
NEXT_SOURCE_PROJECTION_SHA256: str | None = None
REVIEWED_TEST_INVENTORY_LOADER_CONTRACT_SHA256 = (
    "13271a04270843b905f50070b8848f4a6ddc06fdec5b83981ab586f892a7f75f"
)
REVIEWED_TEST_INVENTORY_BOOTSTRAP_SHA256 = (
    "83b807444228d772b60a7b1c4b140d356d8b580fb7842fc01cf99490573b033e"
)
REVIEWED_TEST_INVENTORY_CAPSULE_LAUNCHER_SHA256 = (
    "094f9c5d4bcb0cde833acb2172dad3f9702579058aea99831a5e479f5889aba4"
)
REVIEWED_TEST_INVENTORY_CAPSULE_RUNTIME_SHA256 = (
    "91eaeca9adeecdcd7a3c1d144bd71ba9d027f0c4d8fd3b13d94ddb9ae92cdd00"
)
REVIEWED_TEST_INVENTORY_POSIX_CAPSULE_PROBE_SHA256 = (
    "100123e9c3e9cb964c740ff2afa2d348e5124e1b438324366e5dcc526282449f"
)
REVIEWED_TEST_INVENTORY_LOADER_FUNCTION_NAMES = (
    "_load_build_checker",
    "_reviewed_python_source_module",
    "_reviewed_python_tooling_source_modules",
    "_freeze_python_tooling_execution_closure",
    "_load_python_tooling_closure_dependencies",
    "_python_path_is_within",
    "_python_tooling_path_identity",
    "_python_tooling_path_touches_repository",
    "_python_tooling_live_repo_import_candidate",
    "_python_frozen_module_spec",
    "_python_tooling_execution_imports",
    "_python_tooling_source_skip_review",
    "_verify_python_reviewed_source_module",
    "_verify_python_source_module_binding",
    "_verify_python_source_module_registry",
    "_registered_python_tooling_modules",
    "_python_tooling_suite_contract",
    "_run_python_tooling_root",
)
REVIEWED_TEST_INVENTORY_LOADER_CLASS_NAMES = (
    "_PythonExecutionClosure",
    "_PythonFrozenLoader",
    "_PythonFrozenFinder",
)


class PayloadControllerBinding(NamedTuple):
    payload_artifact_id: str
    source_callsite: str | None = None
    source_selector: str | None = None


PAYLOAD_CONTROLLER_LINKS = {
    "python-launch:bench/tools/run_gemm_sweep_isolated.py:run_one_process:subprocess.run:1": (
        PayloadControllerBinding("compile:build.zig:build:gemm_sweep"),
    ),
    "python-launch:bench/tools/run_level1_report.py:run_once:subprocess.run:1": (
        PayloadControllerBinding(
            "compile:build.zig:build:level1_probe",
            "run_level1_op:run_once:1",
            "args.level1_probe",
        ),
        PayloadControllerBinding(
            "compile:build.zig:build:dcopy_probe",
            "run_copy_op:run_once:1",
            "args.copy_probe",
        ),
    ),
    "python-launch:bench/tools/run_rank_k_report.py:run_one_process:subprocess.run:1": (
        PayloadControllerBinding("compile:build.zig:build:rank_k_probe"),
    ),
    "python-launch:bench/tools/run_rotg_latency_report.py:run_one_process:subprocess.run:1": (
        PayloadControllerBinding("compile:build.zig:build:rotg_latency_probe"),
    ),
    "python-launch:bench/tools/run_symm_report.py:run_one_process:subprocess.run:1": (
        PayloadControllerBinding("compile:build.zig:build:symm_probe"),
    ),
    "python-launch:bench/tools/run_triangular_matrix_report.py:run_one_process:subprocess.run:1": (
        PayloadControllerBinding("compile:build.zig:build:triangular_matrix_probe"),
    ),
}
REQUIRED_GAP_IDS = {
    "gap:process-bounds-deferred",
    "gap:example-optimize-forwarding",
    "gap:cross-target-benchmark-payload-execution",
}
REQUIRED_DERIVED_CANDIDATE_IDS = {
    "derived:include/zynum/blas/blas.h",
    "derived:include/zynum/blas/cblas.h",
    "derived:include/zynum/blas/blas.f90",
    "derived:include/zynum/blas/abi_manifest.json",
    "derived:docs/kernel_coverage.json",
    "derived:pkgconfig/zynum_blas.pc",
    "derived:tools/abi_baseline_observation.json",
    "derived:docs/assets/benchmarks/current_level1_all_types_three_libs.svg",
    "derived:docs/assets/benchmarks/current_level2_all_types_three_libs.svg",
    "derived:docs/assets/benchmarks/current_level3_all_types_more_shapes.svg",
}
REQUIRED_DERIVED_FACT_DIGESTS = {
    "derived:include/zynum/blas/blas.h": "6e9da8d67ac8521c41524651a5acc906447bf24884aabdf4008bff3ca258f1c3",
    "derived:include/zynum/blas/cblas.h": "dac223578719947a5d5fa210ad1bda6a9aeb3d0b7e334206fb86fb4ed285a5fd",
    "derived:include/zynum/blas/blas.f90": "c43410928301c3783ceac435bfab2b39a86289d2bd66737c6dc6c90c7199dce0",
    "derived:include/zynum/blas/abi_manifest.json": "03432d387c814461bae90610bb13520bd15edc9bb440d138a780c1902bb3c557",
    "derived:docs/kernel_coverage.json": "3585fdf4912ec58949ba689e2915c8620e6b8d3d941a54ce582137a2405b5fee",
    "derived:pkgconfig/zynum_blas.pc": "252c8c9903eb2c676d02bed6c7db32f8e50e4632702cbec852ddf1131310dc6c",
    "derived:tools/abi_baseline_observation.json": "2f78f017f1b8d531a4465f5ff54278464954a09169478e709ff6fff1e5c2a99a",
    "derived:docs/assets/benchmarks/current_level1_all_types_three_libs.svg": "91fda3ad65df4cea6bee7e72c8703b78aa59aa8c5fa1f5b9dd05b5b4a688a446",
    "derived:docs/assets/benchmarks/current_level2_all_types_three_libs.svg": "c1da2890fcb93ebbbcdfa18c451c4a50294ed078d7d31b2e1d1d91556ea0aa47",
    "derived:docs/assets/benchmarks/current_level3_all_types_more_shapes.svg": "f8a77a09be351d28bac550fd229c5c2270d6159b6a2f03a17aff4ed33409e90f",
}
REQUIRED_GAP_FACT_DIGESTS = {
    "gap:process-bounds-deferred": "165ca9d0f5a67a60e33c0687c1752307caf444f89df7ff499cf7418694a22ee0",
    "gap:example-optimize-forwarding": "b6d98cfa63851b999c7e144b78bfa654535f9c6733c1a63ce9312f4d075a62ec",
    "gap:cross-target-benchmark-payload-execution": "b7173a413ffc971a81554ff46c4ceedaa68ae606106928b2ec03a75c9f4780a5",
}
REQUIRED_SECTION_FACT_DIGESTS = {
    "option_surfaces": "c4047e8167df65133a70d2e096e8e8617dc26e2207210d4d5dc816a810ff811f",
    "repository_file_classifications": "a3852582e08e104b50a5cb969da73747c0ea2b990ed45ea83de161b2d0f94233",
    "derived_candidates": "5e4195f0d4b498d4a32c6487de38f310b008d81fd7500130a3a6aa4cc89c68ba",
    "current_gaps": "8e202364b2fee4ddf3aed378ab6478abc70c36c90053e4cfffa45326a8ae44a7",
}
REVIEWED_PYTHON_SCRIPT_STOP_DIGESTS = {
    "bench/tools/run_level1_report.py:check_worker_result": "a54ae66023bfd1d936e8e321fe0bb45a9da4555ed7d68283f0ababcc058a2c88",
    "bench/tools/run_level2_report.py:run_one_process": "3f7fbbff7b3d557d0b44c5b2dd3ea4457c1a058aa4171b1e8ef39e5435f64d2b",
}
REVIEWED_ADAPTER_DYNAMIC_POPEN_PROOF_DIGEST = (
    "42e9203df9c3a83ef43d30c6cccc65d03c349e5624bc47d68d466d00891c7a03"
)
REVIEWED_BOUNDED_DYNAMIC_POPEN_PROOF_DIGEST = (
    "1d32044995121c7824cb6e15e240e097c2fd9db33d9645e461367abfe4f4147c"
)


class InventoryError(Exception):
    """A deterministic inventory validation failure."""


class InventoryPublicationIndeterminate(InventoryError):
    """The candidate was installed, but directory durability is uncertain."""


class FrozenInventorySnapshot(NamedTuple):
    """One bounded inventory image and its descriptor-derived identity."""

    bytes: bytes
    identity: tuple[int, int, int, int, int]
    sha256: str
    mode: int


class RefreshedSourceCandidate(NamedTuple):
    """Validated source refresh ready for compare-and-replace publication."""

    inventory: dict[str, Any]
    bytes: bytes
    expected_snapshot: FrozenInventorySnapshot
    projection_sha256: str


class PublicFileUniverse(NamedTuple):
    root: Path
    supplied_root: Path
    paths: tuple[str, ...]
    path_set: frozenset[str]
    mode: str
    nodes: tuple[repository_snapshot.FrozenNode, ...]
    node_index: Mapping[str, repository_snapshot.FrozenNode]
    tree: repository_snapshot.TreeSnapshot
    git_file_set: repository_git.RepositoryFileSet | None
    inventory_node: repository_snapshot.FrozenNode | None

    def node(self, path: str) -> repository_snapshot.FrozenNode:
        try:
            return self.node_index[path]
        except KeyError as exc:
            raise InventoryError(
                f"path is absent from the frozen public universe: {path}"
            ) from exc


class DiscoveryContext(NamedTuple):
    root: Path
    public_files: PublicFileUniverse
    build_roots: tuple[str, ...]
    build_manifests: tuple[dict[str, str], ...]
    inventory_node: repository_snapshot.FrozenNode | None = None


class PythonCarrier(NamedTuple):
    code: str
    shape: str
    semantics_digest: str
    adapter_definition_digest: str | None


class PythonCarrierCallShape(NamedTuple):
    vector: list[ast.expr] | None
    executable: ast.expr | None
    start: int
    shape: str


class PythonCliScan(NamedTuple):
    code: str | None
    proof_digest: str


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _matching_paren(text: str, open_index: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = open_index
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and nxt == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "/" and nxt == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and nxt == "*":
            block_comment = True
            index += 2
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise InventoryError("unterminated build call")


def _code_mask(text: str) -> str:
    """Mask Zig comments and literals while preserving offsets and newlines."""
    result = list(text)
    index = 0
    state = "code"
    quote = ""
    escaped = False
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "line-comment":
            if char == "\n":
                state = "code"
            else:
                result[index] = " "
            index += 1
            continue
        if state == "block-comment":
            if char == "*" and nxt == "/":
                result[index] = result[index + 1] = " "
                state = "code"
                index += 2
            else:
                if char != "\n":
                    result[index] = " "
                index += 1
            continue
        if state == "literal":
            if char != "\n":
                result[index] = " "
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                state = "code"
            index += 1
            continue
        if char == "/" and nxt == "/":
            result[index] = result[index + 1] = " "
            state = "line-comment"
            index += 2
        elif char == "/" and nxt == "*":
            result[index] = result[index + 1] = " "
            state = "block-comment"
            index += 2
        elif char in {'"', "'"}:
            result[index] = " "
            state = "literal"
            quote = char
            escaped = False
            index += 1
        else:
            index += 1
    return "".join(result)


def _calls(text: str, token: str) -> list[tuple[int, str]]:
    boundary = "" if token.startswith(".") else r"(?<![A-Za-z0-9_])"
    pattern = re.compile(boundary + re.escape(token) + r"\s*\(")
    found: list[tuple[int, str]] = []
    for match in pattern.finditer(_code_mask(text)):
        open_index = text.find("(", match.start())
        close_index = _matching_paren(text, open_index)
        found.append((match.start(), text[match.start() : close_index + 1]))
    return found


def _semantic_digest(parts: list[str]) -> str:
    normalized = " ".join(" ".join(part.split()) for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _workflow_semantic_digest(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if part.startswith("block:"):
            kind = b"block"
            payload = part.removeprefix("block:").encode("utf-8")
        else:
            kind = b"structure"
            payload = part.encode("utf-8")
        digest.update(kind)
        digest.update(bytes([0]))
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(bytes([0]))
        digest.update(payload)
    return digest.hexdigest()


def _json_fact_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        text,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _canonical_inventory_bytes(inventory: dict[str, Any]) -> bytes:
    payload = (json.dumps(inventory, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    if len(payload) > SOURCE_REFRESH_MAX_BYTES:
        raise InventoryError(
            f"candidate inventory exceeds {SOURCE_REFRESH_MAX_BYTES} bytes"
        )
    return payload


def _json_structure_error(value: Any) -> str | None:
    """Return a bounded iterative JSON-structure error before recursive checks."""
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > SOURCE_REFRESH_MAX_JSON_NODES:
            return f"build inventory JSON exceeds {SOURCE_REFRESH_MAX_JSON_NODES} nodes"
        if isinstance(current, (dict, list)):
            if depth > SOURCE_REFRESH_MAX_JSON_DEPTH:
                return (
                    "build inventory JSON exceeds maximum depth "
                    f"{SOURCE_REFRESH_MAX_JSON_DEPTH}"
                )
            if len(current) > SOURCE_REFRESH_MAX_COLLECTION_ITEMS:
                return (
                    "build inventory JSON collection exceeds "
                    f"{SOURCE_REFRESH_MAX_COLLECTION_ITEMS} items"
                )
            children = current.values() if isinstance(current, dict) else current
            stack.extend((child, depth + 1) for child in children)
    return None


def _source_projection(inventory: dict[str, Any]) -> dict[str, Any]:
    """Return the exact source-derived facts covered by the reviewed digest slots."""
    projection: dict[str, Any] = {
        "schema_id": SOURCE_PROJECTION_SCHEMA_ID,
        "schema_version": SOURCE_PROJECTION_SCHEMA_VERSION,
    }
    for field in SOURCE_PROJECTION_FIELDS:
        if field not in inventory:
            raise InventoryError(
                f"source projection inventory is missing field {field!r}"
            )
        projection[field] = inventory[field]
    return projection


def _source_projection_digest(inventory: dict[str, Any]) -> str:
    return _json_fact_digest(_source_projection(inventory))


def _reviewed_source_projection_error(inventory: dict[str, Any]) -> str | None:
    slots = (
        ("CURRENT_SOURCE_PROJECTION_SHA256", CURRENT_SOURCE_PROJECTION_SHA256),
        ("NEXT_SOURCE_PROJECTION_SHA256", NEXT_SOURCE_PROJECTION_SHA256),
    )
    populated: list[str] = []
    for name, value in slots:
        if value is None and name == "NEXT_SOURCE_PROJECTION_SHA256":
            continue
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            return f"reviewed source projection policy has invalid {name}"
        populated.append(value)
    if len(populated) != len(set(populated)):
        return "reviewed source projection CURRENT and NEXT digests must differ"
    try:
        observed = _source_projection_digest(inventory)
    except InventoryError as exc:
        return str(exc)
    if observed not in populated:
        return f"unreviewed source projection observed sha256={observed}"
    return None


def _canonical_root(root: Path) -> Path:
    try:
        return repository_git.strict_root(root)
    except repository_git.RepositoryGitError as exc:
        raise InventoryError(f"cannot resolve repository root strictly: {exc}") from exc


def _inventory_relative_path(
    root: Path,
    inventory_path: Path | None,
    *,
    alternate_root: Path | None = None,
) -> str | None:
    if inventory_path is None:
        return None
    candidate = (
        inventory_path if inventory_path.is_absolute() else root / inventory_path
    )
    candidate = Path(os.path.abspath(candidate))
    for candidate_root in (root, alternate_root):
        if candidate_root is None:
            continue
        try:
            relative = candidate.relative_to(candidate_root).as_posix()
            repository_snapshot.relative_path(relative)
            return relative
        except (ValueError, repository_snapshot.RepositorySnapshotError):
            continue
    raise InventoryError("inventory path must be within the canonical root")


def _excluded_public_path(path: str) -> bool:
    return bool(BUILD_ROOT_EXCLUDED_PARTS.intersection(PurePosixPath(path).parts))


def _requires_cached_bytes(path: str, inventory_path: str | None) -> bool:
    pure = PurePosixPath(path)
    return (
        pure.name in {"build.zig", "build.zig.zon"}
        or pure.suffix.lower() in {".py", ".zig"}
        or (
            pure.parent == PurePosixPath(".github/workflows")
            and pure.suffix.lower() in {".yml", ".yaml"}
        )
        or path == DEPENDABOT_CONFIG_PATH
        or path in SNAPSHOT_GENERATOR_PATHS
        or path
        in {inventory_path, "tools/build_inventory.json", "tools/test_inventory.json"}
    )


def _capture_cached_node(
    session: repository_snapshot.SnapshotSession,
    path: str,
) -> repository_snapshot.FrozenNode:
    return session.capture_paths(
        (path,), include_bytes=True, limit=SNAPSHOT_MAX_CACHED_FILE_BYTES
    )[0]


def _make_public_file_universe(
    root: Path, inventory_path: Path | None = None
) -> PublicFileUniverse:
    supplied_root = Path(os.path.abspath(root))
    canonical = _canonical_root(root)
    inventory_rel = _inventory_relative_path(
        supplied_root,
        inventory_path,
        alternate_root=canonical,
    )
    try:
        repository = repository_git.open_repository(canonical)
    except repository_git.RepositoryGitError as exc:
        raise InventoryError(str(exc)) from exc
    try:
        regular_file_limits = repository_snapshot.RegularFileLimits(
            max_file_bytes=SNAPSHOT_MAX_REGULAR_FILE_BYTES,
            max_round_bytes=SNAPSHOT_MAX_REGULAR_ROUND_BYTES,
            max_cached_file_bytes=SNAPSHOT_MAX_CACHED_FILE_BYTES,
            max_cached_total_bytes=SNAPSHOT_MAX_CACHED_TOTAL_BYTES,
        )
        directory_structure_limits = repository_snapshot.DirectoryStructureLimits(
            max_directories=SNAPSHOT_MAX_DIRECTORIES,
            max_entries=SNAPSHOT_MAX_ENTRIES,
            max_depth=SNAPSHOT_MAX_DEPTH,
            max_total_name_bytes=SNAPSHOT_MAX_TOTAL_NAME_BYTES,
            max_total_structure_bytes=SNAPSHOT_MAX_TOTAL_STRUCTURE_BYTES,
        )
        with repository_snapshot.SnapshotSession(
            canonical,
            regular_file_limits=regular_file_limits,
            directory_structure_limits=directory_structure_limits,
        ) as session:
            git_files: repository_git.RepositoryFileSet | None = None
            if repository is not None:
                git_files = repository.snapshot_file_set()  # G0
                session.freeze_directory_structure(BUILD_ROOT_EXCLUDED_PARTS)
                public_paths = tuple(
                    path
                    for path in git_files.present
                    if not _excluded_public_path(path)
                )
                uncached = tuple(
                    path
                    for path in public_paths
                    if not _requires_cached_bytes(path, inventory_rel)
                )
                session.capture_paths(uncached)
                for path in public_paths:
                    if _requires_cached_bytes(path, inventory_rel):
                        _capture_cached_node(session, path)
                for path in git_files.deleted:
                    session.prove_absent(path)
            else:
                session.freeze_directory_structure(BUILD_ROOT_EXCLUDED_PARTS)
                walked = session.walk_archive_root(BUILD_ROOT_EXCLUDED_PARTS)
                public_paths = tuple(
                    node.path for node in walked if node.kind != "directory"
                )
                for path in public_paths:
                    if _requires_cached_bytes(path, inventory_rel):
                        _capture_cached_node(session, path)

            inventory_node: repository_snapshot.FrozenNode | None = None
            if inventory_rel is not None:
                inventory_node = _capture_cached_node(session, inventory_rel)
                if inventory_node.kind != "regular":
                    raise InventoryError("inventory must be a frozen regular file")

            # Repository fixed point: G0 -> bounded structure admission plus
            # capture -> F0 -> G1 -> F1 -> G2.  Archive mode performs the same
            # bounded structure admission before its first content read.
            # There are deliberately no retries.  A mutation after G2 lies
            # beyond the accepted boundary and cannot affect frozen outputs.
            tree = session.seal()  # F0
            if repository is not None:
                observed_git_files = repository.snapshot_file_set()  # G1
                if observed_git_files != git_files:
                    raise InventoryError("Git file set changed during snapshot")
                session.verify(tree)  # F1
                final_git_files = repository.snapshot_file_set()  # G2
                if final_git_files != git_files:
                    raise InventoryError("Git file set changed during snapshot")
    except FileNotFoundError as exc:
        raise InventoryError("repository member disappeared during snapshot") from exc
    except repository_snapshot.RepositorySnapshotError as exc:
        raise InventoryError(f"repository snapshot failed closed: {exc}") from exc
    except repository_git.RepositoryGitError as exc:
        raise InventoryError(str(exc)) from exc

    public_nodes = tuple(tree.node(path) for path in sorted(public_paths))
    node_index = MappingProxyType({node.path: node for node in public_nodes})
    universe = PublicFileUniverse(
        root=canonical,
        supplied_root=supplied_root,
        paths=tuple(node.path for node in public_nodes),
        path_set=frozenset(node.path for node in public_nodes),
        mode="git-checkout" if repository is not None else "archive",
        nodes=public_nodes,
        node_index=node_index,
        tree=tree,
        git_file_set=git_files,
        inventory_node=inventory_node,
    )
    return universe


def _git_public_file_paths(root: Path) -> tuple[list[str], bool]:
    """Compatibility helper for direct tests; validation constructs one context."""
    universe = _make_public_file_universe(root)
    return list(universe.paths), universe.mode == "git-checkout"


def _build_roots_from_universe(
    universe: PublicFileUniverse,
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    roots: list[str] = []
    manifests: list[str] = []
    for rel in universe.paths:
        name = PurePosixPath(rel).name
        if name not in {"build.zig", "build.zig.zon"}:
            continue
        node = universe.node(rel)
        if node.kind != "regular":
            raise InventoryError(
                f"build root candidate must be a non-symlink regular file: {rel}"
            )
        if name == "build.zig":
            roots.append(Path(rel).as_posix())
        else:
            manifests.append(rel)
    manifest_rows: list[dict[str, str]] = []
    for rel in manifests:
        sibling_rel = (PurePosixPath(rel).parent / "build.zig").as_posix()
        if sibling_rel not in universe.path_set:
            raise InventoryError(f"build manifest has no safe sibling build.zig: {rel}")
        if universe.node(sibling_rel).kind != "regular":
            raise InventoryError(f"build manifest has no safe sibling build.zig: {rel}")
        manifest = universe.node(rel)
        if manifest.bytes is None or manifest.sha256 is None:
            raise InventoryError(f"build manifest bytes were not frozen: {rel}")
        manifest_rows.append(
            {
                "id": f"build-manifest:{rel}",
                "path": rel,
                "build_root": sibling_rel,
                "content_sha256": manifest.sha256,
            }
        )
    return tuple(sorted(roots)), tuple(sorted(manifest_rows, key=lambda row: row["id"]))


def _make_discovery_context(
    root: Path, inventory_path: Path | None = None
) -> DiscoveryContext:
    universe = _make_public_file_universe(root, inventory_path)
    build_roots, build_manifests = _build_roots_from_universe(universe)
    return DiscoveryContext(
        universe.root,
        universe,
        build_roots,
        build_manifests,
        universe.inventory_node,
    )


def _context_for(root: Path, context: DiscoveryContext | None) -> DiscoveryContext:
    if context is None:
        return _make_discovery_context(root)
    supplied = Path(os.path.abspath(root))
    if (
        supplied not in {context.root, context.public_files.supplied_root}
        or context.public_files.root != context.root
    ):
        raise InventoryError("discovery context root does not match the canonical root")
    return context


def _discover_build_roots(
    root: Path, context: DiscoveryContext | None = None
) -> list[str]:
    return list(_context_for(root, context).build_roots)


def _discover_build_manifests(
    root: Path, context: DiscoveryContext | None = None
) -> list[dict[str, str]]:
    return [dict(row) for row in _context_for(root, context).build_manifests]


def _frozen_regular_bytes(
    context: DiscoveryContext, rel: str, description: str
) -> bytes:
    node = context.public_files.node(rel)
    if node.kind != "regular":
        raise InventoryError(f"{description} must be a frozen regular file: {rel}")
    if node.bytes is None:
        raise InventoryError(f"{description} bytes were not frozen: {rel}")
    return node.bytes


def _frozen_regular_text(context: DiscoveryContext, rel: str, description: str) -> str:
    try:
        return _frozen_regular_bytes(context, rel, description).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InventoryError(f"{description} is not valid UTF-8: {rel}") from exc


def _repository_file_kind(node: repository_snapshot.FrozenNode) -> str:
    rel = node.path
    if node.kind == "symlink":
        return "symbolic-link"
    if node.kind != "regular":
        return "other-filesystem-object"
    pure = PurePosixPath(rel)
    name = pure.name
    suffix = pure.suffix.lower()
    if name in {
        "LICENSE",
        "COPYING",
        "COPYING.LESSER",
        "NOTICE",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
    }:
        return "legal-governance"
    if name == "build.zig":
        return "zig-build-root"
    if name == "build.zig.zon":
        return "zig-build-manifest"
    if rel.startswith(".github/workflows/") and suffix in {".yml", ".yaml"}:
        return "workflow-definition"
    if suffix == ".zig":
        return "zig-source"
    if suffix == ".py":
        return "python-source"
    if suffix == ".json":
        return "json-data"
    if suffix in {".md", ".mdx", ".rst"}:
        return "documentation"
    if suffix in {".h", ".hpp", ".c", ".cc", ".cpp", ".f90"}:
        return "foreign-language-source"
    if suffix in {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "visual-asset"
    if suffix in {".yml", ".yaml", ".toml", ".zon", ".pc"}:
        return "configuration-metadata"
    if suffix == "":
        return "extensionless-public-file"
    return "other-public-file"


def _discover_repository_file_classifications(
    root: Path, context: DiscoveryContext | None = None
) -> tuple[list[dict[str, str]], bool]:
    active = _context_for(root, context)
    root = active.root
    paths = active.public_files.paths
    return (
        [
            {"path": rel, "kind": _repository_file_kind(active.public_files.node(rel))}
            for rel in paths
        ],
        active.public_files.mode == "git-checkout",
    )


def _canonical_python_ast_value(value: Any) -> str:
    if isinstance(value, ast.AST):
        return _canonical_python_ast(value)
    if isinstance(value, list):
        return "[" + ",".join(_canonical_python_ast_value(item) for item in value) + "]"
    if isinstance(value, tuple):
        items = ",".join(_canonical_python_ast_value(item) for item in value)
        return f"({items}{',' if len(value) == 1 else ''})"
    if (
        value is None
        or value is Ellipsis
        or isinstance(value, (str, bytes, bool, int, float, complex))
    ):
        return repr(value)
    raise TypeError(f"unsupported Python AST field value: {type(value).__name__}")


def _canonical_python_ast(node: ast.AST) -> str:
    fields = []
    for name, value in ast.iter_fields(node):
        if name == "type_params" and value == []:
            continue
        fields.append(f"{name}={_canonical_python_ast_value(value)}")
    return f"{type(node).__name__}({','.join(fields)})"


def _python_call_semantics_digest(
    node: ast.Call,
    canonical_api: str,
    parents: dict[ast.AST, ast.AST],
    *,
    carrier_semantics_digests: tuple[str, ...] = (),
) -> str:
    statement: ast.AST = node
    current: ast.AST = node
    function: ast.AST | None = None
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.stmt) and statement is node:
            statement = current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            function = current
            break
    payload = {
        "version": PYTHON_SEMANTICS_VERSION,
        "canonical_api": canonical_api,
        "call": _canonical_python_ast(node),
        "statement": _canonical_python_ast(statement),
        "lexical_function": _canonical_python_ast(function) if function else "<module>",
    }
    if carrier_semantics_digests:
        payload["carrier_semantics_digests"] = list(carrier_semantics_digests)
    return _json_fact_digest(payload)


def _python_ancestor_control_path(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    controls: list[str] = []
    current = node
    control_types = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        *AST_TRY_STAR_TYPES,
        ast.With,
        ast.AsyncWith,
        ast.Match,
        ast.comprehension,
    )
    while current in parents:
        current = parents[current]
        if isinstance(current, control_types):
            controls.append(_canonical_python_ast(current))
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            break
    return list(reversed(controls))


def _python_scope_token(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    parts: list[str] = []
    current = node
    scopes: list[ast.AST] = []
    while current in parents:
        current = parents[current]
        if isinstance(
            current,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
        ):
            scopes.append(current)
    for scope in reversed(scopes):
        if isinstance(scope, ast.Lambda):
            label = "lambda"
        else:
            label = scope.name
        lexical_parent: ast.AST | None = parents.get(scope)
        while lexical_parent is not None and not isinstance(
            lexical_parent,
            (
                ast.Module,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
                ast.ClassDef,
            ),
        ):
            lexical_parent = parents.get(lexical_parent)
        siblings = []
        if lexical_parent is not None:
            for candidate in ast.walk(lexical_parent):
                if not isinstance(candidate, type(scope)):
                    continue
                candidate_label = (
                    "lambda" if isinstance(candidate, ast.Lambda) else candidate.name
                )
                if candidate_label != label:
                    continue
                candidate_parent = parents.get(candidate)
                while candidate_parent is not None and not isinstance(
                    candidate_parent,
                    (
                        ast.Module,
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.Lambda,
                        ast.ClassDef,
                    ),
                ):
                    candidate_parent = parents.get(candidate_parent)
                if candidate_parent is lexical_parent:
                    siblings.append(candidate)
        ordinal = siblings.index(scope) + 1 if scope in siblings else 1
        parts.append(f"{label}-{ordinal}")
    return "/".join(parts) if parts else "module"


def _ancestor_chain(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> Iterator[ast.AST]:
    current = node
    while current in parents:
        current = parents[current]
        yield current


def _top_level_statement_start(masked: str, start: int, end: int) -> int:
    """Return the first byte after the last top-level semicolon in a region."""
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    statement_start = start
    for index in range(start, end):
        char = masked[index]
        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
            if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                statement_start = index + 1
        elif (
            char == ";" and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0
        ):
            statement_start = index + 1
    return statement_start


def _enclosing_control_context(text: str, position: int) -> list[str]:
    """Capture braced and unbraced control context before a build edge."""
    masked = _code_mask(text[:position])
    stack: list[int] = []
    for index, char in enumerate(masked):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            stack.pop()

    context: list[str] = []
    for stack_index, open_index in enumerate(stack):
        container_start = stack[stack_index - 1] + 1 if stack_index else 0
        header_start = _top_level_statement_start(masked, container_start, open_index)
        header = text[header_start : open_index + 1].strip()
        if header:
            context.append(header)

    container_start = stack[-1] + 1 if stack else 0
    prefix_start = _top_level_statement_start(masked, container_start, position)
    prefix = text[prefix_start:position].strip()
    if prefix:
        context.append(prefix)
    return context


def _enclosing_if_guards(text: str, position: int) -> list[str]:
    masked = _code_mask(text[:position])
    stack: list[int] = []
    for index, char in enumerate(masked):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            stack.pop()

    guards: list[str] = []
    for open_index in stack:
        line_start = text.rfind("\n", 0, open_index) + 1
        header = text[line_start : open_index + 1].strip()
        if re.match(r"^if\s*\(", _code_mask(header)):
            guards.append(header)

    container_start = stack[-1] + 1 if stack else 0
    prefix_start = _top_level_statement_start(masked, container_start, position)
    prefix = text[prefix_start:position].strip()
    if prefix and re.search(r"\bif\s*\(", _code_mask(prefix)):
        guards.append(prefix)
    return guards


def _zig_optional_capture_source(text: str, position: int, capture: str) -> str | None:
    """Resolve a direct optional-payload capture used as a link consumer."""
    pattern = re.compile(
        rf"^if\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*"
        rf"\|\s*{re.escape(capture)}\s*\|\s*\{{$"
    )
    matches = [
        match.group(1)
        for header in _enclosing_control_context(text, position)
        if (match := pattern.fullmatch(_code_mask(header).strip())) is not None
    ]
    if len(matches) > 1:
        raise InventoryError(
            f"ambiguous Zig optional capture source for link consumer {capture!r}"
        )
    return matches[0] if matches else None


def _call_semantics(
    text: str,
    position: int,
    call: str,
    token: str,
    symbol: str,
    provider: str | None,
    *,
    build_context: ZigBuildContext,
) -> str:
    parts = [
        "build-source-sha256=" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        call,
    ]
    if token == "b.option":
        close = position + len(call)
        semicolon = text.find(";", close)
        if semicolon >= 0:
            parts.append(text[close : semicolon + 1])
    if token in {"b.addLibrary", "b.addExecutable", "b.addTest"}:
        module_match = re.search(r"\.root_module\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", call)
        if module_match:
            module = module_match.group(1)
            build = build_context
            receiver_pattern = "|".join(map(re.escape, build.receivers))
            if module not in build.receivers and receiver_pattern:
                declaration = re.search(
                    rf"\bconst\s+{re.escape(module)}\s*=\s*"
                    rf"(?:{receiver_pattern})\s*\.\s*createModule\s*\(",
                    _code_mask(text),
                )
                if declaration:
                    receiver = re.search(
                        rf"(?:{receiver_pattern})\s*\.\s*createModule\s*\(",
                        _code_mask(text)[declaration.start() :],
                    )
                    if receiver is not None:
                        call_start = declaration.start() + receiver.start()
                        open_index = text.find("(", call_start)
                        close_index = _matching_paren(text, open_index)
                        parts.append(text[call_start : close_index + 1])
    if token == "b.step":
        for dependency in re.finditer(
            rf"^.*{re.escape(symbol)}\.dependOn\(.*$", text, re.MULTILINE
        ):
            parts.extend(_enclosing_if_guards(text, dependency.start()))
            parts.append(dependency.group(0).strip())
    if token in {"b.addRunArtifact", "b.addSystemCommand"}:
        launch_lines = [
            line.strip()
            for line in text.splitlines()
            if re.search(
                rf"\b{re.escape(symbol)}\.(?:addArg|addArgs|addFileArg|setCwd|setEnvironmentVariable|step\.dependOn)\b",
                line,
            )
        ]
        parts.extend(launch_lines)
    if provider:
        parts.append(f"provider={provider}")
    return _semantic_digest(parts)


def _symbol_before(text: str, position: int, call: str) -> str:
    statement_start = (
        max(
            text.rfind(";", 0, position),
            text.rfind("{", 0, position),
            text.rfind("}", 0, position),
        )
        + 1
    )
    prefix = text[statement_start:position]
    matches = list(re.finditer(r"\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", prefix))
    if matches:
        return matches[-1].group(1)
    strings = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', call)
    return _slug(strings[0]) if strings else "anonymous"


def _first_string(call: str) -> str | None:
    match = re.search(r'"([^"\\]*(?:\\.[^"\\]*)*)"', call)
    return bytes(match.group(1), "utf-8").decode("unicode_escape") if match else None


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char in "({[":
            depth += 1
        elif char in ")} ]".replace(" ", ""):
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return parts


class ZigBuildContext(NamedTuple):
    receiver: str | None
    receivers: dict[str, int]
    body_start: int
    body_end: int


ZIG_BUILD_RECEIVER_METHODS = {
    token.removeprefix("b.") for token in BUILD_CALLS if token.startswith("b.")
} | {
    "standardTargetOptions",
    "standardTargetOptionsQueryOnly",
    "standardOptimizeOption",
}


def _matching_brace(masked: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise InventoryError("unterminated Zig build function")


def _zig_statement_end(masked: str, start: int, limit: int) -> int:
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    for index in range(start, limit):
        char = masked[index]
        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        elif (
            char == ";" and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0
        ):
            return index
    raise InventoryError("unterminated Zig declaration in build function")


def _zig_brace_depth(masked: str, start: int, position: int) -> int:
    depth = 0
    for char in masked[start:position]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
    return depth


def _zig_member_calls(
    text: str, build: ZigBuildContext, receiver: str, method: str
) -> list[tuple[int, str]]:
    masked = _code_mask(text)
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(receiver)}\s*\.\s*{re.escape(method)}\s*\("
    )
    found: list[tuple[int, str]] = []
    available_at = build.receivers[receiver]
    for match in pattern.finditer(masked, build.body_start, build.body_end):
        if match.start() < available_at:
            raise InventoryError(
                f"Zig build receiver alias {receiver!r} is used before its declaration"
            )
        open_index = masked.find("(", match.start(), match.end())
        close_index = _matching_paren(text, open_index)
        if close_index > build.body_end:
            raise InventoryError("Zig build call escapes the build function")
        found.append((match.start(), text[match.start() : close_index + 1]))
    return found


def _zig_build_context(text: str, rel_path: str) -> ZigBuildContext:
    masked = _code_mask(text)
    matches = list(re.finditer(r"\b(?:pub\s+)?fn\s+build\s*\(", masked))
    if len(matches) != 1:
        raise InventoryError(
            f"{rel_path}: expected exactly one structurally recognizable build function"
        )
    signature = matches[0]
    open_paren = masked.find("(", signature.start(), signature.end())
    close_paren = _matching_paren(text, open_paren)
    open_brace = masked.find("{", close_paren)
    if open_brace < 0 or ";" in masked[close_paren:open_brace]:
        raise InventoryError(f"{rel_path}: build function body is unsupported")
    close_brace = _matching_brace(masked, open_brace)
    parameters = _split_top_level(text[open_paren + 1 : close_paren])
    build_parameters: list[str] = []
    for parameter in parameters:
        match = re.fullmatch(
            r"(?:comptime\s+)?([A-Za-z_][A-Za-z0-9_]*|_)\s*:\s*\*\s*(?:const\s+)?std\.Build",
            _code_mask(parameter).strip(),
        )
        if match is not None and match.group(1) != "_":
            build_parameters.append(match.group(1))
    if len(build_parameters) > 1:
        raise InventoryError(f"{rel_path}: build receiver parameter is ambiguous")

    body_start = open_brace + 1
    body_end = close_brace
    receiver = build_parameters[0] if build_parameters else None
    receivers = {receiver: body_start} if receiver is not None else {}
    declaration_names: set[tuple[int, int]] = set()
    safe_alias_sources: set[tuple[int, int]] = set()
    declaration_pattern = re.compile(
        r"\b(?P<kind>const|var)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        r"(?:\s*:[^=;\n]+)?\s*="
    )
    for declaration in declaration_pattern.finditer(masked, body_start, body_end):
        name = declaration.group("name")
        declaration_names.add(declaration.span("name"))
        statement_end = _zig_statement_end(masked, declaration.end(), body_end)
        rhs = masked[declaration.end() : statement_end].strip()
        if name in receivers:
            raise InventoryError(
                f"{rel_path}: Zig build receiver {name!r} is shadowed or rebound"
            )
        source_receiver = next(
            (candidate for candidate in receivers if rhs == candidate), None
        )
        if source_receiver is not None:
            if (
                declaration.group("kind") != "const"
                or _zig_brace_depth(masked, open_brace, declaration.start()) != 1
            ):
                raise InventoryError(
                    f"{rel_path}: scoped or mutable Zig build receiver aliases are unsupported"
                )
            rhs_prefix = masked[declaration.end() : statement_end]
            source_start = (
                declaration.end() + len(rhs_prefix) - len(rhs_prefix.lstrip())
            )
            safe_alias_sources.add((source_start, source_start + len(source_receiver)))
            receivers[name] = statement_end + 1
            continue

    for candidate in receivers:
        assignment_pattern = re.compile(
            rf"(?<![A-Za-z0-9_.]){re.escape(candidate)}\s*=(?!=)"
        )
        for assignment in assignment_pattern.finditer(masked, body_start, body_end):
            if any(
                start <= assignment.start() < end for start, end in declaration_names
            ):
                continue
            raise InventoryError(
                f"{rel_path}: Zig build receiver {candidate!r} is shadowed or rebound"
            )

    for candidate, available_at in receivers.items():
        use_pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(candidate)}(?![A-Za-z0-9_])"
        )
        for use in use_pattern.finditer(masked, body_start, body_end):
            if any(start <= use.start() < end for start, end in declaration_names):
                continue
            if use.span() in safe_alias_sources:
                continue
            prefix = masked[body_start : use.start()].rstrip()
            if prefix.endswith("."):
                continue
            if use.start() < available_at:
                raise InventoryError(
                    f"{rel_path}: Zig build receiver alias {candidate!r} is used before its declaration"
                )
            suffix = masked[use.end() :].lstrip()
            direct_member = re.match(r"\.\s*[A-Za-z_][A-Za-z0-9_]*", suffix)
            nested_function = any(
                re.search(r"\bfn\b", _code_mask(header))
                and not re.search(r"\bfn\s+build\b", _code_mask(header))
                for header in _enclosing_control_context(text, use.start())
            )
            if direct_member is not None and not nested_function:
                continue
            raise InventoryError(
                f"{rel_path}: Zig build receiver {candidate!r} escapes the analyzed build body"
            )

    build = ZigBuildContext(receiver, receivers, body_start, body_end)
    member_pattern = re.compile(
        rf"(?<![A-Za-z0-9_])(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
        rf"(?P<method>{'|'.join(sorted(map(re.escape, ZIG_BUILD_RECEIVER_METHODS), key=len, reverse=True))})\s*\("
    )
    for call in member_pattern.finditer(masked, body_start, body_end):
        call_receiver = call.group("receiver")
        if call_receiver not in receivers:
            raise InventoryError(
                f"{rel_path}: build API call uses unsupported or ambiguous receiver {call_receiver!r}"
            )
        if call.start() < receivers[call_receiver]:
            raise InventoryError(
                f"{rel_path}: Zig build receiver alias {call_receiver!r} is used before its declaration"
            )
    if receiver is None and member_pattern.search(masked, body_start, body_end):
        raise InventoryError(f"{rel_path}: build receiver parameter is unsupported")
    return build


def _discover_option_surface_semantics(
    root: Path, context: DiscoveryContext | None = None
) -> dict[tuple[str, str], dict[str, str]]:
    active = _context_for(root, context)
    root = active.root
    result: dict[tuple[str, str], dict[str, str]] = {}
    for rel in active.build_roots:
        text = _frozen_regular_text(active, rel, "build root")
        build = _zig_build_context(text, rel)
        option_calls = sorted(
            (
                occurrence
                for receiver in build.receivers
                for occurrence in _zig_member_calls(text, build, receiver, "option")
            ),
            key=lambda occurrence: occurrence[0],
        )
        for position, call in option_calls:
            body = call[call.find("(") + 1 : -1]
            arguments = _split_top_level(body)
            if len(arguments) < 3:
                continue
            name = json.loads(arguments[1])
            description = json.loads(arguments[2])
            semicolon = text.find(";", position + len(call))
            trailer = text[
                position + len(call) : semicolon
                if semicolon >= 0
                else position + len(call)
            ]
            default_match = re.search(r"\borelse\s+\.?(\w+)", trailer)
            default = default_match.group(1) if default_match else "unset"
            result[(rel, name)] = {
                "type": arguments[0],
                "default": default,
                "description": description,
            }
        standard_defaults: dict[str, tuple[str, str, str]] = {
            "target": (
                "target-query",
                "native requested target",
                "Select target triple",
            ),
            "cpu": (
                "cpu-model-and-features",
                "target default CPU selection",
                "Select CPU model and feature set",
            ),
            "ofmt": (
                "object-format",
                "target default object format",
                "Select output object format",
            ),
            "dynamic-linker": (
                "path",
                "target default dynamic linker",
                "Select dynamic linker path",
            ),
        }
        if rel == "build.zig":
            standard_defaults["release"] = (
                "bool",
                "false",
                "Request Zig standard release optimization resolution",
            )
        else:
            standard_defaults["optimize"] = (
                "std.builtin.OptimizeMode",
                "Debug",
                "Select Debug, ReleaseSafe, ReleaseFast, or ReleaseSmall optimization",
            )
        for name, (option_type, default, description) in standard_defaults.items():
            result[(rel, name)] = {
                "type": option_type,
                "default": default,
                "description": description,
            }
    return result


def _discover_build_root(
    root: Path,
    rel_path: str,
    context: DiscoveryContext | None = None,
) -> list[dict[str, Any]]:
    active = _context_for(root, context)
    text = _frozen_regular_text(active, rel_path, "build root")
    build = _zig_build_context(text, rel_path)
    observations: list[dict[str, Any]] = []
    for token, category in BUILD_CALLS.items():
        if token == "linkLibrary":
            occurrences = [
                occurrence
                for occurrence in _calls(text, ".linkLibrary")
                if build.body_start <= occurrence[0] < build.body_end
            ]
        else:
            method = token.removeprefix("b.")
            occurrences = sorted(
                (
                    occurrence
                    for receiver in build.receivers
                    for occurrence in _zig_member_calls(text, build, receiver, method)
                ),
                key=lambda occurrence: occurrence[0],
            )
        symbol_counts: Counter[str] = Counter()
        for position, call in occurrences:
            symbol = _symbol_before(text, position, call)
            semantic_symbol = symbol
            provider: str | None = None
            if token == "linkLibrary":
                line_prefix = text[text.rfind("\n", 0, position) + 1 : position]
                receiver_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", line_prefix)
                if receiver_match:
                    symbol = receiver_match.group(1)
                    semantic_symbol = (
                        _zig_optional_capture_source(text, position, symbol) or symbol
                    )
                argument_match = re.search(
                    r"\.linkLibrary\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)", call
                )
                provider = argument_match.group(1) if argument_match else "unknown"
                if provider == "library":
                    context = text[max(0, position - 1200) : position]
                    bindings = re.findall(
                        r"if\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\|library\|",
                        context,
                    )
                    if bindings:
                        provider = bindings[-1]
            if (
                category == "install"
                and token == "b.installArtifact"
                and symbol == "anonymous"
            ):
                producer_match = re.search(
                    r"\.\s*installArtifact\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)",
                    call,
                )
                if producer_match:
                    symbol = producer_match.group(1)
            symbol_counts[symbol] += 1
            ordinal = symbol_counts[symbol]
            name = _first_string(call)
            if token == "b.option":
                semantic = name or symbol
            elif token == "b.step":
                semantic = name or symbol
            elif token == "linkLibrary":
                semantic = f"{semantic_symbol}<-{provider}"
            elif category == "install" and name:
                semantic = name
            else:
                semantic = symbol if ordinal == 1 else f"{symbol}-{ordinal}"
            observation = {
                "id": f"{category}:{rel_path}:build:{semantic}",
                "category": category,
                "anchor": {
                    "file": rel_path,
                    "enclosing_function": "build",
                    "symbol": symbol,
                    "ordinal": ordinal,
                },
                "call": token,
                "source_digest": _call_semantics(
                    text,
                    position,
                    call,
                    token,
                    symbol,
                    provider,
                    build_context=build,
                ),
            }
            if token == "linkLibrary":
                observation["guard_digest"] = _semantic_digest(
                    _enclosing_control_context(text, position)
                )
            observations.append(observation)
    for call, semantic in (
        ("standardTargetOptions", "target"),
        ("standardTargetOptionsQueryOnly", "target"),
        ("standardOptimizeOption", "optimize"),
    ):
        occurrences = sorted(
            (
                occurrence
                for receiver in build.receivers
                for occurrence in _zig_member_calls(text, build, receiver, call)
            ),
            key=lambda occurrence: occurrence[0],
        )
        for ordinal, (position, actual_call) in enumerate(occurrences, 1):
            symbol = _symbol_before(text, position, actual_call)
            normalized_call = f"b.{call}"
            observations.append(
                {
                    "id": f"option:{rel_path}:build:{semantic}",
                    "category": "option",
                    "anchor": {
                        "file": rel_path,
                        "enclosing_function": "build",
                        "symbol": symbol,
                        "ordinal": ordinal,
                    },
                    "call": normalized_call,
                    "source_digest": _call_semantics(
                        text,
                        position,
                        normalized_call,
                        normalized_call,
                        symbol,
                        None,
                        build_context=build,
                    ),
                }
            )
    if rel_path == "build.zig" and (
        "inventory_cases" in _code_mask(text)
        or any(
            item["id"]
            in {
                TEST_INVENTORY_FACTORY_COMPILE_ID,
                TEST_INVENTORY_FACTORY_LAUNCH_ID,
                TEST_INVENTORY_LINK_STEP_ID,
                TEST_INVENTORY_RUN_STEP_ID,
            }
            for item in observations
        )
    ):
        _annotate_test_inventory_factory(text, observations)
        _annotate_native_feature_test_contract(text, observations)
    if rel_path == "build.zig":
        _annotate_python_tooling_tests(text, observations)
    return observations


TEST_INVENTORY_FACTORY_COMPILE_ID = "compile:build.zig:build:inventory_tests"
TEST_INVENTORY_FACTORY_LAUNCH_ID = "launch:build.zig:build:run_inventory_tests"
TEST_INVENTORY_LINK_STEP_ID = "step:build.zig:build:test-inventory-link"
TEST_INVENTORY_RUN_STEP_ID = "step:build.zig:build:test-inventory"
TEST_INVENTORY_AGGREGATE_STEP_ID = "step:build.zig:build:test"
NATIVE_FEATURE_STEP_ID = "step:build.zig:build:test-native-feature"
NATIVE_FEATURE_LAUNCH_ID = "launch:build.zig:build:run_native_feature_tests"
NATIVE_FEATURE_GUARDS = (
    (
        "step:build.zig:build:test-native-feature-requires-an-explicit-non-baseline-cpu-profile",
        "test-native-feature requires an explicit non-baseline CPU profile",
        "requested CPU model is baseline or determined by arch/OS",
    ),
    (
        "step:build.zig:build:test-native-feature-requires-target-arch-os-abi-ofmt-to-match-the-build-host-exactly",
        "test-native-feature requires target arch/os/abi/ofmt to match the build host exactly",
        "requested target arch, OS, ABI, or object format differs from the build host",
    ),
    (
        "step:build.zig:build:test-native-feature-forbids-external-target-executors",
        "test-native-feature forbids external target executors",
        "QEMU, Rosetta, Wine, Darling, or Wasmtime execution is enabled",
    ),
    (
        "step:build.zig:build:test-native-feature-requested-cpu-features-are-not-supported-by-the-build-host",
        "test-native-feature requested CPU features are not supported by the build host",
        "build-host CPU features are not a superset of requested target CPU features",
    ),
)
TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID = (
    "step:build.zig:build:unsupported_test_inventory_target"
)
TEST_INVENTORY_UNSUPPORTED_TARGET_MESSAGE = (
    "test inventory enumeration is unavailable for the requested target CPU profile"
)
REQUIRED_TEST_INVENTORY_ENUMERATION_MAPPINGS = (
    {
        "architecture": "aarch64",
        "os": "macos",
        "abi": "none",
        "object_format": "macho",
        "cpu_model": "apple_m1",
        "environment_id": "env:aarch64-macos-baseline",
        "enumeration_class_id": "enumeration-class:aarch64-macos-system-macho",
    },
    {
        "architecture": "x86_64",
        "os": "linux",
        "abi": "gnu",
        "object_format": "elf",
        "cpu_model": "x86_64",
        "environment_id": "env:x86-64-linux-gnu-baseline",
        "enumeration_class_id": "enumeration-class:x86-64-linux-gnu-elf",
    },
    {
        "architecture": "aarch64",
        "os": "linux",
        "abi": "gnu",
        "object_format": "elf",
        "cpu_model": "generic",
        "environment_id": "env:aarch64-linux-gnu-baseline",
        "enumeration_class_id": "enumeration-class:aarch64-linux-gnu-elf",
    },
    {
        "architecture": "x86_64",
        "os": "windows",
        "abi": "gnu",
        "object_format": "coff",
        "cpu_model": "x86_64",
        "environment_id": "env:x86-64-windows-gnu-baseline",
        "enumeration_class_id": "enumeration-class:x86-64-windows-gnu-coff",
    },
)
REQUIRED_TEST_INVENTORY_FACTORY_CASES = (
    (
        "zig-root:blas-module-tests",
        "compile:build.zig:build:blas_module_tests",
        "predicate:always",
    ),
    (
        "zig-root:blas-public-surface-contract-tests",
        "compile:build.zig:build:blas_public_surface_contract_tests",
        "predicate:always",
    ),
    ("zig-root:cblas-tests", "compile:build.zig:build:cblas_tests", "predicate:always"),
    (
        "zig-root:fortran-tests",
        "compile:build.zig:build:fortran_tests",
        "predicate:always",
    ),
    (
        "zig-root:gemm-registry-tests",
        "compile:build.zig:build:gemm_registry_tests",
        "predicate:always",
    ),
    (
        "zig-root:header-smoke-tests",
        "compile:build.zig:build:header_smoke_tests",
        "predicate:always",
    ),
    (
        "zig-root:level1-registry-tests",
        "compile:build.zig:build:level1_registry_tests",
        "predicate:always",
    ),
    (
        "zig-root:level2-compact-registry-tests",
        "compile:build.zig:build:level2_compact_registry_tests",
        "predicate:always",
    ),
    (
        "zig-root:level2-fused-registry-tests",
        "compile:build.zig:build:level2_fused_registry_tests",
        "predicate:always",
    ),
    (
        "zig-root:modern-tests",
        "compile:build.zig:build:modern_tests",
        "predicate:always",
    ),
    (
        "zig-root:packed-parallel-tests",
        "compile:build.zig:build:packed_parallel_tests",
        "predicate:always",
    ),
    (
        "zig-root:structured-blocked-tests",
        "compile:build.zig:build:structured_blocked_tests",
        "predicate:always",
    ),
    (
        "zig-root:structured-object-tests",
        "compile:build.zig:build:structured_object_tests",
        "predicate:arch-x86-64",
    ),
    (
        "zig-root:symm-dense-gemm-tests",
        "compile:build.zig:build:symm_dense_gemm_tests",
        "predicate:always",
    ),
    (
        "zig-root:triangular-band-solve-tests",
        "compile:build.zig:build:triangular_band_solve_tests",
        "predicate:always",
    ),
    (
        "zig-root:triangular-band-window-tests",
        "compile:build.zig:build:triangular_band_window_tests",
        "predicate:always",
    ),
    (
        "zig-root:triangular-dense-unit-tests",
        "compile:build.zig:build:triangular_dense_unit_tests",
        "predicate:always",
    ),
    (
        "zig-root:triangular-packed-unit-tests",
        "compile:build.zig:build:triangular_packed_unit_tests",
        "predicate:always",
    ),
    (
        "zig-root:triangular-parallel-tests",
        "compile:build.zig:build:triangular_parallel_tests",
        "predicate:always",
    ),
    (
        "zig-root:vector-stride2-parallel-tests",
        "compile:build.zig:build:vector_stride2_parallel_tests",
        "predicate:always",
    ),
    (
        "zig-root:zynum-public-surface-contract-tests",
        "compile:build.zig:build:zynum_public_surface_contract_tests",
        "predicate:always",
    ),
)
REQUIRED_TEST_INVENTORY_BODY_LAUNCH_IDS = (
    "launch:build.zig:build:run_blas_module_tests",
    "launch:build.zig:build:run_blas_public_surface_contract_tests",
    "launch:build.zig:build:run_cblas_tests",
    "launch:build.zig:build:run_fortran_tests",
    "launch:build.zig:build:run_gemm_registry_tests",
    "launch:build.zig:build:run_header_smoke_tests",
    "launch:build.zig:build:run_level1_registry_tests",
    "launch:build.zig:build:run_level2_compact_registry_tests",
    "launch:build.zig:build:run_level2_fused_registry_tests",
    "launch:build.zig:build:run_modern_tests",
    "launch:build.zig:build:run_packed_parallel_tests",
    "launch:build.zig:build:run_structured_blocked_tests",
    "launch:build.zig:build:run",
    "launch:build.zig:build:run_symm_dense_gemm_tests",
    "launch:build.zig:build:run_triangular_band_solve_tests",
    "launch:build.zig:build:run_triangular_band_window_tests",
    "launch:build.zig:build:run_triangular_dense_unit_tests",
    "launch:build.zig:build:run_triangular_packed_unit_tests",
    "launch:build.zig:build:run_triangular_parallel_tests",
    "launch:build.zig:build:run_vector_stride2_parallel_tests",
    "launch:build.zig:build:run_zynum_public_surface_contract_tests",
)
PYTHON_TOOLING_CAPSULE_PROCESS_LAUNCH_IDS = (
    "python-launch:tools/check_test_inventory.py:_python_tooling_posix_capsule_probe:subprocess.run:1",
    "python-launch:tools/check_test_inventory.py:_run:subprocess.run:1",
    "python-launch:tools/check_test_inventory.py:_run:subprocess.run:2",
    "python-launch:tools/check_test_inventory.py:_run:subprocess.run:3",
)
REVIEWED_ARCHIVE_AND_NATIVE_FEATURE_PYTHON_LAUNCHES = {
    "python-launch:test/abi/baseline/test_package_archive.py:test_archive_cli_rejects_tracked_bytes_hidden_from_status:subprocess.check_output:1": {
        "launch_class": "archive-git-fixture-observation",
        "argv_shape": ["git", "status", "--porcelain=v1", "--", "payload"],
    },
    "python-launch:test/abi/baseline/test_package_archive.py:test_clean_archive_cli_matches_committed_tree_and_summary:subprocess.check_output:1": {
        "launch_class": "archive-git-fixture-observation",
        "argv_shape": ["git", "rev-parse", "HEAD"],
    },
    "python-launch:test/abi/baseline/test_package_archive.py:test_clean_archive_cli_matches_committed_tree_and_summary:subprocess.check_output:2": {
        "launch_class": "archive-git-fixture-observation",
        "argv_shape": [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            "<revision>",
            "--",
            "build.zig.zon",
            "payload",
        ],
    },
    "python-launch:test/abi/baseline/test_package_archive.py:test_clean_archive_cli_matches_committed_tree_and_summary:subprocess.check_output:3": {
        "launch_class": "archive-git-fixture-observation",
        "argv_shape": ["git", "show", "<revision>:<path>"],
    },
    "python-launch:test/build/test_build_inventory.py:test_native_feature_step_rejects_non_native_evidence_profiles:subprocess.run:1": {
        "launch_class": "native-feature-negative-build-guard",
        "argv_shape": [
            "<zig>",
            "build",
            "test-native-feature",
            "<negative-profile-options>",
            "--summary",
            "failures",
        ],
    },
}
REQUIRED_TEST_INVENTORY_PYTHON_LAUNCH_IDS = {
    "python-launch:test/build/test_build_inventory.py:test_unknown_target_configures_and_inventory_steps_fail_closed:subprocess.run:1",
    "python-launch:test/build/test_build_inventory.py:test_unknown_target_configures_and_inventory_steps_fail_closed:subprocess.run:2",
    *REVIEWED_ARCHIVE_AND_NATIVE_FEATURE_PYTHON_LAUNCHES,
    "python-launch:test/build/test_test_inventory.py:test_pending_gap_closure_and_default_cli_fail_closed:subprocess.run:1",
    "python-launch:test/build/test_test_inventory.py:test_pending_gap_closure_and_default_cli_fail_closed:subprocess.run:2",
    "python-launch:test/build/test_test_inventory.py:test_runner_protocol_and_isolated_object_mutations_fail:subprocess.run:1",
    "python-launch:test/build/test_test_inventory.py:test_runner_protocol_and_isolated_object_mutations_fail:subprocess.Popen:1",
    "python-launch:test/build/test_test_inventory.py:run_vector_inventory:subprocess.run:1",
    *PYTHON_TOOLING_CAPSULE_PROCESS_LAUNCH_IDS,
}
TEST_INVENTORY_RUNNER_COMPILE_PYTHON_LAUNCH_ID = (
    "python-launch:test/build/test_test_inventory.py:"
    "test_runner_protocol_and_isolated_object_mutations_fail:subprocess.run:1"
)
TEST_INVENTORY_RUNNER_EXECUTE_PYTHON_LAUNCH_ID = (
    "python-launch:test/build/test_test_inventory.py:"
    "run_vector_inventory:subprocess.run:1"
)
TEST_INVENTORY_RUNNER_RACE_PYTHON_LAUNCH_ID = (
    "python-launch:test/build/test_test_inventory.py:"
    "test_runner_protocol_and_isolated_object_mutations_fail:subprocess.Popen:1"
)
PYTHON_TOOLING_LAUNCH_ID = "launch:build.zig:build:python_tooling_tests"
PYTHON_TOOLING_STEP_ID = "step:build.zig:build:test-python-tooling"
PYTHON_TOOLING_STRUCTURE_BARRIER_ID = (
    "launch:build.zig:build:test_inventory_structure_check"
)
PYTHON_TOOLING_ROOT_ID = "python-root:benchmark-tools-discovery"
LEVEL2_WIDTH_DEFAULT_ARTIFACT_COMPILE_ID = (
    "compile:build.zig:build:level2_width_default_artifact_probe"
)
LEVEL2_WIDTH_DEFAULT_ARTIFACT_LAUNCH_ID = (
    "launch:build.zig:build:run_level2_width_default_artifact_probe"
)
LEVEL2_WIDTH_DEFAULT_ARTIFACT_STEP_ID = (
    "step:build.zig:build:test-level2-width-default-artifact"
)
LEVEL2_WIDTH_DEFAULT_ARTIFACT_LINK_ID = "link:build.zig:build:level2_width_default_artifact_probe_mod<-level2_width_isolated_library"
LEVEL2_WIDTH_DEFAULT_ARTIFACT_CONDITION = (
    "requested target architecture is x86_64 and default production profile is selected"
)
LEVEL2_WIDTH_ENABLED_ARTIFACT_COMPILE_ID = (
    "compile:build.zig:build:level2_width_enabled_artifact_probe"
)
LEVEL2_WIDTH_ENABLED_ARTIFACT_LAUNCH_ID = "launch:build.zig:build:run_probe"
LEVEL2_WIDTH_ENABLED_ARTIFACT_BUILD_STEP_ID = (
    "step:build.zig:build:build-level2-width-enabled-artifact"
)
LEVEL2_WIDTH_ENABLED_ARTIFACT_RUN_STEP_ID = (
    "step:build.zig:build:test-level2-width-enabled-artifact"
)
LEVEL2_WIDTH_ENABLED_ARTIFACT_UNSUPPORTED_STEP_ID = (
    "step:build.zig:build:unsupported_probe"
)
LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID = "link:build.zig:build:level2_width_enabled_artifact_probe_mod<-level2_width_isolated_library"
LEVEL2_WIDTH_ENABLED_ARTIFACT_CONDITION = "requested target is x86_64 with AVX-512F and the Level 2 width production profile is selected"
LEVEL2_WIDTH_ENABLED_ARTIFACT_UNSUPPORTED_CONDITION = "requested target is not x86_64 with AVX-512F or the Level 2 width production profile is not selected"
LEVEL2_WIDTH_ARTIFACT_CONTRACT_PATH = (
    "test/build/level2_width_artifact_probe_contract.zig"
)
LEVEL2_WIDTH_STUB_ROOT_PATH = "src/blas/level2_width_stub_root.zig"
REVIEWED_NEW_WORKFLOW_LAUNCH_FIELDS = {
    "workflow-launch:.github/workflows/ci.yml:source-checks:regenerate-compatibility-headers-and-kernel-coverage": {
        "condition": "always",
        "argv_shape": [
            "zig build generate-headers --summary failures",
            "zig build generate-kernel-coverage --summary failures",
        ],
        "evidence_role": "regenerate tracked references before drift validation",
    },
    "workflow-launch:.github/workflows/ci.yml:source-checks:check-generated-files-are-up-to-date": {
        "condition": "always",
        "argv_shape": [
            "git --no-pager diff --exit-code -- include/zynum/blas docs/kernel_coverage.json",
            "git status --porcelain --untracked-files=all -- include/zynum/blas docs/kernel_coverage.json",
        ],
        "evidence_role": "tracked generated-reference drift gate",
    },
    "workflow-launch:.github/workflows/ci.yml:target-tests:link-test-inventory-for-debug-target": {
        "condition": "matrix.zig_gate == 'link-only'",
        "argv_shape": [
            "zig",
            "build",
            "test-inventory-link",
            "<matrix.target_args>",
            "-Dtest-optimize=Debug",
            "-Dhost-tool-smoke=<matrix.host_tool_smoke>",
            "--summary",
            "failures",
        ],
        "evidence_role": "pending-platform-compile-link-only",
    },
    "workflow-launch:.github/workflows/ci.yml:target-tests:link-test-inventory-for-releasesafe-target": {
        "condition": "matrix.zig_gate == 'link-only'",
        "argv_shape": [
            "zig",
            "build",
            "--release=safe",
            "test-inventory-link",
            "<matrix.target_args>",
            "-Dtest-optimize=ReleaseSafe",
            "-Dhost-tool-smoke=<matrix.host_tool_smoke>",
            "--summary",
            "failures",
        ],
        "evidence_role": "pending-platform-compile-link-only",
    },
    "workflow-launch:.github/workflows/ci.yml:target-tests:link-test-inventory-for-releasefast-target": {
        "condition": "matrix.zig_gate == 'link-only'",
        "argv_shape": [
            "zig",
            "build",
            "--release=fast",
            "test-inventory-link",
            "<matrix.target_args>",
            "-Dtest-optimize=ReleaseFast",
            "-Dhost-tool-smoke=<matrix.host_tool_smoke>",
            "--summary",
            "failures",
        ],
        "evidence_role": "pending-platform-compile-link-only",
    },
    "workflow-launch:.github/workflows/ci.yml:feature-compile:run-matrix-gate": {
        "condition": "always",
        "argv_shape": ["<matrix.command>"],
        "evidence_role": "native-feature-correctness-or-explicit-build-only-by-matrix-row",
        "matrix_contract": {
            "host_native_correctness": [
                "macOS / host-native correctness",
                "Linux / host-native correctness",
            ],
            "host_native_command": "zig build test-native-feature -Dcpu=native -Dtest-optimize=ReleaseSafe --release=safe --summary failures",
            "build_only": [
                "macOS / Apple M4 SME2.1 build-only",
                "Linux / x86_64 v3 build-only",
                "Linux / x86_64 v4 build-only",
            ],
        },
    },
    "workflow-launch:.github/workflows/release.yml:artifacts:check-generated-files-are-up-to-date": {
        "condition": "always",
        "argv_shape": [
            "git --no-pager diff --exit-code -- include/zynum/blas docs/kernel_coverage.json",
            "git status --porcelain --untracked-files=all -- include/zynum/blas docs/kernel_coverage.json",
        ],
        "evidence_role": "release tracked generated-reference drift gate",
    },
    "workflow-launch:.github/workflows/release.yml:artifacts:require-clean-committed-checkout-before-archiving": {
        "condition": "always",
        "argv_shape": [
            "git rev-parse --verify HEAD^{commit}",
            "git --no-pager diff --exit-code",
            "git --no-pager diff --cached --exit-code",
            "git status --porcelain --untracked-files=all",
        ],
        "evidence_role": "release clean committed checkout gate",
    },
}
WINDOWS_PYTHON_TOOLING_FIXTURE_PATH = (
    "test/build/windows_python_tooling_probe_fixture.zig"
)
WINDOWS_PYTHON_TOOLING_FIXTURE_COMPILE_SOURCES = {
    "compile:build.zig:build:rank_k_probe": "bench/rank_k_probe.zig",
    "compile:build.zig:build:rotg_latency_probe": "bench/rotg_latency_probe.zig",
    "compile:build.zig:build:symm_probe": "bench/symm_probe.zig",
    "compile:build.zig:build:triangular_matrix_probe": "bench/triangular_matrix_probe.zig",
}
INSTALL_DYNAMIC_LIBRARY_ID = "install:build.zig:build:install_dynamic_lib"
WINDOWS_PYTHON_TOOLING_FIXTURE_INSTALL_IDS = (
    "install:build.zig:build:install_rank_k_probe",
    "install:build.zig:build:install_rotg_latency_probe",
    "install:build.zig:build:install_symm_probe",
    "install:build.zig:build:install_triangular_matrix_probe",
)
WINDOWS_PYTHON_TOOLING_INSTALL_IDS = (
    INSTALL_DYNAMIC_LIBRARY_ID,
    *WINDOWS_PYTHON_TOOLING_FIXTURE_INSTALL_IDS,
)
WINDOWS_PYTHON_TOOLING_INSTALL_REACHABILITY = {
    INSTALL_DYNAMIC_LIBRARY_ID: (
        "install or install-libraries step is reached, or the canonical native "
        "Windows x86_64 baseline test-python-tooling gate is reached"
    ),
    "install:build.zig:build:install_rank_k_probe": (
        "the build-rank-k-probe named step or the canonical native Windows "
        "x86_64 baseline test-python-tooling gate is reached"
    ),
    "install:build.zig:build:install_rotg_latency_probe": (
        "the build-rotg-latency-probe named step or the canonical native Windows "
        "x86_64 baseline test-python-tooling gate is reached"
    ),
    "install:build.zig:build:install_symm_probe": (
        "the build-symm-probe named step or the canonical native Windows x86_64 "
        "baseline test-python-tooling gate is reached"
    ),
    "install:build.zig:build:install_triangular_matrix_probe": (
        "the build-triangular-matrix-probe named step or the canonical native "
        "Windows x86_64 baseline test-python-tooling gate is reached"
    ),
}
INSTALL_STATIC_LIBRARY_ID = "install:build.zig:build:install_static_lib"
INSTALL_LIBRARIES_STEP_ID = "step:build.zig:build:install-libraries"
WINDOWS_EXCLUDED_DEFAULT_EXECUTABLE_INSTALL_IDS = {
    "install:build.zig:build:bench",
    "install:build.zig:build:gemm_sweep",
    "install:build.zig:build:vector_matrix_sweep",
    "install:build.zig:build:level1_probe",
    "install:build.zig:build:dcopy_probe",
}
NEW_REVIEWED_TEST_INFRASTRUCTURE_CLASSIFICATIONS = {
    LEVEL2_WIDTH_ARTIFACT_CONTRACT_PATH,
    "test/build/level2_width_enabled_artifact_probe.zig",
    WINDOWS_PYTHON_TOOLING_FIXTURE_PATH,
}
PYTHON_TOOLING_CHECKER_PATH = "tools/check_test_inventory.py"
PYTHON_TOOLING_INVENTORY_PATH = "tools/test_inventory.json"


def _compact_zig_contract(source: str) -> str:
    """Remove trivia while retaining literals for an exact local Zig contract."""
    output: list[str] = []
    index = 0
    state = "code"
    quote = ""
    escaped = False
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "line-comment":
            if char == "\n":
                state = "code"
            index += 1
            continue
        if state == "block-comment":
            if char == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if state == "literal":
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                state = "code"
            index += 1
            continue
        if char == "/" and following == "/":
            state = "line-comment"
            index += 2
        elif char == "/" and following == "*":
            state = "block-comment"
            index += 2
        elif char in {'"', "'"}:
            state = "literal"
            quote = char
            output.append(char)
            index += 1
        elif char.isspace():
            index += 1
        else:
            output.append(char)
            index += 1
    return "".join(output)


def _python_tooling_argument_contract() -> dict[str, Any]:
    return {
        "cardinality": "one-explicit-checker-invocation",
        "ordered_argv": [
            {"kind": "literal", "value": "python3"},
            {"kind": "literal", "value": "-B"},
            {
                "kind": "repository-file-argument",
                "value": PYTHON_TOOLING_CHECKER_PATH,
            },
            {"kind": "literal", "value": "--root"},
            {"kind": "repository-directory-argument", "value": "."},
            {"kind": "literal", "value": "--inventory"},
            {
                "kind": "repository-file-argument",
                "value": PYTHON_TOOLING_INVENTORY_PATH,
            },
            {"kind": "literal", "value": "--run-python-tooling-root"},
            {"kind": "literal", "value": PYTHON_TOOLING_ROOT_ID},
        ],
        "conditional_ordered_argv_suffix": {
            "condition": "canonical native Windows x86_64 baseline target",
            "ordered_argv": [
                {
                    "kind": "literal",
                    "value": "--windows-zynum-blas-build-output",
                },
                {
                    "kind": "emitted-artifact-file-argument",
                    "value": "compile:build.zig:build:lib",
                },
                {
                    "kind": "literal",
                    "value": "--windows-zynum-blas-installed-output",
                },
                {
                    "kind": "installed-artifact-path-argument",
                    "value": "zig-out/bin/zynum_blas.dll",
                },
            ],
        },
    }


def _python_tooling_launch_template() -> dict[str, Any]:
    return {
        "owner": "test-infrastructure",
        "detail_status": "process-lifecycle-out-of-scope",
        "cwd_shape": "repository-root",
        "command_shape": "system-command",
        "source_artifact": None,
        "compile_for": "host",
        "execute_on": "host",
        "argv_shape": [
            "python3",
            "-B",
            PYTHON_TOOLING_CHECKER_PATH,
            "--root",
            ".",
            "--inventory",
            PYTHON_TOOLING_INVENTORY_PATH,
            "--run-python-tooling-root",
            PYTHON_TOOLING_ROOT_ID,
        ],
        "launch_class": "validation",
        "launch_role": "inventory-declared-python-tooling-tests",
        "inventory_root_id": PYTHON_TOOLING_ROOT_ID,
        "argument_contract": _python_tooling_argument_contract(),
        "checker_script": {
            "path": PYTHON_TOOLING_CHECKER_PATH,
            "path_source": "b.pathFromRoot",
            "argument_kind": "repository-file",
        },
        "inventory_file": {
            "path": PYTHON_TOOLING_INVENTORY_PATH,
            "path_source": "b.pathFromRoot",
            "argument_kind": "repository-file",
        },
        "test_inventory_barrier": {
            "dependency_step_id": PYTHON_TOOLING_STRUCTURE_BARRIER_ID,
            "relation": "direct-step-dependency-before-python-discovery",
        },
        "windows_artifact_contract": {
            "condition": "canonical native Windows x86_64 baseline target",
            "emitted_artifact_id": "compile:build.zig:build:lib",
            "installed_artifact_path": "zig-out/bin/zynum_blas.dll",
            "ordered_dependency_ids": list(WINDOWS_PYTHON_TOOLING_INSTALL_IDS),
            "dependency_relation": "direct-step-dependencies-before-python-discovery",
        },
    }


def _python_tooling_step_template() -> dict[str, Any]:
    return {
        "owner": "build-composition",
        "description": "Run inventory-declared Python benchmark tooling unit tests",
        "direct_dependencies": [
            {
                "id": PYTHON_TOOLING_LAUNCH_ID,
                "condition": "always",
            }
        ],
        "aggregate_test_membership": "member",
        "aggregate_condition": "always",
        "intentional_orphan": False,
        "orphan_reason": "direct dependency of the canonical correctness aggregate",
        "step_role": "focused-validation",
        "closure_contract": {
            "launch_observation_id": PYTHON_TOOLING_LAUNCH_ID,
            "launch_count": 1,
            "relation": "only-direct-dependency",
        },
    }


def _annotate_python_tooling_tests(
    text: str, observations: list[dict[str, Any]]
) -> None:
    by_id = {item["id"]: item for item in observations}
    tooling_ids = {
        PYTHON_TOOLING_LAUNCH_ID,
        PYTHON_TOOLING_STEP_ID,
    }
    tooling_declaration = re.search(
        r"\b(?:const|var)\s+(?:python_tooling_tests|python_tooling_test_step)\b",
        _code_mask(text),
    )
    if not tooling_ids.intersection(by_id) and tooling_declaration is None:
        return

    required_ids = {
        *tooling_ids,
        PYTHON_TOOLING_STRUCTURE_BARRIER_ID,
        TEST_INVENTORY_AGGREGATE_STEP_ID,
    }
    missing = required_ids - set(by_id)
    if missing:
        raise InventoryError(
            f"Python tooling test observations are incomplete: {sorted(missing)}"
        )

    launch_calls = [
        call
        for position, call in _calls(text, "b.addSystemCommand")
        if _symbol_before(text, position, call) == "python_tooling_tests"
    ]
    expected_launch_call = (
        'b.addSystemCommand(&.{"python3","-B",'
        f'b.pathFromRoot("{PYTHON_TOOLING_CHECKER_PATH}"),'
        '"--root",b.pathFromRoot("."),"--inventory",'
        f'b.pathFromRoot("{PYTHON_TOOLING_INVENTORY_PATH}"),'
        f'"--run-python-tooling-root","{PYTHON_TOOLING_ROOT_ID}",}})'
    )
    if len(launch_calls) != 1 or _compact_zig_contract(launch_calls[0]) != (
        expected_launch_call
    ):
        raise InventoryError(
            "Python tooling launch must preserve the exact explicit checker argv contract"
        )

    exact_calls = {
        "python_tooling_tests.setCwd": (
            'python_tooling_tests.setCwd(b.path("."))',
            "repository working directory",
        ),
    }
    for token, (expected, subject) in exact_calls.items():
        call = _single_zig_call(text, token, f"Python tooling {subject}")
        if _compact_zig_contract(call) != expected:
            raise InventoryError(f"Python tooling launch has an incorrect {subject}")

    environment_calls = [
        _first_string(call)
        for _, call in _calls(text, "python_tooling_tests.removeEnvironmentVariable")
    ]
    if environment_calls != [
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
        "GIT_PAGER",
        "PAGER",
        "LESS",
    ]:
        raise InventoryError(
            "Python tooling launch must remove the exact reviewed Python environment variables"
        )

    dependency_calls = [
        _compact_zig_contract(call) + ";"
        for _, call in _calls(text, "python_tooling_tests.step.dependOn")
    ]
    expected_dependency_calls = [
        "python_tooling_tests.step.dependOn(&test_inventory_structure_check.step);",
        *(
            f"python_tooling_tests.step.dependOn(&{identifier.rsplit(':', 1)[-1]}.step);"
            for identifier in WINDOWS_PYTHON_TOOLING_INSTALL_IDS
        ),
    ]
    if dependency_calls != expected_dependency_calls:
        raise InventoryError(
            "Python tooling launch must preserve the exact structure and Windows artifact dependency closure"
        )

    windows_contract = (
        "if(native_canonical_windows_python_tooling){"
        'python_tooling_tests.addArg("--windows-zynum-blas-build-output");'
        "python_tooling_tests.addFileArg(lib.getEmittedBin());"
        'python_tooling_tests.addArg("--windows-zynum-blas-installed-output");'
        'python_tooling_tests.addArg(b.getInstallPath(.bin,"zynum_blas.dll"));'
        "python_tooling_tests.step.dependOn(&install_dynamic_lib.step);"
        "python_tooling_tests.step.dependOn(&install_rank_k_probe.step);"
        "python_tooling_tests.step.dependOn(&install_rotg_latency_probe.step);"
        "python_tooling_tests.step.dependOn(&install_symm_probe.step);"
        "python_tooling_tests.step.dependOn(&install_triangular_matrix_probe.step);"
        "}"
    )
    if _compact_zig_contract(text).count(windows_contract) != 1:
        raise InventoryError(
            "Python tooling launch must preserve the exact ordered Windows artifact argv and dependency contract"
        )

    source_relations = (
        (
            "python_tooling_test_step.dependOn",
            "python_tooling_test_step.dependOn(&python_tooling_tests.step);",
            "Python tooling named step must close over exactly its launch",
        ),
    )
    for token, relation, message in source_relations:
        calls = [_compact_zig_contract(call) + ";" for _, call in _calls(text, token)]
        if calls != [relation]:
            raise InventoryError(message)
    aggregate_calls = [
        _compact_zig_contract(call) + ";"
        for _, call in _calls(text, "test_step.dependOn")
        if "python_tooling" in call
    ]
    if aggregate_calls != ["test_step.dependOn(python_tooling_test_step);"]:
        raise InventoryError(
            "canonical test aggregate must depend directly on the Python tooling named step"
        )

    by_id[PYTHON_TOOLING_LAUNCH_ID].update(_python_tooling_launch_template())
    by_id[PYTHON_TOOLING_STEP_ID].update(_python_tooling_step_template())


def _test_inventory_enumeration_projection(text: str) -> dict[str, Any]:
    masked = _code_mask(text)
    compact_source = _compact_zig_contract(text)
    required_source_contracts = (
        "constTestInventoryProfile=struct{environment_id:[]constu8,enumeration_class_id:[]constu8,};",
        "consttarget_query=b.standardTargetOptionsQueryOnly(.{});",
        "consttarget=b.resolveTargetQuery(target_query);",
        "constexact_baseline_request=target_query.cpu_model==.baselineandtarget_query.cpu_features_add.isEmpty()andtarget_query.cpu_features_sub.isEmpty();",
        "constexpected_baseline_cpu=std.Target.Cpu.baseline(target.result.cpu.arch,target.result.os);",
        "constresolved_cpu_matches_canonical_baseline=target.result.cpu.model==expected_baseline_cpu.modelandtarget.result.cpu.features.eql(expected_baseline_cpu.features);",
    )
    if any(
        compact_source.count(contract) != 1 for contract in required_source_contracts
    ):
        raise InventoryError(
            "test inventory CPU profile gate must preserve query provenance and canonical baseline resolved features"
        )
    declaration = re.search(
        r"\bconst\s+inventory_profile\s*:\s*\?\s*TestInventoryProfile\s*=",
        masked,
    )
    if declaration is None:
        raise InventoryError(
            "test inventory target CPU profile mapping must be explicitly optional"
        )
    statement_end = _zig_statement_end(masked, declaration.end(), len(masked))
    compact = _compact_zig_contract(text[declaration.start() : statement_end + 1])
    if not compact.endswith("elsenull;"):
        raise InventoryError(
            "test inventory target CPU profile mapping must use a null fallback"
        )
    expected = (
        "constinventory_profile:?TestInventoryProfile=if(exact_baseline_requestandresolved_cpu_matches_canonical_baselineandtarget.result.cpu.arch==.aarch64andtarget.result.cpu.model==&std.Target.aarch64.cpu.apple_m1andtarget.result.os.tag==.macosandtarget.result.abi==.noneandtarget.result.ofmt==.macho)"
        '.{.environment_id="env:aarch64-macos-baseline",.enumeration_class_id="enumeration-class:aarch64-macos-system-macho",}'
        "elseif(exact_baseline_requestandresolved_cpu_matches_canonical_baselineandtarget.result.cpu.arch==.x86_64andtarget.result.cpu.model==&std.Target.x86.cpu.x86_64andtarget.result.os.tag==.linuxandtarget.result.abi==.gnuandtarget.result.ofmt==.elf)"
        '.{.environment_id="env:x86-64-linux-gnu-baseline",.enumeration_class_id="enumeration-class:x86-64-linux-gnu-elf",}'
        "elseif(exact_baseline_requestandresolved_cpu_matches_canonical_baselineandtarget.result.cpu.arch==.aarch64andtarget.result.cpu.model==&std.Target.aarch64.cpu.genericandtarget.result.os.tag==.linuxandtarget.result.abi==.gnuandtarget.result.ofmt==.elf)"
        '.{.environment_id="env:aarch64-linux-gnu-baseline",.enumeration_class_id="enumeration-class:aarch64-linux-gnu-elf",}'
        "elseif(exact_baseline_requestandresolved_cpu_matches_canonical_baselineandtarget.result.cpu.arch==.x86_64andtarget.result.cpu.model==&std.Target.x86.cpu.x86_64andtarget.result.os.tag==.windowsandtarget.result.abi==.gnuandtarget.result.ofmt==.coff)"
        '.{.environment_id="env:x86-64-windows-gnu-baseline",.enumeration_class_id="enumeration-class:x86-64-windows-gnu-coff",}'
        "elsenull;"
    )
    if compact != expected:
        raise InventoryError(
            "test inventory target CPU profile mapping must preserve the exact four baseline mappings"
        )
    return {
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
            dict(mapping) for mapping in REQUIRED_TEST_INVENTORY_ENUMERATION_MAPPINGS
        ],
        "fallback": None,
        "known_branch": "test-inventory-enumerator-factory",
        "unknown_branch_dependency": TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID,
    }


def _single_zig_call(text: str, token: str, subject: str) -> str:
    calls = _calls(text, token)
    if len(calls) != 1:
        raise InventoryError(
            f"test inventory factory requires exactly one {subject}, found {len(calls)}"
        )
    return calls[0][1]


def _test_inventory_case_predicate(text: str, logical_symbol: str) -> str:
    masked = _code_mask(text)
    declaration = re.search(
        rf"\bconst\s+{re.escape(logical_symbol)}\s*=\s*if\s*\(", masked
    )
    if declaration is None:
        return "predicate:always"
    open_index = masked.find("(", declaration.start(), declaration.end())
    close_index = _matching_paren(text, open_index)
    condition = " ".join(text[open_index + 1 : close_index].split())
    if condition == "target.result.cpu.arch == .x86_64":
        return "predicate:arch-x86-64"
    raise InventoryError(
        f"test inventory factory cannot normalize predicate for {logical_symbol}: {condition}"
    )


def _test_inventory_argument_contract(loop_text: str) -> dict[str, Any]:
    file_call = _single_zig_call(
        loop_text,
        "run_inventory_tests.addFileArg",
        "inventory file argument",
    )
    file_match = re.fullmatch(
        r'run_inventory_tests\.addFileArg\s*\(\s*b\.path\s*\(\s*"([^"\\]+)"\s*\)\s*\)',
        file_call,
    )
    if file_match is None:
        raise InventoryError("test inventory factory file argument is unsupported")
    args_call = _single_zig_call(
        loop_text, "run_inventory_tests.addArgs", "inventory argument vector"
    )
    body_match = re.fullmatch(
        r"run_inventory_tests\.addArgs\s*\(\s*&\.\{([\s\S]*)\}\s*\)",
        args_call,
    )
    if body_match is None:
        raise InventoryError("test inventory factory argument vector is unsupported")
    tokens = [token for token in _split_top_level(body_match.group(1)) if token]
    expected = [
        '"--inventory-root"',
        "inventory_case.root_id",
        '"--inventory-mode"',
        "@tagName(test_optimize)",
        '"--inventory-environment"',
        "resolved_inventory_profile.environment_id",
        '"--inventory-class"',
        "resolved_inventory_profile.enumeration_class_id",
    ]
    if tokens != expected:
        raise InventoryError(
            "test inventory factory argument vector must bind root, mode, environment, and enumeration class exactly"
        )
    return {
        "cardinality": "one-launch-per-applicable-expansion-case",
        "ordered_argv": [
            {"kind": "repository-path", "value": file_match.group(1)},
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


def _test_inventory_argv_shape(argument_contract: dict[str, Any]) -> list[str]:
    """Project the exact factory argument contract into reviewed argv notation."""
    token_shapes = {
        ("repository-path", "tools/test_inventory.json"): "tools/test_inventory.json",
        ("literal", "--inventory-root"): "--inventory-root",
        ("expansion-field", "root_id"): "<root-id>",
        ("literal", "--inventory-mode"): "--inventory-mode",
        ("build-option-tag", "test-optimize"): "<test-optimize>",
        ("literal", "--inventory-environment"): "--inventory-environment",
        ("target-derived", "environment-id"): "<environment-id>",
        ("literal", "--inventory-class"): "--inventory-class",
        ("target-derived", "enumeration-class-id"): "<enumeration-class-id>",
    }
    ordered_argv = argument_contract.get("ordered_argv")
    if not isinstance(ordered_argv, list):
        raise InventoryError(
            "test inventory factory ordered argument contract is malformed"
        )
    result: list[str] = []
    for token in ordered_argv:
        if not isinstance(token, dict) or set(token) != {"kind", "value"}:
            raise InventoryError(
                "test inventory factory ordered argument token is malformed"
            )
        shape = token_shapes.get((token["kind"], token["value"]))
        if shape is None:
            raise InventoryError(
                "test inventory factory ordered argument token is unsupported"
            )
        result.append(shape)
    if len(result) != len(token_shapes):
        raise InventoryError(
            "test inventory factory ordered argument contract has the wrong cardinality"
        )
    return result


def _annotate_test_inventory_factory(
    text: str, observations: list[dict[str, Any]]
) -> None:
    """Bind the one-call Zig factory to every logical test root it expands."""
    by_id = {item["id"]: item for item in observations}
    required_ids = {
        TEST_INVENTORY_FACTORY_COMPILE_ID,
        TEST_INVENTORY_FACTORY_LAUNCH_ID,
        TEST_INVENTORY_LINK_STEP_ID,
        TEST_INVENTORY_RUN_STEP_ID,
        TEST_INVENTORY_AGGREGATE_STEP_ID,
        TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID,
    }
    missing = required_ids - set(by_id)
    if missing:
        raise InventoryError(
            f"test inventory factory observations are incomplete: {sorted(missing)}"
        )

    masked = _code_mask(text)
    array_match = re.search(
        r"\bconst\s+inventory_cases\s*=\s*\[(\d+)\]InventoryCase\s*\{",
        masked,
    )
    if array_match is None:
        raise InventoryError("test inventory factory cases array is missing")
    array_open = masked.find("{", array_match.start(), array_match.end())
    array_close = _matching_brace(masked, array_open)
    array_body = text[array_open + 1 : array_close]
    entry_pattern = re.compile(
        r"\.\{\s*\.root_id\s*=\s*\"(?P<root>[^\"\\]+)\"\s*,"
        r"\s*\.logical_tests\s*=\s*(?P<logical>[A-Za-z_][A-Za-z0-9_]*)"
        r"(?:\s*,\s*\.predicate_id\s*=\s*\"(?P<predicate>[^\"\\]+)\")?\s*,?\s*\}",
        re.DOTALL,
    )
    matches = list(entry_pattern.finditer(array_body))
    residue = _code_mask(entry_pattern.sub("", array_body))
    if residue.strip(" \t\r\n,"):
        raise InventoryError("test inventory factory cases contain unsupported syntax")
    declared_count = int(array_match.group(1))
    if declared_count != len(matches) or declared_count != 21:
        raise InventoryError(
            "test inventory factory must declare exactly 21 expansion cases"
        )

    expansion_cases: list[dict[str, str]] = []
    for match in matches:
        root_id = match.group("root")
        logical_symbol = match.group("logical")
        logical_id = f"compile:build.zig:build:{logical_symbol}"
        logical = by_id.get(logical_id)
        if logical is None or logical.get("call") != "b.addTest":
            raise InventoryError(
                f"test inventory case {root_id} does not reference a logical b.addTest observation"
            )
        predicate_id = match.group("predicate") or _test_inventory_case_predicate(
            text, logical_symbol
        )
        expansion_cases.append(
            {
                "root_id": root_id,
                "logical_compile_observation_id": logical_id,
                "predicate_id": predicate_id,
            }
        )
    root_ids = [item["root_id"] for item in expansion_cases]
    logical_ids = [item["logical_compile_observation_id"] for item in expansion_cases]
    if root_ids != sorted(root_ids) or len(set(root_ids)) != 21:
        raise InventoryError(
            "test inventory factory root ids must be unique and canonically sorted"
        )
    if len(set(logical_ids)) != 21:
        raise InventoryError(
            "test inventory factory logical compile observations must be unique"
        )
    actual_case_contract = tuple(
        (
            item["root_id"],
            item["logical_compile_observation_id"],
            item["predicate_id"],
        )
        for item in expansion_cases
    )
    if actual_case_contract != REQUIRED_TEST_INVENTORY_FACTORY_CASES:
        raise InventoryError(
            "test inventory factory cases must preserve the exact 21 logical root bindings"
        )

    if len(REQUIRED_TEST_INVENTORY_BODY_LAUNCH_IDS) != len(expansion_cases):
        raise InventoryError(
            "test inventory official body launch contract has the wrong cardinality"
        )
    normalized_build = " ".join(_code_mask(text).split())
    body_launches: list[dict[str, str]] = []
    for case, launch_id in zip(
        expansion_cases, REQUIRED_TEST_INVENTORY_BODY_LAUNCH_IDS, strict=True
    ):
        launch = by_id.get(launch_id)
        if launch is None or launch.get("call") != "b.addRunArtifact":
            raise InventoryError(
                f"test inventory official body launch is missing: {launch_id}"
            )
        launch_symbol = launch["anchor"]["symbol"]
        dependency = f"{launch_symbol}.step.dependOn(test_inventory_step);"
        if normalized_build.count(dependency) != 1:
            raise InventoryError(
                "test inventory official Zig body launch barrier is missing required "
                f"relation: {launch_id}"
            )
        if case["predicate_id"] == "predicate:arch-x86-64":
            guarded_dependency = (
                "if (run_structured_object_tests) |run| "
                "run.step.dependOn(test_inventory_step);"
            )
            if normalized_build.count(guarded_dependency) != 1:
                raise InventoryError(
                    "test inventory conditional official Zig body launch barrier must "
                    "preserve the x86_64 applicability guard"
                )
        barrier = {
            "dependency_step_id": TEST_INVENTORY_RUN_STEP_ID,
            "relation": "direct-step-dependency-before-body-execution",
        }
        launch["test_inventory_barrier"] = barrier
        body_launches.append(
            {
                **case,
                "launch_observation_id": launch_id,
            }
        )

    known_branch_match = re.search(
        r"\bif\s*\(\s*inventory_profile\s*\)\s*"
        r"\|\s*resolved_inventory_profile\s*\|\s*\{",
        masked[array_close:],
    )
    if known_branch_match is None:
        raise InventoryError("test inventory factory known target branch is missing")
    known_branch_open = array_close + masked[array_close:].find(
        "{", known_branch_match.start(), known_branch_match.end()
    )
    known_branch_close = _matching_brace(masked, known_branch_open)
    loop_match = re.search(
        r"\bfor\s*\(\s*inventory_cases\s*\)\s*\|inventory_case\|\s*\{",
        masked[known_branch_open + 1 : known_branch_close],
    )
    if loop_match is None:
        raise InventoryError("test inventory factory expansion loop is missing")
    loop_open = (
        known_branch_open
        + 1
        + masked[known_branch_open + 1 : known_branch_close].find(
            "{", loop_match.start(), loop_match.end()
        )
    )
    loop_close = _matching_brace(masked, loop_open)
    loop_text = text[loop_open + 1 : loop_close]
    normalized_loop = " ".join(_code_mask(loop_text).split())
    required_fragments = (
        "const official_tests = inventory_case.logical_tests orelse continue;",
        ".root_module = official_tests.root_module,",
        ".path = b.path( ),",
        ".mode = .simple,",
        "const run_inventory_tests = b.addRunArtifact(inventory_tests);",
        "test_inventory_link_step.dependOn(&inventory_tests.step);",
        "test_inventory_step.dependOn(&run_inventory_tests.step);",
    )
    for fragment in required_fragments:
        if fragment not in normalized_loop:
            raise InventoryError(
                f"test inventory factory loop is missing required relation: {fragment}"
            )
    if "test_step.dependOn(test_inventory_step);" not in " ".join(
        _code_mask(text).split()
    ):
        raise InventoryError(
            "canonical test aggregate must depend on the test inventory run step"
        )
    add_test_call = _single_zig_call(loop_text, "b.addTest", "enumerator compile")
    if (
        'b.fmt("inventory-{s}", .{inventory_case.root_id})' not in add_test_call
        or ".root_module = official_tests.root_module" not in add_test_call
    ):
        raise InventoryError(
            "test inventory factory compile must preserve logical name and root-module pointer"
        )
    runner_match = re.search(
        r"\.test_runner\s*=\s*\.\{[\s\S]*?\.path\s*=\s*b\.path\s*\(\s*\"([^\"\\]+)\"\s*\)\s*,"
        r"[\s\S]*?\.mode\s*=\s*\.([A-Za-z_][A-Za-z0-9_]*)\s*,?[\s\S]*?\}",
        add_test_call,
    )
    if runner_match is None:
        raise InventoryError(
            "test inventory factory runner specification is unsupported"
        )
    run_call = _single_zig_call(
        loop_text, "b.addRunArtifact", "enumerator run artifact"
    )
    if not re.fullmatch(r"b\.addRunArtifact\s*\(\s*inventory_tests\s*\)", run_call):
        raise InventoryError(
            "test inventory factory launch must consume the factory compile artifact"
        )

    unknown_branch_match = re.match(r"\s*else\s*\{", masked[known_branch_close + 1 :])
    if unknown_branch_match is None:
        raise InventoryError("test inventory factory unknown target branch is missing")
    unknown_branch_open = (
        known_branch_close
        + 1
        + masked[known_branch_close + 1 :].find(
            "{", unknown_branch_match.start(), unknown_branch_match.end()
        )
    )
    unknown_branch_close = _matching_brace(masked, unknown_branch_open)
    unknown_branch_text = text[unknown_branch_open + 1 : unknown_branch_close]
    fail_call = _single_zig_call(
        unknown_branch_text, "b.addFail", "unknown-target failure step"
    )
    if not re.fullmatch(
        r'b\.addFail\s*\(\s*"'
        + re.escape(TEST_INVENTORY_UNSUPPORTED_TARGET_MESSAGE)
        + r'"\s*,?\s*\)',
        fail_call,
    ):
        raise InventoryError(
            "test inventory unknown target failure message must be generic and exact"
        )
    normalized_unknown_branch = " ".join(_code_mask(unknown_branch_text).split())
    unknown_dependencies = (
        "test_inventory_link_step.dependOn(&unsupported_test_inventory_target.step);",
        "test_inventory_step.dependOn(&unsupported_test_inventory_target.step);",
    )
    if any(
        normalized_unknown_branch.count(dependency) != 1
        for dependency in unknown_dependencies
    ):
        raise InventoryError(
            "test inventory unknown target branch must bind one shared failure dependency to both inventory steps"
        )

    compile_observation = by_id[TEST_INVENTORY_FACTORY_COMPILE_ID]
    compile_observation.update(
        {
            "artifact_role": "test-inventory-enumerator-factory",
            "expansion_relation": "one-per-applicable-logical-zig-root",
            "root_module_relation": "same-pointer",
            "test_runner": {
                "path": runner_match.group(1),
                "mode": runner_match.group(2),
            },
            "expansion_cases": expansion_cases,
            "expansion_case_count": len(expansion_cases),
            "expansion_cases_digest": _json_fact_digest(expansion_cases),
            "enumeration_class_projection": _test_inventory_enumeration_projection(
                text
            ),
        }
    )
    argument_contract = _test_inventory_argument_contract(loop_text)
    by_id[TEST_INVENTORY_FACTORY_LAUNCH_ID].update(
        {
            "launch_role": "test-inventory-enumerator-factory-run",
            "source_factory": TEST_INVENTORY_FACTORY_COMPILE_ID,
            "argument_contract": argument_contract,
            "argv_shape": _test_inventory_argv_shape(argument_contract),
        }
    )
    by_id[TEST_INVENTORY_LINK_STEP_ID].update(
        {
            "step_role": "test-inventory-enumerator-link",
            "direct_dependencies": [
                {
                    "id": TEST_INVENTORY_FACTORY_COMPILE_ID,
                    "condition": "per applicable expansion case for an exact baseline target CPU profile",
                },
                {
                    "id": TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID,
                    "condition": "unknown or nonbaseline target CPU profile",
                },
            ],
        }
    )
    by_id[TEST_INVENTORY_RUN_STEP_ID].update(
        {
            "step_role": "test-inventory-enumerator-run",
            "direct_dependencies": [
                {
                    "id": TEST_INVENTORY_FACTORY_LAUNCH_ID,
                    "condition": "per applicable expansion case for an exact baseline target CPU profile",
                },
                {
                    "id": TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID,
                    "condition": "unknown or nonbaseline target CPU profile",
                },
            ],
            "official_body_launch_barrier": {
                "relation": "every-applicable-official-body-launch-depends-on-this-step",
                "launches": body_launches,
                "launch_count": len(body_launches),
                "launches_digest": _json_fact_digest(body_launches),
            },
        }
    )
    by_id[TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID].update(
        {
            "step_role": "unsupported-test-inventory-target-failure",
            "direct_dependencies": [],
            "error_message": TEST_INVENTORY_UNSUPPORTED_TARGET_MESSAGE,
        }
    )


def _annotate_native_feature_test_contract(
    text: str, observations: list[dict[str, Any]]
) -> None:
    """Bind correctness-only feature runs to the same logical test Compile pointers."""
    by_id = {item["id"]: item for item in observations}
    required_ids = {
        NATIVE_FEATURE_STEP_ID,
        NATIVE_FEATURE_LAUNCH_ID,
        *(identifier for identifier, _, _ in NATIVE_FEATURE_GUARDS),
    }
    missing = required_ids - set(by_id)
    if missing:
        raise InventoryError(
            f"native feature test observations are incomplete: {sorted(missing)}"
        )
    if by_id[NATIVE_FEATURE_STEP_ID].get("call") != "b.step":
        raise InventoryError(
            "native feature entry point must remain a named build step"
        )
    if by_id[NATIVE_FEATURE_LAUNCH_ID].get("call") != "b.addRunArtifact":
        raise InventoryError("native feature body launch must remain a run artifact")
    for identifier, _, _ in NATIVE_FEATURE_GUARDS:
        if by_id[identifier].get("call") != "b.addFail":
            raise InventoryError(
                f"native feature guard must remain fail-closed: {identifier}"
            )

    masked = _code_mask(text)
    feature_start = masked.find("const test_native_feature_step")
    if feature_start < 0:
        raise InventoryError("native feature step source block is missing")
    loop_match = re.search(
        r"\bfor\s*\(\s*inventory_cases\s*\)\s*\|inventory_case\|\s*\{",
        masked[feature_start:],
    )
    if loop_match is None:
        raise InventoryError("native feature expansion loop is missing")
    loop_open = feature_start + masked[feature_start:].find(
        "{", loop_match.start(), loop_match.end()
    )
    loop_close = _matching_brace(masked, loop_open)
    feature_source = text[feature_start : loop_close + 1]
    normalized_feature = " ".join(feature_source.split())
    normalized_code = " ".join(_code_mask(feature_source).split())

    exact_source_contracts = (
        "test_native_feature_step.dependOn(&test_inventory_structure_check.step);",
        "const explicit_non_baseline_cpu_profile = switch (target_query.cpu_model) { .native, .explicit => true, .baseline, .determined_by_arch_os => false, };",
        "const native_feature_target_matches_host = target.result.cpu.arch == b.graph.host.result.cpu.arch and target.result.os.tag == b.graph.host.result.os.tag and target.result.abi == b.graph.host.result.abi and target.result.ofmt == b.graph.host.result.ofmt;",
        "const native_feature_external_executor_enabled = b.enable_qemu or b.enable_rosetta or b.enable_wine or b.enable_darling or b.enable_wasmtime;",
    )
    for contract in exact_source_contracts:
        if normalized_feature.count(contract) != 1:
            raise InventoryError(
                f"native feature source guard contract changed: {contract}"
            )
    ordered_guard_fragments = (
        "if (!explicit_non_baseline_cpu_profile)",
        "else if (!native_feature_target_matches_host)",
        "else if (native_feature_external_executor_enabled)",
        "else if (!b.graph.host.result.cpu.features.isSuperSetOf(target.result.cpu.features))",
        "else &test_inventory_structure_check.step;",
    )
    positions = [normalized_code.find(fragment) for fragment in ordered_guard_fragments]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise InventoryError(
            "native feature profile guard order or success dependency changed"
        )
    for fragment in ordered_guard_fragments:
        if normalized_code.count(fragment) != 1:
            raise InventoryError(
                f"native feature profile guard must occur exactly once: {fragment}"
            )
    for _, message, _ in NATIVE_FEATURE_GUARDS:
        if normalized_feature.count(f'b.addFail("{message}").step') != 1:
            raise InventoryError(
                f"native feature fail-closed guard message changed: {message}"
            )

    loop_text = text[loop_open + 1 : loop_close]
    normalized_loop = " ".join(_code_mask(loop_text).split())
    loop_contracts = (
        "const official_tests = inventory_case.logical_tests orelse continue;",
        "const run_native_feature_tests = b.addRunArtifact(official_tests);",
        "run_native_feature_tests.step.dependOn(native_feature_profile_guard);",
        "test_native_feature_step.dependOn(&run_native_feature_tests.step);",
    )
    for contract in loop_contracts:
        if normalized_loop.count(contract) != 1:
            raise InventoryError(
                f"native feature expansion loop relation changed: {contract}"
            )
    run_call = _single_zig_call(
        loop_text, "b.addRunArtifact", "native feature run artifact"
    )
    if not re.fullmatch(r"b\.addRunArtifact\s*\(\s*official_tests\s*\)", run_call):
        raise InventoryError(
            "native feature launch must consume the same logical test Compile pointer"
        )
    forbidden = (
        "test_inventory_runner",
        "inventory_tests",
        "test_inventory_step",
        "addFileArg",
        "addArgs",
        "test_runner",
    )
    if any(token in loop_text for token in forbidden):
        raise InventoryError(
            "native feature launch must not consume inventory runner, arguments, or barrier"
        )

    factory_cases = by_id[TEST_INVENTORY_FACTORY_COMPILE_ID]["expansion_cases"]
    launch = by_id[NATIVE_FEATURE_LAUNCH_ID]
    launch.update(
        {
            "launch_role": "native-feature-correctness-only",
            "source_compile_relation": "same-logical-test-compile-and-root-module-pointer",
            "expansion_cases": factory_cases,
            "expansion_case_count": len(factory_cases),
            "expansion_cases_digest": _json_fact_digest(factory_cases),
            "dependency_expression": "native_feature_profile_guard",
            "inventory_runner": None,
            "inventory_barrier": None,
            "inventory_evidence": False,
        }
    )
    by_id[NATIVE_FEATURE_STEP_ID].update(
        {
            "step_role": "native-feature-correctness-only",
            "direct_dependencies": [
                {
                    "id": PYTHON_TOOLING_STRUCTURE_BARRIER_ID,
                    "condition": "always",
                },
                {
                    "id": NATIVE_FEATURE_LAUNCH_ID,
                    "condition": "per applicable logical test root after the native profile guard",
                },
            ],
            "inventory_evidence": False,
        }
    )
    for order, (identifier, message, condition) in enumerate(NATIVE_FEATURE_GUARDS, 1):
        by_id[identifier].update(
            {
                "step_role": "native-feature-fail-closed-guard",
                "guard_order": order,
                "guard_condition": condition,
                "direct_dependencies": [],
                "error_message": message,
                "inventory_evidence": False,
            }
        )


def _validate_test_inventory_factory_contract(
    observations: list[dict[str, Any]], errors: list[str]
) -> None:
    by_id = {item.get("id"): item for item in observations}
    factory = by_id.get(TEST_INVENTORY_FACTORY_COMPILE_ID, {})
    expected_cases = [
        {
            "root_id": root_id,
            "logical_compile_observation_id": logical_id,
            "predicate_id": predicate_id,
        }
        for root_id, logical_id, predicate_id in REQUIRED_TEST_INVENTORY_FACTORY_CASES
    ]
    _require(
        factory.get("artifact_role") == "test-inventory-enumerator-factory",
        "test inventory compile observation must have the enumerator factory role",
        errors,
    )
    _require(
        factory.get("expansion_relation") == "one-per-applicable-logical-zig-root",
        "test inventory compile observation has the wrong expansion relation",
        errors,
    )
    _require(
        factory.get("root_module_relation") == "same-pointer",
        "test inventory compile observation must preserve the logical root-module pointer",
        errors,
    )
    _require(
        factory.get("test_runner")
        == {"path": "tools/test_inventory_runner.zig", "mode": "simple"},
        "test inventory compile observation has the wrong runner specification",
        errors,
    )
    _require(
        factory.get("expansion_cases") == expected_cases,
        "test inventory compile observation must bind the exact 21 expansion cases",
        errors,
    )
    _require(
        factory.get("expansion_case_count") == len(expected_cases),
        "test inventory compile observation has the wrong expansion case count",
        errors,
    )
    _require(
        factory.get("expansion_cases_digest") == _json_fact_digest(expected_cases),
        "test inventory compile observation has the wrong canonical expansion digest",
        errors,
    )
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
            dict(mapping) for mapping in REQUIRED_TEST_INVENTORY_ENUMERATION_MAPPINGS
        ],
        "fallback": None,
        "known_branch": "test-inventory-enumerator-factory",
        "unknown_branch_dependency": TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID,
    }
    _require(
        factory.get("enumeration_class_projection") == expected_projection,
        "test inventory compile observation has the wrong optional enumeration class projection",
        errors,
    )
    for case in factory.get("expansion_cases", []):
        _require(
            isinstance(case, dict)
            and set(case)
            == {"root_id", "logical_compile_observation_id", "predicate_id"},
            "test inventory expansion cases must match the schema exactly",
            errors,
        )

    launch = by_id.get(TEST_INVENTORY_FACTORY_LAUNCH_ID, {})
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
    _require(
        launch.get("launch_role") == "test-inventory-enumerator-factory-run",
        "test inventory launch observation has the wrong factory role",
        errors,
    )
    _require(
        launch.get("source_factory") == TEST_INVENTORY_FACTORY_COMPILE_ID,
        "test inventory launch observation has the wrong source factory",
        errors,
    )
    _require(
        launch.get("argument_contract") == expected_argument_contract,
        "test inventory launch observation has the wrong ordered argument contract",
        errors,
    )
    _require(
        launch.get("argv_shape")
        == _test_inventory_argv_shape(expected_argument_contract),
        "test inventory launch argv shape does not match the ordered argument contract",
        errors,
    )

    expected_body_launches = [
        {
            **case,
            "launch_observation_id": launch_id,
        }
        for case, launch_id in zip(
            expected_cases, REQUIRED_TEST_INVENTORY_BODY_LAUNCH_IDS, strict=True
        )
    ]
    expected_launch_barrier = {
        "dependency_step_id": TEST_INVENTORY_RUN_STEP_ID,
        "relation": "direct-step-dependency-before-body-execution",
    }
    for body_launch in expected_body_launches:
        launch_id = body_launch["launch_observation_id"]
        _require(
            by_id.get(launch_id, {}).get("test_inventory_barrier")
            == expected_launch_barrier,
            f"{launch_id}: official Zig body launch is not bound to the test inventory barrier",
            errors,
        )

    expected_steps = {
        TEST_INVENTORY_LINK_STEP_ID: (
            "test-inventory-enumerator-link",
            TEST_INVENTORY_FACTORY_COMPILE_ID,
        ),
        TEST_INVENTORY_RUN_STEP_ID: (
            "test-inventory-enumerator-run",
            TEST_INVENTORY_FACTORY_LAUNCH_ID,
        ),
    }
    for step_id, (role, dependency_id) in expected_steps.items():
        step = by_id.get(step_id, {})
        _require(
            step.get("step_role") == role,
            f"{step_id}: incorrect test inventory step role",
            errors,
        )
        _require(
            step.get("direct_dependencies")
            == [
                {
                    "id": dependency_id,
                    "condition": "per applicable expansion case for an exact baseline target CPU profile",
                },
                {
                    "id": TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID,
                    "condition": "unknown or nonbaseline target CPU profile",
                },
            ],
            f"{step_id}: test inventory factory dependency closure is incomplete",
            errors,
        )
    run_step = by_id.get(TEST_INVENTORY_RUN_STEP_ID, {})
    _require(
        run_step.get("official_body_launch_barrier")
        == {
            "relation": "every-applicable-official-body-launch-depends-on-this-step",
            "launches": expected_body_launches,
            "launch_count": len(expected_body_launches),
            "launches_digest": _json_fact_digest(expected_body_launches),
        },
        "test inventory official body launch barrier contract is incomplete",
        errors,
    )
    unsupported = by_id.get(TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID, {})
    _require(
        unsupported.get("step_role") == "unsupported-test-inventory-target-failure"
        and unsupported.get("direct_dependencies") == []
        and unsupported.get("error_message")
        == TEST_INVENTORY_UNSUPPORTED_TARGET_MESSAGE,
        "test inventory unsupported-target failure step contract is incomplete",
        errors,
    )
    native_launch = by_id.get(NATIVE_FEATURE_LAUNCH_ID, {})
    native_launch_contract = {
        "launch_role": "native-feature-correctness-only",
        "source_compile_relation": "same-logical-test-compile-and-root-module-pointer",
        "expansion_cases": expected_cases,
        "expansion_case_count": len(expected_cases),
        "expansion_cases_digest": _json_fact_digest(expected_cases),
        "dependency_expression": "native_feature_profile_guard",
        "inventory_runner": None,
        "inventory_barrier": None,
        "inventory_evidence": False,
    }
    for field, expected in native_launch_contract.items():
        _require(
            native_launch.get(field) == expected,
            f"{NATIVE_FEATURE_LAUNCH_ID}: recorded {field} changed from the reviewed correctness-only contract",
            errors,
        )
    _require(
        "test_inventory_barrier" not in native_launch,
        f"{NATIVE_FEATURE_LAUNCH_ID}: frozen inventory barrier is forbidden",
        errors,
    )
    native_step = by_id.get(NATIVE_FEATURE_STEP_ID, {})
    _require(
        native_step.get("step_role") == "native-feature-correctness-only"
        and native_step.get("inventory_evidence") is False
        and native_step.get("direct_dependencies")
        == [
            {"id": PYTHON_TOOLING_STRUCTURE_BARRIER_ID, "condition": "always"},
            {
                "id": NATIVE_FEATURE_LAUNCH_ID,
                "condition": "per applicable logical test root after the native profile guard",
            },
        ],
        "native feature step must depend exactly on the structure checker and guarded 21-root correctness launch",
        errors,
    )
    for order, (identifier, message, condition) in enumerate(NATIVE_FEATURE_GUARDS, 1):
        guard = by_id.get(identifier, {})
        _require(
            guard.get("step_role") == "native-feature-fail-closed-guard"
            and guard.get("guard_order") == order
            and guard.get("guard_condition") == condition
            and guard.get("direct_dependencies") == []
            and guard.get("error_message") == message
            and guard.get("inventory_evidence") is False,
            f"{identifier}: native feature fail-closed guard contract changed",
            errors,
        )
    aggregate_dependencies = by_id.get(TEST_INVENTORY_AGGREGATE_STEP_ID, {}).get(
        "direct_dependencies", []
    )
    _require(
        {item.get("id") for item in aggregate_dependencies if isinstance(item, dict)}
        >= {TEST_INVENTORY_RUN_STEP_ID},
        "canonical test aggregate must inventory every native test root before its ordinary launches",
        errors,
    )


def _validate_python_tooling_test_contract(
    observations: list[dict[str, Any]], errors: list[str]
) -> None:
    by_id = {item.get("id"): item for item in observations}
    launch = by_id.get(PYTHON_TOOLING_LAUNCH_ID, {})
    _require(
        "runner_contract" not in launch,
        f"{PYTHON_TOOLING_LAUNCH_ID}: legacy inline runner contract is forbidden",
        errors,
    )
    for field, expected in _python_tooling_launch_template().items():
        _require(
            launch.get(field) == expected,
            f"{PYTHON_TOOLING_LAUNCH_ID}: recorded {field} changed from the reviewed Python tooling launch contract",
            errors,
        )

    step = by_id.get(PYTHON_TOOLING_STEP_ID, {})
    for field, expected in _python_tooling_step_template().items():
        _require(
            step.get(field) == expected,
            f"{PYTHON_TOOLING_STEP_ID}: recorded {field} changed from the reviewed Python tooling step closure",
            errors,
        )

    aggregate_dependencies = by_id.get(TEST_INVENTORY_AGGREGATE_STEP_ID, {}).get(
        "direct_dependencies", []
    )
    expected_dependency = {"id": PYTHON_TOOLING_STEP_ID, "condition": "always"}
    _require(
        isinstance(aggregate_dependencies, list)
        and [
            dependency
            for dependency in aggregate_dependencies
            if isinstance(dependency, dict)
            and dependency.get("id") == PYTHON_TOOLING_STEP_ID
        ]
        == [expected_dependency],
        "canonical test aggregate must record exactly one direct Python tooling step dependency",
        errors,
    )


def _dotted_name(node: ast.expr, aliases: dict[str, str]) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        ordered = list(reversed(parts))
        dotted = ".".join(ordered)
        for length in range(len(ordered), 0, -1):
            prefix = ".".join(ordered[:length])
            if prefix not in aliases:
                continue
            resolved = aliases[prefix]
            if resolved == PYTHON_SHADOWED_ALIAS:
                return None
            remainder = ".".join(ordered[length:])
            return resolved + ("." + remainder if remainder else "")
        root, separator, remainder = dotted.partition(".")
        return root + (separator + remainder if separator else "")
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        key = node.slice.value
        if isinstance(key, str):
            container = _dotted_name(node.value, aliases)
            if container is not None and container.endswith(".__dict__"):
                resolved = container.removesuffix(".__dict__") + "." + key
                suffix = ".".join(reversed(parts))
                return resolved + ("." + suffix if suffix else "")
            if container == "__builtins__" and key in {
                "__import__",
                "eval",
                "exec",
                "globals",
                "locals",
                "vars",
            }:
                resolved = "__import__" if key == "__import__" else f"builtins.{key}"
                suffix = ".".join(reversed(parts))
                return resolved + ("." + suffix if suffix else "")
            if container == PYTHON_GLOBALS_NAMESPACE:
                resolved = aliases.get(key, key)
                if resolved == PYTHON_SHADOWED_ALIAS:
                    return None
                suffix = ".".join(reversed(parts))
                return resolved + ("." + suffix if suffix else "")
            if container == PYTHON_SYS_MODULES_NAMESPACE:
                resolved = key if key in PYTHON_PROCESS_MODULES else None
                if resolved is None:
                    return None
                suffix = ".".join(reversed(parts))
                return resolved + ("." + suffix if suffix else "")
    if isinstance(node, ast.Call):
        called = _dotted_name(node.func, aliases)
        if node.args and not node.keywords:
            key_node = node.args[0]
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
                namespace: str | None = None
                if (
                    called is not None
                    and called.endswith(".__dict__.get")
                    and len(node.args) in {1, 2}
                ):
                    namespace = called.removesuffix(".__dict__.get")
                elif (
                    called is not None
                    and called.endswith(".__dict__.__getitem__")
                    and len(node.args) == 1
                ):
                    namespace = called.removesuffix(".__dict__.__getitem__")
                elif (
                    called is not None
                    and called.endswith(".__getattribute__")
                    and not called.startswith(PYTHON_GLOBALS_NAMESPACE + ".")
                    and len(node.args) == 1
                ):
                    namespace = called.removesuffix(".__getattribute__")
                elif (
                    called == f"{PYTHON_GLOBALS_NAMESPACE}.__getattribute__"
                    and len(node.args) == 1
                    and key
                    in {
                        "get",
                        "__getitem__",
                        "clear",
                        "pop",
                        "popitem",
                        "setdefault",
                        "update",
                        "__delitem__",
                        "__setitem__",
                    }
                ):
                    resolved = f"{PYTHON_GLOBALS_NAMESPACE}.{key}"
                    suffix = ".".join(reversed(parts))
                    return resolved + ("." + suffix if suffix else "")
                elif called in {
                    f"{PYTHON_GLOBALS_NAMESPACE}.get",
                    f"{PYTHON_GLOBALS_NAMESPACE}.__getitem__",
                } and (
                    (called.endswith(".get") and len(node.args) in {1, 2})
                    or (called.endswith(".__getitem__") and len(node.args) == 1)
                ):
                    namespace = PYTHON_GLOBALS_NAMESPACE
                elif called in {
                    "__builtins__.get",
                    f"{PYTHON_SYS_MODULES_NAMESPACE}.get",
                    f"{PYTHON_SYS_MODULES_NAMESPACE}.__getitem__",
                } and (
                    (called.endswith(".get") and len(node.args) in {1, 2})
                    or (called.endswith(".__getitem__") and len(node.args) == 1)
                ):
                    namespace = called.rsplit(".", 1)[0]
                if namespace == PYTHON_SHADOWED_ALIAS:
                    return None
                if namespace is not None:
                    resolved = (
                        aliases.get(key, key)
                        if namespace == PYTHON_GLOBALS_NAMESPACE
                        else key
                        if namespace == PYTHON_SYS_MODULES_NAMESPACE
                        and key in PYTHON_PROCESS_MODULES
                        else namespace + "." + key
                    )
                    if resolved == PYTHON_SHADOWED_ALIAS:
                        return None
                    suffix = ".".join(reversed(parts))
                    return resolved + ("." + suffix if suffix else "")
        if called in PYTHON_NAMESPACE_PRODUCERS and not node.args and not node.keywords:
            suffix = ".".join(reversed(parts))
            return PYTHON_GLOBALS_NAMESPACE + ("." + suffix if suffix else "")
        dynamic_import, imported_module = _dynamic_process_module(node, aliases)
        if dynamic_import and imported_module in PYTHON_PROCESS_MODULES:
            suffix = ".".join(reversed(parts))
            return imported_module + ("." + suffix if suffix else "")
        factory = _dotted_name(node.func, aliases)
        resolved_factory = PYTHON_PROCESS_FACTORIES.get(factory or "")
        if resolved_factory is not None:
            suffix = ".".join(reversed(parts))
            return resolved_factory + ("." + suffix if suffix else "")
    return None


def _set_python_alias(aliases: dict[str, str], name: str, resolved: str) -> None:
    for existing in tuple(aliases):
        if existing == name or existing.startswith(name + "."):
            aliases.pop(existing)
    aliases[name] = resolved


def _shadow_python_alias(aliases: dict[str, str], name: str) -> None:
    _set_python_alias(aliases, name, PYTHON_SHADOWED_ALIAS)


def _normalize_python_process_callable(dotted: str | None) -> str | None:
    if dotted is None:
        return None
    normalized = PYTHON_PROCESS_CALL_ALIASES.get(dotted, dotted)
    while normalized.endswith(".__call__"):
        normalized = normalized.removesuffix(".__call__")
    normalized = PYTHON_PROCESS_CALL_ALIASES.get(normalized, normalized)
    return normalized if normalized in PYTHON_PROCESS_CALLS else dotted


def _is_python_tracked_namespace_mutation(dotted: str | None) -> bool:
    return dotted in PYTHON_NAMESPACE_MUTATIONS


def _is_python_unsupported_namespace_operation(dotted: str | None) -> bool:
    if dotted is None or dotted in (
        PYTHON_NAMESPACE_ACCESSORS
        | PYTHON_NAMESPACE_MUTATIONS
        | PYTHON_IMPORT_HELPERS
        | PYTHON_NAMESPACE_PRODUCERS
        | PYTHON_DYNAMIC_CODE_CALLS
    ):
        return False
    dictionary_namespaces = {
        *PYTHON_PROCESS_MODULES,
        *PYTHON_IMPORT_NAMESPACES,
        "__builtins__",
    }
    return (
        any(
            dotted.startswith(f"{namespace}.__dict__.")
            for namespace in dictionary_namespaces
        )
        or dotted.startswith(PYTHON_GLOBALS_NAMESPACE + ".")
        or dotted.startswith(PYTHON_SYS_MODULES_NAMESPACE + ".")
        or dotted.startswith("__builtins__.")
    )


def _dynamic_process_module(
    node: ast.expr, aliases: dict[str, str]
) -> tuple[bool, str | None]:
    if not isinstance(node, ast.Call):
        return False, None
    called = _dotted_name(node.func, aliases)
    if called not in PYTHON_IMPORT_HELPERS:
        return False, None
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return True, None
    module = node.args[0].value
    if not isinstance(module, str):
        return True, None
    if called in {
        "__import__",
        "builtins.__import__",
        "__builtins__.__import__",
        "importlib.__import__",
    }:
        fromlist: ast.expr | None = node.args[3] if len(node.args) > 3 else None
        level: ast.expr | None = node.args[4] if len(node.args) > 4 else None
        for keyword in node.keywords:
            if keyword.arg == "fromlist":
                fromlist = keyword.value
            elif keyword.arg == "level":
                level = keyword.value
        if level is not None and not (
            isinstance(level, ast.Constant)
            and type(level.value) is int
            and level.value == 0
        ):
            return True, None
        if fromlist is None or (
            isinstance(fromlist, ast.Constant) and not fromlist.value
        ):
            module = module.partition(".")[0]
        elif isinstance(fromlist, (ast.Tuple, ast.List, ast.Set)):
            if not fromlist.elts:
                module = module.partition(".")[0]
        elif not (
            isinstance(fromlist, ast.Constant)
            and isinstance(fromlist.value, str)
            and fromlist.value
        ):
            return True, None
    return True, module


def _python_files(root: Path, context: DiscoveryContext | None = None) -> list[Path]:
    active = _context_for(root, context)
    root = active.root
    python_files: list[Path] = []
    for rel in active.public_files.paths:
        if Path(rel).suffix.lower() != ".py":
            continue
        node = active.public_files.node(rel)
        if node.kind != "regular":
            raise InventoryError(
                f"Python source must be a non-symlink regular file: {rel}"
            )
        if node.bytes is None:
            raise InventoryError(f"Python source bytes were not frozen: {rel}")
        python_files.append(root / rel)
    return sorted(python_files)


def _is_test_inventory_module_binding_verifier_call(
    node: ast.Call,
    rel: str,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Return whether ``node`` executes in the one reviewed verifier scope."""
    if rel != "tools/check_test_inventory.py":
        return False
    scope = _python_scope(node, parents)
    return (
        isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
        and scope.name == "_verify_python_source_module_binding"
        and isinstance(parents.get(scope), ast.Module)
    )


def _is_reviewed_test_inventory_module_binding_lookup(
    node: ast.Call,
    rel: str,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Recognize only the reviewed ``sys.modules`` identity guard AST shape."""
    if not _is_test_inventory_module_binding_verifier_call(node, rel, parents):
        return False
    function = node.func
    if not (
        isinstance(function, ast.Attribute)
        and function.attr == "get"
        and isinstance(function.ctx, ast.Load)
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "modules"
        and isinstance(function.value.ctx, ast.Load)
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "sys"
        and isinstance(function.value.value.ctx, ast.Load)
        and len(node.args) == 1
        and not node.keywords
    ):
        return False
    argument = node.args[0]
    if not (
        isinstance(argument, ast.Attribute)
        and argument.attr == "name"
        and isinstance(argument.ctx, ast.Load)
        and isinstance(argument.value, ast.Name)
        and argument.value.id == "binding"
        and isinstance(argument.value.ctx, ast.Load)
    ):
        return False
    comparison = parents.get(node)
    if not (
        isinstance(comparison, ast.Compare)
        and comparison.left is node
        and len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.IsNot)
        and len(comparison.comparators) == 1
        and isinstance(comparison.comparators[0], ast.Name)
        and comparison.comparators[0].id == "module"
        and isinstance(comparison.comparators[0].ctx, ast.Load)
    ):
        return False
    current: ast.AST = comparison
    inside_boolean_tree = False
    while True:
        parent = parents.get(current)
        if isinstance(parent, ast.BoolOp) and current in parent.values:
            inside_boolean_tree = True
            current = parent
            continue
        return (
            inside_boolean_tree
            and isinstance(parent, ast.If)
            and parent.test is current
        )


def _reviewed_test_inventory_loader_contract_digest(tree: ast.Module) -> str | None:
    constant_names = (
        "_PYTHON_TOOLING_EXECUTION_SOURCE_SHA256",
        "_PYTHON_TOOLING_EXECUTION_MODULES",
        "_PYTHON_TOOLING_EXECUTION_MANIFEST_SHA256",
    )
    definitions: list[tuple[str, ast.AST]] = []
    for name in REVIEWED_TEST_INVENTORY_LOADER_FUNCTION_NAMES:
        direct_definitions = [
            candidate
            for candidate in tree.body
            if isinstance(candidate, ast.FunctionDef) and candidate.name == name
        ]
        all_definitions = [
            candidate
            for candidate in ast.walk(tree)
            if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
            and candidate.name == name
        ]
        if len(direct_definitions) != 1 or all_definitions != direct_definitions:
            return None
        definitions.append((name, direct_definitions[0]))
    for name in REVIEWED_TEST_INVENTORY_LOADER_CLASS_NAMES:
        direct_definitions = [
            candidate
            for candidate in tree.body
            if isinstance(candidate, ast.ClassDef) and candidate.name == name
        ]
        all_definitions = [
            candidate
            for candidate in ast.walk(tree)
            if isinstance(candidate, ast.ClassDef) and candidate.name == name
        ]
        if len(direct_definitions) != 1 or all_definitions != direct_definitions:
            return None
        definitions.append((name, direct_definitions[0]))
    for name in constant_names:
        assignments = [
            candidate
            for candidate in tree.body
            if isinstance(candidate, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in (
                    candidate.targets
                    if isinstance(candidate, ast.Assign)
                    else [candidate.target]
                )
            )
        ]
        if len(assignments) != 1:
            return None
        definitions.append((name, assignments[0]))
    digest = hashlib.sha256()

    def update_token(token: str) -> None:
        encoded = token.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b"\0")
        digest.update(encoded)

    def update_value(value: Any) -> None:
        if isinstance(value, ast.AST):
            update_node(value)
        elif isinstance(value, list):
            update_token("list")
            update_token(str(len(value)))
            for item in value:
                update_value(item)
        elif isinstance(value, tuple):
            update_token("tuple")
            update_token(str(len(value)))
            for item in value:
                update_value(item)
        elif (
            value is None
            or value is Ellipsis
            or isinstance(value, (str, bytes, bool, int, float, complex))
        ):
            update_token("scalar")
            update_token(repr(value))
        else:
            raise TypeError(
                f"unsupported Python AST field value: {type(value).__name__}"
            )

    def update_node(node: ast.AST) -> None:
        update_token(type(node).__name__)
        fields = [
            (name, value)
            for name, value in ast.iter_fields(node)
            if not (name == "type_params" and value == [])
        ]
        update_token(str(len(fields)))
        for name, value in fields:
            update_token(name)
            update_value(value)

    update_token("reviewed-test-inventory-loader-contract-v2-streaming-ast")
    for name, definition in definitions:
        update_token(name)
        update_node(definition)
    return digest.hexdigest()


def _is_reviewed_test_inventory_source_exec(
    node: ast.Call,
    rel: str,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    loader_contract_digest: str | None = None,
) -> bool:
    """Recognize the single frozen-source compile/guard/exec transaction."""

    if loader_contract_digest != REVIEWED_TEST_INVENTORY_LOADER_CONTRACT_SHA256:
        return False

    def direct_name(
        candidate: ast.AST, name: str, context: type[ast.expr_context]
    ) -> bool:
        return (
            isinstance(candidate, ast.Name)
            and candidate.id == name
            and isinstance(candidate.ctx, context)
        )

    def direct_attribute(
        candidate: ast.AST,
        root_name: str,
        attributes: tuple[str, ...],
    ) -> bool:
        current = candidate
        for attribute in reversed(attributes):
            if not (
                isinstance(current, ast.Attribute)
                and current.attr == attribute
                and isinstance(current.ctx, ast.Load)
            ):
                return False
            current = current.value
        return direct_name(current, root_name, ast.Load)

    def direct_call(
        candidate: ast.AST,
        function_name: str,
        arguments: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> bool:
        return (
            isinstance(candidate, ast.Call)
            and direct_name(candidate.func, function_name, ast.Load)
            and not candidate.keywords
            and len(candidate.args) == len(arguments)
            and all(
                direct_attribute(argument, root_name, attributes)
                if attributes
                else direct_name(argument, root_name, ast.Load)
                for argument, (root_name, attributes) in zip(candidate.args, arguments)
            )
        )

    def containing_block(statement: ast.stmt) -> list[ast.stmt] | None:
        parent = parents.get(statement)
        if parent is None:
            return None
        for _, field in ast.iter_fields(parent):
            if isinstance(field, list) and any(item is statement for item in field):
                return field
        return None

    if rel != "tools/check_test_inventory.py":
        return False
    direct_loader_definitions = [
        candidate
        for candidate in tree.body
        if isinstance(candidate, ast.FunctionDef)
        and candidate.name == "_registered_python_tooling_modules"
    ]
    all_loader_definitions = [
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
        and candidate.name == "_registered_python_tooling_modules"
    ]
    if (
        len(direct_loader_definitions) != 1
        or all_loader_definitions != direct_loader_definitions
    ):
        return False
    loader_definition = direct_loader_definitions[0]
    if not (
        len(loader_definition.decorator_list) == 1
        and direct_attribute(
            loader_definition.decorator_list[0], "contextlib", ("contextmanager",)
        )
        and _python_scope(node, parents) is loader_definition
    ):
        return False
    direct_compile_calls = [
        candidate
        for candidate in ast.walk(loader_definition)
        if isinstance(candidate, ast.Call)
        and direct_name(candidate.func, "compile", ast.Load)
    ]
    direct_exec_calls = [
        candidate
        for candidate in ast.walk(loader_definition)
        if isinstance(candidate, ast.Call)
        and direct_name(candidate.func, "exec", ast.Load)
    ]
    if len(direct_compile_calls) != 1 or direct_exec_calls != [node]:
        return False
    exec_loads = [
        candidate
        for candidate in ast.walk(loader_definition)
        if direct_name(candidate, "exec", ast.Load)
    ]
    compile_loads = [
        candidate
        for candidate in ast.walk(loader_definition)
        if direct_name(candidate, "compile", ast.Load)
    ]
    if exec_loads != [node.func] or compile_loads != [direct_compile_calls[0].func]:
        return False
    exec_statement = parents.get(node)
    loop = parents.get(exec_statement) if isinstance(exec_statement, ast.Expr) else None
    if not (
        isinstance(exec_statement, ast.Expr)
        and isinstance(loop, ast.For)
        and direct_name(loop.target, "binding", ast.Store)
        and direct_name(loop.iter, "registry", ast.Load)
        and not loop.orelse
        and exec_statement in loop.body
        and direct_name(node.func, "exec", ast.Load)
        and len(node.args) == 2
        and not node.keywords
        and direct_name(node.args[0], "code", ast.Load)
        and direct_attribute(node.args[1], "binding", ("namespace",))
    ):
        return False
    exec_index = loop.body.index(exec_statement)
    if exec_index < 2 or exec_index + 3 >= len(loop.body):
        return False
    compile_statement = loop.body[exec_index - 2]
    guard_statement = loop.body[exec_index - 1]
    verify_statement = loop.body[exec_index + 1]
    snapshot_statement = loop.body[exec_index + 2]
    digest_statement = loop.body[exec_index + 3]
    execution_with = parents.get(loop)
    if not (isinstance(execution_with, ast.With) and loop in execution_with.body):
        return False
    loop_index = execution_with.body.index(loop)
    if loop_index < 2:
        return False
    registry_statement = execution_with.body[loop_index - 2]
    pre_verify_statement = execution_with.body[loop_index - 1]
    if not (
        isinstance(registry_statement, ast.Assign)
        and len(registry_statement.targets) == 1
        and direct_name(registry_statement.targets[0], "registry", ast.Store)
        and direct_call(
            registry_statement.value,
            "tuple",
            (("bindings", ()),),
        )
        and isinstance(pre_verify_statement, ast.Expr)
        and direct_call(
            pre_verify_statement.value,
            "_verify_python_source_module_registry",
            (("registry", ()), ("reviewed_modules", ())),
        )
    ):
        return False
    if not (
        isinstance(compile_statement, ast.Assign)
        and len(compile_statement.targets) == 1
        and direct_name(compile_statement.targets[0], "code", ast.Store)
        and compile_statement.value is direct_compile_calls[0]
    ):
        return False
    compile_call = direct_compile_calls[0]
    if not (
        len(compile_call.args) == 3
        and len(compile_call.keywords) == 1
        and direct_attribute(
            compile_call.args[0], "binding", ("reviewed", "source_bytes")
        )
        and direct_attribute(compile_call.args[1], "binding", ("file",))
        and isinstance(compile_call.args[2], ast.Constant)
        and compile_call.args[2].value == "exec"
        and compile_call.keywords[0].arg == "dont_inherit"
        and isinstance(compile_call.keywords[0].value, ast.Constant)
        and compile_call.keywords[0].value.value is True
    ):
        return False
    if not (
        isinstance(guard_statement, ast.If)
        and isinstance(guard_statement.test, ast.BoolOp)
        and isinstance(guard_statement.test.op, ast.Or)
        and len(guard_statement.test.values) == 2
        and len(guard_statement.body) == 1
        and not guard_statement.orelse
    ):
        return False
    type_guard, filename_guard = guard_statement.test.values
    if not (
        isinstance(type_guard, ast.Compare)
        and isinstance(type_guard.left, ast.Call)
        and direct_name(type_guard.left.func, "type", ast.Load)
        and len(type_guard.left.args) == 1
        and not type_guard.left.keywords
        and direct_name(type_guard.left.args[0], "code", ast.Load)
        and len(type_guard.ops) == 1
        and isinstance(type_guard.ops[0], ast.IsNot)
        and len(type_guard.comparators) == 1
        and direct_attribute(type_guard.comparators[0], "types", ("CodeType",))
        and isinstance(filename_guard, ast.Compare)
        and direct_attribute(filename_guard.left, "code", ("co_filename",))
        and len(filename_guard.ops) == 1
        and isinstance(filename_guard.ops[0], ast.NotEq)
        and len(filename_guard.comparators) == 1
        and direct_attribute(filename_guard.comparators[0], "binding", ("file",))
    ):
        return False
    guard_raise = guard_statement.body[0]
    if not (
        isinstance(guard_raise, ast.Raise)
        and guard_raise.cause is None
        and isinstance(guard_raise.exc, ast.Call)
        and direct_name(guard_raise.exc.func, "InventoryError", ast.Load)
        and len(guard_raise.exc.args) == 1
        and not guard_raise.exc.keywords
        and isinstance(guard_raise.exc.args[0], ast.Constant)
        and guard_raise.exc.args[0].value
        == "Python tooling compiled code binding changed"
    ):
        return False
    if not (
        isinstance(verify_statement, ast.Expr)
        and direct_call(
            verify_statement.value,
            "_verify_python_source_module_registry",
            (("registry", ()), ("reviewed_modules", ())),
        )
    ):
        return False
    if not (
        isinstance(snapshot_statement, ast.Assign)
        and len(snapshot_statement.targets) == 1
        and direct_name(snapshot_statement.targets[0], "observed", ast.Store)
        and isinstance(snapshot_statement.value, ast.Call)
        and direct_name(
            snapshot_statement.value.func,
            "_read_regular_stable_snapshot",
            ast.Load,
        )
        and len(snapshot_statement.value.args) == 3
        and not snapshot_statement.value.keywords
        and direct_attribute(
            snapshot_statement.value.args[0],
            "binding",
            ("reviewed", "source_path"),
        )
        and direct_name(
            snapshot_statement.value.args[1], "MAX_INVENTORY_BYTES", ast.Load
        )
        and isinstance(snapshot_statement.value.args[2], ast.JoinedStr)
        and len(snapshot_statement.value.args[2].values) == 2
        and isinstance(snapshot_statement.value.args[2].values[0], ast.Constant)
        and snapshot_statement.value.args[2].values[0].value
        == "Python tooling runtime source "
        and isinstance(snapshot_statement.value.args[2].values[1], ast.FormattedValue)
        and snapshot_statement.value.args[2].values[1].conversion == -1
        and snapshot_statement.value.args[2].values[1].format_spec is None
        and direct_attribute(
            snapshot_statement.value.args[2].values[1].value,
            "binding",
            ("reviewed", "inventory_path"),
        )
    ):
        return False
    if not (
        isinstance(digest_statement, ast.If)
        and isinstance(digest_statement.test, ast.Compare)
        and direct_attribute(digest_statement.test.left, "observed", ("sha256",))
        and len(digest_statement.test.ops) == 1
        and isinstance(digest_statement.test.ops[0], ast.NotEq)
        and len(digest_statement.test.comparators) == 1
        and direct_attribute(
            digest_statement.test.comparators[0],
            "binding",
            ("reviewed", "source_sha256"),
        )
        and len(digest_statement.body) == 1
        and not digest_statement.orelse
        and isinstance(digest_statement.body[0], ast.Raise)
        and digest_statement.body[0].cause is None
        and isinstance(digest_statement.body[0].exc, ast.Call)
        and direct_name(digest_statement.body[0].exc.func, "InventoryError", ast.Load)
        and len(digest_statement.body[0].exc.args) == 1
        and not digest_statement.body[0].exc.keywords
        and isinstance(digest_statement.body[0].exc.args[0], ast.Constant)
        and digest_statement.body[0].exc.args[0].value
        == "Python tooling source changed between review and module execution"
    ):
        return False
    code_names = [
        candidate
        for candidate in ast.walk(loader_definition)
        if isinstance(candidate, ast.Name) and candidate.id == "code"
    ]
    permitted_code_names = {
        compile_statement.targets[0],
        type_guard.left.args[0],
        filename_guard.left.value,
        node.args[0],
    }
    if set(code_names) != permitted_code_names or len(code_names) != len(
        permitted_code_names
    ):
        return False

    direct_caller_definitions = [
        candidate
        for candidate in tree.body
        if isinstance(candidate, ast.FunctionDef)
        and candidate.name == "_run_python_tooling_root"
    ]
    all_caller_definitions = [
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
        and candidate.name == "_run_python_tooling_root"
    ]
    if (
        len(direct_caller_definitions) != 1
        or all_caller_definitions != direct_caller_definitions
    ):
        return False
    caller_definition = direct_caller_definitions[0]
    loader_name_loads = [
        candidate
        for candidate in ast.walk(tree)
        if direct_name(candidate, "_registered_python_tooling_modules", ast.Load)
    ]
    if len(loader_name_loads) != 1:
        return False
    loader_call = parents.get(loader_name_loads[0])
    if not (
        isinstance(loader_call, ast.Call)
        and loader_call.func is loader_name_loads[0]
        and len(loader_call.args) == 1
        and not loader_call.keywords
        and direct_name(loader_call.args[0], "reviewed_modules", ast.Load)
        and _python_scope(loader_call, parents) is caller_definition
    ):
        return False
    with_item = parents.get(loader_call)
    with_statement = (
        parents.get(with_item) if isinstance(with_item, ast.withitem) else None
    )
    if not (
        isinstance(with_item, ast.withitem)
        and with_item.context_expr is loader_call
        and direct_name(with_item.optional_vars, "module_registry", ast.Store)
        and isinstance(with_statement, ast.With)
        and len(with_statement.items) == 1
    ):
        return False
    caller_block = containing_block(with_statement)
    if caller_block is None:
        return False
    with_index = caller_block.index(with_statement)
    if with_index == 0:
        return False
    provenance = caller_block[with_index - 1]
    if not (
        isinstance(provenance, ast.Assign)
        and len(provenance.targets) == 1
        and isinstance(provenance.targets[0], ast.Tuple)
        and len(provenance.targets[0].elts) == 4
        and all(
            direct_name(target, name, ast.Store)
            for target, name in zip(
                provenance.targets[0].elts,
                ("_", "_", "dynamic_sites", "reviewed_modules"),
            )
        )
        and isinstance(provenance.value, ast.Call)
        and direct_name(
            provenance.value.func, "_python_tooling_source_skip_review", ast.Load
        )
        and len(provenance.value.args) == 4
        and not provenance.value.keywords
        and direct_name(provenance.value.args[0], "root", ast.Load)
        and isinstance(provenance.value.args[1], ast.Subscript)
        and direct_name(provenance.value.args[1].value, "tooling_root", ast.Load)
        and isinstance(provenance.value.args[1].slice, ast.Constant)
        and provenance.value.args[1].slice.value == "module_paths"
        and direct_name(provenance.value.args[2], "discovery_start", ast.Load)
        and direct_name(provenance.value.args[3], "discovery_pattern", ast.Load)
    ):
        return False
    reviewed_names = [
        candidate
        for candidate in ast.walk(caller_definition)
        if isinstance(candidate, ast.Name) and candidate.id == "reviewed_modules"
    ]
    if (
        set(reviewed_names)
        != {
            provenance.targets[0].elts[3],
            loader_call.args[0],
        }
        or len(reviewed_names) != 2
    ):
        return False
    return True


def _is_reviewed_test_inventory_frozen_exec_regression(
    node: ast.Call,
    rel: str,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    if rel != "test/build/test_test_inventory.py":
        return False
    scope = _python_scope(node, parents)
    outer = parents.get(scope) if scope is not None else None
    return (
        isinstance(scope, ast.FunctionDef)
        and scope.name == "record_frozen_exec"
        and isinstance(outer, ast.FunctionDef)
        and outer.name == "test_predicate_x86_and_mode_row_mutations_fail"
        and isinstance(node.func, ast.Name)
        and node.func.id == "trusted_exec"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "code"
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "namespace"
        and not node.keywords
        and any(
            isinstance(candidate, ast.Assign)
            and len(candidate.targets) == 1
            and isinstance(candidate.targets[0], ast.Name)
            and candidate.targets[0].id == "trusted_exec"
            and isinstance(candidate.value, ast.Name)
            and candidate.value.id == "exec"
            for candidate in outer.body
        )
    )


def _discover_python_launches(
    root: Path,
    context: DiscoveryContext | None = None,
    *,
    _python_files_override: list[Path] | None = None,
    _trees_override: dict[Path, ast.Module] | None = None,
    _embedded_origin: dict[str, Any] | None = None,
    _embedded_budget: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    active = _context_for(root, context)
    root = active.root
    result: list[dict[str, Any]] = []
    python_files = (
        _python_files(root, active)
        if _python_files_override is None
        else _python_files_override
    )
    reviewed_loader_contract_tree: ast.Module | None = None
    if _trees_override is None:
        trees: dict[Path, ast.Module] = {}
        loader_path = root / "tools/check_test_inventory.py"
        if loader_path in python_files:
            loader_rel = loader_path.relative_to(root).as_posix()
            loader_tree = ast.parse(
                _frozen_regular_text(active, loader_rel, "Python source"),
                filename=loader_rel,
            )
            if (
                _reviewed_test_inventory_loader_contract_digest(loader_tree)
                == REVIEWED_TEST_INVENTORY_LOADER_CONTRACT_SHA256
            ):
                reviewed_loader_contract_tree = loader_tree
            trees[loader_path] = loader_tree
        for path in python_files:
            if path in trees:
                continue
            rel = path.relative_to(root).as_posix()
            trees[path] = ast.parse(
                _frozen_regular_text(active, rel, "Python source"),
                filename=rel,
            )
    else:
        trees = _trees_override
        loader_tree = trees.get(root / "tools/check_test_inventory.py")
        if (
            loader_tree is not None
            and _reviewed_test_inventory_loader_contract_digest(loader_tree)
            == REVIEWED_TEST_INVENTORY_LOADER_CONTRACT_SHA256
        ):
            reviewed_loader_contract_tree = loader_tree
    embedded_budget = (
        {"bytes": 0, "nodes": 0, "units": 0}
        if _embedded_budget is None
        else _embedded_budget
    )
    reviewed_adapter_digest: str | None = None
    reviewed_adapter_definition: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    reviewed_adapter_loader_names: tuple[str, str, str] | None = None
    reviewed_adapter_loader_nodes: dict[str, ast.AST] | None = None
    reviewed_adapter_exec_position: tuple[int, int] | None = None
    reviewed_adapter_path = root / "tools/observe_abi_baseline.py"
    reviewed_adapter_consumer_path = (
        root / "test/abi/baseline/test_observe_abi_baseline.py"
    )
    reviewed_adapter_tree = trees.get(reviewed_adapter_path)
    reviewed_consumer_tree = trees.get(reviewed_adapter_consumer_path)

    class NamespaceBindingVisitor(ast.NodeVisitor):
        """Collect writes in one runtime namespace while respecting nested scopes."""

        def __init__(self, name: str) -> None:
            self.name = name
            self.events: list[ast.AST] = []
            self.ignored_comprehension_targets: set[ast.AST] = set()

        def visit_Name(self, node: ast.Name) -> None:
            if (
                node.id == self.name
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node not in self.ignored_comprehension_targets
            ):
                self.events.append(node)

        def visit_Import(self, node: ast.Import) -> None:
            if any(
                (imported.asname or imported.name.partition(".")[0]) == self.name
                for imported in node.names
            ):
                self.events.append(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if any(
                imported.name != "*" and (imported.asname or imported.name) == self.name
                for imported in node.names
            ):
                self.events.append(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function_definition(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function_definition(node)

        def _visit_function_definition(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            if node.name == self.name:
                self.events.append(node)
            for expression in (
                *node.decorator_list,
                *node.args.defaults,
                *(item for item in node.args.kw_defaults if item is not None),
                *(item.annotation for item in node.args.posonlyargs if item.annotation),
                *(item.annotation for item in node.args.args if item.annotation),
                *(item.annotation for item in node.args.kwonlyargs if item.annotation),
                *(item for item in (node.returns,) if item is not None),
            ):
                self.visit(expression)
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                self.visit(node.args.vararg.annotation)
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                self.visit(node.args.kwarg.annotation)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            for expression in (
                *node.args.defaults,
                *(item for item in node.args.kw_defaults if item is not None),
            ):
                self.visit(expression)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if node.name == self.name:
                self.events.append(node)
            for expression in (*node.decorator_list, *node.bases):
                self.visit(expression)
            for keyword in node.keywords:
                self.visit(keyword.value)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.name == self.name:
                self.events.append(node)
            self.generic_visit(node)

        def visit_MatchAs(self, node: ast.MatchAs) -> None:
            if node.name == self.name:
                self.events.append(node)
            self.generic_visit(node)

        def visit_MatchStar(self, node: ast.MatchStar) -> None:
            if node.name == self.name:
                self.events.append(node)

        def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
            if node.rest == self.name:
                self.events.append(node)
            self.generic_visit(node)

        def _visit_comprehension(
            self,
            node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        ) -> None:
            targets = {
                candidate
                for generator in node.generators
                for candidate in ast.walk(generator.target)
                if isinstance(candidate, ast.Name)
                and isinstance(candidate.ctx, ast.Store)
            }
            previous = self.ignored_comprehension_targets
            self.ignored_comprehension_targets = previous | targets
            self.generic_visit(node)
            self.ignored_comprehension_targets = previous

        def visit_ListComp(self, node: ast.ListComp) -> None:
            self._visit_comprehension(node)

        def visit_SetComp(self, node: ast.SetComp) -> None:
            self._visit_comprehension(node)

        def visit_DictComp(self, node: ast.DictComp) -> None:
            self._visit_comprehension(node)

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            self._visit_comprehension(node)

    class DirectGlobalCollector(ast.NodeVisitor):
        """Collect global declarations belonging to one function or class block."""

        def __init__(self) -> None:
            self.names: set[str] = set()

        def visit_Global(self, node: ast.Global) -> None:
            self.names.update(node.names)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            del node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            del node

        def visit_Lambda(self, node: ast.Lambda) -> None:
            del node

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            del node

    def module_namespace_binding_events(tree: ast.Module, name: str) -> list[ast.AST]:
        module_visitor = NamespaceBindingVisitor(name)
        for statement in tree.body:
            module_visitor.visit(statement)
        events = list(module_visitor.events)

        nested_scopes = (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        for scope in nested_scopes:
            globals_in_scope = DirectGlobalCollector()
            for statement in scope.body:
                globals_in_scope.visit(statement)
            if name not in globals_in_scope.names:
                continue
            global_writer = NamespaceBindingVisitor(name)
            for statement in scope.body:
                global_writer.visit(statement)
            events.extend(global_writer.events)

        unique: list[ast.AST] = []
        seen: set[int] = set()
        for event in events:
            if id(event) in seen:
                continue
            seen.add(id(event))
            unique.append(event)
        return unique

    def import_bindings(
        tree: ast.Module,
    ) -> dict[str, tuple[str, ast.stmt, int]]:
        bindings: dict[str, tuple[str, ast.stmt, int]] = {}
        duplicates: set[str] = set()
        for index, statement in enumerate(tree.body):
            if isinstance(statement, ast.Import):
                for imported in statement.names:
                    bound = imported.asname or imported.name.partition(".")[0]
                    resolved = (
                        imported.name
                        if imported.asname
                        else imported.name.partition(".")[0]
                    )
                    if bound in bindings:
                        duplicates.add(bound)
                    bindings[bound] = (resolved, statement, index)
            elif isinstance(statement, ast.ImportFrom) and statement.module:
                for imported in statement.names:
                    if imported.name == "*":
                        continue
                    bound = imported.asname or imported.name
                    resolved = f"{statement.module}.{imported.name}"
                    if bound in bindings:
                        duplicates.add(bound)
                    bindings[bound] = (resolved, statement, index)
        for duplicate in duplicates:
            bindings.pop(duplicate, None)
        return bindings

    def base_identifier(node: ast.AST) -> str | None:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def simple_assignment(statement: ast.stmt) -> tuple[ast.Name, ast.expr] | None:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            return statement.targets[0], statement.value
        return None

    def prove_reviewed_loader(
        tree: ast.Module,
    ) -> tuple[
        tuple[str, str, str], dict[str, ast.AST], tuple[int, int], list[ast.AST]
    ]:
        imports = import_bindings(tree)
        import_aliases = {name: resolved for name, (resolved, _, _) in imports.items()}

        def call_name(node: ast.expr) -> str | None:
            return _dotted_name(node, import_aliases)

        def is_root_producer(value: ast.expr) -> bool:
            return (
                isinstance(value, ast.Subscript)
                and isinstance(value.slice, ast.Constant)
                and type(value.slice.value) is int
                and value.slice.value == 3
                and isinstance(value.value, ast.Attribute)
                and value.value.attr == "parents"
                and isinstance(value.value.value, ast.Call)
                and not value.value.value.args
                and not value.value.value.keywords
                and isinstance(value.value.value.func, ast.Attribute)
                and value.value.value.func.attr == "resolve"
                and isinstance(value.value.value.func.value, ast.Call)
                and call_name(value.value.value.func.value.func) == "pathlib.Path"
                and len(value.value.value.func.value.args) == 1
                and isinstance(value.value.value.func.value.args[0], ast.Name)
                and value.value.value.func.value.args[0].id == "__file__"
                and not value.value.value.func.value.keywords
            )

        assignments = [
            (index, statement, assignment[0], assignment[1])
            for index, statement in enumerate(tree.body)
            if (assignment := simple_assignment(statement)) is not None
        ]
        roots = [item for item in assignments if is_root_producer(item[3])]
        if len(roots) != 1:
            raise InventoryError(
                "test/abi/baseline/test_observe_abi_baseline.py observer loader root is unsupported"
            )
        root_index, root_statement, root_target, _ = roots[0]
        paths = [
            item
            for item in assignments
            if isinstance(item[3], ast.BinOp)
            and isinstance(item[3].op, ast.Div)
            and isinstance(item[3].left, ast.Name)
            and item[3].left.id == root_target.id
            and isinstance(item[3].right, ast.Constant)
            and item[3].right.value == "tools/observe_abi_baseline.py"
        ]
        if len(paths) != 1:
            raise InventoryError(
                "test/abi/baseline/test_observe_abi_baseline.py observer loader path is unsupported"
            )
        path_index, path_statement, path_target, _ = paths[0]
        specs = [
            item
            for item in assignments
            if isinstance(item[3], ast.Call)
            and call_name(item[3].func) == "importlib.util.spec_from_file_location"
            and len(item[3].args) == 2
            and isinstance(item[3].args[0], ast.Constant)
            and item[3].args[0].value == "observe_abi_baseline"
            and isinstance(item[3].args[1], ast.Name)
            and item[3].args[1].id == path_target.id
            and not item[3].keywords
        ]
        if len(specs) != 1:
            raise InventoryError(
                "test/abi/baseline/test_observe_abi_baseline.py observer loader chain is unsupported"
            )
        spec_index, spec_statement, spec_target, _ = specs[0]
        modules = [
            item
            for item in assignments
            if isinstance(item[3], ast.Call)
            and call_name(item[3].func) == "importlib.util.module_from_spec"
            and len(item[3].args) == 1
            and isinstance(item[3].args[0], ast.Name)
            and item[3].args[0].id == spec_target.id
            and not item[3].keywords
        ]
        if len(modules) != 1:
            raise InventoryError(
                "test/abi/baseline/test_observe_abi_baseline.py observer loader chain is unsupported"
            )
        module_index, module_statement, module_target, _ = modules[0]

        registrations: list[tuple[int, ast.Assign]] = []
        executions: list[tuple[int, ast.Expr]] = []
        for index, statement in enumerate(tree.body):
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Subscript)
                and call_name(statement.targets[0].value)
                == PYTHON_SYS_MODULES_NAMESPACE
                and isinstance(statement.targets[0].slice, ast.Attribute)
                and isinstance(statement.targets[0].slice.value, ast.Name)
                and statement.targets[0].slice.value.id == spec_target.id
                and statement.targets[0].slice.attr == "name"
                and isinstance(statement.value, ast.Name)
                and statement.value.id == module_target.id
            ):
                registrations.append((index, statement))
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and call_name(statement.value.func)
                == f"{spec_target.id}.loader.exec_module"
                and len(statement.value.args) == 1
                and isinstance(statement.value.args[0], ast.Name)
                and statement.value.args[0].id == module_target.id
                and not statement.value.keywords
            ):
                executions.append((index, statement))
        if len(registrations) != 1 or len(executions) != 1:
            raise InventoryError(
                "test/abi/baseline/test_observe_abi_baseline.py observer loader chain is unsupported"
            )
        register_index, register_statement = registrations[0]
        exec_index, exec_statement = executions[0]
        if not (
            root_index
            < path_index
            < spec_index
            < module_index
            < register_index
            < exec_index
        ):
            raise InventoryError(
                "test/abi/baseline/test_observe_abi_baseline.py observer loader chain order is unsupported"
            )

        chain_names = {
            root_target.id,
            path_target.id,
            spec_target.id,
            module_target.id,
        }
        if len(chain_names) != 4:
            raise InventoryError(
                "test/abi/baseline/test_observe_abi_baseline.py observer loader variables must be distinct"
            )

        guards = [
            statement
            for statement in tree.body[spec_index + 1 : module_index]
            if isinstance(statement, ast.If)
        ]
        if (
            path_index != root_index + 1
            or spec_index != path_index + 1
            or register_index != module_index + 1
            or exec_index != register_index + 1
            or tree.body[spec_index + 1 : module_index] != guards
            or len(guards) > 1
        ):
            raise InventoryError(
                "test/abi/baseline/test_observe_abi_baseline.py observer loader chain contains unsupported intervening statements"
            )
        if guards:
            guard = guards[0]
            comparisons = (
                guard.test.values
                if isinstance(guard.test, ast.BoolOp)
                and isinstance(guard.test.op, ast.Or)
                else []
            )

            def is_none_check(node: ast.AST, attribute: str | None) -> bool:
                if not (
                    isinstance(node, ast.Compare)
                    and len(node.ops) == 1
                    and isinstance(node.ops[0], ast.Is)
                    and len(node.comparators) == 1
                    and isinstance(node.comparators[0], ast.Constant)
                    and node.comparators[0].value is None
                ):
                    return False
                left = node.left
                if attribute is None:
                    return isinstance(left, ast.Name) and left.id == spec_target.id
                return (
                    isinstance(left, ast.Attribute)
                    and left.attr == attribute
                    and isinstance(left.value, ast.Name)
                    and left.value.id == spec_target.id
                )

            if (
                guard.orelse
                or len(guard.body) != 1
                or not isinstance(guard.body[0], ast.Raise)
                or len(comparisons) != 2
                or not is_none_check(comparisons[0], None)
                or not is_none_check(comparisons[1], "loader")
                or any(
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id in chain_names
                    for node in ast.walk(guard.body[0])
                )
            ):
                raise InventoryError(
                    "test/abi/baseline/test_observe_abi_baseline.py observer loader guard is unsupported"
                )

        allowed_bindings: dict[str, ast.AST] = {
            root_target.id: root_target,
            path_target.id: path_target,
            spec_target.id: spec_target,
            module_target.id: module_target,
        }
        for name, allowed in allowed_bindings.items():
            if any(
                event is not allowed
                for event in module_namespace_binding_events(tree, name)
            ):
                raise InventoryError(
                    f"test/abi/baseline/test_observe_abi_baseline.py observer loader variable {name} is rebound"
                )

        allowed_write_nodes = {register_statement.targets[0]}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Attribute, ast.Subscript)) or not isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                continue
            if base_identifier(node) in chain_names and node not in allowed_write_nodes:
                raise InventoryError(
                    "test/abi/baseline/test_observe_abi_baseline.py observer loader attributes are mutated"
                )

        allowed_pre_exec_statements: set[ast.stmt] = {
            root_statement,
            path_statement,
            spec_statement,
            module_statement,
            register_statement,
            exec_statement,
            *guards,
        }
        for statement in tree.body[: exec_index + 1]:
            if statement in allowed_pre_exec_statements:
                continue
            if any(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in chain_names
                for node in ast.walk(statement)
            ):
                raise InventoryError(
                    "test/abi/baseline/test_observe_abi_baseline.py observer loader alias escapes before exec_module"
                )

        root_constructor = roots[0][3].value.value.func.value.func

        def require_import(expr: ast.expr, before_index: int) -> ast.stmt:
            name = base_identifier(expr)
            binding = imports.get(name or "")
            if binding is None or binding[2] >= before_index:
                raise InventoryError(
                    "test/abi/baseline/test_observe_abi_baseline.py observer loader imports are unsupported"
                )
            _, statement, _ = binding
            if any(
                event is not statement
                for event in module_namespace_binding_events(tree, name or "")
            ):
                raise InventoryError(
                    "test/abi/baseline/test_observe_abi_baseline.py observer loader import is rebound"
                )
            return statement

        critical_imports = sorted(
            {
                require_import(root_constructor, root_index),
                require_import(specs[0][3].func, spec_index),
                require_import(register_statement.targets[0].value, register_index),
            },
            key=lambda statement: (statement.lineno, statement.col_offset),
        )

        nodes = {
            "spec_target": spec_target,
            "spec_value": specs[0][3],
            "module_target": module_target,
            "module_value": modules[0][3],
            "registration_target": register_statement.targets[0],
            "registration_value": register_statement.value,
            "exec_statement": exec_statement,
        }
        digest_nodes: list[ast.AST] = [
            *critical_imports,
            root_statement,
            path_statement,
            spec_statement,
            *guards,
            module_statement,
            register_statement,
            exec_statement,
        ]
        return (
            (spec_target.id, module_target.id, path_target.id),
            nodes,
            (exec_statement.lineno, exec_statement.col_offset),
            digest_nodes,
        )

    if reviewed_adapter_tree is not None:
        direct_definitions = [
            node
            for node in reviewed_adapter_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_command"
        ]
        all_definitions = [
            node
            for node in ast.walk(reviewed_adapter_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_command"
        ]
        if len(direct_definitions) != 1 or all_definitions != direct_definitions:
            raise InventoryError(
                "tools/observe_abi_baseline.py must define exactly one direct unconditional run_command adapter"
            )
        definition = direct_definitions[0]
        reviewed_adapter_definition = definition
        if any(
            event is not definition
            for event in module_namespace_binding_events(
                reviewed_adapter_tree, "run_command"
            )
        ):
            raise InventoryError(
                "tools/observe_abi_baseline.py run_command adapter is rebound"
            )
        adapter_payload = {
            "version": PYTHON_SEMANTICS_VERSION,
            "reviewed_adapter": "tools/observe_abi_baseline.py:run_command",
            "definition": _canonical_python_ast(definition),
        }
        if reviewed_consumer_tree is not None:
            (
                reviewed_adapter_loader_names,
                reviewed_adapter_loader_nodes,
                reviewed_adapter_exec_position,
                loader_digest_nodes,
            ) = prove_reviewed_loader(reviewed_consumer_tree)
            adapter_payload["reviewed_loader_chain"] = [
                _canonical_python_ast(statement) for statement in loader_digest_nodes
            ]
        reviewed_adapter_digest = _json_fact_digest(adapter_payload)
    elif reviewed_consumer_tree is not None:
        raise InventoryError(
            "test/abi/baseline/test_observe_abi_baseline.py requires the reviewed adapter owner"
        )

    def reviewed_adapter_module_binding(
        path: Path,
        target: ast.AST,
        value: ast.expr,
        aliases: dict[str, str],
    ) -> bool:
        """Recognize the one reviewed importlib loader used by the ABI observer test."""
        return (
            path == reviewed_adapter_consumer_path
            and reviewed_adapter_digest is not None
            and reviewed_adapter_loader_nodes is not None
            and target is reviewed_adapter_loader_nodes["module_target"]
            and value is reviewed_adapter_loader_nodes["module_value"]
        )

    def reviewed_adapter_spec_binding(
        path: Path,
        target: ast.AST,
        value: ast.expr,
        aliases: dict[str, str],
    ) -> bool:
        return (
            path == reviewed_adapter_consumer_path
            and reviewed_adapter_loader_nodes is not None
            and target is reviewed_adapter_loader_nodes["spec_target"]
            and value is reviewed_adapter_loader_nodes["spec_value"]
        )

    def reviewed_adapter_module_registration(
        path: Path,
        target: ast.AST,
        value: ast.expr,
        aliases: dict[str, str],
    ) -> bool:
        return (
            path == reviewed_adapter_consumer_path
            and reviewed_adapter_loader_nodes is not None
            and target is reviewed_adapter_loader_nodes["registration_target"]
            and value is reviewed_adapter_loader_nodes["registration_value"]
        )

    def resolve_local_module(
        importer: Path, module: str | None, level: int = 0
    ) -> Path | None:
        module_parts = module.split(".") if module else []
        if level:
            base = importer.parent
            for _ in range(level - 1):
                base = base.parent
            bases = [base]
        else:
            bases = [root, importer.parent]
        for base in bases:
            target = base.joinpath(*module_parts)
            candidates = (
                target.with_suffix(".py") if module_parts else target / "__init__.py",
                target / "__init__.py",
            )
            for candidate in candidates:
                if candidate in trees:
                    return candidate
        return None

    def resolve_imported_submodule(
        importer: Path, module: str | None, level: int, imported_name: str
    ) -> Path | None:
        combined = ".".join(part for part in (module, imported_name) if part)
        return resolve_local_module(importer, combined, level)

    def static_bound_names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return {
                name for element in target.elts for name in static_bound_names(element)
            }
        if isinstance(target, ast.Starred):
            return static_bound_names(target.value)
        return set()

    module_declaration_types = (
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        *AST_TYPE_ALIAS_TYPES,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )
    module_declarations: dict[Path, list[ast.AST]] = {}
    for path, tree in trees.items():
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        declarations: list[ast.AST] = []
        for node in ast.walk(tree):
            if not isinstance(node, module_declaration_types):
                continue
            current = node
            module_scoped = True
            while current in parents:
                current = parents[current]
                if current is tree:
                    break
                if isinstance(
                    current,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        *AST_TYPE_ALIAS_TYPES,
                        ast.Lambda,
                    ),
                ):
                    module_scoped = False
                    break
            if module_scoped:
                declarations.append(node)
        module_declarations[path] = sorted(
            declarations, key=lambda node: (node.lineno, node.col_offset)
        )

    def analyze_module_exports(
        path: Path, known_exports: dict[Path, dict[str, str]]
    ) -> dict[str, str]:
        aliases: dict[str, str] = {}

        def clear(target: ast.AST) -> None:
            for name in static_bound_names(target):
                _shadow_python_alias(aliases, name)

        def bind(target: ast.AST, value: ast.expr) -> None:
            if isinstance(target, ast.Name):
                if reviewed_adapter_spec_binding(path, target, value, aliases):
                    _shadow_python_alias(aliases, target.id)
                    return
                if reviewed_adapter_module_binding(path, target, value, aliases):
                    _shadow_python_alias(aliases, target.id)
                    return
                dynamic_import, imported_module = _dynamic_process_module(
                    value, aliases
                )
                if (
                    dynamic_import
                    and imported_module in PYTHON_PROCESS_MODULES
                    and resolve_local_module(path, imported_module) is not None
                ):
                    raise InventoryError(
                        f"{path.relative_to(root).as_posix()}:{value.lineno}: local modules shadowing dynamically imported process modules are unsupported"
                    )
                resolved = (
                    imported_module
                    if dynamic_import and imported_module in PYTHON_PROCESS_MODULES
                    else _dotted_name(value, aliases)
                    if not dynamic_import
                    else None
                )
                if resolved in PYTHON_TRACKED_ALIAS_VALUES:
                    _set_python_alias(aliases, target.id, resolved)
                else:
                    _shadow_python_alias(aliases, target.id)
                return
            if (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(value, (ast.Tuple, ast.List))
                and len(target.elts) == len(value.elts)
            ):
                for child_target, child_value in zip(target.elts, value.elts):
                    bind(child_target, child_value)
                return
            clear(target)

        for declaration in module_declarations[path]:
            if isinstance(declaration, ast.Import):
                for imported in declaration.names:
                    bound = imported.asname or imported.name.partition(".")[0]
                    imported_root = imported.name.partition(".")[0]
                    source = resolve_local_module(path, imported.name)
                    if source is not None and imported_root in (
                        PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES
                    ):
                        raise InventoryError(
                            f"{path.relative_to(root).as_posix()}:{declaration.lineno}: local modules shadowing tracked standard-library modules are unsupported"
                        )
                    if source is not None:
                        _shadow_python_alias(aliases, bound)
                        prefix = imported.asname or imported.name
                        for name, resolved in known_exports.get(source, {}).items():
                            _set_python_alias(aliases, f"{prefix}.{name}", resolved)
                        continue
                    if imported_root in PYTHON_PROCESS_MODULES:
                        if imported.asname is None or imported.name == imported_root:
                            _set_python_alias(aliases, bound, imported_root)
                        else:
                            _shadow_python_alias(aliases, bound)
                        continue
                    if imported_root in PYTHON_IMPORT_NAMESPACES:
                        if imported.asname is None or imported.name == imported_root:
                            _set_python_alias(aliases, bound, imported_root)
                        else:
                            _shadow_python_alias(aliases, bound)
                        continue
                    _shadow_python_alias(aliases, bound)
            elif isinstance(declaration, ast.ImportFrom):
                local_source = resolve_local_module(
                    path, declaration.module, declaration.level
                )
                if local_source is not None and declaration.module in (
                    PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES
                ):
                    raise InventoryError(
                        f"{path.relative_to(root).as_posix()}:{declaration.lineno}: local modules shadowing tracked standard-library modules are unsupported"
                    )
                if local_source is not None:
                    exports = known_exports.get(local_source, {})
                    for imported in declaration.names:
                        if imported.name == "*":
                            for name, resolved in exports.items():
                                _set_python_alias(aliases, name, resolved)
                        elif imported.name in exports:
                            _set_python_alias(
                                aliases,
                                imported.asname or imported.name,
                                exports[imported.name],
                            )
                        else:
                            bound = imported.asname or imported.name
                            submodule = resolve_imported_submodule(
                                path,
                                declaration.module,
                                declaration.level,
                                imported.name,
                            )
                            submodule_exports = (
                                known_exports.get(submodule, {}) if submodule else {}
                            )
                            _shadow_python_alias(aliases, bound)
                            for name, resolved in submodule_exports.items():
                                _set_python_alias(aliases, f"{bound}.{name}", resolved)
                    continue
                if declaration.module in PYTHON_PROCESS_MODULES:
                    for imported in declaration.names:
                        if imported.name != "*":
                            _set_python_alias(
                                aliases,
                                imported.asname or imported.name,
                                f"{declaration.module}.{imported.name}",
                            )
                    continue
                if declaration.module == "importlib":
                    for imported in declaration.names:
                        bound = imported.asname or imported.name
                        if imported.name == "import_module":
                            _set_python_alias(aliases, bound, "importlib.import_module")
                        elif imported.name == "__import__":
                            _set_python_alias(aliases, bound, "importlib.__import__")
                        else:
                            _shadow_python_alias(aliases, bound)
                    continue
                if declaration.module == "builtins":
                    for imported in declaration.names:
                        bound = imported.asname or imported.name
                        if imported.name == "__import__":
                            _set_python_alias(aliases, bound, "__import__")
                        elif imported.name in {"globals", "locals", "vars"}:
                            _set_python_alias(
                                aliases, bound, f"builtins.{imported.name}"
                            )
                        elif imported.name == "__dict__":
                            _set_python_alias(aliases, bound, "builtins.__dict__")
                        elif imported.name in {"eval", "exec"}:
                            _set_python_alias(
                                aliases, bound, f"builtins.{imported.name}"
                            )
                        else:
                            _shadow_python_alias(aliases, bound)
                    continue
                if declaration.module == "sys":
                    for imported in declaration.names:
                        bound = imported.asname or imported.name
                        if imported.name == "modules":
                            _set_python_alias(aliases, bound, "sys.modules")
                        else:
                            _shadow_python_alias(aliases, bound)
                    continue
                source = resolve_local_module(
                    path, declaration.module, declaration.level
                )
                exports = known_exports.get(source, {}) if source else {}
                for imported in declaration.names:
                    if imported.name == "*":
                        for name, resolved in exports.items():
                            _set_python_alias(aliases, name, resolved)
                    elif imported.name in exports:
                        _set_python_alias(
                            aliases,
                            imported.asname or imported.name,
                            exports[imported.name],
                        )
                    else:
                        bound = imported.asname or imported.name
                        submodule = resolve_imported_submodule(
                            path,
                            declaration.module,
                            declaration.level,
                            imported.name,
                        )
                        submodule_exports = (
                            known_exports.get(submodule, {}) if submodule else {}
                        )
                        _shadow_python_alias(aliases, bound)
                        for name, resolved in submodule_exports.items():
                            _set_python_alias(aliases, f"{bound}.{name}", resolved)
            elif isinstance(declaration, ast.Assign):
                for target in declaration.targets:
                    bind(target, declaration.value)
            elif isinstance(declaration, ast.AnnAssign):
                if declaration.value is None:
                    clear(declaration.target)
                else:
                    bind(declaration.target, declaration.value)
            elif _is_ast_type_alias(declaration):
                clear(declaration.name)
            elif isinstance(
                declaration, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                if (
                    path == reviewed_adapter_path
                    and isinstance(declaration, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and declaration.name == "run_command"
                ):
                    _set_python_alias(
                        aliases, declaration.name, PYTHON_REVIEWED_ADAPTER_CALL
                    )
                    continue
                process_bases = (
                    {
                        _dotted_name(base, aliases)
                        for base in declaration.bases
                        if _dotted_name(base, aliases)
                        in (PYTHON_PROCESS_CALLS | set(PYTHON_PROCESS_FACTORIES))
                    }
                    if isinstance(declaration, ast.ClassDef)
                    else set()
                )
                if len(process_bases) == 1:
                    _set_python_alias(aliases, declaration.name, process_bases.pop())
                else:
                    _shadow_python_alias(aliases, declaration.name)
        if (
            path == reviewed_adapter_consumer_path
            and reviewed_adapter_loader_names is not None
        ):
            module_name = reviewed_adapter_loader_names[1]
            _shadow_python_alias(aliases, module_name)
            _set_python_alias(
                aliases,
                f"{module_name}.run_command",
                PYTHON_REVIEWED_ADAPTER_CALL,
            )
        return {
            name: resolved
            for name, resolved in aliases.items()
            if resolved in PYTHON_TRACKED_ALIAS_VALUES
        }

    analysis_files = sorted(trees)
    module_process_exports = {path: {} for path in analysis_files}
    while True:
        updated = {
            path: analyze_module_exports(path, module_process_exports)
            for path in analysis_files
        }
        if updated == module_process_exports:
            break
        module_process_exports = updated

    for path in python_files:
        rel = path.relative_to(root).as_posix()
        tree = trees[path]
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        reviewed_loader_contract = tree is reviewed_loader_contract_tree
        reviewed_loader_contract_nodes: set[int] = set()
        if reviewed_loader_contract:
            for definition in tree.body:
                if (
                    isinstance(definition, ast.FunctionDef)
                    and definition.name in REVIEWED_TEST_INVENTORY_LOADER_FUNCTION_NAMES
                    or isinstance(definition, ast.ClassDef)
                    and definition.name in REVIEWED_TEST_INVENTORY_LOADER_CLASS_NAMES
                ):
                    reviewed_loader_contract_nodes.update(
                        id(candidate) for candidate in ast.walk(definition)
                    )
        reviewed_dynamic_popen_calls: set[ast.Call] = set()
        reviewed_capsule_process_conditions: dict[ast.Call, str] = {}
        reviewed_capsule_runtime_nodes: set[int] = set()
        if rel == "tools/check_test_inventory.py":
            capsule_launchers = [
                candidate
                for candidate in tree.body
                if isinstance(candidate, ast.ClassDef)
                and candidate.name == "_PythonToolingCapsuleLauncher"
            ]
            if len(capsule_launchers) == 1:
                capsule_launcher = capsule_launchers[0]
                launcher_digest = _json_fact_digest(
                    {
                        "schema": "reviewed-test-inventory-capsule-launcher-v1-ast",
                        "class": _canonical_python_ast(capsule_launcher),
                    }
                )
                capsule_calls = sorted(
                    (
                        candidate
                        for candidate in ast.walk(capsule_launcher)
                        if isinstance(candidate, ast.Call)
                        and isinstance(candidate.func, ast.Name)
                        and candidate.func.id
                        == "_PYTHON_TOOLING_TRUSTED_SUBPROCESS_RUN"
                    ),
                    key=lambda candidate: (candidate.lineno, candidate.col_offset),
                )
                if (
                    launcher_digest == REVIEWED_TEST_INVENTORY_CAPSULE_LAUNCHER_SHA256
                    and len(capsule_calls) == 3
                ):
                    reviewed_capsule_process_conditions = dict(
                        zip(
                            capsule_calls,
                            (
                                "controller argv0 is not the running Python executable",
                                "controller argv0 is the running Python executable and os.name is posix",
                                "controller argv0 is the running Python executable and os.name is nt",
                            ),
                            strict=True,
                        )
                    )
            capsule_runtimes = [
                candidate
                for candidate in tree.body
                if isinstance(candidate, ast.FunctionDef)
                and candidate.name == "_python_tooling_capsule_runtime"
            ]
            if len(capsule_runtimes) == 1:
                capsule_runtime = capsule_runtimes[0]
                runtime_digest = _json_fact_digest(
                    {
                        "schema": "reviewed-test-inventory-capsule-runtime-v1-ast",
                        "function": _canonical_python_ast(capsule_runtime),
                    }
                )
                if runtime_digest == REVIEWED_TEST_INVENTORY_CAPSULE_RUNTIME_SHA256:
                    reviewed_capsule_runtime_nodes = {
                        id(candidate) for candidate in ast.walk(capsule_runtime)
                    }
            posix_capsule_probes = [
                candidate
                for candidate in tree.body
                if isinstance(candidate, ast.FunctionDef)
                and candidate.name == "_python_tooling_posix_capsule_probe"
            ]
            if len(posix_capsule_probes) == 1:
                posix_capsule_probe = posix_capsule_probes[0]
                probe_digest = _json_fact_digest(
                    {
                        "schema": "reviewed-test-inventory-posix-capsule-probe-v1-ast",
                        "function": _canonical_python_ast(posix_capsule_probe),
                    }
                )
                probe_calls = [
                    candidate
                    for candidate in ast.walk(posix_capsule_probe)
                    if isinstance(candidate, ast.Call)
                    and isinstance(candidate.func, ast.Attribute)
                    and isinstance(candidate.func.value, ast.Name)
                    and candidate.func.value.id == "subprocess"
                    and candidate.func.attr == "run"
                ]
                if (
                    probe_digest == REVIEWED_TEST_INVENTORY_POSIX_CAPSULE_PROBE_SHA256
                    and len(probe_calls) == 1
                ):
                    reviewed_capsule_process_conditions[probe_calls[0]] = (
                        "os.name is posix"
                    )
        if path == reviewed_adapter_path and reviewed_adapter_definition is not None:
            reviewed_run_popen_calls = [
                node
                for node in ast.walk(reviewed_adapter_definition)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and node.func.attr == "Popen"
                and _python_scope(node, parents) is reviewed_adapter_definition
            ]
            if reviewed_run_popen_calls:
                named_definitions = {
                    node.name: node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in {"_windows_supervisor_argv", "run_command", "main"}
                }
                if set(named_definitions) != {
                    "_windows_supervisor_argv",
                    "run_command",
                    "main",
                }:
                    raise InventoryError(
                        "tools/observe_abi_baseline.py reviewed Windows supervisor flow changed"
                    )
                main_definition = named_definitions["main"]
                reviewed_main_popen_calls = [
                    node
                    for node in ast.walk(main_definition)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                    and node.func.attr == "Popen"
                    and _python_scope(node, parents) is main_definition
                ]
                if (
                    len(reviewed_run_popen_calls) != 1
                    or len(reviewed_main_popen_calls) != 1
                ):
                    raise InventoryError(
                        "tools/observe_abi_baseline.py reviewed Windows supervisor Popen count changed"
                    )
                ordered_calls = (
                    reviewed_run_popen_calls[0],
                    reviewed_main_popen_calls[0],
                )
                reviewed_statements = [
                    next(
                        ancestor
                        for ancestor in _ancestor_chain(call, parents)
                        if isinstance(ancestor, ast.stmt)
                    )
                    for call in ordered_calls
                ]
                reviewed_popen_proof = _json_fact_digest(
                    {
                        "version": PYTHON_SEMANTICS_VERSION,
                        "owner": "tools/observe_abi_baseline.py:Windows-supervisor-flow",
                        "adapter_definition_digest": reviewed_adapter_digest,
                        "definitions": {
                            name: _canonical_python_ast(definition)
                            for name, definition in sorted(named_definitions.items())
                        },
                        "statements": [
                            _canonical_python_ast(statement)
                            for statement in reviewed_statements
                        ],
                        "calls": [
                            _canonical_python_ast(call) for call in ordered_calls
                        ],
                    }
                )
                if reviewed_popen_proof != REVIEWED_ADAPTER_DYNAMIC_POPEN_PROOF_DIGEST:
                    raise InventoryError(
                        "tools/observe_abi_baseline.py reviewed Windows supervisor Popen proof changed "
                        f"({reviewed_popen_proof})"
                    )
                reviewed_dynamic_popen_calls.update(ordered_calls)
        if rel == "tools/verify_abi_artifact_parity.py":
            bounded_definitions = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "run_bounded"
            ]
            if len(bounded_definitions) != 1:
                raise InventoryError(
                    "tools/verify_abi_artifact_parity.py reviewed run_bounded owner changed"
                )
            bounded_definition = bounded_definitions[0]
            bounded_popen_calls = [
                node
                for node in ast.walk(bounded_definition)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and node.func.attr == "Popen"
                and _python_scope(node, parents) is bounded_definition
            ]
            if len(bounded_popen_calls) != 1:
                raise InventoryError(
                    "tools/verify_abi_artifact_parity.py reviewed run_bounded Popen count changed"
                )
            bounded_call = bounded_popen_calls[0]
            bounded_statement = next(
                ancestor
                for ancestor in _ancestor_chain(bounded_call, parents)
                if isinstance(ancestor, ast.stmt)
            )
            bounded_proof = _json_fact_digest(
                {
                    "version": PYTHON_SEMANTICS_VERSION,
                    "owner": "tools/verify_abi_artifact_parity.py:run_bounded",
                    "definition": _canonical_python_ast(bounded_definition),
                    "statement": _canonical_python_ast(bounded_statement),
                    "call": _canonical_python_ast(bounded_call),
                }
            )
            if bounded_proof != REVIEWED_BOUNDED_DYNAMIC_POPEN_PROOF_DIGEST:
                raise InventoryError(
                    "tools/verify_abi_artifact_parity.py reviewed run_bounded Popen proof changed "
                    f"({bounded_proof})"
                )
            reviewed_dynamic_popen_calls.add(bounded_call)

        definition_time_roots_cache: dict[ast.AST, tuple[ast.AST, ...]] = {}

        def definition_time_roots(scope: ast.AST) -> tuple[ast.AST, ...]:
            cached = definition_time_roots_cache.get(scope)
            if cached is not None:
                return cached
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotations = [
                    argument.annotation
                    for argument in (
                        *scope.args.posonlyargs,
                        *scope.args.args,
                        *scope.args.kwonlyargs,
                    )
                    if argument.annotation is not None
                ]
                if scope.args.vararg is not None and scope.args.vararg.annotation:
                    annotations.append(scope.args.vararg.annotation)
                if scope.args.kwarg is not None and scope.args.kwarg.annotation:
                    annotations.append(scope.args.kwarg.annotation)
                roots = [
                    *scope.decorator_list,
                    *scope.args.defaults,
                    *(item for item in scope.args.kw_defaults if item is not None),
                ]
                if not getattr(scope, "type_params", ()):
                    roots.extend(annotations)
                    roots.extend(item for item in (scope.returns,) if item is not None)
                result = tuple(roots)
            elif isinstance(scope, ast.Lambda):
                result = tuple(
                    [
                        *scope.args.defaults,
                        *(item for item in scope.args.kw_defaults if item is not None),
                    ]
                )
            elif isinstance(scope, ast.ClassDef):
                if getattr(scope, "type_params", ()):
                    result = tuple(scope.decorator_list)
                else:
                    result = tuple(
                        [*scope.decorator_list, *scope.bases, *scope.keywords]
                    )
            else:
                result = ()
            definition_time_roots_cache[scope] = result
            return result

        lexical_scope_cache: dict[ast.AST, ast.AST] = {}

        def lexical_scope(node: ast.AST) -> ast.AST:
            cached = lexical_scope_cache.get(node)
            if cached is not None:
                return cached
            child = node
            while child in parents:
                current = parents[child]
                if isinstance(
                    current,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        *AST_TYPE_ALIAS_TYPES,
                        ast.Lambda,
                        ast.Module,
                    ),
                ):
                    if child in definition_time_roots(current):
                        child = current
                        continue
                    lexical_scope_cache[node] = current
                    return current
                child = current
            lexical_scope_cache[node] = tree
            return tree

        enclosing_alias_scope_cache: dict[ast.AST, ast.AST] = {}

        def enclosing_alias_scope(node: ast.AST) -> ast.AST:
            cached = enclosing_alias_scope_cache.get(node)
            if cached is not None:
                return cached
            current = node
            while current in parents:
                current = parents[current]
                if isinstance(
                    current,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module),
                ):
                    enclosing_alias_scope_cache[node] = current
                    return current
            enclosing_alias_scope_cache[node] = tree
            return tree

        control_flow_nodes = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            *AST_TRY_STAR_TYPES,
            ast.Match,
        )

        control_flow_parent_cache: dict[ast.AST, bool] = {}

        def has_control_flow_parent(node: ast.AST) -> bool:
            if node in control_flow_parent_cache:
                return control_flow_parent_cache[node]
            scope = lexical_scope(node)
            current = node
            while current in parents:
                current = parents[current]
                if current is scope:
                    control_flow_parent_cache[node] = False
                    return False
                if isinstance(current, control_flow_nodes):
                    control_flow_parent_cache[node] = True
                    return True
            control_flow_parent_cache[node] = False
            return False

        def bound_names(target: ast.AST) -> set[str]:
            if isinstance(target, ast.Name):
                return {target.id}
            if isinstance(target, (ast.Tuple, ast.List)):
                return {
                    name for element in target.elts for name in bound_names(element)
                }
            if isinstance(target, ast.Starred):
                return bound_names(target.value)
            return set()

        def reviewed_frozen_import_module_call(value: ast.Call) -> bool:
            if rel != "test/build/test_test_inventory.py":
                return False
            scope = lexical_scope(value)
            scope_parent = parents.get(scope)
            if not (
                isinstance(scope, ast.FunctionDef)
                and isinstance(scope_parent, ast.ClassDef)
                and scope_parent.name == "TestInventoryTests"
            ):
                return False
            if not (
                isinstance(value.func, ast.Attribute)
                and value.func.attr == "pop"
                and isinstance(value.func.value, ast.Attribute)
                and value.func.value.attr == "modules"
                and isinstance(value.func.value.value, ast.Name)
                and value.func.value.value.id == "sys"
                and len(value.args) == 2
                and isinstance(value.args[1], ast.Constant)
                and value.args[1].value is None
                and not value.keywords
            ):
                return False
            if scope.name == "test_python_root_commands_are_directly_executable":
                if not (
                    isinstance(value.args[0], ast.Name) and value.args[0].id == "name"
                ):
                    return False
                current: ast.AST = value
                while current in parents and parents[current] is not scope:
                    current = parents[current]
                    if isinstance(current, ast.For):
                        return (
                            isinstance(current.target, ast.Name)
                            and current.target.id == "name"
                            and isinstance(current.iter, ast.Name)
                            and current.iter.id == "legacy_module_names"
                        )
                return False
            if scope.name != "test_predicate_x86_and_mode_row_mutations_fail":
                return False
            if (
                isinstance(value.args[0], ast.Constant)
                and value.args[0].value == "fractions"
            ):
                return True
            if not (
                isinstance(value.args[0], ast.Name)
                and value.args[0].id == "shadowed_name"
            ):
                return False
            current: ast.AST = value
            while current in parents and parents[current] is not scope:
                current = parents[current]
                if isinstance(current, ast.For):
                    return (
                        isinstance(current.target, ast.Name)
                        and current.target.id == "shadowed_name"
                        and isinstance(current.iter, ast.Tuple)
                        and [
                            element.value
                            for element in current.iter.elts
                            if isinstance(element, ast.Constant)
                        ]
                        == ["csv", "platform", "html"]
                        and all(
                            isinstance(element, ast.Constant)
                            for element in current.iter.elts
                        )
                    )
            return False

        def reviewed_frozen_import_module_probe(
            declaration: ast.Assign | ast.AnnAssign,
        ) -> bool:
            targets = (
                declaration.targets
                if isinstance(declaration, ast.Assign)
                else [declaration.target]
            )
            value = declaration.value
            target_name = (
                targets[0].id
                if len(targets) == 1 and isinstance(targets[0], ast.Name)
                else None
            )
            if not (
                target_name in {"previous_module", "previous_fraction"}
                and isinstance(value, ast.Call)
                and reviewed_frozen_import_module_call(value)
            ):
                return False
            if target_name == "previous_fraction":
                return (
                    isinstance(value.args[0], ast.Constant)
                    and value.args[0].value == "fractions"
                )
            if not (
                isinstance(value.args[0], ast.Name)
                and value.args[0].id == "shadowed_name"
            ):
                return False
            return True

        def reviewed_loader_contract_assignment(
            target: ast.AST, value: ast.expr, active_aliases: dict[str, str]
        ) -> bool:
            if not reviewed_loader_contract:
                return False
            scope = lexical_scope(target)
            if not (
                isinstance(scope, ast.FunctionDef)
                and scope.name == "__init__"
                and isinstance(parents.get(scope), ast.ClassDef)
                and parents[scope].name == "_PythonExecutionClosure"
                and isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                return False
            expected_values = {
                "trusted_compile": "compile",
                "trusted_exec": "exec",
                "trusted_spec_from_file_location": "importlib.util.spec_from_file_location",
                "trusted_import_module": "importlib.import_module",
                "trusted_builtin_find_spec": "importlib.machinery.BuiltinImporter.find_spec",
                "trusted_frozen_find_spec": "importlib.machinery.FrozenImporter.find_spec",
                "trusted_path_find_spec": "importlib.machinery.PathFinder.find_spec",
            }
            return _dotted_name(value, active_aliases) == expected_values.get(
                target.attr
            )

        def reviewed_loader_contract_node(node: ast.AST) -> bool:
            return id(node) in reviewed_loader_contract_nodes

        reviewed_bootstrap_contract = False
        if (
            _embedded_origin is not None
            and _embedded_origin.get("file") == "tools/check_test_inventory.py"
            and _embedded_origin.get("physical_function")
            == "_python_tooling_posix_capsule_probe"
        ):
            physical_tree = trees.get(root / "tools/check_test_inventory.py")

            def exact_bootstrap_constant_assignment(
                name: str,
            ) -> ast.Assign | None:
                if physical_tree is None:
                    return None
                matches = [
                    candidate
                    for candidate in physical_tree.body
                    if isinstance(candidate, ast.Assign)
                    and len(candidate.targets) == 1
                    and isinstance(candidate.targets[0], ast.Name)
                    and candidate.targets[0].id == name
                    and isinstance(candidate.targets[0].ctx, ast.Store)
                    and candidate.type_comment is None
                ]
                return matches[0] if len(matches) == 1 else None

            source_assignment = exact_bootstrap_constant_assignment(
                "_PYTHON_TOOLING_BOOTSTRAP_SOURCE"
            )
            digest_assignment = exact_bootstrap_constant_assignment(
                "_PYTHON_TOOLING_BOOTSTRAP_SHA256"
            )
            if (
                source_assignment is not None
                and isinstance(source_assignment.value, ast.Constant)
                and isinstance(source_assignment.value.value, str)
                and digest_assignment is not None
                and isinstance(digest_assignment.value, ast.Constant)
                and digest_assignment.value.value
                == REVIEWED_TEST_INVENTORY_BOOTSTRAP_SHA256
                and hashlib.sha256(
                    source_assignment.value.value.encode("utf-8")
                ).hexdigest()
                == REVIEWED_TEST_INVENTORY_BOOTSTRAP_SHA256
            ):
                try:
                    reviewed_bootstrap_tree = ast.parse(source_assignment.value.value)
                except (SyntaxError, RecursionError):
                    pass
                else:
                    reviewed_bootstrap_contract = _canonical_python_ast(
                        reviewed_bootstrap_tree
                    ) == _canonical_python_ast(tree)
        reviewed_bootstrap_contract_nodes = (
            {id(candidate) for candidate in ast.walk(tree)}
            if reviewed_bootstrap_contract
            else set()
        )

        def reviewed_bootstrap_contract_node(node: ast.AST) -> bool:
            return id(node) in reviewed_bootstrap_contract_nodes

        def reviewed_process_contract_node(node: ast.AST) -> bool:
            return (
                reviewed_loader_contract_node(node)
                or reviewed_bootstrap_contract_node(node)
                or id(node) in reviewed_capsule_runtime_nodes
            )

        def reviewed_bootstrap_process_sink_assignment(
            candidate_assignment: ast.AST, target: ast.AST, value: ast.expr
        ) -> bool:
            if (
                not reviewed_bootstrap_contract
                or candidate_assignment not in tree.body
                or not isinstance(candidate_assignment, ast.Assign)
                or len(candidate_assignment.targets) != 1
                or candidate_assignment.targets[0] is not target
                or candidate_assignment.value is not value
                or candidate_assignment.type_comment is not None
                or not isinstance(target, ast.Attribute)
                or target.attr != "run"
                or not isinstance(target.ctx, ast.Store)
                or not isinstance(target.value, ast.Name)
                or target.value.id != "subprocess"
                or not isinstance(target.value.ctx, ast.Load)
                or not isinstance(value, ast.Name)
                or value.id != "capsule_run"
                or not isinstance(value.ctx, ast.Load)
            ):
                return False
            process_sink_assignments = [
                candidate
                for candidate in ast.walk(tree)
                if isinstance(candidate, ast.Assign)
                and any(
                    isinstance(candidate_target, ast.Attribute)
                    and isinstance(candidate_target.value, ast.Name)
                    and candidate_target.value.id == "subprocess"
                    for candidate_target in candidate.targets
                )
            ]
            return process_sink_assignments == [candidate_assignment]

        class_scopes = sorted(
            (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)),
            key=lambda node: (node.lineno, node.col_offset),
        )
        scopes: list[ast.AST] = [tree]
        scopes.extend(
            sorted(
                (
                    node
                    for node in ast.walk(tree)
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                    )
                ),
                key=lambda node: (node.lineno, node.col_offset),
            )
        )
        scopes.extend(
            sorted(
                (node for node in ast.walk(tree) if _is_ast_type_alias(node)),
                key=lambda node: (node.lineno, node.col_offset),
            )
        )
        scope_declarations: dict[ast.AST, list[ast.AST]] = {}
        scope_parents: dict[ast.AST, ast.AST | None] = {tree: None}
        for scope in scopes:
            if scope is not tree:
                scope_parents[scope] = enclosing_alias_scope(scope)
            scope_declarations[scope] = sorted(
                (
                    node
                    for node in ast.walk(scope)
                    if node is not scope
                    and lexical_scope(node) is scope
                    and isinstance(
                        node,
                        (
                            ast.Import,
                            ast.ImportFrom,
                            ast.Assign,
                            ast.AnnAssign,
                            ast.AugAssign,
                            ast.Delete,
                            *AST_TYPE_ALIAS_TYPES,
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.ClassDef,
                        ),
                    )
                ),
                key=lambda node: (node.lineno, node.col_offset),
            )
        class_declaration_types = (
            ast.Import,
            ast.ImportFrom,
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Delete,
            *AST_TYPE_ALIAS_TYPES,
        )
        for class_scope in class_scopes:
            scope_parents[class_scope] = enclosing_alias_scope(class_scope)
            scope_declarations[class_scope] = []
        for node in ast.walk(tree):
            if not isinstance(node, class_declaration_types):
                continue
            scope = lexical_scope(node)
            if isinstance(scope, ast.ClassDef):
                scope_declarations[scope].append(node)
        for class_scope in class_scopes:
            scope_declarations[class_scope].sort(
                key=lambda node: (node.lineno, node.col_offset)
            )

        def pattern_bound_names(pattern: ast.pattern) -> set[str]:
            names: set[str] = set()
            for candidate in ast.walk(pattern):
                if isinstance(candidate, (ast.MatchAs, ast.MatchStar)):
                    if candidate.name:
                        names.add(candidate.name)
                elif isinstance(candidate, ast.MatchMapping) and candidate.rest:
                    names.add(candidate.rest)
            return names

        scope_local_names: dict[ast.AST, set[str]] = {}
        scope_binding_counts: dict[ast.AST, Counter[str]] = {}
        for scope in scopes:
            if not isinstance(
                scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                scope_local_names[scope] = set()
                scope_binding_counts[scope] = Counter()
                continue
            local_counts: Counter[str] = Counter()
            global_names: set[str] = set()
            nonlocal_names: set[str] = set()
            parameters = [
                *scope.args.posonlyargs,
                *scope.args.args,
                *scope.args.kwonlyargs,
            ]
            if scope.args.vararg is not None:
                parameters.append(scope.args.vararg)
            if scope.args.kwarg is not None:
                parameters.append(scope.args.kwarg)
            local_counts.update(parameter.arg for parameter in parameters)
            local_counts.update(
                type_parameter.name
                for type_parameter in getattr(scope, "type_params", ())
            )
            for candidate in ast.walk(scope):
                if candidate is scope or lexical_scope(candidate) is not scope:
                    continue
                if isinstance(candidate, ast.Global):
                    global_names.update(candidate.names)
                elif isinstance(candidate, ast.Nonlocal):
                    nonlocal_names.update(candidate.names)
                elif isinstance(candidate, ast.Import):
                    local_counts.update(
                        imported.asname or imported.name.partition(".")[0]
                        for imported in candidate.names
                    )
                elif isinstance(candidate, ast.ImportFrom):
                    local_counts.update(
                        imported.asname or imported.name
                        for imported in candidate.names
                        if imported.name != "*"
                    )
                elif isinstance(candidate, ast.Assign):
                    for target in candidate.targets:
                        local_counts.update(bound_names(target))
                elif isinstance(candidate, (ast.AnnAssign, ast.AugAssign)):
                    local_counts.update(bound_names(candidate.target))
                elif isinstance(candidate, ast.Delete):
                    for target in candidate.targets:
                        local_counts.update(bound_names(target))
                elif _is_ast_type_alias(candidate):
                    local_counts.update(bound_names(candidate.name))
                elif isinstance(candidate, (ast.For, ast.AsyncFor)):
                    local_counts.update(bound_names(candidate.target))
                elif isinstance(candidate, (ast.With, ast.AsyncWith)):
                    for item in candidate.items:
                        if item.optional_vars is not None:
                            local_counts.update(bound_names(item.optional_vars))
                elif isinstance(candidate, ast.NamedExpr):
                    local_counts.update(bound_names(candidate.target))
                elif isinstance(candidate, ast.ExceptHandler) and candidate.name:
                    local_counts[candidate.name] += 1
                elif isinstance(
                    candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    local_counts[candidate.name] += 1
                elif isinstance(candidate, ast.Match):
                    for case in candidate.cases:
                        local_counts.update(pattern_bound_names(case.pattern))
            for declared_global in global_names | nonlocal_names:
                local_counts.pop(declared_global, None)
            scope_local_names[scope] = set(local_counts)
            scope_binding_counts[scope] = local_counts
        module_counts: Counter[str] = Counter()
        for declaration in scope_declarations[tree]:
            if isinstance(declaration, ast.Import):
                module_counts.update(
                    imported.asname or imported.name.partition(".")[0]
                    for imported in declaration.names
                )
            elif isinstance(declaration, ast.ImportFrom):
                module_counts.update(
                    imported.asname or imported.name
                    for imported in declaration.names
                    if imported.name != "*"
                )
            elif isinstance(declaration, ast.Assign):
                for target in declaration.targets:
                    module_counts.update(bound_names(target))
            elif isinstance(declaration, (ast.AnnAssign, ast.AugAssign)):
                module_counts.update(bound_names(declaration.target))
            elif _is_ast_type_alias(declaration):
                module_counts.update(bound_names(declaration.name))
            elif isinstance(declaration, ast.Delete):
                for target in declaration.targets:
                    module_counts.update(bound_names(target))
            elif isinstance(
                declaration, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                module_counts[declaration.name] += 1
        scope_binding_counts[tree] = module_counts

        def replay_declarations(
            scope: ast.AST,
            aliases: dict[str, str],
            declarations: tuple[ast.AST, ...],
        ) -> dict[str, str]:
            def contains_process_value(value: ast.expr) -> bool:
                effective_aliases = aliases
                active_comprehension_names = _active_comprehension_bindings(
                    value, parents
                )
                if active_comprehension_names:
                    effective_aliases = dict(aliases)
                    for active_name in active_comprehension_names:
                        _shadow_python_alias(effective_aliases, active_name)
                if isinstance(value, (ast.Name, ast.Attribute)):
                    resolved_value = _dotted_name(value, effective_aliases)
                    if (
                        resolved_value in PYTHON_TRACKED_ALIAS_VALUES
                        and resolved_value != "sys"
                    ):
                        return True
                    if isinstance(value, ast.Attribute):
                        base = _normalize_python_process_callable(
                            _dotted_name(value.value, effective_aliases)
                        )
                        return base in (
                            PYTHON_PROCESS_CALLS
                            | PYTHON_PROCESS_FACTORY_RESULTS
                            | PYTHON_IMPORT_HELPERS
                        ) or (
                            value.attr == "__dict__"
                            and base
                            in (PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES)
                        )
                    return False
                if isinstance(value, ast.Call):
                    dynamic_import, imported_module = _dynamic_process_module(
                        value, effective_aliases
                    )
                    if (
                        dynamic_import
                        and imported_module in PYTHON_PROCESS_MODULES
                        and resolve_local_module(path, imported_module) is not None
                    ):
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: local modules shadowing dynamically imported process modules are unsupported"
                        )
                    if dynamic_import and imported_module in PYTHON_PROCESS_MODULES:
                        return True
                    called = _normalize_python_process_callable(
                        _dotted_name(value.func, effective_aliases)
                    )
                    values = [
                        *value.args,
                        *(keyword.value for keyword in value.keywords),
                    ]
                    if (
                        called
                        in {
                            "vars",
                            "builtins.vars",
                            "__builtins__.vars",
                        }
                        and len(value.args) == 1
                        and not value.keywords
                    ):
                        return contains_process_value(value.args[0])
                    if called in (
                        PYTHON_PROCESS_CALLS | {PYTHON_REVIEWED_ADAPTER_CALL}
                    ):
                        return any(contains_process_value(item) for item in values)
                return any(
                    contains_process_value(child)
                    for child in ast.iter_child_nodes(value)
                    if isinstance(child, ast.expr)
                )

            def clear_target(target: ast.expr) -> None:
                if isinstance(target, ast.Name):
                    _shadow_python_alias(aliases, target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for element in target.elts:
                        clear_target(element)
                elif isinstance(target, ast.Starred):
                    clear_target(target.value)
                elif isinstance(target, ast.Attribute):
                    resolved = _dotted_name(target, aliases)
                    if resolved in PYTHON_TRACKED_ALIAS_VALUES:
                        if isinstance(
                            declaration, ast.Assign
                        ) and reviewed_bootstrap_process_sink_assignment(
                            declaration, target, declaration.value
                        ):
                            return
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: mutating tracked process alias attributes is unsupported"
                        )
                elif isinstance(target, ast.Subscript):
                    resolved = _dotted_name(target, aliases)
                    if resolved in PYTHON_TRACKED_ALIAS_VALUES:
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: mutating tracked process alias subscripts is unsupported"
                        )

            def bind_target(target: ast.expr, value: ast.expr) -> None:
                if reviewed_adapter_module_registration(path, target, value, aliases):
                    return
                if isinstance(target, ast.Name):
                    if reviewed_adapter_spec_binding(path, target, value, aliases):
                        _shadow_python_alias(aliases, target.id)
                        return
                    if reviewed_adapter_module_binding(path, target, value, aliases):
                        if isinstance(scope, ast.ClassDef):
                            raise InventoryError(
                                f"{rel}:{declaration.lineno}: reviewed adapter aliases in class namespaces are unsupported"
                            )
                        _shadow_python_alias(aliases, target.id)
                        return
                    dynamic_import, imported_module = _dynamic_process_module(
                        value, aliases
                    )
                    if dynamic_import and imported_module is None:
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: dynamic module imports require a literal module name"
                        )
                    if dynamic_import and imported_module not in PYTHON_PROCESS_MODULES:
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: dynamic imports of non-process modules are unsupported"
                        )
                    resolved = (
                        imported_module
                        if dynamic_import and imported_module in PYTHON_PROCESS_MODULES
                        else _dotted_name(value, aliases)
                        if not dynamic_import
                        else None
                    )
                    if resolved in PYTHON_TRACKED_ALIAS_VALUES:
                        if isinstance(scope, ast.ClassDef):
                            raise InventoryError(
                                f"{rel}:{declaration.lineno}: process aliases in class namespaces are unsupported"
                            )
                        _set_python_alias(aliases, target.id, resolved)
                    elif contains_process_value(value):
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: process alias uses an unsupported value expression"
                        )
                    else:
                        _shadow_python_alias(aliases, target.id)
                    return
                if isinstance(target, (ast.Tuple, ast.List)):
                    if isinstance(value, (ast.Tuple, ast.List)) and len(
                        target.elts
                    ) == len(value.elts):
                        for child_target, child_value in zip(target.elts, value.elts):
                            bind_target(child_target, child_value)
                    elif contains_process_value(value):
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: process alias destructuring must have equal static shapes"
                        )
                    else:
                        clear_target(target)
                    return
                if contains_process_value(value):
                    if reviewed_loader_contract_assignment(target, value, aliases):
                        clear_target(target)
                        return
                    raise InventoryError(
                        f"{rel}:{declaration.lineno}: process alias uses an unsupported assignment target"
                    )
                clear_target(target)

            for declaration in declarations:
                rebound_names: set[str] = set()
                if isinstance(declaration, ast.Import):
                    rebound_names.update(
                        imported.asname or imported.name.partition(".")[0]
                        for imported in declaration.names
                    )
                elif isinstance(declaration, ast.ImportFrom):
                    rebound_names.update(
                        imported.asname or imported.name
                        for imported in declaration.names
                        if imported.name != "*"
                    )
                elif isinstance(declaration, ast.Assign):
                    for target in declaration.targets:
                        rebound_names.update(bound_names(target))
                elif isinstance(declaration, (ast.AnnAssign, ast.AugAssign)):
                    rebound_names.update(bound_names(declaration.target))
                elif _is_ast_type_alias(declaration):
                    rebound_names.update(bound_names(declaration.name))
                elif isinstance(declaration, ast.Delete):
                    for target in declaration.targets:
                        rebound_names.update(bound_names(target))
                elif isinstance(
                    declaration,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    rebound_names.add(declaration.name)
                if not isinstance(scope, ast.ClassDef):
                    for rebound_name in rebound_names:
                        if any(
                            (
                                alias_name == rebound_name
                                or alias_name.startswith(rebound_name + ".")
                            )
                            and resolved
                            in {
                                PYTHON_REVIEWED_ADAPTER_MODULE,
                                PYTHON_REVIEWED_ADAPTER_CALL,
                            }
                            for alias_name, resolved in aliases.items()
                        ):
                            raise InventoryError(
                                f"{rel}:{declaration.lineno}: shadowing a reviewed adapter alias is unsupported"
                            )
                if isinstance(declaration, ast.Import):
                    for imported in declaration.names:
                        imported_root = imported.name.partition(".")[0]
                        source = resolve_local_module(path, imported.name)
                        if source is not None and imported_root in (
                            PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES
                        ):
                            raise InventoryError(
                                f"{rel}:{declaration.lineno}: local modules shadowing tracked standard-library modules are unsupported"
                            )
                        if source is not None:
                            bound = imported.asname or imported_root
                            _shadow_python_alias(aliases, bound)
                            exports = module_process_exports.get(source, {})
                            if exports and isinstance(scope, ast.ClassDef):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: repository-local process aliases in class namespaces are unsupported"
                                )
                            if exports and has_control_flow_parent(declaration):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: conditional repository-local process alias imports are unsupported"
                                )
                            prefix = imported.asname or imported.name
                            for name, resolved in exports.items():
                                _set_python_alias(aliases, f"{prefix}.{name}", resolved)
                        elif imported_root in PYTHON_PROCESS_MODULES:
                            if isinstance(scope, ast.ClassDef):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: process aliases in class namespaces are unsupported"
                                )
                            if has_control_flow_parent(declaration):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: conditional process-module imports are unsupported"
                                )
                            bound = imported.asname or imported_root
                            if (
                                imported.asname is not None
                                and imported.name != imported_root
                            ):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: aliased process submodule imports are unsupported"
                                )
                            _set_python_alias(aliases, bound, imported_root)
                        elif imported_root in PYTHON_IMPORT_NAMESPACES:
                            bound = imported.asname or imported_root
                            if (
                                imported.asname is not None
                                and imported.name != imported_root
                            ):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: aliased importlib submodule imports are unsupported"
                                )
                            _set_python_alias(aliases, bound, imported_root)
                        else:
                            bound = imported.asname or imported.name.partition(".")[0]
                            _shadow_python_alias(aliases, bound)
                            exports = (
                                module_process_exports.get(source, {})
                                if source is not None
                                else {}
                            )
                            if exports and isinstance(scope, ast.ClassDef):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: repository-local process aliases in class namespaces are unsupported"
                                )
                            if exports and has_control_flow_parent(declaration):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: conditional repository-local process alias imports are unsupported"
                                )
                            prefix = imported.asname or imported.name
                            for name, resolved in exports.items():
                                _set_python_alias(aliases, f"{prefix}.{name}", resolved)
                elif isinstance(declaration, ast.ImportFrom):
                    local_source = resolve_local_module(
                        path, declaration.module, declaration.level
                    )
                    if local_source is not None and declaration.module in (
                        PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES
                    ):
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: local modules shadowing tracked standard-library modules are unsupported"
                        )
                    if local_source is not None:
                        exports = module_process_exports.get(local_source, {})
                        for imported in declaration.names:
                            if imported.name == "*":
                                if exports:
                                    raise InventoryError(
                                        f"{rel}:{declaration.lineno}: wildcard repository-local process alias imports are unsupported"
                                    )
                                continue
                            bound = imported.asname or imported.name
                            resolved = exports.get(imported.name)
                            if resolved is None:
                                submodule = resolve_imported_submodule(
                                    path,
                                    declaration.module,
                                    declaration.level,
                                    imported.name,
                                )
                                submodule_exports = (
                                    module_process_exports.get(submodule, {})
                                    if submodule is not None
                                    else {}
                                )
                                _shadow_python_alias(aliases, bound)
                                if not submodule_exports:
                                    continue
                                if isinstance(scope, ast.ClassDef):
                                    raise InventoryError(
                                        f"{rel}:{declaration.lineno}: repository-local process aliases in class namespaces are unsupported"
                                    )
                                if has_control_flow_parent(declaration):
                                    raise InventoryError(
                                        f"{rel}:{declaration.lineno}: conditional repository-local process alias imports are unsupported"
                                    )
                                for (
                                    name,
                                    imported_resolved,
                                ) in submodule_exports.items():
                                    _set_python_alias(
                                        aliases,
                                        f"{bound}.{name}",
                                        imported_resolved,
                                    )
                                continue
                            if isinstance(scope, ast.ClassDef):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: repository-local process aliases in class namespaces are unsupported"
                                )
                            if has_control_flow_parent(declaration):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: conditional repository-local process alias imports are unsupported"
                                )
                            _set_python_alias(aliases, bound, resolved)
                    elif declaration.module in PYTHON_PROCESS_MODULES:
                        if isinstance(scope, ast.ClassDef):
                            raise InventoryError(
                                f"{rel}:{declaration.lineno}: process aliases in class namespaces are unsupported"
                            )
                        if has_control_flow_parent(declaration):
                            raise InventoryError(
                                f"{rel}:{declaration.lineno}: conditional process-module imports are unsupported"
                            )
                        for imported in declaration.names:
                            if imported.name == "*":
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: wildcard process-module imports are unsupported"
                                )
                            _set_python_alias(
                                aliases,
                                imported.asname or imported.name,
                                f"{declaration.module}.{imported.name}",
                            )
                    elif declaration.module == "importlib":
                        for imported in declaration.names:
                            bound = imported.asname or imported.name
                            if imported.name == "import_module":
                                _set_python_alias(
                                    aliases, bound, "importlib.import_module"
                                )
                            elif imported.name == "__import__":
                                _set_python_alias(
                                    aliases, bound, "importlib.__import__"
                                )
                            else:
                                _shadow_python_alias(aliases, bound)
                    elif declaration.module == "builtins":
                        for imported in declaration.names:
                            bound = imported.asname or imported.name
                            if imported.name == "__import__":
                                _set_python_alias(aliases, bound, "__import__")
                            elif imported.name in {"globals", "locals", "vars"}:
                                _set_python_alias(
                                    aliases, bound, f"builtins.{imported.name}"
                                )
                            elif imported.name == "__dict__":
                                _set_python_alias(aliases, bound, "builtins.__dict__")
                            elif imported.name in {"eval", "exec"}:
                                _set_python_alias(
                                    aliases, bound, f"builtins.{imported.name}"
                                )
                            else:
                                _shadow_python_alias(aliases, bound)
                    elif declaration.module == "sys":
                        for imported in declaration.names:
                            bound = imported.asname or imported.name
                            if imported.name == "modules":
                                _set_python_alias(aliases, bound, "sys.modules")
                            else:
                                _shadow_python_alias(aliases, bound)
                    else:
                        source = resolve_local_module(
                            path, declaration.module, declaration.level
                        )
                        exports = (
                            module_process_exports.get(source, {})
                            if source is not None
                            else {}
                        )
                        for imported in declaration.names:
                            if imported.name == "*":
                                if exports:
                                    raise InventoryError(
                                        f"{rel}:{declaration.lineno}: wildcard repository-local process alias imports are unsupported"
                                    )
                                continue
                            bound = imported.asname or imported.name
                            resolved = exports.get(imported.name)
                            if resolved is None:
                                submodule = resolve_imported_submodule(
                                    path,
                                    declaration.module,
                                    declaration.level,
                                    imported.name,
                                )
                                submodule_exports = (
                                    module_process_exports.get(submodule, {})
                                    if submodule is not None
                                    else {}
                                )
                                _shadow_python_alias(aliases, bound)
                                if not submodule_exports:
                                    continue
                                if isinstance(scope, ast.ClassDef):
                                    raise InventoryError(
                                        f"{rel}:{declaration.lineno}: repository-local process aliases in class namespaces are unsupported"
                                    )
                                if has_control_flow_parent(declaration):
                                    raise InventoryError(
                                        f"{rel}:{declaration.lineno}: conditional repository-local process alias imports are unsupported"
                                    )
                                for (
                                    name,
                                    imported_resolved,
                                ) in submodule_exports.items():
                                    _set_python_alias(
                                        aliases,
                                        f"{bound}.{name}",
                                        imported_resolved,
                                    )
                                continue
                            if isinstance(scope, ast.ClassDef):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: repository-local process aliases in class namespaces are unsupported"
                                )
                            if has_control_flow_parent(declaration):
                                raise InventoryError(
                                    f"{rel}:{declaration.lineno}: conditional repository-local process alias imports are unsupported"
                                )
                            _set_python_alias(aliases, bound, resolved)
                elif _is_ast_type_alias(declaration):
                    clear_target(declaration.name)
                elif isinstance(declaration, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        declaration.targets
                        if isinstance(declaration, ast.Assign)
                        else [declaration.target]
                    )
                    value = declaration.value
                    if reviewed_frozen_import_module_probe(declaration):
                        for target in targets:
                            clear_target(target)
                        continue
                    guarded_names = {
                        name for target in targets for name in bound_names(target)
                    }
                    guarded_process_alias = any(
                        aliases.get(name) in PYTHON_TRACKED_ALIAS_VALUES
                        for name in guarded_names
                    )
                    guarded_process_value = isinstance(
                        value, ast.expr
                    ) and contains_process_value(value)
                    if reviewed_process_contract_node(declaration) and (
                        guarded_process_alias or guarded_process_value
                    ):
                        for target in targets:
                            clear_target(target)
                        continue
                    if has_control_flow_parent(declaration) and (
                        guarded_process_alias or guarded_process_value
                    ):
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: control-flow-dependent process alias assignment is unsupported"
                        )
                    if isinstance(value, ast.expr):
                        for target in targets:
                            bind_target(target, value)
                    else:
                        for target in targets:
                            clear_target(target)
                elif isinstance(declaration, ast.AugAssign):
                    names = bound_names(declaration.target)
                    if any(
                        aliases.get(name) in PYTHON_TRACKED_ALIAS_VALUES
                        for name in names
                    ) or contains_process_value(declaration.value):
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: augmented process alias assignment is unsupported"
                        )
                    clear_target(declaration.target)
                elif isinstance(declaration, ast.Delete):
                    names = {
                        name
                        for target in declaration.targets
                        for name in bound_names(target)
                    }
                    if any(
                        aliases.get(name) in PYTHON_TRACKED_ALIAS_VALUES
                        for name in names
                    ):
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: deleting a process alias is unsupported"
                        )
                    for target in declaration.targets:
                        target_names = bound_names(target)
                        previous = {name: aliases.get(name) for name in target_names}
                        clear_target(target)
                        if scope is not tree:
                            continue
                        for name in target_names:
                            if previous[name] != PYTHON_SHADOWED_ALIAS:
                                continue
                            if name == "__import__":
                                _set_python_alias(aliases, name, "__import__")
                            elif name in {"globals", "locals", "vars"}:
                                _set_python_alias(aliases, name, name)
                elif isinstance(
                    declaration,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    if (
                        path == reviewed_adapter_path
                        and isinstance(
                            declaration, (ast.FunctionDef, ast.AsyncFunctionDef)
                        )
                        and declaration.name == "run_command"
                    ):
                        _set_python_alias(
                            aliases,
                            declaration.name,
                            PYTHON_REVIEWED_ADAPTER_CALL,
                        )
                        continue
                    process_bases = (
                        {
                            _dotted_name(base, aliases)
                            for base in declaration.bases
                            if _dotted_name(base, aliases)
                            in (PYTHON_PROCESS_CALLS | set(PYTHON_PROCESS_FACTORIES))
                        }
                        if isinstance(declaration, ast.ClassDef)
                        else set()
                    )
                    if len(process_bases) > 1:
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: multiple process-launching class bases are unsupported"
                        )
                    if process_bases:
                        _set_python_alias(
                            aliases, declaration.name, process_bases.pop()
                        )
                    elif isinstance(declaration, ast.ClassDef) and any(
                        contains_process_value(base) for base in declaration.bases
                    ):
                        raise InventoryError(
                            f"{rel}:{declaration.lineno}: process-launching class bases must be statically resolvable"
                        )
                    else:
                        _shadow_python_alias(aliases, declaration.name)
            return aliases

        def apply_reviewed_adapter_exec_overlay(
            scope: ast.AST,
            aliases: dict[str, str],
            position: tuple[int, int] | None,
        ) -> dict[str, str]:
            if (
                scope is tree
                and path == reviewed_adapter_consumer_path
                and reviewed_adapter_loader_names is not None
                and reviewed_adapter_exec_position is not None
                and (position is None or position > reviewed_adapter_exec_position)
            ):
                module_name = reviewed_adapter_loader_names[1]
                _shadow_python_alias(aliases, module_name)
                _set_python_alias(
                    aliases,
                    f"{module_name}.run_command",
                    PYTHON_REVIEWED_ADAPTER_CALL,
                )
            return aliases

        def apply_parameter_bindings(
            scope: ast.AST, aliases: dict[str, str]
        ) -> dict[str, str]:
            if isinstance(scope, (ast.ClassDef, *AST_TYPE_ALIAS_TYPES)):
                for type_parameter in getattr(scope, "type_params", ()):
                    _shadow_python_alias(aliases, type_parameter.name)
                return aliases
            if not isinstance(
                scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                return aliases

            positional = [*scope.args.posonlyargs, *scope.args.args]
            parameters = [*positional, *scope.args.kwonlyargs]
            if scope.args.vararg is not None:
                parameters.append(scope.args.vararg)
            if scope.args.kwarg is not None:
                parameters.append(scope.args.kwarg)
            for parameter in parameters:
                _shadow_python_alias(aliases, parameter.arg)
            for type_parameter in getattr(scope, "type_params", ()):
                _shadow_python_alias(aliases, type_parameter.name)

            definition_scope = lexical_scope(scope)
            definition_aliases = (
                aliases_at(definition_scope, (scope.lineno, scope.col_offset))
                if definition_scope is not scope
                else {}
            )
            defaults = list(
                zip(positional[-len(scope.args.defaults) :], scope.args.defaults)
            )
            defaults.extend(
                (argument, default)
                for argument, default in zip(
                    scope.args.kwonlyargs, scope.args.kw_defaults
                )
                if default is not None
            )
            for parameter, default in defaults:
                resolved = _dotted_name(default, definition_aliases)
                if resolved in PYTHON_TRACKED_ALIAS_VALUES:
                    _set_python_alias(aliases, parameter.arg, resolved)
                    continue
                transports_process_alias = False
                for candidate in ast.walk(default):
                    if isinstance(candidate, ast.Call):
                        if (
                            _dotted_name(candidate, definition_aliases)
                            in PYTHON_PROCESS_FACTORY_RESULTS
                        ):
                            transports_process_alias = True
                            break
                        continue
                    if not isinstance(candidate, (ast.Name, ast.Attribute)):
                        continue
                    candidate_resolved = _dotted_name(candidate, definition_aliases)
                    if candidate_resolved not in PYTHON_TRACKED_ALIAS_VALUES:
                        continue
                    parent = parents.get(candidate)
                    if isinstance(parent, ast.Attribute) and parent.value is candidate:
                        continue
                    if (
                        isinstance(parent, ast.Call)
                        and parent.func is candidate
                        and candidate_resolved in PYTHON_PROCESS_CALLS
                    ):
                        continue
                    transports_process_alias = True
                    break
                if transports_process_alias:
                    raise InventoryError(
                        f"{rel}:{getattr(scope, 'lineno', 1)}: structured process aliases in parameter defaults are unsupported"
                    )
            return aliases

        def apply_static_local_bindings(
            scope: ast.AST, aliases: dict[str, str]
        ) -> dict[str, str]:
            for name in scope_local_names.get(scope, set()):
                _shadow_python_alias(aliases, name)
            return aliases

        missing_alias_value = object()

        class AliasTrackingDict(dict[str, str]):
            """Record one declaration's mutations without copying the full map."""

            def __init__(self, values: dict[str, str]) -> None:
                super().__init__(values)
                self.original: dict[str, object] | None = None

            def begin(self) -> None:
                assert self.original is None
                self.original = {}

            def remember(self, key: str) -> None:
                if self.original is None or key in self.original:
                    return
                self.original[key] = self.get(key, missing_alias_value)

            def __setitem__(self, key: str, value: str) -> None:
                self.remember(key)
                super().__setitem__(key, value)

            def pop(self, key: str, *default: str) -> str:
                self.remember(key)
                return super().pop(key, *default)

            def finish(self) -> dict[str, str | None]:
                assert self.original is not None
                delta: dict[str, str | None] = {}
                for key, original in self.original.items():
                    if key in self:
                        current = dict.__getitem__(self, key)
                        if original is missing_alias_value or current != original:
                            delta[key] = current
                    elif original is not missing_alias_value:
                        delta[key] = None
                self.original = None
                return delta

            def abort(self) -> None:
                assert self.original is not None
                for key, original in self.original.items():
                    if original is missing_alias_value:
                        if key in self:
                            dict.__delitem__(self, key)
                    else:
                        assert isinstance(original, str)
                        dict.__setitem__(self, key, original)
                self.original = None

        class AliasTimeline:
            def __init__(self, scope: ast.AST, base: dict[str, str]) -> None:
                self.scope = scope
                self.declarations = tuple(scope_declarations.get(scope, ()))
                self.positions = tuple(
                    (declaration.lineno, declaration.col_offset)
                    for declaration in self.declarations
                )
                self.aliases = AliasTrackingDict(base)
                self.advanced = 0
                self.deltas: list[dict[str, str | None]] = []
                self.snapshots: dict[int, dict[str, str]] = {0: dict(base)}
                self.snapshot_cutoffs = [0]

            def advance(self, cutoff: int) -> None:
                while self.advanced < cutoff:
                    declaration = self.declarations[self.advanced]
                    self.aliases.begin()
                    try:
                        replay_declarations(self.scope, self.aliases, (declaration,))
                    except Exception:
                        self.aliases.abort()
                        raise
                    self.deltas.append(self.aliases.finish())
                    self.advanced += 1

            def snapshot(self, cutoff: int) -> dict[str, str]:
                cached = self.snapshots.get(cutoff)
                if cached is not None:
                    return dict(cached)
                self.advance(cutoff)
                if cutoff == self.advanced:
                    result = dict(self.aliases)
                else:
                    snapshot_index = (
                        bisect.bisect_right(self.snapshot_cutoffs, cutoff) - 1
                    )
                    start = self.snapshot_cutoffs[snapshot_index]
                    result = dict(self.snapshots[start])
                    for delta in self.deltas[start:cutoff]:
                        for name, resolved in delta.items():
                            if resolved is None:
                                result.pop(name, None)
                            else:
                                result[name] = resolved
                self.snapshots[cutoff] = dict(result)
                bisect.insort(self.snapshot_cutoffs, cutoff)
                return result

        alias_timeline_cache: dict[ast.AST, AliasTimeline] = {}
        full_alias_cache: dict[ast.AST, dict[str, str]] = {}

        def alias_timeline(scope: ast.AST) -> AliasTimeline:
            cached = alias_timeline_cache.get(scope)
            if cached is not None:
                return cached
            parent_scope = scope_parents.get(scope)
            aliases = aliases_full(parent_scope) if parent_scope is not None else {}
            apply_static_local_bindings(scope, aliases)
            apply_parameter_bindings(scope, aliases)
            result = AliasTimeline(scope, aliases)
            alias_timeline_cache[scope] = result
            return result

        def aliases_full(scope: ast.AST) -> dict[str, str]:
            cached = full_alias_cache.get(scope)
            if cached is not None:
                return dict(cached)
            timeline = alias_timeline(scope)
            result = timeline.snapshot(len(timeline.declarations))
            apply_reviewed_adapter_exec_overlay(scope, result, None)
            full_alias_cache[scope] = dict(result)
            return result

        def aliases_at(scope: ast.AST, position: tuple[int, int]) -> dict[str, str]:
            timeline = alias_timeline(scope)
            cutoff = bisect.bisect_left(timeline.positions, position)
            result = timeline.snapshot(cutoff)
            apply_reviewed_adapter_exec_overlay(scope, result, position)
            return result

        generic_annotation_owners: dict[ast.AST, ast.AST] = {}
        for generic_scope in ast.walk(tree):
            if not isinstance(
                generic_scope, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) or not getattr(generic_scope, "type_params", ()):
                continue
            annotation_roots = [
                argument.annotation
                for argument in (
                    *generic_scope.args.posonlyargs,
                    *generic_scope.args.args,
                    *generic_scope.args.kwonlyargs,
                )
                if argument.annotation is not None
            ]
            if generic_scope.args.vararg is not None and (
                generic_scope.args.vararg.annotation is not None
            ):
                annotation_roots.append(generic_scope.args.vararg.annotation)
            if generic_scope.args.kwarg is not None and (
                generic_scope.args.kwarg.annotation is not None
            ):
                annotation_roots.append(generic_scope.args.kwarg.annotation)
            if generic_scope.returns is not None:
                annotation_roots.append(generic_scope.returns)
            for annotation_root in annotation_roots:
                for candidate in ast.walk(annotation_root):
                    generic_annotation_owners[candidate] = generic_scope

        def aliases_for_evaluation(
            node: ast.AST, position: tuple[int, int]
        ) -> dict[str, str]:
            annotation_owner = generic_annotation_owners.get(node)
            if annotation_owner is None:
                aliases = aliases_at(lexical_scope(node), position)
                for name in _active_comprehension_bindings(node, parents):
                    _shadow_python_alias(aliases, name)
                return aliases
            parent_scope = scope_parents.get(annotation_owner)
            aliases = aliases_full(parent_scope) if parent_scope is not None else {}
            for type_parameter in getattr(annotation_owner, "type_params", ()):
                _shadow_python_alias(aliases, type_parameter.name)
            return aliases

        possible_process_roots = {
            *PYTHON_PROCESS_MODULES,
            *PYTHON_IMPORT_NAMESPACES,
            "__import__",
            "globals",
            "locals",
            "vars",
            "eval",
            "exec",
            "__builtins__",
            *(name.partition(".")[0] for name in module_process_exports.get(path, {})),
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    imported_root = imported.name.partition(".")[0]
                    if imported_root in PYTHON_PROCESS_MODULES:
                        possible_process_roots.add(imported.asname or imported_root)
                    elif imported_root in PYTHON_IMPORT_NAMESPACES:
                        possible_process_roots.add(imported.asname or imported_root)
                    else:
                        source = resolve_local_module(path, imported.name)
                        if source is not None and module_process_exports.get(source):
                            possible_process_roots.add(
                                imported.asname or imported.name.partition(".")[0]
                            )
            elif isinstance(node, ast.ImportFrom):
                if node.module in PYTHON_PROCESS_MODULES:
                    possible_process_roots.update(
                        imported.asname or imported.name
                        for imported in node.names
                        if imported.name != "*"
                    )
                elif node.module == "importlib":
                    possible_process_roots.update(
                        imported.asname or imported.name
                        for imported in node.names
                        if imported.name in {"import_module", "__import__"}
                    )
                elif node.module == "builtins":
                    possible_process_roots.update(
                        imported.asname or imported.name
                        for imported in node.names
                        if imported.name
                        in {
                            "__import__",
                            "__dict__",
                            "eval",
                            "exec",
                            "globals",
                            "locals",
                            "vars",
                        }
                    )
                elif node.module == "sys":
                    possible_process_roots.update(
                        imported.asname or imported.name
                        for imported in node.names
                        if imported.name == "modules"
                    )
                else:
                    source = resolve_local_module(path, node.module, node.level)
                    exports = (
                        module_process_exports.get(source, {})
                        if source is not None
                        else {}
                    )
                    for imported in node.names:
                        if imported.name == "*" or imported.name in exports:
                            possible_process_roots.add(imported.asname or imported.name)
                            continue
                        submodule = resolve_imported_submodule(
                            path, node.module, node.level, imported.name
                        )
                        if submodule is not None and module_process_exports.get(
                            submodule
                        ):
                            possible_process_roots.add(imported.asname or imported.name)

        possible_alias_assignments: list[tuple[set[str], ast.expr]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                names = {
                    name for target in node.targets for name in bound_names(target)
                }
                possible_alias_assignments.append((names, node.value))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                possible_alias_assignments.append(
                    (bound_names(node.target), node.value)
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                positional = [*node.args.posonlyargs, *node.args.args]
                possible_alias_assignments.extend(
                    ({parameter.arg}, default)
                    for parameter, default in zip(
                        positional[-len(node.args.defaults) :], node.args.defaults
                    )
                )
                possible_alias_assignments.extend(
                    ({parameter.arg}, default)
                    for parameter, default in zip(
                        node.args.kwonlyargs, node.args.kw_defaults
                    )
                    if default is not None
                )
        changed = True
        while changed:
            changed = False
            for names, value in possible_alias_assignments:
                if names.issubset(possible_process_roots):
                    continue
                if any(
                    isinstance(node, ast.Name) and node.id in possible_process_roots
                    for node in ast.walk(value)
                ):
                    previous_size = len(possible_process_roots)
                    possible_process_roots.update(names)
                    changed = changed or len(possible_process_roots) != previous_size

        statically_tracked_bindings_cache: dict[ast.AST, frozenset[str]] = {}

        def statically_tracked_bindings(scope: ast.AST) -> frozenset[str]:
            cached = statically_tracked_bindings_cache.get(scope)
            if cached is not None:
                return cached
            tracked: set[str] = set()
            for declaration in scope_declarations.get(scope, []):
                if isinstance(declaration, ast.Import):
                    for imported in declaration.names:
                        if imported.name.partition(".")[0] in (
                            PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES
                        ):
                            tracked.add(
                                imported.asname or imported.name.partition(".")[0]
                            )
                elif isinstance(declaration, ast.ImportFrom):
                    if declaration.module in (
                        PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES | {"builtins"}
                    ):
                        tracked.update(
                            imported.asname or imported.name
                            for imported in declaration.names
                            if imported.name != "*"
                        )
                elif isinstance(declaration, (ast.Assign, ast.AnnAssign)):
                    value = declaration.value
                    if value is None:
                        continue
                    before = aliases_at(
                        scope, (declaration.lineno, declaration.col_offset)
                    )
                    dynamic_import, imported_module = _dynamic_process_module(
                        value, before
                    )
                    resolved = _dotted_name(value, before)
                    if (
                        resolved in PYTHON_TRACKED_ALIAS_VALUES
                        or dynamic_import
                        and imported_module in PYTHON_PROCESS_MODULES
                    ):
                        targets = (
                            declaration.targets
                            if isinstance(declaration, ast.Assign)
                            else [declaration.target]
                        )
                        for target in targets:
                            tracked.update(bound_names(target))
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                definition_scope = lexical_scope(scope)
                before = aliases_at(definition_scope, (scope.lineno, scope.col_offset))
                positional = [*scope.args.posonlyargs, *scope.args.args]
                defaults = list(
                    zip(positional[-len(scope.args.defaults) :], scope.args.defaults)
                )
                defaults.extend(
                    (argument, default)
                    for argument, default in zip(
                        scope.args.kwonlyargs, scope.args.kw_defaults
                    )
                    if default is not None
                )
                tracked.update(
                    argument.arg
                    for argument, default in defaults
                    if _dotted_name(default, before) in PYTHON_TRACKED_ALIAS_VALUES
                )
            result = frozenset(tracked)
            statically_tracked_bindings_cache[scope] = result
            return result

        tracked_binding_names = set(PYTHON_PROCESS_MODULES) | {
            *PYTHON_IMPORT_NAMESPACES,
            *PYTHON_NAMESPACE_PRODUCERS,
            *PYTHON_DYNAMIC_CODE_CALLS,
            "__import__",
        }
        tracked_binding_names.update(
            name for scope in scopes for name in statically_tracked_bindings(scope)
        )

        for declaration in ast.walk(tree):
            if isinstance(declaration, (ast.Global, ast.Nonlocal)) and any(
                name in tracked_binding_names for name in declaration.names
            ):
                raise InventoryError(
                    f"{rel}:{declaration.lineno}: global and nonlocal process aliases are unsupported"
                )

        for scope in scopes:
            parent_scope = scope_parents.get(scope)
            if not isinstance(
                scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ) or not isinstance(
                parent_scope,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module),
            ):
                continue
            referenced_names = {
                candidate.id
                for candidate in ast.walk(scope)
                if isinstance(candidate, ast.Name)
                and isinstance(candidate.ctx, ast.Load)
                and lexical_scope(candidate) is scope
            }
            ambiguous = {
                name
                for name in referenced_names & statically_tracked_bindings(parent_scope)
                if scope_binding_counts.get(parent_scope, Counter()).get(name, 0) > 1
                and aliases_at(
                    parent_scope,
                    (getattr(scope, "lineno", 1), getattr(scope, "col_offset", 0)),
                ).get(name)
                in PYTHON_TRACKED_ALIAS_VALUES
            }
            if ambiguous:
                reviewed_loader_scope = (
                    reviewed_loader_contract
                    and ambiguous == {"importlib"}
                    and isinstance(scope, ast.FunctionDef)
                    and (
                        scope.name in REVIEWED_TEST_INVENTORY_LOADER_FUNCTION_NAMES
                        and isinstance(parents.get(scope), ast.Module)
                        or scope.name == "__init__"
                        and isinstance(parents.get(scope), ast.ClassDef)
                        and parents[scope].name == "_PythonExecutionClosure"
                    )
                )
                if reviewed_loader_scope:
                    continue
                names = ", ".join(sorted(ambiguous))
                raise InventoryError(
                    f"{rel}:{getattr(scope, 'lineno', 1)}: process aliases with multiple enclosing bindings are unsupported: {names}"
                )

        for class_scope in class_scopes:
            declarations = scope_declarations[class_scope]
            imports_process_module = any(
                (
                    isinstance(declaration, ast.Import)
                    and any(
                        imported.name in PYTHON_PROCESS_MODULES
                        for imported in declaration.names
                    )
                )
                or (
                    isinstance(declaration, ast.ImportFrom)
                    and declaration.module in PYTHON_PROCESS_MODULES
                )
                for declaration in declarations
            )
            references_process_alias = any(
                isinstance(node, ast.Name) and node.id in possible_process_roots
                for declaration in declarations
                for node in ast.walk(declaration)
            )
            if imports_process_module or references_process_alias:
                aliases_full(class_scope)

        def contains_process_value(value: ast.expr, aliases: dict[str, str]) -> bool:
            if isinstance(value, (ast.Name, ast.Attribute)):
                resolved_value = _dotted_name(value, aliases)
                if (
                    resolved_value in PYTHON_TRACKED_ALIAS_VALUES
                    and resolved_value != "sys"
                ):
                    return True
                if isinstance(value, ast.Attribute):
                    base = _normalize_python_process_callable(
                        _dotted_name(value.value, aliases)
                    )
                    return base in (
                        PYTHON_PROCESS_CALLS
                        | PYTHON_PROCESS_FACTORY_RESULTS
                        | PYTHON_IMPORT_HELPERS
                    ) or (
                        value.attr == "__dict__"
                        and base in (PYTHON_PROCESS_MODULES | PYTHON_IMPORT_NAMESPACES)
                    )
                return False
            if isinstance(value, ast.Call):
                dynamic_import, imported_module = _dynamic_process_module(
                    value, aliases
                )
                if dynamic_import and imported_module in PYTHON_PROCESS_MODULES:
                    return True
                called = _normalize_python_process_callable(
                    _dotted_name(value.func, aliases)
                )
                values = [*value.args, *(keyword.value for keyword in value.keywords)]
                if (
                    called
                    in {
                        "vars",
                        "builtins.vars",
                        "__builtins__.vars",
                    }
                    and len(value.args) == 1
                    and not value.keywords
                ):
                    return contains_process_value(value.args[0], aliases)
                if called in (PYTHON_PROCESS_CALLS | {PYTHON_REVIEWED_ADAPTER_CALL}):
                    return any(contains_process_value(item, aliases) for item in values)
            return any(
                contains_process_value(child, aliases)
                for child in ast.iter_child_nodes(value)
                if isinstance(child, ast.expr)
            )

        def might_contain_process_value(
            value: ast.expr, aliases: dict[str, str]
        ) -> bool:
            roots = {
                name
                for name, resolved in aliases.items()
                if resolved in PYTHON_TRACKED_ALIAS_VALUES
            }
            roots.update(PYTHON_PROCESS_MODULES)
            roots.update(PYTHON_NAMESPACE_PRODUCERS)
            roots.update(PYTHON_DYNAMIC_CODE_CALLS)
            roots.add("__builtins__")
            return any(
                isinstance(node, ast.Name) and node.id in roots
                for node in ast.walk(value)
            )

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            aliases = aliases_at(lexical_scope(node), (node.lineno, node.col_offset))
            process_bases = {
                _dotted_name(base, aliases)
                for base in node.bases
                if _dotted_name(base, aliases)
                in (PYTHON_PROCESS_CALLS | set(PYTHON_PROCESS_FACTORIES))
            }
            if len(process_bases) > 1:
                raise InventoryError(
                    f"{rel}:{node.lineno}: multiple process-launching class bases are unsupported"
                )
            if not process_bases and any(
                might_contain_process_value(base, aliases)
                and contains_process_value(base, aliases)
                for base in node.bases
            ):
                raise InventoryError(
                    f"{rel}:{node.lineno}: process-launching class bases must be statically resolvable"
                )

        def match_bound_names(pattern: ast.pattern) -> set[str]:
            names: set[str] = set()
            for node in ast.walk(pattern):
                if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                    names.add(node.name)
                elif isinstance(node, ast.MatchMapping) and node.rest:
                    names.add(node.rest)
            return names

        for node in ast.walk(tree):
            if not isinstance(node, ast.Match):
                continue
            aliases = aliases_at(
                lexical_scope(node.subject),
                (node.subject.lineno, node.subject.col_offset),
            )
            captured_names = {
                name for case in node.cases for name in match_bound_names(case.pattern)
            }
            shadows_process_alias = any(
                aliases.get(name) in PYTHON_TRACKED_ALIAS_VALUES
                for name in captured_names
            )
            captures_process_value = bool(captured_names) and (
                might_contain_process_value(node.subject, aliases)
                and contains_process_value(node.subject, aliases)
            )
            if shadows_process_alias or captures_process_value:
                raise InventoryError(
                    f"{rel}:{node.lineno}: process aliases in match capture bindings are unsupported"
                )

        unsupported_bindings: tuple[type[ast.AST], ...] = (
            ast.NamedExpr,
            ast.For,
            ast.AsyncFor,
            ast.With,
            ast.AsyncWith,
            ast.comprehension,
        )
        for node in ast.walk(tree):
            if not isinstance(node, unsupported_bindings):
                continue
            if isinstance(node, ast.NamedExpr):
                values = [node.value]
                targets = [node.target]
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                values = [node.iter]
                targets = [node.target]
            else:
                values = [item.context_expr for item in node.items]
                targets = [
                    item.optional_vars
                    for item in node.items
                    if item.optional_vars is not None
                ]
            position_node = node.iter if isinstance(node, ast.comprehension) else node
            position = (position_node.lineno, position_node.col_offset)
            aliases = aliases_at(lexical_scope(position_node), position)
            shadows_process_alias = not isinstance(node, ast.comprehension) and any(
                aliases.get(name) in PYTHON_TRACKED_ALIAS_VALUES
                for target in targets
                for name in bound_names(target)
            )
            binds_process_value = any(
                might_contain_process_value(value, aliases)
                and contains_process_value(value, aliases)
                for value in values
            )
            if shadows_process_alias or binds_process_value:
                raise InventoryError(
                    f"{rel}:{position_node.lineno}: process aliases in {type(node).__name__} bindings are unsupported"
                )

        for node in ast.walk(tree):
            if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Lambda)):
                value = node.body if isinstance(node, ast.Lambda) else node.value
                if value is None:
                    continue
                aliases = aliases_at(
                    lexical_scope(node), (node.lineno, node.col_offset)
                )
                if might_contain_process_value(
                    value, aliases
                ) and contains_process_value(value, aliases):
                    raise InventoryError(
                        f"{rel}:{node.lineno}: returning or yielding a process alias is unsupported"
                    )
        counts: Counter[tuple[str, str]] = Counter()
        carrier_counts: Counter[str] = Counter()
        assignment_cache: dict[Any, Any] = {}

        evaluation_position_cache: dict[ast.AST, tuple[int, int]] = {}

        def evaluation_position(node: ast.AST) -> tuple[int, int]:
            cached = evaluation_position_cache.get(node)
            if cached is not None:
                return cached
            current = node
            while current in parents:
                parent = parents[current]
                if (
                    isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                    and current is parent.value
                ) or (isinstance(parent, ast.NamedExpr) and current is parent.value):
                    result = (parent.lineno, parent.col_offset)
                    evaluation_position_cache[node] = result
                    return result
                current = parent
            result = (node.lineno, node.col_offset)
            evaluation_position_cache[node] = result
            return result

        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript) or not isinstance(
                node.ctx, ast.Load
            ):
                continue
            aliases = aliases_for_evaluation(node, evaluation_position(node))
            if _dotted_name(node.value, aliases) not in {
                PYTHON_GLOBALS_NAMESPACE,
                PYTHON_SYS_MODULES_NAMESPACE,
            }:
                continue
            if not (
                isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                if reviewed_process_contract_node(node):
                    continue
                raise InventoryError(
                    f"{rel}:{node.lineno}: dynamic process namespace lookups are unsupported"
                )

        calls = sorted(
            (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
            key=lambda node: (node.lineno, node.col_offset),
        )
        for node in calls:
            aliases = aliases_for_evaluation(node, evaluation_position(node))
            dynamic_import, imported_module = _dynamic_process_module(node, aliases)
            if dynamic_import and imported_module is None:
                if reviewed_process_contract_node(node):
                    continue
                raise InventoryError(
                    f"{rel}:{node.lineno}: dynamic module imports require a literal module name"
                )
            if dynamic_import:
                if (
                    imported_module in PYTHON_PROCESS_MODULES
                    and resolve_local_module(path, imported_module) is not None
                ):
                    raise InventoryError(
                        f"{rel}:{node.lineno}: local modules shadowing dynamically imported process modules are unsupported"
                    )
                if imported_module not in PYTHON_PROCESS_MODULES:
                    if reviewed_process_contract_node(node):
                        continue
                    raise InventoryError(
                        f"{rel}:{node.lineno}: dynamic imports of non-process modules are unsupported"
                    )
                continue
            dotted = _normalize_python_process_callable(
                _dotted_name(node.func, aliases)
            )
            if dotted in PYTHON_DYNAMIC_CODE_CALLS:
                if (
                    _is_reviewed_test_inventory_source_exec(
                        node,
                        rel,
                        tree,
                        parents,
                        REVIEWED_TEST_INVENTORY_LOADER_CONTRACT_SHA256
                        if reviewed_loader_contract
                        else None,
                    )
                    or _is_reviewed_test_inventory_frozen_exec_regression(
                        node, rel, parents
                    )
                    or reviewed_process_contract_node(node)
                    or (
                        reviewed_process_contract_node(node)
                        and isinstance(_python_scope(node, parents), ast.FunctionDef)
                        and _python_scope(node, parents).name
                        == "_registered_python_tooling_modules"
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "exec"
                        and len(node.args) == 2
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "code"
                        and isinstance(node.args[1], ast.Attribute)
                        and isinstance(node.args[1].value, ast.Name)
                        and node.args[1].value.id == "binding"
                        and node.args[1].attr == "namespace"
                        and not node.keywords
                    )
                ):
                    continue
                raise InventoryError(
                    f"{rel}:{node.lineno}: dynamic Python code execution is unsupported"
                )
            if _is_python_tracked_namespace_mutation(dotted):
                if reviewed_frozen_import_module_call(
                    node
                ) or reviewed_process_contract_node(node):
                    continue
                raise InventoryError(
                    f"{rel}:{node.lineno}: mutating tracked process namespaces is unsupported"
                )
            if _is_python_unsupported_namespace_operation(dotted):
                if reviewed_process_contract_node(node):
                    continue
                raise InventoryError(
                    f"{rel}:{node.lineno}: unsupported tracked process namespace operation"
                )
            if (
                dotted in PYTHON_NAMESPACE_PRODUCERS
                and not node.args
                and not node.keywords
            ):
                continue
            if dotted in {"vars", "builtins.vars", "__builtins__.vars"}:
                if len(node.args) != 1 or node.keywords:
                    raise InventoryError(
                        f"{rel}:{node.lineno}: unsupported vars() call shape"
                    )
                if might_contain_process_value(
                    node.args[0], aliases
                ) and contains_process_value(node.args[0], aliases):
                    raise InventoryError(
                        f"{rel}:{node.lineno}: process alias escapes through vars()"
                    )
                continue
            if dotted in PYTHON_NAMESPACE_ACCESSORS:
                if dotted == f"{PYTHON_SYS_MODULES_NAMESPACE}.get" and (
                    _is_test_inventory_module_binding_verifier_call(node, rel, parents)
                ):
                    if _is_reviewed_test_inventory_module_binding_lookup(
                        node, rel, parents
                    ):
                        continue
                    raise InventoryError(
                        f"{rel}:{node.lineno}: unsupported test-inventory module binding lookup"
                    )
                valid_accessor = False
                if dotted.endswith(".__dict__.get"):
                    valid_accessor = len(node.args) in {1, 2}
                elif dotted.endswith(".__dict__.__getitem__"):
                    valid_accessor = len(node.args) == 1
                elif dotted.endswith(".__getattribute__"):
                    valid_accessor = len(node.args) == 1
                elif dotted == f"{PYTHON_GLOBALS_NAMESPACE}.get":
                    valid_accessor = len(node.args) in {1, 2}
                elif dotted == f"{PYTHON_GLOBALS_NAMESPACE}.__getitem__":
                    valid_accessor = len(node.args) == 1
                elif dotted in {
                    "__builtins__.get",
                    f"{PYTHON_SYS_MODULES_NAMESPACE}.get",
                }:
                    valid_accessor = len(node.args) in {1, 2}
                elif dotted == f"{PYTHON_SYS_MODULES_NAMESPACE}.__getitem__":
                    valid_accessor = len(node.args) == 1
                if (
                    node.keywords
                    or not valid_accessor
                    or not (
                        node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    )
                ):
                    if reviewed_process_contract_node(node):
                        continue
                    raise InventoryError(
                        f"{rel}:{node.lineno}: dynamic process namespace lookups are unsupported"
                    )
                continue
            if dotted in PYTHON_PROCESS_FACTORIES:
                continue
            adapter_digest = (
                reviewed_adapter_digest
                if dotted == PYTHON_REVIEWED_ADAPTER_CALL
                else None
            )
            parent_carrier_digest = (
                _embedded_origin["carrier_semantics_digests"][-1]
                if _embedded_origin is not None
                else None
            )
            carrier = _extract_python_carrier(
                node,
                dotted,
                aliases,
                parents,
                location=f"{rel}:{node.lineno}",
                adapter_definition_digest=adapter_digest,
                parent_carrier_digest=parent_carrier_digest,
                assignment_cache=assignment_cache,
                physical_file=_embedded_origin is None,
                reviewed_dynamic_physical=(
                    node in reviewed_dynamic_popen_calls and _embedded_origin is None
                ),
                reviewed_keyword_expansion=(
                    node in reviewed_capsule_process_conditions
                    and _embedded_origin is None
                ),
            )
            if carrier is not None or dotted in PYTHON_PROCESS_CALLS:
                scope = _python_scope(node, parents)
                function = (
                    scope.name
                    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else "module"
                )
                scope_token = _python_scope_token(node, parents)
            if carrier is not None:
                carrier_counts[scope_token] += 1
                carrier_ordinal = carrier_counts[scope_token]
                carrier_token = f"{scope_token.replace('/', '~')}:c{carrier_ordinal}"
                prior_path = (
                    tuple(_embedded_origin["carrier_path"])
                    if _embedded_origin is not None
                    else ()
                )
                carrier_path = (*prior_path, carrier_token)
                depth = len(carrier_path)
                physical_rel = (
                    _embedded_origin["file"] if _embedded_origin is not None else rel
                )
                physical_function = (
                    _embedded_origin["physical_function"]
                    if _embedded_origin is not None
                    else function
                )
                if depth > EMBEDDED_PYTHON_MAX_DEPTH:
                    raise InventoryError(
                        f"{physical_rel}: embedded Python -c depth limit exceeded"
                    )
                code_bytes = len(carrier.code.encode("utf-8"))
                if embedded_budget["bytes"] + code_bytes > EMBEDDED_PYTHON_MAX_BYTES:
                    raise InventoryError(
                        f"{physical_rel}: embedded Python -c cumulative byte limit exceeded"
                    )
                if embedded_budget["units"] >= EMBEDDED_PYTHON_MAX_UNITS:
                    raise InventoryError(
                        f"{physical_rel}: embedded Python -c unit limit exceeded"
                    )
                embedded_budget["bytes"] += code_bytes
                embedded_budget["units"] += 1
                carrier_semantics_digests = (
                    *(
                        tuple(_embedded_origin["carrier_semantics_digests"])
                        if _embedded_origin is not None
                        else ()
                    ),
                    carrier.semantics_digest,
                )
                carrier_label = ".".join(carrier_path)
                try:
                    embedded_tree = ast.parse(
                        carrier.code,
                        filename=f"{physical_rel}:embedded:{carrier_label}",
                    )
                    node_count = sum(1 for _ in ast.walk(embedded_tree))
                except SyntaxError as exc:
                    raise InventoryError(
                        f"{physical_rel}: embedded Python -c syntax error: {exc.msg}"
                    ) from exc
                except RecursionError as exc:
                    raise InventoryError(
                        f"{physical_rel}: embedded Python -c recursion limit exceeded"
                    ) from exc
                if embedded_budget["nodes"] + node_count > EMBEDDED_PYTHON_MAX_NODES:
                    raise InventoryError(
                        f"{physical_rel}: embedded Python -c cumulative node limit exceeded"
                    )
                embedded_budget["nodes"] += node_count
                virtual_digest = _json_fact_digest(
                    {
                        "file": physical_rel,
                        "carrier_path": carrier_path,
                        "code": carrier.code,
                    }
                )
                virtual_path = root / "__embedded__" / f"{virtual_digest}.py"
                next_origin = {
                    "file": physical_rel,
                    "physical_function": physical_function,
                    "carrier_path": carrier_path,
                    "carrier_semantics_digests": carrier_semantics_digests,
                    "carrier_ordinals": (
                        *(
                            tuple(_embedded_origin["carrier_ordinals"])
                            if _embedded_origin is not None
                            else ()
                        ),
                        carrier_ordinal,
                    ),
                    "carrier_shapes": (
                        *(
                            tuple(_embedded_origin["carrier_shapes"])
                            if _embedded_origin is not None
                            else ()
                        ),
                        carrier.shape,
                    ),
                    "adapter_definition_digest": (
                        carrier.adapter_definition_digest
                        or (
                            _embedded_origin.get("adapter_definition_digest")
                            if _embedded_origin is not None
                            else None
                        )
                    ),
                }
                try:
                    nested_trees = dict(trees)
                    nested_trees[virtual_path] = embedded_tree
                    result.extend(
                        _discover_python_launches(
                            root,
                            active,
                            _python_files_override=[virtual_path],
                            _trees_override=nested_trees,
                            _embedded_origin=next_origin,
                            _embedded_budget=embedded_budget,
                        )
                    )
                except RecursionError as exc:
                    raise InventoryError(
                        f"{physical_rel}: embedded Python -c recursion limit exceeded"
                    ) from exc
            if dotted == PYTHON_REVIEWED_ADAPTER_CALL:
                continue
            if dotted not in PYTHON_PROCESS_CALLS:
                escaped_values = [node.func]
                if dotted != "hasattr":
                    escaped_values.extend(node.args)
                    escaped_values.extend(keyword.value for keyword in node.keywords)
                if any(
                    might_contain_process_value(value, aliases)
                    and contains_process_value(value, aliases)
                    for value in escaped_values
                ):
                    if reviewed_process_contract_node(node):
                        continue
                    raise InventoryError(
                        f"{rel}:{node.lineno}: process alias escapes through an unsupported call expression"
                    )
                continue
            key = (scope_token if _embedded_origin is not None else function, dotted)
            counts[key] += 1
            ordinal = counts[key]
            if _embedded_origin is None:
                launch_id = f"python-launch:{rel}:{function}:{dotted}:{ordinal}"
                anchor = {
                    "file": rel,
                    "enclosing_function": function,
                    "symbol": dotted,
                    "ordinal": ordinal,
                }
                carrier_semantics_digests: tuple[str, ...] = ()
            else:
                physical_rel = _embedded_origin["file"]
                physical_function = _embedded_origin["physical_function"]
                safe_scope = scope_token.replace("/", "~")
                carrier_label = ".".join(_embedded_origin["carrier_path"])
                launch_id = (
                    f"python-launch:{physical_rel}:{physical_function}:{dotted}:"
                    f"embedded-{carrier_label}:scope-{safe_scope}:{ordinal}"
                )
                carrier_semantics_digests = tuple(
                    _embedded_origin["carrier_semantics_digests"]
                )
                anchor = {
                    "file": physical_rel,
                    "enclosing_function": physical_function,
                    "symbol": dotted,
                    "ordinal": ordinal,
                    "origin_kind": "embedded-python-c",
                    "embedded_depth": len(_embedded_origin["carrier_path"]),
                    "carrier_ordinal": _embedded_origin["carrier_ordinals"][-1],
                    "carrier_ordinals": list(_embedded_origin["carrier_ordinals"]),
                    "carrier_path": list(_embedded_origin["carrier_path"]),
                    "carrier_semantics_digests": list(carrier_semantics_digests),
                    "carrier_shapes": list(_embedded_origin["carrier_shapes"]),
                    "adapter_definition_digest": _embedded_origin[
                        "adapter_definition_digest"
                    ],
                    "embedded_scope": scope_token,
                    "embedded_function": function,
                }
            launch_row = {
                "id": launch_id,
                "category": "python-launch",
                "anchor": anchor,
                "call": dotted,
                "call_semantics_digest": _python_call_semantics_digest(
                    node,
                    dotted,
                    parents,
                    carrier_semantics_digests=carrier_semantics_digests,
                ),
            }
            reviewed_condition = reviewed_capsule_process_conditions.get(node)
            if reviewed_condition is not None:
                launch_row["condition"] = reviewed_condition
            if carrier is not None:
                launch_row["carrier_semantics_digest"] = carrier.semantics_digest
            result.append(launch_row)
    return sorted(result, key=lambda item: item["id"])


def _python_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    """Return the runtime namespace used to evaluate ``node``.

    Function/lambda defaults and annotations plus class bases/decorators execute in
    the enclosing namespace. Bodies execute in their newly created namespace.
    """
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.Module):
            return parent
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if current in parent.body:
                return parent
            current = parent
            continue
        if isinstance(parent, ast.Lambda):
            if current is parent.body:
                return parent
            current = parent
            continue
        if isinstance(parent, ast.ClassDef):
            if current in parent.body:
                return parent
            current = parent
            continue
        current = parent
    return current


def _carrier_bound_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {
            name for element in target.elts for name in _carrier_bound_names(element)
        }
    if isinstance(target, ast.Starred):
        return _carrier_bound_names(target.value)
    return set()


def _carrier_pattern_bound_names(pattern: ast.pattern) -> set[str]:
    names: set[str] = set()
    for candidate in ast.walk(pattern):
        if isinstance(candidate, (ast.MatchAs, ast.MatchStar)):
            if candidate.name:
                names.add(candidate.name)
        elif isinstance(candidate, ast.MatchMapping) and candidate.rest:
            names.add(candidate.rest)
    return names


def _active_comprehension_bindings(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> set[str]:
    """Return comprehension targets already bound while ``node`` executes."""
    active: set[str] = set()
    ancestors = {node, *_ancestor_chain(node, parents)}
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(
            parent, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            generators = parent.generators
            relevant = []
            for generator in generators:
                if generator.iter in ancestors:
                    break
                relevant.append(generator)
                if any(condition in ancestors for condition in generator.ifs):
                    break
            for generator in relevant:
                active.update(_carrier_bound_names(generator.target))
        child = parent
    return active


def _cached_active_comprehension_bindings(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    cache: dict[Any, Any],
) -> frozenset[str]:
    key = ("carrier-comprehension-bindings", id(node))
    cached = cache.get(key)
    if isinstance(cached, frozenset):
        return cached
    active = frozenset(_active_comprehension_bindings(node, parents))
    cache[key] = active
    return active


class _CarrierBindingEvent(NamedTuple):
    position: tuple[int, int]
    order: tuple[int, int, int]
    name: str
    value: ast.expr | None
    kind: str
    direct: bool


class _CarrierScopeSummary(NamedTuple):
    events: tuple[_CarrierBindingEvent, ...]
    events_by_name: dict[str, tuple[_CarrierBindingEvent, ...]]
    binding_names: frozenset[str]
    local_names: frozenset[str]
    parameter_names: frozenset[str]
    external_kills: frozenset[str]
    loaded_names: frozenset[str]
    canonical_path_imports: frozenset[tuple[str, str, tuple[int, int]]]


class _CarrierScopeIndex(NamedTuple):
    candidates: dict[int, tuple[ast.AST, ...]]
    child_scopes: dict[int, tuple[ast.AST, ...]]


class _CarrierFactState(NamedTuple):
    value: ast.expr | None
    active_import: bool = False


class _CarrierFactRequest(NamedTuple):
    scope: ast.AST
    name: str
    before: tuple[int, int, int]
    stable: bool


class _CarrierFactPlan(NamedTuple):
    key: tuple[Any, ...]
    summary: _CarrierScopeSummary
    event: _CarrierBindingEvent | None
    dependencies: tuple[_CarrierFactRequest, ...]
    immediate: _CarrierFactState | None


def _carrier_scope_parent(
    scope: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.AST | None:
    """Return the lexical value namespace, skipping class closure boundaries."""

    current = scope
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.ClassDef):
            continue
        if isinstance(
            current, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            return current
    return None


def _discover_level1_payload_source_bindings(
    root: Path, context: DiscoveryContext | None = None
) -> list[dict[str, str]]:
    """Bind the two Level 1 ``run_once`` call sites to their argv0 selectors."""

    relative = "bench/tools/run_level1_report.py"
    active = _context_for(root, context)
    tree = ast.parse(
        _frozen_regular_text(active, relative, "Level 1 payload controller"),
        filename=relative,
    )
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"run_level1_op", "run_copy_op"}
    }
    if set(functions) != {"run_level1_op", "run_copy_op"}:
        raise InventoryError(
            f"{relative}: Level 1 payload source functions are not exact"
        )

    bindings: list[dict[str, str]] = []
    frozen_execution_keywords = {
        "run_level1_op": {
            "expected_binary": "args.level1_probe if artifacts is not None else None",
            "expected_library": "private_library if artifacts is not None else None",
            "artifacts": "artifacts",
        },
        "run_copy_op": {
            "expected_binary": "args.copy_probe if artifacts is not None else None",
            "expected_library": "private_library if artifacts is not None else None",
            "artifacts": "artifacts",
        },
    }
    for function_name in ("run_level1_op", "run_copy_op"):
        function = functions[function_name]
        calls = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_once"
                and _python_scope(node, parents) is function
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        if len(calls) != 1 or len(calls[0].args) != 1:
            raise InventoryError(
                f"{relative}:{function.lineno}: {function_name} must have one exact frozen run_once(argv) call"
            )
        call = calls[0]
        keywords = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
        expected_keywords = frozen_execution_keywords[function_name]
        if len(keywords) != len(call.keywords) or set(keywords) != set(
            expected_keywords
        ):
            raise InventoryError(
                f"{relative}:{call.lineno}: Level 1 run_once frozen execution keywords are not exact"
            )
        for keyword, source in expected_keywords.items():
            expected = ast.parse(source, mode="eval").body
            if _canonical_python_ast(keywords[keyword]) != _canonical_python_ast(
                expected
            ):
                raise InventoryError(
                    f"{relative}:{call.lineno}: Level 1 run_once {keyword} binding is not exact"
                )
        argument = call.args[0]
        if not isinstance(argument, ast.Name):
            raise InventoryError(
                f"{relative}:{call.lineno}: Level 1 run_once argv must be a local name"
            )
        assignments = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == argument.id
            and (node.lineno, node.col_offset) < (call.lineno, call.col_offset)
            and _python_scope(node, parents) is function
        ]
        if len(assignments) != 1:
            raise InventoryError(
                f"{relative}:{call.lineno}: Level 1 run_once argv producer must be unique"
            )
        assignment = assignments[0]
        if (
            not isinstance(assignment.value, (ast.List, ast.Tuple))
            or not assignment.value.elts
        ):
            raise InventoryError(
                f"{relative}:{assignment.lineno}: Level 1 run_once argv0 must be explicit"
            )
        selector = _dotted_name(assignment.value.elts[0], {})
        if selector is None:
            raise InventoryError(
                f"{relative}:{assignment.lineno}: Level 1 payload selector is dynamic"
            )
        bindings.append(
            {
                "source_callsite": f"{function_name}:run_once:1",
                "source_selector": selector,
                "source_semantics_digest": _json_fact_digest(
                    {
                        "version": PYTHON_SEMANTICS_VERSION,
                        "function": _canonical_python_ast(function),
                        "argv_assignment": _canonical_python_ast(assignment),
                        "call": _canonical_python_ast(call),
                    }
                ),
            }
        )
    return bindings


def _carrier_scope_index(
    scope: ast.AST,
    parents: dict[ast.AST, ast.AST],
    cache: dict[Any, Any],
) -> _CarrierScopeIndex:
    """Partition one file AST into execution scopes with one linear DFS."""

    root = scope
    while root in parents:
        root = parents[root]
    key = ("carrier-scope-index", id(root))
    cached = cache.get(key)
    if isinstance(cached, _CarrierScopeIndex):
        return cached

    candidates: dict[int, list[ast.AST]] = {id(root): []}
    child_scopes: dict[int, list[ast.AST]] = {id(root): []}
    scope_nodes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

    def visit(node: ast.AST, active_scope: ast.AST, *, include: bool = True) -> None:
        if include:
            candidates.setdefault(id(active_scope), []).append(node)
        if include and isinstance(node, scope_nodes):
            candidates.setdefault(id(node), [])
            child_scopes.setdefault(id(node), [])
            child_scopes.setdefault(id(active_scope), []).append(node)
        body_ids = (
            {id(statement) for statement in node.body}
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            else frozenset()
        )
        for child in ast.iter_child_nodes(node):
            child_scope = active_scope
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if id(child) in body_ids:
                    child_scope = node
            elif isinstance(node, ast.Lambda) and child is node.body:
                child_scope = node
            visit(child, child_scope)

    visit(root, root, include=False)
    index = _CarrierScopeIndex(
        {identifier: tuple(nodes) for identifier, nodes in candidates.items()},
        {identifier: tuple(nodes) for identifier, nodes in child_scopes.items()},
    )
    cache[key] = index
    return index


def _carrier_scope_summary(
    scope: ast.AST,
    parents: dict[ast.AST, ast.AST],
    cache: dict[Any, Any],
) -> _CarrierScopeSummary:
    key = ("carrier-scope", id(scope))
    cached = cache.get(key)
    if isinstance(cached, _CarrierScopeSummary):
        return cached

    events: list[_CarrierBindingEvent] = []
    local_names: set[str] = set()
    parameter_names: set[str] = set()
    global_names: set[str] = set()
    nonlocal_names: set[str] = set()
    loaded_names: set[str] = set()
    nested_scopes: list[ast.AST] = []
    canonical_path_imports: set[tuple[str, str, tuple[int, int]]] = set()

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        parameters = [
            *scope.args.posonlyargs,
            *scope.args.args,
            *scope.args.kwonlyargs,
        ]
        if scope.args.vararg is not None:
            parameters.append(scope.args.vararg)
        if scope.args.kwarg is not None:
            parameters.append(scope.args.kwarg)
        parameter_names.update(parameter.arg for parameter in parameters)
        local_names.update(parameter_names)

    def record(
        candidate: ast.AST,
        names: set[str],
        *,
        value: ast.expr | None = None,
        kind: str = "kill",
        direct: bool = False,
    ) -> None:
        position = (
            getattr(candidate, "lineno", getattr(scope, "lineno", 1)),
            getattr(candidate, "col_offset", getattr(scope, "col_offset", 0)),
        )
        order_node: ast.AST = candidate
        if kind == "assign" and value is not None:
            order_node = value
        elif isinstance(candidate, (ast.For, ast.AsyncFor)):
            order_node = candidate.iter
        elif isinstance(candidate, (ast.With, ast.AsyncWith)) and candidate.items:
            order_node = candidate.items[-1].context_expr
        elif isinstance(candidate, ast.Match):
            order_node = candidate.subject
        elif isinstance(candidate, ast.NamedExpr):
            order_node = candidate.value
        order = (
            getattr(
                order_node, "end_lineno", getattr(order_node, "lineno", position[0])
            ),
            getattr(
                order_node,
                "end_col_offset",
                getattr(order_node, "col_offset", position[1]),
            ),
            1,
        )
        for name in names:
            events.append(
                _CarrierBindingEvent(position, order, name, value, kind, direct)
            )
            local_names.add(name)

    scope_index = _carrier_scope_index(scope, parents, cache)
    nested_scopes.extend(scope_index.child_scopes.get(id(scope), ()))
    for candidate in scope_index.candidates.get(id(scope), ()):
        direct = parents.get(candidate) is scope
        if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Load):
            loaded_names.add(candidate.id)
        elif isinstance(candidate, ast.Assign):
            for target in candidate.targets:
                names = _carrier_bound_names(target)
                record(
                    candidate,
                    names,
                    value=candidate.value,
                    kind="assign" if isinstance(target, ast.Name) else "kill",
                    direct=direct and isinstance(target, ast.Name),
                )
        elif isinstance(candidate, ast.AnnAssign):
            names = _carrier_bound_names(candidate.target)
            record(
                candidate,
                names,
                value=candidate.value,
                kind=(
                    "assign"
                    if isinstance(candidate.target, ast.Name)
                    and candidate.value is not None
                    else "kill"
                ),
                direct=direct
                and isinstance(candidate.target, ast.Name)
                and candidate.value is not None,
            )
        elif isinstance(candidate, (ast.AugAssign, ast.NamedExpr)):
            record(candidate, _carrier_bound_names(candidate.target))
        elif isinstance(candidate, ast.Delete):
            for target in candidate.targets:
                record(candidate, _carrier_bound_names(target))
        elif isinstance(candidate, ast.Import):
            position = (candidate.lineno, candidate.col_offset)
            canonical_path_imports.update(
                (imported.asname or "pathlib", "module", position)
                for imported in candidate.names
                if imported.name == "pathlib"
            )
            record(
                candidate,
                {
                    imported.asname or imported.name.partition(".")[0]
                    for imported in candidate.names
                },
                kind="import",
                direct=direct,
            )
        elif isinstance(candidate, ast.ImportFrom):
            position = (candidate.lineno, candidate.col_offset)
            if candidate.level == 0 and candidate.module == "pathlib":
                canonical_path_imports.update(
                    (imported.asname or imported.name, "from", position)
                    for imported in candidate.names
                    if imported.name == "Path"
                )
            record(
                candidate,
                {
                    imported.asname or imported.name
                    for imported in candidate.names
                    if imported.name != "*"
                },
                kind="import",
                direct=direct,
            )
        elif isinstance(
            candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            record(candidate, {candidate.name})
        elif isinstance(candidate, (ast.For, ast.AsyncFor)):
            record(candidate, _carrier_bound_names(candidate.target))
        elif isinstance(candidate, (ast.With, ast.AsyncWith)):
            for item in candidate.items:
                if item.optional_vars is not None:
                    record(candidate, _carrier_bound_names(item.optional_vars))
        elif isinstance(candidate, ast.ExceptHandler) and candidate.name:
            record(candidate, {candidate.name})
        elif isinstance(candidate, ast.Match):
            for case in candidate.cases:
                record(candidate, _carrier_pattern_bound_names(case.pattern))
        elif isinstance(candidate, ast.Global):
            global_names.update(candidate.names)
        elif isinstance(candidate, ast.Nonlocal):
            nonlocal_names.update(candidate.names)
        elif _is_ast_type_alias(candidate):
            record(candidate, _carrier_bound_names(candidate.name))

    local_names.difference_update(global_names | nonlocal_names)
    written_names = {
        event.name for event in events if event.name in global_names | nonlocal_names
    }
    for nested_scope in nested_scopes:
        written_names.update(
            _carrier_scope_summary(nested_scope, parents, cache).external_kills
        )
    ordered_events = tuple(
        sorted(events, key=lambda event: (event.order, event.name, event.kind))
    )
    events_by_name: dict[str, list[_CarrierBindingEvent]] = {}
    for event in ordered_events:
        events_by_name.setdefault(event.name, []).append(event)
    summary = _CarrierScopeSummary(
        ordered_events,
        {name: tuple(values) for name, values in events_by_name.items()},
        frozenset(events_by_name),
        frozenset(local_names),
        frozenset(parameter_names),
        frozenset(written_names),
        frozenset(loaded_names),
        frozenset(canonical_path_imports),
    )
    cache[key] = summary
    return summary


def _freeze_carrier_expression(
    value: ast.expr,
    facts: dict[str, ast.expr],
    all_binding_names: frozenset[str],
    active_imports: set[str],
) -> ast.expr:
    class Measure(NamedTuple):
        nodes: int
        depth: int
        items: int
        bytes: int

    measure_cache: dict[int, Measure] = {}
    measuring: set[int] = set()

    def literal_bytes(candidate: ast.AST) -> int:
        if not isinstance(candidate, ast.Constant):
            return 0
        if isinstance(candidate.value, str):
            if len(candidate.value) > CARRIER_FREEZE_MAX_BYTES:
                return CARRIER_FREEZE_MAX_BYTES + 1
            return len(candidate.value.encode("utf-8"))
        if isinstance(candidate.value, bytes):
            return len(candidate.value)
        return len(repr(candidate.value).encode("utf-8"))

    def direct_items(candidate: ast.AST) -> int:
        if isinstance(candidate, (ast.List, ast.Tuple, ast.Set)):
            return len(candidate.elts)
        if isinstance(candidate, ast.Dict):
            return len(candidate.keys)
        return 0

    def check(measure: Measure) -> None:
        if measure.nodes > CARRIER_FREEZE_MAX_NODES:
            raise InventoryError("frozen carrier expression exceeds node limit")
        if measure.depth > CARRIER_FREEZE_MAX_DEPTH:
            raise InventoryError("frozen carrier expression exceeds depth limit")
        if measure.items > CARRIER_FREEZE_MAX_ITEMS:
            raise InventoryError("frozen carrier expression exceeds item limit")
        if measure.bytes > CARRIER_FREEZE_MAX_BYTES:
            raise InventoryError("frozen carrier expression exceeds byte limit")

    def measure_tree(candidate: ast.AST) -> Measure:
        identity = id(candidate)
        cached = measure_cache.get(identity)
        if cached is not None:
            return cached
        if identity in measuring:
            raise InventoryError("frozen carrier expression is cyclic")
        measuring.add(identity)
        nodes = 1
        depth = 0
        items = direct_items(candidate)
        bytes_used = literal_bytes(candidate)
        try:
            for child in ast.iter_child_nodes(candidate):
                child_measure = measure_tree(child)
                nodes += child_measure.nodes
                items += child_measure.items
                bytes_used += child_measure.bytes
                child_depth = child_measure.depth + (
                    1 if isinstance(child, ast.expr) else 0
                )
                depth = max(depth, child_depth)
                check(Measure(nodes, depth, items, bytes_used))
        finally:
            measuring.remove(identity)
        result = Measure(nodes, depth, items, bytes_used)
        check(result)
        measure_cache[identity] = result
        return result

    admitted_nodes = 0
    admitted_depth = 0
    admitted_items = 0
    admitted_bytes = 0
    admitting: set[int] = set()

    def admit(candidate: ast.AST, depth: int) -> None:
        nonlocal admitted_nodes, admitted_depth, admitted_items, admitted_bytes
        identity = id(candidate)
        if identity in admitting:
            raise InventoryError("frozen carrier expression is cyclic")
        if (
            isinstance(candidate, ast.Name)
            and isinstance(candidate.ctx, ast.Load)
            and candidate.id in facts
        ):
            fact_measure = measure_tree(facts[candidate.id])
            admitted_nodes += fact_measure.nodes
            admitted_depth = max(admitted_depth, depth + fact_measure.depth)
            admitted_items += fact_measure.items
            admitted_bytes += fact_measure.bytes
            check(
                Measure(
                    admitted_nodes,
                    admitted_depth,
                    admitted_items,
                    admitted_bytes,
                )
            )
            return
        if (
            isinstance(candidate, ast.Name)
            and isinstance(candidate.ctx, ast.Load)
            and candidate.id in all_binding_names
            and candidate.id not in active_imports
        ):
            admitted_nodes += 1
            admitted_depth = max(admitted_depth, depth)
            check(
                Measure(
                    admitted_nodes,
                    admitted_depth,
                    admitted_items,
                    admitted_bytes,
                )
            )
            return
        admitting.add(identity)
        admitted_nodes += 1
        admitted_depth = max(admitted_depth, depth)
        admitted_items += direct_items(candidate)
        admitted_bytes += literal_bytes(candidate)
        check(
            Measure(
                admitted_nodes,
                admitted_depth,
                admitted_items,
                admitted_bytes,
            )
        )
        try:
            for child in ast.iter_child_nodes(candidate):
                admit(child, depth + (1 if isinstance(child, ast.expr) else 0))
        finally:
            admitting.remove(identity)

    admit(value, 0)

    class Freeze(ast.NodeTransformer):
        def visit_Name(self, candidate: ast.Name) -> ast.AST:
            if not isinstance(candidate.ctx, ast.Load):
                return candidate
            if candidate.id in facts:
                return copy.deepcopy(facts[candidate.id])
            if candidate.id in all_binding_names and candidate.id not in active_imports:
                return ast.copy_location(ast.Constant(value=None), candidate)
            return candidate

    return ast.fix_missing_locations(Freeze().visit(copy.deepcopy(value)))


def _carrier_loaded_names_bounded(value: ast.expr) -> tuple[str, ...]:
    """Collect RHS dependencies without recursive walking or unbounded graph work."""

    names: set[str] = set()
    visiting: set[int] = set()
    finished: set[int] = set()
    nodes = 0
    stack: list[tuple[ast.AST, int, bool]] = [(value, 0, False)]
    while stack:
        candidate, depth, exiting = stack.pop()
        identity = id(candidate)
        if exiting:
            visiting.remove(identity)
            finished.add(identity)
            continue
        if identity in finished:
            continue
        if identity in visiting:
            raise InventoryError("carrier assignment dependency graph is cyclic")
        nodes += 1
        if nodes > CARRIER_FREEZE_MAX_NODES:
            raise InventoryError("frozen carrier expression exceeds node limit")
        if depth > CARRIER_FREEZE_MAX_DEPTH:
            raise InventoryError("frozen carrier expression exceeds depth limit")
        if (
            isinstance(candidate, (ast.List, ast.Tuple, ast.Set))
            and len(candidate.elts) > CARRIER_FREEZE_MAX_ITEMS
        ):
            raise InventoryError("frozen carrier expression exceeds item limit")
        if (
            isinstance(candidate, ast.Dict)
            and len(candidate.keys) > CARRIER_FREEZE_MAX_ITEMS
        ):
            raise InventoryError("frozen carrier expression exceeds item limit")
        if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Load):
            names.add(candidate.id)
        visiting.add(identity)
        stack.append((candidate, depth, True))
        children = tuple(ast.iter_child_nodes(candidate))
        for child in reversed(children):
            stack.append(
                (child, depth + (1 if isinstance(child, ast.expr) else 0), False)
            )
    return tuple(sorted(names))


def _carrier_fact_plan(
    request: _CarrierFactRequest,
    parents: dict[ast.AST, ast.AST],
    cache: dict[Any, Any],
) -> _CarrierFactPlan:
    scope, name, before, stable = request
    summary = _carrier_scope_summary(scope, parents, cache)
    events = summary.events_by_name.get(name, ())
    event: _CarrierBindingEvent | None = None
    if stable:
        if len(events) == 1 and name not in summary.external_kills:
            candidate = events[0]
            if candidate.order < before:
                event = candidate
    elif events:
        index = (
            bisect.bisect_left(events, before, key=lambda candidate: candidate.order)
            - 1
        )
        if index >= 0:
            event = events[index]

    if event is not None:
        key = ("carrier-fact", id(scope), name, event.order, stable)
        if event.kind == "import" and event.direct:
            return _CarrierFactPlan(
                key, summary, event, (), _CarrierFactState(None, True)
            )
        if event.kind == "assign" and event.direct and event.value is not None:
            dependencies = tuple(
                _CarrierFactRequest(scope, dependency, event.order, stable)
                for dependency in _carrier_loaded_names_bounded(event.value)
            )
            return _CarrierFactPlan(key, summary, event, dependencies, None)
        return _CarrierFactPlan(key, summary, event, (), _CarrierFactState(None))

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and (
        name in summary.local_names or name in summary.parameter_names
    ):
        key = ("carrier-fact-empty", id(scope), name, before, stable)
        return _CarrierFactPlan(key, summary, None, (), _CarrierFactState(None))
    parent_scope = _carrier_scope_parent(scope, parents)
    if parent_scope is None:
        key = ("carrier-fact-empty", id(scope), name, before, stable)
        return _CarrierFactPlan(key, summary, None, (), _CarrierFactState(None))
    parent_request = _CarrierFactRequest(
        parent_scope,
        name,
        (
            getattr(scope, "lineno", 1),
            getattr(scope, "col_offset", 0),
            0,
        ),
        True,
    )
    key = ("carrier-fact-parent", id(scope), name)
    return _CarrierFactPlan(key, summary, None, (parent_request,), None)


def _carrier_fact_state(
    scope: ast.AST,
    name: str,
    before: tuple[int, int, int],
    parents: dict[ast.AST, ast.AST],
    cache: dict[Any, Any],
    *,
    stable: bool,
) -> _CarrierFactState:
    """Resolve one frozen fact with an explicit enter/exit dependency worklist."""

    root = _CarrierFactRequest(scope, name, before, stable)
    stack: list[tuple[bool, _CarrierFactRequest, _CarrierFactPlan | None]] = [
        (False, root, None)
    ]
    resolving: set[tuple[Any, ...]] = set()
    root_key: tuple[Any, ...] | None = None
    while stack:
        exiting, request, saved_plan = stack.pop()
        plan = saved_plan or _carrier_fact_plan(request, parents, cache)
        if root_key is None:
            root_key = plan.key
        if exiting:
            facts: dict[str, ast.expr] = {}
            active_imports: set[str] = set()
            for dependency in plan.dependencies:
                dependency_plan = _carrier_fact_plan(dependency, parents, cache)
                dependency_state = cache.get(dependency_plan.key)
                if not isinstance(dependency_state, _CarrierFactState):
                    raise AssertionError("carrier fact dependency was not resolved")
                if dependency_state.value is not None:
                    facts[dependency.name] = dependency_state.value
                if dependency_state.active_import:
                    active_imports.add(dependency.name)
            if plan.event is None:
                dependency_plan = _carrier_fact_plan(
                    plan.dependencies[0], parents, cache
                )
                result = cache[dependency_plan.key]
                assert isinstance(result, _CarrierFactState)
            else:
                assert plan.event.value is not None
                result = _CarrierFactState(
                    _freeze_carrier_expression(
                        plan.event.value,
                        facts,
                        plan.summary.local_names,
                        active_imports,
                    )
                )
            cache[plan.key] = result
            resolving.remove(plan.key)
            continue
        cached = cache.get(plan.key)
        if isinstance(cached, _CarrierFactState):
            continue
        if plan.key in resolving:
            raise InventoryError("carrier assignment dependency is cyclic")
        if plan.immediate is not None:
            cache[plan.key] = plan.immediate
            continue
        resolving.add(plan.key)
        stack.append((True, request, plan))
        for dependency in reversed(plan.dependencies):
            stack.append((False, dependency, None))
    assert root_key is not None
    result = cache[root_key]
    assert isinstance(result, _CarrierFactState)
    return result


def _point_python_assignments(
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    assignment_cache: dict[Any, Any] | None = None,
) -> dict[str, ast.expr]:
    """Return assignment-time-frozen literal/path facts at one call site."""

    scope = _python_scope(call, parents)
    cache = {} if assignment_cache is None else assignment_cache
    call_order = (call.lineno, call.col_offset, 0)
    referenced_names = {
        candidate.id
        for candidate in ast.walk(call)
        if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Load)
    }
    facts: dict[str, ast.expr] = {}
    for name in sorted(referenced_names):
        state = _carrier_fact_state(
            scope,
            name,
            call_order,
            parents,
            cache,
            stable=False,
        )
        if state.value is not None:
            facts[name] = copy.deepcopy(state.value)
    for name in _cached_active_comprehension_bindings(call, parents, cache):
        facts.pop(name, None)
    return facts


def _literal_python_string(
    node: ast.expr,
    assignments: dict[str, ast.expr],
    seen: set[str] | None = None,
    memo: dict[int, str | None] | None = None,
) -> str | None:
    cache = {} if memo is None else memo
    if id(node) in cache:
        return cache[id(node)]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        cache[id(node)] = node.value
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_python_string(node.left, assignments, seen, cache)
        right = _literal_python_string(node.right, assignments, seen, cache)
        value = left + right if left is not None and right is not None else None
        if value is not None and len(value.encode("utf-8")) > EMBEDDED_PYTHON_MAX_BYTES:
            raise InventoryError("literal Python carrier string exceeds the byte limit")
        cache[id(node)] = value
        return value
    if isinstance(node, ast.Name) and node.id in assignments:
        visited = set() if seen is None else set(seen)
        if node.id in visited:
            return None
        visited.add(node.id)
        value = _literal_python_string(
            assignments[node.id], assignments, visited, cache
        )
        cache[id(node)] = value
        return value
    cache[id(node)] = None
    return None


def _is_python_interpreter(
    node: ast.expr,
    assignments: dict[str, ast.expr],
    aliases: dict[str, str],
    seen: set[str] | None = None,
) -> bool:
    if _dotted_name(node, aliases) == "sys.executable":
        return True
    if isinstance(node, ast.Name) and node.id in assignments:
        visited = set() if seen is None else set(seen)
        if node.id in visited:
            return False
        visited.add(node.id)
        if _is_python_interpreter(assignments[node.id], assignments, aliases, visited):
            return True
    value = _literal_python_string(node, assignments)
    if value is None:
        return False
    basename = value.replace("\\", "/").rsplit("/", 1)[-1]
    return _is_python_interpreter_basename(basename)


def _is_python_interpreter_basename(value: str) -> bool:
    return (
        re.fullmatch(r"pythonw?(?:3(?:\.\d+)?t?)?(?:\.exe)?", value, re.IGNORECASE)
        is not None
    )


def _static_python_sequence(
    node: ast.expr, assignments: dict[str, ast.expr]
) -> list[ast.expr] | None:
    def sequence_syntax(
        candidate: ast.expr,
        seen: frozenset[str] = frozenset(),
        depth: int = 0,
    ) -> bool:
        if depth > STATIC_PYTHON_ARGV_MAX_DEPTH:
            raise InventoryError("static Python argv composition exceeds depth limit")
        if isinstance(candidate, (ast.List, ast.Tuple)):
            return True
        if isinstance(candidate, ast.BinOp) and isinstance(candidate.op, ast.Add):
            return sequence_syntax(candidate.left, seen, depth + 1) or sequence_syntax(
                candidate.right, seen, depth + 1
            )
        if isinstance(candidate, ast.Name) and candidate.id in assignments:
            if candidate.id in seen:
                return True
            return sequence_syntax(
                assignments[candidate.id], seen | {candidate.id}, depth + 1
            )
        return False

    def resolve(
        candidate: ast.expr,
        seen: frozenset[str],
        depth: int,
        composing: bool = False,
    ) -> tuple[type[ast.List] | type[ast.Tuple], list[ast.expr]] | None:
        if depth > STATIC_PYTHON_ARGV_MAX_DEPTH:
            raise InventoryError("static Python argv composition exceeds depth limit")
        if isinstance(candidate, (ast.List, ast.Tuple)):
            if composing and any(
                isinstance(element, ast.Starred) for element in candidate.elts
            ):
                raise InventoryError(
                    "static Python argv composition expansion is ambiguous"
                )
            elements = list(candidate.elts)
            if len(elements) > STATIC_PYTHON_ARGV_MAX_ITEMS:
                raise InventoryError(
                    "static Python argv composition exceeds item limit"
                )
            return type(candidate), elements
        if isinstance(candidate, ast.Name) and candidate.id in assignments:
            if candidate.id in seen:
                raise InventoryError("static Python argv composition is cyclic")
            return resolve(
                assignments[candidate.id],
                seen | {candidate.id},
                depth + 1,
                composing,
            )
        if isinstance(candidate, ast.BinOp) and isinstance(candidate.op, ast.Add):
            left_is_sequence = sequence_syntax(candidate.left, seen)
            right_is_sequence = sequence_syntax(candidate.right, seen)
            if not left_is_sequence and not right_is_sequence:
                return None
            left = resolve(candidate.left, seen, depth + 1, True)
            right = resolve(candidate.right, seen, depth + 1, True)
            if left is None or right is None:
                raise InventoryError(
                    "static Python argv concatenation is dynamic or ambiguous"
                )
            if left[0] is not right[0]:
                raise InventoryError(
                    "static Python argv concatenation mixes list and tuple"
                )
            elements = [*left[1], *right[1]]
            if len(elements) > STATIC_PYTHON_ARGV_MAX_ITEMS:
                raise InventoryError(
                    "static Python argv composition exceeds item limit"
                )
            return left[0], elements
        return None

    resolved = resolve(node, frozenset(), 0)
    return None if resolved is None else resolved[1]


def _carrier_visible_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[tuple[ast.AST, tuple[int, int]]]:
    scope = _python_scope(node, parents)
    cutoff = (
        getattr(node, "lineno", 1),
        getattr(node, "col_offset", 0),
    )
    result: list[tuple[ast.AST, tuple[int, int]]] = []
    while scope is not None:
        result.append((scope, cutoff))
        cutoff = (
            getattr(scope, "lineno", cutoff[0]),
            getattr(scope, "col_offset", cutoff[1]),
        )
        scope = _carrier_scope_parent(scope, parents)
    return result


def _carrier_name_is_canonical_builtin(
    name: str,
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    cache: dict[Any, Any],
) -> bool:
    if name in _cached_active_comprehension_bindings(call, parents, cache):
        return False
    for scope, _ in _carrier_visible_scopes(call, parents):
        summary = _carrier_scope_summary(scope, parents, cache)
        if (
            name in summary.parameter_names
            or name in summary.binding_names
            or name in summary.external_kills
        ):
            return False
    return True


def _carrier_has_canonical_path_import(
    bound_name: str,
    import_kind: str,
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    cache: dict[Any, Any],
) -> bool:
    if bound_name in _cached_active_comprehension_bindings(call, parents, cache):
        return False
    for scope, cutoff in _carrier_visible_scopes(call, parents):
        summary = _carrier_scope_summary(scope, parents, cache)
        matching = summary.events_by_name.get(bound_name, ())
        if (
            bound_name in summary.parameter_names
            or bound_name in summary.external_kills
        ):
            return False
        if not matching:
            if (
                isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                and bound_name in summary.local_names
            ):
                return False
            continue
        if (
            len(matching) != 1
            or matching[0].kind != "import"
            or not matching[0].direct
            or matching[0].position >= cutoff
        ):
            return False
        return (
            bound_name,
            import_kind,
            matching[0].position,
        ) in summary.canonical_path_imports
    return False


def _canonical_path_call(
    node: ast.Call,
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    cache: dict[Any, Any],
) -> bool:
    if isinstance(node.func, ast.Name):
        return _carrier_has_canonical_path_import(
            node.func.id, "from", call, parents, cache
        )
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "Path"
        and isinstance(node.func.value, ast.Name)
        and _carrier_has_canonical_path_import(
            node.func.value.id, "module", call, parents, cache
        )
    )


def _absolute_python_script_path(
    node: ast.expr,
    assignments: dict[str, ast.expr],
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    cache: dict[Any, Any],
    physical_file: bool,
    seen: set[str] | None = None,
) -> bool:
    if isinstance(node, ast.Name):
        visited = set() if seen is None else set(seen)
        if node.id in visited:
            return False
        visited.add(node.id)
        producer = assignments.get(node.id)
        return producer is not None and _absolute_python_script_path(
            producer, assignments, call, parents, cache, physical_file, visited
        )
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        return _absolute_python_script_path(
            node.value, assignments, call, parents, cache, physical_file, seen
        )
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
        return (
            node.value.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
            and node.slice.value >= 0
            and _absolute_python_script_path(
                node.value.value,
                assignments,
                call,
                parents,
                cache,
                physical_file,
                seen,
            )
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        suffix = _literal_python_string(node.right, assignments)
        return (
            suffix is not None
            and bool(suffix)
            and not Path(suffix).is_absolute()
            and _absolute_python_script_path(
                node.left,
                assignments,
                call,
                parents,
                cache,
                physical_file,
                seen,
            )
        )
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "resolve":
        return False
    path_call = node.func.value
    return (
        isinstance(path_call, ast.Call)
        and _canonical_path_call(path_call, call, parents, cache)
        and len(path_call.args) == 1
        and isinstance(path_call.args[0], ast.Name)
        and path_call.args[0].id == "__file__"
        and not path_call.keywords
        and physical_file
        and _carrier_name_is_canonical_builtin("__file__", call, parents, cache)
    )


def _reviewed_python_script_stop(
    node: ast.expr,
    assignments: dict[str, ast.expr],
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    location: str,
    cache: dict[Any, Any],
    physical_file: bool,
) -> bool:
    if (
        isinstance(node, ast.Name)
        and node.id == "__file__"
        and physical_file
        and _carrier_name_is_canonical_builtin("__file__", call, parents, cache)
    ):
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
        and not node.keywords
        and _carrier_name_is_canonical_builtin("str", call, parents, cache)
    ):
        if _absolute_python_script_path(
            node.args[0], assignments, call, parents, cache, physical_file
        ):
            return True
    scope = _python_scope(call, parents)
    rel = location.rsplit(":", 1)[0]
    function = (
        scope.name
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
        else "module"
    )
    key = f"{rel}:{function}"
    expected = REVIEWED_PYTHON_SCRIPT_STOP_DIGESTS.get(key)
    if expected is None:
        return False
    actual = _json_fact_digest(
        {
            "version": PYTHON_SEMANTICS_VERSION,
            "file": rel,
            "function": _canonical_python_ast(scope),
            "call": _canonical_python_ast(call),
            "operand": _canonical_python_ast(node),
        }
    )
    if actual != expected:
        raise InventoryError(
            f"{location}: reviewed Python script stop digest changed ({actual})"
        )
    return True


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _shell_python_c_possible(node: ast.expr, assignments: dict[str, ast.expr]) -> bool:
    value = _literal_python_string(node, assignments)
    if value is None:
        return True
    # POSIX, cmd.exe, and PowerShell expand different token syntaxes.  The
    # inventory does not execute a shell or guess a dialect, so any runtime
    # command-name expansion must remain a possible Python carrier.
    if re.search(
        r"\$|`|%[^%\r\n]+%|![^!\r\n]+!|\^|[{}*?\[\];&|<>()+\r\n]",
        value,
    ):
        return True
    try:
        words = shlex.split(value, posix=True)
    except ValueError:
        return True
    elements = [ast.Constant(item) for item in words]
    dummy = ast.Call(func=ast.Name(id="shell", ctx=ast.Load()), args=[], keywords=[])
    for index, word in enumerate(words):
        basename = word.replace("\\", "/").rsplit("/", 1)[-1]
        if not _is_python_interpreter_basename(basename):
            continue
        scan = _scan_cpython_314_cli(
            elements,
            index + 1,
            {},
            dummy,
            {},
            "shell command",
            {},
            False,
        )
        if scan.code is not None:
            return True
    try:
        if elements and _indirect_python_c_carrier(
            elements,
            {},
            {},
            dummy,
            {},
            "shell command",
            {},
            False,
        ):
            return True
    except InventoryError:
        return True
    return False


WINDOWS_PY_DIRECT_LAUNCHERS = {"py", "py.exe", "pyw", "pyw.exe"}
WINDOWS_PY_MANAGER_LAUNCHERS = {
    "pymanager",
    "pymanager.exe",
    "pywmanager",
    "pywmanager.exe",
}
WINDOWS_PY_TERMINAL_COMMANDS = {
    "help",
    "install",
    "list",
    "shortcuts",
    "uninstall",
    "update",
}


def _is_proven_python_process_launcher(
    node: ast.expr,
    assignments: dict[str, ast.expr],
    aliases: dict[str, str],
) -> bool:
    if _is_python_interpreter(node, assignments, aliases):
        return True
    value = _literal_python_string(node, assignments)
    if value is None:
        return False
    basename = value.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return basename in WINDOWS_PY_DIRECT_LAUNCHERS | WINDOWS_PY_MANAGER_LAUNCHERS


def _unsupported_python_launcher_tail_index(
    vector: list[ast.expr],
    assignments: dict[str, ast.expr],
    aliases: dict[str, str],
) -> int | None:
    return next(
        (
            index
            for index, node in enumerate(vector[1:], 1)
            if _is_proven_python_process_launcher(node, assignments, aliases)
        ),
        None,
    )


def _windows_python_launcher_start(
    basename: str,
    values: list[str | None],
    location: str,
) -> int | None:
    """Return the CPython argv start for reviewed Windows launcher forms."""

    lowered_basename = basename.lower()
    direct = lowered_basename in WINDOWS_PY_DIRECT_LAUNCHERS
    manager = lowered_basename in WINDOWS_PY_MANAGER_LAUNCHERS
    if not direct and not manager:
        raise AssertionError("not a reviewed Windows Python launcher")
    if len(values) == 1:
        return 1 if direct else None

    first = values[1]
    if first is None:
        raise InventoryError(f"{location}: Windows Python launcher token is dynamic")
    lowered_first = first.lower()
    if lowered_first in WINDOWS_PY_TERMINAL_COMMANDS or lowered_first in {
        "--help",
        "--list",
        "--list-paths",
        "-0",
        "-0p",
    }:
        return None
    if manager:
        if lowered_first != "exec":
            raise InventoryError(
                f"{location}: unsupported Windows Python manager command"
            )
        start = 2
    else:
        start = 2 if lowered_first == "exec" else 1

    if start >= len(values):
        return start
    selector = values[start]
    if selector is None:
        raise InventoryError(f"{location}: Windows Python launcher selector is dynamic")
    version_selector = re.fullmatch(
        r"-V:(?:[A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]*",
        selector,
        re.IGNORECASE,
    )
    short_selector = re.fullmatch(
        r"-[23](?:\.\d+)?t?(?:-[A-Za-z0-9-]+)?",
        selector,
        re.IGNORECASE,
    )
    if version_selector is not None or short_selector is not None:
        return start + 1
    if selector.lower().startswith("-v:"):
        raise InventoryError(f"{location}: malformed Windows Python launcher selector")
    return start


class IndirectWrapperCommand(NamedTuple):
    interpreter_index: int | None
    split_payload: ast.expr | None = None
    command_index: int | None = None


def _wrapper_literal_value(
    vector: list[ast.expr],
    values: list[str | None],
    index: int,
    option: str,
    location: str,
) -> str:
    if index >= len(vector):
        raise InventoryError(f"{location}: {option} requires a value")
    if isinstance(vector[index], ast.Starred) or values[index] is None:
        raise InventoryError(f"{location}: {option} value is dynamic or ambiguous")
    value = values[index]
    assert value is not None
    if not value:
        raise InventoryError(f"{location}: {option} value is empty")
    return value


def _wrapper_interpreter_at(
    vector: list[ast.expr],
    values: list[str | None],
    index: int,
    wrapper: str,
    location: str,
) -> IndirectWrapperCommand:
    if index >= len(vector):
        return IndirectWrapperCommand(None)
    if isinstance(vector[index], ast.Starred) or values[index] is None:
        raise InventoryError(f"{location}: {wrapper} command is dynamic or ambiguous")
    command = values[index]
    assert command is not None
    basename = command.replace("\\", "/").rsplit("/", 1)[-1]
    return IndirectWrapperCommand(
        index if _is_python_interpreter_basename(basename) else None,
        command_index=index,
    )


def _env_wrapper_command(
    vector: list[ast.expr], values: list[str | None], location: str
) -> IndirectWrapperCommand:
    index = 1
    value_options = {"-u", "--unset", "-C", "--chdir", "-P"}
    flag_options = {
        "-0",
        "--null",
        "-i",
        "--ignore-environment",
        "-v",
        "--debug",
    }
    while index < len(vector):
        if isinstance(vector[index], ast.Starred) or values[index] is None:
            raise InventoryError(
                f"{location}: env option or command is dynamic or ambiguous"
            )
        value = values[index]
        assert value is not None
        if value == "--":
            return _wrapper_interpreter_at(vector, values, index + 1, "env", location)
        if value in {"-S", "--split-string"}:
            _wrapper_literal_value(vector, values, index + 1, value, location)
            return IndirectWrapperCommand(None, vector[index + 1])
        if value.startswith("--split-string="):
            payload = value.partition("=")[2]
            if not payload:
                raise InventoryError(f"{location}: --split-string requires a value")
            return IndirectWrapperCommand(None, ast.Constant(value=payload))
        if value.startswith("-S") and value != "-S":
            payload = value[2:]
            if not payload:
                raise InventoryError(f"{location}: -S requires a value")
            return IndirectWrapperCommand(None, ast.Constant(value=payload))
        if value in value_options:
            _wrapper_literal_value(vector, values, index + 1, value, location)
            index += 2
            continue
        if value.startswith(("--unset=", "--chdir=")):
            if not value.partition("=")[2]:
                raise InventoryError(f"{location}: {value} requires a value")
            index += 1
            continue
        if value in flag_options:
            index += 1
            continue
        if value.startswith("-"):
            raise InventoryError(f"{location}: unsupported env option {value!r}")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", value):
            index += 1
            continue
        return _wrapper_interpreter_at(vector, values, index, "env", location)
    return IndirectWrapperCommand(None)


def _subcommand_wrapper_command(
    wrapper: str,
    vector: list[ast.expr],
    values: list[str | None],
    location: str,
) -> IndirectWrapperCommand:
    if len(vector) == 1:
        return IndirectWrapperCommand(None)
    subcommand = _wrapper_literal_value(
        vector, values, 1, f"{wrapper} subcommand", location
    )
    if subcommand != "run":
        return IndirectWrapperCommand(None)
    if wrapper == "conda":
        value_options = {"-n", "--name", "-p", "--prefix", "--cwd"}
        flag_options = {
            "--dev",
            "--debug-wrapper-scripts",
            "--no-capture-output",
            "--live-stream",
        }
    else:
        value_options = {
            "-p",
            "--python",
            "--directory",
            "--project",
            "--with",
            "--with-editable",
            "--with-requirements",
            "--env-file",
            "--index",
            "--default-index",
            "--index-url",
            "--extra-index-url",
            "--find-links",
            "--resolution",
            "--prerelease",
            "--fork-strategy",
            "--exclude-newer",
            "--link-mode",
            "--config-file",
            "--cache-dir",
            "--python-platform",
            "--python-version",
            "--python-preference",
            "--keyring-provider",
        }
        flag_options = {
            "--isolated",
            "--no-project",
            "--active",
            "--no-sync",
            "--locked",
            "--frozen",
            "--no-cache",
            "--offline",
            "--compile-bytecode",
            "--no-compile-bytecode",
            "--no-config",
            "--managed-python",
            "--no-managed-python",
            "-q",
            "--quiet",
            "-v",
            "--verbose",
        }
    index = 2
    while index < len(vector):
        if isinstance(vector[index], ast.Starred) or values[index] is None:
            raise InventoryError(
                f"{location}: {wrapper} run option or command is dynamic or ambiguous"
            )
        value = values[index]
        assert value is not None
        if value == "--":
            return _wrapper_interpreter_at(
                vector, values, index + 1, f"{wrapper} run", location
            )
        if value in value_options:
            _wrapper_literal_value(vector, values, index + 1, value, location)
            index += 2
            continue
        if any(value.startswith(option + "=") for option in value_options):
            if not value.partition("=")[2]:
                raise InventoryError(f"{location}: {value} requires a value")
            index += 1
            continue
        if value in flag_options:
            index += 1
            continue
        if value.startswith("-"):
            raise InventoryError(
                f"{location}: unsupported {wrapper} run option {value!r}"
            )
        return _wrapper_interpreter_at(
            vector, values, index, f"{wrapper} run", location
        )
    return IndirectWrapperCommand(None)


def _xcrun_wrapper_command(
    vector: list[ast.expr], values: list[str | None], location: str
) -> IndirectWrapperCommand:
    value_options = {"--sdk", "-sdk", "--toolchain", "-toolchain"}
    terminal_value_options = {"--find", "-f"}
    terminal_options = {
        "--show-sdk-path",
        "--show-sdk-version",
        "--show-sdk-build-version",
        "--show-sdk-platform-path",
        "--show-sdk-platform-version",
        "--kill-cache",
    }
    flag_options = {"--run", "-r", "--log", "-l", "--verbose", "-v", "--no-cache", "-n"}
    index = 1
    while index < len(vector):
        if isinstance(vector[index], ast.Starred) or values[index] is None:
            raise InventoryError(
                f"{location}: xcrun option or command is dynamic or ambiguous"
            )
        value = values[index]
        assert value is not None
        if value == "--":
            return _wrapper_interpreter_at(vector, values, index + 1, "xcrun", location)
        if value in terminal_value_options:
            _wrapper_literal_value(vector, values, index + 1, value, location)
            return IndirectWrapperCommand(None)
        if value in terminal_options:
            return IndirectWrapperCommand(None)
        if value in value_options:
            _wrapper_literal_value(vector, values, index + 1, value, location)
            index += 2
            continue
        if value in flag_options:
            index += 1
            continue
        if value.startswith("-"):
            raise InventoryError(f"{location}: unsupported xcrun option {value!r}")
        return _wrapper_interpreter_at(vector, values, index, "xcrun", location)
    return IndirectWrapperCommand(None)


def _nice_wrapper_command(
    vector: list[ast.expr],
    values: list[str | None],
    location: str,
    assignments: dict[str, ast.expr],
    aliases: dict[str, str],
    start: int = 0,
) -> IndirectWrapperCommand:
    def command_at(index: int) -> IndirectWrapperCommand:
        if index < len(vector) and values[index] == "--":
            index += 1
        if index >= len(vector):
            raise InventoryError(f"{location}: nice command is missing")
        if _is_python_interpreter(vector[index], assignments, aliases):
            return IndirectWrapperCommand(index, command_index=index)
        return _wrapper_interpreter_at(vector, values, index, "nice", location)

    index = start + 1
    if index >= len(vector):
        raise InventoryError(f"{location}: nice command is missing")
    value = _wrapper_literal_value(
        vector, values, index, "nice option or command", location
    )
    if value == "--":
        return command_at(index + 1)
    if value == "-n":
        adjustment = _wrapper_literal_value(
            vector, values, index + 1, "nice -n", location
        )
        if re.fullmatch(r"[+-]?\d+", adjustment) is None:
            raise InventoryError(f"{location}: nice -n adjustment is invalid")
        return command_at(index + 2)
    if value.startswith("--adjustment="):
        adjustment = value.partition("=")[2]
        if re.fullmatch(r"[+-]?\d+", adjustment) is None:
            raise InventoryError(
                f"{location}: nice --adjustment value is missing or invalid"
            )
        return command_at(index + 1)
    if value.startswith("-"):
        raise InventoryError(f"{location}: unsupported nice option {value!r}")
    return command_at(index)


def _transparent_nice_interpreter_index(
    vector: list[ast.expr],
    values: list[str | None],
    location: str,
    assignments: dict[str, ast.expr],
    aliases: dict[str, str],
) -> int | None:
    command_index = 0
    while command_index < len(vector):
        command_node = vector[command_index]
        if isinstance(command_node, ast.Starred):
            raise InventoryError(f"{location}: nice command is dynamic or ambiguous")
        if _is_python_interpreter(command_node, assignments, aliases):
            return command_index
        command = values[command_index]
        if command is None:
            raise InventoryError(f"{location}: nice command is dynamic or ambiguous")
        basename = command.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if basename != "nice":
            return None
        wrapper = _nice_wrapper_command(
            vector,
            values,
            location,
            assignments,
            aliases,
            start=command_index,
        )
        if wrapper.command_index is None:
            return None
        command_index = wrapper.command_index
    return None


def _indirect_python_c_carrier(
    vector: list[ast.expr],
    assignments: dict[str, ast.expr],
    aliases: dict[str, str],
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    location: str,
    cache: dict[Any, Any],
    physical_file: bool,
) -> bool:
    values = [_literal_python_string(item, assignments) for item in vector]
    executable = values[0]
    if executable is None:
        if (
            _unsupported_python_launcher_tail_index(vector, assignments, aliases)
            is not None
        ):
            raise InventoryError(
                f"{location}: proven Python launcher appears after unsupported "
                "wrapper/tool tail"
            )
        return False
    basename = executable.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if basename in WINDOWS_PY_DIRECT_LAUNCHERS | WINDOWS_PY_MANAGER_LAUNCHERS:
        start = _windows_python_launcher_start(basename, values, location)
        if start is None:
            return False
        return (
            _scan_cpython_314_cli(
                vector,
                start,
                assignments,
                call,
                parents,
                location,
                cache,
                physical_file,
            ).code
            is not None
        )
    if basename in {"env", "uv", "conda", "xcrun", "nice"}:
        if basename == "env":
            wrapper = _env_wrapper_command(vector, values, location)
        elif basename in {"uv", "conda"}:
            wrapper = _subcommand_wrapper_command(basename, vector, values, location)
        elif basename == "nice":
            wrapper = _nice_wrapper_command(
                vector, values, location, assignments, aliases
            )
        else:
            wrapper = _xcrun_wrapper_command(vector, values, location)
        if wrapper.split_payload is not None:
            return _shell_python_c_possible(wrapper.split_payload, assignments)
        interpreter_index = wrapper.interpreter_index
        if interpreter_index is not None:
            return (
                _scan_cpython_314_cli(
                    vector,
                    interpreter_index + 1,
                    assignments,
                    call,
                    parents,
                    location,
                    cache,
                    physical_file,
                ).code
                is not None
            )
        if wrapper.command_index is not None:
            return _indirect_python_c_carrier(
                vector[wrapper.command_index :],
                assignments,
                aliases,
                call,
                parents,
                location,
                cache,
                physical_file,
            )
        return False
    shell_marker: int | None = None
    if basename in {"bash", "sh", "zsh"}:
        shell_marker = next(
            (
                index
                for index, value in enumerate(values[1:], 1)
                if value is not None
                and (
                    re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", value)
                    or value.startswith("-c")
                )
            ),
            None,
        )
    elif basename in {"cmd", "cmd.exe"}:
        shell_marker = next(
            (
                index
                for index, value in enumerate(values[1:], 1)
                if value is not None
                and (
                    value.lower() in {"/c", "/k"}
                    or value.lower().startswith(("/c", "/k"))
                )
            ),
            None,
        )
    elif basename in {
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
    }:
        shell_marker = next(
            (
                index
                for index, value in enumerate(values[1:], 1)
                if value is not None
                and (
                    value.lower() in {"-c", "-command", "-ec", "-encodedcommand"}
                    or value.lower().startswith(("-c", "-command"))
                )
            ),
            None,
        )
        if shell_marker is not None and values[shell_marker].lower() in {
            "-ec",
            "-encodedcommand",
        }:
            return True
    if basename in {
        "bash",
        "sh",
        "zsh",
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
    } and any(
        values[index] is None or isinstance(vector[index], ast.Starred)
        for index in range(1, shell_marker if shell_marker is not None else len(vector))
    ):
        return True
    if shell_marker is not None:
        option = values[shell_marker]
        if option is not None:
            lowered = option.lower()
            attached_payload: str | None = None
            if basename in {"bash", "sh", "zsh"} and lowered.startswith("-c"):
                attached_payload = option[2:] or None
            elif basename in {"cmd", "cmd.exe"} and lowered.startswith(("/c", "/k")):
                attached_payload = option[2:] or None
            elif basename in {
                "powershell",
                "powershell.exe",
                "pwsh",
                "pwsh.exe",
            }:
                if lowered.startswith(("-command:", "-command=")):
                    attached_payload = option[len("-command:") :]
                elif lowered.startswith("-c") and lowered not in {
                    "-c",
                    "-command",
                }:
                    attached_payload = option[2:]
            if attached_payload is not None:
                return _shell_python_c_possible(
                    ast.Constant(value=attached_payload), assignments
                )
        return shell_marker + 1 >= len(vector) or _shell_python_c_possible(
            vector[shell_marker + 1], assignments
        )
    if basename in {
        "bash",
        "sh",
        "zsh",
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
    }:
        return False
    if (
        _unsupported_python_launcher_tail_index(vector, assignments, aliases)
        is not None
    ):
        raise InventoryError(
            f"{location}: proven Python launcher appears after unsupported "
            "wrapper/tool tail"
        )
    return False


def _scan_cpython_314_cli(
    elements: list[ast.expr],
    start: int,
    assignments: dict[str, ast.expr],
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    location: str,
    cache: dict[Any, Any],
    physical_file: bool,
) -> PythonCliScan:
    """Scan CPython 3.14 options without executing or guessing dynamic argv.

    Short options are interpreted character-by-character because CPython accepts
    clusters such as ``-Ec`` and attached ``-cCODE``.  Once a terminal mode or a
    proved script operand is reached, later argv values are payload and no longer
    participate in interpreter-option discovery.
    """

    proof: list[dict[str, Any]] = []

    def finish(code: str | None, terminal: str) -> PythonCliScan:
        return PythonCliScan(
            code,
            _json_fact_digest(
                {
                    "version": PYTHON_CARRIER_SEMANTICS_VERSION,
                    "tokens": proof,
                    "terminal": terminal,
                }
            ),
        )

    def consume_value(option: str, option_index: int) -> tuple[str, int]:
        error = (
            "Python -c code must be a static string"
            if option == "-c"
            else f"Python option {option} value is dynamic or missing"
        )
        value_index = option_index + 1
        if value_index >= len(elements) or isinstance(
            elements[value_index], ast.Starred
        ):
            raise InventoryError(f"{location}: {error}")
        value = _literal_python_string(elements[value_index], assignments)
        if value is None:
            raise InventoryError(f"{location}: {error}")
        return value, value_index

    index = start
    while index < len(elements):
        element = elements[index]
        if isinstance(element, ast.Starred):
            raise InventoryError(
                f"{location}: Python -c carrier expansion is ambiguous"
            )
        value = _literal_python_string(element, assignments)
        if value is None:
            if _reviewed_python_script_stop(
                element,
                assignments,
                call,
                parents,
                location,
                cache,
                physical_file,
            ):
                proof.append(
                    {
                        "kind": "proved-script",
                        "expression": _canonical_python_ast(element),
                    }
                )
                return finish(None, "script")
            raise InventoryError(f"{location}: Python -c option/marker is dynamic")
        if value == "--":
            proof.append({"kind": "end-options", "token": value})
            return finish(None, "double-dash")
        if value == "-":
            proof.append({"kind": "stdin", "token": value})
            return finish(None, "stdin")
        if not value.startswith("-"):
            proof.append({"kind": "script", "token": value})
            return finish(None, "script")
        if value.startswith("--"):
            if value in {"--help", "--version"}:
                proof.append({"kind": "early-exit", "token": value})
                return finish(None, "early-exit")
            if value != "--check-hash-based-pycs":
                raise InventoryError(
                    f"{location}: unsupported CPython 3.14 option {value!r}"
                )
            option_value, index = consume_value(value, index)
            if option_value not in {"always", "default", "never"}:
                raise InventoryError(
                    f"{location}: invalid --check-hash-based-pycs value"
                )
            proof.append({"kind": "long-option", "token": value, "value": option_value})
            index += 1
            continue

        characters = value[1:]
        if not characters:
            proof.append({"kind": "stdin", "token": value})
            return finish(None, "stdin")
        offset = 0
        while offset < len(characters):
            option = characters[offset]
            if option in "bBdEiIOPqRsSuvx":
                proof.append({"kind": "flag", "option": option})
                offset += 1
                continue
            if option in "h?V":
                proof.append({"kind": "early-exit", "option": option})
                return finish(None, "early-exit")
            if option == "c":
                attached = characters[offset + 1 :]
                if attached:
                    proof.append({"kind": "command", "attached": True})
                    return finish(attached, "command")
                code, index = consume_value("-c", index)
                proof.append({"kind": "command", "attached": False})
                return finish(code, "command")
            if option == "m":
                attached = characters[offset + 1 :]
                if attached:
                    proof.append({"kind": "module", "attached": True})
                else:
                    module_name, index = consume_value("-m", index)
                    proof.append(
                        {"kind": "module", "attached": False, "value": module_name}
                    )
                return finish(None, "module")
            if option in "WX":
                attached = characters[offset + 1 :]
                if attached:
                    proof.append(
                        {"kind": "option-value", "option": option, "attached": True}
                    )
                else:
                    option_value, index = consume_value(f"-{option}", index)
                    proof.append(
                        {
                            "kind": "option-value",
                            "option": option,
                            "attached": False,
                            "value": option_value,
                        }
                    )
                break
            raise InventoryError(
                f"{location}: unsupported CPython 3.14 short option -{option}"
            )
        index += 1
    return finish(None, "argv-end")


def _decode_python_carrier_call_shape(
    call: ast.Call,
    dotted: str | None,
    *,
    location: str,
    reviewed_adapter: bool,
    reviewed_keyword_expansion: bool,
) -> PythonCarrierCallShape | None:
    """Decode only process APIs whose public signatures prove argv placement."""

    capable = {
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "pty.spawn",
        "os.execv",
        "os.execvp",
        "os.execve",
        "os.execvpe",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.spawnv",
        "os.spawnvp",
        "os.spawnve",
        "os.spawnvpe",
        "os.execl",
        "os.execlp",
        "os.execle",
        "os.execlpe",
        "os.spawnl",
        "os.spawnlp",
        "os.spawnle",
        "os.spawnlpe",
        "asyncio.create_subprocess_exec",
        "asyncio.loop.subprocess_exec",
    }
    if dotted not in capable and not reviewed_adapter:
        return None
    if any(keyword.arg is None for keyword in call.keywords) and not (
        reviewed_keyword_expansion
    ):
        raise InventoryError(
            f"{location}: carrier-capable process call keyword expansion is ambiguous"
        )

    subprocess_argv = {
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "pty.spawn",
    }
    fixed_slots = subprocess_argv | {
        "os.execv",
        "os.execvp",
        "os.execve",
        "os.execvpe",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.spawnv",
        "os.spawnvp",
        "os.spawnve",
        "os.spawnvpe",
    }
    if (dotted in fixed_slots or reviewed_adapter) and any(
        isinstance(argument, ast.Starred) for argument in call.args
    ):
        raise InventoryError(
            f"{location}: fixed-slot process call positional expansion is ambiguous"
        )

    if dotted in subprocess_argv or reviewed_adapter:
        argv = call.args[0] if call.args else _keyword(call, "args")
        return PythonCarrierCallShape(
            None if argv is None else [argv],
            _keyword(call, "executable"),
            1,
            "reviewed-adapter-argv0" if reviewed_adapter else "argv-arg0",
        )
    if dotted in {
        "os.execv",
        "os.execvp",
        "os.execve",
        "os.execvpe",
        "os.posix_spawn",
        "os.posix_spawnp",
    }:
        executable = call.args[0] if call.args else _keyword(call, "path")
        argv = call.args[1] if len(call.args) >= 2 else _keyword(call, "argv")
        return PythonCarrierCallShape(
            None if argv is None else [argv], executable, 1, "path+argv"
        )
    if dotted in {"os.spawnv", "os.spawnvp", "os.spawnve", "os.spawnvpe"}:
        executable = call.args[1] if len(call.args) >= 2 else _keyword(call, "path")
        argv = call.args[2] if len(call.args) >= 3 else _keyword(call, "args")
        return PythonCarrierCallShape(
            None if argv is None else [argv], executable, 1, "mode+path+argv"
        )
    if dotted in {"os.execl", "os.execlp", "os.execle", "os.execlpe"}:
        executable = call.args[0] if call.args else None
        if dotted in {"os.execle", "os.execlpe"} and (
            len(call.args) < 3 or isinstance(call.args[-1], ast.Starred)
        ):
            raise InventoryError(
                f"{location}: exec l-form environment slot is ambiguous"
            )
        vector = list(
            call.args[1:-1] if dotted in {"os.execle", "os.execlpe"} else call.args[1:]
        )
        return PythonCarrierCallShape(vector, executable, 1, "path+positional-argv")
    if dotted in {"os.spawnl", "os.spawnlp", "os.spawnle", "os.spawnlpe"}:
        executable = call.args[1] if len(call.args) >= 2 else None
        if dotted in {"os.spawnle", "os.spawnlpe"} and (
            len(call.args) < 4 or isinstance(call.args[-1], ast.Starred)
        ):
            raise InventoryError(
                f"{location}: spawn l-form environment slot is ambiguous"
            )
        vector = list(
            call.args[2:-1]
            if dotted in {"os.spawnle", "os.spawnlpe"}
            else call.args[2:]
        )
        return PythonCarrierCallShape(
            vector, executable, 1, "mode+path+positional-argv"
        )
    if dotted == "asyncio.create_subprocess_exec":
        return PythonCarrierCallShape(list(call.args), None, 1, "asyncio-varargs")
    if dotted == "asyncio.loop.subprocess_exec":
        if call.args and isinstance(call.args[0], ast.Starred):
            raise InventoryError(
                f"{location}: loop protocol-factory expansion is ambiguous"
            )
        return PythonCarrierCallShape(
            list(call.args[1:]), None, 1, "loop-protocol+varargs"
        )
    return None


def _extract_python_carrier(
    call: ast.Call,
    dotted: str | None,
    aliases: dict[str, str],
    parents: dict[ast.AST, ast.AST],
    *,
    location: str,
    adapter_definition_digest: str | None = None,
    parent_carrier_digest: str | None = None,
    assignment_cache: dict[Any, Any] | None = None,
    physical_file: bool = True,
    reviewed_dynamic_physical: bool = False,
    reviewed_keyword_expansion: bool = False,
) -> PythonCarrier | None:
    cache = {} if assignment_cache is None else assignment_cache
    shell_apis = {
        "os.system",
        "os.popen",
        "subprocess.getoutput",
        "subprocess.getstatusoutput",
        "asyncio.create_subprocess_shell",
        "asyncio.loop.subprocess_shell",
    }
    if dotted in shell_apis:
        assignments = _point_python_assignments(call, parents, cache)
        command_node = (
            call.args[1]
            if dotted == "asyncio.loop.subprocess_shell" and len(call.args) > 1
            else call.args[0]
            if call.args
            else _keyword(call, "cmd")
        )
        if command_node is None or _shell_python_c_possible(command_node, assignments):
            raise InventoryError(
                f"{location}: shell Python -c carriers are unsupported"
            )
        return None
    shell_value = _keyword(call, "shell")
    shell_enabled = shell_value is not None and not (
        isinstance(shell_value, ast.Constant) and shell_value.value is False
    )

    decoded_shape = _decode_python_carrier_call_shape(
        call,
        dotted,
        location=location,
        reviewed_adapter=adapter_definition_digest is not None,
        reviewed_keyword_expansion=reviewed_keyword_expansion,
    )
    if decoded_shape is None:
        return None
    assignments = _point_python_assignments(call, parents, cache)
    vector = decoded_shape.vector
    executable_node = decoded_shape.executable
    start = decoded_shape.start
    shape = decoded_shape.shape
    argv_node: ast.expr | None = None
    if (
        vector is not None
        and len(vector) == 1
        and shape
        in {
            "argv-arg0",
            "reviewed-adapter-argv0",
            "path+argv",
            "mode+path+argv",
        }
    ):
        argv_node = vector[0]
        vector = None

    if shell_enabled:
        shell_source = (
            argv_node if argv_node is not None else (vector[0] if vector else None)
        )
        if shell_source is None or _shell_python_c_possible(shell_source, assignments):
            raise InventoryError(
                f"{location}: shell Python -c carriers are unsupported"
            )
        return None
    if vector is None and argv_node is not None:
        vector = _static_python_sequence(argv_node, assignments)
        if vector is None:
            if (
                (dotted == "subprocess.run" and executable_node is None)
                or reviewed_dynamic_physical
                or adapter_definition_digest is not None
            ):
                return None
            raise InventoryError(
                f"{location}: carrier-capable executable or argv proof is dynamic"
            )
    if not vector:
        return None
    interpreter_node = executable_node if executable_node is not None else vector[0]
    if isinstance(vector[0], ast.Starred):
        raise InventoryError(f"{location}: Python carrier argv0 expansion is ambiguous")
    if executable_node is None:
        values = [_literal_python_string(item, assignments) for item in vector]
        first = values[0]
        if first is not None:
            first_basename = first.replace("\\", "/").rsplit("/", 1)[-1].lower()
            if first_basename == "nice":
                nice_interpreter_index = _transparent_nice_interpreter_index(
                    vector, values, location, assignments, aliases
                )
                if nice_interpreter_index is not None:
                    interpreter_node = vector[nice_interpreter_index]
                    start = nice_interpreter_index + 1
                    shape = f"transparent-nice+{shape}"
    carrier_scope = _python_scope(call, parents)
    parameter_names: set[str] = set()
    if isinstance(carrier_scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        parameter_names.update(
            argument.arg
            for argument in (
                *carrier_scope.args.posonlyargs,
                *carrier_scope.args.args,
                *carrier_scope.args.kwonlyargs,
            )
        )
        if carrier_scope.args.vararg is not None:
            parameter_names.add(carrier_scope.args.vararg.arg)
        if carrier_scope.args.kwarg is not None:
            parameter_names.add(carrier_scope.args.kwarg.arg)
    interpreter = not (
        isinstance(interpreter_node, ast.Name)
        and interpreter_node.id in parameter_names
    ) and _is_python_interpreter(interpreter_node, assignments, aliases)
    if not interpreter:
        if _literal_python_string(interpreter_node, assignments) is None:
            possible_scan: PythonCliScan | None = None
            try:
                possible_scan = _scan_cpython_314_cli(
                    vector,
                    start,
                    assignments,
                    call,
                    parents,
                    location,
                    cache,
                    physical_file,
                )
            except InventoryError as exc:
                if "unsupported CPython 3.14" not in str(exc):
                    raise InventoryError(
                        f"{location}: Python carrier executable or CLI is dynamic"
                    ) from exc
            if possible_scan is not None and possible_scan.code is not None:
                raise InventoryError(
                    f"{location}: Python carrier executable is dynamic"
                )
        if _indirect_python_c_carrier(
            vector,
            assignments,
            aliases,
            call,
            parents,
            location,
            cache,
            physical_file,
        ):
            raise InventoryError(
                f"{location}: indirect Python -c carrier is unsupported"
            )
        return None
    cli_scan = _scan_cpython_314_cli(
        vector,
        start,
        assignments,
        call,
        parents,
        location,
        cache,
        physical_file,
    )
    if cli_scan.code is None:
        return None
    payload = {
        "version": PYTHON_CARRIER_SEMANTICS_VERSION,
        "canonical_api": dotted or "reviewed:observe_abi_baseline.run_command",
        "shape": shape,
        "shape_proof": _canonical_python_ast(call),
        "cli_proof_digest": cli_scan.proof_digest,
        "fact_proof_digest": _json_fact_digest(
            {
                name: _canonical_python_ast(value)
                for name, value in sorted(assignments.items())
            }
        ),
        "call": _canonical_python_ast(call),
        "statement": _canonical_python_ast(
            next(
                (
                    ancestor
                    for ancestor in _ancestor_chain(call, parents)
                    if isinstance(ancestor, ast.stmt)
                ),
                call,
            )
        ),
        "lexical_function": _canonical_python_ast(_python_scope(call, parents)),
        "ancestor_control_path": _python_ancestor_control_path(call, parents),
        "adapter_definition_digest": adapter_definition_digest,
        "parent_carrier_digest": parent_carrier_digest,
    }
    return PythonCarrier(
        cli_scan.code,
        shape,
        _json_fact_digest(payload),
        adapter_definition_digest,
    )


def _workflow_mapping(
    text: str,
    rel: str,
    line_number: int,
    *,
    allow_compact_quoted: bool = False,
) -> tuple[str, str] | None:
    match = re.fullmatch(
        r'(?:"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\']*)\'|([A-Za-z0-9_.-]+))[ \t]*:(?:[ \t]+(.*)|[ \t]*)',
        text,
    )
    match_groups = match.groups() if match is not None else None
    if match_groups is None and allow_compact_quoted:
        compact = re.fullmatch(
            r'(?:"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\']*)\')[ \t]*:(\S.*)',
            text,
        )
        if compact is not None:
            compact_double, compact_single, compact_value = compact.groups()
            match_groups = (compact_double, compact_single, None, compact_value)
    if match_groups is None:
        return None
    double_quoted, single_quoted, plain, value = match_groups
    if double_quoted is not None and "\\" in double_quoted:
        raise InventoryError(
            f"{rel}:{line_number}: escaped workflow mapping keys are unsupported"
        )
    if single_quoted is not None and "''" in single_quoted:
        raise InventoryError(
            f"{rel}:{line_number}: escaped workflow mapping keys are unsupported"
        )
    if plain is not None and _workflow_plain_scalar_is_non_string(plain):
        raise InventoryError(
            f"{rel}:{line_number}: workflow mapping keys must be scalar strings"
        )
    return next(
        key for key in (double_quoted, single_quoted, plain) if key is not None
    ), value or ""


def _workflow_flow_fields(content: str, rel: str, line_number: int) -> list[str]:
    if not content.strip():
        return []
    fields: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    nesting: list[str] = []
    for index, char in enumerate(content):
        if quote is not None:
            if escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in "[{":
            if len(nesting) >= 64:
                raise InventoryError(
                    f"{rel}:{line_number}: workflow flow nesting exceeds the supported limit"
                )
            nesting.append(char)
        elif char in "]}":
            expected = "[" if char == "]" else "{"
            if not nesting or nesting[-1] != expected:
                raise InventoryError(
                    f"{rel}:{line_number}: malformed workflow flow collection"
                )
            nesting.pop()
        elif char == "," and not nesting:
            fields.append(content[start:index])
            start = index + 1
    if quote is not None or nesting:
        raise InventoryError(f"{rel}:{line_number}: malformed workflow flow collection")
    fields.append(content[start:])
    if fields and not fields[-1].strip() and content.rstrip().endswith(","):
        fields.pop()
    if any(not field.strip() for field in fields):
        raise InventoryError(f"{rel}:{line_number}: empty workflow flow item")
    return fields


def _workflow_flow_mapping(
    text: str, rel: str, line_number: int
) -> list[tuple[str, str]]:
    if not (text.startswith("{") and text.endswith("}")):
        raise InventoryError(
            f"{rel}:{line_number}: workflow flow mapping must be contained on one line"
        )

    fields = _workflow_flow_fields(text[1:-1], rel, line_number)

    properties: list[tuple[str, str]] = []
    keys: set[str] = set()
    for field in fields:
        mapping = _workflow_mapping(
            field.strip(), rel, line_number, allow_compact_quoted=True
        )
        if mapping is None:
            raise InventoryError(
                f"{rel}:{line_number}: workflow flow property cannot be normalized"
            )
        key, value = mapping
        normalized_value = _workflow_strip_comment(value.strip())
        if normalized_value.startswith(("&", "*", "!")):
            raise InventoryError(
                f"{rel}:{line_number}: workflow anchors, aliases, and tags are unsupported"
            )
        if key in keys:
            raise InventoryError(
                f"{rel}:{line_number}: duplicate workflow flow key {key!r}"
            )
        keys.add(key)
        if normalized_value:
            _validate_workflow_flow_value(
                normalized_value,
                rel,
                line_number,
                "flow mapping value",
                in_flow=True,
            )
        properties.append((key, value))
    return properties


def _validate_workflow_flow_value(
    text: str,
    rel: str,
    line_number: int,
    field: str,
    *,
    in_flow: bool = False,
) -> None:
    stripped = _workflow_strip_comment(text.strip())
    if not stripped:
        return
    if stripped.startswith("{"):
        _workflow_flow_mapping(stripped, rel, line_number)
        return
    if stripped.startswith("["):
        if not stripped.endswith("]"):
            raise InventoryError(
                f"{rel}:{line_number}: workflow flow sequence must be contained on one line"
            )
        for item in _workflow_flow_fields(stripped[1:-1], rel, line_number):
            _validate_workflow_flow_value(
                item,
                rel,
                line_number,
                "flow sequence value",
                in_flow=True,
            )
        return
    if stripped.startswith(('"', "'")):
        _workflow_scalar(stripped, rel, line_number, field)
        return
    if in_flow and (
        stripped.endswith(":") or any(indicator in stripped for indicator in ",[]{}")
    ):
        raise InventoryError(
            f"{rel}:{line_number}: flow indicators are invalid in plain workflow {field}"
        )
    _validate_workflow_plain_scalar_syntax(stripped, rel, line_number, field)


def _validate_workflow_plain_scalar_syntax(
    stripped: str, rel: str, line_number: int, field: str
) -> None:
    if stripped.startswith(("|", ">")):
        if _workflow_block_scalar_indicator(stripped):
            return
        raise InventoryError(
            f"{rel}:{line_number}: invalid workflow block scalar indicator"
        )
    if (
        re.match(r"[-?:][ \t]", stripped)
        or stripped[0]
        in {",", "[", "]", "{", "}", "#", "&", "*", "!", "'", '"', "%", "@", "`"}
        or stripped.endswith(":")
        or re.search(r":\s", stripped)
    ):
        raise InventoryError(
            f"{rel}:{line_number}: invalid plain workflow {field} scalar"
        )


def _workflow_plain_scalar_is_non_string(stripped: str) -> bool:
    lowered = stripped.lower()
    return bool(
        lowered
        in {
            "true",
            "false",
            "null",
            "~",
            ".nan",
            ".inf",
            "+.inf",
            "-.inf",
        }
        or re.fullmatch(
            r"[-+]?(?:(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)(?:e[-+]?[0-9]+)?|0x[0-9a-f_]+|0o[0-7_]+|0b[01_]+)",
            lowered,
        )
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:[tT ].*)?", stripped)
    )


def _workflow_scalar(value: str, rel: str, line_number: int, field: str) -> str:
    stripped = _workflow_strip_comment(value.strip())
    if not stripped or stripped.startswith("#"):
        raise InventoryError(f"{rel}:{line_number}: empty workflow {field} value")
    if stripped[0] in "[{&*!" or stripped.lower() in {"null", "~"}:
        raise InventoryError(
            f"{rel}:{line_number}: workflow {field} must be a scalar string"
        )
    if stripped[0] in {'"', "'"}:
        if len(stripped) < 2 or stripped[-1] != stripped[0]:
            raise InventoryError(
                f"{rel}:{line_number}: multiline quoted workflow {field} is unsupported"
            )
        if stripped[0] == '"' and "\\" in stripped[1:-1]:
            raise InventoryError(
                f"{rel}:{line_number}: escaped double-quoted workflow {field} is unsupported"
            )
        if stripped[0] == "'" and "''" in stripped[1:-1]:
            raise InventoryError(
                f"{rel}:{line_number}: escaped single-quoted workflow {field} is unsupported"
            )
        if stripped[0] in stripped[1:-1]:
            raise InventoryError(
                f"{rel}:{line_number}: multiple quoted workflow {field} tokens are unsupported"
            )
        return stripped[1:-1]
    _validate_workflow_plain_scalar_syntax(stripped, rel, line_number, field)
    if _workflow_plain_scalar_is_non_string(stripped):
        raise InventoryError(
            f"{rel}:{line_number}: workflow {field} must be a scalar string"
        )
    return stripped


def _workflow_strip_comment(text: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "#" and (index == 0 or text[index - 1].isspace()):
            return text[:index].rstrip()
    return text


def _workflow_block_scalar_indicator(value: str) -> bool:
    normalized = _workflow_strip_comment(value.strip())
    return re.fullmatch(r"[|>](?:[1-9][+-]?|[+-][1-9]?)?", normalized) is not None


def _workflow_block_scalar_explicit_indent(value: str) -> int | None:
    normalized = _workflow_strip_comment(value.strip())
    match = re.search(r"[1-9]", normalized)
    return int(match.group()) if match is not None else None


def _discover_workflow_launches(
    root: Path, context: DiscoveryContext | None = None
) -> list[dict[str, Any]]:
    active = _context_for(root, context)
    root = active.root
    launches: list[dict[str, Any]] = []
    workflow_paths = [
        rel
        for rel in active.public_files.paths
        if PurePosixPath(rel).parent == PurePosixPath(".github/workflows")
        and PurePosixPath(rel).suffix.lower() in {".yml", ".yaml"}
    ]
    for rel in workflow_paths:
        node = active.public_files.node(rel)
        if node.kind == "symlink":
            raise InventoryError(f"{rel}: workflow paths must not use symlinks")
        if node.kind != "regular":
            raise InventoryError(f"{rel}: workflow source must be a regular file")
        jobs_indent: int | None = None
        job_indent: int | None = None
        job_property_indent: int | None = None
        steps_indent: int | None = None
        step_indent: int | None = None
        step_property_indent: int | None = None
        job: str | None = None
        step_ordinal = 0
        step_name: str | None = None
        step_has_run = False
        step_semantics: list[str] = []
        step_property_keys: set[str] = set()
        seen_launch_ids: set[str] = set()
        run_block_header_indent: int | None = None
        run_block_content_indent: int | None = None
        run_block_pending_blank_indents: list[int] = []
        run_inline_property_indent: int | None = None
        run_inline_pending_blank = False
        name_inline_property_indent: int | None = None
        name_inline_pending_blank = False
        generic_inline_property_indent: int | None = None
        generic_inline_reject_continuation = False
        generic_block_header_indent: int | None = None
        generic_block_content_indent: int | None = None
        generic_block_pending_blank_indents: list[int] = []
        job_has_steps = False
        job_inline_property_indent: int | None = None
        job_inline_reject_continuation = False
        root_inline_property_indent: int | None = None
        root_inline_reject_continuation = False
        seen_jobs_mapping = False
        seen_job_keys: set[str] = set()
        yaml_mapping_keys: dict[int, set[str]] = {}

        def finish_step() -> None:
            nonlocal step_name, step_has_run, step_semantics, step_property_indent
            nonlocal step_property_keys
            nonlocal run_block_header_indent, run_block_content_indent
            nonlocal run_block_pending_blank_indents
            nonlocal run_inline_property_indent, run_inline_pending_blank
            nonlocal name_inline_property_indent, name_inline_pending_blank
            nonlocal generic_inline_property_indent
            nonlocal generic_inline_reject_continuation
            nonlocal generic_block_header_indent, generic_block_content_indent
            nonlocal generic_block_pending_blank_indents
            if not step_has_run or job is None:
                step_name = None
                step_has_run = False
                step_semantics = []
                step_property_indent = None
                run_block_header_indent = None
                run_block_content_indent = None
                run_block_pending_blank_indents = []
                run_inline_property_indent = None
                run_inline_pending_blank = False
                name_inline_property_indent = None
                name_inline_pending_blank = False
                generic_inline_property_indent = None
                generic_inline_reject_continuation = False
                step_property_keys = set()
                return
            if "uses" in step_property_keys:
                raise InventoryError(
                    f"{rel}: workflow steps cannot contain both run and uses"
                )
            semantic = _slug(step_name) if step_name else f"step-{step_ordinal}"
            if not semantic:
                raise InventoryError(
                    f"{rel}: workflow step name must produce a non-empty identity"
                )
            launch_id = f"workflow-launch:{rel}:{job}:{semantic}"
            if launch_id in seen_launch_ids:
                raise InventoryError(
                    f"{rel}: duplicate workflow launch identity {launch_id}"
                )
            seen_launch_ids.add(launch_id)
            observation = {
                "id": launch_id,
                "category": "workflow-launch",
                "anchor": {
                    "file": rel,
                    "enclosing_function": job,
                    "symbol": step_name or "run",
                    "ordinal": step_ordinal,
                },
                "call": "run",
                "source_digest": _workflow_semantic_digest(
                    [
                        "workflow-file:" + workflow_file_digest,
                        *step_semantics,
                    ]
                ),
            }
            observation.update(REVIEWED_NEW_WORKFLOW_LAUNCH_FIELDS.get(launch_id, {}))
            launches.append(observation)
            step_name = None
            step_has_run = False
            step_semantics = []
            step_property_indent = None
            run_block_header_indent = None
            run_block_content_indent = None
            run_block_pending_blank_indents = []
            run_inline_property_indent = None
            run_inline_pending_blank = False
            name_inline_property_indent = None
            name_inline_pending_blank = False
            generic_inline_property_indent = None
            generic_inline_reject_continuation = False
            step_property_keys = set()

        def accept_property(
            key: str, value: str, line_number: int, property_indent: int
        ) -> None:
            nonlocal step_name, step_has_run, run_block_header_indent
            nonlocal run_block_content_indent
            nonlocal run_block_pending_blank_indents
            nonlocal run_inline_property_indent, run_inline_pending_blank
            nonlocal name_inline_property_indent, name_inline_pending_blank
            nonlocal generic_inline_property_indent
            nonlocal generic_inline_reject_continuation
            nonlocal generic_block_header_indent, generic_block_content_indent
            nonlocal generic_block_pending_blank_indents
            if key in step_property_keys:
                raise InventoryError(
                    f"{rel}:{line_number}: duplicate workflow step key {key!r}"
                )
            step_property_keys.add(key)
            if key == "name":
                if _workflow_block_scalar_indicator(value):
                    raise InventoryError(
                        f"{rel}:{line_number}: block scalar workflow step names are unsupported"
                    )
                step_name = _workflow_scalar(value, rel, line_number, "name")
                name_inline_property_indent = property_indent
                name_inline_pending_blank = False
            elif key == "run":
                if step_has_run:
                    raise InventoryError(
                        f"{rel}:{line_number}: duplicate run key in workflow step"
                    )
                _workflow_scalar(value, rel, line_number, "run")
                step_has_run = True
                normalized = _workflow_strip_comment(value.strip())
                if normalized.startswith(("|", ">")) and not (
                    _workflow_block_scalar_indicator(value)
                ):
                    raise InventoryError(
                        f"{rel}:{line_number}: invalid workflow block scalar indicator"
                    )
                if _workflow_block_scalar_indicator(value):
                    run_block_header_indent = property_indent
                    explicit_indent = _workflow_block_scalar_explicit_indent(value)
                    run_block_content_indent = (
                        property_indent + explicit_indent
                        if explicit_indent is not None
                        else None
                    )
                    run_inline_property_indent = None
                else:
                    run_block_header_indent = None
                    run_block_content_indent = None
                    run_inline_property_indent = property_indent
                run_block_pending_blank_indents = []
                run_inline_pending_blank = False
            else:
                normalized = _workflow_strip_comment(value.strip())
                if normalized:
                    _validate_workflow_flow_value(
                        normalized, rel, line_number, f"step {key} value"
                    )
                    if _workflow_block_scalar_indicator(normalized):
                        generic_block_header_indent = property_indent
                        explicit_indent = _workflow_block_scalar_explicit_indent(value)
                        generic_block_content_indent = (
                            property_indent + explicit_indent
                            if explicit_indent is not None
                            else None
                        )
                        generic_block_pending_blank_indents = []
                    else:
                        generic_inline_property_indent = property_indent
                        generic_inline_reject_continuation = normalized.startswith(
                            ("'", '"', "[", "{")
                        )

        if node.bytes is None:
            raise InventoryError(f"{rel}: workflow source bytes were not frozen")
        source_bytes = node.bytes
        try:
            source_text = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InventoryError(f"{rel}: workflow is not valid UTF-8") from exc
        workflow_file_digest = hashlib.sha256(source_bytes).hexdigest()
        if source_text.startswith("\ufeff"):
            raise InventoryError(f"{rel}:1: UTF-8 BOM is unsupported in workflows")
        for offset, char in enumerate(source_text):
            codepoint = ord(char)
            if (
                codepoint in {0x09, 0x0A, 0x0D, 0x85}
                or 0x20 <= codepoint <= 0x7E
                or 0xA0 <= codepoint <= 0xD7FF
                or 0xE000 <= codepoint <= 0xFFFD
                or 0x10000 <= codepoint <= 0x10FFFF
            ):
                continue
            invalid_line = source_text.count("\n", 0, offset) + 1
            raise InventoryError(
                f"{rel}:{invalid_line}: forbidden YAML control character U+{codepoint:04X}"
            )
        for line_number, line in enumerate(source_text.splitlines(), 1):
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if generic_block_header_indent is not None:
                if not stripped:
                    if generic_block_content_indent is None:
                        generic_block_pending_blank_indents.append(indent)
                    continue
                if indent > generic_block_header_indent:
                    if generic_block_content_indent is None:
                        generic_block_content_indent = indent
                        if any(
                            blank_indent > generic_block_content_indent
                            for blank_indent in generic_block_pending_blank_indents
                        ):
                            raise InventoryError(
                                f"{rel}:{line_number}: inconsistent workflow block scalar indentation"
                            )
                    if indent < generic_block_content_indent:
                        raise InventoryError(
                            f"{rel}:{line_number}: inconsistent workflow block scalar indentation"
                        )
                    continue
                generic_block_header_indent = None
                generic_block_content_indent = None
                generic_block_pending_blank_indents = []
            if run_block_header_indent is not None:
                if not stripped:
                    step_semantics.append("block:" + line)
                    if run_block_content_indent is None:
                        run_block_pending_blank_indents.append(indent)
                    continue
                if indent > run_block_header_indent:
                    if run_block_content_indent is None:
                        run_block_content_indent = indent
                        if any(
                            blank_indent > run_block_content_indent
                            for blank_indent in run_block_pending_blank_indents
                        ):
                            raise InventoryError(
                                f"{rel}:{line_number}: inconsistent workflow block scalar indentation"
                            )
                    if indent < run_block_content_indent:
                        raise InventoryError(
                            f"{rel}:{line_number}: inconsistent workflow block scalar indentation"
                        )
                    step_semantics.append("block:" + line[run_block_content_indent:])
                    continue
                run_block_header_indent = None
                run_block_content_indent = None
                run_block_pending_blank_indents = []
            leading_whitespace = line[: len(line) - len(line.lstrip())]
            if stripped.startswith("#") and "\t" not in leading_whitespace:
                continue
            if generic_inline_property_indent is not None:
                if not stripped:
                    continue
                if indent > generic_inline_property_indent and (
                    generic_inline_reject_continuation
                    or _workflow_mapping(stripped, rel, line_number) is not None
                ):
                    raise InventoryError(
                        f"{rel}:{line_number}: invalid continuation of workflow step scalar"
                    )
                if indent <= generic_inline_property_indent:
                    generic_inline_property_indent = None
                    generic_inline_reject_continuation = False
            if name_inline_property_indent is not None:
                if not stripped:
                    name_inline_pending_blank = True
                    continue
                if indent > name_inline_property_indent:
                    raise InventoryError(
                        f"{rel}:{line_number}: multiline workflow step names are unsupported"
                    )
                name_inline_property_indent = None
                name_inline_pending_blank = False
            if run_inline_property_indent is not None:
                if not stripped:
                    run_inline_pending_blank = True
                    continue
                if indent > run_inline_property_indent:
                    raise InventoryError(
                        f"{rel}:{line_number}: multiline plain workflow run values are unsupported"
                    )
                run_inline_property_indent = None
                run_inline_pending_blank = False
            if (
                line.startswith("\t")
                or line[: len(line) - len(line.lstrip())].find("\t") >= 0
            ):
                raise InventoryError(
                    f"{rel}:{line_number}: tabs are unsupported in workflow indentation"
                )
            if not stripped or stripped.startswith("#"):
                continue
            if re.match(r"(?:---|\.\.\.)(?:\s|$)", stripped):
                raise InventoryError(
                    f"{rel}:{line_number}: workflow document markers are unsupported"
                )
            if stripped.startswith(("?", ":", "!", "&", "*", "{", "[")):
                raise InventoryError(
                    f"{rel}:{line_number}: unsupported workflow mapping syntax"
                )
            if re.match(r"(?:['\"]?<<['\"]?)\s*:", stripped):
                raise InventoryError(
                    f"{rel}:{line_number}: YAML merge keys are unsupported"
                )
            mapping = _workflow_mapping(stripped, rel, line_number)
            if root_inline_property_indent is not None:
                if indent > root_inline_property_indent and (
                    root_inline_reject_continuation or mapping is not None
                ):
                    raise InventoryError(
                        f"{rel}:{line_number}: invalid continuation of workflow root scalar"
                    )
                if indent <= root_inline_property_indent:
                    root_inline_property_indent = None
                    root_inline_reject_continuation = False
            for mapping_indent in tuple(yaml_mapping_keys):
                if mapping_indent > indent:
                    yaml_mapping_keys.pop(mapping_indent)
            if stripped == "-" or stripped.startswith("- "):
                for mapping_indent in tuple(yaml_mapping_keys):
                    if mapping_indent > indent:
                        yaml_mapping_keys.pop(mapping_indent)
                sequence_value = _workflow_strip_comment(stripped[1:].strip())
                if sequence_value.startswith(("&", "*", "!")):
                    raise InventoryError(
                        f"{rel}:{line_number}: workflow anchors, aliases, and tags are unsupported"
                    )
                if sequence_value.startswith(('"', "'")):
                    _workflow_scalar(sequence_value, rel, line_number, "sequence value")
                elif sequence_value.startswith("{"):
                    _workflow_flow_mapping(sequence_value, rel, line_number)
                elif sequence_value.startswith("["):
                    _validate_workflow_flow_value(
                        sequence_value, rel, line_number, "sequence value"
                    )
                else:
                    sequence_mapping = _workflow_mapping(
                        sequence_value, rel, line_number
                    )
                    if sequence_mapping is not None:
                        after_dash = stripped[1:]
                        dash_padding = len(after_dash) - len(after_dash.lstrip(" "))
                        property_indent = indent + 1 + dash_padding
                        keys = yaml_mapping_keys.setdefault(property_indent, set())
                        if sequence_mapping[0] in keys:
                            raise InventoryError(
                                f"{rel}:{line_number}: duplicate workflow mapping key {sequence_mapping[0]!r}"
                            )
                        keys.add(sequence_mapping[0])
                        sequence_mapping_value = _workflow_strip_comment(
                            sequence_mapping[1].strip()
                        )
                        if sequence_mapping_value:
                            _validate_workflow_flow_value(
                                sequence_mapping_value,
                                rel,
                                line_number,
                                "sequence mapping value",
                            )
                            if sequence_mapping[0] not in {
                                "name",
                                "run",
                            } and _workflow_block_scalar_indicator(
                                sequence_mapping_value
                            ):
                                generic_block_header_indent = property_indent
                                explicit_indent = (
                                    _workflow_block_scalar_explicit_indent(
                                        sequence_mapping_value
                                    )
                                )
                                generic_block_content_indent = (
                                    property_indent + explicit_indent
                                    if explicit_indent is not None
                                    else None
                                )
                                generic_block_pending_blank_indents = []
            if mapping is not None:
                mapping_value = _workflow_strip_comment(mapping[1].strip())
                if mapping_value.startswith(("&", "*", "!")):
                    raise InventoryError(
                        f"{rel}:{line_number}: workflow anchors, aliases, and tags are unsupported"
                    )
                if mapping_value.startswith(('"', "'")):
                    _workflow_scalar(mapping_value, rel, line_number, "mapping value")
                elif mapping_value.startswith(("[", "{")):
                    _validate_workflow_flow_value(
                        mapping_value, rel, line_number, "mapping value"
                    )
                elif mapping_value:
                    _validate_workflow_plain_scalar_syntax(
                        mapping_value, rel, line_number, "mapping value"
                    )
                    if _workflow_block_scalar_indicator(mapping_value) and not (
                        step_indent is not None
                        and step_property_indent is not None
                        and indent == step_property_indent
                    ):
                        generic_block_header_indent = indent
                        explicit_indent = _workflow_block_scalar_explicit_indent(
                            mapping_value
                        )
                        generic_block_content_indent = (
                            indent + explicit_indent
                            if explicit_indent is not None
                            else None
                        )
                        generic_block_pending_blank_indents = []
                if (
                    jobs_indent is None
                    and indent == 0
                    and mapping[0] != "jobs"
                    and mapping_value
                    and not _workflow_block_scalar_indicator(mapping_value)
                ):
                    root_inline_property_indent = indent
                    root_inline_reject_continuation = mapping_value.startswith(
                        ("'", '"', "[", "{")
                    )
                is_special_mapping = (
                    mapping[0] == "jobs"
                    and indent == 0
                    or jobs_indent is not None
                    and (job_indent is None or indent == job_indent)
                    or step_indent is not None
                    and step_property_indent is not None
                    and indent == step_property_indent
                )
                if not is_special_mapping and not stripped.startswith("-"):
                    keys = yaml_mapping_keys.setdefault(indent, set())
                    if mapping[0] in keys:
                        raise InventoryError(
                            f"{rel}:{line_number}: duplicate workflow mapping key {mapping[0]!r}"
                        )
                    keys.add(mapping[0])
            if (
                mapping is not None
                and mapping[0] == "jobs"
                and indent == 0
                and not stripped.startswith("-")
            ):
                if mapping[1] and not mapping[1].startswith("#"):
                    raise InventoryError(
                        f"{rel}:{line_number}: jobs must use a block mapping"
                    )
                if seen_jobs_mapping:
                    raise InventoryError(f"{rel}:{line_number}: duplicate jobs mapping")
                seen_jobs_mapping = True
                seen_job_keys = set()
                jobs_indent = indent
                continue
            if jobs_indent is None:
                if mapping is None and indent == 0:
                    raise InventoryError(
                        f"{rel}:{line_number}: workflow top level must use a mapping"
                    )
                continue
            if indent <= jobs_indent:
                if steps_indent is not None and step_indent is None:
                    raise InventoryError(
                        f"{rel}:{line_number}: steps must use a non-empty block sequence"
                    )
                finish_step()
                jobs_indent = None
                job_indent = None
                job_property_indent = None
                steps_indent = None
                step_indent = None
                step_property_indent = None
                job = None
                job_inline_property_indent = None
                job_inline_reject_continuation = False
                continue

            if job_indent is None or (
                indent == job_indent and not stripped.startswith("-")
            ):
                candidate = _workflow_mapping(stripped, rel, line_number)
                if candidate is None or (
                    candidate[1] and not candidate[1].startswith("#")
                ):
                    raise InventoryError(
                        f"{rel}:{line_number}: workflow job must use a block mapping"
                    )
                if steps_indent is not None and step_indent is None:
                    raise InventoryError(
                        f"{rel}:{line_number}: steps must use a non-empty block sequence"
                    )
                finish_step()
                if job_indent is None:
                    job_indent = indent
                job = candidate[0]
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", job) is None:
                    raise InventoryError(
                        f"{rel}:{line_number}: invalid GitHub Actions job identifier {job!r}"
                    )
                if job in seen_job_keys:
                    raise InventoryError(
                        f"{rel}:{line_number}: duplicate workflow job key {job!r}"
                    )
                seen_job_keys.add(job)
                job_property_indent = None
                steps_indent = None
                step_indent = None
                step_property_indent = None
                step_ordinal = 0
                job_has_steps = False
                job_inline_property_indent = None
                job_inline_reject_continuation = False
                continue
            if indent < job_indent:
                raise InventoryError(
                    f"{rel}:{line_number}: inconsistent workflow job indentation"
                )
            if job_inline_property_indent is not None:
                if indent > job_inline_property_indent and (
                    job_inline_reject_continuation or mapping is not None
                ):
                    raise InventoryError(
                        f"{rel}:{line_number}: invalid continuation of workflow job scalar"
                    )
                if indent <= job_inline_property_indent:
                    job_inline_property_indent = None
                    job_inline_reject_continuation = False

            if job_property_indent is None:
                direct_property = _workflow_mapping(stripped, rel, line_number)
                if direct_property is not None:
                    job_property_indent = indent
            if steps_indent is None:
                if indent == job_property_indent:
                    direct_property = _workflow_mapping(stripped, rel, line_number)
                    if (
                        direct_property is not None
                        and direct_property[0] != "steps"
                        and _workflow_strip_comment(direct_property[1].strip())
                        and not _workflow_block_scalar_indicator(direct_property[1])
                    ):
                        normalized_job_value = _workflow_strip_comment(
                            direct_property[1].strip()
                        )
                        job_inline_property_indent = indent
                        job_inline_reject_continuation = (
                            normalized_job_value.startswith(("'", '"', "[", "{"))
                        )
                    if direct_property is not None and direct_property[0] == "steps":
                        if job_has_steps:
                            raise InventoryError(
                                f"{rel}:{line_number}: duplicate steps mapping"
                            )
                        if direct_property[1] and not direct_property[1].startswith(
                            "#"
                        ):
                            raise InventoryError(
                                f"{rel}:{line_number}: steps must use a block sequence"
                            )
                        steps_indent = indent
                        job_has_steps = True
                continue
            sequence_line = stripped == "-" or stripped.startswith("- ")
            if indent < steps_indent or (indent == steps_indent and not sequence_line):
                if step_indent is None:
                    raise InventoryError(
                        f"{rel}:{line_number}: steps must use a non-empty block sequence"
                    )
                finish_step()
                duplicate_steps = _workflow_mapping(stripped, rel, line_number)
                if duplicate_steps is not None and duplicate_steps[0] == "steps":
                    raise InventoryError(
                        f"{rel}:{line_number}: duplicate steps mapping"
                    )
                steps_indent = None
                step_indent = None
                step_property_indent = None
                continue
            if step_indent is None and indent > steps_indent and not sequence_line:
                raise InventoryError(
                    f"{rel}:{line_number}: steps must use a block sequence"
                )

            if sequence_line:
                if step_indent is None:
                    step_indent = indent
                if indent == step_indent:
                    finish_step()
                    step_ordinal += 1
                    structural_line = _workflow_strip_comment(stripped)
                    step_semantics = [structural_line or "-"]
                    after_dash = stripped[1:]
                    dash_padding = len(after_dash) - len(after_dash.lstrip(" "))
                    inline = _workflow_strip_comment(after_dash.strip())
                    if not inline:
                        continue
                    step_property_indent = step_indent + 1 + dash_padding
                    if inline.startswith("{"):
                        if not inline.endswith("}"):
                            raise InventoryError(
                                f"{rel}:{line_number}: unsupported multiline workflow flow mapping"
                            )
                        for key, value in _workflow_flow_mapping(
                            inline, rel, line_number
                        ):
                            if _workflow_block_scalar_indicator(value):
                                raise InventoryError(
                                    f"{rel}:{line_number}: block scalars are unsupported in workflow flow mappings"
                                )
                            accept_property(key, value, line_number, step_indent)
                        continue
                    inline_mapping = _workflow_mapping(inline, rel, line_number)
                    if inline_mapping is None:
                        raise InventoryError(
                            f"{rel}:{line_number}: workflow step must use a mapping"
                        )
                    accept_property(*inline_mapping, line_number, step_property_indent)
                    continue
                if indent < step_indent:
                    raise InventoryError(
                        f"{rel}:{line_number}: inconsistent workflow step indentation"
                    )

            if (
                step_indent is not None
                and step_property_indent is not None
                and step_indent < indent < step_property_indent
            ):
                raise InventoryError(
                    f"{rel}:{line_number}: inconsistent workflow step property indentation"
                )
            if step_indent is not None and indent > step_indent:
                step_semantics.append(stripped)
            if step_indent is not None and step_property_indent is None:
                step_property_indent = indent
            if step_indent is not None and indent == step_property_indent:
                property_mapping = _workflow_mapping(stripped, rel, line_number)
                if property_mapping is None:
                    if re.search(r"(?:[\"']?run[\"']?)\s*:", stripped):
                        raise InventoryError(
                            f"{rel}:{line_number}: workflow run key cannot be normalized"
                        )
                    raise InventoryError(
                        f"{rel}:{line_number}: workflow step property cannot be normalized"
                    )
                accept_property(*property_mapping, line_number, indent)
        if run_block_header_indent is not None:
            step_semantics.append(
                "block:eof-newline="
                + ("1" if source_text.endswith(("\n", "\r")) else "0")
            )
        if steps_indent is not None and step_indent is None:
            raise InventoryError(
                f"{rel}:{len(source_text.splitlines()) or 1}: steps must use a non-empty block sequence"
            )
        finish_step()
    return launches


def _discover_generator_targets(
    root: Path, context: DiscoveryContext | None = None
) -> list[dict[str, Any]]:
    active = _context_for(root, context)
    root = active.root
    targets: list[dict[str, Any]] = []
    specs = (
        (
            "tools/generate_compat_headers.zig",
            "writeGeneratedFile",
            r'writeGeneratedFile\s*\([\s\S]*?"((?:include|docs)/[^"\n]+)"',
        ),
        (
            "tools/generate_kernel_coverage.zig",
            "main",
            r'path\.join\([^;]+"(docs/kernel_coverage\.json)"',
        ),
    )
    for rel, function, pattern in specs:
        if rel not in active.public_files.path_set:
            raise InventoryError(
                f"generator source is absent from the public universe: {rel}"
            )
        node = active.public_files.node(rel)
        if node.kind != "regular":
            raise InventoryError(
                f"generator source must be a non-symlink regular file: {rel}"
            )
        text = _frozen_regular_text(active, rel, "generator source")
        for ordinal, match in enumerate(re.finditer(pattern, text), 1):
            path = match.group(1)
            targets.append(
                {
                    "id": f"generated-target:{path}",
                    "path": path,
                    "anchor": {
                        "file": rel,
                        "enclosing_function": function,
                        "symbol": path,
                        "ordinal": ordinal,
                    },
                }
            )
    return targets


def discover(
    root: Path,
    inventory: dict[str, Any],
    context: DiscoveryContext | None = None,
) -> dict[str, list[dict[str, Any]]]:
    del inventory
    active = _context_for(root, context)
    root = active.root
    build_roots = active.build_roots
    build = [
        item for rel in build_roots for item in _discover_build_root(root, rel, active)
    ]
    build.extend(
        {
            "id": f"step:{rel}:build:{name}",
            "category": "step",
            "anchor": {
                "file": rel,
                "enclosing_function": "build",
                "symbol": name,
                "ordinal": ordinal,
            },
            "call": "implicit",
            "source_digest": _semantic_digest([f"zig-standard-step:{name}"]),
        }
        for rel in build_roots
        for ordinal, name in enumerate(("install", "uninstall"), 1)
    )
    return {
        "build_observations": build,
        "python_launches": _discover_python_launches(root, active),
        "workflow_launches": _discover_workflow_launches(root, active),
        "generator_targets": _discover_generator_targets(root, active),
    }


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _check_unique_ids(
    items: list[dict[str, Any]], section: str, errors: list[str]
) -> None:
    ids = [item.get("id") for item in items]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        errors.append(f"{section}: every entry requires a non-empty id")
    for identifier, count in Counter(
        identifier for identifier in ids if isinstance(identifier, str) and identifier
    ).items():
        if count > 1:
            errors.append(f"{section}: duplicate id {identifier!r}")


def _compare_ids(
    section: str,
    expected: list[dict[str, Any]],
    observed: list[dict[str, Any]],
    errors: list[str],
) -> None:
    expected_ids = {item.get("id") for item in expected}
    observed_ids = {item.get("id") for item in observed}
    for identifier, count in Counter(item.get("id") for item in observed).items():
        if count > 1:
            errors.append(f"{section}: duplicate source occurrence {identifier}")
    for identifier in sorted(observed_ids - expected_ids):
        errors.append(f"{section}: unlisted source occurrence {identifier}")
    for identifier in sorted(expected_ids - observed_ids):
        errors.append(
            f"{section}: inventory entry has no source occurrence {identifier}"
        )
    expected_by_id = {item.get("id"): item for item in expected}
    for item in observed:
        identifier = item.get("id")
        recorded = expected_by_id.get(identifier)
        if recorded is None:
            continue
        for field, observed_value in item.items():
            if recorded.get(field) != observed_value:
                errors.append(
                    f"{section}: source field {field} changed for {identifier}"
                )


def _validate_level2_width_and_windows_artifact_contract(
    inventory: dict[str, Any], errors: list[str]
) -> None:
    observations = {item.get("id"): item for item in inventory["build_observations"]}
    exact_template_ids = (
        LEVEL2_WIDTH_DEFAULT_ARTIFACT_COMPILE_ID,
        LEVEL2_WIDTH_DEFAULT_ARTIFACT_LAUNCH_ID,
        LEVEL2_WIDTH_DEFAULT_ARTIFACT_STEP_ID,
        LEVEL2_WIDTH_DEFAULT_ARTIFACT_LINK_ID,
        LEVEL2_WIDTH_ENABLED_ARTIFACT_COMPILE_ID,
        LEVEL2_WIDTH_ENABLED_ARTIFACT_LAUNCH_ID,
        LEVEL2_WIDTH_ENABLED_ARTIFACT_BUILD_STEP_ID,
        LEVEL2_WIDTH_ENABLED_ARTIFACT_RUN_STEP_ID,
        LEVEL2_WIDTH_ENABLED_ARTIFACT_UNSUPPORTED_STEP_ID,
        LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID,
        INSTALL_DYNAMIC_LIBRARY_ID,
        INSTALL_STATIC_LIBRARY_ID,
        INSTALL_LIBRARIES_STEP_ID,
    )
    for identifier in exact_template_ids:
        observed = observations.get(identifier)
        if observed is None:
            continue
        expected = _new_test_inventory_observation(identifier, inventory)
        for field, value in expected.items():
            _require(
                observed.get(field) == value,
                f"{identifier}: required {field} changed",
                errors,
            )

    for identifier in WINDOWS_PYTHON_TOOLING_FIXTURE_COMPILE_SOURCES:
        observed = observations.get(identifier)
        if observed is None:
            continue
        for field, value in _reviewed_observation_refresh_fields(identifier).items():
            _require(
                observed.get(field) == value,
                f"{identifier}: Windows tooling fixture {field} changed",
                errors,
            )

    for identifier in (
        "compile:build.zig:build:lib",
        "compile:build.zig:build:static_lib",
        *sorted(WINDOWS_EXCLUDED_DEFAULT_EXECUTABLE_INSTALL_IDS),
    ):
        observed = observations.get(identifier)
        if observed is None:
            continue
        for field, value in _reviewed_observation_refresh_fields(identifier).items():
            _require(
                observed.get(field) == value,
                f"{identifier}: reviewed Windows artifact {field} changed",
                errors,
            )

    dynamic = observations.get("compile:build.zig:build:lib", {})
    static = observations.get("compile:build.zig:build:static_lib", {})
    dynamic_windows = dynamic.get("install_destinations_by_target", {}).get(
        "windows", {}
    )
    static_windows = static.get("install_destinations_by_target", {}).get("windows", {})
    _require(
        {
            dynamic_windows.get("primary"),
            dynamic_windows.get("import_library"),
            static_windows.get("primary"),
        }
        == {
            "zig-out/bin/zynum_blas.dll",
            "zig-out/lib/zynum_blas.lib",
            "zig-out/lib/static/zynum_blas.lib",
        },
        "Windows dynamic, import, and static library destinations must be exact and collision-free",
        errors,
    )


def _validate_level2_width_stub_contract(
    context: DiscoveryContext, errors: list[str]
) -> None:
    try:
        source = _frozen_regular_text(
            context,
            LEVEL2_WIDTH_STUB_ROOT_PATH,
            "Level 2 width disabled object root",
        )
    except InventoryError as exc:
        errors.append(f"Level 2 width disabled object root is invalid: {exc}")
        return

    expected = (
        'constobject_format_sections=@import("kernels/isolated/object_format_sections.zig");'
        'constabi=@import("kernels/isolated/x86_64_level2_width_abi.zig");'
        "varenabled:u8linksection(object_format_sections.writable_data)=0;"
        "fnexecute(_:*abi.Request)callconv(.c)u8{return0;}"
        "comptime{"
        '@export(&enabled,.{.name="zynum_internal_x86_64_level2_width_enabled",.visibility=.hidden,});'
        '@export(&execute,.{.name="zynum_internal_x86_64_level2_width_execute",.visibility=.hidden,});'
        "}"
    )
    _require(
        _compact_zig_contract(source) == expected,
        "Level 2 width disabled object root must remain the exact ABI-only byte-zero rejector",
        errors,
    )


def _validate(
    root: Path,
    inventory_path: Path,
    *,
    _context: DiscoveryContext | None = None,
    _inventory_bytes: bytes | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        context = (
            _make_discovery_context(root, inventory_path)
            if _context is None
            else _context_for(root, _context)
        )
    except (InventoryError, OSError, RecursionError) as exc:
        return [f"public-file discovery failed closed: {exc}"]
    root = context.root
    try:
        if _inventory_bytes is None:
            if context.inventory_node is None or context.inventory_node.bytes is None:
                raise InventoryError("inventory bytes were not frozen")
            inventory_bytes = context.inventory_node.bytes
        else:
            inventory_bytes = _inventory_bytes
        if len(inventory_bytes) > SOURCE_REFRESH_MAX_BYTES:
            raise InventoryError(f"inventory exceeds {SOURCE_REFRESH_MAX_BYTES} bytes")
        inventory = _strict_json_loads(inventory_bytes.decode("utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        return [f"cannot read inventory: {exc}"]
    if not isinstance(inventory, dict):
        return ["inventory root must be an object"]
    structure_error = _json_structure_error(inventory)
    if structure_error is not None:
        return [structure_error]
    source_projection_error = _reviewed_source_projection_error(inventory)
    if source_projection_error is not None:
        errors.append(source_projection_error)
    _require(
        set(inventory) == INVENTORY_TOP_LEVEL_KEYS,
        "inventory top-level keys must match the schema exactly",
        errors,
    )
    _require(
        inventory.get("schema_version") == SCHEMA_VERSION,
        "schema_version must be 3",
        errors,
    )
    _require(
        inventory.get("schema_id") == "zynum-build-inventory-v3",
        "schema_id must be zynum-build-inventory-v3",
        errors,
    )
    _require(
        inventory.get("scope") == INVENTORY_SCOPE,
        "scope must match the schema contract exactly",
        errors,
    )
    _require(
        inventory.get("owner_vocabulary") == sorted(OWNER_VOCABULARY),
        "owner_vocabulary must match the schema vocabulary exactly",
        errors,
    )
    observed_build_roots = list(context.build_roots)
    build_roots = inventory.get("build_roots")
    _require(
        build_roots == observed_build_roots,
        "build_roots must exactly match independently observed safe build roots",
        errors,
    )
    observed_build_manifests = [dict(row) for row in context.build_manifests]
    build_manifests = inventory.get("build_manifests")
    _require(
        isinstance(build_manifests, list),
        "build_manifests must be an array",
        errors,
    )
    if isinstance(build_manifests, list):
        _require(
            all(isinstance(row, dict) for row in build_manifests),
            "build_manifests: every entry must be an object",
            errors,
        )
        if all(isinstance(row, dict) for row in build_manifests):
            _check_unique_ids(build_manifests, "build_manifests", errors)
            for row in build_manifests:
                _require(
                    set(row) == {"id", "path", "build_root", "content_sha256"},
                    "build_manifests: every row must match the schema exactly",
                    errors,
                )
            _require(
                build_manifests == observed_build_manifests,
                "build_manifests must exactly match independently observed safe manifests",
                errors,
            )
    root_digests = inventory.get("build_root_digests")
    _require(
        isinstance(root_digests, dict), "build_root_digests must be an object", errors
    )
    if isinstance(root_digests, dict) and build_roots == observed_build_roots:
        _require(
            set(root_digests) == set(observed_build_roots),
            "build_root_digests must cover independently observed build roots exactly",
            errors,
        )
        for rel in observed_build_roots:
            actual_digest = context.public_files.node(rel).sha256
            _require(
                root_digests.get(rel) == actual_digest,
                f"build_root_digests: full source changed for {rel}",
                errors,
            )
    sections_are_structural = True
    for section in (
        "option_surfaces",
        "build_observations",
        "python_launches",
        "workflow_launches",
        "generator_targets",
        "repository_file_classifications",
        "derived_candidates",
        "current_gaps",
    ):
        value = inventory.get(section)
        _require(isinstance(value, list), f"{section} must be an array", errors)
        if isinstance(value, list):
            if all(isinstance(item, dict) for item in value):
                if section == "repository_file_classifications":
                    paths = [item.get("path") for item in value]
                    if len(paths) != len(set(paths)):
                        errors.append("repository_file_classifications: duplicate path")
                        sections_are_structural = False
                else:
                    _check_unique_ids(value, section, errors)
                    if any(
                        not isinstance(item.get("id"), str) or not item.get("id")
                        for item in value
                    ):
                        sections_are_structural = False
                if section in {
                    "generator_targets",
                    "repository_file_classifications",
                    "derived_candidates",
                } and any(
                    not isinstance(item.get("path"), str) or not item.get("path")
                    for item in value
                ):
                    errors.append(
                        f"{section}: every entry requires a non-empty string path"
                    )
                    sections_are_structural = False
            else:
                errors.append(f"{section}: every entry must be an object")
                sections_are_structural = False
        else:
            sections_are_structural = False
    workflow_source_digests = inventory.get("workflow_source_digests")
    _require(
        isinstance(workflow_source_digests, dict),
        "workflow_source_digests must be an object",
        errors,
    )
    if not isinstance(workflow_source_digests, dict):
        sections_are_structural = False
    conditional_link_guard_digests = inventory.get("conditional_link_guard_digests")
    _require(
        isinstance(conditional_link_guard_digests, dict),
        "conditional_link_guard_digests must be an object",
        errors,
    )
    if not isinstance(conditional_link_guard_digests, dict):
        sections_are_structural = False
    classification_digest = inventory.get("repository_file_classifications_digest")
    _require(
        isinstance(classification_digest, str) and bool(classification_digest),
        "repository_file_classifications_digest must be a non-empty string",
        errors,
    )
    if (
        build_roots != observed_build_roots
        or build_manifests != observed_build_manifests
        or not isinstance(root_digests, dict)
    ):
        sections_are_structural = False
    if not sections_are_structural:
        return errors
    _validate_level2_width_stub_contract(context, errors)
    _validate_level2_width_and_windows_artifact_contract(inventory, errors)
    for section, expected_digest in REQUIRED_SECTION_FACT_DIGESTS.items():
        if context.public_files.mode == "archive" and section == "derived_candidates":
            continue
        _require(
            _json_fact_digest(inventory[section]) == expected_digest,
            f"{section} reviewed fact set changed",
            errors,
        )
    root_standard_names = {"target", "cpu", "ofmt", "dynamic-linker", "release"}
    example_standard_names = {"target", "cpu", "ofmt", "dynamic-linker", "optimize"}
    project_names = {
        "test-optimize",
        "host-tool-smoke",
        "level1-fixed-candidates",
        "level1-sve-candidates",
        "level2-fixed-candidates",
        "level2-width-candidates",
        "structured-object-candidates",
        "structured-object-baseline",
        "level2-compact-triangular-baseline",
        "compat-headers",
        "bench-openblas",
        "bench-accelerate",
        "bench-mkl",
        "bench-aocl-blis",
    }
    surfaces_by_root = {
        rel: {
            item.get("name")
            for item in inventory["option_surfaces"]
            if item.get("build_root") == rel
        }
        for rel in inventory["build_roots"]
    }
    _require(
        surfaces_by_root.get("build.zig") == root_standard_names | project_names,
        "build.zig option surfaces must contain exactly 19 standard/project surfaces",
        errors,
    )
    _require(
        surfaces_by_root.get("examples/zig/build.zig") == example_standard_names,
        "example option surfaces must contain target/cpu/ofmt/dynamic-linker/optimize exactly",
        errors,
    )
    option_observation_ids = {
        item["id"]
        for item in inventory["build_observations"]
        if item.get("category") == "option"
    }
    try:
        source_option_semantics = _discover_option_surface_semantics(root, context)
    except (InventoryError, OSError, UnicodeError) as exc:
        errors.append(f"source discovery failed closed: {exc}")
        return errors
    release_resolution = {
        "build.zig": "false resolves to Debug; true or an explicit release request resolves through preferred ReleaseFast",
        "examples/zig/build.zig": "false resolves to Debug; a release request uses Zig standard resolution",
    }
    for item in inventory["option_surfaces"]:
        for field in (
            "build_root",
            "name",
            "option_kind",
            "type",
            "default",
            "description",
            "consumers",
            "role",
            "conflict",
            "precedence",
            "source_observation",
        ):
            _require(
                field in item and item[field] not in (None, "", []),
                f"{item['id']}: missing option field {field}",
                errors,
            )
        _require(
            item.get("source_observation") in option_observation_ids,
            f"{item['id']}: invalid source_observation",
            errors,
        )
        observed_semantics = source_option_semantics.get(
            (item.get("build_root"), item.get("name"))
        )
        _require(
            observed_semantics is not None,
            f"{item['id']}: option surface has no source semantics",
            errors,
        )
        if observed_semantics is not None:
            for field in ("type", "default", "description"):
                _require(
                    item.get(field) == observed_semantics[field],
                    f"{item['id']}: {field} does not match source",
                    errors,
                )
        if item.get("name") == "release":
            _require(
                item.get("resolution_note")
                == release_resolution.get(item.get("build_root")),
                f"{item['id']}: incorrect release resolution note",
                errors,
            )
        if item.get("option_kind") == "standard":
            _require(
                isinstance(item.get("consumer_partition"), dict),
                f"{item['id']}: missing exact consumer_partition",
                errors,
            )
            build_root = item.get("build_root")
            name = item.get("name")
            if build_root == "build.zig" and name in {
                "target",
                "cpu",
                "ofmt",
                "dynamic-linker",
            }:
                expected_partition = {
                    "included": [
                        "requested-target libraries",
                        "requested-target correctness tests",
                        "requested-target benchmark executables",
                        "requested-target probes",
                    ],
                    "excluded": [
                        "host generate_compat_headers tool",
                        "host generate_kernel_coverage tool",
                    ],
                }
            elif build_root == "build.zig" and name == "release":
                expected_partition = {
                    "included": [
                        "normal requested-target libraries",
                        "requested-target benchmark executables",
                        "requested-target probes",
                    ],
                    "excluded": [
                        "correctness test artifacts controlled by test-optimize",
                        "host generate_compat_headers tool fixed at ReleaseSafe",
                        "host generate_kernel_coverage tool fixed at Debug",
                    ],
                }
            elif build_root == "examples/zig/build.zig" and name == "optimize":
                expected_partition = {
                    "included": [
                        "example executable optimization",
                        "root dependency optimize forwarding attempt",
                    ],
                    "excluded": [
                        "no host compile artifact is declared by the example build"
                    ],
                }
            else:
                expected_partition = {
                    "included": [
                        "example executable requested-target configuration",
                        "root dependency requested-target request",
                    ],
                    "excluded": [
                        "no host compile artifact is declared by the example build"
                    ],
                }
            _require(
                item.get("consumer_partition") == expected_partition,
                f"{item['id']}: incorrect standard option consumer partition",
                errors,
            )
        if item.get("name") == "optimize":
            _require(
                item.get("value_domain")
                == ["Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"],
                f"{item['id']}: incorrect optimize value_domain",
                errors,
            )
    for item in inventory["build_observations"]:
        _require(
            item.get("owner") in OWNER_VOCABULARY,
            f"{item['id']}: invalid owner",
            errors,
        )
        anchor = item.get("anchor")
        _require(
            isinstance(anchor, dict)
            and all(
                key in anchor
                for key in ("file", "enclosing_function", "symbol", "ordinal")
            ),
            f"{item['id']}: incomplete stable anchor",
            errors,
        )
        if item.get("category") == "compile":
            for field in (
                "artifact_kind",
                "output_name",
                "root_source",
                "linkage",
                "compile_for",
                "execute_on",
                "optimize_source",
                "condition",
                "produced_outputs",
                "install_destinations",
            ):
                _require(
                    field in item,
                    f"{item['id']}: missing compile field {field}",
                    errors,
                )
            _require(
                item.get("compile_for") in PLATFORMS - {"not-executable"},
                f"{item['id']}: invalid compile_for",
                errors,
            )
            _require(
                item.get("execute_on") in PLATFORMS,
                f"{item['id']}: invalid execute_on",
                errors,
            )
            _require(
                item.get("artifact_kind") in {"library", "executable", "test"},
                f"{item['id']}: invalid artifact_kind",
                errors,
            )
            _require(
                isinstance(item.get("root_source"), list) and bool(item["root_source"]),
                f"{item['id']}: root_source must be non-empty",
                errors,
            )
            _require(
                isinstance(item.get("produced_outputs"), list),
                f"{item['id']}: produced_outputs must be an array",
                errors,
            )
            _require(
                bool(item.get("produced_outputs")),
                f"{item['id']}: produced_outputs must be non-empty",
                errors,
            )
            _require(
                isinstance(item.get("install_destinations"), list),
                f"{item['id']}: install_destinations must be an array",
                errors,
            )
            if item.get("artifact_kind") == "library":
                output_map = item.get("produced_outputs_by_target")
                _require(
                    isinstance(output_map, dict)
                    and set(output_map) == {"elf", "macho", "windows"},
                    f"{item['id']}: incomplete target-conditioned output map",
                    errors,
                )
                install_map = item.get("install_destinations_by_target")
                _require(
                    isinstance(install_map, dict)
                    and set(install_map) == {"elf", "macho", "windows"},
                    f"{item['id']}: incomplete target-conditioned install map",
                    errors,
                )
                name = item.get("output_name")
                if item.get("linkage") == "dynamic":
                    expected_outputs = {
                        "elf": {"primary": f"lib{name}.so"},
                        "macho": {"primary": f"lib{name}.dylib"},
                        "windows": {
                            "primary": f"{name}.dll",
                            "import_library": f"{name}.lib",
                            "debug_side_artifact": f"{name}.pdb when debug information is emitted",
                        },
                    }
                else:
                    expected_outputs = {
                        "elf": {"primary": f"lib{name}.a"},
                        "macho": {"primary": f"lib{name}.a"},
                        "windows": {"primary": f"{name}.lib"},
                    }
                _require(
                    output_map == expected_outputs,
                    f"{item['id']}: target-conditioned outputs do not match artifact kind",
                    errors,
                )
                if item.get("isolated_library"):
                    _require(
                        item.get("install_destinations") == [],
                        f"{item['id']}: isolated libraries are not installed",
                        errors,
                    )
                    _require(
                        install_map == {"elf": {}, "macho": {}, "windows": {}},
                        f"{item['id']}: isolated install map must be empty",
                        errors,
                    )
        if item.get("category") == "step":
            for field in (
                "description",
                "direct_dependencies",
                "aggregate_test_membership",
                "aggregate_condition",
                "intentional_orphan",
                "orphan_reason",
                "step_role",
            ):
                _require(
                    field in item, f"{item['id']}: missing step field {field}", errors
                )
            _require(
                isinstance(item.get("direct_dependencies"), list),
                f"{item['id']}: direct_dependencies must be an array",
                errors,
            )
        if item.get("category") == "install":
            for field in ("producer", "source", "destination", "condition"):
                _require(
                    field in item,
                    f"{item['id']}: missing install field {field}",
                    errors,
                )
            _require(
                "anonymous" not in item["id"],
                f"{item['id']}: anonymous install identity is forbidden",
                errors,
            )
            _require(
                isinstance(item.get("destination"), list) and bool(item["destination"]),
                f"{item['id']}: install destination must be non-empty",
                errors,
            )
        if item.get("category") == "link":
            for field in ("consumer", "provider", "condition"):
                _require(
                    isinstance(item.get(field), str) and bool(item[field]),
                    f"{item['id']}: missing link field {field}",
                    errors,
                )
            semantic_edge = item["id"].rsplit(":", 1)[-1]
            if "<-" in semantic_edge:
                consumer, provider = semantic_edge.split("<-", 1)
                _require(
                    item.get("consumer") == consumer,
                    f"{item['id']}: wrong link consumer",
                    errors,
                )
                _require(
                    item.get("provider") == provider,
                    f"{item['id']}: wrong link provider",
                    errors,
                )
        if item.get("category") == "launch":
            for field in (
                "command_shape",
                "compile_for",
                "execute_on",
                "argv_shape",
                "cwd_shape",
                "launch_class",
                "source_artifact",
            ):
                _require(
                    field in item, f"{item['id']}: missing launch field {field}", errors
                )
            _require(
                item.get("detail_status") == "process-lifecycle-out-of-scope",
                f"{item['id']}: launch detail_status must describe the process-lifecycle scope",
                errors,
            )
    _validate_test_inventory_factory_contract(inventory["build_observations"], errors)
    _validate_python_tooling_test_contract(inventory["build_observations"], errors)
    isolated = [
        item for item in inventory["build_observations"] if item.get("isolated_library")
    ]
    _require(
        len(isolated) == 8,
        f"expected exactly 8 isolated libraries, found {len(isolated)}",
        errors,
    )
    isolated_optimization = {
        "stride2_isolated_library": "optimize",
        "compact_triangular_isolated_library": "optimize",
        "level2_width_isolated_library": "optimize",
        "structured_isolated_library": "optimize",
        "stride2_isolated_test_library": "test-optimize",
        "compact_triangular_isolated_test_library": "test-optimize",
        "level2_width_isolated_test_library": "test-optimize",
        "structured_isolated_test_library": "test-optimize",
    }
    for item in isolated:
        symbol = item["anchor"]["symbol"]
        _require(
            item.get("optimize_source") == isolated_optimization.get(symbol),
            f"{item['id']}: incorrect isolated optimize_source",
            errors,
        )
    parity_step = "step:build.zig:build:test-abi-artifact-parity-verifier"
    for item in (
        entry
        for entry in inventory["build_observations"]
        if entry.get("category") == "step"
    ):
        _require(
            item.get("intentional_orphan") is False,
            f"{item['id']}: no reviewed build step is an intentional orphan",
            errors,
        )
    steps_by_id = {
        item["id"]: item
        for item in inventory["build_observations"]
        if item.get("category") == "step"
    }
    _require(
        steps_by_id.get(parity_step, {}).get("aggregate_test_membership")
        == "conditional-test-coverage"
        and steps_by_id.get(parity_step, {}).get("aggregate_condition")
        == "host-tool-smoke is true via abi_baseline_observer_tests unittest discovery"
        and steps_by_id.get(parity_step, {}).get("orphan_reason")
        == "dedicated step node has no direct aggregate edge; its test file is nevertheless covered by the aggregate unittest discovery launch"
        and steps_by_id.get(parity_step, {}).get("step_role") == "focused-validation",
        "ABI parity step must distinguish missing direct step edge from aggregate test coverage",
        errors,
    )
    root_install_dependencies = {
        *(
            f"install:build.zig:build:{name}"
            for name in (
                "lib",
                "static_lib",
                "bench",
                "gemm_sweep",
                "vector_matrix_sweep",
                "level1_probe",
                "dcopy_probe",
            )
        ),
        *(
            f"install:build.zig:build:{path}"
            for path in (
                "include/zynum/blas/cblas.h",
                "include/zynum/blas/blas.h",
                "include/zynum/blas/blas.f90",
                "include/zynum/blas/abi_manifest.json",
                "pkgconfig/zynum_blas.pc",
            )
        ),
    }
    actual_root_install_entries = steps_by_id["step:build.zig:build:install"].get(
        "direct_dependencies", []
    )
    actual_root_install = {
        dependency.get("id") for dependency in actual_root_install_entries
    }
    expected_root_conditions = {
        identifier: (
            "compat-headers is true"
            if "/" in identifier.rsplit(":", 1)[-1]
            else "always"
        )
        for identifier in root_install_dependencies
    }
    _require(
        actual_root_install == root_install_dependencies,
        "root implicit install dependencies are not the exact 7 artifacts plus 5 conditional files",
        errors,
    )
    _require(
        {
            entry.get("id"): entry.get("condition")
            for entry in actual_root_install_entries
        }
        == expected_root_conditions,
        "root implicit install dependency conditions are incorrect",
        errors,
    )
    actual_example_install = steps_by_id[
        "step:examples/zig/build.zig:build:install"
    ].get("direct_dependencies")
    _require(
        actual_example_install
        == [{"id": "install:examples/zig/build.zig:build:exe", "condition": "always"}],
        "example install must depend only on its executable install",
        errors,
    )
    structured_dependencies = steps_by_id[
        "step:build.zig:build:test-structured-object"
    ].get("direct_dependencies", [])
    _require(
        structured_dependencies
        == [
            {
                "id": "launch:build.zig:build:run",
                "condition": "requested target architecture is x86_64",
            }
        ],
        "test-structured-object launch dependency must be x86_64-conditioned",
        errors,
    )
    observations_by_id = {item["id"]: item for item in inventory["build_observations"]}
    for identifier, condition in WINDOWS_PYTHON_TOOLING_INSTALL_REACHABILITY.items():
        _require(
            observations_by_id[identifier].get("condition") == condition,
            f"{identifier}: incorrect Windows Python tooling install reachability",
            errors,
        )
    for section in ("python_launches", "workflow_launches"):
        for item in inventory[section]:
            _require(
                item.get("owner") in OWNER_VOCABULARY,
                f"{item['id']}: invalid owner",
                errors,
            )
            _require(
                item.get("detail_status") == "process-lifecycle-out-of-scope",
                f"{item['id']}: detail_status must describe the process-lifecycle scope",
                errors,
            )
            for field in ("compile_for", "execute_on", "cwd_shape", "launch_class"):
                _require(
                    field in item and item[field] not in (None, ""),
                    f"{item['id']}: missing launch field {field}",
                    errors,
                )
            if section == "python_launches":
                _require(
                    isinstance(item.get("call_semantics_digest"), str)
                    and bool(item["call_semantics_digest"]),
                    f"{item['id']}: missing call_semantics_digest",
                    errors,
                )
    observations_by_id = {item["id"]: item for item in inventory["build_observations"]}
    launches_by_id = {item["id"]: item for item in inventory["python_launches"]}
    for identifier in (
        TEST_INVENTORY_RUNNER_COMPILE_PYTHON_LAUNCH_ID,
        TEST_INVENTORY_RUNNER_EXECUTE_PYTHON_LAUNCH_ID,
        TEST_INVENTORY_RUNNER_RACE_PYTHON_LAUNCH_ID,
    ):
        reviewed_launch = _new_test_inventory_python_launch(identifier)
        observed_launch = launches_by_id.get(identifier, {})
        for field, value in reviewed_launch.items():
            _require(
                observed_launch.get(field) == value,
                f"{identifier}: reviewed test-inventory runner {field} changed",
                errors,
            )
    payload_identity_fields = {"payload_bindings", "payload_artifact_id"}
    observed_controller_ids = {
        item["id"]
        for item in inventory["python_launches"]
        if payload_identity_fields & set(item)
    }
    _require(
        observed_controller_ids == set(PAYLOAD_CONTROLLER_LINKS),
        "requested-target payload controller identities must match the complete reviewed set",
        errors,
    )
    all_payload_ids = [
        binding.payload_artifact_id
        for bindings in PAYLOAD_CONTROLLER_LINKS.values()
        for binding in bindings
    ]
    _require(
        len(all_payload_ids) == len(set(all_payload_ids)),
        "requested-target payload compile observations must bind exactly one controller source",
        errors,
    )
    try:
        level1_sources = {
            item["source_callsite"]: item
            for item in _discover_level1_payload_source_bindings(root, context)
        }
    except (InventoryError, OSError, SyntaxError, UnicodeError) as exc:
        errors.append(f"Level 1 payload source discovery failed closed: {exc}")
        return errors
    for launch_id, bindings in PAYLOAD_CONTROLLER_LINKS.items():
        launch = launches_by_id.get(launch_id, {})
        _require(
            bool(launch),
            f"{launch_id}: missing requested-target payload controller",
            errors,
        )
        _require(
            launch.get("compile_for") == "requested-target",
            f"{launch_id}: controller payload compile_for must be requested-target",
            errors,
        )
        _require(
            launch.get("execute_on") == "host",
            f"{launch_id}: controller itself must execute on host",
            errors,
        )
        expected_bindings: list[dict[str, str]] = []
        for binding in bindings:
            expected = {
                "payload_artifact_id": binding.payload_artifact_id,
                "execution_transport": "direct-host",
                "compatibility_requirement": "requested-target-must-be-host-runnable",
            }
            if binding.source_callsite is not None:
                source = level1_sources.get(binding.source_callsite, {})
                _require(
                    bool(source),
                    f"{launch_id}: missing payload source callsite {binding.source_callsite}",
                    errors,
                )
                _require(
                    source.get("source_selector") == binding.source_selector,
                    f"{launch_id}: {binding.source_callsite} payload selector does not match {binding.source_selector}",
                    errors,
                )
                expected.update(
                    {
                        "source_callsite": binding.source_callsite,
                        "source_selector": binding.source_selector or "",
                        "source_semantics_digest": source.get(
                            "source_semantics_digest", ""
                        ),
                    }
                )
            expected_bindings.append(expected)
            payload = observations_by_id.get(binding.payload_artifact_id, {})
            _require(
                bool(payload),
                f"{launch_id}: payload artifact {binding.payload_artifact_id} does not exist",
                errors,
            )
            _require(
                payload.get("category") == "compile"
                and payload.get("artifact_kind") == "executable",
                f"{launch_id}: payload {binding.payload_artifact_id} must be an executable compile observation",
                errors,
            )
            _require(
                payload.get("compile_for") == launch.get("compile_for"),
                f"{launch_id}: payload {binding.payload_artifact_id} compile_for disagrees with controller record",
                errors,
            )
        _require(
            launch.get("payload_bindings") == expected_bindings,
            f"{launch_id}: payload_bindings must exactly bind payload, transport, compatibility, and reviewed source selection",
            errors,
        )
        _require(
            "payload_artifact_id" not in launch,
            f"{launch_id}: singular payload_artifact_id is forbidden",
            errors,
        )
    level2_controller = launches_by_id.get(
        "python-launch:bench/tools/run_level2_report.py:run_one_process:subprocess.run:1",
        {},
    )
    _require(
        level2_controller.get("compile_for") == "host"
        and level2_controller.get("execute_on") == "host",
        "the separate Level 2 host controller launch must remain host/host",
        errors,
    )
    classifications = inventory["repository_file_classifications"]
    _require(
        inventory.get("repository_file_classifications_digest")
        == _json_fact_digest(classifications),
        "repository_file_classifications_digest must bind the complete reviewed ledger",
        errors,
    )
    try:
        observed_classifications, repository_complete = (
            _discover_repository_file_classifications(root, context)
        )
    except (InventoryError, OSError) as exc:
        errors.append(f"repository file discovery failed closed: {exc}")
        return errors
    observed_by_path = {item["path"]: item for item in observed_classifications}
    classified_by_path = {item.get("path"): item for item in classifications}
    observed_paths = set(observed_by_path)
    classified_paths = set(classified_by_path)
    if repository_complete:
        _require(
            classified_paths == observed_paths,
            "repository_file_classifications must exactly partition the current Git cached/nonignored-untracked public-file universe: "
            f"unclassified={sorted(observed_paths - classified_paths)}, stale={sorted(classified_paths - observed_paths)}",
            errors,
        )
    else:
        _require(
            observed_paths <= classified_paths,
            "archive public files are not all classified: "
            f"unclassified={sorted(observed_paths - classified_paths)}",
            errors,
        )
    if DEPENDABOT_CONFIG_PATH in observed_paths:
        dependabot_bytes = _frozen_regular_bytes(
            context, DEPENDABOT_CONFIG_PATH, "Dependabot configuration"
        )
        _require(
            hashlib.sha256(dependabot_bytes).hexdigest()
            == REVIEWED_DEPENDABOT_CONFIG_SHA256,
            "reviewed Dependabot configuration changed",
            errors,
        )
    enforced_classifications = (
        classified_by_path
        if repository_complete
        else {
            path: item
            for path, item in classified_by_path.items()
            if path in observed_paths
        }
    )
    for path, item in enforced_classifications.items():
        _require(
            set(item) == {"path", "kind", "class", "owner"},
            f"classification:{path}: keys must be path/kind/class/owner exactly",
            errors,
        )
        if path in observed_by_path:
            _require(
                item.get("kind") == observed_by_path[path]["kind"],
                f"classification:{path}: mechanically observed kind changed",
                errors,
            )
        _require(
            item.get("class") in DERIVED_CLASSES,
            f"classification:{path}: invalid class",
            errors,
        )
        _require(
            item.get("owner") in OWNER_VOCABULARY,
            f"classification:{path}: invalid owner",
            errors,
        )
    candidates = inventory["derived_candidates"]
    candidate_paths = {item.get("path") for item in candidates}
    candidate_ids = {item.get("id") for item in candidates}
    required_candidate_ids = (
        REQUIRED_DERIVED_CANDIDATE_IDS
        if repository_complete
        else {
            identifier
            for identifier in REQUIRED_DERIVED_CANDIDATE_IDS
            if identifier.removeprefix("derived:") in observed_paths
        }
    )
    _require(
        required_candidate_ids <= candidate_ids,
        f"missing required reviewed derived details: {sorted(required_candidate_ids - candidate_ids)}",
        errors,
    )
    if not repository_complete:
        _require(
            candidate_ids <= REQUIRED_DERIVED_CANDIDATE_IDS,
            "derived_candidates reviewed fact set changed",
            errors,
        )
    _require(
        len(candidate_paths) == len(candidates),
        "derived_candidates: duplicate path",
        errors,
    )
    candidates_by_id = {item.get("id"): item for item in candidates}
    for identifier, expected_digest in REQUIRED_DERIVED_FACT_DIGESTS.items():
        candidate = candidates_by_id.get(identifier)
        if candidate is not None:
            _require(
                _json_fact_digest(candidate) == expected_digest,
                f"{identifier}: reviewed fact set changed",
                errors,
            )
    for item in candidates:
        _require(
            item.get("id") == f"derived:{item.get('path')}",
            f"{item.get('id')}: id must derive from path",
            errors,
        )
        candidate_rel = item.get("path")
        candidate_present = candidate_rel in context.public_files.path_set
        if repository_complete:
            _require(
                candidate_present,
                f"{item['id']}: derived candidate path must belong to the public universe",
                errors,
            )
        elif not candidate_present:
            _require(
                item.get("id") in REQUIRED_DERIVED_CANDIDATE_IDS,
                f"{item['id']}: derived candidate path must belong to the public universe",
                errors,
            )
            continue
        if candidate_rel in context.public_files.path_set:
            candidate = context.public_files.node(candidate_rel)
            _require(
                candidate.kind == "regular",
                f"{item['id']}: derived candidate must be a non-symlink regular file",
                errors,
            )
        _require(
            item.get("class") in DERIVED_CLASSES,
            f"{item['id']}: invalid derived class",
            errors,
        )
        _require(
            item.get("owner") in OWNER_VOCABULARY,
            f"{item['id']}: invalid owner",
            errors,
        )
        _require(
            item.get("tracking_status") in {"tracked", "current-untracked-gap"},
            f"{item['id']}: invalid tracking_status",
            errors,
        )
        derived_class = item.get("class")
        if derived_class == "designated-reproducible-contract-reference":
            for field in (
                "authoritative_inputs",
                "canonical_command",
                "deterministic_drift_gate",
                "deterministic_drift_gate_ids",
                "consumer_gates",
                "consumer_gate_ids",
                "source_package_disposition",
                "binary_package_disposition",
            ):
                _require(
                    field in item,
                    f"{item['id']}: missing generated-artifact field {field}",
                    errors,
                )
            expected_command = (
                "zig build generate-kernel-coverage --summary failures"
                if item.get("path") == "docs/kernel_coverage.json"
                else "zig build generate-headers --summary failures"
            )
            _require(
                item.get("canonical_command") == expected_command,
                f"{item['id']}: incorrect canonical command",
                errors,
            )
            generator_input = (
                "tools/generate_kernel_coverage.zig"
                if item.get("path") == "docs/kernel_coverage.json"
                else "tools/generate_compat_headers.zig"
            )
            _require(
                generator_input in item.get("authoritative_inputs", []),
                f"{item['id']}: generator source missing from authoritative_inputs",
                errors,
            )
        elif derived_class == "curated-documentation-asset":
            for field in (
                "public_safe_provenance",
                "claim_scope",
                "review_date",
                "freshness_criteria",
                "replacement_criteria",
                "deterministic_regeneration_claim",
                "raw_inputs_disposition",
            ):
                _require(
                    field in item and item[field] not in (None, ""),
                    f"{item['id']}: missing curated provenance field {field}",
                    errors,
                )
            try:
                date.fromisoformat(item.get("review_date", ""))
            except (TypeError, ValueError):
                errors.append(f"{item['id']}: review_date must be an ISO calendar date")
            _require(
                item.get("deterministic_regeneration_claim") is False,
                f"{item['id']}: curated SVG cannot claim deterministic regeneration",
                errors,
            )
        elif item.get("path") == "pkgconfig/zynum_blas.pc":
            for field in ("authorship", "package_disposition", "install_disposition"):
                _require(
                    field in item and item[field] not in (None, ""),
                    f"{item['id']}: missing package metadata field {field}",
                    errors,
                )
        if item.get("path") in {
            "docs/kernel_coverage.json",
            "tools/abi_baseline_observation.json",
        }:
            if item.get("tracking_status") == "current-untracked-gap":
                _require(
                    isinstance(item.get("package_closure_gap"), str)
                    and bool(item["package_closure_gap"]),
                    f"{item['id']}: missing dirty/clean package closure gap",
                    errors,
                )
            else:
                _require(
                    item.get("tracking_status") == "tracked"
                    and "package_closure_gap" not in item,
                    f"{item['id']}: tracked package subject retains a stale closure gap",
                    errors,
                )
    generated_paths = {item.get("path") for item in inventory["generator_targets"]}
    _require(
        len(generated_paths) == len(inventory["generator_targets"]),
        "generator_targets: duplicate path",
        errors,
    )
    candidates_by_path = {item.get("path"): item for item in candidates}
    for path, classification in enforced_classifications.items():
        if classification.get("class") not in {
            "designated-reproducible-contract-reference",
            "curated-documentation-asset",
        }:
            continue
        detail = candidates_by_path.get(path)
        _require(
            detail is not None,
            f"classification:{path}: designated/curated file requires derived_candidates detail",
            errors,
        )
        if detail is not None:
            _require(
                detail.get("class") == classification.get("class"),
                f"classification:{path}: detail class mismatch",
                errors,
            )
            _require(
                detail.get("owner") == classification.get("owner"),
                f"classification:{path}: detail owner mismatch",
                errors,
            )
    for item in candidates:
        if not repository_complete and item.get("path") not in observed_paths:
            continue
        classification = enforced_classifications.get(item.get("path"))
        _require(
            classification is not None,
            f"{item['id']}: detailed candidate has no whole-ledger classification",
            errors,
        )
        if classification is not None:
            _require(
                item.get("class") == classification.get("class"),
                f"{item['id']}: class must match whole-ledger classification",
                errors,
            )
            _require(
                item.get("owner") == classification.get("owner"),
                f"{item['id']}: owner must match whole-ledger classification",
                errors,
            )
    for target in inventory["generator_targets"]:
        _require(
            target.get("owner") in OWNER_VOCABULARY,
            f"{target['id']}: invalid generator owner",
            errors,
        )
        if not repository_complete and target.get("path") not in observed_paths:
            continue
        candidate = candidates_by_path.get(target.get("path"), {})
        _require(
            candidate.get("class") == "designated-reproducible-contract-reference",
            f"{target['id']}: generated target must map to a designated reproducible candidate",
            errors,
        )
        _require(
            candidate.get("owner") == target.get("owner"),
            f"{target['id']}: owner must match its derived candidate",
            errors,
        )
    enforced_generated_paths = (
        generated_paths if repository_complete else generated_paths & observed_paths
    )
    _require(
        enforced_generated_paths <= candidate_paths,
        "every generator target must be a derived candidate",
        errors,
    )
    svg_paths = {
        rel
        for rel in context.public_files.paths
        if PurePosixPath(rel).parent == PurePosixPath("docs/assets/benchmarks")
        and PurePosixPath(rel).suffix.lower() == ".svg"
    }
    pc_paths = {
        rel
        for rel in context.public_files.paths
        if PurePosixPath(rel).parent == PurePosixPath("pkgconfig")
        and PurePosixPath(rel).suffix.lower() == ".pc"
    }
    for path in svg_paths:
        candidate = candidates_by_path.get(path, {})
        _require(
            candidate.get("class") == "curated-documentation-asset",
            f"derived:{path}: benchmark SVG must be curated documentation",
            errors,
        )
        _require(
            candidate.get("owner") == "documentation-maintainers",
            f"derived:{path}: benchmark SVG owner is incorrect",
            errors,
        )
    for path in pc_paths:
        candidate = candidates_by_path.get(path, {})
        _require(
            candidate.get("class") == "non-generated-source",
            f"derived:{path}: package metadata must be non-generated source",
            errors,
        )
        _require(
            candidate.get("owner") == "package-metadata",
            f"derived:{path}: package metadata owner is incorrect",
            errors,
        )
    baseline_candidate = candidates_by_path.get(
        "tools/abi_baseline_observation.json", {}
    )
    _require(
        baseline_candidate.get("class") == "non-generated-source",
        "ABI baseline observation must be non-generated source",
        errors,
    )
    _require(
        baseline_candidate.get("owner") == "abi-compatibility",
        "ABI baseline observation owner is incorrect",
        errors,
    )
    gate_ids = {
        item.get("id")
        for section in ("build_observations", "python_launches", "workflow_launches")
        for item in inventory[section]
    }
    for item in candidates:
        if item.get("class") != "designated-reproducible-contract-reference":
            continue
        drift_ids = item.get("deterministic_drift_gate_ids", [])
        consumer_ids = item.get("consumer_gate_ids", [])
        _require(
            isinstance(drift_ids, list),
            f"{item['id']}: deterministic_drift_gate_ids must be an array",
            errors,
        )
        _require(
            isinstance(consumer_ids, list),
            f"{item['id']}: consumer_gate_ids must be an array",
            errors,
        )
        if isinstance(drift_ids, list):
            _require(
                set(drift_ids) <= gate_ids,
                f"{item['id']}: unknown deterministic drift gate id",
                errors,
            )
        if isinstance(consumer_ids, list):
            _require(
                set(consumer_ids) <= gate_ids,
                f"{item['id']}: unknown consumer gate id",
                errors,
            )
        if item.get("tracking_status") == "tracked":
            _require(
                bool(drift_ids),
                f"{item['id']}: tracked reproducible artifact requires a drift gate id",
                errors,
            )
            _require(
                bool(consumer_ids),
                f"{item['id']}: tracked reproducible artifact requires a consumer gate id",
                errors,
            )
    try:
        observed = discover(root, inventory, context)
    except (
        InventoryError,
        SyntaxError,
        UnicodeError,
        OSError,
        RecursionError,
    ) as exc:
        errors.append(f"source discovery failed closed: {exc}")
        return errors
    for section in (
        "build_observations",
        "python_launches",
        "workflow_launches",
        "generator_targets",
    ):
        _compare_ids(section, inventory[section], observed[section], errors)
    observed_workflow_digests = {
        item["id"]: item["source_digest"] for item in observed["workflow_launches"]
    }
    _require(
        inventory.get("workflow_source_digests") == observed_workflow_digests,
        "workflow_source_digests must exactly match every normalized workflow run step",
        errors,
    )
    observed_link_guard_digests = {
        item["id"]: item["guard_digest"]
        for item in observed["build_observations"]
        if item.get("category") == "link"
    }
    _require(
        inventory.get("conditional_link_guard_digests") == observed_link_guard_digests,
        "conditional_link_guard_digests must exactly match every conditional link edge",
        errors,
    )
    expected_links = [
        item
        for item in inventory["build_observations"]
        if item.get("category") == "link"
    ]
    _require(
        len(expected_links) == 28,
        f"expected 28 current conditional link edges, found {len(expected_links)}",
        errors,
    )
    gap_ids = {item.get("id") for item in inventory.get("current_gaps", [])}
    _require(
        gap_ids == REQUIRED_GAP_IDS,
        f"current gap ids must match the complete reviewed set: missing={sorted(REQUIRED_GAP_IDS - gap_ids)}, extra={sorted(gap_ids - REQUIRED_GAP_IDS)}",
        errors,
    )
    gaps_by_id = {item.get("id"): item for item in inventory.get("current_gaps", [])}
    for identifier, expected_digest in REQUIRED_GAP_FACT_DIGESTS.items():
        gap = gaps_by_id.get(identifier)
        if gap is not None:
            _require(
                _json_fact_digest(gap) == expected_digest,
                f"{identifier}: reviewed fact set changed",
                errors,
            )
    for identifier in REQUIRED_GAP_IDS:
        for field in ("classification", "observed_result", "status", "owner"):
            _require(
                isinstance(gaps_by_id.get(identifier, {}).get(field), str)
                and bool(gaps_by_id[identifier][field]),
                f"{identifier}: missing gap field {field}",
                errors,
            )
        _require(
            gaps_by_id.get(identifier, {}).get("owner") in OWNER_VOCABULARY,
            f"{identifier}: invalid gap owner",
            errors,
        )
    payload_gap = gaps_by_id.get("gap:cross-target-benchmark-payload-execution", {})
    expected_controller_ids = sorted(PAYLOAD_CONTROLLER_LINKS)
    expected_payload_ids = sorted(all_payload_ids)
    _require(
        payload_gap.get("controller_ids") == expected_controller_ids
        and payload_gap.get("payload_artifact_ids") == expected_payload_ids
        and payload_gap.get("controller_count") == len(expected_controller_ids)
        and payload_gap.get("payload_count") == len(expected_payload_ids),
        "cross-target payload gap must bind the exact reviewed controller/payload set and counts",
        errors,
    )
    _require(
        payload_gap.get("execution_transport") == "direct-host"
        and payload_gap.get("compatibility_requirement")
        == "requested-target-must-be-host-runnable",
        "cross-target payload gap must bind direct-host transport and host-runnable compatibility",
        errors,
    )
    _require(
        payload_gap.get("observed_result")
        == (
            "no emulator or remote runner is wired for "
            f"{len(expected_controller_ids)} requested-target benchmark payload "
            f"controllers and {len(expected_payload_ids)} exact executable payloads"
        ),
        "cross-target payload gap observed_result does not match the mechanical controller/payload counts",
        errors,
    )
    _require(
        gaps_by_id.get("gap:example-optimize-forwarding", {}).get(
            "reproduction_command"
        )
        == "cd examples/zig && zig build --help",
        "example optimize gap reproduction command is incorrect",
        errors,
    )
    example_gap = gaps_by_id.get("gap:example-optimize-forwarding", {})
    _require(
        example_gap.get("observed_exit_code") == 0,
        "example optimize gap exit code must be 0",
        errors,
    )
    _require(
        example_gap.get("stderr_contains") == "error: invalid option: -Doptimize",
        "example optimize gap stderr observation is incorrect",
        errors,
    )
    return errors


def validate(root: Path, inventory_path: Path) -> list[str]:
    try:
        return _validate(root, inventory_path)
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        return [f"inventory structure is invalid: {type(exc).__name__}: {exc}"]


def _reviewed_observation_refresh_fields(identifier: str) -> dict[str, Any]:
    if identifier in {
        TEST_INVENTORY_RUNNER_COMPILE_PYTHON_LAUNCH_ID,
        TEST_INVENTORY_RUNNER_EXECUTE_PYTHON_LAUNCH_ID,
        TEST_INVENTORY_RUNNER_RACE_PYTHON_LAUNCH_ID,
    }:
        return _new_test_inventory_python_launch(identifier)
    install_reachability = WINDOWS_PYTHON_TOOLING_INSTALL_REACHABILITY.get(identifier)
    if install_reachability is not None:
        return {"condition": install_reachability}
    fixture_source = WINDOWS_PYTHON_TOOLING_FIXTURE_COMPILE_SOURCES.get(identifier)
    if fixture_source is not None:
        return {
            "root_source": [fixture_source, WINDOWS_PYTHON_TOOLING_FIXTURE_PATH],
            "root_source_by_target": {
                "windows": WINDOWS_PYTHON_TOOLING_FIXTURE_PATH,
                "non-windows": fixture_source,
            },
            "evidence_role_by_target": {
                "windows": "python-tooling-fixture-only-not-benchmark-runtime-evidence",
                "non-windows": "benchmark-probe-runtime-evidence",
            },
        }
    if identifier == LEVEL2_WIDTH_DEFAULT_ARTIFACT_COMPILE_ID:
        return {
            "root_source": [
                "test/build/level2_width_default_artifact_probe.zig",
                LEVEL2_WIDTH_ARTIFACT_CONTRACT_PATH,
            ],
            "probe_contract_source": LEVEL2_WIDTH_ARTIFACT_CONTRACT_PATH,
        }
    if identifier == "compile:build.zig:build:lib":
        return {
            "install_destinations": [
                "zig-out/lib/libzynum_blas.so",
                "zig-out/lib/libzynum_blas.dylib",
                "zig-out/bin/zynum_blas.dll",
                "zig-out/lib/zynum_blas.lib",
            ]
        }
    if identifier == "compile:build.zig:build:static_lib":
        return {
            "produced_outputs": ["libzynum_blas.a", "zynum_blas.lib"],
            "install_destinations": [
                "zig-out/lib/libzynum_blas.a",
                "zig-out/lib/static/zynum_blas.lib",
            ],
            "install_destinations_by_target": {
                "elf": {"primary": "zig-out/lib/libzynum_blas.a"},
                "macho": {"primary": "zig-out/lib/libzynum_blas.a"},
                "windows": {"primary": "zig-out/lib/static/zynum_blas.lib"},
            },
        }
    if identifier in WINDOWS_EXCLUDED_DEFAULT_EXECUTABLE_INSTALL_IDS:
        return {
            "condition": "requested target OS is not Windows and install step is reached"
        }
    return {}


def _apply_reviewed_build_inventory_migrations(inventory: dict[str, Any]) -> None:
    inventory["scope"] = INVENTORY_SCOPE
    for item in inventory["build_observations"]:
        if item.get("category") == "launch":
            item["detail_status"] = "process-lifecycle-out-of-scope"
    for section in ("python_launches", "workflow_launches"):
        for item in inventory[section]:
            item["detail_status"] = "process-lifecycle-out-of-scope"

    classifications = {
        item["path"]: item for item in inventory["repository_file_classifications"]
    }
    for path in NEW_REVIEWED_TEST_INFRASTRUCTURE_CLASSIFICATIONS:
        classifications[path] = {
            "path": path,
            "kind": "zig-source",
            "class": "non-generated-source",
            "owner": "test-infrastructure",
        }
    classifications.update(
        {
            ".github/dependabot.yml": {
                "path": ".github/dependabot.yml",
                "kind": "configuration-metadata",
                "class": "non-generated-source",
                "owner": "workflow-maintainers",
            },
            ".github/ISSUE_TEMPLATE/config.yml": {
                "path": ".github/ISSUE_TEMPLATE/config.yml",
                "kind": "configuration-metadata",
                "class": "non-generated-source",
                "owner": "workflow-maintainers",
            },
            ".github/ISSUE_TEMPLATE/question.yml": {
                "path": ".github/ISSUE_TEMPLATE/question.yml",
                "kind": "configuration-metadata",
                "class": "non-generated-source",
                "owner": "workflow-maintainers",
            },
            "SUPPORT.md": {
                "path": "SUPPORT.md",
                "kind": "documentation",
                "class": "non-generated-source",
                "owner": "project-governance",
            },
            "COPYING": {
                "path": "COPYING",
                "kind": "legal-governance",
                "class": "non-generated-source",
                "owner": "project-governance",
            },
            "COPYING.LESSER": {
                "path": "COPYING.LESSER",
                "kind": "legal-governance",
                "class": "non-generated-source",
                "owner": "project-governance",
            },
        }
    )
    inventory["repository_file_classifications"] = [
        classifications[path] for path in sorted(classifications)
    ]
    inventory["repository_file_classifications_digest"] = _json_fact_digest(
        inventory["repository_file_classifications"]
    )
    drift_gate_ids = [
        "workflow-launch:.github/workflows/ci.yml:source-checks:regenerate-compatibility-headers-and-kernel-coverage",
        "workflow-launch:.github/workflows/ci.yml:source-checks:check-generated-files-are-up-to-date",
    ]
    derived_candidates = {item["id"]: item for item in inventory["derived_candidates"]}
    for path in (
        "include/zynum/blas/blas.h",
        "include/zynum/blas/cblas.h",
        "include/zynum/blas/blas.f90",
        "include/zynum/blas/abi_manifest.json",
    ):
        derived_candidates[f"derived:{path}"]["deterministic_drift_gate_ids"] = (
            drift_gate_ids
        )
    derived_candidates["derived:docs/kernel_coverage.json"] = {
        "id": "derived:docs/kernel_coverage.json",
        "path": "docs/kernel_coverage.json",
        "class": "designated-reproducible-contract-reference",
        "owner": "kernel-coverage",
        "tracking_status": "tracked",
        "authoritative_inputs": [
            "tools/generate_kernel_coverage.zig",
            "src/blas/kernel_coverage_root.zig",
            "src/blas/kernels/coverage.zig",
            "kernel registry and coverage declarations",
        ],
        "canonical_command": "zig build generate-kernel-coverage --summary failures",
        "deterministic_drift_gate": [
            "CI source-checks regenerates and rejects changes to docs/kernel_coverage.json"
        ],
        "deterministic_drift_gate_ids": drift_gate_ids,
        "consumer_gates": ["CI generated-file drift validation"],
        "consumer_gate_ids": [
            "workflow-launch:.github/workflows/ci.yml:source-checks:check-generated-files-are-up-to-date"
        ],
        "source_package_disposition": "included",
        "binary_package_disposition": "not installed",
    }
    derived_candidates["derived:tools/abi_baseline_observation.json"] = {
        "id": "derived:tools/abi_baseline_observation.json",
        "path": "tools/abi_baseline_observation.json",
        "class": "non-generated-source",
        "owner": "abi-compatibility",
        "tracking_status": "tracked",
        "authorship": "observed ABI baseline policy source; not a generator output",
        "package_disposition": "included",
        "install_disposition": "not installed",
    }
    inventory["derived_candidates"] = [
        derived_candidates[item["id"]] for item in inventory["derived_candidates"]
    ]
    closed_gaps = {
        "gap:windows-library-install-collision",
        "gap:windows-default-install-executables",
        "gap:kernel-coverage-untracked",
    }
    inventory["current_gaps"] = [
        item for item in inventory["current_gaps"] if item["id"] not in closed_gaps
    ]
    process_bounds_gap = next(
        item
        for item in inventory["current_gaps"]
        if item["id"] == "gap:process-bounds-deferred"
    )
    process_bounds_gap["classification"] = (
        "launch occurrences are classified here; command construction, bounds, exit, "
        "cancellation, and descendant cleanup remain outside this inventory's scope"
    )
    process_bounds_gap["observed_result"] = (
        "occurrence coverage is enforced without claiming detailed subprocess "
        "lifecycle semantics"
    )
    process_bounds_gap["status"] = (
        "process lifecycle guarantees are not claimed by this inventory"
    )


def _new_test_inventory_observation(
    identifier: str, inventory: dict[str, Any]
) -> dict[str, Any]:
    if identifier == PYTHON_TOOLING_LAUNCH_ID:
        return _python_tooling_launch_template()
    if identifier == PYTHON_TOOLING_STEP_ID:
        return _python_tooling_step_template()
    if identifier == LEVEL2_WIDTH_DEFAULT_ARTIFACT_COMPILE_ID:
        return {
            "owner": "test-infrastructure",
            "artifact_kind": "executable",
            "output_name": "zynum-level2-width-default-artifact-probe",
            "root_source": [
                "test/build/level2_width_default_artifact_probe.zig",
                LEVEL2_WIDTH_ARTIFACT_CONTRACT_PATH,
            ],
            "probe_contract_source": LEVEL2_WIDTH_ARTIFACT_CONTRACT_PATH,
            "linkage": "not-applicable",
            "compile_for": "requested-target",
            "execute_on": "requested-target",
            "optimize_source": "test-optimize",
            "condition": LEVEL2_WIDTH_DEFAULT_ARTIFACT_CONDITION,
            "produced_outputs": [
                "zynum-level2-width-default-artifact-probe",
                "zynum-level2-width-default-artifact-probe.exe",
            ],
            "install_destinations": [],
        }
    if identifier == LEVEL2_WIDTH_DEFAULT_ARTIFACT_LAUNCH_ID:
        return {
            "owner": "test-infrastructure",
            "detail_status": "process-lifecycle-out-of-scope",
            "cwd_shape": "repository-root",
            "command_shape": "run-artifact",
            "source_artifact": LEVEL2_WIDTH_DEFAULT_ARTIFACT_COMPILE_ID,
            "compile_for": "requested-target",
            "execute_on": "requested-target",
            "argv_shape": [],
            "launch_class": "validation",
        }
    if identifier == LEVEL2_WIDTH_DEFAULT_ARTIFACT_STEP_ID:
        return {
            "owner": "build-composition",
            "description": "Run the default x86 Level 2 width production-artifact probe",
            "direct_dependencies": [
                {
                    "id": LEVEL2_WIDTH_DEFAULT_ARTIFACT_LAUNCH_ID,
                    "condition": LEVEL2_WIDTH_DEFAULT_ARTIFACT_CONDITION,
                }
            ],
            "aggregate_test_membership": "conditional-member",
            "aggregate_condition": LEVEL2_WIDTH_DEFAULT_ARTIFACT_CONDITION,
            "intentional_orphan": False,
            "orphan_reason": "conditionally reachable from canonical test aggregate",
            "step_role": "focused-validation",
        }
    if identifier == LEVEL2_WIDTH_DEFAULT_ARTIFACT_LINK_ID:
        return {
            "owner": "build-composition",
            "consumer": "level2_width_default_artifact_probe_mod",
            "provider": "level2_width_isolated_library",
            "condition": LEVEL2_WIDTH_DEFAULT_ARTIFACT_CONDITION,
        }
    if identifier == LEVEL2_WIDTH_ENABLED_ARTIFACT_COMPILE_ID:
        return {
            "owner": "test-infrastructure",
            "artifact_kind": "executable",
            "output_name": "zynum-level2-width-enabled-artifact-probe",
            "root_source": [
                "test/build/level2_width_enabled_artifact_probe.zig",
                LEVEL2_WIDTH_ARTIFACT_CONTRACT_PATH,
            ],
            "probe_contract_source": LEVEL2_WIDTH_ARTIFACT_CONTRACT_PATH,
            "linkage": "not-applicable",
            "compile_for": "requested-target",
            "execute_on": "requested-target",
            "optimize_source": "optimize",
            "condition": LEVEL2_WIDTH_ENABLED_ARTIFACT_CONDITION,
            "produced_outputs": [
                "zynum-level2-width-enabled-artifact-probe",
                "zynum-level2-width-enabled-artifact-probe.exe",
            ],
            "install_destinations": [],
        }
    if identifier == LEVEL2_WIDTH_ENABLED_ARTIFACT_LAUNCH_ID:
        return {
            "owner": "test-infrastructure",
            "detail_status": "process-lifecycle-out-of-scope",
            "cwd_shape": "repository-root",
            "command_shape": "run-artifact",
            "source_artifact": LEVEL2_WIDTH_ENABLED_ARTIFACT_COMPILE_ID,
            "compile_for": "requested-target",
            "execute_on": "requested-target",
            "argv_shape": [],
            "launch_class": "validation",
            "condition": LEVEL2_WIDTH_ENABLED_ARTIFACT_CONDITION,
        }
    if identifier in {
        LEVEL2_WIDTH_ENABLED_ARTIFACT_BUILD_STEP_ID,
        LEVEL2_WIDTH_ENABLED_ARTIFACT_RUN_STEP_ID,
    }:
        is_build = identifier == LEVEL2_WIDTH_ENABLED_ARTIFACT_BUILD_STEP_ID
        return {
            "owner": "build-composition",
            "description": (
                "Compile the enabled x86 Level 2 width production-artifact probe"
                if is_build
                else "Run the enabled x86 Level 2 width production-artifact probe on AVX-512 hardware"
            ),
            "direct_dependencies": [
                {
                    "id": (
                        LEVEL2_WIDTH_ENABLED_ARTIFACT_COMPILE_ID
                        if is_build
                        else LEVEL2_WIDTH_ENABLED_ARTIFACT_LAUNCH_ID
                    ),
                    "condition": LEVEL2_WIDTH_ENABLED_ARTIFACT_CONDITION,
                },
                {
                    "id": LEVEL2_WIDTH_ENABLED_ARTIFACT_UNSUPPORTED_STEP_ID,
                    "condition": LEVEL2_WIDTH_ENABLED_ARTIFACT_UNSUPPORTED_CONDITION,
                },
            ],
            "aggregate_test_membership": "not-member",
            "aggregate_condition": "explicit named step only",
            "intentional_orphan": False,
            "orphan_reason": "valid standalone fail-closed production-artifact probe entry point",
            "step_role": "focused-validation",
        }
    if identifier == LEVEL2_WIDTH_ENABLED_ARTIFACT_UNSUPPORTED_STEP_ID:
        return {
            "owner": "build-composition",
            "description": "the enabled Level 2 width artifact probe requires an x86_64 AVX-512 target and -Dlevel2-width-candidates=true",
            "direct_dependencies": [],
            "aggregate_test_membership": "conditional-named-step-guard",
            "aggregate_condition": LEVEL2_WIDTH_ENABLED_ARTIFACT_UNSUPPORTED_CONDITION,
            "intentional_orphan": False,
            "orphan_reason": "shared fail-closed dependency of both enabled artifact-probe steps",
            "step_role": "fail-closed-platform-guard",
        }
    if identifier == LEVEL2_WIDTH_ENABLED_ARTIFACT_LINK_ID:
        return {
            "owner": "build-composition",
            "consumer": "level2_width_enabled_artifact_probe_mod",
            "provider": "level2_width_isolated_library",
            "condition": LEVEL2_WIDTH_ENABLED_ARTIFACT_CONDITION,
        }
    if identifier == INSTALL_DYNAMIC_LIBRARY_ID:
        return {
            "owner": "build-composition",
            "producer": "compile:build.zig:build:lib",
            "source": "emitted artifact",
            "destination": [
                "zig-out/lib/libzynum_blas.so",
                "zig-out/lib/libzynum_blas.dylib",
                "zig-out/bin/zynum_blas.dll",
                "zig-out/lib/zynum_blas.lib",
            ],
            "condition": WINDOWS_PYTHON_TOOLING_INSTALL_REACHABILITY[
                INSTALL_DYNAMIC_LIBRARY_ID
            ],
        }
    if identifier == INSTALL_STATIC_LIBRARY_ID:
        return {
            "owner": "build-composition",
            "producer": "compile:build.zig:build:static_lib",
            "source": "emitted artifact",
            "destination": [
                "zig-out/lib/libzynum_blas.a",
                "zig-out/lib/static/zynum_blas.lib",
            ],
            "condition": "install or install-libraries step is reached",
        }
    if identifier == INSTALL_LIBRARIES_STEP_ID:
        return {
            "owner": "build-composition",
            "description": "Install the shared and static Zynum BLAS libraries without tools",
            "direct_dependencies": [
                {"id": INSTALL_DYNAMIC_LIBRARY_ID, "condition": "always"},
                {"id": INSTALL_STATIC_LIBRARY_ID, "condition": "always"},
            ],
            "aggregate_test_membership": "not-member",
            "aggregate_condition": "not-applicable",
            "intentional_orphan": False,
            "orphan_reason": "valid standalone packaging entry point used by Windows CI",
            "step_role": "packaging",
        }
    system_command_argv = {
        "launch:build.zig:build:test_inventory_security_tests": [
            "python3",
            "-B",
            "test/build/test_test_inventory.py",
        ],
        "launch:build.zig:build:test_inventory_structure_check": [
            "python3",
            "-B",
            "tools/check_test_inventory.py",
            "--root",
            ".",
            "--structure-only",
        ],
    }
    if identifier in system_command_argv:
        return {
            "owner": "test-infrastructure",
            "detail_status": "process-lifecycle-out-of-scope",
            "cwd_shape": "repository-root",
            "command_shape": "system-command",
            "source_artifact": None,
            "compile_for": "host",
            "execute_on": "host",
            "argv_shape": system_command_argv[identifier],
            "launch_class": "validation",
        }
    if identifier == "step:build.zig:build:test-test-inventory":
        return {
            "owner": "build-composition",
            "description": "Run the complete test-inventory security regression suite",
            "direct_dependencies": [
                {
                    "id": "launch:build.zig:build:test_inventory_security_tests",
                    "condition": "always",
                }
            ],
            "aggregate_test_membership": "not-member",
            "aggregate_condition": "not-applicable",
            "intentional_orphan": False,
            "orphan_reason": "valid standalone entry point; aggregate test membership is not intended",
            "step_role": "focused-validation",
        }
    if identifier == "option:build.zig:build:target":
        return {
            "owner": "build-composition",
            "observation_role": "source declaration for option surfaces",
        }
    if identifier == TEST_INVENTORY_FACTORY_COMPILE_ID:
        observations = {item["id"]: item for item in inventory["build_observations"]}
        logical_sources = sorted(
            {
                source
                for _, logical_id, _ in REQUIRED_TEST_INVENTORY_FACTORY_CASES
                for source in observations[logical_id]["root_source"]
            }
        )
        return {
            "owner": "test-infrastructure",
            "artifact_kind": "test",
            "output_name": "inventory-{root-id}",
            "root_source": logical_sources,
            "linkage": "not-applicable",
            "compile_for": "requested-target",
            "execute_on": "requested-target",
            "optimize_source": "test-optimize",
            "condition": "one artifact for each applicable logical Zig root",
            "produced_outputs": [
                "inventory-{root-id}",
                "inventory-{root-id}.exe",
            ],
            "install_destinations": [],
        }
    if identifier == TEST_INVENTORY_FACTORY_LAUNCH_ID:
        return {
            "owner": "test-infrastructure",
            "detail_status": "process-lifecycle-out-of-scope",
            "cwd_shape": "repository-root",
            "command_shape": "run-artifact",
            "source_artifact": TEST_INVENTORY_FACTORY_COMPILE_ID,
            "compile_for": "requested-target",
            "execute_on": "requested-target",
            "argv_shape": [
                "tools/test_inventory.json",
                "--inventory-root",
                "<root-id>",
                "--inventory-mode",
                "<test-optimize>",
                "--inventory-environment",
                "<environment-id>",
                "--inventory-class",
                "<enumeration-class-id>",
            ],
            "launch_class": "validation",
        }
    if identifier == TEST_INVENTORY_LINK_STEP_ID:
        return {
            "owner": "build-composition",
            "description": "Compile every applicable test-inventory enumerator without running it",
            "aggregate_test_membership": "supporting-entry-point",
            "aggregate_condition": "not-applicable",
            "intentional_orphan": False,
            "orphan_reason": "explicit compile-only closure for every applicable enumerator expansion",
        }
    if identifier == TEST_INVENTORY_RUN_STEP_ID:
        return {
            "owner": "build-composition",
            "description": "Run and verify the exact native test inventory without executing test bodies",
            "aggregate_test_membership": "member",
            "aggregate_condition": "always",
            "intentional_orphan": False,
            "orphan_reason": "direct dependency of the canonical correctness aggregate",
        }
    if identifier == NATIVE_FEATURE_STEP_ID:
        return {
            "owner": "build-composition",
            "description": "Run correctness-only tests for an explicit host-supported non-baseline CPU profile; not inventory evidence",
            "aggregate_test_membership": "not-member",
            "aggregate_condition": "explicit named step only",
            "intentional_orphan": False,
            "orphan_reason": "standalone host-native feature correctness entry point outside frozen inventory evidence",
        }
    if identifier == NATIVE_FEATURE_LAUNCH_ID:
        return {
            "owner": "test-infrastructure",
            "detail_status": "process-lifecycle-out-of-scope",
            "cwd_shape": "repository-root",
            "command_shape": "run-artifact",
            "source_artifact": "same logical test Compile pointer per expansion case",
            "compile_for": "requested-target-exactly-matching-build-host",
            "execute_on": "build-host-without-external-executor",
            "argv_shape": [],
            "launch_class": "native-feature-correctness-only",
        }
    native_guard_templates = {
        guard_id: (message, condition)
        for guard_id, message, condition in NATIVE_FEATURE_GUARDS
    }
    if identifier in native_guard_templates:
        message, condition = native_guard_templates[identifier]
        return {
            "owner": "build-composition",
            "description": message,
            "aggregate_test_membership": "conditional-named-step-guard",
            "aggregate_condition": condition,
            "intentional_orphan": False,
            "orphan_reason": "fail-closed dependency selected before native feature test body execution",
        }
    if identifier == TEST_INVENTORY_UNSUPPORTED_TARGET_STEP_ID:
        return {
            "owner": "build-composition",
            "description": TEST_INVENTORY_UNSUPPORTED_TARGET_MESSAGE,
            "aggregate_test_membership": "conditional-test-guard",
            "aggregate_condition": "unknown or nonbaseline target CPU profile",
            "intentional_orphan": False,
            "orphan_reason": "shared failure dependency of both inventory entry points for unsupported targets",
        }
    raise InventoryError(f"no reviewed template for new observation {identifier}")


def _new_test_inventory_python_launch(identifier: str) -> dict[str, Any]:
    if identifier not in REQUIRED_TEST_INVENTORY_PYTHON_LAUNCH_IDS:
        raise InventoryError(f"no reviewed template for new Python launch {identifier}")
    reviewed_launch = REVIEWED_ARCHIVE_AND_NATIVE_FEATURE_PYTHON_LAUNCHES.get(
        identifier
    )
    if reviewed_launch is not None:
        return {
            "detail_status": "process-lifecycle-out-of-scope",
            "compile_for": "host",
            "execute_on": "host",
            "cwd_shape": "temporary fixture repository",
            "owner": "test-infrastructure",
            **reviewed_launch,
        }
    if identifier == TEST_INVENTORY_RUNNER_COMPILE_PYTHON_LAUNCH_ID:
        return {
            "detail_status": "process-lifecycle-out-of-scope",
            "compile_for": "host",
            "execute_on": "host",
            "cwd_shape": "temporary fixture repository",
            "owner": "test-infrastructure",
            "launch_class": "test-inventory-runner-fixture-compile",
            "argv_shape": [
                "zig",
                "test",
                "<fixture>/digest_vectors.zig",
                "--name",
                "digest_vectors",
                "-O",
                "Debug",
                "-target",
                "<host-native-baseline-target>",
                "-mcpu",
                "baseline",
                "--test-runner",
                "<fixture>/tools/test_inventory_vector_runner.zig",
                "--test-no-exec",
                "-femit-bin=<fixture>/test-inventory-digest-vectors",
            ],
        }
    if identifier == TEST_INVENTORY_RUNNER_EXECUTE_PYTHON_LAUNCH_ID:
        return {
            "detail_status": "process-lifecycle-out-of-scope",
            "compile_for": "host",
            "execute_on": "host",
            "cwd_shape": "temporary fixture repository",
            "owner": "test-infrastructure",
            "launch_class": "test-inventory-runner-fixture-execute",
            "argv_shape": [
                "./test-inventory-digest-vectors",
                "<fixture-inventory-path>",
                "--inventory-environment",
                "<host-native-inventory-environment>",
                "--inventory-root",
                "zig-root:header-smoke-tests",
                "--inventory-mode",
                "Debug",
                "--inventory-class",
                "<host-native-enumeration-class>",
            ],
        }
    if identifier == TEST_INVENTORY_RUNNER_RACE_PYTHON_LAUNCH_ID:
        return {
            "detail_status": "process-lifecycle-out-of-scope",
            "compile_for": "host",
            "execute_on": "host",
            "cwd_shape": "temporary fixture repository",
            "owner": "test-infrastructure",
            "launch_class": "test-inventory-runner-fixture-execute",
            "argv_shape": [
                "./test-inventory-digest-vectors",
                "<fixture-inventory-path>",
                "--inventory-environment",
                "<host-native-inventory-environment>",
                "--inventory-root",
                "zig-root:header-smoke-tests",
                "--inventory-mode",
                "Debug",
                "--inventory-class",
                "<host-native-enumeration-class>",
            ],
        }
    capsule_argv = [
        "<current-python>",
        "-I",
        "-S",
        "-B",
        "-c",
        "<reviewed-python-tooling-bootstrap>",
        "<capsule-transport-descriptor>",
        "<capsule-sha256>",
        "<bootstrap-sha256>",
        "<capsule-mode>",
        "<reviewed-controller-target>",
        "<controller-arguments>",
    ]
    capsule_templates = {
        PYTHON_TOOLING_CAPSULE_PROCESS_LAUNCH_IDS[0]: {
            "launch_class": "python-tooling-posix-capsule-probe",
            "argv_shape": capsule_argv,
            "transport_shape": "inherited-posix-file-descriptor",
        },
        PYTHON_TOOLING_CAPSULE_PROCESS_LAUNCH_IDS[1]: {
            "launch_class": "python-tooling-non-python-controller-passthrough",
            "argv_shape": ["<original-controller-argv>"],
            "transport_shape": "unchanged-trusted-subprocess-run",
        },
        PYTHON_TOOLING_CAPSULE_PROCESS_LAUNCH_IDS[2]: {
            "launch_class": "python-tooling-posix-capsule-controller",
            "argv_shape": capsule_argv,
            "transport_shape": "inherited-posix-file-descriptor",
        },
        PYTHON_TOOLING_CAPSULE_PROCESS_LAUNCH_IDS[3]: {
            "launch_class": "python-tooling-windows-capsule-controller",
            "argv_shape": capsule_argv,
            "transport_shape": "explicit-inherited-windows-handle",
        },
    }
    capsule_template = capsule_templates.get(identifier)
    if capsule_template is not None:
        return {
            "detail_status": "process-lifecycle-out-of-scope",
            "compile_for": "host",
            "execute_on": "host",
            "cwd_shape": "declared by controller call or inherited repository root",
            "owner": "test-infrastructure",
            **capsule_template,
        }
    return {
        "detail_status": "process-lifecycle-out-of-scope",
        "compile_for": "host",
        "execute_on": "host",
        "cwd_shape": "declared by call or inherited from invoking repository command",
        "owner": "test-infrastructure",
        "launch_class": "test-fixture",
    }


def _new_test_inventory_workflow_launch(identifier: str) -> dict[str, Any]:
    identifiers = {
        "workflow-launch:.github/workflows/ci.yml:source-checks:check-build-inventory",
        "workflow-launch:.github/workflows/ci.yml:source-checks:check-test-inventory-structure",
        "workflow-launch:.github/workflows/ci.yml:build-inventory-security:run-build-inventory-security-suite",
        "workflow-launch:.github/workflows/ci.yml:ci-gate:require-every-ci-gate-to-succeed",
        "workflow-launch:.github/workflows/ci.yml:test-inventory-security:run-test-inventory-security-suite",
        "workflow-launch:.github/workflows/ci.yml:target-tests:build-windows-python-tooling-executable-fixtures-and-libraries",
        "workflow-launch:.github/workflows/ci.yml:target-tests:check-windows-library-layout-and-tooling-fixture-boundary",
        "workflow-launch:.github/workflows/ci.yml:target-tests:run-windows-python-tooling-inventory-gate",
        "workflow-launch:.github/workflows/ci.yml:capability-builds:compile-enabled-level-2-width-production-artifact-probe",
        "workflow-launch:.github/workflows/release.yml:build-inventory-security:require-current-only-build-inventory-policy",
        "workflow-launch:.github/workflows/release.yml:build-inventory-security:require-current-only-test-inventory-policy",
        "workflow-launch:.github/workflows/release.yml:build-inventory-security:run-build-inventory-security-suite",
        "workflow-launch:.github/workflows/release.yml:test-inventory-security:require-current-only-test-inventory-policy",
        "workflow-launch:.github/workflows/release.yml:test-inventory-security:run-test-inventory-security-suite",
        "workflow-launch:.github/workflows/release.yml:artifacts:require-current-only-build-inventory-policy",
        "workflow-launch:.github/workflows/release.yml:artifacts:require-current-only-test-inventory-policy",
        "workflow-launch:.github/workflows/release.yml:artifacts:provision-fresh-publication-workspace",
        "workflow-launch:.github/workflows/release.yml:artifacts:verify-publication-workspace",
        *REVIEWED_NEW_WORKFLOW_LAUNCH_FIELDS,
    }
    if identifier not in identifiers:
        raise InventoryError(
            f"no reviewed template for new workflow launch {identifier}"
        )
    return {
        "owner": "release-validation",
        "launch_class": "workflow",
        "detail_status": "process-lifecycle-out-of-scope",
        "compile_for": "host",
        "execute_on": "workflow-runner",
        "cwd_shape": "workflow checkout",
        **REVIEWED_NEW_WORKFLOW_LAUNCH_FIELDS.get(identifier, {}),
    }


def _read_regular_stable_snapshot(
    path: Path | str,
    maximum_bytes: int,
    subject: str,
    *,
    directory_fd: int | None = None,
) -> FrozenInventorySnapshot:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise InventoryError(f"cannot read {subject}: {path}") from exc
    close_error: OSError | None = None
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
        snapshot = FrozenInventorySnapshot(
            bytes=data,
            identity=after_identity,
            sha256=hashlib.sha256(data).hexdigest(),
            mode=after.st_mode,
        )
    except OSError as exc:
        raise InventoryError(f"cannot read {subject}: {path}") from exc
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            close_error = exc
    if close_error is not None:
        raise InventoryError(f"cannot close {subject}: {path}") from close_error
    return snapshot


def _snapshot_matches_node(
    snapshot: FrozenInventorySnapshot, node: repository_snapshot.FrozenNode
) -> bool:
    return (
        node.kind == "regular"
        and node.bytes == snapshot.bytes
        and node.sha256 == snapshot.sha256
        and node.identity.mode == snapshot.mode
        and (
            node.identity.device,
            node.identity.inode,
            node.identity.size,
            node.identity.mtime_ns,
            node.identity.ctime_ns,
        )
        == snapshot.identity
    )


def _snapshot_from_node(
    node: repository_snapshot.FrozenNode,
) -> FrozenInventorySnapshot:
    if node.kind != "regular" or node.bytes is None or node.sha256 is None:
        raise InventoryError("build inventory was not frozen as a regular file")
    return FrozenInventorySnapshot(
        bytes=node.bytes,
        identity=(
            node.identity.device,
            node.identity.inode,
            node.identity.size,
            node.identity.mtime_ns,
            node.identity.ctime_ns,
        ),
        sha256=node.sha256,
        mode=node.identity.mode,
    )


def _parse_refresh_inventory(snapshot: FrozenInventorySnapshot) -> dict[str, Any]:
    try:
        inventory = _strict_json_loads(snapshot.bytes.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InventoryError(f"cannot parse inventory for refresh: {exc}") from exc
    structure_error = _json_structure_error(inventory)
    if structure_error is not None:
        raise InventoryError(structure_error)
    if not isinstance(inventory, dict):
        raise InventoryError("inventory root must be an object")
    return inventory


def _prepare_refreshed_source_candidate(
    root: Path, inventory_path: Path
) -> RefreshedSourceCandidate:
    """Build and fully validate one approved candidate without writing the filesystem."""
    context: DiscoveryContext | None = None
    if os.name == "posix":
        existing_snapshot = _read_regular_stable_snapshot(
            inventory_path,
            SOURCE_REFRESH_MAX_BYTES,
            "build inventory for refresh",
        )
    else:
        context = _make_discovery_context(root, inventory_path)
        if context.inventory_node is None:
            raise InventoryError("build inventory was not frozen")
        existing_snapshot = _snapshot_from_node(context.inventory_node)
    inventory = _parse_refresh_inventory(existing_snapshot)
    existing_projection_error = _reviewed_source_projection_error(inventory)
    if existing_projection_error is not None:
        raise InventoryError(existing_projection_error)

    if context is None:
        context = _make_discovery_context(root, inventory_path)
    if context.inventory_node is None or not _snapshot_matches_node(
        existing_snapshot, context.inventory_node
    ):
        raise InventoryError("build inventory changed after its bounded snapshot")

    observed = discover(root, inventory, context)
    for section in (
        "build_observations",
        "python_launches",
        "workflow_launches",
        "generator_targets",
    ):
        recorded = {item["id"]: item for item in inventory[section]}
        refreshed: list[dict[str, Any]] = []
        for source_item in observed[section]:
            identifier = source_item["id"]
            item = copy.deepcopy(recorded.get(identifier, {}))
            if not item:
                if section == "build_observations":
                    item = _new_test_inventory_observation(identifier, inventory)
                elif section == "python_launches":
                    item = _new_test_inventory_python_launch(identifier)
                elif section == "workflow_launches":
                    item = _new_test_inventory_workflow_launch(identifier)
                else:
                    raise InventoryError(
                        f"no reviewed template for new generator target {identifier}"
                    )
            if identifier == PYTHON_TOOLING_LAUNCH_ID:
                item.pop("runner_contract", None)
            item.update(_reviewed_observation_refresh_fields(identifier))
            item.update(source_item)
            refreshed.append(item)
        inventory[section] = sorted(refreshed, key=lambda item: item["id"])

    inventory["build_roots"] = list(context.build_roots)
    inventory["build_manifests"] = [dict(row) for row in context.build_manifests]
    observations = {item["id"]: item for item in inventory["build_observations"]}
    aggregate = observations[TEST_INVENTORY_AGGREGATE_STEP_ID]
    aggregate_dependencies = (
        {"id": TEST_INVENTORY_RUN_STEP_ID, "condition": "always"},
        {"id": PYTHON_TOOLING_STEP_ID, "condition": "always"},
    )
    migrated_dependency_ids = {
        dependency["id"] for dependency in aggregate_dependencies
    }
    aggregate["direct_dependencies"] = [
        *aggregate_dependencies,
        *[
            dependency
            for dependency in aggregate["direct_dependencies"]
            if dependency.get("id") not in migrated_dependency_ids
        ],
    ]
    inventory["build_root_digests"] = {
        rel: hashlib.sha256(
            _frozen_regular_bytes(context, rel, "build root")
        ).hexdigest()
        for rel in context.build_roots
    }
    inventory["conditional_link_guard_digests"] = {
        item["id"]: item["guard_digest"]
        for item in inventory["build_observations"]
        if item.get("category") == "link"
    }
    inventory["workflow_source_digests"] = {
        item["id"]: item["source_digest"] for item in inventory["workflow_launches"]
    }
    _apply_reviewed_build_inventory_migrations(inventory)
    candidate_bytes = _canonical_inventory_bytes(inventory)
    projection_sha256 = _source_projection_digest(inventory)
    projection_error = _reviewed_source_projection_error(inventory)
    if projection_error is not None:
        raise InventoryError(projection_error)
    candidate_errors = _validate(
        root,
        inventory_path,
        _context=context,
        _inventory_bytes=candidate_bytes,
    )
    if candidate_errors:
        raise InventoryError(
            "refreshed inventory is invalid: " + "; ".join(candidate_errors)
        )
    return RefreshedSourceCandidate(
        inventory=inventory,
        bytes=candidate_bytes,
        expected_snapshot=existing_snapshot,
        projection_sha256=projection_sha256,
    )


def _publication_capability_error() -> str | None:
    if os.name != "posix":
        return "source refresh publication requires POSIX filesystem semantics"
    dir_fd_functions = os.supports_dir_fd
    for function in (
        os.mkdir,
        os.open,
        os.rename,
        os.rmdir,
        os.stat,
        os.unlink,
    ):
        if function not in dir_fd_functions:
            return (
                "source refresh publication requires anchored dir_fd support for "
                f"{function.__name__}"
            )
    follow_functions = os.supports_follow_symlinks
    if os.stat not in follow_functions:
        return "source refresh publication requires no-follow anchored stat support"
    return None


_UNKNOWN_TEMPORARY_IDENTITY = object()


def _inventory_unclaimed_candidate_error(
    summary: str,
    directory_descriptor: int,
    directory_path: Path,
    temporary_name: str,
    cause: BaseException | None = None,
) -> InventoryError:
    candidate_path = directory_path / temporary_name
    try:
        os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        observation = "presence is uncertain"
    else:
        observation = "was observed present"
    error = InventoryError(
        f"{summary}; unclaimed candidate {observation} at {candidate_path}"
    )
    if cause is not None:
        error.__cause__ = cause
    return error


def _verify_inventory_temporary_claim(
    descriptor: int,
    path_metadata: os.stat_result,
    temporary_identity: tuple[int, int],
    expected_temporary_bytes: bytes,
) -> repository_snapshot.ClaimVerification:
    try:
        before = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino) != temporary_identity
            or (path_metadata.st_dev, path_metadata.st_ino) != temporary_identity
            or before.st_size > SOURCE_REFRESH_MAX_BYTES
        ):
            return repository_snapshot.ClaimVerification.FOREIGN
        chunks: list[bytes] = []
        remaining = SOURCE_REFRESH_MAX_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError:
        return repository_snapshot.ClaimVerification.UNKNOWN
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_mode,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_mode,
    )
    if (
        before_identity != after_identity
        or len(data) != after.st_size
        or data != expected_temporary_bytes
        or hashlib.sha256(data).digest()
        != hashlib.sha256(expected_temporary_bytes).digest()
    ):
        return repository_snapshot.ClaimVerification.FOREIGN
    return repository_snapshot.ClaimVerification.MATCH


def _inventory_cleanup_error(
    outcome: repository_snapshot.CleanupOutcome,
) -> InventoryError:
    issue_codes = {issue.code for issue in outcome.issues}
    if issue_codes & {
        "cleanup_claim_destination_fsync_failed",
        "cleanup_claim_source_fsync_failed",
    }:
        summary = "inventory temporary cleanup claim persistence failed"
    elif "cleanup_claimed_foreign" in issue_codes:
        summary = "inventory temporary cleanup claimed unexpected bytes"
    elif issue_codes & {
        "cleanup_claimed_unknown",
        "cleanup_claimed_uninspectable",
        "cleanup_claimed_descriptor_close_uncertain",
    }:
        summary = "inventory temporary cleanup claim could not be checked"
    elif "cleanup_public_name_reappeared" in issue_codes:
        summary = "inventory temporary public pathname reappeared after cleanup claim"
    elif issue_codes & {
        "cleanup_quarantine_descriptor_close_uncertain",
        "cleanup_quarantine_fsync_failed",
        "cleanup_quarantine_teardown_failed",
        "cleanup_source_fsync_failed",
    }:
        summary = "inventory temporary cleanup quarantine removal failed"
    else:
        summary = "inventory temporary cleanup claim failed"
    details: list[str] = []
    for issue in outcome.issues:
        label = {
            "cleanup_claim_destination_fsync_failed": "quarantine directory fsync failed",
            "cleanup_claim_source_fsync_failed": "source directory fsync failed",
            "cleanup_source_fsync_failed": "source directory fsync failed",
        }.get(issue.code, issue.code)
        if issue.error is not None:
            label += f": {issue.error}"
        details.append(label)
    recovery = ", ".join(os.fspath(path) for path in outcome.recovery_paths)
    candidates = ", ".join(os.fspath(path) for path in outcome.candidate_paths)
    message = summary
    if details:
        message += "; " + "; ".join(details)
    if recovery:
        message += f"; recovery material retained as {recovery}"
    if candidates:
        if outcome.public_candidate is repository_snapshot.PublicCandidate.UNKNOWN:
            observation = "presence is uncertain"
        elif outcome.public_candidate is repository_snapshot.PublicCandidate.PRESENT:
            observation = "was observed present"
        else:
            observation = "requires inspection"
        message += f"; unclaimed candidate {observation} at {candidates}"
    elif outcome.public_candidate is not repository_snapshot.PublicCandidate.ABSENT:
        message += (
            "; unclaimed public candidate state is "
            f"{outcome.public_candidate.name.lower()}"
        )
    if (
        outcome.disposition is repository_snapshot.CleanupDisposition.UNADDRESSABLE
        or outcome.arena_binding is not repository_snapshot.ArenaBinding.BOUND
    ):
        status = (
            "unaddressable"
            if outcome.disposition
            is repository_snapshot.CleanupDisposition.UNADDRESSABLE
            else "tainted"
        )
        message += (
            f"; cleanup namespace is {status}; arena binding is "
            f"{outcome.arena_binding.name.lower()}"
        )
    error = InventoryError(message)
    cause = next((issue.error for issue in outcome.issues if issue.error), None)
    if cause is not None:
        error.__cause__ = cause
    return error


def _cleanup_inventory_temporary(
    directory_descriptor: int,
    directory_path: Path,
    temporary_name: str,
    temporary_identity: tuple[int, int],
    expected_temporary_bytes: bytes,
) -> None:
    try:
        outcome = repository_snapshot.claim_and_remove(
            repository_snapshot.DirectoryAnchor(
                directory_descriptor,
                directory_path,
            ),
            temporary_name,
            lambda descriptor, metadata: _verify_inventory_temporary_claim(
                descriptor,
                metadata,
                temporary_identity,
                expected_temporary_bytes,
            ),
            quarantine_prefix=f".{temporary_name}.",
            quarantine_suffix=".quarantine",
        )
    except repository_snapshot.CleanupFailure as exc:
        raise _inventory_cleanup_error(exc.outcome) from exc
    if outcome.disposition not in {
        repository_snapshot.CleanupDisposition.ABSENT,
        repository_snapshot.CleanupDisposition.REMOVED,
    }:
        raise _inventory_cleanup_error(outcome)


def _publish_inventory_atomic(
    inventory_path: Path,
    candidate_bytes: bytes,
    expected_snapshot: FrozenInventorySnapshot,
) -> None:
    """Optimistically reverify and atomically install already-validated bytes.

    POSIX exposes no portable pathname compare-and-swap primitive.  The final
    descriptor-derived comparison therefore rejects observed lost updates but
    does not claim to defeat every non-cooperating external writer.
    """
    if len(candidate_bytes) > SOURCE_REFRESH_MAX_BYTES:
        raise InventoryError(
            f"candidate inventory exceeds {SOURCE_REFRESH_MAX_BYTES} bytes"
        )
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
    projection_error = _reviewed_source_projection_error(candidate)
    if projection_error is not None:
        raise InventoryError(projection_error)
    capability_error = _publication_capability_error()
    if capability_error is not None:
        raise InventoryError(capability_error)

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_descriptor = -1
    descriptor = -1
    temporary_name: str | None = None
    temporary_identity: tuple[int, int] | object | None = None
    temporary_cleanup_prohibited = False
    offset = 0
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
            temporary_identity = _UNKNOWN_TEMPORARY_IDENTITY
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("inventory temporary is not a regular file")
            temporary_identity = (metadata.st_dev, metadata.st_ino)
            break
        else:
            raise OSError("cannot allocate a unique inventory temporary file")

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
            temporary_cleanup_prohibited = True
            raise _inventory_unclaimed_candidate_error(
                "inventory publication temporary descriptor close failed; "
                "exactly one close attempt was made and operating-system "
                f"descriptor state is unknown: {exc}",
                directory_descriptor,
                inventory_path.parent,
                temporary_name,
                exc,
            ) from exc

        if temporary_name is None or not isinstance(temporary_identity, tuple):
            raise InventoryError("inventory publication temporary identity is missing")
        temporary_snapshot = _read_regular_stable_snapshot(
            temporary_name,
            SOURCE_REFRESH_MAX_BYTES,
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
            SOURCE_REFRESH_MAX_BYTES,
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
        finalization_errors: list[Exception] = []
        if descriptor >= 0:
            closing_descriptor = descriptor
            descriptor = -1
            try:
                os.close(closing_descriptor)
            except OSError as exc:
                temporary_cleanup_prohibited = True
                finalization_errors.append(
                    _inventory_unclaimed_candidate_error(
                        "inventory publication temporary descriptor close failed; "
                        "exactly one close attempt was made and operating-system "
                        f"descriptor state is unknown: {exc}",
                        directory_descriptor,
                        inventory_path.parent,
                        temporary_name,
                        exc,
                    )
                )
        cleanup_error: Exception | None = None
        try:
            if (
                temporary_name is not None
                and directory_descriptor >= 0
                and not temporary_cleanup_prohibited
            ):
                if temporary_identity is _UNKNOWN_TEMPORARY_IDENTITY:
                    finalization_errors.append(
                        _inventory_unclaimed_candidate_error(
                            "inventory publication temporary identity is unknown; "
                            "refusing unverified pathname cleanup",
                            directory_descriptor,
                            inventory_path.parent,
                            temporary_name,
                        )
                    )
                elif isinstance(temporary_identity, tuple):
                    _cleanup_inventory_temporary(
                        directory_descriptor,
                        inventory_path.parent,
                        temporary_name,
                        temporary_identity,
                        candidate_bytes[:offset],
                    )
                elif temporary_identity is not None:
                    finalization_errors.append(
                        _inventory_unclaimed_candidate_error(
                            "inventory publication temporary ownership state is "
                            "invalid; refusing unverified pathname cleanup",
                            directory_descriptor,
                            inventory_path.parent,
                            temporary_name,
                        )
                    )
        except Exception as exc:
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
        if finalization_errors:
            if cleanup_error is not None:
                finalization_errors.append(cleanup_error)
            raise InventoryError(
                "inventory publication finalization failed: "
                + "; ".join(str(error) for error in finalization_errors)
            )
        if cleanup_error is not None:
            raise cleanup_error


def refresh_source_derived_inventory(
    root: Path, inventory_path: Path
) -> RefreshedSourceCandidate:
    """Return an approved, fully validated source refresh without publishing it."""
    return _prepare_refreshed_source_candidate(root, inventory_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        help="inventory JSON (default: ROOT/tools/build_inventory.json)",
    )
    parser.add_argument(
        "--refresh-source-derived",
        action="store_true",
        help="refresh deterministic source observations before validation",
    )
    parser.add_argument(
        "--require-current-only",
        action="store_true",
        help="reject an open build source-projection NEXT migration window",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(os.path.abspath(args.root))
    inventory = (
        Path(os.path.abspath(args.inventory))
        if args.inventory
        else root / "tools/build_inventory.json"
    )
    if args.require_current_only and NEXT_SOURCE_PROJECTION_SHA256 is not None:
        print(
            "build inventory error: current-only policy requires "
            "NEXT_SOURCE_PROJECTION_SHA256 to be empty",
            file=sys.stderr,
        )
        return 1
    refreshed_candidate: RefreshedSourceCandidate | None = None
    if args.refresh_source_derived:
        try:
            refreshed_candidate = refresh_source_derived_inventory(root, inventory)
            _publish_inventory_atomic(
                inventory,
                refreshed_candidate.bytes,
                refreshed_candidate.expected_snapshot,
            )
        except InventoryPublicationIndeterminate as exc:
            print(f"build inventory error: {exc}", file=sys.stderr)
            return 3
        except (
            AttributeError,
            IndexError,
            InventoryError,
            KeyError,
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
            RecursionError,
        ) as exc:
            print(f"build inventory error: refresh failed: {exc}", file=sys.stderr)
            return 1
    errors = [] if refreshed_candidate is not None else validate(root, inventory)
    if errors:
        for error in errors:
            print(f"build inventory error: {error}", file=sys.stderr)
        return 1
    print(f"build inventory valid: {inventory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
