#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Shared reproducibility identity for benchmark controller metadata."""

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import benchmark_artifacts

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_GIT_PATH = _REPOSITORY_ROOT / "tools" / "repository_git.py"
_REPOSITORY_GIT_SPEC = importlib.util.spec_from_file_location(
    "_zynum_repository_git", _REPOSITORY_GIT_PATH
)
if _REPOSITORY_GIT_SPEC is None or _REPOSITORY_GIT_SPEC.loader is None:
    raise RuntimeError("unable to load the repository Git policy")
repository_git = importlib.util.module_from_spec(_REPOSITORY_GIT_SPEC)
sys.modules[_REPOSITORY_GIT_SPEC.name] = repository_git
_REPOSITORY_GIT_SPEC.loader.exec_module(repository_git)
_REPOSITORY_SNAPSHOT_PATH = _REPOSITORY_ROOT / "tools" / "repository_snapshot.py"
_REPOSITORY_SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "_zynum_benchmark_repository_snapshot", _REPOSITORY_SNAPSHOT_PATH
)
if _REPOSITORY_SNAPSHOT_SPEC is None or _REPOSITORY_SNAPSHOT_SPEC.loader is None:
    raise RuntimeError("unable to load the repository snapshot policy")
repository_snapshot = importlib.util.module_from_spec(_REPOSITORY_SNAPSHOT_SPEC)
sys.modules[_REPOSITORY_SNAPSHOT_SPEC.name] = repository_snapshot
_REPOSITORY_SNAPSHOT_SPEC.loader.exec_module(repository_snapshot)


SCHEMA_VERSION = 2
PUBLIC_PROJECTION_SCHEMA_VERSION = 1
COVERAGE_SCHEMA_VERSION = 1
BUILD_DECLARATION_SCHEMA_VERSION = 1
BUILD_OPTIMIZATIONS = ("Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall")
MAX_CONTROL_ARTIFACT_BYTES = 16 * 1024 * 1024
BUILD_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._,+:-]{0,255}\Z")
COVERAGE_ENTRY_STRING_FIELDS = (
    "level",
    "stable_id",
    "operation",
    "scalar",
    "implementation",
    "specialization",
    "capability",
    "availability",
    "lifecycle",
    "state",
    "evidence_note",
)
COVERAGE_EVIDENCE_FIELDS = (
    "build",
    "native_correctness",
    "native_performance",
)
COVERAGE_SUMMARY_FIELDS = (
    "total",
    "implemented",
    "experimental",
    "rejected",
    "missing",
    "unsupported",
    "build_tested",
    "native_correctness_tested",
    "native_performance_tested",
)
COVERAGE_ENUM_FIELDS = {
    "level": {"level1", "level2", "level3_gemm", "level3_structured"},
    "availability": {"implemented", "missing", "rejected", "unsupported"},
    "lifecycle": {"experimental", "portable_fallback", "production", "rejected"},
    "state": {
        "aarch64_streaming_sm",
        "aarch64_streaming_za",
        "apple_amx",
        "none",
        "x86_64_amx",
    },
}
PUBLIC_CONTROLLERS = frozenset(
    {
        "run_gemm_sweep_isolated.py",
        "run_level1_report.py",
        "run_level2_report.py",
        "run_rank_k_report.py",
        "run_rotg_latency_report.py",
        "run_symm_report.py",
        "run_triangular_matrix_report.py",
    }
)
_PUBLIC_EXCLUDED_KEYS = frozenset(
    {
        "argv",
        "cwd",
        "detection_source",
        "executable",
        "path",
        "repository_root",
        "snapshot_manifest",
        "status_short",
    }
)
_PUBLIC_METADATA_KEYS = frozenset(
    {
        "aggregate_metric",
        "alphas",
        "banded_profiles",
        "benchmark_identity",
        "betas",
        "binaries",
        "calls_per_sample",
        "case_count_per_library",
        "cases",
        "copy_byte_coverage",
        "copy_byte_sizes",
        "copy_case_policy",
        "copy_only",
        "copy_seconds",
        "correctness_check",
        "detected_cpu_count",
        "diagonals",
        "environment",
        "generated_at_unix",
        "git_revision",
        "groups",
        "harness",
        "interleave_libraries",
        "isolate_kind",
        "isolate_shape",
        "isolation",
        "kinds",
        "level1_stride_pairs",
        "level1_strides",
        "level1_variants",
        "libraries",
        "n",
        "negative_stride_policy",
        "operations",
        "ops",
        "os",
        "packed_profiles",
        "platform",
        "probe",
        "probes",
        "process_metric",
        "process_repeats",
        "python_version",
        "reps",
        "reps_large",
        "reps_small",
        "routines",
        "samples",
        "schedule",
        "seconds",
        "shapes",
        "sides",
        "sizes",
        "source",
        "stable_negative_operations",
        "transposes",
        "uplos",
        "zig_version",
        "zynum_maximum_threads",
        "zynum_maximum_threads_detected",
    }
)
_LEGACY_TOOL_PATH_KEYS = frozenset({"copy_probe", "level1_probe"})
_ABSOLUTE_HOST_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'=:(])(?:file://)?(?:/[^\s\"',;)]*|[A-Za-z]:[\\/][^\s\"',;)]*|\\\\[^\s\"',;)]*)"
)
_DROP_PUBLIC_VALUE = object()


def _reject_duplicate_object_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key {!r}".format(key))
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value):
    raise ValueError("non-standard JSON constant {!r}".format(value))


def _decode_strict_json_object(contents):
    document = json.loads(
        contents.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_object_keys,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(document, dict):
        raise ValueError("JSON document must be an object")
    return document


def _build_token(value):
    if not isinstance(value, str) or BUILD_TOKEN_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "build target/CPU must be a 1-256 character restricted ASCII token"
        )
    return value


def add_identity_arguments(parser):
    parser.add_argument(
        "--build-target",
        type=_build_token,
        metavar="TRIPLE",
        help="Target triple requested when building the measured payload.",
    )
    parser.add_argument(
        "--build-cpu",
        type=_build_token,
        metavar="CPU",
        help="CPU model/features requested when building the measured payload.",
    )
    parser.add_argument(
        "--build-optimize",
        choices=BUILD_OPTIMIZATIONS,
        help="Zig optimization mode requested for the measured payload.",
    )
    parser.add_argument(
        "--registry-id",
        action="append",
        default=[],
        metavar="ID",
        help=(
            "Stable Zynum kernel registry ID exercised by this run. May be "
            "repeated; IDs are validated against docs/kernel_coverage.json."
        ),
    )
    parser.add_argument(
        "--selected-path",
        action="append",
        default=[],
        metavar="SCOPE=ID",
        help=(
            "Observed or deliberately forced selection for a case/shape scope. "
            "May be repeated and implies --registry-id for the selected ID."
        ),
    )
    parser.add_argument(
        "--target-capability",
        action="append",
        default=[],
        metavar="CAPABILITY",
        help=(
            "Capability of the measured artifact/host that cannot be inferred "
            "reliably by the controller. May be repeated."
        ),
    )
    parser.add_argument(
        "--source-identity",
        metavar="JSON",
        help=(
            "Source snapshot exported by the synchronizing workspace when the "
            "benchmark directory is not itself a Git worktree."
        ),
    )


def sha256_file(path):
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command):
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def repository_root():
    return Path(__file__).resolve().parents[2]


def _control_artifact(root, path):
    anchored_root = Path(os.path.abspath(root))
    supplied_path = Path(os.path.abspath(path))
    try:
        relative = supplied_path.relative_to(anchored_root).as_posix()
    except ValueError as error:
        raise ValueError(
            "control artifact must be beneath the repository root"
        ) from error
    return repository_snapshot.capture_control_artifact(
        anchored_root,
        relative,
        max_bytes=MAX_CONTROL_ARTIFACT_BYTES,
    )


def source_snapshot(root=None, identity_file=None):
    root = Path(os.path.abspath(root or repository_root()))

    if identity_file:
        path = Path(os.path.abspath(identity_file))
        try:
            artifact = _control_artifact(root, path)
            exported = _decode_strict_json_object(artifact.bytes)
        except (ValueError, repository_snapshot.RepositorySnapshotError) as error:
            raise ValueError(
                "cannot read --source-identity {}: {}".format(path, error)
            ) from error
        revision = exported.get("revision")
        if not isinstance(revision, str) or revision == "":
            raise ValueError(
                "--source-identity must be a JSON object with a non-empty revision"
            )
        optional_strings = {}
        for field in (
            "branch",
            "repository_root",
            "status_short",
            "snapshot_tree_sha256",
            "snapshot_created_utc",
        ):
            value = exported.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    "--source-identity {} must be null or a string".format(field)
                )
            optional_strings[field] = value
        exported_dirty = exported.get("dirty")
        if exported_dirty is not None and type(exported_dirty) is not bool:
            raise ValueError("--source-identity dirty must be null or a boolean")
        return {
            "repository_root": optional_strings["repository_root"],
            "revision": revision,
            "branch": optional_strings["branch"],
            "dirty": exported_dirty,
            "status_short": optional_strings["status_short"],
            "identity_status": "exported",
            "identity_diagnostic": "source_identity_manifest_loaded",
            "cleanliness_status": (
                "known" if exported_dirty is not None else "unknown"
            ),
            "snapshot_manifest": str(path),
            "snapshot_manifest_sha256": artifact.sha256,
            "snapshot_tree_sha256": optional_strings["snapshot_tree_sha256"],
            "snapshot_created_utc": optional_strings["snapshot_created_utc"],
        }

    def unresolved(status, diagnostic):
        return {
            "repository_root": str(root),
            "revision": None,
            "branch": None,
            "dirty": None,
            "status_short": None,
            "identity_status": status,
            "identity_diagnostic": diagnostic,
            "cleanliness_status": "unknown",
        }

    try:
        repository = repository_git.open_repository(root)
    except repository_git.RepositoryGitUnavailable:
        return unresolved("unavailable", "git_executable_unavailable")
    except repository_git.RepositoryGitError:
        return unresolved("unreadable", "git_repository_boundary_unreadable")
    if repository is None:
        return unresolved("no_git", "source_has_no_git_marker")

    try:
        identity = repository.observe_identity()
    except repository_git.RepositoryGitError:
        return unresolved("unreadable", "git_identity_unreadable")
    if identity.revision is None:
        return unresolved("unavailable", "git_head_unavailable")
    status = "\n".join(identity.status_lines)

    return {
        "repository_root": str(repository.root),
        "revision": identity.revision,
        "branch": identity.branch,
        "dirty": bool(status),
        "status_short": status,
        "identity_status": "git",
        "identity_diagnostic": "git_source_identity_loaded",
        "cleanliness_status": "known",
    }


def _decode_zig_string(literal):
    if len(literal) < 2 or literal[0] != '"' or literal[-1] != '"':
        raise ValueError("expected a Zig string literal")

    result = []
    index = 1
    escapes = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "\\": "\\",
        '"': '"',
        "'": "'",
    }
    while index < len(literal) - 1:
        character = literal[index]
        if character in "\r\n":
            raise ValueError("unescaped newline in Zig string literal")
        if character != "\\":
            result.append(character)
            index += 1
            continue

        index += 1
        if index >= len(literal) - 1:
            raise ValueError("truncated escape in Zig string literal")
        escape = literal[index]
        if escape in escapes:
            result.append(escapes[escape])
            index += 1
        elif escape == "x":
            digits = literal[index + 1 : index + 3]
            if len(digits) != 2:
                raise ValueError("truncated hexadecimal escape in Zig string literal")
            try:
                result.append(chr(int(digits, 16)))
            except ValueError as error:
                raise ValueError(
                    "invalid hexadecimal escape in Zig string literal"
                ) from error
            index += 3
        elif escape == "u" and literal[index + 1 : index + 2] == "{":
            closing = literal.find("}", index + 2)
            if closing == -1 or closing >= len(literal) - 1:
                raise ValueError("unterminated Unicode escape in Zig string literal")
            digits = literal[index + 2 : closing]
            try:
                codepoint = int(digits, 16)
                result.append(chr(codepoint))
            except (ValueError, OverflowError) as error:
                raise ValueError(
                    "invalid Unicode escape in Zig string literal"
                ) from error
            index = closing + 1
        else:
            raise ValueError("unsupported escape in Zig string literal")
    return "".join(result)


def _zig_object_fields(environment):
    """Parse top-level fields from `zig env` without evaluating Zig source."""

    text = environment.strip()
    if not text.startswith(".{"):
        raise ValueError("output is neither JSON nor a Zig object literal")

    fields = {}
    index = 2
    length = len(text)
    while True:
        while index < length and text[index].isspace():
            index += 1
        if index < length and text.startswith("//", index):
            newline = text.find("\n", index + 2)
            index = length if newline == -1 else newline + 1
            continue
        if index >= length:
            raise ValueError("unterminated Zig object literal")
        if text[index] == "}":
            index += 1
            if text[index:].strip():
                raise ValueError("unexpected content after Zig object literal")
            return fields
        if text[index] != ".":
            raise ValueError("expected a Zig object field")

        index += 1
        name_start = index
        while index < length and (text[index].isalnum() or text[index] == "_"):
            index += 1
        name = text[name_start:index]
        if not name or name in fields:
            raise ValueError("invalid or duplicate Zig object field")
        while index < length and text[index].isspace():
            index += 1
        if index >= length or text[index] != "=":
            raise ValueError("expected '=' after Zig object field")

        index += 1
        while index < length and text[index].isspace():
            index += 1
        value_start = index
        delimiters = []
        in_string = False
        while index < length:
            character = text[index]
            if in_string:
                if character == "\\":
                    index += 2
                    continue
                if character == '"':
                    in_string = False
                index += 1
                continue
            if character == '"':
                in_string = True
                index += 1
                continue
            if character in "{[(":
                delimiters.append({"{": "}", "[": "]", "(": ")"}[character])
                index += 1
                continue
            if delimiters and character == delimiters[-1]:
                delimiters.pop()
                index += 1
                continue
            if not delimiters and character == ",":
                fields[name] = text[value_start:index].strip()
                index += 1
                break
            if not delimiters and character == "}":
                raise ValueError("Zig object field is missing a trailing comma")
            index += 1
        else:
            raise ValueError("unterminated Zig object field")


def _parse_zig_environment(environment):
    try:
        parsed = json.loads(environment)
    except json.JSONDecodeError:
        fields = _zig_object_fields(environment)
        format_name = "zig-object"
        try:
            target = _decode_zig_string(fields["target"])
        except KeyError as error:
            raise ValueError("zig env object has no target field") from error
        zig_exe_literal = fields.get("zig_exe")
        zig_exe = (
            _decode_zig_string(zig_exe_literal) if zig_exe_literal is not None else None
        )
    else:
        if not isinstance(parsed, dict):
            raise ValueError("zig env JSON is not an object")
        format_name = "json"
        target = parsed.get("target")
        zig_exe = parsed.get("zig_exe")

    if not isinstance(target, str) or not target:
        raise ValueError("zig env target is not a non-empty string")
    if zig_exe is not None and (not isinstance(zig_exe, str) or not zig_exe):
        raise ValueError("zig env zig_exe is not a non-empty string")
    return target, zig_exe, format_name


def compiler_snapshot():
    executable = shutil.which("zig")
    version = command_output(["zig", "version"])
    environment = command_output(["zig", "env"])
    target = None
    if environment is None:
        diagnostic = {
            "status": "unavailable",
            "classification": "zig_env_command_failed",
            "message": "zig env did not produce output",
        }
    else:
        try:
            target, reported_executable, format_name = _parse_zig_environment(
                environment
            )
        except ValueError as error:
            diagnostic = {
                "status": "invalid",
                "classification": "zig_env_parse_error",
                "message": str(error),
            }
        else:
            executable = reported_executable or executable
            diagnostic = {
                "status": "ok",
                "classification": "zig_env_parsed",
                "format": format_name,
            }
    return {
        "name": "zig",
        "version": version,
        "executable": executable,
        "default_target": target,
        "default_target_diagnostic": diagnostic,
    }


def _linux_capabilities():
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return []
    keys = ("flags", "features")
    for line in cpuinfo.read_text(errors="replace").splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip().lower() in keys:
            return sorted(set(value.strip().lower().split()))
    return []


def _macos_capabilities():
    values = []
    for name in (
        "machdep.cpu.features",
        "machdep.cpu.leaf7_features",
        "machdep.cpu.extfeatures",
        "hw.optional.arm.FEAT_SVE",
        "hw.optional.arm.FEAT_SME",
        "hw.optional.arm.FEAT_SME2",
        "hw.optional.arm.FEAT_SME2P1",
    ):
        value = command_output(["sysctl", "-n", name])
        if value is None:
            continue
        if name.startswith("hw.optional"):
            if value.strip() == "1":
                values.append(name.rsplit(".", 1)[-1].lower())
        else:
            values.extend(value.lower().split())
    return sorted(set(values))


def host_snapshot():
    system = platform.system()
    if system == "Linux":
        detected = _linux_capabilities()
        detection_source = "/proc/cpuinfo"
    elif system == "Darwin":
        detected = _macos_capabilities()
        detection_source = "sysctl"
    else:
        detected = []
        detection_source = "unavailable"
    return {
        "platform": platform.platform(),
        "system": system,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "detected_capabilities": detected,
        "detection_source": detection_source,
    }


def build_declaration(args):
    requested = {
        "target_triple": getattr(args, "build_target", None),
        "cpu": getattr(args, "build_cpu", None),
        "optimization": getattr(args, "build_optimize", None),
    }
    for field in ("target_triple", "cpu"):
        value = requested[field]
        if value is not None:
            try:
                requested[field] = _build_token(value)
            except ValueError as exc:
                raise ValueError(f"invalid payload build {field}: {exc}") from exc
    optimization = requested["optimization"]
    if optimization is not None and optimization not in BUILD_OPTIMIZATIONS:
        raise ValueError(
            "payload build optimization must be one of {}".format(
                ", ".join(BUILD_OPTIMIZATIONS)
            )
        )

    missing_fields = [field for field, value in requested.items() if value is None]
    if len(missing_fields) == len(requested):
        declaration_status = "unspecified"
    elif missing_fields:
        declaration_status = "partial"
    else:
        declaration_status = "complete"
    return {
        "schema_version": BUILD_DECLARATION_SCHEMA_VERSION,
        "requested": requested,
        "declaration_status": declaration_status,
        "missing_fields": missing_fields,
        "source": {
            "kind": "controller_cli",
            "options": {
                "target_triple": "--build-target",
                "cpu": "--build-cpu",
                "optimization": "--build-optimize",
            },
        },
        "validation": {
            "syntax": "valid",
            "claim_scope": "requested_configuration_only",
            "artifact_match": "not_verified",
        },
    }


def _coverage_document_stable_ids(document):
    if not isinstance(document, dict):
        raise ValueError("document is not an object")
    schema_version = document.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != COVERAGE_SCHEMA_VERSION
    ):
        raise ValueError("schema_version must be {}".format(COVERAGE_SCHEMA_VERSION))
    if (
        not isinstance(document.get("generator"), str)
        or not document["generator"].strip()
    ):
        raise ValueError("generator must be a non-empty string")
    streaming_vector_bytes = document.get("streaming_vector_bytes")
    if (
        not isinstance(streaming_vector_bytes, int)
        or isinstance(streaming_vector_bytes, bool)
        or streaming_vector_bytes <= 0
    ):
        raise ValueError("streaming_vector_bytes must be a positive integer")

    summary = document.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")
    for field in COVERAGE_SUMMARY_FIELDS:
        value = summary.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("summary.{} must be a non-negative integer".format(field))

    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError("entries must be an array")

    stable_ids = []
    expected_summary = {field: 0 for field in COVERAGE_SUMMARY_FIELDS}
    expected_summary["total"] = len(entries)
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("entries[{}] must be an object".format(index))
        for field in COVERAGE_ENTRY_STRING_FIELDS:
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "entries[{}].{} must be a non-empty string".format(index, field)
                )
        for field, allowed_values in COVERAGE_ENUM_FIELDS.items():
            if entry[field] not in allowed_values:
                raise ValueError(
                    "entries[{}].{} has unsupported value {!r}".format(
                        index, field, entry[field]
                    )
                )

        stable_id = entry["stable_id"]
        if stable_id in stable_ids:
            raise ValueError("duplicate stable_id {!r}".format(stable_id))
        stable_ids.append(stable_id)

        evidence = entry.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("entries[{}].evidence must be an object".format(index))
        for field in COVERAGE_EVIDENCE_FIELDS:
            if not isinstance(evidence.get(field), bool):
                raise ValueError(
                    "entries[{}].evidence.{} must be a boolean".format(index, field)
                )

        if entry["availability"] in (
            "implemented",
            "rejected",
            "missing",
            "unsupported",
        ):
            expected_summary[entry["availability"]] += 1
        if (
            entry["availability"] == "implemented"
            and entry["lifecycle"] == "experimental"
        ):
            expected_summary["experimental"] += 1
        expected_summary["build_tested"] += int(evidence["build"])
        expected_summary["native_correctness_tested"] += int(
            evidence["native_correctness"]
        )
        expected_summary["native_performance_tested"] += int(
            evidence["native_performance"]
        )

    for field, expected in expected_summary.items():
        if summary[field] != expected:
            raise ValueError(
                "summary.{} is {}, expected {} from entries".format(
                    field, summary[field], expected
                )
            )
    return stable_ids


def coverage_snapshot(root=None):
    root = Path(os.path.abspath(root or repository_root()))
    path = root / "docs" / "kernel_coverage.json"
    result = {
        "path": str(path),
        "sha256": None,
        "schema_version": None,
        "generator": None,
        "summary": None,
        "stable_ids": [],
        "authority_status": "missing",
        "authority_diagnostic": "kernel_coverage_missing",
    }
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return result
    except OSError:
        result.update(
            {
                "authority_status": "unreadable",
                "authority_diagnostic": "kernel_coverage_stat_failed",
            }
        )
        return result
    if not stat.S_ISREG(metadata.st_mode):
        result.update(
            {
                "authority_status": "unreadable",
                "authority_diagnostic": "kernel_coverage_unsafe_file_type",
            }
        )
        return result
    try:
        artifact = _control_artifact(root, path)
    except (ValueError, repository_snapshot.RepositorySnapshotError):
        result.update(
            {
                "authority_status": "unreadable",
                "authority_diagnostic": "kernel_coverage_read_failed",
            }
        )
        return result
    result["sha256"] = artifact.sha256
    try:
        document = _decode_strict_json_object(artifact.bytes)
    except UnicodeDecodeError:
        result.update(
            {
                "authority_status": "unreadable",
                "authority_diagnostic": "kernel_coverage_text_decode_failed",
            }
        )
        return result
    except ValueError:
        result.update(
            {
                "authority_status": "malformed",
                "authority_diagnostic": "kernel_coverage_invalid_json",
            }
        )
        return result
    try:
        stable_ids = _coverage_document_stable_ids(document)
    except ValueError as error:
        result.update(
            {
                "authority_status": "malformed",
                "authority_diagnostic": "kernel_coverage_invalid_document",
                "authority_detail": str(error),
            }
        )
        return result
    result.update(
        {
            "schema_version": document.get("schema_version"),
            "generator": document.get("generator"),
            "summary": document.get("summary"),
            "stable_ids": stable_ids,
            "authority_status": "valid" if stable_ids else "empty",
            "authority_diagnostic": (
                "kernel_coverage_loaded"
                if stable_ids
                else "kernel_coverage_has_no_stable_ids"
            ),
        }
    )
    return result


def parse_selected_paths(values):
    result = []
    for value in values:
        scope, separator, registry_id = value.rpartition("=")
        scope = scope.strip()
        registry_id = registry_id.strip()
        if not separator or not scope or not registry_id:
            raise ValueError("--selected-path must be SCOPE=ID, got {!r}".format(value))
        result.append({"scope": scope, "registry_id": registry_id})
    return result


def artifact_records(items):
    """Compatibility live-hashing projection for unmigrated controllers only."""

    records = []
    for name, path in items:
        records.append(
            {
                "name": name,
                "path": path,
                "sha256": sha256_file(path),
            }
        )
    return records


def _frozen_artifact_records(items, expected_role):
    records = []
    for item in items:
        if not isinstance(item, benchmark_artifacts.FrozenArtifact):
            raise TypeError(
                "frozen artifact identity inputs must be FrozenArtifact instances"
            )
        if item.role != expected_role:
            raise ValueError(
                "frozen artifact role {!r} does not match {!r} identity projection".format(
                    item.role, expected_role
                )
            )
        record = item.metadata_record()
        if list(record) != ["name", "path", "sha256"]:
            raise ValueError("frozen artifact metadata record has an unstable schema")
        if record != {
            "name": item.name,
            "path": item.path,
            "sha256": item.sha256,
        }:
            raise ValueError("frozen artifact metadata projection is inconsistent")
        records.append(record)
    return records


def _collect_benchmark_identity_records(
    args, *, library_records, binary_records, root=None
):
    root = Path(root or repository_root())
    source = source_snapshot(root, getattr(args, "source_identity", None))
    build = build_declaration(args)
    coverage = coverage_snapshot(root)
    selected_paths = parse_selected_paths(getattr(args, "selected_path", []))
    registry_ids = _unique(
        list(getattr(args, "registry_id", []))
        + [entry["registry_id"] for entry in selected_paths]
    )
    known_ids = set(coverage["stable_ids"])
    unknown_ids = [
        registry_id for registry_id in registry_ids if registry_id not in known_ids
    ]
    if registry_ids and coverage["authority_status"] != "valid":
        raise ValueError(
            "kernel registry IDs require valid docs/kernel_coverage.json authority; "
            "coverage is {} ({})".format(
                coverage["authority_status"], coverage["authority_diagnostic"]
            )
        )
    if unknown_ids:
        raise ValueError(
            "kernel registry IDs are absent from docs/kernel_coverage.json: {}".format(
                ", ".join(unknown_ids)
            )
        )
    coverage = dict(coverage)
    coverage.pop("stable_ids", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "controller": {
            "host": host_snapshot(),
            "compiler": compiler_snapshot(),
            "python": {
                "version": sys.version,
                "executable": sys.executable,
            },
        },
        "payload": {
            "build": build,
            "declared_capabilities": _unique(getattr(args, "target_capability", [])),
            "kernel_selection": {
                "registry_ids": registry_ids,
                "selected_paths": selected_paths,
                "selection_observation": (
                    "explicit case/shape-to-registry mapping supplied by controller"
                    if selected_paths
                    else "no per-call selection mapping supplied; do not infer a selected kernel from declared capability alone"
                ),
                "coverage_artifact": coverage,
            },
            "artifacts": {
                "hash_claim_scope": (
                    "content_identity_only; artifact hashes do not prove build flags"
                ),
                "binaries": binary_records,
                "libraries": library_records,
            },
        },
    }


def collect_benchmark_identity_from_frozen(
    args, *, libraries=(), binaries=(), root=None
):
    """Collect identity from already-frozen artifact records only.

    This is the migration target for benchmark controllers.  It never hashes a
    live executable or library path; each record is projected by the owning
    :class:`benchmark_artifacts.ArtifactSnapshotSet` after private-copy
    verification.
    """

    library_records = _frozen_artifact_records(libraries, "library")
    binary_records = _frozen_artifact_records(binaries, "binary")
    return _collect_benchmark_identity_records(
        args,
        library_records=library_records,
        binary_records=binary_records,
        root=root,
    )


def collect_benchmark_identity(args, libraries=(), binaries=(), root=None):
    """Compatibility collector for controllers that still pass live paths.

    New and migrated controllers must use
    :func:`collect_benchmark_identity_from_frozen` instead.
    """

    return _collect_benchmark_identity_records(
        args,
        library_records=artifact_records(libraries),
        binary_records=artifact_records(binaries),
        root=root,
    )


def identity_snapshot(args, libraries=(), binaries=(), root=None):
    """Compatibility wrapper for the preferred identity collector."""

    return collect_benchmark_identity(args, libraries, binaries, root)


def legacy_source_snapshot(source):
    """Project private diagnostics onto the legacy source metadata shape.

    The public serializer removes ``status_short`` from this compatibility
    shape before publication.
    """

    return {
        "revision": source.get("revision"),
        "branch": source.get("branch"),
        "dirty": source.get("dirty"),
        "status_short": source.get("status_short"),
    }


def source_git_revision(source):
    """Project the legacy Git revision from an already-collected source."""

    return source.get("revision")


def _contains_absolute_host_path(value):
    return bool(_ABSOLUTE_HOST_PATH_PATTERN.search(value))


def _public_value(value):
    """Project a private diagnostic value into the public metadata domain."""

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("benchmark metadata object keys must be strings")
            if key in _PUBLIC_EXCLUDED_KEYS or _contains_absolute_host_path(key):
                continue
            if key in _LEGACY_TOOL_PATH_KEYS and isinstance(item, str):
                continue
            projected = _public_value(item)
            if projected is not _DROP_PUBLIC_VALUE:
                result[key] = projected
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            projected = _public_value(item)
            if projected is not _DROP_PUBLIC_VALUE:
                result.append(projected)
        return result
    if isinstance(value, str) and _contains_absolute_host_path(value):
        return _DROP_PUBLIC_VALUE
    return value


def public_safe_projection(private_metadata, *, controller, parameter_keys=()):
    """Return the publishable projection of private controller diagnostics.

    Controllers may collect absolute paths, raw process arguments, and detailed
    Git status for local failure diagnosis.  Published ``.meta.json`` files use
    this separate, allowlisted projection and retain only logical artifact
    names/digests and non-host-specific reproduction declarations.
    """

    if not isinstance(private_metadata, dict):
        raise TypeError("private benchmark metadata must be an object")
    if controller not in PUBLIC_CONTROLLERS:
        raise ValueError(
            "unsupported public benchmark controller {!r}".format(controller)
        )
    projected = _public_value(
        {
            key: value
            for key, value in private_metadata.items()
            if key in _PUBLIC_METADATA_KEYS
        }
    )
    if projected is _DROP_PUBLIC_VALUE:
        raise ValueError("benchmark metadata object cannot be projected")

    parameters = {}
    for key in parameter_keys:
        if not isinstance(key, str):
            raise TypeError("public command parameter keys must be strings")
        if key not in projected:
            raise ValueError(
                "public command parameter {!r} is absent after projection".format(key)
            )
        parameters[key] = projected[key]
    projected["command"] = {
        "controller": controller,
        "parameters": parameters,
    }
    projected["metadata_projection"] = {
        "schema_version": PUBLIC_PROJECTION_SCHEMA_VERSION,
        "audience": "public",
        "private_diagnostics": "excluded",
    }
    return projected


def serialize_public_metadata(private_metadata, *, controller, parameter_keys=()):
    """Serialize the sole public-safe metadata projection as deterministic JSON."""

    public_metadata = public_safe_projection(
        private_metadata,
        controller=controller,
        parameter_keys=parameter_keys,
    )
    return (
        json.dumps(public_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _unique(values):
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
