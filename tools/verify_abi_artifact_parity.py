#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Build fresh Darwin ABI artifacts and verify them against a frozen baseline.

This verifier deliberately does not run package or source-archive consumers.  Its
subject is the two installed binary artifacts produced by the exact command in
the checked-in ABI policy.  The byte parser remains ``abi_artifact_parity.py``;
this file owns process isolation, filesystem provenance, independent source and
symbol accounting, and system-tool cross-checks.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_NAME = "zynum.abi-fresh-artifact-parity"
SCHEMA_VERSION = 1
POLICY_SCHEMA_VERSION = 3
OBSERVATION_SCHEMA = {"name": "zynum.abi-baseline-observation", "version": 3}
RECEIPT_SCHEMA = "zynum.private-abi-artifact-build-receipt-v1"
PARSER_ID = "tools/abi_artifact_parity.py"
BENCHMARK_ARTIFACTS_ID = "bench/tools/benchmark_artifacts.py"
VERIFIER_ID = "tools/verify_abi_artifact_parity.py"
MAX_CAPTURE_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_HASH_BYTES = 512 * 1024 * 1024
PROCESS_CLEANUP_GRACE_SECONDS = 1.0
PLACEHOLDERS = {
    "<temporary-install-prefix>": "install",
    "<isolated-local-cache>": "local-cache",
    "<isolated-global-cache>": "global-cache",
}


class VerificationError(RuntimeError):
    """One fail-closed verification condition was not met."""

    def __init__(self, code: str, message: str, *, locator: str = "verifier") -> None:
        self.code = code
        self.locator = locator
        self.message = message
        super().__init__(f"{locator}: {message}")

    def json(self) -> dict[str, str]:
        return {"code": self.code, "locator": self.locator, "message": self.message}


class SensitivePathRedactor:
    """Deterministically replace registered generated roots in report values."""

    def __init__(self) -> None:
        self._roots: dict[str, str] = {}

    def add_root(self, path: str | os.PathLike[str], placeholder: str) -> None:
        if not placeholder.startswith("<") or not placeholder.endswith(">"):
            raise ValueError("sensitive-path placeholders must be stable tokens")
        try:
            absolute = os.path.abspath(os.fspath(path))
        except (TypeError, ValueError, OSError):
            return
        if absolute == os.path.abspath(os.sep):
            return
        candidates = {absolute}
        try:
            candidates.add(os.path.realpath(absolute))
        except OSError:
            pass
        for candidate in candidates:
            if candidate and candidate != os.path.abspath(os.sep):
                self._roots[candidate] = placeholder

    def add_snapshot_artifact(self, artifact: Any) -> str:
        execution_path = artifact.execution_path
        self.add_root(Path(execution_path).parent, "<private-artifact-root>")
        return execution_path

    def sanitize_text(self, value: str) -> str:
        for root, placeholder in sorted(
            self._roots.items(), key=lambda item: (-len(item[0]), item[0])
        ):
            value = value.replace(root, placeholder)
        return value

    def sanitize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.sanitize_text(value)
        if isinstance(value, bytes):
            for root, placeholder in sorted(
                self._roots.items(), key=lambda item: (-len(item[0]), item[0])
            ):
                value = value.replace(os.fsencode(root), placeholder.encode("ascii"))
            return value
        if isinstance(value, Mapping):
            return {
                self.sanitize_value(key): self.sanitize_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.sanitize_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.sanitize_value(item) for item in value)
        return value


@dataclasses.dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    truncated: bool
    cleanup_failure: str | None
    duration_ms: int

    @property
    def ok(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.truncated
            and self.cleanup_failure is None
        )

    def sanitized(self) -> dict[str, Any]:
        return {
            "argv0": Path(self.argv[0]).name if self.argv else None,
            "argument_count": len(self.argv),
            "returncode": self.returncode,
            "stdout_size": len(self.stdout),
            "stderr_size": len(self.stderr),
            "stdout_sha256": sha256_bytes(self.stdout),
            "stderr_sha256": sha256_bytes(self.stderr),
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "cleanup_failure": self.cleanup_failure,
            "duration_ms": self.duration_ms,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


@dataclasses.dataclass(frozen=True)
class ExactRegularFile:
    """One bounded regular-file read tied to a stable leaf and descriptor."""

    data: bytes
    size: int
    sha256: str
    metadata: os.stat_result


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_exact_regular_file(
    path: Path, *, max_bytes: int, locator: str
) -> ExactRegularFile:
    """Read exactly the descriptor's initial size and reject path/content drift."""

    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    try:
        before_leaf = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise VerificationError(
            "file_stat_failed", "file leaf cannot be inspected", locator=locator
        ) from exc
    if not stat.S_ISREG(before_leaf.st_mode):
        raise VerificationError(
            "file_not_regular", "file leaf must be regular", locator=locator
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise VerificationError(
            "file_open_failed", "file leaf cannot be opened safely", locator=locator
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stable_file_identity(
            opened
        ) != _stable_file_identity(before_leaf):
            raise VerificationError(
                "file_leaf_rebound",
                "file leaf changed while it was opened",
                locator=locator,
            )
        size = opened.st_size
        if size < 0 or size > max_bytes:
            raise VerificationError(
                "file_size_out_of_range",
                f"file exceeds the {max_bytes}-byte verifier limit",
                locator=locator,
            )
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            try:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
            except OSError as exc:
                raise VerificationError(
                    "file_read_failed", "exact file read failed", locator=locator
                ) from exc
            if not chunk:
                raise VerificationError(
                    "file_short_read",
                    "file ended before its frozen size",
                    locator=locator,
                )
            if len(chunk) > remaining:
                raise VerificationError(
                    "file_read_overflow",
                    "file read exceeded its frozen size",
                    locator=locator,
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        try:
            growth = os.read(descriptor, 1)
        except OSError as exc:
            raise VerificationError(
                "file_read_failed", "growth probe failed", locator=locator
            ) from exc
        if growth:
            raise VerificationError(
                "file_grew_during_read",
                "file supplied bytes beyond its frozen size",
                locator=locator,
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_leaf = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise VerificationError(
            "file_leaf_rebound",
            "file leaf disappeared after it was read",
            locator=locator,
        ) from exc
    identity = _stable_file_identity(opened)
    if (
        not stat.S_ISREG(after.st_mode)
        or not stat.S_ISREG(after_leaf.st_mode)
        or _stable_file_identity(after) != identity
        or _stable_file_identity(after_leaf) != identity
    ):
        raise VerificationError(
            "file_changed_during_read",
            "file identity or metadata changed during its exact read",
            locator=locator,
        )
    data = b"".join(chunks)
    if len(data) != size:
        raise VerificationError(
            "file_short_read",
            "file read did not produce its frozen size",
            locator=locator,
        )
    return ExactRegularFile(data, size, sha256_bytes(data), opened)


def sha256_file(path: Path, *, locator: str | None = None) -> str:
    return read_exact_regular_file(
        path,
        max_bytes=MAX_HASH_BYTES,
        locator=locator or str(path),
    ).sha256


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(
                "duplicate_json_key", f"duplicate JSON key {key!r}", locator="json"
            )
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise VerificationError(
        "nonfinite_json_number",
        f"non-finite JSON number {value!r} is not permitted",
        locator="json",
    )


def load_json(path: Path, *, locator: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = read_exact_regular_file(
            path, max_bytes=MAX_JSON_BYTES, locator=locator
        ).data
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except VerificationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("invalid_json", str(exc), locator=locator) from exc
    if not isinstance(value, dict):
        raise VerificationError(
            "invalid_json_type", "expected a JSON object", locator=locator
        )
    return value, raw


def load_parity_module(root: Path) -> Any:
    path = root / PARSER_ID
    spec = importlib.util.spec_from_file_location("zynum_abi_artifact_parity", path)
    if spec is None or spec.loader is None:
        raise VerificationError("parser_import_failed", "could not create module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise VerificationError("parser_import_failed", str(exc)) from exc
    return module


def load_benchmark_artifacts_module(root: Path) -> Any:
    """Dynamically load the shared artifact snapshot owner from the repository."""

    path = root / BENCHMARK_ARTIFACTS_ID
    spec = importlib.util.spec_from_file_location("zynum_benchmark_artifacts", path)
    if spec is None or spec.loader is None:
        raise VerificationError(
            "artifact_snapshot_import_failed", "could not create module spec"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise VerificationError(
            "artifact_snapshot_import_failed", "could not load artifact snapshot owner"
        ) from exc
    return module


def _snapshot_failure(exc: BaseException) -> VerificationError:
    code = getattr(exc, "code", "artifact_snapshot_failed")
    if not isinstance(code, str) or not code:
        code = "artifact_snapshot_failed"
    return VerificationError(
        code,
        "private artifact snapshot verification or cleanup failed",
        locator="artifact_snapshot",
    )


def snapshot_artifact_bytes(
    artifact: Any, *, locator: str, redactor: SensitivePathRedactor | None = None
) -> ExactRegularFile:
    """Read one freshly verified private snapshot and bind its captured metadata."""

    path = Path(
        artifact.execution_path
        if redactor is None
        else redactor.add_snapshot_artifact(artifact)
    )
    captured = read_exact_regular_file(path, max_bytes=MAX_HASH_BYTES, locator=locator)
    if captured.size != artifact.size or captured.sha256 != artifact.sha256:
        raise VerificationError(
            "artifact_snapshot_metadata_mismatch",
            "private artifact bytes differ from captured metadata",
            locator=locator,
        )
    return captured


def _terminate_group(process: subprocess.Popen[bytes]) -> str | None:
    """Ensure the process group is gone; report any inability to prove cleanup."""

    # ``start_new_session=True`` makes the child's PID its process-group ID.
    # Keep that known ID even after the direct child exits so descendants
    # cannot escape cleanup in the parent-exited-first case.
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return None
    except OSError as exc:
        return f"SIGTERM process-group cleanup failed: {exc}"
    deadline = time.monotonic() + PROCESS_CLEANUP_GRACE_SECONDS
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return None
        except OSError as exc:
            return f"process-group cleanup probe failed: {exc}"
        time.sleep(0.01)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return None
    except OSError as exc:
        return f"SIGKILL process-group cleanup failed: {exc}"
    deadline = time.monotonic() + PROCESS_CLEANUP_GRACE_SECONDS
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return None
        except OSError as exc:
            return f"post-SIGKILL process-group probe failed: {exc}"
        time.sleep(0.01)
    return "process group survived SIGKILL"


def run_bounded(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    max_output: int = MAX_CAPTURE_BYTES,
) -> CommandResult:
    """Run one command in a new session with bounded capture and hard cleanup."""

    if not argv or any(not isinstance(item, str) or "\0" in item for item in argv):
        raise VerificationError(
            "invalid_command", "command arguments must be nonempty strings"
        )
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise VerificationError(
            "command_launch_failed", str(exc), locator=f"command.{Path(argv[0]).name}"
        ) from exc
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = threading.Event()

    def drain(name: str, pipe: Any) -> None:
        try:
            while chunk := pipe.read(65536):
                buffer = buffers[name]
                remaining = max_output + 1 - len(buffer)
                if remaining > 0:
                    buffer.extend(chunk[:remaining])
                if len(buffer) > max_output or len(chunk) > remaining:
                    truncated.set()
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except (OSError, ProcessLookupError):
                        pass
        finally:
            pipe.close()

    assert process.stdout is not None and process.stderr is not None
    threads = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    cleanup_failure = _terminate_group(process)
    try:
        process.wait(timeout=PROCESS_CLEANUP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        cleanup_failure = (
            cleanup_failure or "direct child did not exit after group cleanup"
        )
    for thread in threads:
        thread.join(PROCESS_CLEANUP_GRACE_SECONDS)
        if thread.is_alive():
            cleanup_failure = cleanup_failure or "output-drain thread did not terminate"
    stdout = bytes(buffers["stdout"][:max_output])
    stderr = bytes(buffers["stderr"][:max_output])
    return CommandResult(
        argv=tuple(argv),
        returncode=process.poll(),
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        truncated=truncated.is_set(),
        cleanup_failure=cleanup_failure,
        duration_ms=round((time.monotonic() - started) * 1000),
    )


def require_command(result: CommandResult, *, locator: str) -> None:
    if result.cleanup_failure is not None:
        raise VerificationError(
            "process_cleanup_failed", result.cleanup_failure, locator=locator
        )
    if result.timed_out:
        raise VerificationError(
            "command_timed_out", "command exceeded its timeout", locator=locator
        )
    if result.truncated:
        raise VerificationError(
            "command_output_truncated",
            "command output exceeded its bound",
            locator=locator,
        )
    if result.returncode != 0:
        raise VerificationError(
            "command_failed",
            f"command exited with status {result.returncode}",
            locator=locator,
        )


def _is_empty_directory(path: Path) -> bool:
    try:
        return path.is_dir() and next(path.iterdir(), None) is None
    except OSError as exc:
        raise VerificationError(
            "fresh_root_inspection_failed",
            "fresh root cannot be inspected safely",
            locator="fresh_root",
        ) from exc


def prepare_fresh_roots(fresh_root: Path) -> dict[str, Path]:
    if fresh_root.is_symlink():
        raise VerificationError(
            "fresh_root_symlink",
            "fresh root must not be a symlink",
            locator="fresh_root",
        )
    if fresh_root.exists() and not _is_empty_directory(fresh_root):
        raise VerificationError(
            "fresh_root_not_empty",
            "fresh root must not exist or must be empty",
            locator="fresh_root",
        )
    try:
        fresh_root.mkdir(parents=True, exist_ok=True)
        roots = {name: fresh_root / dirname for name, dirname in PLACEHOLDERS.items()}
        for path in roots.values():
            path.mkdir(mode=0o700)
    except OSError as exc:
        raise VerificationError(
            "fresh_root_create_failed",
            "fresh root or one of its isolated children cannot be created",
            locator="fresh_root",
        ) from exc
    resolved = [path.resolve(strict=True) for path in roots.values()]
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                raise VerificationError(
                    "fresh_roots_overlap",
                    "install and cache roots must be mutually non-overlapping",
                    locator="fresh_root",
                )
    return roots


def materialize_build_command(
    configuration: Mapping[str, Any], roots: Mapping[str, Path]
) -> tuple[list[str], list[str]]:
    template = configuration.get("command_template")
    if (
        not isinstance(template, list)
        or not template
        or any(not isinstance(item, str) for item in template)
    ):
        raise VerificationError(
            "invalid_command_template",
            "command_template must be a nonempty string array",
            locator="policy.artifact_build_configuration.command_template",
        )
    counts = collections.Counter(item for item in template if item in PLACEHOLDERS)
    if counts != collections.Counter({placeholder: 1 for placeholder in PLACEHOLDERS}):
        raise VerificationError(
            "invalid_command_placeholders",
            "canonical command must contain each fresh-root placeholder exactly once",
            locator="policy.artifact_build_configuration.command_template",
        )
    if any("<" in item or ">" in item for item in template if item not in PLACEHOLDERS):
        raise VerificationError(
            "unknown_command_placeholder",
            "canonical command contains an unknown placeholder",
            locator="policy.artifact_build_configuration.command_template",
        )
    command = [str(roots[item]) if item in roots else item for item in template]
    return command, list(template)


def validate_control_documents(
    policy: Mapping[str, Any],
    observation: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise VerificationError(
            "policy_schema_mismatch",
            "policy schema_version must be 3",
            locator="policy.schema_version",
        )
    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise VerificationError(
            "observation_schema_mismatch",
            "frozen observation schema is not v3",
            locator="observation.schema",
        )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise VerificationError(
            "receipt_schema_mismatch",
            "unexpected frozen receipt schema",
            locator="receipt.schema",
        )
    configuration = policy.get("artifact_build_configuration")
    if not isinstance(configuration, dict):
        raise VerificationError(
            "configuration_missing",
            "policy configuration is absent",
            locator="policy.artifact_build_configuration",
        )
    if observation.get("artifact_build_configuration") != configuration:
        raise VerificationError(
            "observation_configuration_mismatch",
            "policy and observation configurations differ",
            locator="observation.artifact_build_configuration",
        )
    if receipt.get("configuration") != configuration:
        raise VerificationError(
            "receipt_configuration_mismatch",
            "policy and receipt configurations differ",
            locator="receipt.configuration",
        )
    configuration_id = configuration.get("configuration_id")
    if not isinstance(configuration_id, str) or not configuration_id:
        raise VerificationError(
            "configuration_id_invalid",
            "configuration_id must be a nonempty string",
            locator="policy.artifact_build_configuration.configuration_id",
        )
    return configuration


def validate_frozen_artifact(
    artifact: Any,
    *,
    kind: str,
    observation: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_key = f"{kind}_library"
    observation_key = "dynamic" if kind == "dynamic" else "static"
    receipt_artifact = receipt.get("artifacts", {}).get(receipt_key)
    observed_artifact = observation.get("artifacts", {}).get(observation_key)
    if not isinstance(receipt_artifact, dict) or not isinstance(
        observed_artifact, dict
    ):
        raise VerificationError(
            "artifact_record_missing",
            f"missing {kind} artifact record",
            locator=f"receipt.artifacts.{receipt_key}",
        )
    try:
        requested = os.path.normcase(os.path.abspath(os.fspath(artifact.path)))
        recorded = os.path.normcase(
            os.path.abspath(os.fspath(receipt_artifact["path"]))
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(
            "artifact_path_invalid",
            "receipt artifact path is invalid",
            locator=f"receipt.artifacts.{receipt_key}.path",
        ) from exc
    if requested != recorded:
        raise VerificationError(
            "artifact_path_mismatch",
            "argument does not name the receipt artifact",
            locator=f"receipt.artifacts.{receipt_key}.path",
        )
    metadata = artifact.metadata_record()
    actual_size = artifact.size
    actual_sha = metadata.get("sha256")
    if type(actual_size) is not int or not isinstance(actual_sha, str):
        raise VerificationError(
            "artifact_snapshot_metadata_invalid",
            "captured artifact metadata is incomplete",
            locator=f"receipt.artifacts.{receipt_key}",
        )
    if receipt_artifact.get("size") != actual_size:
        raise VerificationError(
            "artifact_size_mismatch",
            "receipt size does not match artifact",
            locator=f"receipt.artifacts.{receipt_key}.size",
        )
    if receipt_artifact.get("sha256") != actual_sha:
        raise VerificationError(
            "artifact_sha256_mismatch",
            "receipt SHA-256 does not match artifact",
            locator=f"receipt.artifacts.{receipt_key}.sha256",
        )
    if (
        observed_artifact.get("status") != "observed"
        or observed_artifact.get("sha256") != actual_sha
    ):
        raise VerificationError(
            "observation_artifact_mismatch",
            "observation does not bind the frozen artifact",
            locator=f"observation.artifacts.{observation_key}",
        )
    return {"size": actual_size, "sha256": actual_sha}


def _exact_version_line(raw: bytes, *, locator: str) -> str:
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise VerificationError(
            "tool_output_not_utf8", str(exc), locator=locator
        ) from exc
    if not text or "\n" in text:
        raise VerificationError(
            "tool_output_ambiguous",
            "expected exactly one nonempty line",
            locator=locator,
        )
    return text


def verify_toolchain(
    configuration: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    root: Path,
    timeout: float,
    runner: Callable[..., CommandResult] = run_bounded,
) -> dict[str, Any]:
    toolchain = receipt.get("toolchain")
    if not isinstance(toolchain, dict):
        raise VerificationError(
            "toolchain_receipt_missing",
            "receipt.toolchain must be an object",
            locator="receipt.toolchain",
        )
    template = configuration["command_template"]
    zig_name = template[0]
    zig_path_text = shutil.which(zig_name)
    if zig_path_text is None:
        raise VerificationError(
            "zig_not_found", f"cannot resolve {zig_name!r}", locator="toolchain.zig"
        )
    zig_path = Path(zig_path_text).resolve(strict=True)
    zig_version_result = runner((str(zig_path), "version"), cwd=root, timeout=timeout)
    require_command(zig_version_result, locator="toolchain.zig.version")
    zig_version = _exact_version_line(
        zig_version_result.stdout, locator="toolchain.zig.version"
    )
    expected_version = toolchain.get("zig_version")
    if (
        zig_version != expected_version
        or configuration.get("resolved", {}).get("zig_version") != expected_version
    ):
        raise VerificationError(
            "zig_version_mismatch",
            "Zig version differs from configuration or receipt",
            locator="toolchain.zig.version",
        )
    zig_sha = sha256_file(zig_path)
    if zig_sha != toolchain.get("zig_executable_sha256"):
        raise VerificationError(
            "zig_executable_mismatch",
            "Zig executable hash differs from receipt",
            locator="toolchain.zig.sha256",
        )

    commands: dict[str, CommandResult] = {"zig_version": zig_version_result}

    def query(name: str, argv: Sequence[str]) -> str:
        result = runner(argv, cwd=root, timeout=timeout)
        commands[name] = result
        require_command(result, locator=f"toolchain.{name}")
        return (result.stdout + result.stderr).decode("utf-8", "strict").strip()

    sdk_version = query("sdk_version", ("xcrun", "--show-sdk-version"))
    if sdk_version != toolchain.get("host_sdk_version"):
        raise VerificationError(
            "host_sdk_mismatch",
            "active SDK differs from receipt",
            locator="toolchain.sdk_version",
        )
    sdk_path = Path(query("sdk_path", ("xcrun", "--show-sdk-path"))).resolve(
        strict=True
    )
    settings = sdk_path / "SDKSettings.json"
    if sha256_file(settings) != toolchain.get("host_sdk_settings_sha256"):
        raise VerificationError(
            "sdk_settings_mismatch",
            "SDKSettings.json hash differs from receipt",
            locator="toolchain.sdk_settings",
        )
    ld_path = Path(query("ld_path", ("xcrun", "--find", "ld"))).resolve(strict=True)
    ld_version = query("linker_version", (str(ld_path), "-v"))
    if str(toolchain.get("linker_version")) not in ld_version:
        raise VerificationError(
            "linker_version_mismatch",
            "linker version differs from receipt",
            locator="toolchain.linker_version",
        )
    clang_version = query("clang_version", ("xcrun", "clang", "--version"))
    expected_clang = str(toolchain.get("clang_version"))
    if not re.search(rf"\b{re.escape(expected_clang)}\b", clang_version):
        raise VerificationError(
            "clang_version_mismatch",
            "clang version differs from receipt",
            locator="toolchain.clang_version",
        )
    artifact_sdk = configuration.get("resolved", {}).get("artifact_sdk")
    if artifact_sdk != toolchain.get("artifact_sdk_version"):
        raise VerificationError(
            "artifact_sdk_receipt_mismatch",
            "artifact SDK differs between policy and receipt",
            locator="toolchain.artifact_sdk_version",
        )
    return {
        "zig_version": zig_version,
        "zig_executable_sha256": zig_sha,
        "host_sdk_version": sdk_version,
        "host_sdk_settings_sha256": toolchain["host_sdk_settings_sha256"],
        "artifact_sdk_version": artifact_sdk,
        "linker_version": str(toolchain["linker_version"]),
        "clang_version": expected_clang,
        "commands": {
            name: result.sanitized() for name, result in sorted(commands.items())
        },
        "verdict": "pass",
    }


def _contained(path: Path, root: Path, *, locator: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        boundary = root.resolve(strict=True)
    except OSError as exc:
        raise VerificationError(
            "provenance_path_missing",
            "artifact-referenced provenance path cannot be resolved",
            locator=locator,
        ) from exc
    if not resolved.is_relative_to(boundary):
        raise VerificationError(
            "provenance_outside_cache",
            "artifact-referenced path is outside its cache root",
            locator=locator,
        )
    return resolved


def _unique_provenance(artifact: Any, kind: str, *, locator: str) -> Any:
    values = [item for item in artifact.provenance if item.kind == kind]
    if len(values) != 1:
        raise VerificationError(
            "provenance_locator_count",
            f"expected exactly one {kind} locator, found {len(values)}",
            locator=locator,
        )
    return values[0]


def _bind_builtin(
    locator: Any,
    *,
    cache_root: Path,
    expected_sha256: str,
    label: str,
    receipt_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = _contained(Path(locator.path), cache_root, locator=label)
    if path.name != "builtin.zig":
        raise VerificationError(
            "provenance_basename_mismatch",
            "provenance does not name builtin.zig",
            locator=label,
        )
    captured = read_exact_regular_file(path, max_bytes=MAX_HASH_BYTES, locator=label)
    digest = captured.sha256
    if digest != expected_sha256:
        raise VerificationError(
            "builtin_sha256_mismatch",
            "artifact-bound builtin.zig differs from policy",
            locator=label,
        )
    if receipt_record is not None:
        try:
            recorded_path = Path(receipt_record["path"]).resolve(strict=True)
        except (KeyError, OSError, TypeError) as exc:
            raise VerificationError(
                "receipt_builtin_invalid",
                "receipt builtin record cannot be resolved",
                locator=f"{label}.receipt",
            ) from exc
        if recorded_path != path or receipt_record.get("sha256") != digest:
            raise VerificationError(
                "receipt_builtin_mismatch",
                "receipt builtin path or SHA-256 is not artifact-referenced provenance",
                locator=f"{label}.receipt",
            )
    return {
        "kind": locator.kind,
        "source_locator": locator.source_locator,
        "relative_cache_path": path.relative_to(
            cache_root.resolve(strict=True)
        ).as_posix(),
        "sha256": digest,
        "verdict": "pass",
    }


def bind_artifact_provenance(
    parity: Any,
    dynamic: Any,
    static: Any,
    *,
    local_cache: Path,
    global_cache: Path,
    expected_dynamic_sha: str,
    expected_static_sha: str,
    label: str,
    receipt_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    oso = _unique_provenance(
        dynamic, "dynamic_cache_object", locator=f"{label}.dynamic.N_OSO"
    )
    object_path = _contained(
        Path(oso.path), local_cache, locator=f"{label}.dynamic.N_OSO"
    )
    object_file = read_exact_regular_file(
        object_path,
        max_bytes=MAX_HASH_BYTES,
        locator=f"{label}.dynamic.N_OSO",
    )
    mtime = object_file.metadata.st_mtime_ns // 1_000_000_000
    binding = parity.bind_provenance(oso, exists=True, mtime_seconds=mtime)
    if binding["verdict"] != "pass":
        raise VerificationError(
            "n_oso_mtime_mismatch",
            "; ".join(binding["failures"]),
            locator=f"{label}.dynamic.N_OSO",
        )
    try:
        dynamic_object = parity.parse_macho_object(
            object_file.data,
            label=f"{label}-dynamic-object",
            locator_prefix="dynamic_object",
        )
    except (OSError, parity.ParityError) as exc:
        raise VerificationError(
            "dynamic_object_parse_failed", str(exc), locator=f"{label}.dynamic.object"
        ) from exc
    dynamic_builtin = _unique_provenance(
        dynamic_object, "static_builtin", locator=f"{label}.dynamic.object.DWARF"
    )
    static_builtin = _unique_provenance(
        static, "static_builtin", locator=f"{label}.static.DWARF"
    )
    return {
        "dynamic": {
            "object_relative_cache_path": object_path.relative_to(
                local_cache.resolve(strict=True)
            ).as_posix(),
            "object_mtime_seconds": mtime,
            "n_oso_mtime_seconds": oso.expected_mtime_seconds,
            "builtin": _bind_builtin(
                dynamic_builtin,
                cache_root=global_cache,
                expected_sha256=expected_dynamic_sha,
                label=f"{label}.dynamic.builtin",
                receipt_record=(
                    receipt_inputs.get("dynamic_builtin_zig")
                    if receipt_inputs is not None
                    else None
                ),
            ),
            "verdict": "pass",
        },
        "static": {
            "builtin": _bind_builtin(
                static_builtin,
                cache_root=global_cache,
                expected_sha256=expected_static_sha,
                label=f"{label}.static.builtin",
                receipt_record=(
                    receipt_inputs.get("static_builtin_zig")
                    if receipt_inputs is not None
                    else None
                ),
            ),
            "verdict": "pass",
        },
        "cache_presence_sufficient": False,
        "verdict": "pass",
    }


def receipt_cache_roots(
    receipt: Mapping[str, Any], *, redactor: SensitivePathRedactor | None = None
) -> tuple[Path, Path]:
    configuration = receipt["configuration"]
    template = configuration["command_template"]
    argv = receipt.get("execution", {}).get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) != len(template)
        or any(not isinstance(item, str) for item in argv)
    ):
        raise VerificationError(
            "receipt_command_invalid",
            "receipt execution argv does not match canonical command shape",
            locator="receipt.execution.argv",
        )
    substitutions: dict[str, Path] = {}
    for expected, actual in zip(template, argv):
        if expected in PLACEHOLDERS:
            substitutions[expected] = Path(actual)
        elif expected != actual:
            raise VerificationError(
                "receipt_command_mismatch",
                "receipt execution argv differs from canonical command",
                locator="receipt.execution.argv",
            )
    if set(substitutions) != set(PLACEHOLDERS):
        raise VerificationError(
            "receipt_command_invalid",
            "receipt execution argv lacks cache roots",
            locator="receipt.execution.argv",
        )
    if redactor is not None:
        for placeholder, replacement in (
            ("<temporary-install-prefix>", "<frozen-install-root>"),
            ("<isolated-local-cache>", "<frozen-local-cache>"),
            ("<isolated-global-cache>", "<frozen-global-cache>"),
        ):
            redactor.add_root(substitutions[placeholder], replacement)
    try:
        roots = [item.resolve(strict=True) for item in substitutions.values()]
    except OSError as exc:
        raise VerificationError(
            "receipt_root_invalid",
            "receipt install or cache root cannot be resolved",
            locator="receipt.execution.argv",
        ) from exc
    if len(set(roots)) != 3:
        raise VerificationError(
            "receipt_roots_overlap",
            "receipt install/cache roots are not distinct",
            locator="receipt.execution.argv",
        )
    return substitutions["<isolated-local-cache>"], substitutions[
        "<isolated-global-cache>"
    ]


def symbol_axis(records: Iterable[Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        "external_defined": [],
        "external_undefined": [],
        "local_non_stabs": [],
    }
    for record in records:
        if record.stab:
            continue
        value = {
            "name": record.name,
            "base_type": record.base_type,
            "section": record.section,
            "private_external": record.private_external,
            "type_code": record.type_code,
        }
        undefined = record.base_type in {"undefined", "prebound-undefined"}
        if record.external and undefined:
            result["external_undefined"].append(value)
        elif record.external:
            result["external_defined"].append(value)
        else:
            result["local_non_stabs"].append(value)
    return result


def compare_symbol_axes(
    left: Mapping[str, list[dict[str, Any]]], right: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for name in ("external_defined", "external_undefined", "local_non_stabs"):
        left_values = left[name]
        right_values = right[name]
        left_counter = collections.Counter(
            canonical_json_bytes(item) for item in left_values
        )
        right_counter = collections.Counter(
            canonical_json_bytes(item) for item in right_values
        )
        axes[name] = {
            "left_count": len(left_values),
            "right_count": len(right_values),
            "order_equal": left_values == right_values,
            "multiset_equal": left_counter == right_counter,
            "types_equal": collections.Counter(
                (item["base_type"], item["section"], item["type_code"])
                for item in left_values
            )
            == collections.Counter(
                (item["base_type"], item["section"], item["type_code"])
                for item in right_values
            ),
        }
        if not all(
            axes[name][key] for key in ("order_equal", "multiset_equal", "types_equal")
        ):
            raise VerificationError(
                "symbol_axis_mismatch",
                f"{name} order, multiset, or types differ",
                locator=f"symbols.{name}",
            )
    return {"axes": axes, "verdict": "pass"}


def source_names_from_observation(
    observation: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    declarations = observation.get("sources", {}).get("declarations")
    if not isinstance(declarations, list):
        raise VerificationError(
            "source_declarations_missing",
            "frozen observation lacks source declarations",
            locator="observation.sources.declarations",
        )
    public: list[str] = []
    hidden: list[str] = []
    for index, item in enumerate(declarations):
        if not isinstance(item, dict):
            raise VerificationError(
                "source_declaration_invalid",
                "source declaration must be an object",
                locator=f"observation.sources.declarations[{index}]",
            )
        name = item.get("exported_name")
        visibility = item.get("visibility")
        if (
            not isinstance(name, str)
            or not name
            or visibility not in {"default", "hidden"}
        ):
            raise VerificationError(
                "source_declaration_invalid",
                "invalid exported_name or visibility",
                locator=f"observation.sources.declarations[{index}]",
            )
        (hidden if visibility == "hidden" else public).append(name)
    return public, hidden


def _normalize_exact_symbol(name: str, source_names: set[str]) -> str | None:
    if name in source_names:
        return name
    if name.startswith("_") and name[1:] in source_names:
        return name[1:]
    return None


def source_accounting(
    records: Iterable[Any], public: Sequence[str], hidden: Sequence[str]
) -> dict[str, Any]:
    public_set = set(public)
    hidden_set = set(hidden)
    # The observation deliberately preserves declaration sites, so multiple
    # source declarations may export the same ABI name.  Name accounting is a
    # set operation; only contradictory public/hidden classification is invalid.
    if public_set & hidden_set:
        raise VerificationError(
            "source_name_ambiguity",
            "public and hidden source names overlap",
            locator="observation.sources.declarations",
        )
    public_matched: set[str] = set()
    hidden_matched: set[str] = set()
    for record in records:
        if record.stab or record.base_type in {"undefined", "prebound-undefined"}:
            continue
        if record.external and not record.private_external:
            normalized = _normalize_exact_symbol(record.name, public_set)
            if normalized is not None:
                public_matched.add(normalized)
        else:
            normalized = _normalize_exact_symbol(record.name, hidden_set)
            if normalized is not None:
                hidden_matched.add(normalized)
    result = {
        "public": {
            "expected": sorted(public_set),
            "matched": sorted(public_matched),
            "missing": sorted(public_set - public_matched),
        },
        "hidden": {
            "expected": sorted(hidden_set),
            "matched": sorted(hidden_matched),
            "missing": sorted(hidden_set - hidden_matched),
        },
    }
    result["verdict"] = "pass"
    return result


def comparison_summary(comparison: Any) -> dict[str, Any]:
    allowed_bytes = sum(
        item["left"]["size"] + item["right"]["size"]
        for item in comparison.allowed_edit_atoms
    )
    derived_bytes = sum(
        item["left"]["size"] + item["right"]["size"]
        for item in comparison.derived_fields
    )
    return {
        "verdict": comparison.verdict,
        "failures": list(comparison.failures),
        "left_raw_sha256": comparison.left_raw_sha256,
        "right_raw_sha256": comparison.right_raw_sha256,
        "left_size": comparison.left_size,
        "right_size": comparison.right_size,
        "left_canonical_digest": comparison.left_canonical_digest,
        "right_canonical_digest": comparison.right_canonical_digest,
        "structured_axes_equal": bool(comparison.structured_axes.get("equal")),
        "symbol_order_equal": bool(comparison.symbol_axes.get("order_equal")),
        "symbol_multiset_equal": bool(comparison.symbol_axes.get("multiset_equal")),
        "allowed_atom_count": len(comparison.allowed_edit_atoms),
        "allowed_bytes_both_sides": allowed_bytes,
        "derived_atom_count": len(comparison.derived_fields),
        "derived_bytes_both_sides": derived_bytes,
        "uncovered_left_bytes": list(comparison.uncovered_left_bytes),
        "uncovered_right_bytes": list(comparison.uncovered_right_bytes),
        "unpaired_atoms": list(comparison.unpaired_atoms),
        "invalid_derived_fields": list(comparison.invalid_derived_fields),
    }


def require_comparison(comparison: Any, *, locator: str) -> None:
    if comparison.verdict != "pass":
        raise VerificationError(
            "artifact_parity_failed", "; ".join(comparison.failures), locator=locator
        )
    if comparison.left_canonical_digest != comparison.right_canonical_digest:
        raise VerificationError(
            "canonical_digest_mismatch", "canonical digests differ", locator=locator
        )
    if any(
        (
            comparison.uncovered_left_bytes,
            comparison.uncovered_right_bytes,
            comparison.unpaired_atoms,
            comparison.invalid_derived_fields,
            comparison.failures,
        )
    ):
        raise VerificationError(
            "comparison_not_closed",
            "comparison has uncovered, unpaired, invalid, or failed evidence",
            locator=locator,
        )


def _decode_tool(result: CommandResult, locator: str) -> str:
    require_command(result, locator=locator)
    try:
        return (result.stdout + result.stderr).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError(
            "tool_output_not_utf8", str(exc), locator=locator
        ) from exc


def parse_otool(loads: str, libraries: str) -> dict[str, Any]:
    blocks = re.split(r"(?=^Load command \d+\s*$)", loads, flags=re.MULTILINE)
    build = [
        block
        for block in blocks
        if re.search(r"^\s*cmd LC_BUILD_VERSION\s*$", block, re.MULTILINE)
    ]
    if len(build) != 1:
        raise VerificationError(
            "otool_build_version_count",
            f"expected one LC_BUILD_VERSION block, found {len(build)}",
            locator="system.otool",
        )

    def one(pattern: str, text: str, field: str) -> str:
        values = re.findall(pattern, text, re.MULTILINE)
        if len(values) != 1:
            raise VerificationError(
                "otool_field_count",
                f"expected one {field}, found {len(values)}",
                locator=f"system.otool.{field}",
            )
        return values[0]

    platform = one(r"^\s*platform\s+(\S+)", build[0], "platform")
    platform = {"1": "MACOS", "MACOS": "MACOS"}.get(platform, platform)
    minimum = one(r"^\s*minos\s+(\S+)", build[0], "minimum_platform")
    sdk = one(r"^\s*sdk\s+(\S+)", build[0], "sdk")
    id_blocks = [
        block
        for block in blocks
        if re.search(r"^\s*cmd LC_ID_DYLIB\s*$", block, re.MULTILINE)
    ]
    install = (
        one(r"^\s*name\s+(\S+)", id_blocks[0], "install_name")
        if len(id_blocks) == 1
        else None
    )
    if install is None:
        raise VerificationError(
            "otool_install_name_count",
            f"expected one LC_ID_DYLIB block, found {len(id_blocks)}",
            locator="system.otool.install_name",
        )
    dependency_commands = "LC_(?:LOAD|LOAD_WEAK|REEXPORT|LAZY_LOAD|LOAD_UPWARD)_DYLIB"
    dependency_blocks = [
        block
        for block in blocks
        if re.search(rf"^\s*cmd {dependency_commands}\s*$", block, re.MULTILINE)
    ]
    dependencies = [
        one(r"^\s*name\s+(\S+)", block, "dependency") for block in dependency_blocks
    ]
    rpath_blocks = [
        block
        for block in blocks
        if re.search(r"^\s*cmd LC_RPATH\s*$", block, re.MULTILINE)
    ]
    rpaths = [one(r"^\s*path\s+(\S+)", block, "rpath") for block in rpath_blocks]
    library_names = []
    for line in libraries.splitlines()[1:]:
        match = re.match(r"\s*(\S+)\s+\(compatibility version", line)
        if match:
            library_names.append(match.group(1))
    if (
        not library_names
        or library_names[0] != install
        or library_names[1:] != dependencies
    ):
        raise VerificationError(
            "otool_libraries_mismatch",
            "otool -L disagrees with otool -l",
            locator="system.otool.libraries",
        )
    return {
        "platform": platform,
        "minimum_platform": minimum,
        "sdk": sdk,
        "install_name": install,
        "dependencies": dependencies,
        "rpaths": rpaths,
    }


def core_macho_metadata(dynamic: Any) -> dict[str, Any]:
    build_versions = dynamic.structured_axes["build_versions"]
    install_names = dynamic.structured_axes["install_name"]
    if len(build_versions) != 1 or len(install_names) != 1:
        raise VerificationError(
            "core_metadata_count",
            "core parser did not find unique build/install metadata",
            locator="structured.dynamic",
        )
    build = build_versions[0]
    platform = {1: "MACOS"}.get(build["platform"])
    if platform is None:
        raise VerificationError(
            "core_platform_unknown",
            f"unsupported Mach-O platform value {build['platform']!r}",
            locator="structured.dynamic.platform",
        )

    def display_version(value: str) -> str:
        parts = value.split(".")
        while len(parts) > 2 and parts[-1] == "0":
            parts.pop()
        return ".".join(parts)

    return {
        "platform": platform,
        "minimum_platform": display_version(build["minimum_platform"]),
        "sdk": display_version(build["sdk"]),
        "install_name": install_names[0]["name"],
        "dependencies": [
            item["name"] for item in dynamic.structured_axes["dependencies"]
        ],
        "rpaths": list(dynamic.structured_axes["rpaths"]),
    }


def parse_nm_axes(text: str) -> dict[str, collections.Counter[tuple[str, str]]]:
    axes = {
        name: collections.Counter()
        for name in ("external_defined", "external_undefined", "local_non_stabs")
    }
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.endswith(":"):
            continue
        match = re.search(
            r"\(([^)]+)\)\s+"
            r"(private external|external|non-external)\s+"
            r"(.+?)(?:\s+\(from [^)]+\))?$",
            line,
        )
        if match is None:
            continue
        descriptor, visibility, name = match.groups()
        name = name.strip()
        if name.startswith("[cold func] "):
            name = name[len("[cold func] ") :]
        if not name or name.startswith("("):
            continue
        lower = descriptor.lower()
        # ``nm -a`` renders STABS debug records as ``(?)``; the nlist-based
        # axes intentionally exclude all STABS records.
        if lower == "?":
            continue
        symbol_type = (
            "undefined"
            if lower == "undefined"
            else "section:" + descriptor
            if "," in descriptor
            else lower.replace(" ", "_")
        )
        if "stab" in lower:
            continue
        if visibility in {"external", "private external"}:
            axis = (
                "external_undefined"
                if symbol_type == "undefined"
                else "external_defined"
            )
        else:
            axis = "local_non_stabs"
        axes[axis][(name, symbol_type)] += 1
    return axes


def core_nm_axes(artifact: Any) -> dict[str, collections.Counter[tuple[str, str]]]:
    result = {
        name: collections.Counter()
        for name in ("external_defined", "external_undefined", "local_non_stabs")
    }
    parsed_objects = (
        [artifact]
        if hasattr(artifact, "segments")
        else [member.object for member in artifact.members if member.object is not None]
    )
    sections = {
        section.index: f"section:{section.segment},{section.name}"
        for parsed in parsed_objects
        for segment in parsed.segments
        for section in segment.sections
    }
    for axis, values in symbol_axis(artifact.symbols).items():
        result[axis].update(
            (
                item["name"],
                "undefined"
                if item["base_type"] == "undefined"
                else sections[item["section"]]
                if item["base_type"] == "section"
                else item["base_type"],
            )
            for item in values
        )
    return result


def parse_ar_tv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        mode, uid_gid = fields[0], fields[1]
        if "/" not in uid_gid or not re.fullmatch(r"[rwx-]{9}", mode):
            continue
        uid, gid = uid_gid.split("/", 1)
        size_index = next(
            (index for index in range(2, len(fields)) if fields[index].isdigit()), None
        )
        if size_index is None:
            continue
        permission = 0
        for character, bit in zip(
            mode, (0o400, 0o200, 0o100, 0o040, 0o020, 0o010, 0o004, 0o002, 0o001)
        ):
            if character != "-":
                permission |= bit
        rows.append(
            {
                "name": fields[-1],
                "uid": int(uid),
                "gid": int(gid),
                "mode": permission,
                "size": int(fields[size_index]),
                "date_fields": fields[size_index + 1 : -1],
            }
        )
    return rows


def system_cross_checks(
    dynamic_snapshot: Any,
    static_snapshot: Any,
    dynamic: Any,
    static: Any,
    *,
    root: Path,
    timeout: float,
    runner: Callable[..., CommandResult] = run_bounded,
    redactor: SensitivePathRedactor | None = None,
) -> dict[str, Any]:
    commands: dict[str, CommandResult] = {}

    def run(name: str, argv: Sequence[str]) -> str:
        result = runner(argv, cwd=root, timeout=timeout)
        commands[name] = result
        return _decode_tool(result, f"system.{name}")

    def artifact_path(snapshot: Any) -> str:
        # Every system-tool invocation gets a newly reverified private path.
        if redactor is None:
            return snapshot.execution_path
        return redactor.add_snapshot_artifact(snapshot)

    loads = run(
        "otool_load_commands",
        ("otool", "-l", artifact_path(dynamic_snapshot)),
    )
    libraries = run("otool_libraries", ("otool", "-L", artifact_path(dynamic_snapshot)))
    otool_metadata = parse_otool(loads, libraries)
    core_metadata = core_macho_metadata(dynamic)
    if otool_metadata != core_metadata:
        raise VerificationError(
            "otool_core_mismatch",
            "otool metadata differs from core parser",
            locator="system.otool",
        )
    for name, snapshot, artifact in (
        ("dynamic", dynamic_snapshot, dynamic),
        ("static", static_snapshot, static),
    ):
        nm_text = run(f"nm_{name}", ("nm", "-a", "-m", artifact_path(snapshot)))
        if parse_nm_axes(nm_text) != core_nm_axes(artifact):
            raise VerificationError(
                "nm_core_mismatch",
                f"nm axes differ for {name} artifact",
                locator=f"system.nm.{name}",
            )
    ar_names = [
        line.strip()
        for line in run(
            "ar_members", ("ar", "-t", artifact_path(static_snapshot))
        ).splitlines()
        if line.strip()
    ]
    expected_names = [member.name for member in static.members]
    if ar_names != expected_names:
        raise VerificationError(
            "ar_member_order_mismatch",
            "ar -t differs from core member order",
            locator="system.ar.members",
        )
    tv_rows = parse_ar_tv(
        run("ar_metadata", ("ar", "-tv", artifact_path(static_snapshot)))
    )
    if [row["name"] for row in tv_rows] != expected_names:
        raise VerificationError(
            "ar_metadata_order_mismatch",
            "ar -tv differs from core member order",
            locator="system.ar.metadata",
        )
    normalized = static.structured_axes["normalized_metadata"]
    logical_sizes = [
        member.stored_size - (member.data_offset - member.header_offset - 60)
        for member in static.members
    ]
    if any(
        row["uid"] != meta["uid"]
        or row["gid"] != meta["gid"]
        or row["mode"] != (meta["mode"] & 0o777)
        or row["size"] != expected_size
        or meta["date"] != 0
        or not {"Jan", "1", "1970"}.issubset(set(row["date_fields"]))
        for row, meta, expected_size in zip(tv_rows, normalized, logical_sizes)
    ):
        raise VerificationError(
            "ar_metadata_mismatch",
            "ar ownership metadata differs from core parser",
            locator="system.ar.metadata",
        )
    run(
        "codesign_verify",
        ("codesign", "--verify", "--strict", artifact_path(dynamic_snapshot)),
    )
    codesign_details = run(
        "codesign_details",
        ("codesign", "-d", "--verbose=4", artifact_path(dynamic_snapshot)),
    )
    flags = re.search(r"flags=0x([0-9a-fA-F]+)", codesign_details)
    if (
        flags is None
        or int(flags.group(1), 16) != 0x20002
        or "adhoc" not in codesign_details.lower()
        or "linker-signed" not in codesign_details.lower()
    ):
        raise VerificationError(
            "codesign_flags_mismatch",
            "codesign did not report ad-hoc linker-signed identity",
            locator="system.codesign",
        )
    return {
        "otool": {"metadata": otool_metadata, "matches_core": True},
        "nm": {
            "dynamic_counts": {
                key: sum(value.values()) for key, value in core_nm_axes(dynamic).items()
            },
            "static_counts": {
                key: sum(value.values()) for key, value in core_nm_axes(static).items()
            },
            "matches_core": True,
        },
        "ar": {"member_order": expected_names, "metadata_matches_core": True},
        "codesign": {"flags": "0x20002", "adhoc": True, "linker_signed": True},
        "commands": {
            name: result.sanitized() for name, result in sorted(commands.items())
        },
        "verdict": "pass",
    }


def _report_identity(path: Path, root: Path) -> dict[str, str]:
    return {"identity": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def verify(
    args: argparse.Namespace, *, redactor: SensitivePathRedactor | None = None
) -> dict[str, Any]:
    active_redactor = redactor or SensitivePathRedactor()
    active_redactor.add_root(args.fresh_root, "<fresh-root>")
    root = args.root.resolve(strict=True)
    policy, policy_raw = load_json(args.policy, locator="policy")
    observation, observation_raw = load_json(
        args.frozen_observation, locator="observation"
    )
    receipt, receipt_raw = load_json(args.frozen_receipt, locator="receipt")
    configuration = validate_control_documents(policy, observation, receipt)
    parity = load_parity_module(root)
    snapshots = load_benchmark_artifacts_module(root)
    frozen_requests = [
        snapshots.ArtifactRequest.library("frozen-dynamic", args.frozen_dynamic),
        snapshots.ArtifactRequest.library("frozen-static", args.frozen_static),
    ]
    report: dict[str, Any]
    try:
        with snapshots.ArtifactSnapshotSet.capture(
            frozen_requests,
            max_artifact_bytes=MAX_HASH_BYTES,
            max_artifacts=2,
            max_total_bytes=2 * MAX_HASH_BYTES,
        ) as frozen_set:
            frozen_dynamic_snapshot, frozen_static_snapshot = frozen_set.artifacts
            active_redactor.add_snapshot_artifact(frozen_dynamic_snapshot)
            active_redactor.add_snapshot_artifact(frozen_static_snapshot)
            frozen_raw = {
                "dynamic": validate_frozen_artifact(
                    frozen_dynamic_snapshot,
                    kind="dynamic",
                    observation=observation,
                    receipt=receipt,
                ),
                "static": validate_frozen_artifact(
                    frozen_static_snapshot,
                    kind="static",
                    observation=observation,
                    receipt=receipt,
                ),
            }
            fresh_roots = prepare_fresh_roots(args.fresh_root)
            command, sanitized_command = materialize_build_command(
                configuration, fresh_roots
            )
            toolchain = verify_toolchain(
                configuration, receipt, root=root, timeout=args.timeout
            )
            build_result = run_bounded(command, cwd=root, timeout=args.timeout)
            require_command(build_result, locator="fresh_build")
            fresh_dynamic_path = (
                fresh_roots["<temporary-install-prefix>"]
                / "lib"
                / args.frozen_dynamic.name
            )
            fresh_static_path = (
                fresh_roots["<temporary-install-prefix>"]
                / "lib"
                / args.frozen_static.name
            )
            fresh_requests = [
                snapshots.ArtifactRequest.library("fresh-dynamic", fresh_dynamic_path),
                snapshots.ArtifactRequest.library("fresh-static", fresh_static_path),
            ]
            with snapshots.ArtifactSnapshotSet.capture(
                fresh_requests,
                max_artifact_bytes=MAX_HASH_BYTES,
                max_artifacts=2,
                max_total_bytes=2 * MAX_HASH_BYTES,
            ) as fresh_set:
                fresh_dynamic_snapshot, fresh_static_snapshot = fresh_set.artifacts
                active_redactor.add_snapshot_artifact(fresh_dynamic_snapshot)
                active_redactor.add_snapshot_artifact(fresh_static_snapshot)
                frozen_dynamic_file = snapshot_artifact_bytes(
                    frozen_dynamic_snapshot,
                    locator="artifact.frozen.dynamic",
                    redactor=active_redactor,
                )
                frozen_static_file = snapshot_artifact_bytes(
                    frozen_static_snapshot,
                    locator="artifact.frozen.static",
                    redactor=active_redactor,
                )
                fresh_dynamic_file = snapshot_artifact_bytes(
                    fresh_dynamic_snapshot,
                    locator="artifact.fresh.dynamic",
                    redactor=active_redactor,
                )
                fresh_static_file = snapshot_artifact_bytes(
                    fresh_static_snapshot,
                    locator="artifact.fresh.static",
                    redactor=active_redactor,
                )
                frozen_dynamic = parity.parse_dynamic_artifact(
                    frozen_dynamic_file.data, label="frozen-dynamic"
                )
                frozen_static = parity.parse_static_archive(
                    frozen_static_file.data, label="frozen-static"
                )
                fresh_dynamic = parity.parse_dynamic_artifact(
                    fresh_dynamic_file.data, label="fresh-dynamic"
                )
                fresh_static = parity.parse_static_archive(
                    fresh_static_file.data, label="fresh-static"
                )
                dynamic_comparison = parity.compare_dynamic_artifact_pair(
                    frozen_dynamic, fresh_dynamic
                )
                static_comparison = parity.compare_static_archive_pair(
                    frozen_static, fresh_static
                )
                require_comparison(dynamic_comparison, locator="comparison.dynamic")
                require_comparison(static_comparison, locator="comparison.static")
                expected_hashes = configuration["resolved_build_inputs"][
                    "generated_builtin_zig_sha256"
                ]
                frozen_local, frozen_global = receipt_cache_roots(
                    receipt, redactor=active_redactor
                )
                receipt_inputs = receipt.get("resolved_build_inputs")
                if not isinstance(receipt_inputs, dict):
                    raise VerificationError(
                        "receipt_build_inputs_missing",
                        "receipt.resolved_build_inputs must be an object",
                        locator="receipt.resolved_build_inputs",
                    )
                frozen_provenance = bind_artifact_provenance(
                    parity,
                    frozen_dynamic,
                    frozen_static,
                    local_cache=frozen_local,
                    global_cache=frozen_global,
                    expected_dynamic_sha=expected_hashes["dynamic_library"],
                    expected_static_sha=expected_hashes["static_library"],
                    label="frozen",
                    receipt_inputs=receipt_inputs,
                )
                fresh_provenance = bind_artifact_provenance(
                    parity,
                    fresh_dynamic,
                    fresh_static,
                    local_cache=fresh_roots["<isolated-local-cache>"],
                    global_cache=fresh_roots["<isolated-global-cache>"],
                    expected_dynamic_sha=expected_hashes["dynamic_library"],
                    expected_static_sha=expected_hashes["static_library"],
                    label="fresh",
                )
                public_names, hidden_names = source_names_from_observation(observation)
                frozen_symbol_axes = {
                    "dynamic": symbol_axis(frozen_dynamic.symbols),
                    "static": symbol_axis(frozen_static.symbols),
                }
                fresh_symbol_axes = {
                    "dynamic": symbol_axis(fresh_dynamic.symbols),
                    "static": symbol_axis(fresh_static.symbols),
                }
                symbols = {
                    kind: compare_symbol_axes(
                        frozen_symbol_axes[kind], fresh_symbol_axes[kind]
                    )
                    for kind in ("dynamic", "static")
                }
                source_checks: dict[str, Any] = {}
                for label, artifact in (
                    ("frozen_dynamic", frozen_dynamic),
                    ("frozen_static", frozen_static),
                    ("fresh_dynamic", fresh_dynamic),
                    ("fresh_static", fresh_static),
                ):
                    source_checks[label] = source_accounting(
                        artifact.symbols, public_names, hidden_names
                    )
                    if source_checks[label]["public"]["missing"]:
                        raise VerificationError(
                            "source_symbol_accounting_failed",
                            "a public source declaration is missing",
                            locator=f"source_accounting.{label}",
                        )
                for kind in ("dynamic", "static"):
                    frozen_accounting = source_checks[f"frozen_{kind}"]
                    fresh_accounting = source_checks[f"fresh_{kind}"]
                    if frozen_accounting != fresh_accounting:
                        raise VerificationError(
                            "source_symbol_accounting_mismatch",
                            f"fresh {kind} public/hidden accounting differs from frozen",
                            locator=f"source_accounting.{kind}",
                        )
                source_checks["parity"] = {
                    "dynamic": True,
                    "static": True,
                    "verdict": "pass",
                }
                system = {
                    "frozen": system_cross_checks(
                        frozen_dynamic_snapshot,
                        frozen_static_snapshot,
                        frozen_dynamic,
                        frozen_static,
                        root=root,
                        timeout=args.timeout,
                        redactor=active_redactor,
                    ),
                    "fresh": system_cross_checks(
                        fresh_dynamic_snapshot,
                        fresh_static_snapshot,
                        fresh_dynamic,
                        fresh_static,
                        root=root,
                        timeout=args.timeout,
                        redactor=active_redactor,
                    ),
                }
                fresh_raw = {
                    "dynamic": {
                        "size": fresh_dynamic_file.size,
                        "sha256": fresh_dynamic_file.sha256,
                    },
                    "static": {
                        "size": fresh_static_file.size,
                        "sha256": fresh_static_file.sha256,
                    },
                }
                report = {
                    "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
                    "verifier": _report_identity(root / VERIFIER_ID, root),
                    "parser": _report_identity(root / PARSER_ID, root),
                    "inputs": {
                        "policy_sha256": sha256_bytes(policy_raw),
                        "observation_sha256": sha256_bytes(observation_raw),
                        "receipt_sha256": sha256_bytes(receipt_raw),
                        "configuration_id": configuration["configuration_id"],
                        "configuration_digest": sha256_bytes(
                            canonical_json_bytes(configuration)
                        ),
                    },
                    "artifacts": {"frozen": frozen_raw, "fresh": fresh_raw},
                    "fresh_build": {
                        "canonical_command": sanitized_command,
                        "cwd": "<repository-root>",
                        "fresh_root_was_empty": True,
                        "roots_mutually_non_overlapping": True,
                        "artifacts_preserved": True,
                        "result": build_result.sanitized(),
                    },
                    "toolchain": toolchain,
                    "comparisons": {
                        "dynamic": comparison_summary(dynamic_comparison),
                        "static": comparison_summary(static_comparison),
                    },
                    "structured_checks": {
                        "dynamic": dynamic_comparison.structured_axes,
                        "static": static_comparison.structured_axes,
                        "verdict": "pass",
                    },
                    "symbol_checks": symbols,
                    "source_checks": source_checks,
                    "provenance_checks": {
                        "frozen": frozen_provenance,
                        "fresh": fresh_provenance,
                        "verdict": "pass",
                    },
                    "system_checks": system,
                    "uncovered_left_bytes": [],
                    "uncovered_right_bytes": [],
                    "unpaired_atoms": [],
                    "invalid_derived_fields": [],
                    "failures": [],
                    "verdict": "pass",
                }
                # The report is complete in memory before either subject is
                # finalized.  Context exit must then prove private cleanup.
                frozen_set.finalize()
                fresh_set.finalize()
    except snapshots.ArtifactSnapshotError as exc:
        raise _snapshot_failure(exc) from None
    return report


def failure_report(
    exc: BaseException, *, redactor: SensitivePathRedactor | None = None
) -> dict[str, Any]:
    active_redactor = redactor or SensitivePathRedactor()
    if isinstance(exc, VerificationError):
        failure = {
            "code": exc.code,
            "locator": active_redactor.sanitize_text(exc.locator),
            "message": active_redactor.sanitize_text(exc.message),
        }
    else:
        message = active_redactor.sanitize_text(str(exc))
        failure = {
            "code": "verification_exception",
            "locator": "verifier",
            "message": message or "unexpected verification failure",
        }
    return active_redactor.sanitize_value(
        {
            "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "uncovered_left_bytes": [],
            "uncovered_right_bytes": [],
            "unpaired_atoms": [],
            "invalid_derived_fields": [],
            "failures": [failure],
            "verdict": "fail",
        }
    )


def write_report(
    path: Path,
    report: Mapping[str, Any],
    *,
    redactor: SensitivePathRedactor | None = None,
) -> None:
    active_redactor = redactor or SensitivePathRedactor()
    try:
        raw = canonical_json_bytes(active_redactor.sanitize_value(report)) + b"\n"
    except (TypeError, ValueError) as exc:
        raise VerificationError(
            "report_serialization_failed",
            "verification report could not be serialized",
            locator="output",
        ) from exc
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    except OSError as exc:
        raise VerificationError(
            "report_write_failed",
            "verification report could not be published",
            locator="output",
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--frozen-observation", type=Path, required=True)
    parser.add_argument("--frozen-receipt", type=Path, required=True)
    parser.add_argument("--frozen-dynamic", type=Path, required=True)
    parser.add_argument("--frozen-static", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.timeout > 3600:
        parser.error("--timeout must be greater than zero and at most 3600 seconds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    redactor = SensitivePathRedactor()
    redactor.add_root(args.fresh_root, "<fresh-root>")
    try:
        report = verify(args, redactor=redactor)
    except Exception as exc:
        report = failure_report(exc, redactor=redactor)
        try:
            write_report(args.output, report, redactor=redactor)
        except VerificationError as write_exc:
            print(
                canonical_json_bytes(
                    failure_report(write_exc, redactor=redactor)
                ).decode("ascii"),
                file=sys.stderr,
            )
        return 1
    try:
        write_report(args.output, report, redactor=redactor)
    except VerificationError as exc:
        print(
            canonical_json_bytes(failure_report(exc, redactor=redactor)).decode(
                "ascii"
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
