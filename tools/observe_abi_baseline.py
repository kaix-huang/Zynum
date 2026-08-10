#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Deterministically observe source, projection, artifact, and consumer ABI facts.

This program is deliberately read-only except for the path passed to ``--output``.
It describes evidence; it does not declare or approve a public ABI.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


_REPOSITORY_GIT_PATH = Path(__file__).with_name("repository_git.py")
_REPOSITORY_GIT_SPEC = importlib.util.spec_from_file_location(
    "_zynum_abi_repository_git", _REPOSITORY_GIT_PATH
)
if _REPOSITORY_GIT_SPEC is None or _REPOSITORY_GIT_SPEC.loader is None:
    raise RuntimeError("unable to load the repository Git policy")
repository_git = importlib.util.module_from_spec(_REPOSITORY_GIT_SPEC)
sys.modules[_REPOSITORY_GIT_SPEC.name] = repository_git
_REPOSITORY_GIT_SPEC.loader.exec_module(repository_git)


SCHEMA_NAME = "zynum.abi-baseline-observation"
SCHEMA_VERSION = 3
OBSERVER_ID = "tools/observe_abi_baseline.py"
MAX_CAPTURE_BYTES = 65536
MAX_ARTIFACT_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_ARCHIVE_MEMBERS = 4096
DEFAULT_TIMEOUT_SECONDS = 20.0
PROCESS_CLEANUP_GRACE_SECONDS = 1.0
MAX_ARCHIVE_MEMBERS = 20000
MAX_ARCHIVE_FILE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 200
WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
MACHO_FAT_MAGICS = frozenset(
    (
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
        b"\xca\xfe\xba\xbf",
        b"\xbf\xba\xfe\xca",
    )
)
DARWIN_RUNTIME_SANDBOX = "sandbox-exec"
ARTIFACT_BUILD_CONFIGURATION: dict[str, Any] = {
    "configuration_id": "aarch64-macos-releasefast-v1",
    "command_template": [
        "zig",
        "build",
        "install",
        "-j1",
        "-Dtarget=aarch64-macos",
        "--release=fast",
        "--prefix",
        "<temporary-install-prefix>",
        "--cache-dir",
        "<isolated-local-cache>",
        "--global-cache-dir",
        "<isolated-global-cache>",
        "--summary",
        "failures",
    ],
    "declared_target": "aarch64-macos",
    "resolved": {
        "optimize": "ReleaseFast",
        "cpu": "apple_m1",
        "cpu_resolution": "Zig 0.16.0 explicit-target default",
        "strip_debug_info": False,
        "minimum_platform": "13.0",
        "artifact_sdk": "26.4",
        "zig_version": "0.16.0",
    },
    "resolved_build_inputs": {
        "generated_builtin_zig_sha256": {
            "dynamic_library": "87a6d1418ffc4acad526edf9f21836a937a0456a868ed521cee9c264d5f30c78",
            "static_library": "9c61cd4da525518fd5660ac7b84c3750781ad2ef5ad2e93d364b04421bcf9498",
        },
        "matched_across_two_independent_cache_roots": True,
        "installed_artifact_provenance_binding": {
            "cache_presence_sufficient": False,
            "dynamic_library": {
                "binding_chain": [
                    {
                        "from": "installed_dylib",
                        "via": "unique_N_OSO",
                        "to": "cache_object",
                    },
                    {
                        "from": "cache_object",
                        "via": "DWARF_v4_line_directory",
                        "to": "generated_builtin_zig",
                    },
                ],
                "expected_sha256_field": "generated_builtin_zig_sha256.dynamic_library",
                "match_required": True,
            },
            "static_library": {
                "binding_chain": [
                    {
                        "from": "installed_archive",
                        "via": "unique_ZCU_member",
                        "to": "ZCU_object",
                    },
                    {
                        "from": "ZCU_object",
                        "via": "DWARF_v4_line_directory",
                        "to": "generated_builtin_zig",
                    },
                ],
                "expected_sha256_field": "generated_builtin_zig_sha256.static_library",
                "match_required": True,
            },
        },
        "verification_boundary": {
            "independent_parser": "tools/abi_artifact_parity.py",
            "build_and_verify_cli": "tools/verify_abi_artifact_parity.py",
            "local_receipt_cross_check": True,
        },
    },
    "raw_artifacts": {
        "sha256_retained": True,
        "cross_cache_raw_byte_equality_claimed": False,
        "allowed_volatility_classes": [
            "Mach-O LC_UUID",
            "N_OSO cache path/object mtime",
            "derived adhoc signature",
            "static-object DWARF global-cache path",
        ],
    },
    "fresh_rebuild_parity": {
        "required_parity_axes": [
            "artifact_metadata",
            "archive_structure",
            "symbol_sets",
            "source_symbol_accounting",
            "generated_build_input_provenance",
        ],
        "required_artifact_fields": [
            "format",
            "architecture",
            "platform",
            "minimum_platform",
            "sdk",
            "install_name",
            "dependencies",
            "rpaths",
        ],
        "required_archive_fields": [
            "members",
            "index",
            "normalized_metadata",
        ],
        "required_symbol_sets": [
            {
                "scope": "external",
                "definition": "defined",
                "fields": ["name", "type"],
            },
            {
                "scope": "external",
                "definition": "undefined",
                "fields": ["name", "type"],
            },
            {
                "scope": "local",
                "exclude": "STABS",
                "fields": ["name", "type"],
            },
        ],
        "required_source_symbol_accounting": ["public", "hidden"],
    },
}

LEVEL1 = frozenset(
    "swap copy axpy axpby dot dotu dotc dotu_sub dotc_sub sdsdot dsdot nrm2 asum "
    "amax rotg rot rotm rotmg scal iamax iamax_sub".split()
)
LEVEL2 = frozenset(
    "gemv gbmv hemv hbmv hpmv symv sbmv spmv trmv tbmv tpmv trsv tbsv tpsv "
    "ger geru gerc her hpr her2 hpr2 syr spr syr2 spr2".split()
)
LEVEL3 = frozenset("gemm symm hemm syrk herk syr2k her2k trmm trsm".split())
RETAINED = frozenset("axpby iamax_sub".split())
UTILITY = frozenset(("lsame", "xerbla", "xerbla_array", "scabs1", "dcabs1"))
RUNTIME = frozenset(("zynum_blas_shutdown",))
PLATFORMS = (
    "aarch64-macos",
    "x86_64-linux-gnu",
    "aarch64-linux-gnu",
    "x86_64-windows-gnu",
)
TARGET_BINDINGS = {
    "aarch64-macos": {
        "format": "Mach-O",
        "architectures": {"arm64"},
        "platform": "macOS",
    },
    "x86_64-linux-gnu": {"format": "ELF", "architectures": {"x86_64"}},
    "aarch64-linux-gnu": {"format": "ELF", "architectures": {"arm64"}},
    "x86_64-windows-gnu": {"format": "PE/COFF", "architectures": {"x86_64"}},
}
CONSUMER_SOURCE_PATHS = (
    "examples/cblas/dgemm.c",
    "examples/fortran/dgemm.f90",
    "test/abi/baseline/consumers/cpp_main.cpp",
)
PACKAGE_REPRODUCTION_PATHS = (
    "build.zig",
    "build.zig.zon",
    "tools/check_package_paths.py",
)


class ObservationError(RuntimeError):
    """Raised when the observer schema or an invariant is invalid."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def relative_display(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return "<external>/" + path.name


def normalized_arg(arg: str, root: Path, transient_roots: Sequence[Path] = ()) -> str:
    normalized = arg
    replacements = [(str(root), "<root>"), (str(root.resolve()), "<root>")]
    replacements.extend(
        replacement
        for item in transient_roots
        for replacement in (
            (str(item), "<temporary>"),
            (str(item.resolve()), "<temporary>"),
        )
    )
    for original, replacement in sorted(
        set(replacements), key=lambda item: len(item[0]), reverse=True
    ):
        normalized = normalized.replace(original, replacement)
    if normalized != arg:
        return normalized.replace(os.sep, "/")
    candidate = Path(arg)
    if not candidate.is_absolute():
        return arg.replace(os.sep, "/")
    resolved = candidate.resolve()
    for item in transient_roots:
        try:
            suffix = resolved.relative_to(item.resolve()).as_posix()
            return "<temporary>" + ("/" + suffix if suffix != "." else "")
        except ValueError:
            pass
    return relative_display(resolved, root)


def not_observed(reason: str) -> dict[str, str]:
    if not reason:
        raise ObservationError("not_observed requires a reason")
    return {"status": "not_observed", "reason": reason}


@dataclasses.dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    status: str
    exit_code: Optional[int]
    signal_number: Optional[int]
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    reason: Optional[str] = None

    def observation(
        self, root: Path, transient_roots: Sequence[Path] = ()
    ) -> dict[str, Any]:
        def output(data: bytes, truncated: bool) -> dict[str, Any]:
            excerpt = data.decode("utf-8", "replace")
            replacements = [(str(root), "<root>"), (str(root.resolve()), "<root>")]
            replacements.extend(
                replacement
                for item in transient_roots
                for replacement in (
                    (str(item), "<temporary>"),
                    (str(item.resolve()), "<temporary>"),
                )
            )
            for arg in self.argv:
                candidate = Path(arg)
                if candidate.is_absolute():
                    replacements.append(
                        (str(candidate), normalized_arg(arg, root, transient_roots))
                    )
            for original, replacement in sorted(
                set(replacements), key=lambda item: len(item[0]), reverse=True
            ):
                excerpt = excerpt.replace(original, replacement)
            excerpt = re.sub(r"\b0x[0-9A-Fa-f]+\b", "<address>", excerpt)
            excerpt = re.sub(r"\b(dyld|sandbox-exec)\[\d+\]", r"\1[<pid>]", excerpt)
            return {
                "sha256": sha256_bytes(excerpt.encode("utf-8")),
                "excerpt": excerpt,
                "truncated": truncated,
            }

        value: dict[str, Any] = {
            "argv": [normalized_arg(arg, root, transient_roots) for arg in self.argv],
            "exit_code": self.exit_code,
            "signal": self.signal_number,
            "status": self.status,
            "stdout": output(self.stdout, self.stdout_truncated),
            "stderr": output(self.stderr, self.stderr_truncated),
        }
        if self.reason is not None:
            value["reason"] = self.reason
        return value


def command_succeeded(result: CommandResult) -> bool:
    return result.status == "exited" and result.exit_code == 0


def command_observed(result: CommandResult) -> bool:
    return (
        command_succeeded(result)
        and not result.stdout_truncated
        and not result.stderr_truncated
    )


def command_failed(result: CommandResult) -> bool:
    return not command_observed(result) or b"error:" in result.stderr.lower()


def _windows_supervisor_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Wrap an untrusted Windows command in a trusted start-gated supervisor."""
    return (
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-windows-supervisor",
        *argv,
    )


class _WindowsJob:
    """Windows Job Object that kills the complete assigned process tree."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        import ctypes
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = (
                ("per_process_user_time_limit", ctypes.c_longlong),
                ("per_job_user_time_limit", ctypes.c_longlong),
                ("limit_flags", wintypes.DWORD),
                ("minimum_working_set_size", ctypes.c_size_t),
                ("maximum_working_set_size", ctypes.c_size_t),
                ("active_process_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority_class", wintypes.DWORD),
                ("scheduling_class", wintypes.DWORD),
            )

        class IoCounters(ctypes.Structure):
            _fields_ = tuple(
                (name, ctypes.c_ulonglong)
                for name in (
                    "read_operation_count",
                    "write_operation_count",
                    "other_operation_count",
                    "read_transfer_count",
                    "write_transfer_count",
                    "other_transfer_count",
                )
            )

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = (
                ("basic_limit_information", BasicLimitInformation),
                ("io_info", IoCounters),
                ("process_memory_limit", ctypes.c_size_t),
                ("job_memory_limit", ctypes.c_size_t),
                ("peak_process_memory_used", ctypes.c_size_t),
                ("peak_job_memory_used", ctypes.c_size_t),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        information = ExtendedLimitInformation()
        information.basic_limit_information.limit_flags = (
            WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not kernel32.SetInformationJobObject(
            handle,
            9,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise OSError(error, "SetInformationJobObject failed")
        if not kernel32.AssignProcessToJobObject(
            handle, wintypes.HANDLE(int(process._handle))
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise OSError(error, "AssignProcessToJobObject failed")
        self._kernel32 = kernel32
        self._handle = handle

    def terminate(self) -> bool:
        return bool(self._kernel32.TerminateJobObject(self._handle, 1))

    def close(self) -> bool:
        if self._handle is None:
            return True
        result = bool(self._kernel32.CloseHandle(self._handle))
        self._handle = None
        return result


def _bind_and_release_windows_supervisor(
    process: subprocess.Popen[bytes],
) -> _WindowsJob:
    """Bind the trusted supervisor before allowing it to launch the target."""
    job = _WindowsJob(process)
    assert process.stdin is not None
    try:
        process.stdin.write(b"1")
        process.stdin.flush()
    except OSError:
        job.terminate()
        job.close()
        raise
    finally:
        process.stdin.close()
    return job


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output: int = MAX_CAPTURE_BYTES,
    env: Optional[Mapping[str, str]] = None,
    inherit_env: bool = True,
) -> CommandResult:
    """Run a trusted tool with bounded time/output and same-session cleanup."""
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        raise ObservationError("command argv must contain non-empty strings")
    if max_output <= 0:
        raise ObservationError("max_output must be greater than zero")
    executable = argv[0]
    if os.sep not in executable and shutil.which(executable) is None:
        return CommandResult(
            tuple(argv), "tool_missing", None, None, b"", b"", False, False, executable
        )
    started = time.monotonic()
    try:
        process_env = dict(os.environ) if inherit_env else {}
        if env is not None:
            process_env.update(env)
        process_env.update({"LANG": "C", "LC_ALL": "C", "TZ": "UTC"})
        launch_argv = _windows_supervisor_argv(argv) if os.name == "nt" else tuple(argv)
        process = subprocess.Popen(
            launch_argv,
            cwd=str(cwd),
            env=process_env,
            stdin=subprocess.PIPE if os.name == "nt" else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=os.name != "nt",
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
    except FileNotFoundError:
        return CommandResult(
            tuple(argv), "tool_missing", None, None, b"", b"", False, False, executable
        )
    except OSError as exc:
        return CommandResult(
            tuple(argv),
            "launch_error",
            None,
            None,
            b"",
            b"",
            False,
            False,
            exc.__class__.__name__,
        )
    windows_job: Optional[_WindowsJob] = None
    if os.name == "nt":
        try:
            windows_job = _bind_and_release_windows_supervisor(process)
        except OSError as exc:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=PROCESS_CLEANUP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                return CommandResult(
                    tuple(argv),
                    "cleanup_error",
                    None,
                    None,
                    b"",
                    b"",
                    False,
                    False,
                    "trusted Windows supervisor did not terminate after launch failure",
                )
            return CommandResult(
                tuple(argv),
                "launch_error",
                None,
                None,
                b"",
                b"",
                False,
                False,
                exc.__class__.__name__,
            )
    assert process.stdout is not None
    assert process.stderr is not None

    captures: dict[str, tuple[bytes, bool]] = {}
    reader_errors: list[str] = []

    def drain(name: str, stream: Any) -> None:
        captured = bytearray()
        truncated = False
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                remaining = max_output - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated = True
        except OSError as exc:
            reader_errors.append(exc.__class__.__name__)
        finally:
            stream.close()
            captures[name] = (bytes(captured), truncated)

    readers = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    cleanup_ok = True
    try:
        remaining = max(0.0, started + timeout - time.monotonic())
        code = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        timed_out = True
        code = None
    try:
        if os.name == "nt":
            assert windows_job is not None
            cleanup_ok = windows_job.terminate() and cleanup_ok
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        cleanup_ok = False
    finally:
        if windows_job is not None:
            cleanup_ok = windows_job.close() and cleanup_ok
    if process.poll() is None:
        try:
            code = process.wait(timeout=PROCESS_CLEANUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            cleanup_ok = False
            try:
                code = process.wait(timeout=PROCESS_CLEANUP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                code = process.returncode
    elif code is None:
        code = process.returncode
    cleanup_deadline = time.monotonic() + PROCESS_CLEANUP_GRACE_SECONDS
    for reader in readers:
        reader.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
    if any(reader.is_alive() for reader in readers):
        cleanup_ok = False
        process.stdout.close()
        process.stderr.close()
    stdout, stdout_truncated = captures.get("stdout", (b"", False))
    stderr, stderr_truncated = captures.get("stderr", (b"", False))
    if not cleanup_ok or reader_errors or code is None:
        return CommandResult(
            tuple(argv),
            "cleanup_error",
            code if code is not None and code >= 0 else None,
            -code if code is not None and code < 0 else None,
            stdout,
            stderr,
            stdout_truncated,
            stderr_truncated,
            "process-tree cleanup did not complete",
        )
    if timed_out:
        return CommandResult(
            tuple(argv),
            "timeout",
            None,
            -code if code < 0 else None,
            stdout,
            stderr,
            stdout_truncated,
            stderr_truncated,
            f"timeout after {timeout:g} seconds",
        )
    return CommandResult(
        tuple(argv),
        "exited" if code >= 0 else "signaled",
        code if code >= 0 else None,
        -code if code < 0 else None,
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
    )


def _sandbox_path(path: Path, *, resolve: bool = True) -> str:
    value = path.resolve() if resolve else path
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _darwin_runtime_sandbox_profile(
    cwd: Path,
    argv: Sequence[str],
    read_roots: Sequence[Path],
    *,
    observation_root: Optional[Path] = None,
    observation_transient_roots: Sequence[Path] = (),
) -> str:
    allowed_subpaths = {
        "/System",
        "/usr/lib",
        "/usr/share",
        "/Library/Apple",
        "/private/preboot/Cryptexes",
        "/private/var/db/dyld",
        "/private/var/run/dyld",
        _sandbox_path(cwd),
        _sandbox_path(cwd, resolve=False),
        *(_sandbox_path(path) for path in read_roots),
        *(_sandbox_path(path, resolve=False) for path in read_roots),
    }
    allowed_literals = {"/", "/dev/null", "/dev/urandom"}
    executable = Path(argv[0])
    if executable.is_absolute():
        allowed_literals.update(
            (
                _sandbox_path(executable),
                _sandbox_path(executable, resolve=False),
            )
        )
        allowed_subpaths.update(
            (
                _sandbox_path(executable.parent),
                _sandbox_path(executable.parent, resolve=False),
            )
        )

    def evidence_sort_key(clause: str) -> tuple[str, str]:
        evidence = (
            normalized_arg(clause, observation_root, observation_transient_roots)
            if observation_root is not None
            else clause
        )
        return evidence, clause

    subpath_filters = {f'(subpath "{path}")' for path in allowed_subpaths}
    literal_filters = {f'(literal "{path}")' for path in allowed_literals}
    read_filters = "".join(sorted(subpath_filters, key=evidence_sort_key)) + "".join(
        sorted(literal_filters, key=evidence_sort_key)
    )
    return (
        "(version 1)"
        '(import "dyld-support.sb")'
        "(allow default)"
        "(deny process-fork)"
        "(deny network*)"
        "(deny file-write*)"
        f"(deny file-read-data (require-not (require-any {read_filters})))"
    )


def run_candidate_executable(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    max_output: int = MAX_CAPTURE_BYTES,
    read_roots: Sequence[Path] = (),
    observation_root: Optional[Path] = None,
    observation_transient_roots: Sequence[Path] = (),
) -> CommandResult:
    """Execute candidate code only under a strict platform runtime boundary."""
    if os.name == "nt":
        return CommandResult(
            tuple(argv),
            "containment_unavailable",
            None,
            None,
            b"",
            b"",
            False,
            False,
            "strict candidate runtime containment is unavailable on Windows",
        )
    if sys.platform == "darwin":
        sandbox = shutil.which(DARWIN_RUNTIME_SANDBOX)
        if sandbox is None:
            return CommandResult(
                tuple(argv),
                "containment_unavailable",
                None,
                None,
                b"",
                b"",
                False,
                False,
                "the Darwin runtime sandbox is unavailable",
            )
        return run_command(
            (
                sandbox,
                "-p",
                _darwin_runtime_sandbox_profile(
                    cwd,
                    argv,
                    read_roots,
                    observation_root=observation_root,
                    observation_transient_roots=observation_transient_roots,
                ),
                *argv,
            ),
            cwd=cwd,
            timeout=timeout,
            max_output=max_output,
            inherit_env=False,
        )
    return CommandResult(
        tuple(argv),
        "containment_unavailable",
        None,
        None,
        b"",
        b"",
        False,
        False,
        "strict candidate runtime containment is unavailable on this platform",
    )


def _balanced(text: str, start: int, opening: str = "(", closing: str = ")") -> int:
    depth = 0
    quote: Optional[str] = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ObservationError(f"unbalanced {opening}{closing} expression")


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _routine_family(name: str) -> str:
    base = name.lower()
    if base.startswith("cblas_"):
        base = base[6:]
    base = base.rstrip("_")
    if base in UTILITY or base in RUNTIME:
        return base
    for prefix in (
        "is",
        "id",
        "ic",
        "iz",
        "sc",
        "dz",
        "ss",
        "sd",
        "ds",
        "cs",
        "zd",
        "ch",
        "zh",
    ):
        if (
            base.startswith(prefix)
            and base[len(prefix) :] in LEVEL1 | LEVEL2 | LEVEL3 | RETAINED
        ):
            return base[len(prefix) :]
    if base and base[0] in "sdcz" and base[1:] in LEVEL1 | LEVEL2 | LEVEL3 | RETAINED:
        return base[1:]
    return base


def classify_symbol(name: str, source_path: str, visibility: str) -> str:
    family = _routine_family(name)
    if name.startswith("zynum_internal_") or visibility == "hidden":
        return "internal_bridge"
    if "kernels/arch/" in source_path or name.startswith("zynum_blas_amx_"):
        return "architecture_extension"
    if family in UTILITY:
        return "utility_error"
    if family in RUNTIME:
        return "runtime_control"
    if family in RETAINED:
        return "retained_extension"
    if family in LEVEL1:
        return "standard_level1"
    if family in LEVEL2:
        return "standard_level2"
    if family in LEVEL3:
        return "standard_level3"
    return "unclassified"


def scan_zig_exports(source_root: Path, repository_root: Path) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for path in sorted(source_root.rglob("*.zig")):
        text = path.read_text(encoding="utf-8")
        display = path.relative_to(repository_root).as_posix()
        occupied: list[tuple[int, int]] = []
        for match in re.finditer(
            r"\bpub\s+export\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text
        ):
            close = _balanced(text, text.find("(", match.start()))
            body = text[
                match.start() : text.find("{", close)
                if "{" in text[close:]
                else close + 1
            ]
            callconv_match = re.search(r"callconv\s*\(\s*\.([A-Za-z0-9_]+)\s*\)", body)
            name = match.group(1)
            entry = {
                "source_path": display,
                "line": _line_number(text, match.start()),
                "source_target": name,
                "exported_name": name,
                "callconv": callconv_match.group(1)
                if callconv_match
                else "unspecified",
                "visibility": "default",
                "declaration_kind": "pub_export_fn",
            }
            entry["category"] = classify_symbol(name, display, "default")
            declarations.append(entry)
            occupied.append((match.start(), close))
        for match in re.finditer(r"@export\s*\(", text):
            if any(start <= match.start() <= end for start, end in occupied):
                continue
            open_offset = text.find("(", match.start())
            close = _balanced(text, open_offset)
            expression = text[open_offset + 1 : close]
            target_match = re.search(r"&\s*([A-Za-z_][A-Za-z0-9_]*)", expression)
            name_match = re.search(r"\.name\s*=\s*\"([^\"]+)\"", expression)
            visibility_match = re.search(
                r"\.visibility\s*=\s*\.([A-Za-z0-9_]+)", expression
            )
            if not target_match or not name_match:
                declarations.append(
                    {
                        "source_path": display,
                        "line": _line_number(text, match.start()),
                        "source_target": "not_observed",
                        "exported_name": "not_observed",
                        "callconv": "not_observed",
                        "visibility": "not_observed",
                        "declaration_kind": "at_export",
                        "category": "unclassified",
                        "reason": "export expression did not expose a literal target and name",
                    }
                )
                continue
            target = target_match.group(1)
            name = name_match.group(1)
            visibility = visibility_match.group(1) if visibility_match else "default"
            fn_match = re.search(
                r"\bfn\s+" + re.escape(target) + r"\s*\([^)]*\)[^{;]*", text
            )
            callconv_match = (
                re.search(r"callconv\s*\(\s*\.([A-Za-z0-9_]+)\s*\)", fn_match.group(0))
                if fn_match
                else None
            )
            entry = {
                "source_path": display,
                "line": _line_number(text, match.start()),
                "source_target": target,
                "exported_name": name,
                "callconv": callconv_match.group(1)
                if callconv_match
                else "unspecified",
                "visibility": visibility,
                "declaration_kind": "at_export",
            }
            entry["category"] = classify_symbol(name, display, visibility)
            declarations.append(entry)
    return sorted(
        declarations,
        key=lambda item: (item["source_path"], item["line"], item["exported_name"]),
    )


def strip_c_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)


def split_c_params(text: str) -> list[str]:
    if text.strip() in ("", "void"):
        return []
    result: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(" ".join(text[start:index].split()))
            start = index + 1
    result.append(" ".join(text[start:].split()))
    return result


def parse_c_header(text: str, identity: str) -> dict[str, Any]:
    clean = strip_c_comments(text)
    macros = []
    for match in re.finditer(r"^\s*#\s*define\s+(\w+)(?:\s+(.*?))?\s*$", clean, re.M):
        macros.append({"name": match.group(1), "value": (match.group(2) or "").strip()})
    structs = []
    for match in re.finditer(
        r"typedef\s+struct(?:\s+\w+)?\s*\{(.*?)\}\s*(\w+)\s*;", clean, re.S
    ):
        fields = []
        for declaration in match.group(1).split(";"):
            declaration = " ".join(declaration.split())
            field = re.match(r"(.+?)\s+([A-Za-z_]\w*)$", declaration)
            if field:
                fields.append(
                    {
                        "name": field.group(2),
                        "type": field.group(1),
                        "declaration": declaration,
                    }
                )
        structs.append({"name": match.group(2), "fields": fields})
    enums = []
    for match in re.finditer(
        r"typedef\s+enum\s+(\w+)\s*\{(.*?)\}\s*(\w+)\s*;", clean, re.S
    ):
        constants = []
        for part in match.group(2).split(","):
            constant = re.match(r"\s*(\w+)\s*=\s*([^,]+)\s*$", part)
            if constant:
                constants.append(
                    {"name": constant.group(1), "value": constant.group(2).strip()}
                )
        enums.append(
            {"tag": match.group(1), "alias": match.group(3), "constants": constants}
        )
    aliases = []
    scrubbed = re.sub(r"typedef\s+(?:struct|enum).*?}\s*\w+\s*;", "", clean, flags=re.S)
    for match in re.finditer(r"\btypedef\s+(.+?)\s+(\w+)\s*;", scrubbed, re.S):
        source = " ".join(match.group(1).split())
        if "{" not in source:
            aliases.append({"name": match.group(2), "target": source})
    functions = []
    function_clean = re.sub(r"^\s*#.*$", "", clean, flags=re.M)
    function_clean = re.sub(r"extern\s+\"C\"\s*\{", "", function_clean)
    function_clean = re.sub(r"^\s*}\s*$", "", function_clean, flags=re.M)
    for statement in function_clean.split(";"):
        normalized = " ".join(statement.split())
        match = re.search(
            r"(?:^|\})\s*([A-Za-z_][\w\s*]*?)\s+([A-Za-z_]\w*)\s*\((.*)\)\s*$",
            normalized,
        )
        if not match or normalized.startswith("typedef"):
            continue
        return_type = " ".join(match.group(1).split())
        name = match.group(2)
        params = []
        for raw in split_c_params(match.group(3)):
            item = re.match(r"(.+?)([A-Za-z_]\w*)$", raw)
            params.append(
                {
                    "declaration": raw,
                    "name": item.group(2) if item else "not_observed",
                    "type": item.group(1).strip() if item else raw,
                }
            )
        functions.append(
            {
                "name": name,
                "return": return_type,
                "params": params,
                "prototype": normalized,
            }
        )
    guard = re.search(r"#\s*ifndef\s+(\w+)", clean)
    return {
        "identity": identity,
        "include_guard": guard.group(1) if guard else "not_observed",
        "macros": sorted(macros, key=lambda item: item["name"]),
        "structs": sorted(structs, key=lambda item: item["name"]),
        "aliases": sorted(aliases, key=lambda item: item["name"]),
        "enums": sorted(enums, key=lambda item: item["alias"]),
        "functions": sorted(functions, key=lambda item: item["name"]),
    }


def parse_fortran_module(text: str, identity: str) -> dict[str, Any]:
    logical_lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if pending:
            line = pending + " " + line.lstrip("& ")
            pending = ""
        if line.endswith("&"):
            pending = line[:-1].rstrip()
        else:
            logical_lines.append(" ".join(line.split()))
    if pending:
        logical_lines.append(pending)
    logical = "\n".join(logical_lines)
    module_match = re.search(r"^module\s+(\w+)", logical, re.M | re.I)
    use_match = re.search(
        r"use,\s*intrinsic\s*::\s*(\w+),\s*only:\s*([^\n]+)", logical, re.I
    )
    parameters = []
    for match in re.finditer(
        r"^integer,\s*parameter\s*::\s*(\w+)\s*=\s*([^\n]+)$", logical, re.M | re.I
    ):
        parameters.append({"name": match.group(1), "value": match.group(2).strip()})
    procedures = []
    pattern = re.compile(
        r"^(function|subroutine)\s+(\w+)\s*\((.*?)\)\s*bind\s*\(\s*C\s*,\s*name\s*=\s*\"([^\"]+)\"\s*\)"
        r"(?:\s*result\s*\(\s*(\w+)\s*\))?\n(.*?)^end\s+(?:function|subroutine)\s+\2\s*$",
        re.M | re.S | re.I,
    )
    for match in pattern.finditer(logical):
        kind, name, args_text, bind_name, result_name, body = match.groups()
        args = [item.strip() for item in args_text.split(",") if item.strip()]
        imports: list[str] = []
        declarations: dict[str, dict[str, Any]] = {}
        result_declaration: Any = not_observed("procedure has no result")
        for line in body.splitlines():
            import_match = re.match(r"import\s*::\s*(.*)", line, re.I)
            if import_match:
                imports.extend(
                    item.strip() for item in import_match.group(1).split(",")
                )
                continue
            declaration_match = re.match(r"(.+?)\s*::\s*(.+)$", line)
            if not declaration_match:
                continue
            prefix, declared = declaration_match.groups()
            intent_match = re.search(r"intent\s*\(\s*(inout|in|out)\s*\)", prefix, re.I)
            value_flag = bool(re.search(r"(?:^|,)\s*value(?:,|$)", prefix, re.I))
            for item in [part.strip() for part in declared.split(",")]:
                variable = re.match(r"(\w+)(\(.*\))?$", item)
                if not variable:
                    continue
                detail = {
                    "declaration": f"{prefix} :: {item}",
                    "type": prefix.split(",", 1)[0].strip(),
                    "intent": intent_match.group(1).lower()
                    if intent_match
                    else "unspecified",
                    "value": value_flag,
                    "array": variable.group(2) or "scalar",
                }
                declarations[variable.group(1)] = detail
                if result_name and variable.group(1).lower() == result_name.lower():
                    result_declaration = detail
        procedures.append(
            {
                "name": name,
                "bind_name": bind_name,
                "procedure_kind": kind.lower(),
                "result_name": result_name or "not_applicable",
                "result": result_declaration,
                "imports": sorted(set(imports)),
                "params": [
                    {
                        "name": arg,
                        **declarations.get(
                            arg,
                            {
                                "declaration": "not_observed",
                                "intent": "not_observed",
                                "value": False,
                                "array": "not_observed",
                                "type": "not_observed",
                            },
                        ),
                    }
                    for arg in args
                ],
            }
        )
    aliases = [
        item
        for item in parameters
        if item["value"] in {entry["name"] for entry in parameters}
    ]
    return {
        "identity": identity,
        "module": module_match.group(1) if module_match else "not_observed",
        "intrinsic_module": use_match.group(1) if use_match else "not_observed",
        "imports": sorted(item.strip() for item in use_match.group(2).split(","))
        if use_match
        else [],
        "public_parameters": sorted(parameters, key=lambda item: item["name"]),
        "compatibility_aliases": sorted(aliases, key=lambda item: item["name"]),
        "procedures": sorted(procedures, key=lambda item: item["bind_name"]),
    }


def parse_pkgconfig(text: str, identity: str) -> dict[str, Any]:
    variables = []
    fields = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and (":" not in line or line.index("=") < line.index(":")):
            name, value = line.split("=", 1)
            variables.append({"name": name.strip(), "value": value.strip()})
        elif ":" in line:
            name, value = line.split(":", 1)
            fields.append({"name": name.strip(), "value": value.strip()})
    return {"identity": identity, "variables": variables, "fields": fields}


def observe_projections(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    paths = {
        "manifest": Path("include/zynum/blas/abi_manifest.json"),
        "blas_header": Path("include/zynum/blas/blas.h"),
        "cblas_header": Path("include/zynum/blas/cblas.h"),
        "fortran_module": Path("include/zynum/blas/blas.f90"),
        "pkgconfig": Path("pkgconfig/zynum_blas.pc"),
    }
    inputs = []
    contents: dict[str, str] = {}
    for key, relative in paths.items():
        path = root / relative
        if not path.is_file():
            raise ObservationError(
                f"required projection missing: {relative.as_posix()}"
            )
        data = path.read_bytes()
        contents[key] = data.decode("utf-8")
        inputs.append({"path": relative.as_posix(), "sha256": sha256_bytes(data)})
    manifest = json.loads(contents["manifest"])
    blas = parse_c_header(contents["blas_header"], paths["blas_header"].as_posix())
    cblas = parse_c_header(contents["cblas_header"], paths["cblas_header"].as_posix())
    fortran = parse_fortran_module(
        contents["fortran_module"], paths["fortran_module"].as_posix()
    )
    pkg = parse_pkgconfig(contents["pkgconfig"], paths["pkgconfig"].as_posix())
    c_symbols = {item["name"] for item in blas["functions"]} | {
        item["name"] for item in cblas["functions"]
    }
    f_symbols = {item["bind_name"] for item in fortran["procedures"]}
    manifest_symbols: set[str] = set()
    cross_reference = []
    for group in ("fortran", "cblas"):
        for item in manifest.get(group, {}).get("exports", []):
            name = item["name"]
            manifest_symbols.add(name)
            expected_c = name in c_symbols
            expected_f = group == "fortran" and name in f_symbols
            cross_reference.append(
                {
                    "symbol": name,
                    "manifest_group": group,
                    "manifest_mapping": item,
                    "c_prototype": next(
                        (
                            entry
                            for entry in blas["functions"] + cblas["functions"]
                            if entry["name"] == name
                        ),
                        not_observed("no matching C prototype"),
                    ),
                    "fortran_procedure": next(
                        (
                            entry
                            for entry in fortran["procedures"]
                            if entry["bind_name"] == name
                        ),
                        not_observed(
                            "not a Fortran projection symbol"
                            if group != "fortran"
                            else "no matching Fortran procedure"
                        ),
                    ),
                    "missing": sorted(
                        (["c"] if not expected_c else [])
                        + (["fortran"] if group == "fortran" and not expected_f else [])
                    ),
                }
            )
    extra_c = sorted(c_symbols - manifest_symbols)
    extra_fortran = sorted(f_symbols - manifest_symbols)
    projection = {
        "manifest": manifest,
        "c_headers": {"blas": blas, "cblas": cblas},
        "fortran_module": fortran,
        "pkgconfig": pkg,
        "cross_reference": sorted(cross_reference, key=lambda item: item["symbol"]),
        "missing": sorted(
            {
                name
                for item in cross_reference
                for name in ([item["symbol"]] if item["missing"] else [])
            }
        ),
        "extra": {"c": extra_c, "fortran": extra_fortran},
        "unclassified": [],
        "integer_abi": {
            "model": manifest.get("blas_integer_abi", "not_observed"),
            "bits": manifest.get("blas_integer_bits", "not_observed"),
        },
    }
    counts = {
        "manifest_fortran": len(manifest.get("fortran", {}).get("exports", [])),
        "manifest_cblas": len(manifest.get("cblas", {}).get("exports", [])),
        "c_prototypes": len(c_symbols),
        "fortran_procedures": len(f_symbols),
    }
    return projection, inputs, counts


def parse_nm_symbols(text: str) -> list[dict[str, Any]]:
    symbols = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.endswith(":"):
            continue
        fields = line.split()
        symbol_type = "not_observed"
        name = fields[-1]
        if len(fields) >= 2 and len(fields[-2]) == 1:
            symbol_type = fields[-2]
        else:
            descriptor = re.search(r"\(([^)]+)\)", line)
            if descriptor:
                value = descriptor.group(1).strip()
                symbol_type = (
                    "undefined"
                    if value.lower() == "undefined"
                    else "section:" + value
                    if "," in value
                    else value.lower().replace(" ", "_")
                )
        if re.match(r"^[A-Za-z_?$@.][\w?$@.]*$", name):
            lower = line.lower()
            visibility = (
                "hidden"
                if "private external" in lower
                else ("default" if "external" in lower else "local")
            )
            symbols.append(
                {
                    "name": name,
                    "type": symbol_type,
                    "visibility": visibility,
                    "defined": symbol_type.lower() not in ("u", "undefined"),
                    "declaration": line,
                }
            )
    unique = {
        (item["name"], item["type"], item["declaration"]): item for item in symbols
    }
    return sorted(
        unique.values(),
        key=lambda item: (item["name"], item["type"], item["declaration"]),
    )


def parse_elf_metadata(text: str) -> dict[str, Any]:
    soname = re.search(r"\(SONAME\).*?\[(.*?)\]", text)
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", text)
    rpath = re.findall(r"\((?:RPATH)\).*?\[(.*?)\]", text)
    runpath = re.findall(r"\((?:RUNPATH)\).*?\[(.*?)\]", text)
    versions = sorted(
        set(re.findall(r"\b([A-Za-z][A-Za-z0-9_.-]+_[0-9][A-Za-z0-9_.-]*)\b", text))
    )
    elf_class = re.search(r"^\s*Class:\s*(.+)$", text, re.M)
    data = re.search(r"^\s*Data:\s*(.+)$", text, re.M)
    machine = re.search(r"^\s*Machine:\s*(.+)$", text, re.M)
    file_type = re.search(r"^\s*Type:\s*(.+)$", text, re.M)
    return {
        "class": elf_class.group(1).strip()
        if elf_class
        else not_observed("ELF class was absent"),
        "data_encoding": data.group(1).strip()
        if data
        else not_observed("ELF data encoding was absent"),
        "machine": machine.group(1).strip()
        if machine
        else not_observed("ELF machine was absent"),
        "type": file_type.group(1).strip()
        if file_type
        else not_observed("ELF type was absent"),
        "soname": soname.group(1) if soname else not_observed("SONAME was absent"),
        "needed": sorted(set(needed)),
        "symbol_versions": versions,
        "rpath": sorted(set(rpath)),
        "runpath": sorted(set(runpath)),
    }


def parse_macho_metadata(load_commands: str, libraries: str = "") -> dict[str, Any]:
    blocks = re.split(r"(?=^Load command \d+\s*$)", load_commands, flags=re.M)
    identifier_block = next(
        (
            block
            for block in blocks
            if re.search(r"^\s*cmd LC_ID_DYLIB\s*$", block, re.M)
        ),
        "",
    )
    id_value: Any = not_observed("install identifier was absent")
    current: Any = not_observed("current version was absent")
    compatibility: Any = not_observed("compatibility version was absent")
    identifier = re.search(r"^\s*name\s+([^\s]+)", identifier_block, re.M)
    current_match = re.search(r"^\s*current version\s+([^\n]+)", identifier_block, re.M)
    compatibility_match = re.search(
        r"^\s*compatibility version\s+([^\n]+)", identifier_block, re.M
    )
    if identifier:
        id_value = identifier.group(1).strip()
    if current_match:
        current = current_match.group(1).strip()
    if compatibility_match:
        compatibility = compatibility_match.group(1).strip()
    deps = []
    for line in libraries.splitlines()[1:]:
        match = re.match(
            r"\s*(\S+)\s+\(compatibility version ([^,]+), current version ([^)]+)\)",
            line,
        )
        if match and match.group(1) != id_value:
            deps.append(
                {
                    "name": match.group(1),
                    "compatibility_version": match.group(2),
                    "current_version": match.group(3),
                }
            )
    rpaths = [
        match.group(1)
        for block in blocks
        if re.search(r"^\s*cmd LC_RPATH\s*$", block, re.M)
        for match in [re.search(r"^\s*path\s+(\S+)", block, re.M)]
        if match
    ]
    build_block = next(
        (
            block
            for block in blocks
            if re.search(r"^\s*cmd LC_BUILD_VERSION\s*$", block, re.M)
        ),
        "",
    )
    minimum = re.findall(r"\bminos\s+([0-9.]+)", build_block)
    if not minimum:
        minimum = re.findall(
            r"LC_VERSION_MIN_\w+.*?\bversion\s+([0-9.]+)", load_commands, re.S
        )
    platform_values = re.findall(r"^\s*platform\s+(\S+)", build_block, re.M)
    platform_names = {"1": "macOS", "2": "iOS", "3": "tvOS", "4": "watchOS"}
    platform = (
        platform_names.get(platform_values[0], platform_values[0])
        if platform_values
        else not_observed("platform was absent")
    )
    sdk_values = re.findall(r"^\s*sdk\s+([0-9.]+)", build_block, re.M)
    return {
        "id": id_value,
        "dependencies": sorted(deps, key=lambda item: item["name"]),
        "current_version": current,
        "compatibility_version": compatibility,
        "platform": platform,
        "minimum_platform": minimum[0]
        if minimum
        else not_observed("minimum platform was absent"),
        "sdk": sdk_values[0] if sdk_values else not_observed("SDK version was absent"),
        "rpath": sorted(set(rpaths)),
    }


def parse_pe_metadata(text: str) -> dict[str, Any]:
    dll_name = re.search(r"^\s*Name\s*:\s*([^\s]+\.dll)\b", text, re.I | re.M)
    imports = sorted(set(re.findall(r"DLL Name:\s*([^\s]+)", text, re.I)))
    image_base = re.search(r"ImageBase\s+([0-9A-Fa-fx]+)", text)
    subsystem = re.search(r"Subsystem\s+([0-9A-Fa-fx]+)(?:\s+(.+))?", text)
    characteristics = sorted(
        set(
            re.findall(
                r"^\s*(?:DLL|EXEC_P|HAS_RELOC|LARGE_ADDRESS_AWARE|DYNAMIC_BASE|NX_COMPAT|HIGH_ENTROPY_VA)\s*$",
                text,
                re.M,
            )
        )
    )
    crt = sorted(
        {name for name in imports if re.search(r"(?:ucrt|msvcr|vcruntime)", name, re.I)}
    )
    exports = []
    for raw in text.splitlines():
        match = re.match(
            r"\s*([0-9]+)\s+([0-9A-Fa-f]+)\s+([A-Za-z_?$@][^\s]*)\s*$", raw
        )
        if match:
            name = match.group(3)
            exports.append(
                {
                    "ordinal": int(match.group(1)),
                    "rva": match.group(2),
                    "name": name,
                    "decoration": "decorated"
                    if name.startswith(("_", "?")) or "@" in name
                    else "plain",
                }
            )
    return {
        "dll_name": dll_name.group(1)
        if dll_name
        else not_observed("DLL name was absent"),
        "exports": sorted(exports, key=lambda item: (item["ordinal"], item["name"])),
        "imports": imports,
        "crt": crt,
        "image_base": image_base.group(1)
        if image_base
        else not_observed("image base was absent"),
        "subsystem": (
            {
                "value": subsystem.group(1),
                "description": (subsystem.group(2) or "").strip(),
            }
            if subsystem
            else not_observed("subsystem was absent")
        ),
        "characteristics": characteristics
        if characteristics
        else not_observed("image characteristics were absent"),
        "import_library": not_observed("import library was not supplied"),
    }


def _canonical_architectures(text: str) -> list[str]:
    aliases = (
        (r"(?<![A-Za-z0-9])(?:aarch64|arm64)(?![A-Za-z0-9])", "arm64"),
        (r"(?<![A-Za-z0-9])(?:x86_64|x86-64)(?![A-Za-z0-9])", "x86_64"),
        (r"(?<![A-Za-z0-9])i386(?![A-Za-z0-9])", "i386"),
    )
    lowered = text.lower()
    return sorted(
        {canonical for pattern, canonical in aliases if re.search(pattern, lowered)}
    )


def _architecture_value(text: str) -> Any:
    architectures = _canonical_architectures(text)
    if not architectures:
        return "not_observed"
    return architectures[0] if len(architectures) == 1 else architectures


def _read_prefix(path: Path, size: int) -> bytes:
    with path.open("rb") as stream:
        return stream.read(size)


def _detect_format(path: Path, file_text: str) -> tuple[str, Any]:
    lowered = file_text.lower()
    arch = _architecture_value(file_text)
    if "mach-o" in lowered:
        return "Mach-O", arch
    if "elf" in lowered:
        return "ELF", arch
    if "pe32" in lowered or "coff" in lowered:
        return "PE/COFF", arch
    data = _read_prefix(path, 4)
    if data == b"\x7fELF":
        return "ELF", arch
    if data[:2] == b"MZ":
        return "PE/COFF", arch
    if data in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf") or data in MACHO_FAT_MAGICS:
        return "Mach-O", arch
    if data == b"!<ar":
        return "archive", arch
    return "unknown", arch


def _is_fat_or_universal_macho(path: Path, file_text: str) -> bool:
    lowered = file_text.lower()
    data = _read_prefix(path, 4)
    return (
        data in MACHO_FAT_MAGICS
        or "universal binary" in lowered
        or (
            "non-fat file" not in lowered
            and "not a fat file" not in lowered
            and ("fat file" in lowered or "architectures in" in lowered)
        )
    )


def _detect_binary_header(path: Path) -> tuple[str, str]:
    header = _read_prefix(path, 64)
    if header[:2] != b"MZ":
        return _detect_object_format(header)
    if len(header) < 64:
        return "unknown", "not_observed"
    pe_offset = int.from_bytes(header[60:64], "little")
    if pe_offset < 64 or pe_offset > MAX_ARTIFACT_OUTPUT_BYTES:
        return "unknown", "not_observed"
    with path.open("rb") as stream:
        stream.seek(pe_offset)
        pe_header = stream.read(6)
    if len(pe_header) != 6 or pe_header[:4] != b"PE\0\0":
        return "unknown", "not_observed"
    machine = int.from_bytes(pe_header[4:6], "little")
    architecture = {0x8664: "x86_64", 0xAA64: "arm64"}.get(machine, "not_observed")
    return "PE/COFF", architecture


def _detect_object_format(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\x7fELF") and len(data) >= 20:
        byte_order = "little" if data[5] == 1 else "big" if data[5] == 2 else "unknown"
        machine = (
            int.from_bytes(data[18:20], byte_order) if byte_order != "unknown" else -1
        )
        architecture = {62: "x86_64", 183: "arm64"}.get(machine, "not_observed")
        return "ELF", architecture
    macho_magics = {
        b"\xcf\xfa\xed\xfe": "little",
        b"\xfe\xed\xfa\xcf": "big",
        b"\xce\xfa\xed\xfe": "little",
        b"\xfe\xed\xfa\xce": "big",
    }
    byte_order = macho_magics.get(data[:4])
    if byte_order is not None and len(data) >= 8:
        cpu_type = int.from_bytes(data[4:8], byte_order)
        architecture = {0x01000007: "x86_64", 0x0100000C: "arm64"}.get(
            cpu_type, "not_observed"
        )
        return "Mach-O", architecture
    if len(data) >= 2:
        machine = int.from_bytes(data[:2], "little")
        if machine in (0x8664, 0xAA64):
            return "PE/COFF", {0x8664: "x86_64", 0xAA64: "arm64"}[machine]
    return "unknown", "not_observed"


def _field_observed(value: Any) -> bool:
    return not (isinstance(value, dict) and value.get("status") == "not_observed")


def _not_observed_values_have_reasons(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("status") == "not_observed" and not value.get("reason"):
            return False
        return all(_not_observed_values_have_reasons(child) for child in value.values())
    if isinstance(value, list):
        return all(_not_observed_values_have_reasons(child) for child in value)
    return True


def _metadata_issues(artifact_format: str, metadata: Any) -> list[str]:
    if not isinstance(metadata, dict) or metadata.get("status") == "not_observed":
        return ["format metadata was not observed"]
    required = {
        "ELF": ("class", "data_encoding", "machine", "type", "soname"),
        "Mach-O": (
            "id",
            "current_version",
            "compatibility_version",
            "platform",
            "minimum_platform",
            "sdk",
        ),
        "PE/COFF": (
            "dll_name",
            "image_base",
            "subsystem",
            "characteristics",
            "import_library",
        ),
    }.get(artifact_format, ())
    return [
        f"metadata field {field} was not observed"
        for field in required
        if not _field_observed(metadata.get(field))
    ]


def _normalized_artifact_symbol(name: str, source_names: set[str]) -> str:
    if name in source_names:
        return name
    if name.startswith("_") and name[1:] in source_names:
        return name[1:]
    return name


def _is_toolchain_symbol(name: str) -> bool:
    normalized = name.lstrip("_")
    return normalized in {
        "TMC_END",
        "bss_start",
        "dso_handle",
        "edata",
        "end",
        "fini",
        "init",
        "mh_dylib_header",
        "mh_execute_header",
    }


def observe_artifact(
    path: Optional[Path],
    kind: str,
    root: Path,
    public_source_names: set[str],
    hidden_source_names: Optional[set[str]] = None,
) -> dict[str, Any]:
    if path is None:
        return {"kind": kind, **not_observed("artifact path was not supplied")}
    display = relative_display(path, root)
    if not path.is_file():
        return {
            "kind": kind,
            "path": display,
            **not_observed("artifact does not exist"),
        }
    file_result = run_command(("file", "-b", str(path)), cwd=root)
    file_text = (
        file_result.stdout.decode("utf-8", "replace")
        if command_observed(file_result)
        else ""
    )
    artifact_format, arch = _detect_format(path, file_text)
    expected_formats = {"dynamic": {"ELF", "Mach-O", "PE/COFF"}, "static": {"archive"}}
    issues = (
        []
        if artifact_format in expected_formats[kind]
        else [f"{kind} artifact has unexpected {artifact_format} format"]
    )
    if not command_observed(file_result):
        issues.append("file description query failed")
    if isinstance(arch, list):
        issues.append("file description reported multiple architectures")
    if artifact_format == "Mach-O" and _is_fat_or_universal_macho(path, file_text):
        issues.append(
            "fat or universal Mach-O artifact is not a single-target observation"
        )
    if kind == "dynamic" and artifact_format in expected_formats[kind]:
        header_format, header_architecture = _detect_binary_header(path)
        if header_format != artifact_format:
            issues.append("binary header format did not match the file description")
        if header_architecture == "not_observed":
            issues.append("binary header architecture was not recognized")
        elif arch == "not_observed":
            arch = header_architecture
        elif isinstance(arch, str) and arch != header_architecture:
            issues.append(
                "binary header architecture did not match the file description"
            )
    architecture_commands: list[CommandResult] = []
    if artifact_format == "archive" and arch == "not_observed":
        lipo_result = run_command(("lipo", "-info", str(path)), cwd=root)
        architecture_commands.append(lipo_result)
        if command_observed(lipo_result):
            lipo_text = lipo_result.stdout.decode("utf-8", "replace")
            if _is_fat_or_universal_macho(path, lipo_text):
                issues.append(
                    "fat or universal archive is not a single-target observation"
                )
            lipo_arches = re.search(
                r"(?:architecture:\s*|are:\s*)([A-Za-z0-9_ -]+)\s*$", lipo_text
            )
            if lipo_arches:
                arch = _architecture_value(lipo_arches.group(1))
                if (
                    isinstance(arch, list)
                    and "file description reported multiple architectures" not in issues
                ):
                    issues.append("file description reported multiple architectures")
    nm_result = run_command(
        ("nm", "-g", "-m", str(path)), cwd=root, max_output=MAX_ARTIFACT_OUTPUT_BYTES
    )
    nm_public_commands = [nm_result]
    if not command_observed(nm_result):
        nm_result = run_command(
            ("nm", "-g", str(path)), cwd=root, max_output=MAX_ARTIFACT_OUTPUT_BYTES
        )
        nm_public_commands.append(nm_result)
    nm_all_result = run_command(
        ("nm", "-m", str(path)), cwd=root, max_output=MAX_ARTIFACT_OUTPUT_BYTES
    )
    nm_all_commands = [nm_all_result]
    if not command_observed(nm_all_result):
        nm_all_result = run_command(
            ("nm", str(path)), cwd=root, max_output=MAX_ARTIFACT_OUTPUT_BYTES
        )
        nm_all_commands.append(nm_all_result)
    symbols: Any = (
        parse_nm_symbols(nm_all_result.stdout.decode("utf-8", "replace"))
        if command_observed(nm_all_result)
        else not_observed("nm symbol query failed or was truncated")
    )
    public_declarations = (
        parse_nm_symbols(nm_result.stdout.decode("utf-8", "replace"))
        if command_observed(nm_result)
        else []
    )
    all_source_names = public_source_names | (hidden_source_names or set())
    public_names: Any = (
        sorted(
            {
                _normalized_artifact_symbol(item["name"], all_source_names)
                for item in public_declarations
                if item["defined"] and item["visibility"] != "hidden"
            }
        )
        if command_observed(nm_result)
        else not_observed("nm public-symbol query failed or was truncated")
    )
    if isinstance(symbols, list) and isinstance(public_names, list):
        all_defined = {
            _normalized_artifact_symbol(item["name"], all_source_names)
            for item in symbols
            if item["defined"]
        }
        hidden_names: Any = sorted(all_defined - set(public_names))
        matched: Any = sorted(set(public_names) & public_source_names)
        missing: Any = sorted(public_source_names - set(public_names))
        hidden_matched: Any = sorted(set(hidden_names) & (hidden_source_names or set()))
        hidden_missing: Any = sorted((hidden_source_names or set()) - set(hidden_names))
        toolchain_symbols: Any = sorted(
            name for name in public_names if _is_toolchain_symbol(name)
        )
        unclassified: Any = sorted(
            set(public_names) - public_source_names - set(toolchain_symbols)
        )
    else:
        hidden_names = not_observed(
            "complete and public symbol observations are required"
        )
        matched = not_observed("public symbols were not observed")
        missing = not_observed("public symbols were not observed")
        hidden_matched = not_observed("hidden symbols were not observed")
        hidden_missing = not_observed("hidden symbols were not observed")
        toolchain_symbols = not_observed("public symbols were not observed")
        unclassified = not_observed("public symbols were not observed")
    metadata: Any = not_observed("format metadata tool was not available")
    metadata_commands = []
    metadata_query_complete = False
    if artifact_format == "ELF":
        result = run_command(
            ("readelf", "-h", "-d", "--version-info", str(path)),
            cwd=root,
            max_output=MAX_ARTIFACT_OUTPUT_BYTES,
        )
        metadata_commands.append(result.observation(root))
        metadata_query_complete = command_observed(result)
        metadata = (
            parse_elf_metadata(result.stdout.decode("utf-8", "replace"))
            if metadata_query_complete
            else not_observed("readelf query failed or was truncated")
        )
    elif artifact_format == "Mach-O":
        loads = run_command(
            ("otool", "-l", str(path)), cwd=root, max_output=MAX_ARTIFACT_OUTPUT_BYTES
        )
        libs = run_command(
            ("otool", "-L", str(path)), cwd=root, max_output=MAX_ARTIFACT_OUTPUT_BYTES
        )
        metadata_commands.extend((loads.observation(root), libs.observation(root)))
        metadata_query_complete = command_observed(loads) and command_observed(libs)
        metadata = (
            parse_macho_metadata(
                loads.stdout.decode("utf-8", "replace"),
                libs.stdout.decode("utf-8", "replace"),
            )
            if metadata_query_complete
            else not_observed("otool query failed or was truncated")
        )
    elif artifact_format == "PE/COFF":
        result = run_command(
            ("objdump", "-p", str(path)), cwd=root, max_output=MAX_ARTIFACT_OUTPUT_BYTES
        )
        metadata_commands.append(result.observation(root))
        metadata_query_complete = command_observed(result)
        metadata = (
            parse_pe_metadata(result.stdout.decode("utf-8", "replace"))
            if metadata_query_complete
            else not_observed("PE/COFF metadata query failed or was truncated")
        )
    archive: Any = not_observed("artifact is not a static archive")
    if kind == "static":
        listing = run_command(
            ("ar", "-t", str(path)), cwd=root, max_output=MAX_ARTIFACT_OUTPUT_BYTES
        )
        members = (
            sorted(
                line.strip()
                for line in listing.stdout.decode("utf-8", "replace").splitlines()
                if line.strip()
            )
            if command_observed(listing)
            else []
        )
        duplicate_members = sorted(
            member for member, count in Counter(members).items() if count > 1
        )
        if not command_observed(listing):
            issues.append("archive member listing failed")
        if len(members) > MAX_ARTIFACT_ARCHIVE_MEMBERS:
            issues.append("archive member count exceeded the observation budget")
        if duplicate_members:
            issues.append("archive contains duplicate member names")
        index_members = [
            member
            for member in members
            if member in ("/", "__.SYMDEF", "__.SYMDEF SORTED")
        ]
        index_commands: list[CommandResult] = []
        if not index_members:
            armap = run_command(
                ("nm", "--print-armap", str(path)),
                cwd=root,
                max_output=MAX_ARTIFACT_OUTPUT_BYTES,
            )
            index_commands.append(armap)
            if command_observed(armap) and b"Archive index:" in armap.stdout:
                index_members = ["<nm-armap>"]
        object_members = []
        object_commands: list[CommandResult] = []
        for member in (
            item
            for item in members[:MAX_ARTIFACT_ARCHIVE_MEMBERS]
            if item not in ("/", "__.SYMDEF", "__.SYMDEF SORTED")
        ):
            if member in duplicate_members:
                object_members.append(
                    {
                        "name": member,
                        "format": "unknown",
                        "architecture": not_observed(
                            "duplicate member names make object lookup ambiguous"
                        ),
                        "command": not_observed(
                            "duplicate member names were rejected before object lookup"
                        ),
                    }
                )
                continue
            query = run_command(
                ("ar", "-p", str(path), member), cwd=root, max_output=64
            )
            object_commands.append(query)
            member_format, member_architecture = (
                _detect_object_format(query.stdout)
                if command_succeeded(query)
                else ("unknown", "not_observed")
            )
            object_members.append(
                {
                    "name": member,
                    "format": member_format,
                    "architecture": (
                        member_architecture
                        if member_architecture != "not_observed"
                        else not_observed("object architecture was not recognized")
                    ),
                    "command": query.observation(root),
                }
            )
        object_formats = sorted({item["format"] for item in object_members})
        object_architectures = sorted(
            {
                item["architecture"]
                for item in object_members
                if isinstance(item["architecture"], str)
            }
        )
        if arch == "not_observed" and len(object_architectures) == 1:
            arch = object_architectures[0]
        elif (
            isinstance(arch, str)
            and len(object_architectures) == 1
            and arch != object_architectures[0]
        ):
            issues.append("archive description and object architectures disagree")
        archive = {
            "members": members,
            "member_count": len(members),
            "duplicate_members": duplicate_members,
            "objects": object_members,
            "object_formats": object_formats,
            "object_architectures": object_architectures,
            "index": (
                {"status": "observed", "present": True, "members": index_members}
                if index_members
                else not_observed(
                    "archive listing did not expose a recognized index member"
                )
            ),
            "command": listing.observation(root),
            "index_commands": [result.observation(root) for result in index_commands],
        }
        archive["object_query_commands"] = [
            result.observation(root) for result in object_commands
        ]
    architecture = (
        arch
        if arch != "not_observed"
        else not_observed("architecture was absent from the file description")
    )
    if not _field_observed(architecture):
        issues.append("architecture was not observed")
    if not isinstance(symbols, list):
        issues.append("complete symbol declarations were not observed")
    elif any(
        item.get("type") == "not_observed" or item.get("visibility") == "not_observed"
        for item in symbols
    ):
        issues.append("a symbol declaration field was not recognized")
    if not isinstance(public_names, list):
        issues.append("public symbols were not observed")
    if kind == "dynamic":
        if not metadata_query_complete:
            issues.append("format metadata query failed")
        issues.extend(_metadata_issues(artifact_format, metadata))
    if kind == "static":
        if not isinstance(archive, dict) or archive.get("status") == "not_observed":
            issues.append("archive metadata was not observed")
        else:
            if archive.get("member_count", 0) == 0:
                issues.append("archive member list was empty")
            if not _field_observed(archive.get("index")):
                issues.append("archive index was not observed")
            if not archive.get("objects"):
                issues.append("archive object members were not observed")
            if "unknown" in archive.get("object_formats", ()):
                issues.append("archive object format was not recognized")
            if len(archive.get("object_formats", ())) > 1:
                issues.append("archive contains multiple object formats")
            if not archive.get("object_architectures"):
                issues.append("archive object architectures were not observed")
            if any(
                not isinstance(item.get("architecture"), str)
                for item in archive.get("objects", ())
            ):
                issues.append("an archive object architecture was not recognized")
            if len(archive.get("object_architectures", ())) > 1:
                issues.append("archive contains multiple object architectures")
    if isinstance(arch, list):
        issues.append("multiple architectures were observed")
    status = "failed" if issues else "observed"
    return {
        "kind": kind,
        "status": status,
        **(
            {"reason": "; ".join(sorted(set(issues))), "issues": sorted(set(issues))}
            if issues
            else {}
        ),
        "path": display,
        "sha256": sha256_file(path),
        "format": artifact_format,
        "architecture": architecture,
        "file_description": file_text.strip()
        if file_text.strip()
        else not_observed("file query failed"),
        "symbol_declarations": symbols,
        "public_symbols": public_names,
        "hidden_symbols": hidden_names,
        "source_matched_symbols": matched,
        "source_missing_symbols": missing,
        "hidden_source_matched_symbols": hidden_matched,
        "hidden_source_missing_symbols": hidden_missing,
        "toolchain_symbols": toolchain_symbols,
        "unclassified_symbols": unclassified,
        "metadata": metadata,
        "archive": archive,
        "commands": [
            file_result.observation(root),
            *(result.observation(root) for result in architecture_commands),
            *(result.observation(root) for result in nm_public_commands),
            *(result.observation(root) for result in nm_all_commands),
        ]
        + metadata_commands,
    }


def git_observation(root: Path) -> dict[str, Any]:
    def failed(reason: str) -> dict[str, Any]:
        return {
            field: not_observed(reason)
            for field in ("head", "index_sha256", "status_sha256", "status_summary")
        }

    try:
        repository = repository_git.open_repository(root)
        if repository is None:
            return failed("repository has no Git marker")
        identity = repository.observe_identity(include_index=True)
    except repository_git.RepositoryGitError:
        return failed("coherent Git identity observation failed")
    if identity.revision is None or identity.index_sha256 is None:
        return failed("repository HEAD is not available")

    state_counts: dict[str, int] = {}
    for line in identity.status_lines:
        code = line[:2]
        state_counts[code] = state_counts.get(code, 0) + 1
    return {
        "head": identity.revision,
        "index_sha256": identity.index_sha256,
        "status_sha256": identity.status_sha256,
        "status_summary": {
            "clean": not identity.status_lines,
            "entry_count": len(identity.status_lines),
            "states": dict(sorted(state_counts.items())),
        },
    }


def consumer_rows() -> list[dict[str, str]]:
    rows = [
        {"environment": "clean_source_archive", "language": "Zig", "linkage": "source"}
    ]
    for language in ("C", "C++", "Fortran"):
        for linkage in ("shared", "static"):
            rows.append(
                {
                    "environment": "binary_install_prefix",
                    "language": language,
                    "linkage": linkage,
                }
            )
    return rows


def _consumer_result(
    row: dict[str, str], status: str, reason: str, steps: list[dict[str, Any]]
) -> dict[str, Any]:
    return {**row, "status": status, "reason": reason, "steps": steps}


def _runtime_assurance() -> dict[str, Any]:
    if sys.platform == "darwin" and shutil.which(DARWIN_RUNTIME_SANDBOX) is not None:
        return {
            "status": "available",
            "backend": "darwin_sandbox",
            "environment": "scrubbed",
            "fork": "denied",
            "network": "denied",
            "file_writes": "denied",
            "file_data_reads": "default_denied",
            "read_allowlist": [
                "system_runtime",
                "consumer_workspace",
                "explicit_runtime_dependencies",
            ],
        }
    return not_observed("strict candidate runtime containment is unavailable")


def _darwin_runtime_dependency_roots(
    executable: Path,
    *,
    cwd: Path,
    timeout: float,
) -> tuple[CommandResult, tuple[Path, ...]]:
    result = run_command(("otool", "-L", str(executable)), cwd=cwd, timeout=timeout)
    roots: set[Path] = set()
    if command_observed(result):
        for line in result.stdout.decode("utf-8", "replace").splitlines()[1:]:
            dependency = line.strip().split(" (", 1)[0]
            path = Path(dependency)
            if path.is_absolute():
                roots.add(path.parent)
    return result, tuple(sorted(roots, key=lambda path: str(path)))


def _archive_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise ObservationError("source archive extraction exceeded its time budget")


def _validate_archive_member(
    member: tarfile.TarInfo,
    destination: Path,
    seen: set[Path],
) -> Path:
    target = (destination / member.name).resolve()
    try:
        target.relative_to(destination.resolve())
    except ValueError as exc:
        raise ObservationError("source archive contains an unsafe path") from exc
    if target in seen:
        raise ObservationError("source archive contains a duplicate path")
    seen.add(target)
    if member.issym() or member.islnk():
        raise ObservationError("source archive links are not accepted")
    if not (member.isfile() or member.isdir()):
        raise ObservationError("source archive contains an unsupported entry type")
    if member.isfile() and member.size > MAX_ARCHIVE_FILE_BYTES:
        raise ObservationError("source archive member exceeds the per-file budget")
    if getattr(member, "sparse", None) or any(
        key.startswith("GNU.sparse") for key in member.pax_headers
    ):
        raise ObservationError("source archive sparse members are not accepted")
    return target


def _extract_tar_contents(
    source: Path,
    destination: Path,
    timeout: float,
    max_members: int,
    max_file_bytes: int,
    max_total_bytes: int,
    max_compression_ratio: int,
) -> None:
    deadline = time.monotonic() + timeout
    compressed_size = source.stat().st_size
    members: list[tuple[tarfile.TarInfo, Path]] = []
    seen: set[Path] = set()
    total_size = 0
    with tarfile.open(source, "r:*") as archive:
        for member in archive:
            _archive_deadline(deadline)
            if len(members) >= max_members:
                raise ObservationError("source archive exceeds the member-count budget")
            target = _validate_archive_member(member, destination, seen)
            if member.isfile() and member.size > max_file_bytes:
                raise ObservationError(
                    "source archive member exceeds the per-file budget"
                )
            total_size += member.size if member.isfile() else 0
            if total_size > max_total_bytes:
                raise ObservationError(
                    "source archive exceeds the expanded-size budget"
                )
            members.append((member, target))
        if total_size > max(1, compressed_size) * max_compression_ratio:
            raise ObservationError(
                "source archive exceeds the compression-ratio budget"
            )
        for member, target in members:
            _archive_deadline(deadline)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source_stream = archive.extractfile(member)
            if source_stream is None:
                raise ObservationError("source archive member could not be read")
            remaining = member.size
            with source_stream, target.open("xb") as output:
                while remaining:
                    _archive_deadline(deadline)
                    chunk = source_stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ObservationError("source archive member was truncated")
                    output.write(chunk)
                    remaining -= len(chunk)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


def _safe_extract_tar(source: Path, destination: Path, timeout: float) -> None:
    if not source.is_file():
        raise ObservationError("source archive does not exist")
    if not destination.is_dir() or any(destination.iterdir()):
        raise ObservationError("source archive destination must be an empty directory")
    command = run_command(
        (
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-extract-archive",
            str(source.resolve()),
            str(destination.resolve()),
            f"{timeout:g}",
            str(MAX_ARCHIVE_MEMBERS),
            str(MAX_ARCHIVE_FILE_BYTES),
            str(MAX_ARCHIVE_TOTAL_BYTES),
            str(MAX_ARCHIVE_COMPRESSION_RATIO),
        ),
        cwd=destination,
        timeout=timeout,
        max_output=8192,
    )
    if not command_observed(command):
        detail = command.stderr.decode("utf-8", "replace").strip()
        raise ObservationError(
            detail or command.reason or "source archive extraction failed"
        )


def run_consumers(
    root: Path,
    install_prefix: Optional[Path],
    source_archive: Optional[Path],
    timeout: float,
) -> list[dict[str, Any]]:
    results = []
    with tempfile.TemporaryDirectory(prefix="zynum-abi-observe-") as temp_name:
        temp = Path(temp_name)
        transient = (
            (temp, Path.home())
            if install_prefix is None
            else (temp, install_prefix, Path.home())
        )
        runtime_assurance = _runtime_assurance()
        zig_row = consumer_rows()[0]
        if source_archive is None:
            results.append(
                _consumer_result(
                    zig_row, "not_observed", "source archive was not supplied", []
                )
            )
        elif not source_archive.is_file():
            results.append(
                _consumer_result(
                    zig_row, "not_observed", "source archive does not exist", []
                )
            )
        else:
            extracted = temp / "source"
            extracted.mkdir()
            try:
                _safe_extract_tar(source_archive, extracted, timeout)
                children = sorted(extracted.iterdir())
                cwd = (
                    children[0]
                    if len(children) == 1 and children[0].is_dir()
                    else extracted
                )
                source_cache = temp / "source-local-cache"
                source_global_cache = temp / "source-global-cache"
                output = temp / "source-zig-consumer"
                compile_command = run_command(
                    (
                        "zig",
                        "build-exe",
                        "-lc",
                        "--dep",
                        "zynum",
                        "-Mroot=examples/zig/matrix_multiply.zig",
                        "-lc",
                        "-Mzynum=src/zynum.zig",
                        "--cache-dir",
                        str(source_cache),
                        "--global-cache-dir",
                        str(source_global_cache),
                        "-femit-bin=" + str(output),
                    ),
                    cwd=cwd,
                    timeout=timeout,
                )
                steps = [compile_command.observation(root, transient)]
                if compile_command.status == "tool_missing":
                    result = _consumer_result(
                        zig_row, "not_observed", "Zig compiler was not found", steps
                    )
                elif command_failed(compile_command):
                    result = _consumer_result(
                        zig_row, "fail", "source consumer compile failed", steps
                    )
                else:
                    runtime_roots: tuple[Path, ...] = ()
                    dependencies_failed = False
                    if sys.platform == "darwin":
                        dependency_query, runtime_roots = (
                            _darwin_runtime_dependency_roots(
                                output,
                                cwd=temp,
                                timeout=timeout,
                            )
                        )
                        steps.append(dependency_query.observation(root, transient))
                        dependencies_failed = command_failed(dependency_query)
                    if dependencies_failed:
                        result = _consumer_result(
                            zig_row, "fail", "runtime dependency query failed", steps
                        )
                    else:
                        execute = run_candidate_executable(
                            (str(output),),
                            cwd=temp,
                            timeout=timeout,
                            read_roots=runtime_roots,
                            observation_root=root,
                            observation_transient_roots=transient,
                        )
                        steps.append(execute.observation(root, transient))
                        if execute.status == "containment_unavailable":
                            result = _consumer_result(
                                zig_row,
                                "not_observed",
                                execute.reason or "runtime containment unavailable",
                                steps,
                            )
                        else:
                            failed = command_failed(execute)
                            result = _consumer_result(
                                zig_row,
                                "fail" if failed else "pass",
                                "source consumer execution failed"
                                if failed
                                else "completed",
                                steps,
                            )
                results.append(result)
            except (tarfile.TarError, ObservationError) as exc:
                results.append(_consumer_result(zig_row, "fail", str(exc), []))
        for row in consumer_rows()[1:]:
            if install_prefix is None:
                results.append(
                    _consumer_result(
                        row, "not_observed", "install prefix was not supplied", []
                    )
                )
                continue
            include_dir = install_prefix / "include"
            lib_dir = install_prefix / "lib"
            if not include_dir.is_dir() or not lib_dir.is_dir():
                results.append(
                    _consumer_result(
                        row, "not_observed", "install prefix is incomplete", []
                    )
                )
                continue
            language = row["language"]
            source = {
                "C": root / "examples/cblas/dgemm.c",
                "C++": root / "test/abi/baseline/consumers/cpp_main.cpp",
                "Fortran": root / "examples/fortran/dgemm.f90",
            }[language]
            if not source.is_file():
                results.append(
                    _consumer_result(
                        row, "not_observed", "consumer source was not found", []
                    )
                )
                continue
            source_copy = temp / (
                "source-" + language.replace("+", "p") + source.suffix
            )
            shutil.copyfile(source, source_copy)
            compiler = {"C": "cc", "C++": "c++", "Fortran": "gfortran"}[language]
            output = temp / f"consumer-{language.replace('+', 'p')}-{row['linkage']}"
            library = None
            if row["linkage"] == "static":
                candidates = sorted(lib_dir.glob("libzynum_blas.a"))
                library = candidates[0] if candidates else None
                if library is None:
                    results.append(
                        _consumer_result(
                            row, "not_observed", "static library was not found", []
                        )
                    )
                    continue
            pkg_env = {
                "PKG_CONFIG_PATH": str(lib_dir / "pkgconfig"),
                "PKG_CONFIG_LIBDIR": str(lib_dir / "pkgconfig"),
            }
            cflags_result = run_command(
                ("pkg-config", "--cflags", "zynum_blas"),
                cwd=temp,
                timeout=timeout,
                env=pkg_env,
            )
            libs_argv = ["pkg-config"]
            if row["linkage"] == "static":
                libs_argv.append("--static")
            libs_argv.extend(("--libs", "zynum_blas"))
            libs_result = run_command(libs_argv, cwd=temp, timeout=timeout, env=pkg_env)
            steps = [
                cflags_result.observation(root, transient),
                libs_result.observation(root, transient),
            ]
            if (
                cflags_result.status == "tool_missing"
                or libs_result.status == "tool_missing"
            ):
                results.append(
                    _consumer_result(
                        row, "not_observed", "pkg-config was not found", steps
                    )
                )
                continue
            pkg_failed = any(
                command_failed(result) for result in (cflags_result, libs_result)
            )
            if pkg_failed:
                results.append(
                    _consumer_result(
                        row, "fail", "installed package metadata query failed", steps
                    )
                )
                continue
            try:
                cflags = shlex.split(cflags_result.stdout.decode("utf-8", "strict"))
                link_flags = shlex.split(libs_result.stdout.decode("utf-8", "strict"))
            except (UnicodeDecodeError, ValueError):
                results.append(
                    _consumer_result(
                        row,
                        "fail",
                        "installed package metadata output was invalid",
                        steps,
                    )
                )
                continue
            if any(str(root.resolve()) in flag for flag in cflags + link_flags):
                results.append(
                    _consumer_result(
                        row,
                        "fail",
                        "installed package metadata referenced the checkout",
                        steps,
                    )
                )
                continue
            compile_sources = [source_copy]
            if language == "Fortran":
                module_source = include_dir / "zynum/blas/blas.f90"
                if module_source.is_file():
                    compile_sources.insert(0, module_source)
            argv = [
                compiler,
                *(str(item) for item in compile_sources),
                *cflags,
                "-o",
                str(output),
            ]
            if language == "Fortran":
                argv.extend(("-I", str(include_dir / "zynum/blas")))
            if library is not None:
                argv.append(str(library))
                prefix_library_flag = "-L" + str(lib_dir)
                argv.extend(
                    flag
                    for flag in link_flags
                    if flag not in ("-lzynum_blas", prefix_library_flag)
                )
            else:
                argv.extend(link_flags)
                if os.name != "nt":
                    argv.append(f"-Wl,-rpath,{lib_dir}")
            compile_result = run_command(argv, cwd=temp, timeout=timeout)
            steps.append(compile_result.observation(root, transient))
            compile_failed = command_failed(compile_result)
            if compile_result.status == "tool_missing":
                results.append(
                    _consumer_result(
                        row, "not_observed", "compiler was not found", steps
                    )
                )
            elif compile_failed:
                results.append(
                    _consumer_result(row, "fail", "compile or link failed", steps)
                )
            else:
                runtime_roots = (lib_dir,)
                dependencies_failed = False
                if sys.platform == "darwin":
                    dependency_query, dependency_roots = (
                        _darwin_runtime_dependency_roots(
                            output,
                            cwd=temp,
                            timeout=timeout,
                        )
                    )
                    steps.append(dependency_query.observation(root, transient))
                    runtime_roots = (*runtime_roots, *dependency_roots)
                    dependencies_failed = command_failed(dependency_query)
                if dependencies_failed:
                    result = _consumer_result(
                        row, "fail", "runtime dependency query failed", steps
                    )
                else:
                    execute = run_candidate_executable(
                        (str(output),),
                        cwd=temp,
                        timeout=timeout,
                        read_roots=runtime_roots,
                        observation_root=root,
                        observation_transient_roots=transient,
                    )
                    steps.append(execute.observation(root, transient))
                    if execute.status == "containment_unavailable":
                        result = _consumer_result(
                            row,
                            "not_observed",
                            execute.reason or "runtime containment unavailable",
                            steps,
                        )
                    else:
                        failed = command_failed(execute)
                        result = _consumer_result(
                            row,
                            "fail" if failed else "pass",
                            "execution failed" if failed else "completed",
                            steps,
                        )
                results.append(result)
        for result in results:
            result.setdefault("runtime_assurance", runtime_assurance)
            result.setdefault("candidate_build_scripts_executed", False)
    return results


def _architecture_matches(value: Any, expected: set[str]) -> bool:
    return isinstance(value, str) and value in expected


def _artifact_matches_target(artifact: dict[str, Any], target: str) -> bool:
    if artifact.get("status") != "observed":
        return False
    binding = TARGET_BINDINGS[target]
    if (
        artifact.get("kind") == "dynamic"
        and artifact.get("format") != binding["format"]
    ):
        return False
    if artifact.get("kind") == "static" and artifact.get("format") != "archive":
        return False
    if not _architecture_matches(
        artifact.get("architecture"), binding["architectures"]
    ):
        return False
    if artifact.get("kind") == "static":
        archive = artifact.get("archive")
        if not isinstance(archive, dict) or archive.get("object_formats") != [
            binding["format"]
        ]:
            return False
        if archive.get("object_architectures") != sorted(binding["architectures"]):
            return False
    expected_platform = binding.get("platform")
    if expected_platform is not None and artifact.get("kind") == "dynamic":
        metadata = artifact.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("platform") != expected_platform
        ):
            return False
    return True


def observe_platforms(
    targets: Sequence[str], artifacts: Mapping[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for platform in PLATFORMS:
        if platform not in targets:
            rows.append(
                {
                    "target": platform,
                    **not_observed("target was not explicitly requested"),
                }
            )
        elif all(
            _artifact_matches_target(artifact, platform)
            for artifact in artifacts.values()
        ):
            rows.append(
                {
                    "target": platform,
                    "status": "observed",
                    "reason": "both artifacts match the explicit target",
                }
            )
        else:
            rows.append(
                {
                    "target": platform,
                    "status": "failed",
                    "reason": "artifact format, architecture, or platform metadata did not match the explicit target",
                }
            )
    return rows


def validate_policy(policy: Any) -> None:
    if not isinstance(policy, dict):
        raise ObservationError("policy must be a JSON object")
    if policy.get("role") != "observer_not_abi_authority":
        raise ObservationError("policy role must be observer_not_abi_authority")
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise ObservationError("policy schema version does not match the observer")
    if policy.get("artifact_build_configuration") != ARTIFACT_BUILD_CONFIGURATION:
        raise ObservationError(
            "policy artifact build configuration does not match the observer"
        )
    if tuple(policy.get("platform_matrix", ())) != PLATFORMS:
        raise ObservationError("policy platform matrix does not match the schema")
    if policy.get("consumer_matrix") != consumer_rows():
        raise ObservationError("policy consumer matrix does not match the schema")
    if policy.get("installed_zig_module_abi") is not False:
        raise ObservationError("policy must not introduce an installed Zig module ABI")
    if policy.get("artifact_policy", {}).get("required_kinds") != ["dynamic", "static"]:
        raise ObservationError("policy must require both dynamic and static artifacts")
    expected_bindings = {
        target: {
            "architectures": sorted(binding["architectures"]),
            "dynamic_format": binding["format"],
            "static_member_format": binding["format"],
            **(
                {"dynamic_platform": binding["platform"]}
                if "platform" in binding
                else {}
            ),
        }
        for target, binding in TARGET_BINDINGS.items()
    }
    if policy.get("artifact_policy", {}).get("target_bindings") != expected_bindings:
        raise ObservationError("policy target bindings do not match the observer")
    expected_archive_limits = {
        "max_members": MAX_ARCHIVE_MEMBERS,
        "max_file_bytes": MAX_ARCHIVE_FILE_BYTES,
        "max_total_bytes": MAX_ARCHIVE_TOTAL_BYTES,
        "max_compression_ratio": MAX_ARCHIVE_COMPRESSION_RATIO,
        "hard_timeout": True,
        "links_sparse_and_special_entries": "rejected",
    }
    if (
        policy.get("source_archive_policy", {}).get("extraction_limits")
        != expected_archive_limits
    ):
        raise ObservationError(
            "policy source archive extraction limits do not match the observer"
        )
    if (
        policy.get("artifact_policy", {}).get("max_archive_members")
        != MAX_ARTIFACT_ARCHIVE_MEMBERS
    ):
        raise ObservationError(
            "policy artifact archive limit does not match the observer"
        )
    expected_execution_policy = {
        "candidate_build_scripts_executed": False,
        "compile_and_link_tools": "trusted_toolchain",
        "runtime": {
            "darwin": {
                "backend": "sandbox-exec",
                "environment": "scrubbed",
                "fork": "denied",
                "network": "denied",
                "file_writes": "denied",
                "file_data_reads": "default_denied",
                "read_allowlist": [
                    "system_runtime",
                    "consumer_workspace",
                    "explicit_runtime_dependencies",
                ],
            },
            "windows": "not_observed_without_filesystem_and_network_isolation",
            "other": "not_observed_without_strict_runtime_containment",
        },
    }
    if policy.get("consumer_execution_policy") != expected_execution_policy:
        raise ObservationError(
            "policy consumer execution boundary does not match the observer"
        )
    allowed = set(policy.get("classification", {}).get("allowed", ()))
    if "unclassified" not in allowed or not policy.get("classification", {}).get(
        "unknown_is_failure"
    ):
        raise ObservationError("policy must fail closed for unknown classifications")


def build_observation(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    policy_path = args.policy.resolve()
    if not (root / "src").is_dir():
        raise ObservationError("root does not contain src")
    policy_data = policy_path.read_bytes()
    policy = json.loads(policy_data)
    validate_policy(policy)
    requested_targets = sorted(set(args.target))
    unknown_targets = sorted(set(requested_targets) - set(PLATFORMS))
    if unknown_targets:
        raise ObservationError(
            "requested target is outside the declared platform matrix"
        )
    if len(requested_targets) > 1:
        raise ObservationError("one observation may bind at most one platform target")
    artifacts_requested = (
        args.dynamic_library is not None or args.static_library is not None
    )
    if artifacts_requested and (
        args.dynamic_library is None or args.static_library is None
    ):
        raise ObservationError(
            "whole-artifact observation requires both dynamic and static libraries"
        )
    if artifacts_requested != bool(requested_targets):
        raise ObservationError(
            "artifact observation and one declared platform target must be requested together"
        )
    artifact_build_configuration = copy.deepcopy(ARTIFACT_BUILD_CONFIGURATION)
    configured_target = artifact_build_configuration["declared_target"]
    if artifacts_requested and requested_targets != [configured_target]:
        raise ObservationError(
            "artifact target does not match the canonical artifact build configuration"
        )
    source_declarations = scan_zig_exports(root / "src", root)
    projection, projection_inputs, projection_counts = observe_projections(root)
    public_source_names = {
        item["exported_name"]
        for item in source_declarations
        if item["exported_name"] != "not_observed" and item["visibility"] != "hidden"
    }
    hidden_source_names = {
        item["exported_name"]
        for item in source_declarations
        if item["exported_name"] != "not_observed" and item["visibility"] == "hidden"
    }
    generator_declarations = [
        item
        for item in source_declarations
        if item["source_path"] in ("src/blas/abi/cblas.zig", "src/blas/abi/fortran.zig")
    ]
    generator_names = {item["exported_name"] for item in generator_declarations}
    manifest_names = {item["symbol"] for item in projection["cross_reference"]}
    for item in projection["cross_reference"]:
        item["source_declarations"] = [
            entry
            for entry in generator_declarations
            if entry["exported_name"] == item["symbol"]
        ]
        if not item["source_declarations"]:
            item["missing"].append("source")
    projection["source_missing"] = sorted(manifest_names - generator_names)
    projection["source_extra"] = sorted(generator_names - manifest_names)
    projection["missing"] = sorted(
        set(projection["missing"]) | set(projection["source_missing"])
    )
    dynamic = observe_artifact(
        args.dynamic_library.resolve() if args.dynamic_library else None,
        "dynamic",
        root,
        public_source_names,
        hidden_source_names,
    )
    static = observe_artifact(
        args.static_library.resolve() if args.static_library else None,
        "static",
        root,
        public_source_names,
        hidden_source_names,
    )
    artifacts = {"dynamic": dynamic, "static": static}
    platform_rows = observe_platforms(args.target, artifacts)
    if args.run_consumers:
        consumers = run_consumers(
            root,
            args.install_prefix.resolve() if args.install_prefix else None,
            args.source_archive.resolve() if args.source_archive else None,
            args.timeout,
        )
    else:
        consumers = [
            {**row, **not_observed("consumer execution was not requested"), "steps": []}
            for row in consumer_rows()
        ]
    unclassified = [
        item for item in source_declarations if item["category"] == "unclassified"
    ]
    source_counts = {
        "declaration_sites": len(source_declarations),
        "generator_visible": sum(
            item["source_path"]
            in ("src/blas/abi/cblas.zig", "src/blas/abi/fortran.zig")
            for item in source_declarations
        ),
        "fortran_generator_visible": sum(
            item["source_path"] == "src/blas/abi/fortran.zig"
            for item in source_declarations
        ),
        "cblas_generator_visible": sum(
            item["source_path"] == "src/blas/abi/cblas.zig"
            for item in source_declarations
        ),
        "default_architecture_extension": sum(
            item["visibility"] == "default"
            and item["category"] == "architecture_extension"
            for item in source_declarations
        ),
        "explicit_hidden_sites": sum(
            item["visibility"] == "hidden" for item in source_declarations
        ),
        "unclassified": len(unclassified),
    }
    observer_path = root / OBSERVER_ID
    git = git_observation(root)
    inputs = sorted(
        projection_inputs
        + [
            {
                "path": relative_display(policy_path, root),
                "sha256": sha256_bytes(policy_data),
            },
            {"path": OBSERVER_ID, "sha256": sha256_file(observer_path)},
            *(
                {"path": path, "sha256": sha256_file(root / path)}
                for path in CONSUMER_SOURCE_PATHS + PACKAGE_REPRODUCTION_PATHS
            ),
        ],
        key=lambda item: item["path"],
    )
    source_commitment = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted((root / "src").rglob("*.zig"))
    ]
    consumer_input_commitment = []
    if args.source_archive and args.source_archive.is_file():
        consumer_input_commitment.append(
            {
                "path": "source-archive/" + args.source_archive.name,
                "sha256": sha256_file(args.source_archive),
            }
        )
    if args.install_prefix and args.install_prefix.is_dir():
        for path in sorted(
            item for item in args.install_prefix.rglob("*") if item.is_file()
        ):
            consumer_input_commitment.append(
                {
                    "path": "install-prefix/"
                    + path.relative_to(args.install_prefix).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    commitment_subject = {
        "git": git,
        "inputs": inputs,
        "source_files": source_commitment,
        "targets": sorted(set(args.target)),
        "artifact_digests": sorted(
            item["sha256"]
            for item in artifacts.values()
            if item.get("status") == "observed"
        ),
        "consumer_inputs": consumer_input_commitment,
    }
    invariant_results = {
        "policy_role_is_observer": policy["role"] == "observer_not_abi_authority",
        "artifact_target_matches_build_configuration": (
            requested_targets == [configured_target] if artifacts_requested else True
        ),
        "source_only_build_configuration_declared": (
            artifacts_requested
            or artifact_build_configuration == ARTIFACT_BUILD_CONFIGURATION
        ),
        "unknown_is_not_pass": len(unclassified) == 0,
        "artifact_unknown_is_not_pass": all(
            not isinstance(item.get("unclassified_symbols"), list)
            or not item["unclassified_symbols"]
            for item in artifacts.values()
        ),
        "artifact_observations_complete": (
            all(item.get("status") == "observed" for item in artifacts.values())
            if artifacts_requested
            else all(
                item.get("status") == "not_observed" for item in artifacts.values()
            )
        ),
        "artifact_public_surface_complete": (
            all(
                item.get("source_missing_symbols") == []
                and item.get("unclassified_symbols") == []
                for item in artifacts.values()
            )
            if artifacts_requested
            else True
        ),
        "platform_binding_complete": (
            sum(item["status"] == "observed" for item in platform_rows) == 1
            if artifacts_requested
            else all(item["status"] == "not_observed" for item in platform_rows)
        ),
        "consumer_matrix_complete": (
            all(item["status"] in ("pass", "fail") for item in consumers)
            if args.run_consumers
            else all(item["status"] == "not_observed" for item in consumers)
        ),
        "git_observation_complete": all(
            _field_observed(git[field])
            for field in ("head", "index_sha256", "status_sha256")
        ),
        "projection_unknown_is_not_pass": not projection["unclassified"],
        "projection_cross_map_complete": not projection["missing"]
        and not projection["source_extra"]
        and not projection["extra"]["c"]
        and not projection["extra"]["fortran"],
        "not_observed_has_reason": _not_observed_values_have_reasons(
            [artifacts, platform_rows, consumers, projection, git]
        ),
        "source_site_records_are_not_deduplicated": len(source_declarations)
        == source_counts["declaration_sites"],
    }
    summary = {
        "sources": source_counts,
        "projections": projection_counts,
        "artifacts_observed": sum(
            item.get("status") == "observed" for item in artifacts.values()
        ),
        "platforms_observed": sum(
            item["status"] == "observed" for item in platform_rows
        ),
        "consumer_statuses": {
            status: sum(item["status"] == status for item in consumers)
            for status in ("pass", "fail", "not_observed")
        },
        "invariants_pass": all(invariant_results.values()),
    }
    observation = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "observer": {
            "identity": OBSERVER_ID,
            "sha256": sha256_file(observer_path),
            "role": policy["role"],
        },
        "subject": {
            "commitment_sha256": sha256_bytes(canonical_json_bytes(commitment_subject)),
            "inputs": commitment_subject,
        },
        "artifact_build_configuration": artifact_build_configuration,
        "accepted_public_zig_digest_reference": args.public_zig_contract_digest
        if args.public_zig_contract_digest
        else not_observed("digest reference was not supplied"),
        "installed_zig_module_abi": {"exists": False, "policy": "not_introduced"},
        "sources": {
            "declarations": source_declarations,
            "counts": source_counts,
            "unclassified": unclassified,
        },
        "projections": projection,
        "artifacts": artifacts,
        "platforms": platform_rows,
        "consumers": consumers,
        "summary": summary,
        "invariants": invariant_results,
    }
    failed_invariants = sorted(
        name for name, passed in invariant_results.items() if not passed
    )
    if failed_invariants:
        raise ObservationError(
            "observation invariants failed: " + ", ".join(failed_invariants)
        )
    return observation


def validate_observation(value: Any) -> None:
    if not isinstance(value, dict) or value.get("schema") != {
        "name": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
    }:
        raise ObservationError("invalid observation schema")
    if value.get("observer", {}).get("role") != "observer_not_abi_authority":
        raise ObservationError("invalid observer role")
    if value.get("artifact_build_configuration") != ARTIFACT_BUILD_CONFIGURATION:
        raise ObservationError("invalid artifact build configuration")

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("status") == "not_observed" and not item.get("reason"):
                raise ObservationError("not_observed value has no reason")
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamic-library", type=Path)
    parser.add_argument("--static-library", type=Path)
    parser.add_argument("--install-prefix", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--public-zig-contract-digest")
    parser.add_argument("--run-consumers", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.timeout > 600:
        parser.error("--timeout must be greater than zero and at most 600 seconds")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args[:1] == ["--internal-windows-supervisor"]:
        if os.name != "nt" or len(raw_args) < 2:
            return 1
        try:
            if sys.stdin.buffer.read(1) != b"1":
                return 1
            child = subprocess.Popen(raw_args[1:], stdin=subprocess.DEVNULL)
            return child.wait()
        except OSError:
            return 1
    if raw_args[:1] == ["--internal-extract-archive"]:
        if len(raw_args) != 8:
            print(
                "source archive extraction failed: invalid internal arguments",
                file=sys.stderr,
            )
            return 1
        try:
            _extract_tar_contents(
                Path(raw_args[1]),
                Path(raw_args[2]),
                float(raw_args[3]),
                int(raw_args[4]),
                int(raw_args[5]),
                int(raw_args[6]),
                int(raw_args[7]),
            )
            return 0
        except (ObservationError, OSError, ValueError, tarfile.TarError) as exc:
            print(f"source archive extraction failed: {exc}", file=sys.stderr)
            return 1
    try:
        args = parse_args(raw_args)
        observation = build_observation(args)
        validate_observation(observation)
        output = args.output.resolve()
        if not output.parent.is_dir():
            raise ObservationError("output parent directory does not exist")
        output.write_bytes(canonical_json_bytes(observation))
        return 0 if observation["summary"]["invariants_pass"] else 2
    except (ObservationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"observation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
