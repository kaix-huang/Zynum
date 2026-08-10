# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools/observe_abi_baseline.py"
SPEC = importlib.util.spec_from_file_location("observe_abi_baseline", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load observer")
observer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observer
SPEC.loader.exec_module(observer)
FIXTURES = Path(__file__).with_name("fixtures")


def arguments(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "root": ROOT,
        "policy": ROOT / "tools/abi_baseline_observation.json",
        "output": Path("unused.json"),
        "dynamic_library": None,
        "static_library": None,
        "install_prefix": None,
        "source_archive": None,
        "target": [],
        "public_zig_contract_digest": None,
        "run_consumers": False,
        "timeout": 1.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def policy() -> dict[str, object]:
    return json.loads(
        (ROOT / "tools/abi_baseline_observation.json").read_text(encoding="utf-8")
    )


class PolicyTests(unittest.TestCase):
    def test_artifact_build_configuration_is_exact(self) -> None:
        value = policy()
        observer.validate_policy(value)
        self.assertEqual(
            observer.ARTIFACT_BUILD_CONFIGURATION, value["artifact_build_configuration"]
        )

    def test_missing_or_wrong_artifact_build_configuration_is_rejected(self) -> None:
        cases = {
            "missing": None,
            "wrong": {
                **observer.ARTIFACT_BUILD_CONFIGURATION,
                "configuration_id": "not-canonical",
            },
        }
        for label, replacement in cases.items():
            with self.subTest(label=label):
                value = policy()
                if replacement is None:
                    value.pop("artifact_build_configuration")
                else:
                    value["artifact_build_configuration"] = replacement
                with self.assertRaisesRegex(
                    observer.ObservationError, "artifact build configuration"
                ):
                    observer.validate_policy(value)

    def test_canonical_command_is_explicit_releasefast_and_path_free(self) -> None:
        command = observer.ARTIFACT_BUILD_CONFIGURATION["command_template"]
        self.assertEqual(
            [
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
            command,
        )
        self.assertFalse(any(Path(argument).is_absolute() for argument in command))
        self.assertNotIn(
            "ReleaseSafe", json.dumps(observer.ARTIFACT_BUILD_CONFIGURATION)
        )

    def test_resolved_configuration_and_build_input_identities_are_exact(self) -> None:
        configuration = observer.ARTIFACT_BUILD_CONFIGURATION
        self.assertEqual(
            {
                "optimize": "ReleaseFast",
                "cpu": "apple_m1",
                "cpu_resolution": "Zig 0.16.0 explicit-target default",
                "strip_debug_info": False,
                "minimum_platform": "13.0",
                "artifact_sdk": "26.4",
                "zig_version": "0.16.0",
            },
            configuration["resolved"],
        )
        self.assertEqual(
            {
                "dynamic_library": "87a6d1418ffc4acad526edf9f21836a937a0456a868ed521cee9c264d5f30c78",
                "static_library": "9c61cd4da525518fd5660ac7b84c3750781ad2ef5ad2e93d364b04421bcf9498",
            },
            configuration["resolved_build_inputs"]["generated_builtin_zig_sha256"],
        )
        encoded = json.dumps(configuration, sort_keys=True)
        self.assertNotIn(
            "e6d6e63d5498623a821935479418bcb52541b36a66f2c87362d5bae9b6f4ae09", encoded
        )

    def test_installed_artifact_provenance_binding_is_exact(self) -> None:
        build_inputs = observer.ARTIFACT_BUILD_CONFIGURATION["resolved_build_inputs"]
        self.assertTrue(build_inputs["matched_across_two_independent_cache_roots"])
        self.assertEqual(
            {
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
            build_inputs["installed_artifact_provenance_binding"],
        )
        self.assertNotIn("installed_artifact_inference_or_verification", build_inputs)
        self.assertEqual(
            {
                "independent_parser": "tools/abi_artifact_parity.py",
                "build_and_verify_cli": "tools/verify_abi_artifact_parity.py",
                "local_receipt_cross_check": True,
            },
            build_inputs["verification_boundary"],
        )

    def test_wrong_static_artifact_provenance_binding_is_rejected(self) -> None:
        value = policy()
        build_inputs = value["artifact_build_configuration"]["resolved_build_inputs"]
        binding = build_inputs["installed_artifact_provenance_binding"][
            "static_library"
        ]
        binding["binding_chain"][1]["via"] = "cache_presence"
        with self.assertRaisesRegex(
            observer.ObservationError, "artifact build configuration"
        ):
            observer.validate_policy(value)

    def test_raw_hash_claim_and_volatility_classes_are_exact(self) -> None:
        raw = observer.ARTIFACT_BUILD_CONFIGURATION["raw_artifacts"]
        self.assertTrue(raw["sha256_retained"])
        self.assertFalse(raw["cross_cache_raw_byte_equality_claimed"])
        self.assertEqual(
            [
                "Mach-O LC_UUID",
                "N_OSO cache path/object mtime",
                "derived adhoc signature",
                "static-object DWARF global-cache path",
            ],
            raw["allowed_volatility_classes"],
        )

    def test_fresh_rebuild_artifact_and_archive_parity_fields_are_exact(self) -> None:
        parity = observer.ARTIFACT_BUILD_CONFIGURATION["fresh_rebuild_parity"]
        self.assertEqual(
            [
                "artifact_metadata",
                "archive_structure",
                "symbol_sets",
                "source_symbol_accounting",
                "generated_build_input_provenance",
            ],
            parity["required_parity_axes"],
        )
        self.assertEqual(
            [
                "format",
                "architecture",
                "platform",
                "minimum_platform",
                "sdk",
                "install_name",
                "dependencies",
                "rpaths",
            ],
            parity["required_artifact_fields"],
        )
        self.assertEqual(
            ["members", "index", "normalized_metadata"],
            parity["required_archive_fields"],
        )

    def test_fresh_rebuild_symbol_and_source_accounting_are_exact(self) -> None:
        parity = observer.ARTIFACT_BUILD_CONFIGURATION["fresh_rebuild_parity"]
        self.assertEqual(
            [
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
                {"scope": "local", "exclude": "STABS", "fields": ["name", "type"]},
            ],
            parity["required_symbol_sets"],
        )
        self.assertEqual(
            ["public", "hidden"], parity["required_source_symbol_accounting"]
        )

    def test_observation_validation_rejects_wrong_build_configuration(self) -> None:
        value = {
            "schema": {
                "name": observer.SCHEMA_NAME,
                "version": observer.SCHEMA_VERSION,
            },
            "observer": {"role": "observer_not_abi_authority"},
            "artifact_build_configuration": {"configuration_id": "not-canonical"},
        }
        with self.assertRaisesRegex(
            observer.ObservationError, "artifact build configuration"
        ):
            observer.validate_observation(value)


class SourceTests(unittest.TestCase):
    def test_real_repository_counts_and_cross_map(self) -> None:
        declarations = observer.scan_zig_exports(ROOT / "src", ROOT)
        self.assertEqual(334, len(declarations))
        self.assertEqual(
            161,
            sum(
                item["source_path"] == "src/blas/abi/fortran.zig"
                for item in declarations
            ),
        )
        self.assertEqual(
            150,
            sum(
                item["source_path"] == "src/blas/abi/cblas.zig" for item in declarations
            ),
        )
        self.assertEqual(
            8,
            sum(
                item["category"] == "architecture_extension"
                and item["visibility"] == "default"
                for item in declarations
            ),
        )
        self.assertEqual(
            15, sum(item["visibility"] == "hidden" for item in declarations)
        )
        self.assertFalse(
            [item for item in declarations if item["category"] == "unclassified"]
        )

        projection, _, counts = observer.observe_projections(ROOT)
        self.assertEqual(161, counts["manifest_fortran"])
        self.assertEqual(150, counts["manifest_cblas"])
        self.assertEqual(311, counts["c_prototypes"])
        self.assertEqual(161, counts["fortran_procedures"])
        self.assertEqual([], projection["missing"])
        self.assertEqual({"c": [], "fortran": []}, projection["extra"])
        self.assertTrue(
            all(item["manifest_mapping"] for item in projection["cross_reference"])
        )

    def test_synthetic_unknown_default_hidden_and_duplicate_sites(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "src"
            source.mkdir()
            (source / "one.zig").write_text(
                textwrap.dedent("""
                pub export fn mystery() callconv(.c) void {}
                fn hiddenTarget() callconv(.c) void {}
                comptime {
                    @export(&hiddenTarget, .{
                        .name = "zynum_internal_duplicate",
                        .visibility = .hidden,
                    });
                }
            """),
                encoding="utf-8",
            )
            (source / "two.zig").write_text(
                textwrap.dedent("""
                fn otherTarget() callconv(.c) void {}
                comptime { @export(&otherTarget, .{ .name = "zynum_internal_duplicate", .visibility = .hidden }); }
            """),
                encoding="utf-8",
            )
            declarations = observer.scan_zig_exports(source, root)
            self.assertEqual(3, len(declarations))
            unknown = next(
                item for item in declarations if item["exported_name"] == "mystery"
            )
            self.assertEqual("default", unknown["visibility"])
            self.assertEqual("unclassified", unknown["category"])
            duplicates = [
                item
                for item in declarations
                if item["exported_name"] == "zynum_internal_duplicate"
            ]
            self.assertEqual(2, len(duplicates))
            self.assertTrue(all(item["visibility"] == "hidden" for item in duplicates))

    def test_projection_preserves_type_and_fortran_details(self) -> None:
        projection, _, _ = observer.observe_projections(ROOT)
        cblas = projection["c_headers"]["cblas"]
        self.assertTrue(
            any(
                item["name"] == "zynum_blas_complex_float" and len(item["fields"]) == 2
                for item in cblas["structs"]
            )
        )
        self.assertTrue(
            any(
                item["alias"] == "CBLAS_TRANSPOSE" and len(item["constants"]) == 3
                for item in cblas["enums"]
            )
        )
        procedure = next(
            item
            for item in projection["fortran_module"]["procedures"]
            if item["bind_name"] == "dgemm_"
        )
        self.assertEqual("subroutine", procedure["procedure_kind"])
        self.assertTrue(procedure["imports"])
        self.assertTrue(
            all(item["declaration"] != "not_observed" for item in procedure["params"])
        )
        self.assertTrue(any(item["array"] == "(*)" for item in procedure["params"]))


class MetadataTests(unittest.TestCase):
    def test_architecture_aliases_are_canonical_and_universal_is_preserved(
        self,
    ) -> None:
        self.assertEqual(["arm64"], observer._canonical_architectures("aarch64 arm64"))
        self.assertEqual(["x86_64"], observer._canonical_architectures("x86-64 x86_64"))
        self.assertEqual(
            ["arm64", "x86_64"],
            observer._canonical_architectures("Mach-O universal x86_64 and arm64"),
        )

    def test_object_magic_binds_archive_members_to_platform_formats(self) -> None:
        self.assertEqual(
            ("ELF", "x86_64"),
            observer._detect_object_format(b"\x7fELF\x02\x01" + b"\0" * 12 + b"\x3e\0"),
        )
        self.assertEqual(
            ("ELF", "arm64"),
            observer._detect_object_format(b"\x7fELF\x02\x01" + b"\0" * 12 + b"\xb7\0"),
        )
        self.assertEqual(
            ("Mach-O", "arm64"),
            observer._detect_object_format(b"\xcf\xfa\xed\xfe\x0c\0\0\x01"),
        )
        self.assertEqual(
            ("PE/COFF", "x86_64"), observer._detect_object_format(b"\x64\x86")
        )
        self.assertEqual(
            ("unknown", "not_observed"),
            observer._detect_object_format(b"not-an-object"),
        )

    def test_nm_symbol_fields_are_explicit_or_fail_closed(self) -> None:
        parsed = observer.parse_nm_symbols(
            "00000000 (__TEXT,__text) external _function\n"
            "         (undefined) external _dependency\n"
            "00000010 D _gnu_data\n"
        )
        by_name = {item["name"]: item for item in parsed}
        self.assertEqual("section:__TEXT,__text", by_name["_function"]["type"])
        self.assertEqual("undefined", by_name["_dependency"]["type"])
        self.assertFalse(by_name["_dependency"]["defined"])
        self.assertEqual("D", by_name["_gnu_data"]["type"])
        self.assertTrue(
            all(item["visibility"] in ("default", "hidden", "local") for item in parsed)
        )

    def test_binary_header_detection_is_independent_of_file_description(self) -> None:
        cases = {
            "elf": (
                b"\x7fELF\x02\x01" + b"\0" * 12 + b"\x3e\0",
                ("ELF", "x86_64"),
            ),
            "macho": (b"\xcf\xfa\xed\xfe\x0c\0\0\x01", ("Mach-O", "arm64")),
        }
        for label, (contents, expected) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as name:
                artifact = Path(name) / "library"
                artifact.write_bytes(contents)
                self.assertEqual(expected, observer._detect_binary_header(artifact))

        for magic in observer.MACHO_FAT_MAGICS:
            with (
                self.subTest(fat_magic=magic.hex()),
                tempfile.TemporaryDirectory() as name,
            ):
                artifact = Path(name) / "library"
                artifact.write_bytes(magic + b"\0" * 64)
                self.assertEqual(
                    ("Mach-O", "not_observed"), observer._detect_format(artifact, "")
                )
                self.assertEqual(
                    ("unknown", "not_observed"),
                    observer._detect_binary_header(artifact),
                )
                self.assertTrue(observer._is_fat_or_universal_macho(artifact, ""))

        with tempfile.TemporaryDirectory() as name:
            artifact = Path(name) / "library.dll"
            contents = bytearray(70)
            contents[:2] = b"MZ"
            contents[60:64] = (64).to_bytes(4, "little")
            contents[64:68] = b"PE\0\0"
            contents[68:70] = (0x8664).to_bytes(2, "little")
            artifact.write_bytes(contents)
            self.assertEqual(
                ("PE/COFF", "x86_64"), observer._detect_binary_header(artifact)
            )

    def test_elf_fixture(self) -> None:
        parsed = observer.parse_elf_metadata(
            (FIXTURES / "elf_readelf.txt").read_text(encoding="utf-8")
        )
        self.assertEqual("libzynum_blas.so.1", parsed["soname"])
        self.assertEqual(["libc.so.6"], parsed["needed"])
        self.assertEqual(["$ORIGIN"], parsed["runpath"])
        self.assertIn("ZYNUM_BLAS_1.0", parsed["symbol_versions"])

    def test_macho_fixture(self) -> None:
        parsed = observer.parse_macho_metadata(
            (FIXTURES / "macho_load_commands.txt").read_text(encoding="utf-8"),
            (FIXTURES / "macho_libraries.txt").read_text(encoding="utf-8"),
        )
        self.assertEqual("@rpath/libzynum_blas.1.dylib", parsed["id"])
        self.assertEqual("1.2.0", parsed["current_version"])
        self.assertEqual("1.0.0", parsed["compatibility_version"])
        self.assertEqual("macOS", parsed["platform"])
        self.assertEqual("13.0", parsed["minimum_platform"])
        self.assertEqual("15.0", parsed["sdk"])
        self.assertEqual(["@loader_path"], parsed["rpath"])
        self.assertEqual(1, len(parsed["dependencies"]))
        self.assertEqual(
            "/usr/lib/libSystem.B.dylib", parsed["dependencies"][0]["name"]
        )

    def test_pe_fixture(self) -> None:
        parsed = observer.parse_pe_metadata(
            (FIXTURES / "pe_objdump.txt").read_text(encoding="utf-8")
        )
        self.assertEqual("zynum_blas.dll", parsed["dll_name"])
        self.assertEqual(["KERNEL32.dll", "ucrtbase.dll"], parsed["imports"])
        self.assertEqual(["ucrtbase.dll"], parsed["crt"])
        self.assertEqual("decorated", parsed["exports"][1]["decoration"])
        self.assertEqual("not_observed", parsed["import_library"]["status"])


class BoundaryTests(unittest.TestCase):
    def test_windows_supervisor_preserves_argv_behind_internal_gate(self) -> None:
        wrapped = observer._windows_supervisor_argv(("tool", "argument with spaces"))
        self.assertEqual(sys.executable, wrapped[0])
        self.assertEqual("--internal-windows-supervisor", wrapped[2])
        self.assertEqual(("tool", "argument with spaces"), wrapped[3:])

    def test_windows_supervisor_is_bound_before_gate_release(self) -> None:
        events: list[str] = []

        class Gate(io.BytesIO):
            def write(self, data: bytes) -> int:
                self.assert_bound()
                events.append("release")
                return super().write(data)

            def flush(self) -> None:
                events.append("flush")
                super().flush()

            def close(self) -> None:
                events.append("close")
                super().close()

            @staticmethod
            def assert_bound() -> None:
                if events != ["bind"]:
                    raise AssertionError(
                        "supervisor gate was released before Job binding"
                    )

        class Process:
            stdin = Gate()

        class Job:
            def __init__(self, _: object) -> None:
                events.append("bind")

            def terminate(self) -> bool:
                return True

            def close(self) -> bool:
                return True

        with mock.patch.object(observer, "_WindowsJob", Job):
            observer._bind_and_release_windows_supervisor(Process())  # type: ignore[arg-type]
        self.assertEqual(["bind", "release", "flush", "close"], events)

    def test_missing_artifact_and_tool_are_observations(self) -> None:
        missing = observer.observe_artifact(
            Path("does-not-exist"), "dynamic", ROOT, set()
        )
        self.assertEqual("not_observed", missing["status"])
        command = observer.run_command(
            ("command-that-cannot-exist-abi-observer",), cwd=ROOT
        )
        self.assertEqual("tool_missing", command.status)

    def test_timeout_and_output_bound(self) -> None:
        timeout = observer.run_command(
            (sys.executable, "-c", "import time; time.sleep(1)"), cwd=ROOT, timeout=0.01
        )
        self.assertEqual("timeout", timeout.status)
        bounded = observer.run_command(
            (sys.executable, "-c", "print('x' * 10000)"), cwd=ROOT, max_output=64
        )
        self.assertEqual(64, len(bounded.stdout))
        self.assertTrue(bounded.stdout_truncated)

    @unittest.skipUnless(os.name == "posix", "POSIX process-group behavior")
    def test_parent_exit_still_cleans_descendant_before_return(self) -> None:
        script = (
            "import subprocess,sys;"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)']);"
            "raise SystemExit(7)"
        )
        started = time.monotonic()
        result = observer.run_command(
            (sys.executable, "-c", script), cwd=ROOT, timeout=0.2
        )
        elapsed = time.monotonic() - started
        self.assertEqual("exited", result.status)
        self.assertEqual(7, result.exit_code)
        self.assertLess(elapsed, 0.5)

    def test_candidate_runtime_refuses_platform_without_strict_containment(
        self,
    ) -> None:
        with (
            mock.patch.object(observer.os, "name", "posix"),
            mock.patch.object(observer.sys, "platform", "unsupported"),
            mock.patch.object(observer, "run_command") as run,
        ):
            result = observer.run_candidate_executable(
                ("candidate",), cwd=ROOT, timeout=1.0
            )
        self.assertEqual("containment_unavailable", result.status)
        run.assert_not_called()

    @unittest.skipUnless(sys.platform == "darwin", "Darwin runtime sandbox integration")
    def test_darwin_candidate_runtime_denies_escape_and_host_access(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            marker = root / "marker"
            source = root / "sandbox_probe.c"
            helper = root / "sandbox_probe"
            source.write_text(
                textwrap.dedent("""
                #include <arpa/inet.h>
                #include <errno.h>
                #include <fcntl.h>
                #include <netinet/in.h>
                #include <sys/socket.h>
                #include <sys/types.h>
                #include <unistd.h>

                static int denied(void) {
                    return errno == EPERM || errno == EACCES;
                }

                int main(void) {
                    errno = 0;
                    pid_t child = fork();
                    if (child >= 0 || errno != EPERM) return 10;

                    errno = 0;
                    int output = open("marker", O_WRONLY | O_CREAT, 0600);
                    if (output >= 0 || !denied()) return 11;

                    errno = 0;
                    int input = open("/etc/hosts", O_RDONLY);
                    if (input >= 0 || !denied()) return 12;

                    errno = 0;
                    int network = socket(AF_INET, SOCK_STREAM, 0);
                    if (network < 0) return denied() ? 0 : 13;
                    struct sockaddr_in address = {0};
                    address.sin_family = AF_INET;
                    address.sin_port = htons(9);
                    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
                    errno = 0;
                    int connected = connect(network, (struct sockaddr *)&address, sizeof(address));
                    close(network);
                    return connected < 0 && denied() ? 0 : 14;
                }
            """),
                encoding="utf-8",
            )
            compile_result = observer.run_command(
                ("cc", str(source), "-o", str(helper)),
                cwd=root,
                timeout=10.0,
            )
            self.assertTrue(
                observer.command_observed(compile_result),
                compile_result.observation(ROOT, (root,)),
            )
            result = observer.run_candidate_executable(
                (str(helper),),
                cwd=root,
                timeout=2.0,
            )
            self.assertTrue(
                observer.command_observed(result), result.observation(ROOT, (root,))
            )
            self.assertFalse(marker.exists())

    @unittest.skipUnless(sys.platform == "darwin", "Darwin runtime sandbox integration")
    def test_darwin_candidate_runtime_denies_non_allowlisted_file_data(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = observer.run_candidate_executable(
                ("/bin/cat", "/etc/hosts"),
                cwd=root,
                timeout=2.0,
            )
        self.assertTrue(observer.command_failed(result))
        self.assertEqual(b"", result.stdout)
        self.assertNotEqual(0, result.exit_code)

    def test_volatile_addresses_are_normalized_before_hashing(self) -> None:
        first = observer.CommandResult(
            ("tool",), "exited", 0, None, b"at 0x1234 in call\n", b"", False, False
        )
        second = observer.CommandResult(
            ("tool",), "exited", 0, None, b"at 0xABCDEF in call\n", b"", False, False
        )
        self.assertEqual(
            first.observation(ROOT)["stdout"], second.observation(ROOT)["stdout"]
        )

    def test_composite_arguments_normalize_transient_paths(self) -> None:
        transient = Path(tempfile.gettempdir()) / "observer-prefix"
        argument = f"-Wl,-rpath,{transient}/lib"
        self.assertEqual(
            "-Wl,-rpath,<temporary>/lib",
            observer.normalized_arg(argument, ROOT, (transient,)),
        )

    @unittest.skipUnless(
        sys.platform == "darwin", "Darwin sandbox profile normalization"
    )
    def test_darwin_sandbox_profile_is_transient_root_deterministic(self) -> None:
        transient_roots = (
            Path("/var/folders/zz/example/T/zynum-abi-observe-fixed"),
            Path("/tmp/zynum-abi-observe-fixed"),
        )
        observed_argv = []
        for transient in transient_roots:
            with self.subTest(transient=transient):
                executable = transient / "consumer-C-shared"
                runtime_root = transient / "runtime"
                profile = observer._darwin_runtime_sandbox_profile(
                    transient,
                    (str(executable),),
                    (runtime_root,),
                    observation_root=ROOT,
                    observation_transient_roots=(transient,),
                )
                for path in {
                    observer._sandbox_path(transient),
                    observer._sandbox_path(transient, resolve=False),
                    observer._sandbox_path(runtime_root),
                    observer._sandbox_path(runtime_root, resolve=False),
                }:
                    self.assertIn(f'(subpath "{path}")', profile)
                for path in {
                    observer._sandbox_path(executable),
                    observer._sandbox_path(executable, resolve=False),
                }:
                    self.assertIn(f'(literal "{path}")', profile)
                self.assertNotIn("<temporary>", profile)
                result = observer.CommandResult(
                    ("sandbox-exec", "-p", profile, str(executable)),
                    "exited",
                    0,
                    None,
                    b"",
                    b"",
                    False,
                    False,
                )
                observed_argv.append(result.observation(ROOT, (transient,))["argv"])
        self.assertEqual(observed_argv[0], observed_argv[1])
        self.assertEqual(4, observed_argv[0][2].count('(subpath "<temporary>'))

    def test_archive_observer_only_uses_read_only_queries(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            archive = Path(name) / "libsample.a"
            archive.write_bytes(b"!<arch>\n")
            seen: list[tuple[str, ...]] = []

            def fake(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                seen.append(command)
                stdout = b"object.o\n" if command[:2] == ("ar", "-t") else b""
                return observer.CommandResult(
                    command, "exited", 0, None, stdout, b"", False, False
                )

            with mock.patch.object(observer, "run_command", side_effect=fake):
                result = observer.observe_artifact(archive, "static", ROOT, set())
            self.assertEqual(["object.o"], result["archive"]["members"])
            self.assertIn(("ar", "-t", str(archive)), seen)
            self.assertFalse(
                any(command and command[0] == "ranlib" for command in seen)
            )

    def test_artifact_visibility_matching_and_wrong_kind_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            artifact = Path(name) / "wrong.a"
            artifact.write_bytes(b"\xcf\xfa\xed\xfe")

            def fake(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                if command[:4] == ("nm", "-g", "-m", str(artifact)):
                    stdout = (
                        b"00000000 (__TEXT,__text) external _public_api\n"
                        b"00000010 (__TEXT,__text) private external _hidden_bridge\n"
                        b"00000020 (__DATA,__data) external ___dso_handle\n"
                    )
                elif command[:2] == ("nm", "-m"):
                    stdout = (
                        b"00000000 (__TEXT,__text) external _public_api\n"
                        b"00000010 (__TEXT,__text) private external _hidden_bridge\n"
                        b"00000020 (__DATA,__data) external ___dso_handle\n"
                    )
                else:
                    stdout = b""
                return observer.CommandResult(
                    command, "exited", 0, None, stdout, b"", False, False
                )

            with mock.patch.object(observer, "run_command", side_effect=fake):
                result = observer.observe_artifact(
                    artifact,
                    "static",
                    ROOT,
                    {"public_api"},
                    {"hidden_bridge"},
                )
            self.assertEqual("failed", result["status"])
            self.assertIn("unexpected Mach-O format", result["reason"])
            self.assertEqual(["public_api"], result["source_matched_symbols"])
            self.assertEqual(["hidden_bridge"], result["hidden_source_matched_symbols"])
            self.assertEqual(["___dso_handle"], result["toolchain_symbols"])
            self.assertEqual([], result["unclassified_symbols"])

    def test_failed_consumers_never_become_pass(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            prefix = Path(name) / "prefix"
            (prefix / "include").mkdir(parents=True)
            (prefix / "lib").mkdir()
            (prefix / "lib/libzynum_blas.a").write_bytes(b"archive")
            failed = observer.CommandResult(
                ("cc",), "exited", 1, None, b"", b"error: failed", False, False
            )
            with mock.patch.object(observer, "run_command", return_value=failed):
                results = observer.run_consumers(ROOT, prefix, None, 1.0)
            self.assertEqual("not_observed", results[0]["status"])
            self.assertTrue(all(item["status"] == "fail" for item in results[1:]))
            self.assertFalse(any(item["status"] == "pass" for item in results))

    def test_installed_consumers_forward_observation_roots_to_runtime_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            prefix = Path(name) / "prefix"
            (prefix / "include").mkdir(parents=True)
            (prefix / "lib").mkdir()
            (prefix / "lib/libzynum_blas.a").write_bytes(b"archive")

            def fake(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                return observer.CommandResult(
                    command, "exited", 0, None, b"", b"", False, False
                )

            executed = observer.CommandResult(
                ("candidate",), "exited", 0, None, b"", b"", False, False
            )
            with (
                mock.patch.object(observer, "run_command", side_effect=fake),
                mock.patch.object(
                    observer, "run_candidate_executable", return_value=executed
                ) as run_candidate,
            ):
                results = observer.run_consumers(ROOT, prefix, None, 1.0)
            self.assertTrue(all(item["status"] == "pass" for item in results[1:]))
            self.assertEqual(6, run_candidate.call_count)
            for call in run_candidate.call_args_list:
                self.assertEqual(ROOT, call.kwargs["observation_root"])
                self.assertIn(
                    prefix,
                    call.kwargs["observation_transient_roots"],
                )

    def test_truncated_successful_consumer_steps_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            prefix = root / "prefix"
            (prefix / "include").mkdir(parents=True)
            (prefix / "lib").mkdir()
            (prefix / "lib/libzynum_blas.a").write_bytes(b"archive")
            archive = root / "source.tar"
            with tarfile.open(archive, "w"):
                pass

            def truncated(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                return observer.CommandResult(
                    command, "exited", 0, None, b"output", b"", True, False
                )

            with mock.patch.object(observer, "run_command", side_effect=truncated):
                results = observer.run_consumers(ROOT, prefix, archive, 1.0)
            self.assertTrue(all(item["status"] == "fail" for item in results))
            self.assertFalse(any(item["status"] == "pass" for item in results))

    def test_clean_source_row_runs_the_zig_consumer_example(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            archive = Path(name) / "source.tar"
            with tarfile.open(archive, "w"):
                pass
            seen: list[tuple[str, ...]] = []

            def fake(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                seen.append(command)
                return observer.CommandResult(
                    command, "exited", 0, None, b"", b"", False, False
                )

            executed = observer.CommandResult(
                ("candidate",), "exited", 0, None, b"", b"", False, False
            )
            with (
                mock.patch.object(observer, "run_command", side_effect=fake),
                mock.patch.object(
                    observer, "run_candidate_executable", return_value=executed
                ) as run_candidate,
            ):
                results = observer.run_consumers(ROOT, None, archive, 1.0)
            self.assertEqual("pass", results[0]["status"])
            command = next(
                command for command in seen if command[:2] == ("zig", "build-exe")
            )
            self.assertIn("-Mroot=examples/zig/matrix_multiply.zig", command)
            self.assertIn("-Mzynum=src/zynum.zig", command)
            self.assertIn("--cache-dir", command)
            self.assertIn("--global-cache-dir", command)
            self.assertFalse(any("build.zig" in argument for argument in command))
            self.assertFalse(results[0]["candidate_build_scripts_executed"])
            run_candidate.assert_called_once()
            self.assertEqual(ROOT, run_candidate.call_args.kwargs["observation_root"])
            observation_transient_roots = run_candidate.call_args.kwargs[
                "observation_transient_roots"
            ]
            self.assertEqual(Path.home(), observation_transient_roots[-1])
            self.assertTrue(
                observation_transient_roots[0].name.startswith("zynum-abi-observe-")
            )

    def test_every_consumer_row_records_runtime_assurance(self) -> None:
        with mock.patch.object(
            observer,
            "_runtime_assurance",
            return_value={"status": "not_observed", "reason": "test"},
        ):
            results = observer.run_consumers(ROOT, None, None, 1.0)
        self.assertEqual(7, len(results))
        self.assertTrue(
            all(
                item["runtime_assurance"]["status"] == "not_observed"
                for item in results
            )
        )
        self.assertTrue(
            all(item["candidate_build_scripts_executed"] is False for item in results)
        )

    def test_archive_extraction_rejects_file_count_size_and_sparse_budgets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            oversized = root / "oversized.tar"
            with tarfile.open(oversized, "w") as archive:
                payload = b"x" * 32
                member = tarfile.TarInfo("oversized")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            destination = root / "oversized-out"
            destination.mkdir()
            with mock.patch.object(observer, "MAX_ARCHIVE_FILE_BYTES", 16):
                with self.assertRaisesRegex(
                    observer.ObservationError, "per-file budget"
                ):
                    observer._safe_extract_tar(oversized, destination, 1.0)
            self.assertEqual([], list(destination.iterdir()))

            crowded = root / "crowded.tar"
            with tarfile.open(crowded, "w") as archive:
                for filename in ("one", "two"):
                    payload = filename.encode()
                    member = tarfile.TarInfo(filename)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
            destination = root / "crowded-out"
            destination.mkdir()
            with mock.patch.object(observer, "MAX_ARCHIVE_MEMBERS", 1):
                with self.assertRaisesRegex(
                    observer.ObservationError, "member-count budget"
                ):
                    observer._safe_extract_tar(crowded, destination, 1.0)
            self.assertEqual([], list(destination.iterdir()))

            sparse = tarfile.TarInfo("sparse")
            sparse.size = 1
            sparse.pax_headers = {"GNU.sparse.map": "0,1"}
            with self.assertRaisesRegex(observer.ObservationError, "sparse"):
                observer._validate_archive_member(sparse, root, set())

    def test_archive_extraction_rejects_total_ratio_and_time_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            expanded = root / "expanded.tar"
            with tarfile.open(expanded, "w") as archive:
                for filename in ("one", "two"):
                    payload = b"x" * 8
                    member = tarfile.TarInfo(filename)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
            destination = root / "expanded-out"
            destination.mkdir()
            with mock.patch.object(observer, "MAX_ARCHIVE_TOTAL_BYTES", 10):
                with self.assertRaisesRegex(
                    observer.ObservationError, "expanded-size budget"
                ):
                    observer._safe_extract_tar(expanded, destination, 1.0)
            self.assertEqual([], list(destination.iterdir()))

            compressed = root / "compressed.tar.gz"
            with tarfile.open(compressed, "w:gz") as archive:
                payload = b"x" * 4096
                member = tarfile.TarInfo("payload")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            destination = root / "compressed-out"
            destination.mkdir()
            with mock.patch.object(observer, "MAX_ARCHIVE_COMPRESSION_RATIO", 1):
                with self.assertRaisesRegex(
                    observer.ObservationError, "compression-ratio budget"
                ):
                    observer._safe_extract_tar(compressed, destination, 1.0)
            self.assertEqual([], list(destination.iterdir()))

            timeout_result = observer.CommandResult(
                (sys.executable,),
                "timeout",
                None,
                9,
                b"",
                b"",
                False,
                False,
                "timeout after 0.1 seconds",
            )
            destination = root / "timeout-out"
            destination.mkdir()
            with mock.patch.object(
                observer, "run_command", return_value=timeout_result
            ) as run:
                with self.assertRaisesRegex(observer.ObservationError, "timeout after"):
                    observer._safe_extract_tar(expanded, destination, 0.1)
            self.assertEqual(0.1, run.call_args.kwargs["timeout"])

            truncated_result = observer.CommandResult(
                (sys.executable,), "exited", 0, None, b"partial", b"", True, False
            )
            destination = root / "truncated-out"
            destination.mkdir()
            with mock.patch.object(
                observer, "run_command", return_value=truncated_result
            ):
                with self.assertRaisesRegex(
                    observer.ObservationError, "source archive extraction failed"
                ):
                    observer._safe_extract_tar(expanded, destination, 0.1)

    def test_platform_matrix_is_fail_closed(self) -> None:
        artifacts = {
            "dynamic": {
                "kind": "dynamic",
                "status": "observed",
                "format": "Mach-O",
                "architecture": "arm64",
                "metadata": {"platform": "macOS"},
            },
            "static": {
                "kind": "static",
                "status": "observed",
                "format": "archive",
                "architecture": "arm64",
                "archive": {
                    "object_formats": ["Mach-O"],
                    "object_architectures": ["arm64"],
                },
            },
        }
        rows = observer.observe_platforms(("aarch64-macos",), artifacts)
        self.assertEqual("observed", rows[0]["status"])
        self.assertTrue(all(item["status"] == "not_observed" for item in rows[1:]))
        mismatch = observer.observe_platforms(("x86_64-linux-gnu",), artifacts)
        self.assertEqual("failed", mismatch[1]["status"])
        artifacts["static"]["architecture"] = "x86_64"
        artifacts["static"]["archive"] = {
            "object_formats": ["Mach-O"],
            "object_architectures": ["x86_64"],
        }
        artifacts["dynamic"] = {
            "kind": "dynamic",
            "status": "observed",
            "format": "ELF",
            "architecture": "x86_64",
            "metadata": {"machine": "Advanced Micro Devices X86-64"},
        }
        wrong_static_format = observer.observe_platforms(
            ("x86_64-linux-gnu",), artifacts
        )
        self.assertEqual("failed", wrong_static_format[1]["status"])

    def test_artifacts_require_one_known_target(self) -> None:
        for targets in (
            ["unknown-target"],
            ["aarch64-macos", "x86_64-linux-gnu"],
            ["aarch64-macos"],
        ):
            with self.subTest(targets=targets):
                with self.assertRaises(observer.ObservationError):
                    observer.build_observation(arguments(target=targets))

    def test_artifact_target_must_match_canonical_build_configuration(self) -> None:
        with self.assertRaisesRegex(
            observer.ObservationError, "canonical artifact build configuration"
        ):
            observer.build_observation(
                arguments(
                    dynamic_library=Path("libzynum_blas.dylib"),
                    static_library=Path("libzynum_blas.a"),
                    target=["x86_64-linux-gnu"],
                )
            )

    def test_missing_symbol_or_metadata_query_marks_artifact_failed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            artifact = Path(name) / "library.dylib"
            artifact.write_bytes(b"\xcf\xfa\xed\xfe")

            def fake(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                if command[0] == "file":
                    return observer.CommandResult(
                        command,
                        "exited",
                        0,
                        None,
                        b"Mach-O 64-bit arm64",
                        b"",
                        False,
                        False,
                    )
                if command[0] == "nm":
                    return observer.CommandResult(
                        command,
                        "exited",
                        0,
                        None,
                        b"0000 (__TEXT,__text) external _public_api\n",
                        b"",
                        False,
                        False,
                    )
                return observer.CommandResult(
                    command, "exited", 1, None, b"", b"query failed", False, False
                )

            with mock.patch.object(observer, "run_command", side_effect=fake):
                result = observer.observe_artifact(
                    artifact, "dynamic", ROOT, {"public_api"}
                )
            self.assertEqual("failed", result["status"])
            self.assertIn("format metadata query failed", result["issues"])

    def test_universal_dynamic_artifact_fails_closed(self) -> None:
        descriptions = (
            (
                b"Mach-O universal binary with 2 architectures: [x86_64] [arm64]",
                ["arm64", "x86_64"],
            ),
            (b"Mach-O universal binary with 2 architectures: [arm64] [ppc64]", "arm64"),
        )
        for description, expected_architecture in descriptions:
            with (
                self.subTest(description=description),
                tempfile.TemporaryDirectory() as name,
            ):
                artifact = Path(name) / "library.dylib"
                artifact.write_bytes(b"\xca\xfe\xba\xbe")

                def fake(argv: object, **_: object) -> object:
                    command = tuple(argv)  # type: ignore[arg-type]
                    if command[0] == "file":
                        stdout = description
                    elif command[:2] == ("otool", "-l"):
                        stdout = (FIXTURES / "macho_load_commands.txt").read_bytes()
                    elif command[:2] == ("otool", "-L"):
                        stdout = (FIXTURES / "macho_libraries.txt").read_bytes()
                    else:
                        stdout = b"0000 (__TEXT,__text) external _public_api\n"
                    return observer.CommandResult(
                        command, "exited", 0, None, stdout, b"", False, False
                    )

                with mock.patch.object(observer, "run_command", side_effect=fake):
                    result = observer.observe_artifact(
                        artifact, "dynamic", ROOT, {"public_api"}
                    )
                self.assertEqual("failed", result["status"])
                self.assertEqual(expected_architecture, result["architecture"])
                self.assertIn(
                    "fat or universal Mach-O artifact is not a single-target observation",
                    result["issues"],
                )
                self.assertFalse(
                    observer._artifact_matches_target(result, "aarch64-macos")
                )

    def test_dynamic_artifact_rejects_file_description_header_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            artifact = Path(name) / "library.dylib"
            artifact.write_bytes(b"\xcf\xfa\xed\xfe\x0c\0\0\x01")

            def fake(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                if command[0] == "file":
                    stdout = b"Mach-O 64-bit dynamically linked shared library x86_64"
                elif command[:2] == ("otool", "-l"):
                    stdout = (FIXTURES / "macho_load_commands.txt").read_bytes()
                elif command[:2] == ("otool", "-L"):
                    stdout = (FIXTURES / "macho_libraries.txt").read_bytes()
                else:
                    stdout = b"0000 (__TEXT,__text) external _public_api\n"
                return observer.CommandResult(
                    command, "exited", 0, None, stdout, b"", False, False
                )

            with mock.patch.object(observer, "run_command", side_effect=fake):
                result = observer.observe_artifact(
                    artifact, "dynamic", ROOT, {"public_api"}
                )
            self.assertEqual("failed", result["status"])
            self.assertIn(
                "binary header architecture did not match the file description",
                result["issues"],
            )
            self.assertFalse(observer._artifact_matches_target(result, "aarch64-macos"))

    def test_dynamic_artifact_rejects_unrecognized_symbol_fields(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            artifact = Path(name) / "library.dylib"
            artifact.write_bytes(b"\xcf\xfa\xed\xfe\x0c\0\0\x01")

            def fake(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                if command[0] == "file":
                    stdout = b"Mach-O 64-bit dynamically linked shared library arm64"
                elif command[:2] == ("otool", "-l"):
                    stdout = (FIXTURES / "macho_load_commands.txt").read_bytes()
                elif command[:2] == ("otool", "-L"):
                    stdout = (FIXTURES / "macho_libraries.txt").read_bytes()
                else:
                    stdout = b"0000 external _public_api\n"
                return observer.CommandResult(
                    command, "exited", 0, None, stdout, b"", False, False
                )

            with mock.patch.object(observer, "run_command", side_effect=fake):
                result = observer.observe_artifact(
                    artifact, "dynamic", ROOT, {"public_api"}
                )
            self.assertEqual("failed", result["status"])
            self.assertIn(
                "a symbol declaration field was not recognized", result["issues"]
            )
            self.assertFalse(observer._artifact_matches_target(result, "aarch64-macos"))

    def test_dynamic_artifact_rejects_multiple_described_architectures(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            artifact = Path(name) / "library.dylib"
            artifact.write_bytes(b"\xcf\xfa\xed\xfe\x0c\0\0\x01")

            def fake(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                if command[0] == "file":
                    stdout = b"Mach-O 64-bit arm64 x86_64"
                elif command[:2] == ("otool", "-l"):
                    stdout = (FIXTURES / "macho_load_commands.txt").read_bytes()
                elif command[:2] == ("otool", "-L"):
                    stdout = (FIXTURES / "macho_libraries.txt").read_bytes()
                else:
                    stdout = b"0000 (__TEXT,__text) external _public_api\n"
                return observer.CommandResult(
                    command, "exited", 0, None, stdout, b"", False, False
                )

            with mock.patch.object(observer, "run_command", side_effect=fake):
                result = observer.observe_artifact(
                    artifact, "dynamic", ROOT, {"public_api"}
                )
            self.assertEqual("failed", result["status"])
            self.assertIn(
                "file description reported multiple architectures", result["issues"]
            )
            self.assertFalse(observer._artifact_matches_target(result, "aarch64-macos"))

    def test_fat_archive_with_unknown_slice_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            artifact = Path(name) / "library.a"
            artifact.write_bytes(b"!<arch>\n")

            def fake(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                if command[0] == "file":
                    stdout = b"current ar archive"
                elif command[:2] == ("lipo", "-info"):
                    stdout = (
                        b"Architectures in the fat file: library.a are: arm64 ppc64\n"
                    )
                elif command[:2] == ("ar", "-t"):
                    stdout = b"__.SYMDEF\nobject.o\n"
                elif command[:2] == ("ar", "-p"):
                    stdout = b"\xcf\xfa\xed\xfe\x0c\0\0\x01"
                else:
                    stdout = b"0000 T public_api\n"
                return observer.CommandResult(
                    command, "exited", 0, None, stdout, b"", False, False
                )

            with mock.patch.object(observer, "run_command", side_effect=fake):
                result = observer.observe_artifact(
                    artifact, "static", ROOT, {"public_api"}
                )
            self.assertEqual("failed", result["status"])
            self.assertEqual("arm64", result["architecture"])
            self.assertIn(
                "fat or universal archive is not a single-target observation",
                result["issues"],
            )
            self.assertFalse(observer._artifact_matches_target(result, "aarch64-macos"))

    def test_static_archive_rejects_mixed_unknown_and_duplicate_members(self) -> None:
        def elf(machine: int) -> bytes:
            return b"\x7fELF\x02\x01" + b"\0" * 12 + machine.to_bytes(2, "little")

        cases = {
            "mixed": (b"x86.o\narm.o\n", {"x86.o": elf(62), "arm.o": elf(183)}),
            "unknown": (b"x86.o\nriscv.o\n", {"x86.o": elf(62), "riscv.o": elf(243)}),
            "duplicate": (b"same.o\nsame.o\n", {"same.o": elf(62)}),
        }
        for label, (listing, members) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as name:
                artifact = Path(name) / "library.a"
                artifact.write_bytes(b"!<arch>\n")

                def fake(argv: object, **_: object) -> object:
                    command = tuple(argv)  # type: ignore[arg-type]
                    if command[0] == "file":
                        return observer.CommandResult(
                            command,
                            "exited",
                            0,
                            None,
                            b"current ar archive",
                            b"",
                            False,
                            False,
                        )
                    if command[:2] == ("lipo", "-info"):
                        return observer.CommandResult(
                            command, "tool_missing", None, None, b"", b"", False, False
                        )
                    if command[:2] == ("ar", "-t"):
                        return observer.CommandResult(
                            command, "exited", 0, None, listing, b"", False, False
                        )
                    if command[:2] == ("ar", "-p"):
                        return observer.CommandResult(
                            command,
                            "exited",
                            0,
                            None,
                            members[command[-1]],
                            b"",
                            False,
                            False,
                        )
                    if command[:2] == ("nm", "--print-armap"):
                        return observer.CommandResult(
                            command,
                            "exited",
                            0,
                            None,
                            b"Archive index:\n",
                            b"",
                            False,
                            False,
                        )
                    stdout = b"0000 T public_api\n"
                    return observer.CommandResult(
                        command, "exited", 0, None, stdout, b"", False, False
                    )

                with mock.patch.object(observer, "run_command", side_effect=fake):
                    result = observer.observe_artifact(
                        artifact, "static", ROOT, {"public_api"}
                    )
                self.assertEqual("failed", result["status"])
                if label == "mixed":
                    self.assertIn(
                        "archive contains multiple object architectures",
                        result["issues"],
                    )
                elif label == "unknown":
                    self.assertIn(
                        "an archive object architecture was not recognized",
                        result["issues"],
                    )
                else:
                    self.assertIn(
                        "archive contains duplicate member names", result["issues"]
                    )

    def test_missing_nm_query_and_failed_artifact_propagate(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            artifact = Path(name) / "library.dylib"
            artifact.write_bytes(b"\xcf\xfa\xed\xfe")

            def fake(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                if command[0] == "file":
                    stdout = b"Mach-O 64-bit arm64"
                    return observer.CommandResult(
                        command, "exited", 0, None, stdout, b"", False, False
                    )
                if command[0] == "nm":
                    return observer.CommandResult(
                        command, "exited", 1, None, b"", b"query failed", False, False
                    )
                stdout = (
                    FIXTURES
                    / (
                        "macho_load_commands.txt"
                        if command[1] == "-l"
                        else "macho_libraries.txt"
                    )
                ).read_bytes()
                return observer.CommandResult(
                    command, "exited", 0, None, stdout, b"", False, False
                )

            with mock.patch.object(observer, "run_command", side_effect=fake):
                result = observer.observe_artifact(
                    artifact, "dynamic", ROOT, {"public_api"}
                )
            self.assertEqual("failed", result["status"])
            self.assertIn(
                "complete symbol declarations were not observed", result["issues"]
            )
            self.assertIn("public symbols were not observed", result["issues"])

            failed = {"kind": "dynamic", "status": "failed", "reason": "query failed"}
            missing = {"kind": "static", "status": "failed", "reason": "query failed"}
            with (
                mock.patch.object(
                    observer, "observe_artifact", side_effect=(failed, missing)
                ),
                self.assertRaisesRegex(
                    observer.ObservationError, "artifact_observations_complete"
                ),
            ):
                observer.build_observation(
                    arguments(
                        dynamic_library=artifact,
                        static_library=artifact,
                        target=["aarch64-macos"],
                    )
                )

    def test_truncated_required_query_and_oversized_archive_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            dynamic = Path(name) / "library.dylib"
            dynamic.write_bytes(b"\xcf\xfa\xed\xfe")

            def truncated(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                if command[0] == "file":
                    return observer.CommandResult(
                        command,
                        "exited",
                        0,
                        None,
                        b"Mach-O 64-bit arm64",
                        b"",
                        False,
                        False,
                    )
                if command[0] == "nm":
                    return observer.CommandResult(
                        command,
                        "exited",
                        0,
                        None,
                        b"0000 external _public_api\n",
                        b"",
                        True,
                        False,
                    )
                fixture = (
                    "macho_load_commands.txt"
                    if command[1] == "-l"
                    else "macho_libraries.txt"
                )
                return observer.CommandResult(
                    command,
                    "exited",
                    0,
                    None,
                    (FIXTURES / fixture).read_bytes(),
                    b"",
                    False,
                    False,
                )

            with mock.patch.object(observer, "run_command", side_effect=truncated):
                result = observer.observe_artifact(
                    dynamic, "dynamic", ROOT, {"public_api"}
                )
            self.assertEqual("failed", result["status"])
            self.assertIn(
                "complete symbol declarations were not observed", result["issues"]
            )

            static = Path(name) / "library.a"
            static.write_bytes(b"!<arch>\n")

            def crowded(argv: object, **_: object) -> object:
                command = tuple(argv)  # type: ignore[arg-type]
                stdout = b""
                if command[0] == "file":
                    stdout = b"current ar archive"
                elif command[:2] == ("lipo", "-info"):
                    stdout = b"Non-fat file: library.a is architecture: arm64"
                elif command[:2] == ("ar", "-t"):
                    stdout = b"__.SYMDEF\none.o\ntwo.o\n"
                elif command[:2] == ("ar", "-p"):
                    stdout = b"\xcf\xfa\xed\xfe\x0c\0\0\x01"
                elif command[0] == "nm":
                    stdout = b"0000 external _public_api\n"
                return observer.CommandResult(
                    command, "exited", 0, None, stdout, b"", False, False
                )

            with (
                mock.patch.object(observer, "MAX_ARTIFACT_ARCHIVE_MEMBERS", 1),
                mock.patch.object(observer, "run_command", side_effect=crowded),
            ):
                result = observer.observe_artifact(
                    static, "static", ROOT, {"public_api"}
                )
            self.assertEqual("failed", result["status"])
            self.assertIn(
                "archive member count exceeded the observation budget", result["issues"]
            )

    def test_unknown_source_classification_raises(self) -> None:
        unknown = [
            {
                "exported_name": "mystery",
                "source_path": "src/mystery.zig",
                "visibility": "default",
                "category": "unclassified",
            }
        ]
        with mock.patch.object(observer, "scan_zig_exports", return_value=unknown):
            with self.assertRaisesRegex(
                observer.ObservationError, "unknown_is_not_pass"
            ):
                observer.build_observation(arguments())

    def test_schema_rejects_reasonless_not_observed(self) -> None:
        value = {
            "schema": {"name": observer.SCHEMA_NAME, "version": 1},
            "observer": {"role": "observer_not_abi_authority"},
            "bad": {"status": "not_observed"},
        }
        with self.assertRaises(observer.ObservationError):
            observer.validate_observation(value)


class GitObservationTests(unittest.TestCase):
    def test_git_observation_uses_one_coherent_repository_identity(self) -> None:
        identity = observer.repository_git.RepositoryGitIdentity(
            revision="1" * 40,
            branch="main",
            detached=False,
            status_bytes=b" M tracked.txt\n?? new.txt\n",
            status_lines=(" M tracked.txt", "?? new.txt"),
            status_sha256="status-digest",
            index_sha256="index-digest",
        )
        repository = mock.Mock()
        repository.observe_identity.return_value = identity
        with (
            mock.patch.object(
                observer.repository_git,
                "open_repository",
                return_value=repository,
            ),
            mock.patch.object(
                observer,
                "run_command",
                side_effect=AssertionError("legacy Git query must not run"),
            ),
        ):
            result = observer.git_observation(ROOT)

        repository.observe_identity.assert_called_once_with(include_index=True)
        self.assertEqual(result["head"], "1" * 40)
        self.assertEqual(result["index_sha256"], "index-digest")
        self.assertEqual(result["status_sha256"], "status-digest")
        self.assertEqual(
            result["status_summary"],
            {"clean": False, "entry_count": 2, "states": {" M": 1, "??": 1}},
        )

    def test_any_git_failure_marks_every_git_fact_not_observed(self) -> None:
        repository = mock.Mock()
        repository.observe_identity.side_effect = (
            observer.repository_git.RepositoryGitCommandError("status observation")
        )
        with mock.patch.object(
            observer.repository_git,
            "open_repository",
            return_value=repository,
        ):
            result = observer.git_observation(ROOT)

        self.assertEqual(
            set(result),
            {"head", "index_sha256", "status_sha256", "status_summary"},
        )
        self.assertTrue(
            all(value["status"] == "not_observed" for value in result.values())
        )
        self.assertNotIn("clean", result["status_summary"])

    def test_repository_without_head_marks_every_git_fact_not_observed(self) -> None:
        repository = mock.Mock()
        repository.observe_identity.return_value = (
            observer.repository_git.RepositoryGitIdentity(
                revision=None,
                branch="main",
                detached=False,
                status_bytes=b"",
                status_lines=(),
                status_sha256="empty-status-digest",
                index_sha256="empty-index-digest",
            )
        )
        with mock.patch.object(
            observer.repository_git,
            "open_repository",
            return_value=repository,
        ):
            result = observer.git_observation(ROOT)

        self.assertTrue(
            all(value["status"] == "not_observed" for value in result.values())
        )
        self.assertNotIn("clean", result["status_summary"])


class DeterminismTests(unittest.TestCase):
    def test_source_only_observation_is_byte_deterministic(self) -> None:
        first = observer.build_observation(arguments())
        second = observer.build_observation(arguments())
        observer.validate_observation(first)
        self.assertEqual(
            observer.canonical_json_bytes(first), observer.canonical_json_bytes(second)
        )
        self.assertEqual(7, len(first["consumers"]))
        self.assertEqual(
            observer.PLATFORMS, tuple(item["target"] for item in first["platforms"])
        )
        self.assertEqual(
            observer.ARTIFACT_BUILD_CONFIGURATION, first["artifact_build_configuration"]
        )
        self.assertTrue(first["invariants"]["source_only_build_configuration_declared"])
        self.assertTrue(
            first["invariants"]["artifact_target_matches_build_configuration"]
        )
        self.assertEqual([], first["projections"]["source_missing"])
        self.assertEqual([], first["projections"]["source_extra"])
        self.assertTrue(
            all(
                item["source_declarations"]
                for item in first["projections"]["cross_reference"]
            )
        )
        encoded = observer.canonical_json_bytes(first)
        self.assertNotIn(tempfile.gettempdir().encode(), encoded)

    def test_policy_hash_remains_in_the_subject(self) -> None:
        observation = observer.build_observation(arguments())
        policy_path = ROOT / "tools/abi_baseline_observation.json"
        policy_input = next(
            item
            for item in observation["subject"]["inputs"]["inputs"]
            if item["path"] == "tools/abi_baseline_observation.json"
        )
        self.assertEqual(observer.sha256_file(policy_path), policy_input["sha256"])


if __name__ == "__main__":
    unittest.main()
