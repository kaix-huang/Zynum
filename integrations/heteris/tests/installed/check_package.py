#!/usr/bin/env python3
"""Plan/run installed Linux CPU package checks; never builds Heteris Runtime."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[4]
HEADERS = {
    "heteris.h": "resources/include/heteris.h",
    "heteris_runtime_provider.h": "resources/include/heteris_runtime_provider.h",
    "heteris_data.h": "data/include/heteris_data.h",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--heteris-source", required=True, type=Path)
    parser.add_argument("--install-root", type=Path, help="Separate installs/logs inside cache; reuses producer build directories")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    cache = args.cache_root.expanduser().resolve()
    install_root = args.install_root.expanduser().resolve() if args.install_root else cache
    if install_root != cache and cache not in install_root.parents:
        parser.error("install-root must stay within the assigned cache")
    heteris = args.heteris_source.resolve()
    if cache == SOURCE or SOURCE in cache.parents or cache in SOURCE.parents:
        parser.error("cache must be outside the source and its ancestors")
    contract = heteris / "interfaces/production/public/contracts"
    inputs = {name: contract / relative for name, relative in HEADERS.items()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
    version = re.search(r'[.]version\s*=\s*"([^"]+)"', (SOURCE / "build.zig.zon").read_text()).group(1)
    cases = [(config, enabled) for config in ("Debug", "Release") for enabled in (False, True)]
    plan = {
        "source": str(SOURCE), "version": version, "cache": str(cache), "install_root": str(install_root), "heteris_source": str(heteris),
        "contract_sha256": hashes,
        "cases": [{"configuration": config, "standalone": enabled} for config, enabled in cases],
        "commands": [
            "cmake -S SOURCE -B CACHE/CONFIG-on|off/producer -G Ninja -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=CONFIG -DHETERIS_ABI_INCLUDE_DIR=CACHE/sdk/include -DZYNUM_BUILD_STANDALONE=ON|OFF -DCMAKE_INSTALL_PREFIX=CASE/original -DCMAKE_INSTALL_LIBDIR=lib64",
            "cmake --build CACHE/CONFIG-on|off/producer --parallel 2; cmake --install CACHE/CONFIG-on|off/producer",
            "cmake -S tests/installed -B CASE/consumer -G Ninja -DCMAKE_PREFIX_PATH=PREFIX -DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "cmake --build CASE/consumer; ctest --test-dir CASE/consumer --output-on-failure",
            "negative find_package(required missing standalone / unknown component); optional missing standalone",
            "nm -g --defined-only libzynum.a (no naive provider symbols)",
            "move original prefix to relocated; repeat independent consumers with original absent",
            "missing optional archive/header: real core/default and OPTIONAL consumers pass; required standalone fails",
            "reject conflicting existing imported targets; repeated same-prefix core then standalone succeeds",
            "reject each of seven declared link/compile/source usage mismatches on fresh core and standalone targets",
            "linked empty-interface default/core/optional/required and same-prefix repeat controls succeed",
            "inspect compile_commands and Ninja link commands for installed include/archive closure",
        ],
        "boundaries": "C11/C++20 Linux CPU package only; no full Core, material, MPI, GPU, performance or cluster qualification",
    }
    print(json.dumps(plan, indent=2))
    if not args.run:
        return 0
    cache.mkdir(parents=True, exist_ok=True)
    install_root.mkdir(parents=True, exist_ok=True)
    binding = cache / "source-binding.txt"
    if binding.exists() and binding.read_text().strip() != str(SOURCE):
        parser.error("cache belongs to a different source")
    binding.write_text(str(SOURCE) + "\n")
    sdk = cache / "sdk/include"
    sdk.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        shutil.copyfile(path, sdk / name)
    poison = cache / "poison"
    poison.mkdir(exist_ok=True)
    marker = cache / "forbidden-tool-invoked"
    for name in ("cargo", "rustc", "mpicc", "mpicxx", "mpiexec", "nvcc"):
        tool = poison / name
        tool.write_text("#!/bin/sh\necho " + name + " >> '" + str(marker) + "'\nexit 91\n")
        tool.chmod(0o755)
    guard = poison / "guard.cmake"
    guard.write_text("""macro(find_package)
    if("${ARGV0}" MATCHES "^(Heteris|ScientificPlatform|MPI|CUDAToolkit|CUDA|Rust|Cargo)$")
        message(FATAL_ERROR "Forbidden package discovery: ${ARGV0}")
    endif()
    _find_package(${ARGV})
endmacro()
""")
    env = dict(os.environ, PATH=str(poison) + os.pathsep + os.environ["PATH"])
    report = dict(plan, source_head=subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
                  heteris_head=subprocess.check_output(["git", "-C", str(heteris), "rev-parse", "HEAD"], text=True).strip(),
                  results=[])
    logfile = install_root / "run.log"
    if logfile.exists():
        shutil.copyfile(logfile, install_root / "run.previous.log")
    with logfile.open("w") as log:
        def run(command, expected=0):
            command = [str(item) for item in command]
            print("+ " + " ".join(command), flush=True)
            result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write("+ " + " ".join(command) + "\n" + result.stdout + "\n")
            log.flush()
            report["results"].append({"command": command, "returncode": result.returncode, "expected": expected})
            if (expected == 0 and result.returncode != 0) or (expected != 0 and result.returncode == 0):
                print(result.stdout, file=sys.stderr)
                raise RuntimeError("unexpected command result")
            return result.stdout

        def consumer(case, label, prefix, components="", expected=0, optional=False):
            build = case / label
            command = ["cmake", "-S", SOURCE / "integrations/heteris/tests/installed", "-B", build, "-G", "Ninja",
                       "-DCMAKE_PREFIX_PATH=" + str(prefix), "-DCMAKE_BUILD_TYPE=" + case.name.split("-")[0],
                       "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DZYNUM_TEST_COMPONENTS=" + components,
                       "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=" + str(guard),
                       "-DZYNUM_TEST_OPTIONAL_STANDALONE=" + ("ON" if optional else "OFF")]
            output = run(command, expected)
            if expected:
                if "standalone" not in output and "Unknown Zynum component" not in output:
                    raise RuntimeError("negative case failed for an unrelated reason")
                return
            run(["cmake", "--build", build, "--parallel", "2"])
            run(["ctest", "--test-dir", build, "--output-on-failure"])
            compile_commands = (build / "compile_commands.json").read_text()
            links = run(["ninja", "-C", build, "-t", "commands"])
            # Every compile includes the relocated installed SDK only; neither
            # source public headers nor the producer's flat SDK can satisfy it.
            for forbidden in (str(SOURCE / "include"), str(SOURCE / "integrations/heteris/include"), str(sdk)):
                if forbidden in compile_commands or forbidden in links:
                    raise RuntimeError("source/producer include path leaked: " + forbidden)
            if str(prefix / "include") not in compile_commands or str(prefix / "lib64/libzynum.a") not in links:
                raise RuntimeError("installed target include/archive closure missing")
            for forbidden in ("libheteris_runtime", "libmpi", "libcuda", "libcudart", "libzynum-standalone.a" if not components else "unused-sentinel"):
                if forbidden in links:
                    raise RuntimeError("unexpected link dependency: " + forbidden)

        for config, enabled in cases:
            case_name = config + ("-on" if enabled else "-off")
            case = install_root / case_name
            producer = cache / case_name / "producer"
            prefix, relocated = case / "original", case / "relocated"
            # A previous accepted relocation is reused as the next install
            # prefix; no recursive deletion or extra source/compiler copy.
            if relocated.exists() and not prefix.exists():
                relocated.rename(prefix)
            run(["cmake", "-S", SOURCE, "-B", producer, "-G", "Ninja",
                 "-DBUILD_TESTING=OFF", "-DCMAKE_BUILD_TYPE=" + config,
                 "-DHETERIS_ABI_INCLUDE_DIR=" + str(sdk), "-DZYNUM_BUILD_STANDALONE=" + ("ON" if enabled else "OFF"),
                 "-DCMAKE_INSTALL_PREFIX=" + str(prefix), "-DCMAKE_INSTALL_LIBDIR=lib64",
                 "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=" + str(guard)])
            run(["cmake", "--build", producer, "--parallel", "2"])
            run(["cmake", "--install", producer])
            if version not in (prefix / "share/cmake/Zynum/ZynumConfig.cmake").read_text():
                raise RuntimeError("source prerelease version lost")
            symbols = run(["nm", "-g", "--defined-only", prefix / "lib64/libzynum.a"])
            if "zynum_resources_create_naive" in symbols or "zynum_resources_destroy_naive" in symbols:
                raise RuntimeError("standalone provider leaked into core")
            if not enabled and ((prefix / "lib64/libzynum-standalone.a").exists() or (prefix / "include/zynum_standalone.h").exists()):
                raise RuntimeError("core-only installation contains standalone artifacts")
            consumer(case, "consumer-default", prefix)
            consumer(case, "consumer-core", prefix, "core")
            consumer(case, "unknown-required", prefix, "not_a_component", expected=1)
            if enabled:
                consumer(case, "consumer-standalone", prefix, "core;standalone")
            else:
                consumer(case, "missing-standalone", prefix, "standalone", expected=1)
                consumer(case, "optional-missing-standalone", prefix, optional=True)
            prefix.rename(relocated)
            if prefix.exists():
                raise RuntimeError("original prefix still available")
            consumer(case, "consumer-relocated", relocated, "core;standalone" if enabled else "core")
        # These regressions reproduce real discovery failures independently of
        # target generation. Missing optional artifacts also get linked callers.
        selected = install_root / "Debug-on/relocated"
        other = install_root / "Release-on/relocated"
        for relative in ("lib64/libzynum-standalone.a", "include/zynum_standalone.h"):
            artifact = selected / relative
            saved = artifact.with_name(artifact.name + ".package-probe-saved")
            if not artifact.resolve().is_relative_to(install_root) or saved.exists():
                raise RuntimeError("unsafe or stale optional-artifact probe")
            label = "missing-" + artifact.name.replace(".", "-")
            try:
                artifact.rename(saved)
                consumer(install_root / "Debug-on", label + "-default", selected)
                consumer(install_root / "Debug-on", label + "-core", selected, "core")
                consumer(install_root / "Debug-on", label + "-optional", selected, optional=True)
                consumer(install_root / "Debug-on", label + "-required", selected, "standalone", expected=1)
            finally:
                saved.rename(artifact)

        def find(prefix, components="core"):
            return ('set(Zynum_DIR "' + str(prefix / "share/cmake/Zynum") + '")\n'
                    'find_package(Zynum CONFIG REQUIRED COMPONENTS ' + components + ')\n')

        def probe(label, body, expected=0, diagnostic=""):
            source = install_root / "discovery-probes" / (label + "-source")
            source.mkdir(parents=True, exist_ok=True)
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\nproject(Probe LANGUAGES NONE)\n" + body)
            output = run(["cmake", "-S", source, "-B", source.with_name(label), "-G", "Ninja",
                          "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=" + str(guard)], expected)
            if diagnostic and diagnostic not in output:
                raise RuntimeError("probe did not diagnose the selected condition: " + label)

        probe("same-prefix-core-then-standalone",
              find(selected) + find(selected, "core standalone") +
              'if(NOT TARGET Zynum::core OR NOT TARGET Zynum::standalone)\n'
              'message(FATAL_ERROR "Repeated discovery lost targets")\nendif()\n')
        conflict = "Zynum target origin conflict"
        probe("two-prefix-core", find(selected) + find(other), 1, conflict)
        standalone = ('add_library(Zynum::standalone STATIC IMPORTED)\n'
                      'set_target_properties(Zynum::standalone PROPERTIES IMPORTED_LOCATION "' +
                      str(selected / "lib64/libzynum-standalone.a") +
                      '" INTERFACE_INCLUDE_DIRECTORIES "' + str(selected / "include") +
                      '" INTERFACE_LINK_LIBRARIES Zynum::core)\n')
        probe("foreign-standalone", standalone + find(other, "core standalone"), 1, conflict)
        for label, target_type, include, location_override, links in (
                ("wrong-core-type", "INTERFACE", other / "include", "", "m"),
                ("wrong-core-include", "STATIC", selected / "include", "", "m"),
                ("wrong-core-configuration", "STATIC", other / "include",
                 ' IMPORTED_LOCATION_RELEASE "' + str(selected / "lib64/libzynum.a") + '"', "m"),
                ("wrong-core-link-interface", "STATIC", other / "include", "", "m;foreign")):
            body = ('add_library(Zynum::core ' + target_type + ' IMPORTED)\n'
                    'set_target_properties(Zynum::core PROPERTIES IMPORTED_LOCATION "' +
                    str(other / "lib64/libzynum.a") + '" INTERFACE_INCLUDE_DIRECTORIES "' +
                    str(include) + '" INTERFACE_LINK_LIBRARIES "' + links + '"' + location_override + ')\n')
            probe(label, body + find(other), 1, conflict)
        for property_name in ("location", "include"):
            label = "relative-core-" + property_name
            source = install_root / "discovery-probes" / (label + "-source")
            library = str(other / "lib64/libzynum.a")
            include = str(other / "include")
            if property_name == "location":
                library = os.path.relpath(library, source)
            else:
                include = os.path.relpath(include, source)
            body = ('add_library(Zynum::core STATIC IMPORTED)\n'
                    'set_target_properties(Zynum::core PROPERTIES IMPORTED_LOCATION "' +
                    library + '" INTERFACE_INCLUDE_DIRECTORIES "' + include +
                    '" INTERFACE_LINK_LIBRARIES m)\n')
            probe(label, body + find(other), 1, conflict)
        # Required core artifacts remain mandatory.
        for relative in ("lib64/libzynum.a", "include/heteris_data.h"):
            artifact = selected / relative
            saved = artifact.with_name(artifact.name + ".package-probe-saved")
            if not artifact.resolve().is_relative_to(install_root) or saved.exists():
                raise RuntimeError("unsafe or stale core-artifact probe")
            try:
                artifact.rename(saved)
                probe("missing-core-" + artifact.name.replace(".", "-"), find(selected), 1, artifact.name)
            finally:
                saved.rename(artifact)
        def usage_interface_probe(kind, property_name=None, mode="required"):
            # A real C consumer makes these usage requirements observable at
            # generation/link time if discovery ever stops rejecting them.
            label = kind + "-" + (property_name.lower() if property_name else mode)
            source = install_root / "usage-interface-probes" / label / "source"
            build = source.with_name("build")
            source.mkdir(parents=True, exist_ok=True)
            (source / "foreign-search").mkdir(exist_ok=True)
            (source / "extra.c").write_text("int foreign_source_probe(void) { return 0; }\n")
            main = ("#include <zynum_native.h>\n" +
                    ("#include <zynum_standalone.h>\n" if kind == "standalone" else "") +
                    "int main(void) { zynum_native_context_requirements_v1 r = {0};\n"
                    "if (zynum_native_context_query_v1(&r) != ZYNUM_NATIVE_OK) return 1;\n" +
                    ("zynum_resources p; if (zynum_resources_create_naive(&p) != HETERIS_OK) return 2;\n"
                     "if (zynum_resources_destroy_naive(&p) != HETERIS_OK) return 3;\n"
                     if kind == "standalone" else "") + "return 0; }\n")
            (source / "main.c").write_text(main)
            body = ""
            if mode != "repeat":
                body += ('add_library(Zynum::core STATIC IMPORTED)\n'
                         'set_target_properties(Zynum::core PROPERTIES IMPORTED_LOCATION "' +
                         str(other / "lib64/libzynum.a") + '" INTERFACE_INCLUDE_DIRECTORIES "' +
                         str(other / "include") + '" INTERFACE_LINK_LIBRARIES m)\n')
                if kind == "standalone":
                    body += ('add_library(Zynum::standalone STATIC IMPORTED)\n'
                             'set_target_properties(Zynum::standalone PROPERTIES IMPORTED_LOCATION "' +
                             str(other / "lib64/libzynum-standalone.a") +
                             '" INTERFACE_INCLUDE_DIRECTORIES "' + str(other / "include") +
                             '" INTERFACE_LINK_LIBRARIES Zynum::core)\n')
            if property_name:
                values = {
                    "INTERFACE_LINK_LIBRARIES_DIRECT": "foreign_dependency",
                    "INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE": "Zynum::" + kind,
                    "INTERFACE_LINK_OPTIONS": "-Wl,--zynum-foreign-probe",
                    "INTERFACE_LINK_DIRECTORIES": str(source / "foreign-search"),
                    "INTERFACE_COMPILE_OPTIONS": "-DZ_FOREIGN_OPTION",
                    "INTERFACE_COMPILE_DEFINITIONS": "Z_FOREIGN_DEFINITION",
                    "INTERFACE_SOURCES": str(source / "extra.c"),
                }
                body += ('set_property(TARGET Zynum::' + kind + ' PROPERTY ' +
                         property_name + ' "' + values[property_name] + '")\n')
            if mode == "repeat":
                body += find(other) + find(other, "core standalone")
                body += ('set(Zynum_DIR "' + str(other / "share/cmake/Zynum") +
                         '")\nfind_package(Zynum CONFIG REQUIRED)\n')
                body += find(other, "core OPTIONAL_COMPONENTS standalone")
                body += find(other, "core standalone")
            elif mode == "default":
                body += ('set(Zynum_DIR "' + str(other / "share/cmake/Zynum") +
                         '")\nfind_package(Zynum CONFIG REQUIRED)\n')
            else:
                components = ("core OPTIONAL_COMPONENTS standalone" if mode == "optional" else
                              "core standalone" if kind == "standalone" else "core")
                body += find(other, components)
            if kind == "standalone":
                body += ('if(NOT Zynum_standalone_FOUND OR NOT TARGET Zynum::standalone)\n'
                         'message(FATAL_ERROR "Installed standalone control unavailable")\nendif()\n')
            body += ("add_executable(usage_interface_probe main.c)\n"
                     "target_compile_features(usage_interface_probe PRIVATE c_std_11)\n"
                     "target_link_libraries(usage_interface_probe PRIVATE Zynum::" + kind + ")\n"
                     "add_test(NAME usage_interface_probe COMMAND usage_interface_probe)\n")
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\nproject(UsageProbe LANGUAGES C)\n"
                "include(CTest)\n" + body)
            output = run(["cmake", "-S", source, "-B", build, "-G", "Ninja",
                          "-DCMAKE_BUILD_TYPE=Release",
                          "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=" + str(guard)],
                         1 if property_name else 0)
            if property_name:
                if conflict not in output or property_name not in output:
                    raise RuntimeError("usage mismatch was rejected for an unrelated reason: " + label)
            else:
                run(["cmake", "--build", build, "--parallel", "2"])
                run(["ctest", "--test-dir", build, "--output-on-failure"])
                commands = run(["ninja", "-C", build, "-t", "commands"])
                if str(other / "lib64/libzynum.a") not in commands:
                    raise RuntimeError("control lost the selected core archive")
                if kind == "standalone" and str(other / "lib64/libzynum-standalone.a") not in commands:
                    raise RuntimeError("control lost the selected standalone archive")
                if any(value in commands for value in ("foreign_dependency", "foreign-search",
                                                       "Z_FOREIGN", "extra.c")):
                    raise RuntimeError("foreign usage leaked into an empty-interface control")

        for kind in ("core", "standalone"):
            for property_name in (
                    "INTERFACE_LINK_LIBRARIES_DIRECT", "INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE",
                    "INTERFACE_LINK_OPTIONS", "INTERFACE_LINK_DIRECTORIES", "INTERFACE_COMPILE_OPTIONS",
                    "INTERFACE_COMPILE_DEFINITIONS", "INTERFACE_SOURCES"):
                usage_interface_probe(kind, property_name)
        for kind, mode in (("core", "default"), ("core", "core"),
                           ("standalone", "optional"), ("standalone", "required"),
                           ("standalone", "repeat")):
            usage_interface_probe(kind, mode=mode)
        after = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
        if after != hashes:
            raise RuntimeError("canonical Heteris contract changed during qualification")
        if marker.exists():
            raise RuntimeError("forbidden external tool invoked")
    report["status"] = "passed"
    (install_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Installed package checks passed; report: " + str(install_root / "report.json"))
    return 0

if __name__ == "__main__":
    sys.exit(main())
