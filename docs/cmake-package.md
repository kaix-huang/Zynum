# Installed CMake package

Zynum's CMake integration installs an independent `Zynum` config package.
It does not require the Scientific Platform superproject package.

```sh
cmake -S . -B /external/build/zynum -G Ninja \
  -DHETERIS_ABI_INCLUDE_DIR=/selected/flat-contract-sdk/include \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX=/selected/install/zynum
cmake --build /external/build/zynum
cmake --install /external/build/zynum
```

The producer requires the selected global Zig 0.17.1 and the canonical Heteris
Data/resource C headers. A flat SDK contains `heteris.h`, `heteris_data.h`
and `heteris_runtime_provider.h`, without source-relative forwarding includes.
The existing embedded build may instead obtain canonical inputs from the
owning Heteris target. No additional ABI source copies are maintained in Zynum.

Consumers use:

```cmake
find_package(Zynum CONFIG REQUIRED COMPONENTS core)
target_link_libraries(my_consumer PRIVATE Zynum::core)
```

Omitting components selects only `core`. This target contains the native archive,
installed public headers, the bundled canonical Heteris contracts and, on Unix,
`libm`. It never finds Heteris Runtime, Cargo/Rust, MPI or CUDA. The installed
consumer needs C/C++ tools, not Zig or Rust. An integrated application explicitly
selects and links its resource provider separately; the native context never
constructs a fallback provider.

Build with `-DZYNUM_BUILD_STANDALONE=ON` to install the explicit lightweight
provider archive and header. A consumer then requests `COMPONENTS core standalone`
and links `Zynum::standalone`, which links the same `Zynum::core`. A required
missing or unknown component rejects discovery with a diagnostic. An optional
missing standalone component leaves core usable. Default/core-only discovery
does not create the standalone target even in an installation containing it.
Use a fresh prefix for a clean artifact inventory: installing OFF over an old ON
prefix does not delete artifacts owned by the earlier installation.

Archive, header and config destinations follow `GNUInstallDirs`. Package paths
are computed from the installed prefix; no source/build path is exported.
The installed archive is the configuration selected by `cmake --install --config`
(or `CMAKE_BUILD_TYPE` for single-config Ninja). Configurations installed to one
prefix replace that same archive; use separate prefixes to retain both Debug and
Release. Missing archives or public contract headers reject discovery.

The source version comes from `build.zig.zon`, currently `0.0.1-beta`. The former
integration-only CMake label was `0.1.0`; it now derives the numeric coordinate
`0.0.1` from that same authority. The numeric coordinate is solely a CMake
comparison coordinate, with no stable-release or future compatibility promise.
The full prerelease string remains `Zynum_VERSION`; range requests and numeric
`EXACT` requests do not imply an exact prerelease match.

Installed `share/licenses/zynum` retains Zynum's LICENSE, COPYING, upstream source
record and Zig MIT notice. Bundled Heteris contracts remain exact copies of the
producer-selected canonical inputs; `HeterisContractHeaders.sha256` records their
hashes. No Heteris implementation or new ABI definitions are added. The current
selected Heteris source has no separate top-level license file; this package does
not infer or relabel its licensing. Distribution provenance/permissions remain a
separate delivery responsibility.

## Focused checks

From Linux, inspect the plan before executing it:

```sh
python3 integrations/heteris/tests/installed/check_package.py \
  --cache-root /external/cache/zynum-package \
  --heteris-source /selected/original/heteris --plan
python3 integrations/heteris/tests/installed/check_package.py \
  --cache-root /external/cache/zynum-package \
  --heteris-source /selected/original/heteris --run
```

The dedicated check uses one external source-bound cache, Debug/Release producers,
standalone OFF/ON, C11/C++20 linked consumers, existing explicit-provider lifecycle
and Linux x86-64 complex ABI fixtures, missing/unknown/optional components,
a non-default lib64 layout, dependency/tool poison guards, symbol inventory and
relocation with the original install prefix absent. It checks the three canonical
input hashes before and after. It is a Linux CPU packaging check, not full Core,
material/SCF, GPU, MPI, Slurm or performance qualification. Release here uses the
existing ReleaseSafe archive policy; existing ReleaseFast mathematical evidence
is not rerun or enlarged by this package check. The parent integration/installed
matrix remains required before a milestone.

## Optional artifacts and repeated discovery

An ON installation with a missing standalone archive or header still supports
default/core-only discovery. Requesting standalone as an optional component
sets `Zynum_standalone_FOUND` false and keeps core usable; requiring it rejects
discovery. Core's archive and public contract headers always remain required.

Repeated discovery from the same installation may first load core and later add
standalone. At discovery, existing targets must match the package's declared
static-imported type, absolute archive/include paths and checked configuration
locations. The supported usage-interface consistency checks are:

- `INTERFACE_LINK_LIBRARIES`: the package's Unix `m` dependency for core, or
  `Zynum::core` for standalone.
- `INTERFACE_SYSTEM_INCLUDE_DIRECTORIES`,
  `INTERFACE_LINK_LIBRARIES_DIRECT`, `INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE`,
  `INTERFACE_LINK_OPTIONS`, `INTERFACE_LINK_DIRECTORIES`,
  `INTERFACE_COMPILE_OPTIONS`, `INTERFACE_COMPILE_DEFINITIONS` and
  `INTERFACE_SOURCES`: empty for both targets.

A mismatch in this declared contract causes a target-origin diagnostic instead
of silently reusing the target. These checks concern the target presented when
`find_package` selects the package. They neither enumerate all CMake properties
nor prevent a consumer from deliberately modifying targets afterward; they are
package consistency checks, not a security sandbox.

The package harness exercises both missing optional artifacts with real default,
core and optional-component linked callers, required-component rejection, foreign
core/standalone targets and each supported link/compile/source usage mismatch in
real C consumer projects. Linked default/core/optional/required and same-prefix
repeat controls exercise the normal empty interfaces. Its optional `--install-root /external/cache/zynum-package/v2` keeps
installs, consumer builds and logs separate while reusing the source-bound
producer build directories under `--cache-root`. The install root must stay
inside that cache. This allows a previously accepted installed prefix to remain
available to other readers during package repair.
