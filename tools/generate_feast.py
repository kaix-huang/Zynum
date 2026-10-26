# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate a finite FEAST LP64 boundary; native numerical algorithms stay in Core."""
import argparse
import hashlib
import json
from pathlib import Path

PROFILE = "feast4.lp64.abplas-v1"
SYMBOLS = ["feastinit_", "zfeast_hcsrev_"]

def validate_profile(profile, *, declaration=False):
    schema = "zynum-external-interface-declarations-v1" if declaration else "zynum-external-interface-profile-v1"
    if (not isinstance(profile, dict) or profile.get("schema") != schema or profile.get("profile") != PROFILE or profile.get("target") != "apple-arm64-default-c-v1"
        or type(profile.get("integer_bits")) is not int or profile["integer_bits"] != 32
        or type(profile.get("complex_f64_bytes")) is not int or profile["complex_f64_bytes"] != 16
        or profile.get("symbols") != SYMBOLS or profile.get("full_feast_compatibility") is not False):
        raise ValueError("unsupported or altered finite FEAST profile")

def validate_manifest(manifest):
    validate_profile(manifest, declaration=True)
    if (manifest.get("schema") != "zynum-external-interface-declarations-v1"
        or manifest.get("runtime_exported") is not False
        or manifest.get("status") != "candidate-runtime-and-ABP-integration-pending"):
        raise ValueError("invalid finite FEAST declaration manifest")
    for name in ("profile_sha256", "generator_sha256"):
        value = manifest.get(name)
        if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid finite FEAST source identity")
    outputs = manifest.get("generated_sha256")
    expected={"feast_lp64.h", "feast_exports.zig", "FEASTConfig.cmake", "FEASTConfigVersion.cmake"}
    if not isinstance(outputs,dict) or set(outputs)!=expected or any(not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v) for v in outputs.values()):
        raise ValueError("invalid finite FEAST generated identities")
    return tuple(SYMBOLS)

def generate(profile, destination, profile_path):
    validate_profile(profile)
    header = """/* Generated FEAST4 LP64 ABPLaS driver declarations. */
#ifndef ZYNUM_FEAST4_LP64_ABPLAS_V1_H
#define ZYNUM_FEAST4_LP64_ABPLAS_V1_H
#include <stdint.h>
#if !defined(__APPLE__) || !defined(__aarch64__)
#error "Selected FEAST ABI profile requires Apple arm64"
#endif
#ifdef __cplusplus
static_assert(sizeof(int)==4 && sizeof(double)==8 && sizeof(void*)==8,"FEAST LP64 ABI");
extern "C" {
#else
_Static_assert(sizeof(int)==4 && sizeof(double)==8 && sizeof(void*)==8,"FEAST LP64 ABI");
#endif
enum { ZYNUM_FEAST_UNSUPPORTED_MODE = -1000 };
void feastinit_(int*);
void zfeast_hcsrev_(char*,int*,double*,int*,int*,int*,double*,int*,double*,double*,int*,double*,double*,int*,double*,int*);
#ifdef __cplusplus
}
#endif
#endif
"""
    zig = """// Generated declarations only; numerical and resource ownership is injected.
const B=@import("feast-interface-owner");
pub export fn feastinit_(f:[*c]i32) callconv(.c) void { B.initialize(f); }
pub export fn zfeast_hcsrev_(u:[*c]u8,n:[*c]i32,a:[*c]f64,ia:[*c]i32,ja:[*c]i32,f:[*c]i32,e:[*c]f64,l:[*c]i32,lo:[*c]f64,hi:[*c]f64,m:[*c]i32,v:[*c]f64,x:[*c]f64,k:[*c]i32,r:[*c]f64,i:[*c]i32) callconv(.c) void { B.hcsrev(u,n,a,ia,ja,f,e,l,lo,hi,m,v,x,k,r,i); }
"""
    files = {"feast_lp64.h": header.encode(), "feast_exports.zig": zig.encode()}
    templates = Path(__file__).with_name("interface_templates")
    for name,template in (("FEASTConfig.cmake","feast_config.cmake.in"),("FEASTConfigVersion.cmake","feast_config_version.cmake.in")):
        files[name]=(templates/template).read_bytes()
    digest = lambda data: hashlib.sha256(data).hexdigest()
    manifest = {**profile, "schema": "zynum-external-interface-declarations-v1", "runtime_exported": False,
                "status": "candidate-runtime-and-ABP-integration-pending", "profile_sha256": digest(profile_path.read_bytes()),
                "generator_sha256": digest(Path(__file__).read_bytes()), "generated_sha256": {p:digest(b) for p,b in files.items()}}
    validate_manifest(manifest)
    # Validate before creating any output; generated files contain no math/state.
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items(): (destination/name).write_bytes(data)
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--profile",type=Path,default=Path(__file__).with_name("feast_lp64_profile.json"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    generate(json.loads(args.profile.read_text()),args.output,args.profile)
