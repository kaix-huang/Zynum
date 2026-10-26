#!/usr/bin/env python3
# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check installed LP64 GELS in standalone and unified C/C++ consumers."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    SDK=args.prefix.resolve();D=args.output.resolve();ROOT=Path(__file__).resolve().parents[1]
    if platform.system()!='Darwin' or platform.machine()!='arm64':parser.error('requires the reviewed Apple arm64 C ABI target')
    manifest=SDK/'include/zynum/lapacke/lp64/abi_manifest.json'
    try:plan=json.loads(manifest.read_text())
    except (OSError,ValueError) as error:parser.error('cannot read installed LAPACKE manifest: '+str(error))
    components={c['plan']['component']:c['plan'] for c in plan.get('components',[])}
    if not {'gels','controls'}<=components.keys():parser.error('requires generated LP64 GELS and NaN control components')
    wanted=['LAPACKE_'+tag+'gels'+suffix for tag in 'sdcz' for suffix in ['', '_work']]
    if components['gels']['profile']!='mkl2026.lp64' or components['gels']['symbols']!=wanted:parser.error('installed GELS profile or symbol contract differs')
    D.mkdir(parents=True,exist_ok=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    payloads={str(p.relative_to(SDK)):dict(sha256=sha(p)) for p in SDK.rglob('*') if p.is_file()}
    def guard():
     for name,row in payloads.items():assert sha(SDK/name)==row['sha256'],name
    cfile=ROOT/'test/headers/lapacke_gels_smoke.c'
    runs = []
    try:
        guard()
        env = dict(os.environ)
        env.pop('DYLD_LIBRARY_PATH',None)
        env.pop('DYLD_FALLBACK_LIBRARY_PATH',None)
        sysroot = subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip()
        for family in ['abplas','lapacke']:
            for kind in ['static','shared']:
                library = SDK / ('lib/libzynum_'+family+'_lp64.'+('a' if kind=='static' else 'dylib'))
                for language, standard, custom in [('c','c11',False),('c++','c++20',False),('c++','c++20',True)]:
                    label=family+'-'+kind+'-'+standard+('-std-complex' if custom else '')
                    binary=D/label
                    argv=['/usr/bin/'+('clang++' if language=='c++' else 'clang'),'-x',language,'-std='+standard,'-isysroot',sysroot,'-Werror','-I'+str(SDK/'include')]
                    if custom:argv+=['-DUSE_STD_COMPLEX']
                    argv += [str(cfile),'-x','none',str(library),'-Wl,-rpath,'+str(SDK/'lib'),'-o',str(binary)]
                    cp=subprocess.run(argv,env=env,capture_output=True,text=True)
                    (D/(label+'-compile.stdout.txt')).write_text(cp.stdout);(D/(label+'-compile.stderr.txt')).write_text(cp.stderr)
                    assert cp.returncode==0,(label,'compile',cp.returncode)
                    native=subprocess.run([str(binary),'--',str(binary if kind=='static' else library)],env=env,capture_output=True,text=True)
                    (D/(label+'-run.stdout.txt')).write_text(native.stdout);(D/(label+'-run.stderr.txt')).write_text(native.stderr)
                    assert native.returncode==0,(label,'runtime',native.returncode,native.stderr[-2000:])
                    counts=json.loads(native.stdout);assert counts==dict(queries=52,computes=384,boundaries=84,owners=8)
                    links=subprocess.check_output(['/usr/bin/otool','-L',str(binary)],text=True);imports=subprocess.check_output(['/usr/bin/nm','-u',str(binary)],text=True)
                    (D/(label+'-links.txt')).write_text(links);(D/(label+'-imports.txt')).write_text(imports)
                    if kind=='shared':assert library.name in links and all('_LAPACKE_'+s in imports for s in ['sgels','dgels','cgels','zgels'])
                    else:assert 'libzynum_' not in links and 'LAPACKE_' not in imports
                    runs.append(dict(label=label,compiler_argv=argv,compile_exit=cp.returncode,actual_native_exit=native.returncode,actual_counts=counts,binary_sha256=sha(binary),library_sha256=sha(library)))
                    (D/'progress.json').write_text(json.dumps(dict(runs=runs),indent=2)+'\n')
                    print(dict(consumer=label,actual_exit=0,**counts),flush=True)
        guard()
        (D/'completion.json').write_text(json.dumps(dict(qualified=True,actual_external_consumers=len(runs),runs=runs,total_queries=52*len(runs),total_computes=384*len(runs),total_boundaries=84*len(runs),total_symbol_owner_checks=8*len(runs),guarded_installed_files=len(payloads),LP64_only=True,ILP64_GELS_claim=False,ABP_new_GELS_call_claim=False,full_LAPACK_profile_claim=False),indent=2)+'\n')
    except Exception as ex:
        (D/'failure.json').write_text(json.dumps(dict(qualified=False,error_type=type(ex).__name__,error=str(ex),runs_completed=len(runs)),indent=2)+'\n')
        raise


if __name__ == "__main__":
    main()
