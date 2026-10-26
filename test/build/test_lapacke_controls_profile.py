"""Check generated controls, C integer types and exact configuration ownership."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import generate_lapacke_component as generator
import lapacke_components as components

ZIG = shutil.which('zig')
CLANG = shutil.which('clang')
TARGET = 'apple-arm64-default-c-v1'

class ControlsProfileTests(unittest.TestCase):
    def base(self, profile):
        width = 'ilp64' if profile.endswith('ilp64') else 'lp64'
        return json.loads((ROOT / 'interface_subset' / profile / ('getri-ordinary-' + width + '-v1') / 'abi_manifest.json').read_text())

    def test_both_profiles_compose_without_new_math_operation_or_owner(self):
        for profile in components.PROFILES:
            with self.subTest(profile=profile):
                control = components.manifest_for(profile, 'controls', TARGET)
                base = self.base(profile)
                combined = components.compose(base, [control])
                self.assertEqual({'LAPACKE_set_nancheck','LAPACKE_get_nancheck'},
                    set(combined['plan']['symbols']) - set(base['plan']['symbols']))
                self.assertTrue(combined['one_existing_owner'])
                self.assertFalse(control['plan']['generated_owner_or_numerics'])
                self.assertTrue(all('operation' not in row for row in control['plan']['bindings']))
                self.assertTrue(all(row['control_integer_bits'] == 32 for row in control['plan']['bindings']))

    def test_edited_forwarding_width_and_owner_manifest_are_rejected(self):
        for change in ['forwarding', 'width', 'owner']:
            with self.subTest(change=change):
                control = components.manifest_for('mkl2026.lp64', 'controls', TARGET)
                if change == 'forwarding':
                    control['plan']['bindings'][0]['configuration_forward'] = 'unreviewedConfiguration'
                elif change == 'width':
                    control['plan']['bindings'][0]['control_integer_bits'] = 64
                else:
                    control['plan']['owner_imports'] = [{'name': 'copied-owner'}]
                with self.assertRaisesRegex(ValueError, 'source-owned plan'):
                    components.compose(self.base('mkl2026.lp64'), [control])

    def test_duplicate_unordered_and_mixed_profile_components_are_rejected(self):
        base = self.base('mkl2026.lp64')
        control = components.manifest_for('mkl2026.lp64', 'controls', TARGET)
        potri = components.manifest_for('mkl2026.lp64', 'potri', TARGET)
        for values in [[control, control], [potri, control],
                       [components.manifest_for('mkl2026.ilp64', 'controls', TARGET)]]:
            with self.subTest(components=[v['plan']['component'] for v in values]):
                with self.assertRaises(ValueError):
                    components.compose(base, values)

    @unittest.skipUnless(ZIG and CLANG, 'Zig and Clang are required for compiled control type checks')
    def test_actual_C_and_CXX_control_types_stay_C_int_in_both_profiles(self):
        with tempfile.TemporaryDirectory(prefix='controls-header-types-') as directory:
            root = Path(directory)
            for profile in components.PROFILES:
                outputs = generator.outputs(profile, 'controls', TARGET, ZIG)
                (root / 'lapacke_controls.h').write_text(outputs['lapacke_controls.h'])
                (root / 'lapacke_base.h').write_text('typedef ' + ('long long' if profile.endswith('ilp64') else 'int') + ' lapack_int;\n')
                for language, standard, extension in [('c','c11','c'),('c++','c++20','cpp')]:
                    source = root / ('types.' + extension)
                    checks = ('#include <type_traits>\n'
                        'static_assert(std::is_same<decltype(&LAPACKE_set_nancheck),void(*)(int)>::value,"setter");\n'
                        'static_assert(std::is_same<decltype(&LAPACKE_get_nancheck),int(*)(void)>::value,"getter");\n'
                        'static_assert(sizeof(int)==4,"control integer width");\n') if language == 'c++' else (
                        '_Static_assert(__builtin_types_compatible_p(__typeof__(&LAPACKE_set_nancheck),void(*)(int)),"setter");\n'
                        '_Static_assert(__builtin_types_compatible_p(__typeof__(&LAPACKE_get_nancheck),int(*)(void)),"getter");\n'
                        '_Static_assert(sizeof(int)==4,"control integer width");\n')
                    source.write_text('#include "lapacke_controls.h"\n' + checks)
                    result = subprocess.run(['clang','-x',language,'-std='+standard,'-Wall','-Wextra','-Werror','-fsyntax-only',str(source)],capture_output=True,text=True)
                    self.assertEqual(0, result.returncode, result.stderr)

    @unittest.skipUnless(ZIG, 'Zig is required for owner and width compiler guards')
    def test_real_composer_compiler_rejects_same_Core_different_configuration_owner_and_wrong_width(self):
        with tempfile.TemporaryDirectory(prefix='controls-owner-guard-') as directory:
            root = Path(directory)
            (root / 'core.zig').write_text('// Identity-only Core fixture.\n')
            binding = ('pub const Provider = struct { pub const OwnerIdentity = @import("zynum-core"); };\n'
                'pub const shared_services = struct {\n'
                'pub fn setNancheck(flag: c_int) void { _ = flag; }\n'
                'pub fn getNancheck() c_int { return 1; }\n};\n')
            for name in ['binding_a','binding_b']:
                (root / (name + '.zig')).write_text(binding)
            cases = [('mkl2026.lp64',False,False,None), ('mkl2026.ilp64',False,False,None),
                     ('mkl2026.lp64',True,False,'share the math configuration owner'),
                     ('mkl2026.ilp64',False,True,'integer profiles differ')]
            for i, (profile, split_owner, wrong_width, error) in enumerate(cases):
                with self.subTest(profile=profile, split_owner=split_owner, wrong_width=wrong_width):
                    outputs = generator.outputs(profile, 'controls', TARGET, ZIG)
                    (root / 'controls.zig').write_text(outputs['exports.zig'])
                    width = 'c_longlong' if profile.endswith('ilp64') and not wrong_width else 'c_int'
                    (root / 'base.zig').write_text('pub const WireInt = '+width+';\n')
                    argv = ['zig','build-obj','-ODebug','-target','aarch64-macos','-mcpu','baseline',
                        '--dep','zynum-core=core','--dep','lapacke-interface-owner=binding_a',
                        '--dep','zynum-lapacke-base-exports=base','--dep','zynum-lapacke-controls-exports=controls',
                        '-Mroot='+str(ROOT/'src/lapacke_controls_composed.zig'),
                        '--dep','lapacke-interface-owner='+('binding_b' if split_owner else 'binding_a'),
                        '-Mcontrols='+str(root/'controls.zig'),'--dep','zynum-core=core','-Mbinding_a='+str(root/'binding_a.zig')]
                    if split_owner:
                        argv += ['--dep','zynum-core=core','-Mbinding_b='+str(root/'binding_b.zig')]
                    argv += ['-Mcore='+str(root/'core.zig'),'-Mbase='+str(root/'base.zig'),
                        '--cache-dir',str(root/('cache'+str(i))),'-femit-bin='+str(root/('out'+str(i)+'.o'))]
                    result = subprocess.run(argv,capture_output=True,text=True,cwd=root)
                    if error is None:
                        self.assertEqual(0, result.returncode, result.stderr)
                    else:
                        self.assertNotEqual(0, result.returncode)
                        self.assertIn(error, result.stderr)

if __name__ == '__main__':
    unittest.main(verbosity=2)
