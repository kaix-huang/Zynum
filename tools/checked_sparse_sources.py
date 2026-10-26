# Copyright (C) 2026 Zynum contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reviewed Sparse entry-source identity; not a complete compiler dependency closure."""
import hashlib
from pathlib import Path
from checked_sparse_schema import SparseProfileError

ROOT = Path(__file__).resolve().parents[1]
REVIEWED_SOURCES = {'tools/checked_complex_wire.py': '8aabb44c94f1f0f6fce466af19d26cde86616b9855e63b7c8b030e520f1ed730', 'include/zynum/checked/checked.h': '6d6ca6c9259d3b56612c1ed741c1b9deab3d3881f4f7147c1675107cc1a94c54', 'src/checked/abi_types.zig': 'd80b207134bc498e83f8bafa35b3024d810a81c07e46897a1218fa6e2b7adec7', 'src/checked/implementation.zig': 'bf05e8adfac3a678ab2db7a2b5fa8588a8ebba105ce358d7e8c27e6d238090c5', 'src/checked_sparse.zig': '48c4869497f482fdab6037c2fa758a2dd825c0ca99797887c6b24dbb688c0f31', 'src/facades/sparse.zig': '33c466b84222ad137d90e0a0bd22a7ac2689c2ed655425296fedbdb057c96ab4', 'src/numerics/dense.zig': '4dbdbc980179822174afb2ed89b9d2c569169c4ce0320cd862a103d46681214b', 'src/numerics/operator.zig': '861b29caa6f1c0846f84467dee4e59b7029f5c570546d8e27f3c8b8a2f2e01f2', 'src/numerics/storage.zig': '2b615abc2d108049e678e08e0280a9db42ee2e03fbc27494ee69725682ec7cf0', 'src/numerics/synchronization.zig': 'a6412c75c4dabc553066af4fe8faf27d91fa47c051f5b1432e3c51ba11fc122e', 'src/sparse.zig': '20ce7f4f08cdcad68c31fb82f8ff5bb5e5198a0bffc213281909dc09893ea1d4', 'src/sparse/core/product.zig': '10ae035ab5eec98ae5097063c82552f0593420173353f8dcf76d2f93ab062822', 'src/sparse/operations.zig': '5adf63051132ca579b5faaaef443efb93738a6eb1dd72abe0c680facbcb0826a', 'src/sparse/operator.zig': 'e5d32e8198f7e3f74dd3b08f9ad812ca5bbbce0409f25841ce290e4d7fca4ced', 'src/sparse/pattern.zig': 'e4b0f4ec040be891151ba49780dad03e47e470278de6287ff30ee497e476d4f5', 'src/sparse/scalar.zig': '1132c4f780ff224459132600835335070edf0d82ed9fabb3ba97873942e88fb4', 'src/sparse/views.zig': 'd77e6a9cf6ff8027f81876afd23812b43367ac6a6107c91a6a16fc1ad7fe26af', 'tools/checked_abi_schema.py': '27dce62299ea7b51481aaccdaeb501bb9e0fd75552a31d6d4413362d2b900951', 'tools/checked_sparse_profile.py': 'a8484cbc9963c6c14c1c7baaf02bc411902af51acd936c1faf7ebf80165dc280', 'tools/checked_sparse_schema.py': '48d1236432d37c1a27115df9eb4afb6d3238b826192775d0225472c5f2ad7f0a', 'tools/numerical_operation_schema.py': '3d28ad4bc5f5cac7bbdeeb113e2bece9f1faddf7c82dc12ccbf0b0e591bacd75'}


def verify_sources(root=ROOT):
    root = Path(root)
    for relative, expected in REVIEWED_SOURCES.items():
        source = root / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise SparseProfileError('Reviewed Sparse source changed or missing: ' + relative)
    return {'scope': 'reviewed entry sources only; not full numerical dependency closure',
            'sha256': dict(REVIEWED_SOURCES)}
