# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""UT for the private SHMEM binding's native path and load diagnostics."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.shmem import _bindings


class TestShmemBindings(unittest.TestCase):
    """Cover loader behavior without initializing any native communicator."""

    def test_source_payload_precedes_installed_and_missing_is_actionable(self):
        """A source checkout resolves its own payload, with no legacy fallback."""
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        module = root / "hyper_parallel/core/multicore/shmem/_bindings.py"
        relative = Path("core/multicore/shmem/lib/framework/torch/libaclshmem_torch.so")
        source = root / "build/native/payload/hyper_parallel" / relative
        installed = root / "hyper_parallel" / relative
        (root / "setup.py").touch()
        for path in (source, installed):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        with patch.object(_bindings, "__file__", str(module)):
            self.assertEqual(_bindings._require_library(), source)
            source.unlink()
            self.assertEqual(_bindings._require_library(), installed)
            installed.unlink()
            with self.assertRaisesRegex(ImportError, "HP-NATIVE-PAYLOAD-MISSING.*--multicore on"):
                _bindings._require_library()

    def test_native_load_failure_preserves_empty_cache(self):
        """ABI and dynamic-link errors are actionable, never hidden as success."""
        with (
            patch.object(_bindings, "_manager", None),
            patch.object(_bindings, "_ops", None),
            patch.object(_bindings, "_require_library", return_value=Path("/payload/libaclshmem_torch.so")),
            patch.object(_bindings.torch.ops, "load_library", side_effect=OSError("wrong ABI")),
        ):
            with self.assertRaisesRegex(ImportError, "HP-NATIVE-LOAD-FAILED.*wrong ABI"):
                _bindings.get_ops()
            self.assertIsNone(_bindings._manager)
            self.assertIsNone(_bindings._ops)
