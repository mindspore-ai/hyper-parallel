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
"""VLM facade export boundary, pinned after the stage-6 move.

The top-level ``hyper_parallel.data`` package must not re-export VLM
symbols flat. The facade ``__all__``/signature snapshot that originally
lived in this module was removed: importing ``hyper_parallel.data.vlm``
pulls in ``transformers.AutoProcessor``, which transitively imports
``torchvision`` — a broken third-party package on the UT gate executors.
"""
# pylint: disable=wrong-import-position

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from tests.common.mark_utils import arg_mark

_FACADE_ALL = [
    "VLMBatchProcessor",
    "VLMGetBatch",
    "build_processor",
    "build_vlm_collator",
    "build_vlm_data_transform",
    "build_vlm_dataset",
    "build_vlm_get_batch",
]


class TestVlmFacadeExports(unittest.TestCase):
    """The data root package keeps the VLM facade out of its flat exports."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_data_root_does_not_flat_export_vlm(self):
        """The top-level data package never re-exports VLM symbols flat."""
        import hyper_parallel.data as data

        exported = set(getattr(data, "__all__", None) or [])
        for name in _FACADE_ALL:
            self.assertNotIn(name, exported, f"data root flat-exports {name}")
            self.assertFalse(hasattr(data, name), f"data root has attribute {name}")


if __name__ == "__main__":
    unittest.main()
