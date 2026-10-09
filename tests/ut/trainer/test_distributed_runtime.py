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
"""Unit tests for trainer distributed-runtime lifecycle helpers."""

import unittest
from unittest.mock import patch

from hyper_parallel.core.pipeline_parallel._p2p import _P2P_MULTI_STREAM_GROUPS
from hyper_parallel.trainer.runtime.distributed import destroy_process_group
from tests.common.mark_utils import arg_mark


class TestDestroyProcessGroup(unittest.TestCase):
    """Verify process-group teardown clears caches from their current owners."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    @patch("hyper_parallel.trainer.runtime.distributed.dist.is_initialized", return_value=False)
    def test_clears_pipeline_p2p_group_cache(self, mock_is_initialized) -> None:
        """Teardown imports and clears the cache from core pipeline parallel."""
        del mock_is_initialized
        _P2P_MULTI_STREAM_GROUPS[("send", 0, 1)] = object()

        destroy_process_group()

        self.assertEqual(_P2P_MULTI_STREAM_GROUPS, {})


if __name__ == "__main__":
    unittest.main()
