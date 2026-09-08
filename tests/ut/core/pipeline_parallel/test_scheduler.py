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
"""Unit tests for the pipeline-schedule name resolver + shared-parameter info."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# pylint: disable=C0413
import torch
from torch import nn

from hyper_parallel.core.pipeline_parallel.stage import PipelineStage, SharedParameterInfo


class TestSharedParameterInfoReFetch(unittest.TestCase):
    """``SharedParameterInfo`` re-fetches the live parameter from its owner.

    Needed for the meta-init pipeline path, where the stage is materialized
    (``to_empty`` replaces the ``nn.Parameter`` object) *after* the shared info
    is created — a stale capture would have ``grad=None`` at sync time.
    """

    def test_refetches_current_param_from_owner(self):
        """With ``owner_module``/``param_name``, ``parameter`` tracks replacement."""
        module = nn.Linear(4, 4)
        original = module.weight
        info = SharedParameterInfo(module.weight, [0, 1], owner_module=module, param_name="weight")
        assert info.parameter is original, "should start at the captured parameter"
        # Simulate ``to_empty`` swapping the Parameter object in place.
        module.weight = nn.Parameter(torch.zeros(4, 4))
        assert info.parameter is module.weight, "must re-fetch the live parameter"
        assert info.parameter is not original, "must not return the stale parameter"

    def test_static_capture_without_owner(self):
        """Without ``owner_module`` the captured parameter object is returned as-is."""
        module = nn.Linear(4, 4)
        original = module.weight
        info = SharedParameterInfo(module.weight, [0, 1])
        module.weight = nn.Parameter(torch.zeros(4, 4))
        assert info.parameter is original, "no owner -> keep the captured parameter"


class TestSyncSharedParametersGradGuards(unittest.TestCase):
    """``sync_shared_parameters_grad`` guards and zero-fill semantics.

    A shared info whose materialization handshake has not completed has
    ``group=None`` and must skip the collective. A tied end whose backward has
    not run has ``grad=None`` but must still ENTER the collective with a zero
    contribution — skipping would desync the peers' matching ``all_reduce``.
    """

    @staticmethod
    def _make_info(grad, group, requires_grad=True):
        """Build a ``SharedParameterInfo`` with a given grad and group."""
        param = nn.Parameter(torch.zeros(2, 2), requires_grad=requires_grad)
        param.grad = grad
        info = SharedParameterInfo(param, [0, 1])
        info.group = group
        return info

    def _call(self, info):
        """Invoke the unbound method on a duck-typed stage that owns ``info``."""
        fake_stage = SimpleNamespace(_shared_parameters=[info], _has_backward=True)
        with patch("hyper_parallel.core.pipeline_parallel.stage.platform") as mock_platform:
            mock_platform.all_reduce.return_value = (torch.zeros(2, 2), None)
            PipelineStage.sync_shared_parameters_grad(fake_stage)
        return mock_platform

    def test_zero_fills_when_grad_is_none(self):
        """A ``None`` grad still enters the collective with a zero contribution."""
        info = self._make_info(grad=None, group="pp_group")
        mock_platform = self._call(info)
        mock_platform.full_like.assert_called_once()
        assert mock_platform.all_reduce.call_count == 1, \
            (f"a grad-less shared end must still all-reduce (zeros), "
             f"got {mock_platform.all_reduce.call_count} calls")

    def test_skips_when_group_is_none(self):
        """A ``None`` group (handshake not yet run) must not reach ``all_reduce``."""
        info = self._make_info(grad=torch.ones(2, 2), group=None)
        mock_platform = self._call(info)
        mock_platform.all_reduce.assert_not_called()

    def test_skips_when_frozen(self):
        """``requires_grad=False`` peers all skip (tied/frozen is group-wide)."""
        info = self._make_info(grad=None, group="pp_group", requires_grad=False)
        mock_platform = self._call(info)
        mock_platform.all_reduce.assert_not_called()

    def test_reduces_when_grad_and_group_ready(self):
        """A ready entry reduces its grad exactly once over its group."""
        grad = torch.ones(2, 2)
        info = self._make_info(grad=grad, group="pp_group")
        mock_platform = self._call(info)
        assert mock_platform.all_reduce.call_count == 1, \
            (f"ready shared grad should all-reduce once, "
             f"got {mock_platform.all_reduce.call_count} calls")
        reduced = mock_platform.all_reduce.call_args.args[0]
        assert reduced is grad, \
            (f"all_reduce should receive the live grad tensor, "
             f"got {type(reduced)}")


if __name__ == "__main__":
    unittest.main()
