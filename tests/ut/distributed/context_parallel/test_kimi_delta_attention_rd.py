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
"""CPU checks for RD configuration and fail-before-communication contracts."""
from types import SimpleNamespace
import inspect
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.distributed.context_parallel import kimi_delta_attention_rd as rd
from hyper_parallel.models.kimi_k3.adapter.distributed import context_parallel as adapter
from tests.common.mark_utils import arg_mark


class TestCachedDoubling(unittest.TestCase):
    """Keep environment gating and invalid-input checks independent of hardware."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_transport_qualification(self):
        """Feature: RD transport selection.

        Description: Vary the runtime versions, device and explicit public override.
        Expectation: Private coalescing is selected only for the validated environment.
        """
        args = ("2.10.0", "2.10.0", "9.1.0-beta.3", "Ascend910B3", True)
        self.assertTrue(rd.select_transport(*args).direct)
        self.assertFalse(rd.select_transport(*args, force_public=True).direct)
        for index, value in ((0, "2.11.0"), (1, "2.9.0"), (2, "9.2.0"), (3, "Ascend950"), (4, False)):
            changed = list(args)
            changed[index] = value
            self.assertFalse(rd.select_transport(*changed).direct)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_inputs_fail_before_exchange(self):
        """Feature: RD input validation.

        Description: Pass incompatible peers, shape, dtype and invocation caches.
        Expectation: Invalid calls fail before any peer exchange.
        """
        with patch.object(rd.dist, "get_backend", return_value="gloo"):
            protocol = rd.CachedRecursiveDoubling(None, 2, (0, 2, 4))
            with self.assertRaisesRegex(ValueError, "ordered"):
                rd.CachedRecursiveDoubling(None, 0, (2, 0, 4))
        state = torch.zeros(1, 2, 128, 128)
        with patch.object(protocol, "exchange") as exchange:
            for matrix in (state.double(), state[..., :64], state.transpose(-1, -2)):
                with self.assertRaises(ValueError):
                    protocol.forward(state, matrix)
            for cache in ((), (state,), [state, state], (state, state.double()),
                          (state, state.transpose(-1, -2))):
                with self.assertRaises(ValueError):
                    protocol.backward(state, cache)
            exchange.assert_not_called()
        with self.assertRaisesRegex(ValueError, "outside"):
            protocol.exchange(state, 3, None)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_adapter_passes_explicit_rd_configuration(self):
        """Feature: Unified KDA state CP configuration.

        Description: Select recursive doubling while retaining older wrapper defaults.
        Expectation: The executor records RD and fixed chunk64; legacy signatures are unchanged.
        """
        mesh = SimpleNamespace(size=lambda: 4)
        with patch.object(adapter, "KimiDeltaAttentionLayerP2PCP") as executor:
            request = adapter.kimi_delta_attention_cp_wrapper(
                nn.Identity(), None, None, mesh, None, backend="triton", state_cp_method="recursive_doubling",
            )
        self.assertEqual(executor.call_args.kwargs["state_cp_method"], "recursive_doubling")
        self.assertEqual(executor.call_args.kwargs["chunk_size"], 64)
        self.assertNotIn("chunk_size", inspect.signature(adapter.kimi_delta_attention_cp_wrapper).parameters)
        for wrapper in (adapter.kimi_delta_attention_p2p_cp_wrapper, adapter.kimi_delta_attention_ulysses_cp_wrapper):
            parameters = inspect.signature(wrapper).parameters
            self.assertEqual(parameters["chunk_size"].default, 64)
            self.assertNotIn("state_cp_method", parameters)
        self.assertEqual(request.companion_attrs["_hp_kda_cp_config"]["state_cp_method"], "recursive_doubling")
