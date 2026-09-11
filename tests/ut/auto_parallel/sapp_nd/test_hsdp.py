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
"""Unit tests for HSDP dimension and cost-model variables."""
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest
from types import SimpleNamespace
from typing import Any

from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.common._cost_model_variables import (
    _CostModVar,
)

from ._test_helpers import _ConcreteParser


class TestHSDPDimension(unittest.TestCase):

    def test_hsdp_dimension_properties(self):
        self.assertEqual(Dim.HSDP.name, "HSDP")
        self.assertEqual(Dim.HSDP.cost_model_var, "d_shard")
        self.assertEqual(Dim.HSDP.default, 1)

    def test_hsdp_d_shard_exceeds_dp_is_invalid(self):
        dims = Dim.Dimensions(
            [
                (Dim.DP, 4),
                (Dim.EP, 1),
                (Dim.TP, 2),
                (Dim.CP, 1),
                (Dim.PP, 1),
                (Dim.VPP, 1),
                (Dim.MBN, 1),
                (Dim.MBS, 8),
                (Dim.SP, False),
                (Dim.OP, 1),
                (Dim.FSDP, True),
                (Dim.HSDP, 8),
            ],
            all_dims=Dim.ALL_DIMS.copy(),
        )
        self.assertFalse(dims.is_valid())

    def test_hsdp_d_shard_less_than_dp_is_valid(self):
        dims = Dim.Dimensions(
            [
                (Dim.DP, 8),
                (Dim.EP, 1),
                (Dim.TP, 2),
                (Dim.CP, 1),
                (Dim.PP, 1),
                (Dim.VPP, 1),
                (Dim.MBN, 1),
                (Dim.MBS, 8),
                (Dim.SP, False),
                (Dim.OP, 1),
                (Dim.FSDP, True),
                (Dim.HSDP, 4),
            ],
            all_dims=Dim.ALL_DIMS.copy(),
        )
        self.assertTrue(dims.is_valid())


class TestHSDPCostModelVariables(unittest.TestCase):

    def test_hsdp_default_values(self):
        ccfg = _CostModVar(None, None, None, None)
        self.assertEqual(ccfg.d_shard, 1)
        self.assertEqual(ccfg.d_replicate, 1)
        self.assertEqual(ccfg.comm_hsdp, 0)


class TestHSDPConfigFsdpShard(unittest.TestCase):

    def _make_ccfg(self, **kwargs: Any) -> SimpleNamespace:
        defaults = {
            "config": None,
            "d": 8,
            "t": 2,
            "cp": 1,
            "ep": 1,
            "n_exp": 1,
            "d_exp": 8,
            "t_exp": 2,
            "d_shard": 4,
            "has_op": True,
            "has_grad_shard": False,
            "fsdp": True,
            "os_max_shard": 16,
            "sp": 1,
            "shard_grad_exp": 0,
            "shard_grad_exp_partial": 0,
        }
        defaults.update(kwargs)
        defaults["d_shard_or_d"] = defaults["d_shard"] if defaults["d_shard"] > 0 else defaults["d"]
        return SimpleNamespace(**defaults)

    def test_hsdp_shard_non_exp_uses_d_shard(self):
        ccfg = self._make_ccfg(d=8, d_shard=4, t=2, cp=1)
        parser = _ConcreteParser(ccfg)
        parser.config_fsdp_shard(ccfg)
        self.assertEqual(ccfg.shard_p_fsdp_non_exp, 4 * 1 * 2)
        self.assertNotEqual(ccfg.shard_p_fsdp_non_exp, 8 * 1 * 2)

    def test_hsdp_d_replicate_set(self):
        ccfg = self._make_ccfg(d=8, d_shard=4)
        parser = _ConcreteParser(ccfg)
        parser.config_fsdp_shard(ccfg)
        self.assertEqual(ccfg.d_replicate, 2)


class TestHSDPConfigCommFlag(unittest.TestCase):

    def _make_ccfg(self, **kwargs: Any) -> SimpleNamespace:
        defaults = {
            "config": None,
            "d": 8,
            "t": 2,
            "cp": 1,
            "ep": 1,
            "n_exp": 1,
            "d_exp": 8,
            "t_exp": 2,
            "d_shard": 4,
            "has_op": True,
            "has_grad_shard": True,
            "fsdp": True,
            "os_max_shard": 16,
            "sp": 1,
            "shard_grad_exp": 0,
            "shard_grad_exp_partial": 0,
        }
        defaults.update(kwargs)
        defaults["d_shard_or_d"] = defaults["d_shard"] if defaults["d_shard"] > 0 else defaults["d"]
        return SimpleNamespace(**defaults)

    def test_comm_hsdp_set_when_hsdp_active(self):
        ccfg = self._make_ccfg(fsdp=True, d=8, d_shard=4)
        parser = _ConcreteParser(ccfg)
        parser.config_comm_flag(ccfg)
        self.assertEqual(ccfg.comm_hsdp, 1.0)


class TestHSDPSetStrategy(unittest.TestCase):

    def test_set_strategy_d_shard_kwarg(self):
        from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig

        parser = SimpleNamespace(
            config_shard_emb=lambda: None,
            config_dp_tp_exp=lambda cfg: None,
            config_fsdp_shard=lambda cfg: None,
            config_comm_flag=lambda cfg: None,
        )
        cost_cfg = CostModelConfig()
        cost_cfg.__dict__.update(
            vars(_CostModVar(None, None, None, None))
        )
        cost_cfg.parser = parser
        cost_cfg.multimodal = False
        cost_cfg.hooks_dict = {}
        cost_cfg.is_consistent_pp_config = lambda: True

        cost_cfg.set_strategy(
            d=8, mp=2, pp=1, cp=1, mbs=8, mb=1,
            ep=1, vp=1, op=1, sp=1, fsdp=True, d_shard=4,
        )
        self.assertEqual(cost_cfg.d_shard, 4)


if __name__ == "__main__":
    unittest.main()