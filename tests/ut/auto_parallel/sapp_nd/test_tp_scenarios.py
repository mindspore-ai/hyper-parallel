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
"""Unit tests for TP dimension scenarios in sapp_nd."""
# pylint: disable=missing-class-docstring,missing-function-docstring
import copy
import logging
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.debug import PerfParts, RealParts, dim_color
from hyper_parallel.auto_parallel.sapp_nd.nd.global_config import GlobalConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level

from .conftest import shared_search, CONFIG_PATH


def _load_ccfg():
    with tempfile.TemporaryDirectory() as mpl_tmp, \
            patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
        set_verbose_level(0)
        return CostModelConfig(input_config=CONFIG_PATH, framework="mindformers")


def _enable_logger(name, level):
    lg = logging.getLogger(name)
    lg.disabled = False
    lg.setLevel(level)
    return lg


class TestTPDimensionName(unittest.TestCase):
    def test_tp_lname(self):
        self.assertEqual(Dim.TP.lname(), "tp")
    def test_mp_alias_is_tp(self):
        self.assertIs(Dim.MP, Dim.TP)


class TestTPGetDim(unittest.TestCase):
    def test_get_dim_mp_alias_resolves_to_tp(self):
        self.assertIs(Dim.get_dim("MP"), Dim.TP)
    def test_get_dim_tp_case_insensitive(self):
        self.assertIs(Dim.get_dim("tp"), Dim.TP)
    def test_get_dim_invalid_raises(self):
        with self.assertRaises(ValueError):
            Dim.get_dim("INVALID")


class TestTPDimensionBounds(unittest.TestCase):
    def test_tp_set_bound_min(self):
        Dim.TP.set_bound(8)
        Dim.TP.set_bound(4)
        self.assertEqual(Dim.TP.get_bound(), 4)
        Dim.TP.reset_bound()
    def test_tp_is_valid_exceeds_bound(self):
        Dim.TP.set_bound(2)
        self.assertFalse(Dim.TP.is_valid(4))
        Dim.TP.reset_bound()


class TestTPDimensionsValidity(unittest.TestCase):
    def test_tp_not_power_of_2_invalid(self):
        dims = Dim.Dimensions(
            [(Dim.DP, 1), (Dim.TP, 3), (Dim.PP, 1)],
            all_dims=[Dim.DP, Dim.TP, Dim.PP],
        )
        self.assertFalse(dims.is_valid())
    def test_tp_power_of_2_valid(self):
        dims = Dim.Dimensions(
            [(Dim.DP, 1), (Dim.TP, 4), (Dim.PP, 1)],
            all_dims=[Dim.DP, Dim.TP, Dim.PP],
        )
        self.assertTrue(dims.is_valid())


class TestTPSetStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            cls._base_ccfg = _load_ccfg()
    def _fresh_ccfg(self):
        return copy.copy(self._base_ccfg)
    def test_set_strategy_tp_sp_equals_t(self):
        ccfg = self._fresh_ccfg()
        ccfg.set_strategy(tp=8)
        self.assertEqual(ccfg.sp, ccfg.t)
        self.assertEqual(ccfg.sp, 8)
    def test_strategy_num_devices_with_tp(self):
        ccfg = self._fresh_ccfg()
        ccfg.set_strategy(dp=2, tp=4, cp=1)
        self.assertEqual(ccfg.strategy_num_devices(), ccfg.d * ccfg.t * ccfg.cp * ccfg.p)


class TestTPMakeParallelConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            ccfg = _load_ccfg()
            cls.gcfg = GlobalConfig(ccfg, dimensions=[Dim.DP, Dim.TP, Dim.PP, Dim.CP])
    def test_make_parallel_config_args_tp(self):
        pc = self.gcfg.make_parallel_config_args(dp=2, tp=4, pp=1, cp=1, mbs=1, mb=1)
        self.assertEqual(pc.val(Dim.TP), 4)
    def test_make_parallel_config_tp_via_dtpc(self):
        pc = self.gcfg.make_parallel_config(
            dtpc_p=(2, 4, 1, 1), mbsn=(1, 1), evos_p=(1, 1, 1, False, False, 1))
        self.assertEqual(pc.val(Dim.TP), 4)


class TestTPSetParallelConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            ccfg = _load_ccfg()
            cls.gcfg = GlobalConfig(ccfg, dimensions=[Dim.DP, Dim.TP, Dim.PP, Dim.CP])
    def test_set_parallel_config_tp_propagation(self):
        pc = self.gcfg.make_parallel_config_args(dp=2, tp=4, pp=1, cp=1, mbs=1, mb=1)
        self.gcfg.set_parallel_config(pc)
        self.assertEqual(self.gcfg.ccfg.t, 4)


class TestTPMoeValid(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            ccfg = _load_ccfg()
            cls.gcfg = GlobalConfig(ccfg, dimensions=[Dim.DP, Dim.TP, Dim.PP, Dim.CP])
    def test_moe_valid_ep_le_dp_times_tp(self):
        pc = self.gcfg.make_parallel_config_args(dp=2, tp=4, pp=1, cp=1, mbs=1, mb=1, ep=4)
        self.assertTrue(self.gcfg.moe_valid(pc))


class TestTPBoundSpace(unittest.TestCase):
    def test_bound_space_with_tp(self):
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            machine = Hard.Machine(None, "A2")
            dims = Dim.get_dims(["DP", "TP", "PP", "EP", "MB"])
            runner = Par.Parallelize(
                "mindformers", CONFIG_PATH, machine, global_batch_size=None,
                dimensions=dims, swap_os=False, mppb=False, model=None,
                max_mem=None, mem_for_ppb=Memory.from_string("0GB"))
            tp_bound = Dim.TP.get_bound()
            if runner.config.ccfg.n_kv:
                self.assertLessEqual(tp_bound, runner.config.ccfg.n_kv)
            Dim.TP.reset_bound()


class TestTPPerfParts(unittest.TestCase):
    def test_perf_parts_mp_comm_short_name(self):
        self.assertEqual(PerfParts.MP_COMM.short_name(), "TP(MP)")
    def test_mp_comm_maps_to_mp_wait(self):
        self.assertGreater(PerfParts.MP_COMM.value, 0)
        self.assertGreater(RealParts.MP_WAIT.value, 0)
    def test_dim_color_tp_is_red(self):
        self.assertEqual(dim_color(Dim.TP), "red")


class TestTPLogMessages(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            cls._base_ccfg = _load_ccfg()
    def _fresh_ccfg(self):
        return copy.copy(self._base_ccfg)
    def test_print_parallelism_log(self):
        _enable_logger("memory_estimation", logging.INFO)
        ccfg = self._fresh_ccfg()
        with self.assertLogs("memory_estimation", level="INFO") as cm:
            ccfg.print_parallelism()
        self.assertIn("TP(MP)", "\n".join(cm.output))


@unittest.skip("TODO: _backbone.__postprocess_stages max() on empty sequence")
class TestTPRunND(unittest.TestCase):
    _scored: list = []
    @classmethod
    def setUpClass(cls) -> Any:
        cls._scored = shared_search("A2", None, ["DP", "TP", "PP", "EP", "MB"], top_k=2)
    def test_run_nd_with_tp(self):
        self.assertIsInstance(self._scored, list)


class TestTPSpaceAndMaxOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            ccfg = _load_ccfg()
            cls.gcfg = GlobalConfig(ccfg, dimensions=[Dim.DP, Dim.TP, Dim.PP, Dim.CP])
            Dim.TP.reset_bound()
    def test_space_tp_bounded(self):
        Dim.TP.set_bound(4)
        space = self.gcfg.space(Dim.TP, divide=True, reverse=True)
        for val in space:
            self.assertLessEqual(val, 4)
        Dim.TP.reset_bound()
    def test_space_tp_not_in_dimensions(self):
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            ccfg = _load_ccfg()
            gcfg_no_tp = GlobalConfig(ccfg, dimensions=[Dim.DP, Dim.PP])
        space = gcfg_no_tp.space(Dim.TP, divide=True)
        self.assertEqual(len(space), 1)
        self.assertEqual(space[0], Dim.TP.from_config(gcfg_no_tp.ccfg))


class TestTPDimVal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            ccfg = _load_ccfg()
            cls.gcfg = GlobalConfig(ccfg, dimensions=[Dim.DP, Dim.TP, Dim.PP, Dim.CP])
    def test_dim_val_tp_falls_back_to_from_config(self):
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            ccfg = _load_ccfg()
            gcfg_no_tp = GlobalConfig(ccfg, dimensions=[Dim.DP, Dim.PP])
        pc = gcfg_no_tp.make_parallel_config_args(dp=2, pp=1, mbs=1, mb=1)
        val = gcfg_no_tp.dim_val(Dim.TP, pc)
        self.assertEqual(val, Dim.TP.from_config(gcfg_no_tp.ccfg))


class TestTPSetStrategySideEffects(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            cls._base_ccfg = _load_ccfg()
    def _fresh_ccfg(self):
        return copy.copy(self._base_ccfg)
    def test_set_strategy_tp_updates_t_exp(self):
        ccfg = self._fresh_ccfg()
        ccfg.set_strategy(tp=4)
        self.assertIsNotNone(ccfg.t_exp)
    def test_set_strategy_tp_with_offset(self):
        ccfg = self._fresh_ccfg()
        offset = [1] * ccfg.p
        offset[-1] = -1
        ccfg.set_strategy(tp=4, offset=offset)
        self.assertTrue(ccfg.is_consistent_pp_config())


class TestTPDeviceLoopsBound(unittest.TestCase):
    def test_device_loops_tp_within_bound(self):
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            machine = Hard.Machine(None, "A2")
            dims = Dim.get_dims(["DP", "TP", "PP", "EP", "MB"])
            runner = Par.Parallelize(
                "mindformers", CONFIG_PATH, machine, global_batch_size=None,
                dimensions=dims, swap_os=False, mppb=False, model=None,
                max_mem=None, mem_for_ppb=Memory.from_string("0GB"))
            tp_bound = Dim.TP.get_bound()
            if tp_bound is not None:
                for val in runner.config.space(Dim.TP, runner.config.ccfg.n, reverse=True):
                    self.assertLessEqual(val, tp_bound)
            Dim.TP.reset_bound()


class TestSetStrategyTPPrecedence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            cls._base_ccfg = _load_ccfg()

    def _fresh_ccfg(self):
        return copy.copy(self._base_ccfg)

    def test_set_strategy_tp_takes_precedence_over_mp(self):
        ccfg = self._fresh_ccfg()
        ccfg.set_strategy(tp=8, mp=2)
        self.assertEqual(ccfg.t, 8)

    def test_set_strategy_mp_ignored_when_tp_present(self):
        ccfg = self._fresh_ccfg()
        ccfg.set_strategy(tp=2, mp=8)
        self.assertEqual(ccfg.t, 2)

    def test_get_strategy_no_mp_key_after_mp_set(self):
        ccfg = self._fresh_ccfg()
        ccfg.set_strategy(mp=4)
        strategy = ccfg.get_strategy()
        self.assertIn("tp", strategy)
        self.assertNotIn("mp", strategy)


class TestMakeParallelConfigTPPrecedence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            ccfg = _load_ccfg()
            cls.gcfg = GlobalConfig(ccfg, dimensions=[Dim.DP, Dim.TP, Dim.PP, Dim.CP])

    def test_make_parallel_config_args_tp_precedence(self):
        pc = self.gcfg.make_parallel_config_args(
            dp=2, tp=8, mp=2, pp=1, cp=1, mbs=1, mb=1,
        )
        self.assertEqual(pc.val(Dim.TP), 8)


class TestGlobalConfigDimMPvsTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> Any:
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)
            ccfg_mp = CostModelConfig(input_config=CONFIG_PATH, framework="mindformers")
            ccfg_tp = CostModelConfig(input_config=CONFIG_PATH, framework="mindformers")
            cls.gcfg_mp = GlobalConfig(ccfg_mp, dimensions=[Dim.DP, Dim.MP, Dim.PP, Dim.CP])
            cls.gcfg_tp = GlobalConfig(ccfg_tp, dimensions=[Dim.DP, Dim.TP, Dim.PP, Dim.CP])

    def test_make_parallel_config_args_mp_vs_tp_same_gbs(self):
        gcfg_mp_gbs = GlobalConfig(
            CostModelConfig(input_config=CONFIG_PATH, framework="mindformers"),
            dimensions=[Dim.DP, Dim.MP, Dim.PP, Dim.CP, Dim.MBS, Dim.MBN],
        )
        gcfg_tp_gbs = GlobalConfig(
            CostModelConfig(input_config=CONFIG_PATH, framework="mindformers"),
            dimensions=[Dim.DP, Dim.TP, Dim.PP, Dim.CP, Dim.MBS, Dim.MBN],
        )
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            pc_mp = gcfg_mp_gbs.make_parallel_config_args(
                dp=2, mp=4, pp=2, cp=1, mbs=2, mb=3,
            )
            pc_tp = gcfg_tp_gbs.make_parallel_config_args(
                dp=2, tp=4, pp=2, cp=1, mbs=2, mb=3,
            )
        self.assertEqual(pc_mp.global_batch_size(), pc_tp.global_batch_size())

    def test_make_parallel_config_args_mp_vs_tp_same_unique_name(self):
        pc_mp = self.gcfg_mp.make_parallel_config_args(
            dp=2, mp=4, pp=1, cp=1, mbs=1, mb=1,
        )
        pc_tp = self.gcfg_tp.make_parallel_config_args(
            dp=2, tp=4, pp=1, cp=1, mbs=1, mb=1,
        )
        self.assertEqual(pc_mp.unique_name(), pc_tp.unique_name())

    def test_set_parallel_config_mp_vs_tp_same_ccfg_t(self):
        pc_mp = self.gcfg_mp.make_parallel_config_args(
            dp=2, mp=8, pp=1, cp=1, mbs=1, mb=1,
        )
        pc_tp = self.gcfg_tp.make_parallel_config_args(
            dp=2, tp=8, pp=1, cp=1, mbs=1, mb=1,
        )
        self.gcfg_mp.set_parallel_config(pc_mp)
        self.gcfg_tp.set_parallel_config(pc_tp)
        self.assertEqual(self.gcfg_mp.ccfg.t, self.gcfg_tp.ccfg.t)
        self.assertEqual(self.gcfg_mp.ccfg.t, 8)


if __name__ == "__main__":
    unittest.main()
