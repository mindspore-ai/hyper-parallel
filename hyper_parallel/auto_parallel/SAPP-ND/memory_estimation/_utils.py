# Copyright 2025 Huawei Technologies Co., Ltd
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
"""getters, setters, printers"""
from __future__ import annotations
from typing import TYPE_CHECKING

from memory_estimation._backbone import _Backbone
from memory_estimation.logger import logger

if TYPE_CHECKING:
    from typing import Dict, Union, Tuple


class _Utils(_Backbone):
    """utils class"""

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def get_model_name(self) -> str:
        """accessor"""
        return self._ccfg.model_name

    def get_strategy(self) -> Dict:
        """return parallelism/recompute strategies"""
        return self._ccfg.get_strategy()

    def get_max_device_memory(self) -> float:
        """accessor for max device memory in MB"""
        return self._ccfg.device_capacity.to_mb().size

    def get_num_layers(self) -> Union[Tuple, int]:
        """tuple of all L if multimodal"""
        if not self._ccfg.multimodal:
            return self._ccfg.n_lay + self._ccfg.n_mtp
        return [
            self._ccfg.mm_ccfgs[m].n_lay + self._ccfg.mm_ccfgs[m].n_mtp
            for m in self._ccfg.mm_order
        ]

    def set_layer_custom(self, lc=None) -> None:
        """setting ccfg.layer_custom_config (inner call only)"""
        if not lc:
            self._ccfg.layer_custom_config = [(self._ccfg.n_lay, None)]
        elif isinstance(lc, list):
            self._ccfg.layer_custom_config = lc

    def set_config(self, config) -> None:
        """Explicitly Assign a new config ccfg"""
        self._ccfg = config

    def all_stage_micro_factors(self) -> None:
        """get all stage's warmup micros for current schedule"""
        sched = self._ccfg.pp_sched
        for stage_id in range(self._ccfg.p):
            chunk_micro = []
            for chunk_id in range(self._ccfg.vp):
                self._ctx.current_stage_id = stage_id
                self._ctx.current_chunk_id = chunk_id
                micro = self._ctx.pp_micro_eval[sched](self._ccfg, self._ctx)
                chunk_micro += [micro]
            logger.info(
                "%s stage _%s = %s",
                self._ctx.pp_micro_eval[sched].__name__,
                stage_id,
                str(chunk_micro),
            )

    # Printers

    def print_ccfg(self) -> None:
        """pretty printer for ccfg"""
        if not self._ccfg.multimodal:
            print(self._ccfg)
        else:
            for m in self._ccfg.mm_order:
                print("Module", m)
                print(self._ccfg.mm_ccfgs[m])

    def print_ctx(self) -> None:
        """pretty printer for ctx"""
        print("Eval Context attributes:\n" + str(self._ctx))

    def print_node_eval(self) -> None:
        """get all (P,stat,dyn)"""
        return self._ctx.print_node_eval()

    def print_stages(self, stages: list, spec_stage_id: int = -1) -> None:
        """can target a stage id"""
        self._ccfg.print_stages(stages, spec_stage_id)
