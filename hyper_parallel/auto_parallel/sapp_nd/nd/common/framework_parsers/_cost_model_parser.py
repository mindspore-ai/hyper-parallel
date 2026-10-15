# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""cost model parser module"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple

import logging
from abc import ABC
from abc import abstractmethod

from hyper_parallel.auto_parallel._layer_stack import derive_layers, resolve_layers
from hyper_parallel.auto_parallel._op_profiles import infer_arch, load_op_profile, resolve_ops
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger

# What a parse says a run should read: the memory logger is off below -v 4.
parse_logger = logging.getLogger(__name__)

# The optimizer a config that states none is priced as, with a warning: the
# shipped recipes state AdamW and Muon both, so none is assumed silently.
DEFAULT_OPTIMIZER = "adamw"

if TYPE_CHECKING:
    from typing import Mapping, Optional
    from hyper_parallel.auto_parallel._layer_stack import LayerStack
    from hyper_parallel.auto_parallel._model_spec import OpCounts
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import _CostModVar


class _CostModelParser(ABC):
    """abstract parser class"""

    def __init__(self, ccfg: _CostModVar):
        self.ccfg = ccfg
        self.config = ccfg.config

    @abstractmethod
    def parse(self):
        """Parse the cost model configuration and populate the cost model variables.

        Subclasses must implement this method to read framework-specific
        configuration values into the shared _CostModVar instance.
        """

    @staticmethod
    def stated_optimizer(stated: Any) -> Tuple[str, int]:
        """Name the optimizer a config states, and the states it keeps a parameter, which its name decides.

        Muon keeps one momentum a matrix and AdamW two moments a parameter.
        A config that states no optimizer is priced as AdamW, and the parser
        says so: the HyperParallel parser named one silently and charged its
        two states, which a Muon run described without its optimizer does
        not keep, and the MindFormers parser failed on the missing section
        (M3). The name is what the state count is read from (M5).

        Returns:
            The optimizer's name and the states it keeps a parameter.
        """
        if not stated:
            parse_logger.warning("the config states no optimizer: it is priced as %s, two states a parameter",
                                 DEFAULT_OPTIMIZER)
            stated = DEFAULT_OPTIMIZER
        name = str(stated)
        return name, 1 if "muon" in name.lower() else 2

    @staticmethod
    def state_optimizer(ccfg: Any, stated: Any) -> None:
        """Set the optimizer a config states and its state count on *ccfg* (``stated_optimizer``)."""
        ccfg.optimizer, ccfg.optimizer_states = _CostModelParser.stated_optimizer(stated)

    def config_op_counts(
        self,
        ccfg: _CostModVar,
        arch: Optional[str] = None,
        ops: Optional[Mapping[str, OpCounts]] = None,
    ) -> None:
        """Settle the op profile the model is priced with.

        A parser that reads a declared arch passes it, with any op counts the
        spec declares; one that has only a model name leaves the family to
        :func:`infer_arch`.
        """
        ccfg.arch = arch or infer_arch(ccfg.model_name)
        ccfg.op_counts = resolve_ops(ccfg.arch, ops)

    @staticmethod
    def config_layer_stack(ccfg: _CostModVar, stack: Optional[LayerStack] = None) -> None:
        """Settle the layer stack the estimators price.

        Args:
            ccfg: The config, with its arch, op counts and layer counts parsed.
            stack: The stack the model states.  By default, the one its layer
                counts imply: dense layers first, two halves, or one kind.
        """
        if stack is None:
            layers = derive_layers(
                load_op_profile(ccfg.arch), int(ccfg.n_lay), int(ccfg.n_mtp or 0),
                first_k_dense=int(ccfg.k_1st_dense or 0),
            )
            stack = resolve_layers(ccfg.arch, layers, ccfg.op_counts)
        ccfg.layer_stack = stack
        if len(stack.groups) > 1:
            logger.info(
                "%s layer stack: %s", ccfg.model_name,
                ", ".join(f"{group.count}x{group.kind.name}" for group in stack.groups),
            )

    # def config_op_level(self, ccfg, strategy):
    #     def full_partial():
    #         return Config({"full":0, "partial":0})
    #     def exp_or_not():
    #         return Config({
    #             "non_exp":full_partial(),
    #             "exp":full_partial()
    #         })
    #     ccfg.op = Config({
    #         "p":exp_or_not(),
    #         "os":exp_or_not(),
    #         "grad"exp_or_not()
    #     })
    #     shard_strat = {
    #         "grad":0, #zero 1
    #         "os+grad":0, #zero 2
    #         "p+os+grad":0, # zero 3
    #         "p+os":0 # zero2 mindspore
    #     }
    #     shard_strat[strategy]

    def init_hff(self):
        """MindFormers format for FFn hidden size"""
        # Assuming following 3 variables are already parsed
        hidden_size = self.ccfg.h
        ffn_dim_multiplier = self.ccfg.fdm
        multiple_of = self.ccfg.multiple_of
        hff = 4 * hidden_size
        if ffn_dim_multiplier:
            hff = int((ffn_dim_multiplier + 0.01) * hff)
        hff = int(2 * hff / 3)
        hff = multiple_of * ((hff + multiple_of - 1) // multiple_of)
        return hff
