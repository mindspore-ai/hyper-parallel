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
"""Getters"""

from copy import deepcopy
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import perf_logger as logger


# Configs are updated depending on the type of the transformer layer
def get_layer_custom_configs(cfg):
    """Stores each configuration along with how many layers are affected by it
    in ascending order of execution in a forward pass
    """

    if cfg.layer_custom_config is None or any(
        func is None for (_, func) in cfg.layer_custom_config
    ):
        return [(cfg, cfg.n_lay)]

    lccfgs = []
    for nb_layers, func in cfg.layer_custom_config:
        lccfg = deepcopy(cfg)
        func(lccfg)
        lccfgs.append((lccfg, nb_layers))

    return lccfgs


def get_recomp_factor(lccfg, layer, op_name):
    """recomputation factor"""
    if layer == LayerType.FULL_REC_LAYER:
        return 1
    if layer == LayerType.NOT_REC_LAYER:
        return 0
    if layer == LayerType.SEL_REC_LAYER:
        return getattr(lccfg.rec_op, op_name, 0)
    logger.warning("Unrecognized recompute type %s", layer)
    return 0


def get_table_quantity(lccfg, table, layer, with_recomp):
    """op compute load from given table"""
    qt_layer = 0
    for op, quantity in table.items():
        op_name = op[2:]

        qt_layer += (
            (1 + with_recomp * get_recomp_factor(lccfg, layer, op_name))
            * getattr(lccfg, op)
            * quantity
        )
    return qt_layer
