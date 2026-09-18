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
"""Walkthrough of the EvaluatorV2 memory estimation API, see README.md."""
from typing import Any

from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hooks.template import Template
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger


def my_attn_num_param(ccfg: Any, ctx: Any) -> float:
    """Attention parameter count that overrides the default formula."""
    del ctx
    return 10 * ccfg.h * ccfg.h


def custom(ccfg: Any) -> None:
    """Cost model variables that override the parsed configuration."""
    ccfg.bytes_compute = 1
    ccfg.s = 1024
    ccfg.n_attMM = 5


def main() -> None:
    """Estimate, inspect and customize the memory of the bundled test cases."""
    # Instantiate evaluator with a model configuration,
    #  log_level=0 removes warning messages
    e = EvaluatorV2("./test_cases/mixtral/default.yaml", log_level=0)

    # Check all defined node type
    logger.output("%s", list(LayerType))

    # Estimate peak memory (in Megabytes)
    peak_mem = e.estimate_peak(verbose=True)
    # Check whether estimation fits in device's max memory
    e.mem_fit(peak_mem)

    # Estimate static memory of a specific pipeline stage (in Megabytes)
    logger.output("%s", e.static_mem_stage(1))
    # Estimate dynamic memory of a specific pipeline stage (in Megabytes)
    logger.output("%s", e.dynamic_mem_stage(1))
    # Estimate static memory of a specific layer and stage (in Megabytes)
    logger.output("%s", e.static_mem_layer(LayerType.FULL_REC_LAYER, 1))
    # Estimate dynamic memory of a specific layer and stage (in Megabytes)
    logger.output("%s", e.dynamic_mem_layer(LayerType.FULL_REC_LAYER, 1))
    # Retrieve the memory estimation logs of a specific stage (in Megabytes)
    logger.output("%s", e.logs_mem_stage(1))
    # Fetch memory insights from each pipeline stage
    logger.output("%s", e.estimate_peak_insight())
    # PPB Input
    logger.output("%s", e.estimate_layer_memory())

    # Inspect a specific stage (here is the first one)
    e.estimate_peak(spec_stage_id=0, verbose=True)

    # Plot
    e.estimate_peak(plot=True)

    e = EvaluatorV2("./test_cases/deepseek3/default.yaml", log_level=0)

    # Overwriting context function
    e.set_attn_eval_fun(num_p=my_attn_num_param)

    # Overwriting a training feature
    e.set_passes(swap_os=True)

    # Overwriting cost model variables
    e.set_ccfg(custom)

    # Overwriting strategy
    logger.output("%s", e.get_strategy())
    e.set_strategy(dp=8, tp=8, m=128)
    logger.output("%s", e.get_strategy())

    e.estimate_peak(verbose=True)
    # Inspect ccfg object (cost model variables)
    e.print_ccfg()
    # Inspect ctx object (evaluation variables and functions)
    e.print_ctx()

    # Load a hook class
    # ... when declaring an Evaluator
    e = EvaluatorV2(
        "./test_cases/deepseek3/default.yaml", log_level=0, hook_cls=Template()
    )
    # ... by using load_hook_cls()
    e.load_hook_cls(Template())


if __name__ == "__main__":
    main()
