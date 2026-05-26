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
"""memory estimation API"""
# pylint: disable=protected-access
from __future__ import annotations
from typing import TYPE_CHECKING
import argparse
import os
import copy

# import pprint
import json
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._utils import _Utils
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._hook_manager import _HookManager
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory

if TYPE_CHECKING:
    from typing import Any, Dict, Union
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hook_base import MemEvalHook


class EvaluatorV2(_Utils, _HookManager):
    """Memory evaluator class"""

    def __init__(self, *args, **kwargs):
        self.ppb = None
        super().__init__(*args, **kwargs)
        self._child_cls = self

    def reset_config(self) -> None:
        """reset current config"""
        self.update_config(self.config_path)

    def load_hook_cls(self, hook_cls: MemEvalHook) -> None:
        """load an instance of MemEvalHook"""
        self.hook_cls = hook_cls
        self.reset_config()
        if not self._ccfg.multimodal:
            hook = list(self._ccfg.hooks_dict.values())[0]
            hook(self)

    def estimate_peak(
        self,
        stages: list = None,
        verbose: bool = False,
        spec_stage_id: int = -1,
        plot: bool = False,
    ) -> float:
        """Peak stage memory estimation"""
        original_ccfg = copy.deepcopy(self._ccfg)
        res = self._estimate_backbone(
            stages, verbose, False, spec_stage_id, plot
        )
        self._ccfg = original_ccfg
        self._ccfg.parser.ccfg = self._ccfg
        self._overhead_obj._ccfg = self._ccfg
        insights, _ = res
        stage_mems = [i["Static"] + i["Dynamic"] for i in insights]
        peak_mem = max(stage_mems)
        peak_stage = stage_mems.index(peak_mem)

        logger.output(
            "model_name: %s, peak memory : \033[1m%s MB\033[0m (stage _%s)",
            self._ccfg.model_name,
            peak_mem,
            peak_stage,
        )
        return peak_mem

    def estimate_peak_insight(self, stages: list = None) -> list:
        """subcomponents' proportion estimation"""
        original_ccfg = copy.deepcopy(self._ccfg)
        insights, _ = self._estimate_backbone(stages, False, False, -1, False)
        self._ccfg = original_ccfg
        self._ccfg.parser.ccfg = self._ccfg
        self._overhead_obj._ccfg = self._ccfg
        return insights

    def estimate_layer_memory(
        self, stages: list = None, ppb_format=1, device_type=Hard.Device_910B
    ) -> Dict:
        """PPB's input"""
        logger.info(device_type)
        if self.ppb:
            return self.ppb
        original_ccfg = copy.deepcopy(self._ccfg)
        res = self._estimate_backbone(stages, False, ppb_format, -1, False)
        self._ccfg = original_ccfg
        self._ccfg.parser.ccfg = self._ccfg
        self._overhead_obj._ccfg = self._ccfg
        _, ppb = res
        self.ppb = ppb
        return ppb

    # Specific estimation

    def static_mem_stage(self, stage_id: int) -> float:
        """stage estimation proportions"""
        insights = self.estimate_peak_insight()
        return insights[stage_id]["Static"]

    def dynamic_mem_stage(self, stage_id: int) -> float:
        """stage estimation proportions"""
        insights = self.estimate_peak_insight()
        return insights[stage_id]["Dynamic"]

    def logs_mem_stage(self, stage_id: int) -> list:
        """stage estimation hook calls trace"""
        insights = self.estimate_peak_insight()
        return insights[stage_id]["Node Log"]

    def static_mem_layer(
        self, node: Union[str, LayerType], stage_id: int
    ) -> Union[float, list]:
        """accounting all hooks for this node"""
        logs = self.logs_mem_stage(stage_id)
        extracted_log = set()
        for k in logs.keys():
            if k[3] == node.name[0]:
                extracted_log.add(logs[k]["_param"])
        extracted_log = list(extracted_log)
        if len(extracted_log) == 1:
            return extracted_log[0]
        return extracted_log

    def dynamic_mem_layer(
        self, node: Union[str, LayerType], stage_id: int
    ) -> Union[float, list]:
        """accounting all hooks for this node"""
        logs = self.logs_mem_stage(stage_id)
        extracted_log = set()
        for k in logs.keys():
            if k[3] == node.name[0]:
                extracted_log.add(logs[k]["_activ"] + logs[k]["_comm"])
        extracted_log = list(extracted_log)
        if len(extracted_log) == 1:
            return extracted_log[0]
        return extracted_log

    def mem_fit(
        self, mem: float, tolerance: float = 0, margin: float = 0
    ) -> bool:
        """check if input memory fits in device"""
        # Expect arguments to be in MB
        memory = Memory.from_mb(mem)
        tolerance_mem = Memory.from_mb(tolerance)
        margin_mem = Memory.from_mb(margin)
        cap = self._ccfg.device_capacity - margin_mem
        diff = abs(memory - cap)
        is_close = diff <= tolerance_mem
        is_fit = memory <= cap

        if Memory.zero() < tolerance_mem and is_close:
            logger.info(
                "Prediction is CLOSE to memory device (%s of diff)", diff
            )
            return True
        if is_fit:
            logger.output(
                "estimation FITS into device memory (%s<=%s-%s)",
                mem,
                self._ccfg.device_capacity,
                margin_mem,
            )
        else:
            logger.output(
                "estimation DOES NOT FIT into device memory (%s>%s-%s)",
                mem,
                self._ccfg.device_capacity,
                margin_mem,
            )
        return is_fit


def estimate_memory(config: Any) -> bool:
    """fast usage for estimation/fit for given input config"""
    e = EvaluatorV2(config)
    peak = e.estimate_peak()
    return e.mem_fit(peak)


def main():
    """commandline"""
    parser = argparse.ArgumentParser(
        description="Command line usage: Estimate peak stage memory"
    )

    parser.add_argument(
        "model_config_path",
        nargs=1,
        help="Model config file (MindFormer YAML or MindSpeed JSON)",
    )

    parser.add_argument(
        "--framework",
        default=None,
        type=str,
        help="Specify a framework name",
    )

    parser.add_argument(
        "--code-path",
        default=None,
        type=str,
        help="Specify a source code path (Additional parsing)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Show estimation trace"
    )
    parser.add_argument("--plot", action="store_true", help="Plot estimation")
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Check if estimation fits in device memory",
    )
    parser.add_argument(
        "--stage", default=-1, type=int, help="Specify pipeline stage ID"
    )
    parser.add_argument(
        "--hook",
        default=None,
        type=str,
        help="Specify hook class (defined in hooks/)",
    )
    parser.add_argument(
        "--trace-fun",
        default=None,
        type=str,
        help="Specify a formula function name to get it traced",
    )
    parser.add_argument(
        "--ppb",
        action="store_true",
        help="Generate pipeline balancing layers description",
    )
    parser.add_argument(
        "--ppb-new",
        action="store_true",
        help="Generate pipeline balancing layers description (New format)",
    )
    parser.add_argument(
        "--ctx", action="store_true", help="Show ctx variables"
    )
    parser.add_argument(
        "--ccfg", action="store_true", help="Show ccfg variables"
    )
    parser.add_argument(
        "--warnings", action="store_true", help="Show warnings"
    )
    args = parser.parse_args()

    path = args.model_config_path[0]
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"`{path}` was not found")
    if not path.endswith((".yaml", ".json",".toml")):
        raise argparse.ArgumentTypeError(f"`{path}` has invalid file type")

    e = EvaluatorV2(
        path,
        framework=args.framework,
        source_code=args.code_path,
        log_level=args.warnings,
        hook_cls=args.hook,
        trace_fun=args.trace_fun,
    )
    if args.ctx:
        e.print_ctx()
    if args.ccfg:
        e.print_ccfg()
    if args.ppb:
        ppb = e.estimate_layer_memory()
        print(json.dumps(ppb, indent=2))
    elif args.ppb_new:
        ppb = e.estimate_layer_memory(ppb_format=2)
        print(json.dumps(ppb, indent=2))
    else:
        peak_mem = e.estimate_peak(
            verbose=args.verbose, spec_stage_id=args.stage, plot=args.plot
        )

        if args.fit:
            e.mem_fit(peak_mem)


if __name__ == "__main__":
    main()
