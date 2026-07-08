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
"""cost model variables"""
from __future__ import annotations
from typing import TYPE_CHECKING

import importlib
import ast
import os
from dataclasses import dataclass
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger

if TYPE_CHECKING:
    from typing import Any, Optional, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
MAPPING_YML = os.path.join(current_dir, "framework_parsers/mapping.yaml")


@dataclass
class _CostModVar:
    """cost model variables class"""

    config: any = None
    config_format: str = None
    multimodal: bool = False
    model_name: str = None
    device_capacity: Memory = Memory.zero()  # float = 0
    mm_ccfgs: any = None
    mm_order: list = None
    layer_custom_config: list = None
    overwrite_eval_functions: dict = None
    parser: any = None

    # Strategy
    d: float = 0
    t: float = 0
    p: float = 0
    cp: float = 0
    ep: float = 0
    sp: float = 0
    vp: float = 0
    os_max_shard: float = 0
    op_weight_shard: float = 0
    offset: Union[list, int] = None
    full_rec: Union[list, bool] = None
    sel_rec: Union[list, bool] = None
    pp_sched: str = None
    n_s_split: float = 0
    cp_algo: str = "colossalai_cp"
    rec_op: any = None
    pp_partition: list = None

    # hyperparameters
    h: float = 0
    hff: float = 0
    v: float = 0
    s: float = 0
    s_fa: float = 0
    a: float = 0
    n_lay: float = 0
    n_kv: float = 0
    dh: float = 0
    dc_kv: float = 0
    dc_q: float = 0
    dhr: float = 0
    k_1st_dense: float = 0
    n_mtp: float = 0
    is_mtp_in_offset: bool = True
    multiple_of: float = 0
    fdm: float = 0

    # MoE
    t_exp: float = 0
    d_exp: float = 0
    hff_exp: float = 0
    n_exp: float = 0
    n_chosen_exp: float = 0
    n_shared_exp: float = 0
    cap_fact: float = 0
    etp: float = 0
    tokens_per_expert: list = None  # per-expert token count per mb (all EP, pre a2a); None=balanced

    # CP modeling
    kv_lora_rank: float = 0
    attention_type: str = None
    device_per_node: float = 8
    bw_intra: float = 400.0
    bw_inter: float = 25.0
    sp_enabled: bool = False

    # ZeRO
    shard_p_os_non_exp_partial: float = 0
    shard_p_os_non_exp: float = 0
    shard_grad_non_exp: float = 0
    shard_p_os_exp_partial: float = 0
    shard_p_os_exp: float = 0
    shard_grad_exp: float = 0
    shard_grad_exp_partial: float = 0

    # comm flag
    comm_d_non_exp: float = 0
    comm_d_exp: float = 0
    comm_t: float = 0
    comm_ep: float = 0
    comm_cp: float = 0
    # Transitional comm overlap correction (see comm_time.py).
    # Fraction of comm volume hidden behind compute, applied as
    #   comm[dim] *= (1 - overlap_dim)
    # Applies to both FLOP and TIME paths; the TIME path's
    # estimate_comm_score(overlap=...) call site is zeroed so this is the
    # single source of overlap.
    # Defaults (dp=0.9, tp=0.5) are validated on MindFormers: they are the
    # overlap that made the model match real MindFormers step times.  Raw
    # volume over-counts because communication really does overlap with
    # compute.  Re-validating for the hyper-parallel target is a follow-up.
    # Follow-up: source from hardware profiling, then fold into
    # estimate_comm_score (which also needs DP dedup, EP/CP terms, latency).
    comm_dp_overlap: float = 0.9
    comm_tp_overlap: float = 0.5

    # feature flag
    has_op: bool = False
    has_grad_shard: bool = False
    freeze: bool = False
    has_fa: bool = False
    # vp_less_mem: bool = False
    has_clip: bool = False
    gmm: bool = False
    vocab_emb_dp: float = 0
    tie_emb_out: bool = False
    emb_out_in_offset: bool = False

    # batch
    b: float = 0
    m: float = 0
    gbs: float = 0

    # shard
    shard_embed: float = 0
    shard_output_activ: float = 0
    shard_recompute_input: float = 0
    is_shard_mtp_param: bool = True

    # bytes
    bytes_p: float = 0
    bytes_compute: float = 0
    bytes_softmax: float = 0
    bytes_grad: float = 0
    bytes_os: float = 0
    bytes_norm: float = 0

    def __init__(self, input_config: Any, hook_cls: Any, framework: Optional[str], source_code: Optional[str]) -> None:
        """Initialise from a config path and optional hooks/framework/source."""
        super().__init__()
        if input_config:
            self.update_config(input_config, hook_cls, framework, source_code)

    def _load_parser_cls(self, module_name):
        """hook_class in eval yaml"""
        target_mod_path = None
        try:
            # search in folder 'framework_parsers'
            fram_dir = os.path.join(current_dir, "framework_parsers")
            for f in os.listdir(fram_dir):
                if f.endswith(".py"):
                    mod_path = f"hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.{f.split('.')[0]}"
                    spec = importlib.util.find_spec(mod_path)
                    if spec is None or spec.origin is None:
                        continue
                    with open(spec.origin, "r", encoding="utf-8") as mf:
                        source = mf.read()
                        tree = ast.parse(source)
                        mod_cls = None
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef) and node.name == module_name:
                                mod_cls = node
                                break
                        if mod_cls:
                            target_mod_path = mod_path
                            break
            if target_mod_path:
                module = importlib.import_module(target_mod_path)
                return getattr(module, module_name)
        except (ModuleNotFoundError, ImportError) as e:
            print(e)
        return None

    def get_framework_parser_naive(self, input_config: str) -> Optional[Any]:
        """Return parser class based on file extension (naive heuristic)."""
        mod_name = None
        if isinstance(input_config, str):
            if input_config.endswith("yaml"):
                mod_name = "CostModelParserMindformers"
            if input_config.endswith("json"):
                mod_name = "CostModelParserMindspeed"
            if input_config.endswith("toml"):
                mod_name = "CostModelParserHyperparallel"
            if not mod_name:
                raise AttributeError(f"Unhandled input format '{input_config}'")
            return self._load_parser_cls(mod_name)
        return None

    def get_framework_parser(self, framework: str) -> Any:
        """Look up and return the parser class for the given framework name.

        Uses the mapping YAML file to find the corresponding parser module
        class name, then loads and returns it. Raises AttributeError if the
        framework name is not found in the mapping.
        """
        yml = Config(MAPPING_YML)
        mod_name = next((e["module"] for e in yml.framework_parser if e["name"] == framework), None)
        if not mod_name:
            raise AttributeError(f"Cannot find parser module name from arg '{framework}'")
        return self._load_parser_cls(mod_name)

    def update_config(
        self,
        input_config: Any,
        hook_cls: Any = None,
        framework: Optional[str] = None,
        source_code: Optional[str] = None,
    ) -> None:
        """process input config"""
        self.hooks_dict = None if not hook_cls else hook_cls.get_hooks()
        self.source_code = source_code
        if isinstance(input_config, str):
            self.config = Config(input_config)
            # get parser
            if framework:
                logger.debug("Find parser module based on input framework name")
                parser_cls = self.get_framework_parser(framework.lower())
            else:
                logger.debug("Naive way to find parser module")
                parser_cls = self.get_framework_parser_naive(input_config)
            if parser_cls:
                self.parser = parser_cls(self)
                logger.debug("Parser module: %s", self.parser.__class__)
                self.parser.parse()
            return
        if isinstance(input_config, dict):
            self.config = Config(input_config)
        elif isinstance(input_config, Config):
            self.config = input_config
        else:
            raise TypeError(
                f"Expecting path string or Config object for {input_config}"
            )
        #MindFormers format by default
        self.parser = self.get_framework_parser_naive("yaml")(self)
        self.parser.parse()
