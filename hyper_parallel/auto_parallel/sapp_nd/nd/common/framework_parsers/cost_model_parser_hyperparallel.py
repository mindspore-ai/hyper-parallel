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
"""parser child class"""
import ast
from pathlib import Path
import importlib.util
import time
import sys
import os
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import _CostModelParser
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger


class CostModelParserHyperparallel(_CostModelParser):
    """parser class for HyperParallel format"""

    def parse(self):
        """Parse a HyperParallel TOML configuration."""
        if self.ccfg.source_code:
            specs = self.__parse_init_code()
            self.ccfg.specs = specs
            self.__parse_toml()
        else:
            now = time.time()
            home_path = os.path.expanduser("~")
            if home_path not in sys.path:
                sys.path.append(home_path)
            spec_torch = importlib.util.find_spec("torch")
            spec_torchtitan = importlib.util.find_spec("torchtitan")
            if spec_torch is not None and spec_torchtitan is not None:
                # existing torchtitan package
                spec_path = spec_torchtitan.submodule_search_locations[0]
                if spec_path not in sys.path:
                    sys.path.append(spec_path)
                try:
                    logger.info(
                        "found torchtitan package from homedir: %s",
                        spec_path,
                    )
                    logger.info("importing getter from torchitan...")
                    os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
                    # pylint: disable=import-outside-toplevel
                    import torchtitan.protocols.train_spec as train_spec_module
                    logger.info(
                        "import time: %s sec",
                        round(time.time() - now, 2),
                    )
                    train_spec = train_spec_module.get_train_spec(
                        self.config.model.name
                    )
                    model_args = train_spec.model_args[self.config.model.flavor]
                    # Convert recursively to Config
                    # pprint.pprint(train_spec.model_args)
                    # pprint.pprint(model_args)
                    # model = train_spec.model_cls(model_args)
                    # print(model)
                    # print(dir(model))
                    # print("NAMED PARAMETERS")
                    # for k,_ in model.named_parameters():
                    #     print(k)
                    specs = self.__obj_to_config(model_args)
                    self.ccfg.specs = specs
                    self.__parse_toml()
                    return
                except ModuleNotFoundError:
                    pass
            raise AttributeError(
                "Hyperparallel config: Could not find torch/torchtitan package. "
                "Please specify argument --code-path with __init__.py path "
            )

    def __obj_to_config(self, obj):
        """convert to Config"""
        if obj and not isinstance(obj, (str, int, float, bool, list)):
            res = Config({})
            for k, v in obj.__dict__.items():
                setattr(res, k, self.__obj_to_config(v))
            return res
        return obj

    def __parse_init_code(self):
        """parse __init__.py through ast"""
        def parse_args(res, arg):
            """parse config from target flavor (var = dict: flavor -> config)"""
            for kw in arg.keywords:
                if not isinstance(kw.value, ast.Call):
                    res[kw.arg] = ast.literal_eval(kw.value)
                else:
                    res[kw.arg] = {}
                    parse_args(res[kw.arg], kw.value)

        path = Path(self.ccfg.source_code)
        source_code = path.read_text(encoding="utf-8")
        tree = ast.parse(source_code, filename=path.name)
        # fetch get_spec() AST
        tree_get_spec = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "get_train_spec"
        )
        # fetch model_args variable AST
        var_model_args = next(
            k.value.id
            for k in tree_get_spec.body[0].value.keywords
            if k.arg == "model_args"
        )
        var_tree = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.targets[0].id == var_model_args:
                var_tree = node
                break
        # fetch dict of hyperparameters according to flavor
        params = {}
        for k, v in zip(var_tree.value.keys, var_tree.value.values):
            if k.value == self.config.model.flavor:
                parse_args(params, v)
                break
        return Config(params)

    def __parse_toml(self):
        """main parsing order"""
        self.ccfg.model_name = self.config.model.name
        self.ccfg.config_format = "toml"
        self.ccfg.multimodal = False
        self.ccfg.device_capacity = Memory.from_string("56GB")  # important
        self.ccfg.mm_ccfgs = None
        self.ccfg.mm_order = None
        self.__parse_feature_flag()
        self.__parse_hyperparam()
        self.__parse_strat()
        self.__parse_moe()
        self.config_optimizer_shard(self.ccfg)  # need to adapt FSDP
        self.config_comm_flag(self.ccfg)
        self.__parse_batch()
        self.__init_shard()
        self.__init_bytes()
        self.ccfg.n_mtp = 0
        self.ccfg.layer_custom_config = [(self.ccfg.n_lay, None)]
        self.ccfg.overwrite_eval_functions = {}

    def __parse_strat(self):
        """strategy vars"""
        self.ccfg.d = max(
            1,
            self.config.parallelism.data_parallel_replicate_degree
            * self.config.parallelism.data_parallel_shard_degree,
        )  # need correction
        self.ccfg.t = max(1, self.config.parallelism.tensor_parallel_degree)
        self.ccfg.p = max(1, self.config.parallelism.pipeline_parallel_degree)
        self.ccfg.cp = max(1, self.config.parallelism.context_parallel_degree)
        self.ccfg.ep = max(1, self.config.parallelism.expert_parallel_degree)
        self.ccfg.sp = self.ccfg.t
        self.ccfg.vp = 1
        self.ccfg.op_weight_shard = (
            self.config.parallelism.data_parallel_shard_degree * self.ccfg.t
        )
        self.ccfg.os_max_shard = (
            self.ccfg.op_weight_shard if self.ccfg.op_weight_shard >= 1
            else self.ccfg.d * self.ccfg.t
        )  # need correction
        self.ccfg.offset = 0  # important
        self.ccfg.full_rec = self.config.activation_checkpoint.mode == "full"
        self.ccfg.sel_rec = self.config.activation_checkpoint.mode == "selective"
        self.ccfg.pp_sched = self.config.parallelism.pipeline_parallel_schedule
        if self.ccfg.pp_sched:
            # From Pytorch code
            schedule_map = {
                "1F1B": "1f1b",
                "Interleaved1F1B": "1f1b",
                "GPipe": "gpipe",
                "FlexibleInterleaved1F1B": "1f1b",  # ??
                "LoopedBFS": "1f1b",
                "InterleavedZeroBubble": "1f1b",
                "ScheduleZBVZeroBubble": "zero_bubble_v",
                "PipelineScheduleSingle": None,
                "PipelineScheduleMulti": None,
            }
            if "Interleaved" in self.ccfg.pp_sched and self.ccfg.p > 1:
                self.ccfg.vp = 2
            if self.ccfg.pp_sched in schedule_map:
                self.ccfg.pp_sched = schedule_map[self.ccfg.pp_sched]
            else:
                logger.warning(
                    "Unsupported pipeline_parallel_schedule '%s'. "
                    "Defaulting to 1f1b.",
                    self.ccfg.pp_sched,
                )
                self.ccfg.pp_sched = "1f1b"
        else:
            self.ccfg.pp_sched = "1f1b"
        self.ccfg.emb_out_in_offset = True
        self.ccfg.n_s_split = 1
        self.ccfg.cp_algo = "colossalai_cp"
        self.ccfg.rec_op = Config({
            "attBMM": 1,
            "headCast": 1,
            "dropout": 1,
            "softmax": 1,
            "normOp": 1,
            "gather": 1,
            "ffAct": 1,
        })
        self.ccfg.pp_partition = None

    def __parse_hyperparam(self):
        """hyperparameter vars"""
        self.ccfg.multiple_of = max(1, self.ccfg.specs.multiple_of)
        self.ccfg.fdm = max(1, self.ccfg.specs.ffn_dim_multiplier)
        self.ccfg.h = self.ccfg.specs.dim
        self.ccfg.hff = self.ccfg.specs.inter_dim
        if not self.ccfg.hff:
            self.ccfg.hff = self.ccfg.specs.hidden_dim
        if not self.ccfg.hff:
            if "llama" in self.ccfg.model_name:
                self.ccfg.hff = self.init_hff()
            else:
                self.ccfg.hff = self.ccfg.h
        self.ccfg.v = self.ccfg.specs.vocab_size
        self.ccfg.s = self.config.training.seq_len
        self.ccfg.a = self.ccfg.specs.n_heads
        self.ccfg.s_fa = (
            (self.ccfg.s / self.ccfg.a) if self.ccfg.has_fa else self.ccfg.s
        )
        self.ccfg.n_lay = self.ccfg.specs.n_layers
        self.ccfg.n_kv = self.ccfg.specs.n_kv_heads
        if not self.ccfg.n_kv:
            self.ccfg.n_kv = self.ccfg.a
        self.ccfg.dh = self.ccfg.h / self.ccfg.a
        self.ccfg.dc_kv = self.ccfg.specs.kv_lora_rank
        self.ccfg.dc_q = self.ccfg.specs.q_lora_rank
        self.ccfg.dhr = self.ccfg.specs.qk_rope_head_dim
        self.ccfg.k_1st_dense = self.ccfg.specs.n_dense_layers
        self.ccfg.is_mtp_in_offset = True

    def __parse_moe(self):
        """MoE vars"""
        self.ccfg.hff_exp = (
            self.ccfg.specs.moe_inter_dim if self.ccfg.specs.moe_inter_dim
            else self.ccfg.hff
        )
        if (
            not hasattr(self.ccfg.specs, "moe_enabled")
            or self.ccfg.specs.moe_enabled
        ):
            if self.ccfg.specs.moe_args:
                self.ccfg.n_exp = self.ccfg.specs.moe_args.num_experts
                self.ccfg.n_chosen_exp = self.ccfg.specs.moe_args.top_k
                self.ccfg.n_shared_exp = (
                    self.ccfg.specs.moe_args.num_shared_experts
                )
        else:
            self.ccfg.n_exp = 1
            self.ccfg.n_chosen_exp = 1
            self.ccfg.n_shared_exp = 0
        self.ccfg.cap_fact = 1  # Assuming
        self.ccfg.etp = self.config.parallelism.expert_tensor_parallel_degree
        self.config_dp_tp_exp(self.ccfg)  # need verification in code

    def __parse_feature_flag(self):
        """training feature vars"""
        self.ccfg.has_op = True  # Assuming
        self.ccfg.has_grad_shard = True  # Assuming FSDP
        self.ccfg.freeze = False
        self.ccfg.has_fa = True  # Assuming
        self.ccfg.vp_less_mem = False
        self.ccfg.has_clip = False
        self.ccfg.gmm = True
        self.ccfg.vocab_emb_dp = True
        self.ccfg.tie_emb_out = self.ccfg.specs.enable_weight_tying

    def __parse_batch(self):
        """batch related vars"""
        self.ccfg.b = self.config.training.local_batch_size
        self.ccfg.m = self.ccfg.p
        self.ccfg.gbs = self.ccfg.b * self.ccfg.d * self.ccfg.m

    def __init_shard(self):
        """sharding vars"""
        self.ccfg.shard_embed = self.ccfg.t
        self.ccfg.shard_output_activ = True
        self.ccfg.shard_recompute_input = True
        self.ccfg.is_shard_mtp_param = True

    def __init_bytes(self):
        """fp bytes vars"""
        self.ccfg.bytes_p = 4
        self.ccfg.bytes_compute = 2
        self.ccfg.bytes_softmax = 4
        self.ccfg.bytes_grad = 4
        self.ccfg.bytes_os = 4
        self.ccfg.bytes_norm = 4
