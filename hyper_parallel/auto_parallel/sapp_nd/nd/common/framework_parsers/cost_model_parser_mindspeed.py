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
"""parser child class"""
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config, YamlObject
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import _CostModelParser
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger


class CostModelParserMindspeed(_CostModelParser):
    """parser class for MindSpeed format"""

    def parse(self):
        self.__config_parse_json_multimodals()

    # MindSpeed (multimodal)
    def __config_parse_json_multimodals(self):
        """MindSpeed format for multimodal"""
        self.ccfg.device_capacity = Memory.from_gb(55)  # 55 * 1024 * 1024 * 1024
        self.ccfg.model_name = self.config.model_id
        # Assume it exists a hook module with the same name as model_name
        self.ccfg.n_lay = 0  # SUM ALL
        self.ccfg.pp_sched = "1f1b"
        self.ccfg.p = self.config.tmp.pp  # PP
        self.ccfg.m = self.ccfg.p
        self.ccfg.b = self.config.tmp.mbs
        self.ccfg.d = self.config.tmp.dp  # DP
        self.ccfg.t = self.config.tmp.tp  # TP
        self.ccfg.os_max_shard = self.ccfg.d * self.ccfg.t
        self.ccfg.cp = self.config.tmp.cp  # CP
        self.ccfg.vp = self.config.tmp.vpp  # VPP
        self.ccfg.ep = self.config.tmp.ep  # EP
        ccfgs = self.__search_and_parse_mods_ccfg(self.ccfg.config)
        self.ccfg.multimodal = len(ccfgs) > 1
        if self.ccfg.multimodal:
            if not self.ccfg.hooks_dict:
                raise TypeError(
                    "Currently for multimodals, 'hook_cls' required for evaluator"
                )
            missing_hooks = set(ccfgs.keys()) - set(self.ccfg.hooks_dict.keys())
            if missing_hooks:
                raise TypeError(
                    f"Missing hooks for submodules {list(missing_hooks)}"
                )
            self.ccfg.mm_ccfgs = ccfgs
            self.ccfg.mm_order = list(ccfgs.keys())

            # Update each mod's offset, layer_custom_config, pp_partition
            for m in self.ccfg.mm_ccfgs:
                cc = self.ccfg.mm_ccfgs[m]
                num_layer_per_stage = max(1, cc.n_lay // self.ccfg.p // self.ccfg.vp)
                if cc.pp_partition:
                    # print("here",m,cc.pp_partition,num_layer_per_stage)
                    # Making sure the format is offset[vp][p] like in MF
                    if isinstance(cc.pp_partition[0], list):
                        cc.offset = [
                            [
                                cc.pp_partition[v_idx][idx] - num_layer_per_stage
                                for idx in range(self.ccfg.p)
                            ]
                            for v_idx in range(self.ccfg.vp)
                        ]
                    else:
                        cc.offset = [
                            [
                                cc.pp_partition[idx] // self.ccfg.vp
                                - num_layer_per_stage
                                for idx in range(self.ccfg.p)
                            ]
                            for v_idx in range(self.ccfg.vp)
                        ]
                else:
                    self.__complete_unimodal_pp_plan(m, cc, num_layer_per_stage)
        self.ccfg.overwrite_eval_functions = {}

    def __complete_unimodal_pp_plan(self, m, cc, num_layer_per_stage):
        """Try to follow previous pp plan"""
        stage_insert_idx, chunk_insert_idx = 0, 0
        cc.offset = [
            [-num_layer_per_stage for _ in range(self.ccfg.p)]
            for _ in range(self.ccfg.vp)
        ]
        previous_mod_idx = (
            self.ccfg.mm_order.index(m) - 1
        )  # look for previous mod pp partition
        if previous_mod_idx >= 0:
            previous_mod_partition = self.ccfg.mm_ccfgs[
                self.ccfg.mm_order[previous_mod_idx]
            ].pp_partition
            if isinstance(previous_mod_partition[0], list):  # vpp
                put = False
                for v_idx in range(self.ccfg.vp - 1, -1, -1):
                    for s_idx in range(self.ccfg.p - 1, -1, -1):
                        if previous_mod_partition[v_idx][s_idx]:
                            stage_insert_idx = s_idx
                            chunk_insert_idx = v_idx
                            put = True
                            break
                    if put:
                        break
            else:
                stage_insert_idx = self.ccfg.p - 1
                for p in previous_mod_partition[::-1]:
                    if p > 0:
                        break
                    stage_insert_idx = max(0, stage_insert_idx - 1)
        cc.offset[chunk_insert_idx][stage_insert_idx] = (
            cc.n_lay - num_layer_per_stage
        )
        cc.pp_partition = [
            [
                num_layer_per_stage + cc.offset[v_idx][idx]
                for idx in range(self.ccfg.p)
            ]
            for v_idx in range(self.ccfg.vp)
        ]
        cc.p, cc.vp = self.ccfg.p, self.ccfg.vp
        cc.full_rec = (
            self.ccfg.mm_ccfgs[self.ccfg.mm_order[previous_mod_idx]].full_rec is True
        )
        cc.sel_rec = False

    def __search_and_parse_mods_ccfg(self, field):
        """extract multimodal submodules (MindSpeed format)"""
        res = {}
        for _, v in vars(field).items():
            if isinstance(v, YamlObject):
                res.update(self.__search_and_parse_mods_ccfg(v))
                if v.model_id:
                    logger.info("Detected model config: %s", v.model_id)
                    res[v.model_id] = self.__config_parse_json(
                        v
                    )  # Build cost model variable foreach configs
        return res

    def __config_parse_json_parallelism(self, cc, mod):
        """MindSpeed format for parallelism"""
        cc.t = max(cc.tensor_model_parallel_size, self.config.tmp.tp)
        cc.p = max(cc.pipeline_model_parallel_size, self.config.tmp.pp)
        cc.cp = self.config.tmp.cp
        cc.d = self.config.tmp.dp
        cc.ep = max(cc.expert_model_parallel_size, self.config.tmp.ep)
        cc.sp = cc.t if mod.sequence_parallel else 1
        if cc.cp > 1 and cc.sp > 1:
            logger.warning(
                "sequence parallelism and context parallelism are both enabled"
            )
        cc.pp_partition = (
            mod.pipeline_num_layers
        )  # Offset regardless of even distribution

        # Interleaving
        cc.n_s_split = 1  # seqpipe
        cc.pp_sched = "1f1b"
        cc.vp = self.config.tmp.vpp  # VPP

    def __config_parse_json_hyperparameters(self, cc, mod):
        """MindSpeed format for hyperparams"""
        cc.n_lay = mod.num_layers
        cc.h = mod.hidden_size
        cc.hff = mod.ffn_hidden_size
        cc.v = mod.vocab_size
        cc.s = self.config.tmp.seqlen  # SEQ_LEN
        cc.a = mod.num_attention_heads
        cc.n_kv = (
            mod.num_query_groups if mod.num_query_groups else cc.a
        )  # NOT SURE
        cc.dh = (
            mod.kv_channels if mod.kv_channels else (cc.h / cc.a)
        )  # Per head dimension
        # print(cc.a, cc.h, cc.dh)
        cc.dc_kv = mod.k_lora_rank  # KV compression dimension #NOT SURE
        cc.dc_q = mod.q_lora_rank  # Q compression dimension
        cc.dhr = mod.qk_rope_head_dim  # decoupled QK per head dimension

    def __config_parse_json_moe(self, cc, mod):
        """MindSpeed format for MoE infos"""
        cc.n_exp = max(1, mod.num_moe_experts)
        cc.n_chosen_exp = max(1, mod.moe_router_topk)
        cc.n_shared_exp = mod.n_shared_exp
        cc.cap_fact = 1
        cc.t_exp, cc.d_exp = cc.t, cc.d
        cc.hff_exp = (
            mod.moe_intermediate_size if mod.moe_intermediate_size else cc.hff
        )
        cc.k_1st_dense = mod.first_k_dense_replace
        # temporary
        cc.etp = self.config.tmp.etp  # ETP

    def __config_parse_json_op_recompute(self, cc):
        """MindSpeed format for select recompute"""
        cc.rec_op = Config(
            {}
        )  # recomputed operators (selective recompute only)
        cc.rec_op.attBMM = 1
        cc.rec_op.headCast = 1
        cc.rec_op.dropout = 1
        cc.rec_op.softmax = 1
        cc.rec_op.normOp = 1
        cc.rec_op.gather = 1
        cc.rec_op.ffAct = 1

    def __config_parse_json(self, mod):
        """MindSpeed format for unimodal"""
        # def mod_hook(M) :
        cc = type(self.ccfg)({}) # CostModelConfig({})
        cc.parser = self
        cc.config_format = "json"
        cc.model_name = mod.model_id
        cc.freeze = mod.freeze  # for later
        cc.has_fa = True
        cc.has_op = True  # mod.use_distributed_optimizer
        cc.has_grad_shard = True
        # cc.vp_less_mem = False
        cc.has_clip = False
        cc.cp_algo = "colossalai_cp"
        cc.gmm = mod.moe_grouped_gemm
        cc.vocab_emb_dp = False

        cc.offset = 0
        # Parallel dimensions
        self.__config_parse_json_parallelism(cc, mod)

        cc.full_rec = mod.recompute_num_layers
        cc.sel_rec = False
        if mod.recompute_num_layers and isinstance(
            mod.recompute_num_layers, int
        ):
            cc.full_rec = [mod.recompute_num_layers] * cc.p
            if cc.vp > 1:
                cc.full_rec = [cc.full_rec] * cc.vp

        # Hyperparameters
        self.__config_parse_json_hyperparameters(cc, mod)

        # Microbatch infos
        cc.b = self.config.tmp.mbs  # MBS # Microbatch size
        cc.m = cc.p  # Number of microbatches
        # if cc.m<=0 : logger.warning("num_micro is negative")

        # MoE infos
        self.__config_parse_json_moe(cc, mod)
        self.config_dp_tp_exp(cc)

        # FP byte storages
        cc.bytes_p = self.ccfg.fp_bytes(mod.params_dtype)  # parameters
        cc.bytes_compute = 2
        cc.bytes_softmax = (
            4 if mod.attention_softmax_in_fp32 else 2
        )  # softmax output
        cc.bytes_grad = 4
        cc.bytes_os = 4
        cc.bytes_norm = 4

        # Optimizer parallel factors
        cc.os_max_shard = cc.d * cc.t
        self.config_optimizer_shard(cc)

        # Other factors
        cc.shard_embed = cc.t * cc.d
        cc.shard_output_activ = 1
        cc.shard_recompute_input = 1
        cc.s_fa = (
            cc.s if not cc.has_fa else cc.s / cc.a
        )  # flash attention factor [HYPOTHESIS]
        cc.comm_d_non_exp = (
            0
            if ((cc.d == 1) or not cc.has_op)
            else (2 if not cc.has_grad_shard else 3)
        )  # data parallel comm factor
        cc.comm_d_exp = (
            0
            if ((cc.d_exp == 1) or not cc.has_op)
            else (2 if not cc.has_grad_shard else 3)
        )  # data parallel comm factor
        cc.comm_t = float(cc.t > 1)  # tensor parallel comm factor
        cc.comm_ep = float(
            cc.ep > 1 or cc.n_exp > 1
        )  # expert parallel comm factor
        cc.comm_cp = float(cc.cp > 1)  # context parallel comm factor
        cc.gbs = cc.b * cc.d * cc.m
        cc.n_mtp = mod.mtp_num_layers
        # Recomputation
        self.__config_parse_json_op_recompute(cc)
        cc.layer_custom_config = [(cc.n_lay, None)]
        # By default, 100% of layers use a unique custom config (if specified)
        cc.overwrite_eval_functions = {}
        return cc  # mod_hook
