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
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import _CostModelParser
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger


class CostModelParserMindformers(_CostModelParser):
    """parser class for MindFormers format"""

    def parse(self):
        self.__config_parse_yaml()

    def __config_parse_yaml_parallelism(self):
        """MindFormer format for strategy"""
        self.ccfg.has_op = self.config.parallel.enable_parallel_optimizer
        op_cfg = self.config.parallel.parallel_optimizer_config
        if op_cfg:
            self.ccfg.has_grad_shard = op_cfg.gradient_accumulation_shard
        self.ccfg.vocab_emb_dp = self.config.parallel_config.vocab_emb_dp
        self.ccfg.tie_emb_out = self.config.model.model_config.tie_word_embeddings
        self.ccfg.cp_algo = (
            self.config.parallel_config.context_parallel_algo
            if self.config.parallel_config.context_parallel_algo
            else "colossalai_cp"
        )
        self.ccfg.d = max(
            1, self.config.parallel_config.data_parallel
        )  # Data parallel
        self.ccfg.t = max(
            1, self.config.parallel_config.model_parallel
        )  # Tensor parallel
        self.ccfg.p = max(
            1, self.config.parallel_config.pipeline_stage
        )  # Pipeline parallel
        self.ccfg.cp = max(
            1, self.config.parallel_config.context_parallel
        )  # Context parallel
        self.ccfg.ep = max(
            1, self.config.parallel_config.expert_parallel
        )  # Expert parallel
        self.ccfg.sp = (
            self.ccfg.t if self.config.parallel_config.use_seq_parallel else 1
        )  # Sequence parallel factor
        if self.ccfg.cp > 1 and self.ccfg.sp > 1:
            logger.warning(
                "sequence parallelism and context parallelism are both enabled"
            )

        # Interleaving
        self.ccfg.n_s_split = 1  # seqpipe
        self.ccfg.pp_sched = "1f1b"
        self.ccfg.vp = max(1, self.config.model.model_config.pp_interleave_num)
        if self.config.parallel.pipeline_config:
            if self.config.parallel.pipeline_config.pipeline_interleave:
                self.ccfg.n_s_split = max(
                    1, self.config.parallel_config.seq_split_num
                )
            if self.config.parallel.pipeline_config.pipeline_scheduler:
                self.ccfg.pp_sched = (
                    self.config.parallel.pipeline_config.pipeline_scheduler
                )

    def __config_parse_yaml_hyperparameters(self):
        """MindFormer format for hyperparams"""
        self.ccfg.n_lay = (
            self.config.model.model_config.num_layers
            if self.config.model.model_config.num_layers
            else self.config.model.model_config.num_hidden_layers
        )
        if self.config.model.model_config.is_encoder_decoder:
            self.ccfg.n_lay *= 2
        self.ccfg.h = self.config.model.model_config.hidden_size  # Hidden size
        self.ccfg.hff = (
            self.config.model.model_config.intermediate_size
            if self.config.model.model_config.intermediate_size
            else self.init_hff()
        )  # Expanded hidden size
        self.ccfg.v = self.config.model.model_config.vocab_size  # Vocabulary size
        self.ccfg.s = self.config.model.model_config.seq_length  # Sequence length
        self.ccfg.a = (
            self.config.model.model_config.num_heads
        )  # Number of attention (query) heads
        if not self.ccfg.a:
            self.ccfg.a = self.config.model.model_config.num_attention_heads
        self.ccfg.n_kv = self.config.model.model_config.n_kv_heads # Number of keys - values heads
        if not self.ccfg.n_kv:
            self.ccfg.n_kv = self.config.model.model_config.num_key_value_heads
        if not self.ccfg.n_kv:
            self.ccfg.n_kv = self.ccfg.a
        self.ccfg.dh = self.ccfg.h / self.ccfg.a  # Per head dimension
        self.ccfg.dc_kv = (
            self.config.model.model_config.kv_lora_rank
            if self.config.model.model_config.kv_lora_rank
            else 0
        )  # KV compression dimension
        self.ccfg.dc_q = (
            self.config.model.model_config.q_lora_rank
            if self.config.model.model_config.q_lora_rank
            else 0
        )  # Q compression dimension
        self.ccfg.dhr = (
            self.config.model.model_config.qk_rope_head_dim
            if self.config.model.model_config.qk_rope_head_dim
            else 0
        )  # decoupled QK per head dimension

        # Microbatch infos
        self.ccfg.b = max(
            1, self.config.runner_config.batch_size
        )  # Microbatch size
        self.ccfg.m = (
            self.config.parallel_config.micro_batch_num
        )  # Number of microbatches
        if self.ccfg.m <= 0:
            logger.warning("num_micro is negative")

    def __config_parse_yaml_moe(self):
        """MindFormer format for MoE infos"""
        self.ccfg.n_exp, self.ccfg.n_chosen_exp, self.ccfg.n_shared_exp = 1, 1, 0
        self.ccfg.hff_exp, self.ccfg.cap_fact = self.ccfg.hff, 1
        self.ccfg.t_exp, self.ccfg.d_exp = self.ccfg.t, self.ccfg.d
        if self.config.moe_config:
            self.ccfg.n_exp = max(
                1, self.config.moe_config.expert_num
            )  # Total number of experts
            self.ccfg.n_chosen_exp = max(
                1, self.config.moe_config.num_experts_chosen
            )  # Number of chosen experts
            self.ccfg.n_shared_exp = self.config.moe_config.shared_expert_num
            if self.config.moe_config.moe_intermediate_size:
                self.ccfg.hff_exp = self.config.moe_config.moe_intermediate_size
            self.ccfg.cap_fact = max(
                1, self.config.moe_config.capacity_factor
            )  # Capacity factor
            self.ccfg.k_1st_dense = self.config.moe_config.first_k_dense_replace
            self.ccfg.etp = self.config.moe_config.expert_model_parallel
            self.config_dp_tp_exp(self.ccfg)
            self.ccfg.gmm = self.config.moe_config.use_gmm
        else:
            cfg = self.config.model.model_config
            self.ccfg.n_exp = max(self.ccfg.n_exp, cfg.n_routed_experts)
            self.ccfg.n_chosen_exp = max(self.ccfg.n_chosen_exp, cfg.num_experts_per_tok)
            self.ccfg.n_shared_exp = max(self.ccfg.n_shared_exp, cfg.n_shared_experts)
            if cfg.moe_intermediate_size:
                self.ccfg.hff_exp = cfg.moe_intermediate_size
            self.ccfg.k_1st_dense = max(self.ccfg.k_1st_dense, cfg.first_k_dense_replace)
            self.config_dp_tp_exp(self.ccfg)
            self.ccfg.gmm = cfg.moe_grouped_gemm

    def __config_parse_yaml_op_recompute(self):
        """MindFormer format for select recompute"""
        # [HYPOTHESIS]
        self.ccfg.rec_op = Config(
            {}
        )  # recomputed operators (selective recompute only)
        self.ccfg.rec_op.attBMM = int(
            not (
                self.config.recompute_config.select_recompute
                and not self.ccfg.has_fa
                and self.ccfg.sp > 1
            )
        )
        self.ccfg.rec_op.headCast = int(
            not (self.config.recompute_config.select_recompute and self.ccfg.has_fa)
        )
        self.ccfg.rec_op.dropout = 1
        self.ccfg.rec_op.softmax = int(
            not (
                self.config.recompute_config.select_recompute
                and not self.ccfg.has_fa
            )
        )
        self.ccfg.rec_op.normOp = int(
            not (self.config.recompute_config.select_recompute and self.ccfg.sp > 1)
        )
        self.ccfg.rec_op.gather = int(
            not (
                self.config.recompute_config.select_comm_recompute
                and self.ccfg.sp > 1
            )
        )
        self.ccfg.rec_op.ffAct = int(
            not (self.config.recompute_config.select_recompute and self.ccfg.sp > 1)
        )

    def config_shard_emb(self):
        """Configure embedding and output activation sharding."""
        self.ccfg.shard_embed = (
            self.ccfg.d
            if (self.ccfg.vocab_emb_dp and self.ccfg.p == 1)
            else (self.ccfg.t * self.ccfg.d)
        )

    def __config_parse_yaml(self):
        """MindFormer format for unimodal"""
        self.ccfg.config_format = "yaml"
        self.ccfg.model_name = self.config.trainer.model_name
        self.ccfg.device_capacity = Memory.from_string(
            self.config.context.max_device_memory
        )
        # (
        #     float(self.config.context.max_device_memory[:-2])
        #     * 1024
        #     * 1024
        #     * 1024
        # )
        self.ccfg.has_fa = self.config.model.model_config.use_flash_attention
        # self.ccfg.vp_less_mem = False
        self.ccfg.has_clip = self.config.runner_wrapper.use_clip_grad
        self.ccfg.gmm = False
        self.ccfg.freeze = False
        op_cfg = self.config.parallel.parallel_optimizer_config
        if op_cfg:
            self.ccfg.op_weight_shard = op_cfg.optimizer_weight_shard_size
        self.ccfg.optimizer = self.config.optimizer.type
        self.ccfg.multiple_of = (
            self.config.model.model_config.multiple_of
            if self.config.model.model_config.multiple_of
            else 1
        )
        self.ccfg.fdm = (
            self.config.model.model_config.ffn_dim_multiplier
            if self.config.model.model_config.ffn_dim_multiplier
            else 1
        )
        self.__config_parse_yaml_parallelism()
        self.__config_parse_yaml_hyperparameters()
        self.__config_parse_yaml_moe()

        # FP byte storages
        self.ccfg.bytes_p = self.ccfg.fp_bytes(
            self.config.model.model_config.param_init_type,
        )  # parameters
        if not self.ccfg.bytes_p:
            self.ccfg.bytes_p = self.ccfg.fp_bytes(
                self.config.model.model_config.params_dtype
            )
        self.ccfg.bytes_compute = self.ccfg.fp_bytes(
            self.config.model.model_config.compute_dtype
        )  # activations
        self.ccfg.bytes_softmax = self.ccfg.fp_bytes(
            self.config.model.model_config.softmax_compute_type
        )  # softmax output
        if not self.ccfg.bytes_softmax:
            self.ccfg.bytes_softmax = self.ccfg.fp_bytes(
                self.config.model.model_config.softmax_compute_dtype
            )
        if not self.ccfg.bytes_p:
            raise AttributeError("bytes_p not positive")
        if not self.ccfg.bytes_compute:
            raise AttributeError("bytes_compute not positive")

        # Optimizer parallel factors
        if self.ccfg.op_weight_shard:
            self.ccfg.os_max_shard = self.ccfg.op_weight_shard
        elif self.ccfg.has_op:
            self.ccfg.os_max_shard = self.ccfg.d * self.ccfg.t
        else:
            self.ccfg.os_max_shard = 1
        self.config_optimizer_shard(self.ccfg)

        # Other factors
        self.config_shard_emb()
        self.ccfg.shard_output_activ = 1
        self.ccfg.shard_recompute_input = (
            self.ccfg.t
            if self.config.recompute_config.recompute_slice_activation
            else 1
        )
        self.ccfg.s_fa = (
            self.ccfg.s
            if not self.config.model.model_config.use_flash_attention
            else self.ccfg.s / self.ccfg.a
        )  # flash attention factor [HYPOTHESIS]
        self.config_comm_flag(self.ccfg)
        self.ccfg.gbs = self.ccfg.b * self.ccfg.d * self.ccfg.m
        self.ccfg.n_mtp = (
            self.config.model.model_config.mtp_depth
            if self.config.model.model_config.mtp_depth
            else 0
        )
        if not self.ccfg.n_mtp:
            self.ccfg.n_mtp = (
                self.config.model.model_config.num_nextn_predict_layers
            )
            self.ccfg.is_mtp_in_offset = False

        # Layer custom config
        # [(num layers selected, layer custom config to apply)]
        self.ccfg.layer_custom_config = [(self.ccfg.n_lay + self.ccfg.n_mtp, None)]

        self.__config_parse_yaml_op_recompute()
        # By default, 100% of layers use a unique custom config (if specified)
        self.ccfg.offset = self.config.model.model_config.offset
        self.ccfg.sel_rec = self.config.recompute_config.select_recompute
        self.ccfg.full_rec = self.config.recompute_config.recompute
        self.ccfg.overwrite_eval_functions = {}
