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
"""HyperParallel AutoModels ``train.yaml`` parser (Hyper V2).

Parses the HyperParallel YAML configuration format
(``examples/demo_trainer/train.yaml``) and populates
a :class:`CostModelConfig` for memory estimation.

Model hyperparameters are resolved with the same Transformers
``AutoConfig.from_pretrained`` path used by AutoModels. Legacy
``model.config_overrides`` remains supported for standalone search configs.

Expected YAML structure::

    model:
      _target_: hyper_parallel.auto_models._transformers.HyperAutoModelForCausalLM.from_pretrained
      pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
      torch_dtype: bfloat16

    training:
      global_batch_size: 4
      micro_batch_size: 1

    accelerator:
      tp_size: 1
      pp_size: 1

    fsdp_config:
      dp_shard_size: 4

    activation_checkpoint:
      mode: full
"""
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
import logging
from typing import Any, Dict

from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config, YamlObject
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import _CostModelParser
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import (
    CWrap,
    check_and_apply_custom_hook,
    custom_default_transformer,
)
from hyper_parallel.auto_parallel._hf_model_spec import (
    is_auto_models_schema,
    resolve_hf_model_spec,
)

logger = logging.getLogger(__name__)



def custom_vision_tower(ccfg: Any) -> None:
    """Operation profile of a vision-transformer tower.

    Lives here rather than in ``arch_hooks`` because it is only ever used by
    the multimodal split below.
    """
    custom_default_transformer(ccfg)
    # ViT blocks use a plain two-matmul MLP, not the LLM's gated triple.
    ccfg.n_ffMM = 2
    ccfg.n_ffParamCast = ccfg.n_ffMM if not ccfg.has_op else 0
    ccfg.n_normOp = 2
    ccfg.n_gather = 4


def custom_vision_tower_hook(evaluator: Any) -> None:
    """Apply the vision-tower profile to an evaluator or a config."""
    if not hasattr(evaluator, "set_ccfg"):
        evaluator = CWrap(evaluator)
    evaluator.set_ccfg(custom_vision_tower)


class CostModelParserHyperV2(_CostModelParser):
    """Parser for HyperParallel native train.yaml configuration format.

    This parser replaces the placeholder ``CostModelParserHyperparallel``
    which was written for an older TorchTitan TOML format. It reads the
    AutoModels Trainer schema and resolves model parameters through the
    Transformers configuration pipeline. Legacy ``config_overrides`` input
    remains available for standalone cost-model search files.
    """

    def parse(self) -> None:
        """Main parsing entry point."""
        self.ccfg.config_format = "yaml"
        self.ccfg.multimodal = False
        self.ccfg.mm_ccfgs = None
        self.ccfg.mm_order = None
        self._vision_spec = None

        # Resolve model hyperparameters via AutoModels' Transformers pipeline.
        self._resolve_model_config_pipeline()
        self._resolve_sequence_length()

        # --- Parallelism ---
        self._parse_parallelism()

        # --- Batch ---
        self._parse_batch()

        # --- Feature flags ---
        self._parse_feature_flags()

        # Flash attention factor
        if self.ccfg.has_fa and self.ccfg.a > 0:
            self.ccfg.s_fa = self.ccfg.s / self.ccfg.a
        else:
            self.ccfg.s_fa = self.ccfg.s

        # --- Recompute ---
        self._parse_recompute()

        # --- n_s_split ---
        self.ccfg.n_s_split = 1

        # --- Bytes ---
        self._init_bytes()

        # --- Post-processing ---
        self._init_moe_strategy()
        self.config_optimizer_shard(self.ccfg)
        self.config_comm_flag(self.ccfg)
        self._init_shard()
        self.ccfg.layer_custom_config = [(self.ccfg.n_lay + self.ccfg.n_mtp, None)]
        self._init_offset()
        self.ccfg.overwrite_eval_functions = {}

        # --- Multimodal split (vision-language models only) ---
        self._resolve_multimodal()

    def _resolve_model_config_pipeline(self):
        """Resolve model hyperparameters to populate ``ccfg``.

        Delegates to the shared AutoModels resolver, which reads the same
        Transformers configuration as the Trainer and falls back to
        ``model.config_overrides`` for standalone cost-model search files.
        A vision-language config additionally yields a ``vision`` sub-spec,
        held here until :meth:`_resolve_multimodal` can build its submodule.
        """
        spec = resolve_hf_model_spec(
            self._model_section(), self._visual_seq_len_override()
        )
        self._vision_spec = spec.pop("vision", None)
        self._apply_spec(self.ccfg, spec)
        self._resolve_device_capacity()

    def _model_section(self) -> Dict[str, Any]:
        """Return the ``model`` section as a plain mapping."""
        model_raw = self._config_to_flat_dict(
            self._get_cfg_attr(self.config, "model", Config({}))
        )
        return model_raw if isinstance(model_raw, dict) else {}

    def _visual_seq_len_override(self) -> int:
        """Return ``context.visual_seq_len`` when the config declares one."""
        ctx = self._get_cfg_attr(self.config, "context", Config({}))
        return int(self._get_cfg_attr(ctx, "visual_seq_len", 0) or 0)

    @staticmethod
    def _spec_int(spec: Dict[str, Any], name: str, default: int = 0) -> int:
        """Return an integer spec field, treating ``None`` as absent."""
        return int(spec.get(name, default) or default)

    def _apply_spec(self, ccfg: Any, spec: Dict[str, Any]) -> None:
        """Populate one cost-model config object from a canonical model spec.

        Shared by the main model, the ``config_overrides`` fallback and every
        multimodal submodule, so all three agree on field semantics.
        """
        self._apply_core_spec(ccfg, spec)
        self._apply_moe_spec(ccfg, spec)
        self._apply_scaling_spec(ccfg, spec)

    def _apply_core_spec(self, ccfg: Any, spec: Dict[str, Any]) -> None:
        """Map common transformer geometry and attention fields."""
        ccfg.model_name = str(spec.get("name", "custom"))
        ccfg.h = self._spec_int(spec, "hidden_size")
        ccfg.n_lay = self._spec_int(spec, "num_hidden_layers")
        ccfg.a = self._spec_int(spec, "num_attention_heads")
        ccfg.hff = self._spec_int(spec, "intermediate_size")
        ccfg.v = self._spec_int(spec, "vocab_size")
        ccfg.s = (
            self._spec_int(spec, "max_position_embeddings")
            or self._spec_int(spec, "seq_length")
            or 4096
        )
        ccfg.n_kv = self._spec_int(spec, "num_key_value_heads") or ccfg.a
        # Transformers exposes head_dim explicitly and it is not always
        # hidden_size / num_attention_heads (Qwen3 has h/a = 64, head_dim 128).
        head_dim = self._spec_int(spec, "head_dim")
        ccfg.dh = head_dim if head_dim else (ccfg.h / ccfg.a if ccfg.a else 0)
        ccfg.dc_kv = self._spec_int(spec, "kv_lora_rank")
        ccfg.dc_q = self._spec_int(spec, "q_lora_rank")
        ccfg.dhr = self._spec_int(spec, "qk_rope_head_dim")

    def _apply_moe_spec(self, ccfg: Any, spec: Dict[str, Any]) -> None:
        """Map dense defaults and optional MoE fields."""
        ccfg.n_exp = 1
        ccfg.n_chosen_exp = 1
        ccfg.n_shared_exp = 0
        ccfg.hff_exp = ccfg.hff
        ccfg.cap_fact = 1
        ccfg.t_exp = ccfg.t
        ccfg.d_exp = ccfg.d
        ccfg.gmm = False
        ccfg.k_1st_dense = 0
        num_exp = self._spec_int(spec, "num_experts", 1)
        if num_exp <= 1:
            return
        ccfg.n_exp = num_exp
        ccfg.n_chosen_exp = max(1, self._spec_int(spec, "num_experts_per_tok", 1))
        ccfg.n_shared_exp = self._spec_int(spec, "num_shared_experts")
        moe_inter = self._spec_int(spec, "moe_intermediate_size")
        if moe_inter:
            ccfg.hff_exp = moe_inter
        ccfg.k_1st_dense = self._spec_int(spec, "first_k_dense_replace")
        cap_val = spec.get("capacity_factor", spec.get("cap_fact"))
        if cap_val is not None:
            ccfg.cap_fact = max(1, float(cap_val))
        ccfg.gmm = bool(spec.get("use_gmm", spec.get("gmm", True)))

    def _apply_scaling_spec(self, ccfg: Any, spec: Dict[str, Any]) -> None:
        """Map MTP and feed-forward scaling fields."""
        ccfg.n_mtp = self._spec_int(spec, "mtp_depth")
        # Match the MF parser: when the model declares MTP layers they
        # participate in pipeline offset balancing (default True); when there
        # is none they are excluded, mirroring the MF parser's
        # ``num_nextn_predict_layers`` fallback which sets
        # ``is_mtp_in_offset = False``.
        ccfg.is_mtp_in_offset = bool(ccfg.n_mtp)
        ccfg.multiple_of = self._spec_int(spec, "multiple_of", 256)
        ccfg.fdm = float(spec.get("ffn_dim_multiplier", 1.0) or 1.0)

    # -- Multimodal ----------------------------------------------------

    def _resolve_multimodal(self) -> None:
        """Split a vision-language model into cost-model submodules.

        The vision tower and the language model run on one shared pipeline,
        so both submodules inherit the parent's strategy and only their
        geometry, sequence length and layer placement differ. The parent
        keeps the strategy and drops to ``n_lay = 0`` so the backbone sums
        the submodules rather than its own layer count.
        """
        if not self._vision_spec:
            return

        text_ccfg = self._clone_submodule(self.ccfg.model_name)
        vision_ccfg = self._build_vision_submodule(self._vision_spec)

        self.ccfg.multimodal = True
        self.ccfg.mm_ccfgs = {"vision": vision_ccfg, "text": text_ccfg}
        # Vision runs first; the language model drives the search space.
        self.ccfg.mm_order = ["vision", "text"]
        self.ccfg.mm_main = "text"
        self.ccfg.hooks_dict = {
            "vision": custom_vision_tower_hook,
            "text": check_and_apply_custom_hook,
        }
        self.ccfg.n_lay = 0
        self.ccfg.layer_custom_config = []
        logger.info(
            "Multimodal cost model: vision %s (%d layers, s=%d) + text %s "
            "(%d layers, s=%d)",
            vision_ccfg.model_name, vision_ccfg.n_lay, vision_ccfg.s,
            text_ccfg.model_name, text_ccfg.n_lay, text_ccfg.s,
        )

    def _clone_submodule(self, name: str) -> Any:
        """Return a submodule cost config seeded from the parsed parent."""
        cc = type(self.ccfg)({})
        for key, value in self.ccfg.__dict__.items():
            if key in ("mm_ccfgs", "mm_order", "mm_main", "hooks_dict"):
                continue
            setattr(cc, key, value)
        cc.multimodal = False
        cc.mm_ccfgs = None
        cc.mm_order = None
        cc.parser = self
        cc.model_name = name
        cc.rec_op = Config(dict(self.ccfg.rec_op.__dict__))
        cc.layer_custom_config = [(cc.n_lay + cc.n_mtp, None)]
        cc.offset = self._even_offset()
        return cc

    def _build_vision_submodule(self, vision_spec: Dict[str, Any]) -> Any:
        """Return the vision-tower submodule config.

        The tower is dense, carries no vocabulary embedding, and consumes the
        visual token count rather than the text sequence length. It is placed
        entirely on the first pipeline stage, which is where the runtime puts
        it unless an MPipe-style schedule moves it.
        """
        cc = self._clone_submodule(str(vision_spec.get("name", "vision")))
        self._apply_spec(cc, vision_spec)
        cc.v = 0  # patch embedding, not a vocabulary table
        cc.vocab_emb_dp = False
        cc.n_mtp = 0
        cc.is_mtp_in_offset = False
        cc.s_fa = cc.s / cc.a if cc.has_fa and cc.a > 0 else cc.s
        cc.layer_custom_config = [(cc.n_lay, None)]
        cc.offset = self._front_loaded_offset(cc.n_lay)
        return cc

    def _even_offset(self):
        """Return a balanced offset of the shape ``is_consistent_pp_config`` wants."""
        if self.ccfg.vp > 1:
            return [[0] * self.ccfg.p for _ in range(self.ccfg.vp)]
        return [0] * self.ccfg.p

    def _front_loaded_offset(self, n_lay: int):
        """Return an offset placing every layer on the first pipeline stage."""
        per_stage = max(0, n_lay // max(1, self.ccfg.p) // max(1, self.ccfg.vp))
        head = n_lay - per_stage
        if self.ccfg.vp > 1:
            chunks = [[-per_stage] * self.ccfg.p for _ in range(self.ccfg.vp)]
            chunks[0][0] = head
            return chunks
        stages = [-per_stage] * self.ccfg.p
        stages[0] = head
        return stages

    def _resolve_sequence_length(self) -> None:
        """Prefer the Trainer dataset sequence length over the model limit."""
        data_raw = self._get_cfg_attr(self.config, "data", Config({}))
        legacy_seq_len = self._get_cfg_attr(data_raw, "max_seq_len", 0)

        dataset_raw = self._get_cfg_attr(self.config, "dataset", Config({}))
        transform_raw = self._get_cfg_attr(
            dataset_raw, "data_transform", Config({}),
        )
        trainer_seq_len = self._get_cfg_attr(transform_raw, "max_seq_len", 0)
        seq_len = int(trainer_seq_len or legacy_seq_len or self.ccfg.s or 4096)
        self.ccfg.s = seq_len

    def _resolve_device_capacity(self) -> None:
        """Set device capacity from config or default (64 GB)."""
        ctx = self._get_cfg_attr(self.config, "context", Config({}))
        device_mem_str = (
            ctx.__dict__.get("max_device_memory", None)
            if isinstance(ctx, (Config, YamlObject))
            else None
        )
        if device_mem_str:
            self.ccfg.device_capacity = Memory.from_string(str(device_mem_str))
        else:
            self.ccfg.device_capacity = Memory.from_string("64GB")

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _get_cfg_attr(cfg: Any, attr: str, default: Any = None) -> Any:
        """Get an attribute from ``Config`` / ``YamlObject`` safely.

        ``YamlObject.__getattr__`` returns ``0`` for missing attributes
        instead of raising ``AttributeError``, which breaks Python's
        ``getattr(obj, attr, default)`` fallback protocol. This helper
        checks ``__dict__`` directly.
        """
        if isinstance(cfg, (Config, YamlObject)):
            return cfg.__dict__.get(attr, default)
        return getattr(cfg, attr, default)

    @staticmethod
    def _config_to_flat_dict(cfg: Any) -> Dict[str, Any]:
        """Recursively convert a ``Config`` or ``YamlObject`` to a flat dict."""
        if isinstance(cfg, (Config, YamlObject)):
            return {k: CostModelParserHyperV2._config_to_flat_dict(v)
                    for k, v in cfg.__dict__.items()
                    if not k.startswith("_")}
        if isinstance(cfg, (int, float, str, bool)):
            return cfg  # type: ignore[return-value]
        if isinstance(cfg, list):
            return [CostModelParserHyperV2._config_to_flat_dict(i) for i in cfg]
        return cfg

    @staticmethod
    def _bytes_from_dtype(dtype_str: Any) -> int:
        """Parse a dtype string (e.g. ``\"float32\"``) to byte size.

        Returns ``4`` for float32, ``2`` for bfloat16/float16, etc.
        Defaults to ``4`` when parsing fails.
        """
        import re  # pylint: disable=import-outside-toplevel
        dtype_str = str(dtype_str)
        m = re.search(r"(\d+)", dtype_str)
        if m:
            return max(1, int(m.group(1)) // 8)
        return 4

    def _parse_parallelism(self):
        """Extract parallelism from the AutoModels or legacy Trainer schema."""
        train_raw = self._get_cfg_attr(self.config, "train", Config({}))
        legacy_accel = self._get_cfg_attr(train_raw, "accelerator", Config({}))
        accel = self._get_cfg_attr(self.config, "accelerator", legacy_accel)
        fsdp = self._get_cfg_attr(self.config, "fsdp_config", Config({}))

        dp_shard = self._parse_parallel_dimensions(accel, fsdp)
        self._parse_sequence_parallelism(accel)
        self._parse_optimizer_parallelism(accel, dp_shard)

    def _parse_parallel_dimensions(self, accel, fsdp) -> int:
        """Populate mesh dimensions and return the data shard degree."""

        dp_shard = int(
            self._get_cfg_attr(fsdp, "dp_shard_size", 0)
            or self._get_cfg_attr(accel, "dp_shard", 1)
            or 1
        )
        dp_replicate = int(self._get_cfg_attr(accel, "dp_replicate", 1) or 1)
        tp = int(
            self._get_cfg_attr(accel, "tp_size", 0)
            or self._get_cfg_attr(accel, "tp_degree", 1)
            or 1
        )
        pp = int(
            self._get_cfg_attr(accel, "pp_size", 0)
            or self._get_cfg_attr(accel, "pipeline_parallel_degree", 1)
            or 1
        )
        cp = int(
            self._get_cfg_attr(accel, "cp_size", 0)
            or self._get_cfg_attr(accel, "context_parallel_degree", 1)
            or 1
        )
        ep = int(
            self._get_cfg_attr(accel, "ep_size", 0)
            or self._get_cfg_attr(accel, "expert_parallel_degree", 1)
            or 1
        )
        etp = int(self._get_cfg_attr(accel, "expert_tensor_parallel_degree", 0) or 0)
        ep = max(ep, 1)

        self.ccfg.t = max(1, tp)
        self.ccfg.p = max(1, pp)
        self.ccfg.cp = max(1, cp)
        self.ccfg.ep = max(1, ep)
        self.ccfg.d = self._resolve_data_parallel(dp_replicate, dp_shard)
        self.ccfg.sp = self.ccfg.t  # Sequence parallel factor
        self.ccfg.etp = etp
        self.ccfg.vp = max(1, int(
            self._get_cfg_attr(accel, "pp_interleave_num", 1) or 1
        ))
        return dp_shard

    def _resolve_data_parallel(self, dp_replicate: int, dp_shard: int) -> int:
        """Return the data-parallel degree, preferring an explicit device count.

        ND reads ``d * t * cp * p`` back as the cluster size whenever no
        device count is supplied on the command line. The AutoModels schema
        has no replicate field, since the runtime derives it from the world
        size, so without ``context.device_num`` an HSDP run would understate
        the cluster by exactly its replicate factor.
        """
        ctx = self._get_cfg_attr(self.config, "context", Config({}))
        device_num = int(self._get_cfg_attr(ctx, "device_num", 0) or 0)
        if not device_num:
            if is_auto_models_schema(self.config) and dp_replicate == 1:
                logger.warning(
                    "AutoModels config carries no context.device_num; assuming "
                    "dp_replicate=1 (d = dp_shard_size = %d). Pass -d/--devices "
                    "for HSDP runs.", dp_shard,
                )
            return max(1, dp_replicate * dp_shard)
        denom = self.ccfg.t * self.ccfg.p * self.ccfg.cp
        if denom < 1 or device_num % denom:
            raise ValueError(
                f"context.device_num={device_num} is not divisible by "
                f"t*p*cp={denom}"
            )
        return max(1, device_num // denom)

    def _parse_sequence_parallelism(self, accel) -> None:
        """Populate sequence-parallel and pipeline scheduler settings."""
        use_sp = bool(
            self._get_cfg_attr(accel, "sequence_parallel", False)
            or self._get_cfg_attr(accel, "use_seq_parallel", False)
        )
        self.ccfg.sp = self.ccfg.t if use_sp else 1
        self.ccfg.pp_sched = str(
            self._get_cfg_attr(accel, "pipeline_scheduler", "1f1b")
        )

    def _parse_optimizer_parallelism(self, accel, dp_shard: int) -> None:
        """Populate optimizer and gradient sharding settings."""
        is_auto_models = is_auto_models_schema(self.config)
        self.ccfg.has_op = (
            dp_shard > 1
            if is_auto_models
            else bool(self._get_cfg_attr(
                accel, "enable_parallel_optimizer", True,
            ))
        )
        self.ccfg.op_weight_shard = max(1, int(
            self._get_cfg_attr(accel, "optimizer_weight_shard_size", 0)
        ) or (dp_shard if is_auto_models else self.ccfg.d * self.ccfg.t))
        self.ccfg.has_grad_shard = bool(self._get_cfg_attr(accel,
                                                             "gradient_accumulation_shard",
                                                             False))
        self.ccfg.os_max_shard = (
            self.ccfg.op_weight_shard if self.ccfg.op_weight_shard >= 1
            else self.ccfg.d * self.ccfg.t
        )

    def _parse_batch(self):
        """Extract batch settings from ``training`` or legacy ``train``."""
        legacy_train = self._get_cfg_attr(self.config, "train", Config({}))
        train_raw = self._get_cfg_attr(self.config, "training", legacy_train)
        self.ccfg.b = max(1, int(self._get_cfg_attr(train_raw, "micro_batch_size", 1) or 1))
        m = int(self._get_cfg_attr(train_raw, "micro_batch_num", 0) or 0)
        gbs = int(self._get_cfg_attr(train_raw, "global_batch_size", 0) or 0)
        if m > 0:
            self.ccfg.m = m
        elif gbs > 0 and gbs % (self.ccfg.b * self.ccfg.d) == 0:
            self.ccfg.m = max(1, gbs // (self.ccfg.b * self.ccfg.d))
        else:
            self.ccfg.m = self.ccfg.p
        if gbs > 0:
            self.ccfg.gbs = gbs
        else:
            self.ccfg.gbs = self.ccfg.b * self.ccfg.d * self.ccfg.m

    def _parse_feature_flags(self):
        """Set training feature flags."""
        self.ccfg.has_fa = True
        self.ccfg.vocab_emb_dp = True
        self.ccfg.tie_emb_out = False
        self.ccfg.freeze = False
        legacy_train = self._get_cfg_attr(self.config, "train", Config({}))
        training = self._get_cfg_attr(self.config, "training", legacy_train)
        optimizer = self._get_cfg_attr(
            self.config,
            "optimizer",
            self._get_cfg_attr(legacy_train, "optimizer", Config({})),
        )
        max_grad_norm = float(
            self._get_cfg_attr(training, "max_grad_norm", None)
            or self._get_cfg_attr(optimizer, "max_grad_norm", 0.0)
            or 0.0
        )
        self.ccfg.has_clip = max_grad_norm > 0
        self.ccfg.vp_less_mem = False
        accel = self._get_cfg_attr(
            self.config,
            "accelerator",
            self._get_cfg_attr(legacy_train, "accelerator", Config({})),
        )
        cp_algo = self._get_cfg_attr(accel, "context_parallel_algo", None)
        if cp_algo:
            self.ccfg.cp_algo = cp_algo
        else:
            self.ccfg.cp_algo = "colossalai_cp"
            if self.ccfg.cp and self.ccfg.cp > 1:
                logger.warning(
                    "context_parallel_algo not set; defaulting to "
                    "'colossalai_cp' (Ring CP). Set "
                    "train.accelerator.context_parallel_algo explicitly "
                    "to 'ulysses_cp' if Ulysses CP is intended."
                )
        # Optimizer type — used by GlobalConfig.max_op to detect muon-based
        # optimizers.  Matches the MF parser's
        # ``self.ccfg.optimizer = self.config.optimizer.type``.
        opt_type = (
            self._get_cfg_attr(optimizer, "_target_", None)
            or self._get_cfg_attr(optimizer, "type", None)
        )
        # Always a string: GlobalConfig.max_op only bounds OP by the data
        # parallel degree when this reads as a non-muon optimizer name, and
        # the generated cost-model yaml carries no optimizer section.
        self.ccfg.optimizer = str(opt_type) if opt_type else "adamw"

    def _parse_recompute(self):
        """Parse recompute mode.

        Reads ``activation_checkpoint.mode`` from the AutoModels schema, with
        the legacy ``train.gradient_checkpointing`` path as a fallback. When
        ``config_overrides`` supplies ``full_rec`` or ``sel_rec`` (matching
        the MF parser's ``recompute_config.recompute`` /
        ``recompute_config.select_recompute`` fields), those values take
        precedence so that Hyper YAML demo files can express per-stage
        recompute lists for side-by-side comparisons with MindFormers.
        """
        model_raw = self._get_cfg_attr(self.config, "model", Config({}))
        overrides = self._get_cfg_attr(model_raw, "config_overrides", Config({}))
        full_rec_override = self._get_cfg_attr(overrides, "full_rec", None)
        sel_rec_override = self._get_cfg_attr(overrides, "sel_rec", None)

        train_raw = self._get_cfg_attr(self.config, "train", Config({}))
        gc = self._get_cfg_attr(train_raw, "gradient_checkpointing", Config({}))
        activation_checkpoint = self._get_cfg_attr(
            self.config, "activation_checkpoint", Config({}),
        )
        ac_mode = str(
            self._get_cfg_attr(activation_checkpoint, "mode", None)
            or self._get_cfg_attr(gc, "activation_checkpoint", "none")
        )
        if ac_mode == "off":
            ac_mode = "none"

        if full_rec_override is not None:
            self.ccfg.full_rec = full_rec_override
        else:
            self.ccfg.full_rec = ac_mode == "full"

        if sel_rec_override is not None:
            self.ccfg.sel_rec = sel_rec_override
        else:
            self.ccfg.sel_rec = ac_mode == "selective"

        self.ccfg.rec_op = Config({
            "attBMM": 1,
            "headCast": 1,
            "dropout": 1,
            "softmax": 1,
            "normOp": 1,
            "gather": 1,
            "ffAct": 1,
        })

    def _init_bytes(self):
        """Set FP byte sizes from AutoModels or legacy dtype fields."""
        model_raw = self._get_cfg_attr(self.config, "model", Config({}))
        fsdp = self._get_cfg_attr(self.config, "fsdp_config", Config({}))
        mix_precision = self._get_cfg_attr(fsdp, "mix_precision", Config({}))
        model_dtype = self._get_cfg_attr(model_raw, "torch_dtype", None)
        self.ccfg.bytes_p = self._bytes_from_dtype(
            self._get_cfg_attr(mix_precision, "param_dtype", None)
            or model_dtype
            or self._get_cfg_attr(model_raw, "param_init_type", "float32")
        )
        self.ccfg.bytes_compute = self._bytes_from_dtype(
            model_dtype
            or self._get_cfg_attr(model_raw, "compute_dtype", "bfloat16")
        )
        self.ccfg.bytes_softmax = self._bytes_from_dtype(
            self._get_cfg_attr(model_raw, "softmax_compute_type", "float32"))
        self.ccfg.bytes_grad = 4
        self.ccfg.bytes_os = 4
        self.ccfg.bytes_norm = 4

    def _init_moe_strategy(self):
        """Initialize MoE strategy variables via base helper.

        For MoE models (``n_exp > 1``), ``etp`` defaults to 1 when
        absent from the YAML, matching the MF parser's
        ``expert_model_parallel`` default.  For dense models the
        existing ``etp=0`` path continues to produce ``t_exp = t,
        d_exp = d``.

        Catches invalid MoE combinations (e.g., ``d_exp = 0`` when
        ``dp < ep``) so the search engine can proceed — invalid combos
        will later be filtered by the memory budget check.
        """
        if self.ccfg.n_exp > 1 and self.ccfg.etp == 0:
            self.ccfg.etp = 1
        try:
            self.config_dp_tp_exp(self.ccfg)
        except TypeError:
            logger.warning(
                "MoE config_dp_tp_exp failed for d=%d t=%d ep=%d etp=%d "
                "n_exp=%d — clamping to minimum values.",
                self.ccfg.d, self.ccfg.t, self.ccfg.ep,
                self.ccfg.etp, self.ccfg.n_exp,
            )
            self.ccfg.d_exp = max(1, self.ccfg.d_exp)
            self.ccfg.t_exp = max(1, self.ccfg.t_exp)
            self.ccfg.hff_exp = max(1, self.ccfg.hff_exp)
            self.ccfg.n_exp = max(1, self.ccfg.n_exp)

    def _init_offset(self):
        """Initialize the pipeline offset.

        The MF parser reads ``model.model_config.offset`` directly from the
        YAML.  When it is a list (e.g. ``[1, 1, ..., -1]``),
        ``CostModelConfig.is_consistent_pp_config`` requires
        ``len(offset) == pp``, so strategies whose pipeline degree differs
        are rejected until ``GlobalConfig.adapt_config`` regenerates a
        matching offset.  A scalar ``0`` is always accepted.

        To match the MF parser's *list*-based filtering behaviour (used by
        DeepSeek-V3 and other models that declare an explicit offset), this
        parser emits a list offset of length ``pp`` (all zeros = even
        balancing) by default.  An explicit offset supplied via
        ``config_overrides.offset`` overrides this — a list is used as-is,
        and a non-zero int is broadcast to ``[int] * pp``.
        """
        model_raw = self._get_cfg_attr(self.config, "model", Config({}))
        overrides = self._get_cfg_attr(model_raw, "config_overrides", Config({}))
        explicit = self._get_cfg_attr(overrides, "offset", None)
        if explicit is None:
            explicit = self._get_cfg_attr(model_raw, "offset", None)
        if isinstance(explicit, list):
            self.ccfg.offset = list(explicit)
        elif isinstance(explicit, int):
            if explicit == 0:
                self.ccfg.offset = 0
            else:
                self.ccfg.offset = [explicit] * self.ccfg.p
        else:
            self.ccfg.offset = [0] * self.ccfg.p

    def config_shard_emb(self) -> None:
        """Configure embedding sharding based on current parallelism.

        Mirrors ``CostModelParserMindformers.config_shard_emb`` so that
        ``set_strategy`` recomputes ``shard_embed`` whenever the parallel
        configuration changes.  When ``vocab_emb_dp`` is enabled and pipeline
        parallelism is disabled (``p == 1``), the embedding is sharded only
        along the data-parallel dimension (``d``); otherwise it is sharded
        along ``t * d``.

        Without this method, ``CostModelConfig.set_strategy`` skips the
        ``config_shard_emb`` call (guarded by ``hasattr``) and the initial
        ``shard_embed`` value computed in ``_init_shard`` is never refreshed,
        producing an embedding-memory mismatch versus the MF parser.
        """
        self.ccfg.shard_embed = (
            self.ccfg.d
            if (self.ccfg.vocab_emb_dp and self.ccfg.p == 1)
            else (self.ccfg.t * self.ccfg.d)
        )

    def config_shard_recompute(self) -> None:
        """Recompute ``shard_recompute_input`` after strategy changes.

        When ``recompute_slice_activation`` is ``True``, the recompute input
        is sharded by the current tensor-parallel degree ``t``; otherwise it
        is not sharded (value ``1``).  This method is called by
        ``set_strategy`` (via ``hasattr`` guard) so that changing ``t``
        during search correctly updates the sharding factor.

        Without this method, ``shard_recompute_input`` retains the value
        computed at initial parse time (using the default ``t`` from the
        YAML), causing memory-estimation errors when the search explores
        strategies with different ``t`` values.
        """
        self.ccfg.shard_recompute_input = (
            self.ccfg.t if self._recompute_slice_activation else 1
        )

    def _init_shard(self):
        """Initialize sharding variables.

        ``shard_embed`` is computed via :meth:`config_shard_emb` so the
        initial value follows the same rule used on subsequent
        ``set_strategy`` calls.  ``shard_output_activ`` defaults to 1 (no
        sharding), matching the MF parser's default; the ``custom_qwen``
        arch hook overrides it to ``ccfg.t`` for Qwen-family models via
        ``check_and_apply_custom_hook``.

        ``shard_recompute_input`` mirrors the MF parser's
        ``recompute_config.recompute_slice_activation`` flag: when the flag
        is ``True`` (DeepSeek-V3), activations are sharded by ``ccfg.t``;
        when ``False`` (Qwen), they are not sharded.  The flag is stored
        as ``self._recompute_slice_activation`` so that
        :meth:`config_shard_recompute` can recompute the value after
        ``set_strategy`` changes ``t``.  Per-model arch hooks
        (e.g. ``custom_qwen``) may override this during ``EvaluatorV2``
        initialisation.
        """
        self.config_shard_emb()
        self.ccfg.shard_output_activ = 1
        train_raw = self._get_cfg_attr(self.config, "train", Config({}))
        gc = self._get_cfg_attr(train_raw, "gradient_checkpointing", Config({}))
        fsdp = self._get_cfg_attr(self.config, "fsdp_config", Config({}))
        ac = self._get_cfg_attr(self.config, "activation_checkpoint", Config({}))
        self._recompute_slice_activation = bool(self._get_cfg_attr(
            ac,
            "recompute_slice_activation",
            self._get_cfg_attr(
                fsdp,
                "recompute_slice_activation",
                self._get_cfg_attr(gc, "recompute_slice_activation", False),
            ),
        ))
        self.config_shard_recompute()
        self.ccfg.is_shard_mtp_param = True
