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
"""HyperParallel native train.yaml parser (Hyper V2).

Parses the HyperParallel YAML configuration format
(``examples/qwen3_5_0_8b_base/train.yaml``) and populates
a :class:`CostModelConfig` for memory estimation.

Model hyperparameters are resolved by importing the model module's
``_build_config`` function directly (by naming convention) rather than
through the ``ModelSpec`` registry.  This avoids coupling the
memory-estimation pipeline to the model registration interface.

Expected YAML structure::

    model:
      name: qwen3_5                    # model spec key
      config_overrides:
        num_hidden_layers: 4
        hidden_size: 3584
        ...

    train:
      accelerator:
        dp_shard: 4                    # FSDP shard degree
      gradient_checkpointing:
        activation_checkpoint: none    # none | full | selective
      global_batch_size: 4
      micro_batch_size: 1

    data:
      max_seq_len: 64
"""
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
import logging
from typing import Any, Dict

from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config, YamlObject
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import _CostModelParser
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory

logger = logging.getLogger(__name__)


class CostModelParserHyperV2(_CostModelParser):
    """Parser for HyperParallel native train.yaml configuration format.

    This parser replaces the placeholder ``CostModelParserHyperparallel``
    which was written for an older TorchTitan TOML format.  It reads the
    current Hyper YAML schema (``model.name`` + ``train.accelerator``)
    and resolves model parameters by importing ``_build_config`` directly
    from the model module.  When the model module is not available it
    falls back to reading hyperparameters from ``config_overrides``.
    """

    def parse(self) -> None:
        """Main parsing entry point."""
        self.ccfg.config_format = "yaml"
        self.ccfg.multimodal = False
        self.ccfg.mm_ccfgs = None
        self.ccfg.mm_order = None

        # Resolve model hyperparameters via Hyper's own config pipeline
        self._resolve_model_config_pipeline()

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
        self.ccfg.offset = 0
        self.ccfg.overwrite_eval_functions = {}

    def _resolve_model_config_pipeline(self):
        """Resolve model hyperparameters to populate ``ccfg``.

        Tries to import ``_build_config`` directly from the model module.
        On failure, falls back to reading hyperparameters from
        ``config_overrides``.  This avoids coupling the ModelSpec
        registration interface to the memory-estimation pipeline.
        """
        try:
            self._resolve_via_direct_import()
        except ValueError:
            logger.debug(
                "Direct _build_config import failed; "
                "falling back to config_overrides"
            )
            self._resolve_from_config_overrides()
        except TypeError as exc:
            raise ValueError(
                f"Failed to build HyperTrainerConfig: {exc}"
            ) from exc

    def _resolve_via_direct_import(self):
        """Import ``_build_config`` directly from the model module.

        Each model module may expose an internal ``_build_config(cfg)``
        function (by naming convention) that returns a model-specific
        Config object.  The parser imports it directly rather than
        going through the ``ModelSpec`` registry.

        Raises:
            ValueError: If the model module or ``_build_config`` is not found.
        """
        # pylint: disable=import-outside-toplevel
        from hyper_parallel.trainer.config import (
            _instantiate_recursive, HyperTrainerConfig,
        )

        config_dict = self._config_to_flat_dict(self.config)
        trainer_cfg = _instantiate_recursive(HyperTrainerConfig, config_dict)
        self.ccfg.model_name = trainer_cfg.model.name

        import importlib  # pylint: disable=import-outside-toplevel
        try:
            mod = importlib.import_module(
                f"hyper_parallel.models.{trainer_cfg.model.name}")
        except ImportError as exc:
            raise ValueError(
                f"Model module '{trainer_cfg.model.name}' not found: {exc}"
            ) from exc

        build_config_fn = getattr(mod, "_build_config", None)
        if build_config_fn is None:
            raise ValueError(
                f"Model module '{trainer_cfg.model.name}' has no "
                f"_build_config function. The parser falls back to "
                f"config_overrides."
            )

        model_config = build_config_fn(trainer_cfg)
        self._map_model_config_to_ccfg(model_config)

    def _map_model_config_to_ccfg(self, model_config) -> None:
        """Map model-specific Config object fields to ``ccfg``."""
        self.ccfg.h = int(model_config.hidden_size)
        self.ccfg.n_lay = int(model_config.num_hidden_layers)
        self.ccfg.a = int(model_config.num_attention_heads)
        self.ccfg.hff = int(model_config.intermediate_size)
        self.ccfg.v = int(model_config.vocab_size)
        self.ccfg.s = int(model_config.max_position_embeddings)

        self.ccfg.n_kv = int(getattr(model_config, "num_key_value_heads", 0))
        if not self.ccfg.n_kv:
            self.ccfg.n_kv = self.ccfg.a
        self.ccfg.dh = self.ccfg.h / self.ccfg.a
        self.ccfg.dc_kv = int(getattr(model_config, "kv_lora_rank", 0))
        self.ccfg.dc_q = int(getattr(model_config, "q_lora_rank", 0))
        self.ccfg.dhr = int(getattr(model_config, "qk_rope_head_dim", 0))

        self.ccfg.n_exp = 1
        self.ccfg.n_chosen_exp = 1
        self.ccfg.n_shared_exp = 0
        self.ccfg.hff_exp = self.ccfg.hff
        self.ccfg.cap_fact = 1
        self.ccfg.t_exp = self.ccfg.t
        self.ccfg.d_exp = self.ccfg.d
        self.ccfg.gmm = False
        self.ccfg.k_1st_dense = 0
        num_exp = int(getattr(model_config, "num_experts", 1))
        if num_exp > 1:
            self.ccfg.n_exp = max(1, num_exp)
            self.ccfg.n_chosen_exp = max(
                1, int(getattr(model_config, "num_experts_per_tok", 1)))
            self.ccfg.n_shared_exp = int(
                getattr(model_config, "n_shared_experts",
                        getattr(model_config, "num_shared_experts", 0)))
            moe_inter = int(getattr(model_config, "moe_intermediate_size", 0))
            if moe_inter:
                self.ccfg.hff_exp = moe_inter
            self.ccfg.k_1st_dense = int(
                getattr(model_config, "first_k_dense_replace", 0))
            self.ccfg.gmm = True

        self.ccfg.n_mtp = int(getattr(model_config, "mtp_depth", 0))
        self.ccfg.is_mtp_in_offset = bool(self.ccfg.n_mtp)
        self.ccfg.multiple_of = int(getattr(model_config, "multiple_of", 256))
        self.ccfg.fdm = float(getattr(model_config, "ffn_dim_multiplier", 1.0))

        self._resolve_device_capacity()

    def _resolve_from_config_overrides(self) -> None:
        """Populate ``ccfg`` directly from ``config_overrides``.

        Used when the model is not registered in ``ModelSpec``.
        """
        model_raw = self._get_cfg_attr(self.config, "model", Config({}))
        overrides = self._get_cfg_attr(model_raw, "config_overrides", Config({}))
        data_raw = self._get_cfg_attr(self.config, "data", Config({}))

        self.ccfg.model_name = str(
            self._get_cfg_attr(model_raw, "name", "custom"))
        self.ccfg.h = int(self._get_cfg_attr(overrides, "hidden_size", 0))
        self.ccfg.n_lay = int(self._get_cfg_attr(overrides, "num_hidden_layers", 0))
        self.ccfg.a = int(self._get_cfg_attr(overrides, "num_attention_heads", 0))
        self.ccfg.hff = int(self._get_cfg_attr(overrides, "intermediate_size", 0))
        self.ccfg.v = int(self._get_cfg_attr(overrides, "vocab_size", 0))

        # seq_len: data.max_seq_len > overrides > default
        self.ccfg.s = int(
            self._get_cfg_attr(data_raw, "max_seq_len", 0)
            or self._get_cfg_attr(overrides, "max_position_embeddings", 0)
            or self._get_cfg_attr(overrides, "seq_length", 0)
            or 4096
        )

        self.ccfg.n_kv = int(
            self._get_cfg_attr(overrides, "num_key_value_heads", 0))
        if not self.ccfg.n_kv:
            self.ccfg.n_kv = self.ccfg.a
        self.ccfg.dh = self.ccfg.h / self.ccfg.a if self.ccfg.a else 0
        self.ccfg.dc_kv = int(
            self._get_cfg_attr(overrides, "kv_lora_rank", 0))
        self.ccfg.dc_q = int(
            self._get_cfg_attr(overrides, "q_lora_rank", 0))
        self.ccfg.dhr = int(
            self._get_cfg_attr(overrides, "qk_rope_head_dim", 0))

        # MoE
        self.ccfg.n_exp = 1
        self.ccfg.n_chosen_exp = 1
        self.ccfg.n_shared_exp = 0
        self.ccfg.hff_exp = self.ccfg.hff
        self.ccfg.cap_fact = 1
        self.ccfg.t_exp = self.ccfg.t
        self.ccfg.d_exp = self.ccfg.d
        self.ccfg.gmm = False
        self.ccfg.k_1st_dense = 0
        num_exp = int(self._get_cfg_attr(overrides, "num_experts", 1))
        if num_exp > 1:
            self.ccfg.n_exp = max(1, num_exp)
            self.ccfg.n_chosen_exp = max(
                1, int(self._get_cfg_attr(overrides, "num_experts_per_tok", 1)))
            self.ccfg.n_shared_exp = int(
                self._get_cfg_attr(overrides, "num_shared_experts", 0))
            moe_inter = int(
                self._get_cfg_attr(overrides, "moe_intermediate_size", 0))
            if moe_inter:
                self.ccfg.hff_exp = moe_inter
            self.ccfg.k_1st_dense = int(
                self._get_cfg_attr(overrides, "first_k_dense_replace", 0))
            self.ccfg.gmm = True

        self.ccfg.n_mtp = int(self._get_cfg_attr(overrides, "mtp_depth", 0))
        self.ccfg.is_mtp_in_offset = bool(self.ccfg.n_mtp)
        self.ccfg.multiple_of = int(
            self._get_cfg_attr(overrides, "multiple_of", 256))
        self.ccfg.fdm = float(
            self._get_cfg_attr(overrides, "ffn_dim_multiplier", 1.0))

        self._resolve_device_capacity()

    def _resolve_device_capacity(self) -> None:
        """Set device capacity from config or default (64 GB)."""
        ctx = self._get_cfg_attr(self.config, "context", Config({}))
        device_mem_str = ctx.__dict__.get("max_device_memory", None) if isinstance(ctx, (Config, YamlObject)) else None
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
            return max(2, int(m.group(1)) // 8)
        return 4

    def _parse_parallelism(self):
        """Extract parallelism settings from ``train.accelerator``."""
        train_raw = self._get_cfg_attr(self.config, "train", Config({}))
        accel = self._get_cfg_attr(train_raw, "accelerator", Config({}))

        dp_shard = int(self._get_cfg_attr(accel, "dp_shard", 1) or 1)
        dp_replicate = int(self._get_cfg_attr(accel, "dp_replicate", 1) or 1)
        tp = int(self._get_cfg_attr(accel, "tp_degree", 1) or 1)
        pp = int(self._get_cfg_attr(accel, "pipeline_parallel_degree", 1) or 1)
        cp = int(self._get_cfg_attr(accel, "context_parallel_degree", 1) or 1)
        ep = int(self._get_cfg_attr(accel, "expert_parallel_degree", 1) or 1)
        etp = int(self._get_cfg_attr(accel, "expert_tensor_parallel_degree", 0) or 0)
        ep = max(ep, 1)

        # FSDP: d = replicate * shard (when shard > 1)
        self.ccfg.d = max(1, dp_replicate * dp_shard)
        self.ccfg.t = max(1, tp)
        self.ccfg.p = max(1, pp)
        self.ccfg.cp = max(1, cp)
        self.ccfg.ep = max(1, ep)
        self.ccfg.sp = self.ccfg.t  # Sequence parallel factor
        self.ccfg.etp = etp
        self.ccfg.vp = max(1, int(
            self._get_cfg_attr(accel, "pp_interleave_num", 1) or 1
        ))
        use_sp = bool(self._get_cfg_attr(accel, "use_seq_parallel", False))
        self.ccfg.sp = self.ccfg.t if use_sp else 1
        self.ccfg.pp_sched = str(
            self._get_cfg_attr(accel, "pipeline_scheduler", "1f1b")
        )

        # Optimizer parallel sharding
        self.ccfg.has_op = bool(self._get_cfg_attr(accel,
                                                     "enable_parallel_optimizer",
                                                     True))
        self.ccfg.op_weight_shard = max(1, int(
            self._get_cfg_attr(accel, "optimizer_weight_shard_size", 0)
        ) or (self.ccfg.d * self.ccfg.t))
        self.ccfg.has_grad_shard = bool(self._get_cfg_attr(accel,
                                                             "gradient_accumulation_shard",
                                                             False))
        self.ccfg.os_max_shard = (
            self.ccfg.op_weight_shard if self.ccfg.op_weight_shard >= 1
            else self.ccfg.d * self.ccfg.t
        )

    def _parse_batch(self):
        """Extract batch settings from ``train`` section."""
        train_raw = self._get_cfg_attr(self.config, "train", Config({}))
        self.ccfg.b = max(1, int(self._get_cfg_attr(train_raw, "micro_batch_size", 1) or 1))
        m = int(self._get_cfg_attr(train_raw, "micro_batch_num", 0) or 0)
        if m > 0:
            self.ccfg.m = m
        else:
            self.ccfg.m = self.ccfg.p
        gbs = int(self._get_cfg_attr(train_raw, "global_batch_size", 0) or 0)
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
        train_raw = self._get_cfg_attr(self.config, "train", Config({}))
        optimizer = self._get_cfg_attr(train_raw, "optimizer", Config({}))
        max_grad_norm = float(
            self._get_cfg_attr(optimizer, "max_grad_norm", 0.0) or 0.0
        )
        self.ccfg.has_clip = max_grad_norm > 0
        self.ccfg.vp_less_mem = False
        self.ccfg.cp_algo = "colossalai_cp"

    def _parse_recompute(self):
        """Parse recompute mode from ``train.gradient_checkpointing``."""
        train_raw = self._get_cfg_attr(self.config, "train", Config({}))
        gc = self._get_cfg_attr(train_raw, "gradient_checkpointing", Config({}))
        ac_mode = str(self._get_cfg_attr(gc, "activation_checkpoint", "none"))
        self.ccfg.full_rec = ac_mode == "full"
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
        """Set FP byte sizes from dtype fields in the model section."""
        model_raw = self._get_cfg_attr(self.config, "model", Config({}))
        self.ccfg.bytes_p = self._bytes_from_dtype(
            self._get_cfg_attr(model_raw, "param_init_type", "float32"))
        self.ccfg.bytes_compute = self._bytes_from_dtype(
            self._get_cfg_attr(model_raw, "compute_dtype", "bfloat16"))
        self.ccfg.bytes_softmax = self._bytes_from_dtype(
            self._get_cfg_attr(model_raw, "softmax_compute_type", "float32"))
        self.ccfg.bytes_grad = 4
        self.ccfg.bytes_os = 4
        self.ccfg.bytes_norm = 4

    def _init_moe_strategy(self):
        """Initialize MoE strategy variables via base helper."""
        self.config_dp_tp_exp(self.ccfg)

    def _init_shard(self):
        """Initialize sharding variables.

        Note: ``shard_output_activ`` and ``shard_recompute_input`` are set to
        ``ccfg.t`` directly instead of ``True`` (which would evaluate to 1).
        This ensures activation memory is correctly divided by the tensor
        parallel degree, matching the behavior of the MF parser.  The
        ``custom_qwen`` arch hook (in ``arch_hooks.py``) would also set these
        to ``ccfg.t`` via ``check_and_apply_custom_hook``, but relying on the
        hook is fragile — it may fail silently when the model name convention
        changes or when the ``set_ccfg`` mechanism is bypassed.
        """
        self.ccfg.shard_embed = self.ccfg.t * self.ccfg.d
        self.ccfg.shard_output_activ = self.ccfg.t
        self.ccfg.shard_recompute_input = self.ccfg.t
        self.ccfg.is_shard_mtp_param = True
