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
(``hyper_parallel/models/qwen3_moe/recipes/train.yaml``) and
populates a :class:`CostModelConfig` for memory estimation.

Model hyperparameters are resolved with the same Transformers
``AutoConfig.from_pretrained`` path used by AutoModels. Legacy
``model.config_overrides`` remains supported for standalone search configs.

Expected YAML structure.  Only two things are required: the checkpoint to
read the dimensions from, and the training sequence length::

    model:
      _target_: hyper_parallel.models._transformers.HyperAutoModelForCausalLM.from_pretrained
      pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
      torch_dtype: bfloat16          # optional, default bfloat16

    dataset:
      data_transform:
        max_seq_len: 4096            # or the legacy data.max_seq_len
                                     # absent: the model's CONTEXT LIMIT is
                                     # costed, which is rarely what you meant

Everything below is optional, and on the search path it is ignored: the
search varies these dimensions itself, and the device count, the batch size
and the memory budget come from ``-d``, ``-b`` and ``-M`` (or from the
search config).  Give them only to cost one fixed strategy::

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

    context:
      max_device_memory: "64GB"
      device_num: 64
      census: true                   # price each layer kind's activations
                                     # from a census of a fake layer of it
      expert_shard: group            # shard each strategy's routed experts
                                     # over its whole expert DP group

``model.config_overrides`` stays supported for standalone search configs,
and wins over anything read from the checkpoint.
"""
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
import logging
from typing import Any, Dict, Optional, Tuple

from hyper_parallel.auto_parallel.sapp_nd.nd.balancing_adapter import front_loaded_offset
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config, YamlObject
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import _CostModelParser
from hyper_parallel.auto_parallel.sapp_nd.nd.common.apply_exec import apply_exec
from hyper_parallel.auto_parallel._exec_spec import ExecSpec
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import read_activation_checkpoint_mode
from hyper_parallel.auto_parallel._hf_model_spec import (
    is_auto_models_schema,
    resolve_hf_model_spec,
)
from hyper_parallel.auto_parallel._layer_census import replacement_specs
from hyper_parallel.auto_parallel._layer_stack import LinearAttentionDims, resolve_layers
from hyper_parallel.auto_parallel._model_spec import (
    KindActivations,
    activations_from_dict,
    layers_from_list,
    ops_from_dict,
)
from hyper_parallel.auto_parallel._op_profiles import VISION_ARCH

logger = logging.getLogger(__name__)

# The legacy precision keys of a model section, which the AutoModels trainer
# hands on to the model's Transformers config and never reads, by what it
# reads instead.
_UNREAD_PRECISION_KEYS = {
    "param_init_type": "the run keeps its weights in model_init_dtype, else model.torch_dtype",
    "compute_dtype": (
        "the run computes in fsdp_config.mix_precision.param_dtype, else model_init_dtype, "
        "else model.torch_dtype"
    ),
}


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
        self._model_seq_len = 0
        self._tie_word_embeddings = False

        # The model, through AutoModels' Transformers pipeline.
        self._resolve_model_config_pipeline()

        # The run, as the train yaml states it.  An expert layout the
        # degrees cannot hold is clamped rather than refused: the search can
        # still reach it, and rejects it by memory.
        apply_exec(self.ccfg, self._exec_spec(), strict=False)
        self.ccfg.moe_dispatch = self._moe_dispatch()
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
        The AutoModels trainer trains the model Transformers builds, which
        has no MTP layer (:meth:`_without_mtp`), and a vision tower only
        where its model class builds one (:meth:`_builds_vision_tower`).
        ``context.census`` has the resolver run a census of each layer kind
        (:meth:`_config_census`), of the modules the run's ``plan_overrides``
        install (:meth:`_replacements`), and ``context.census_spec`` has it
        state the census's spec of the model rather than its own.
        """
        spec = resolve_hf_model_spec(
            self._model_section(), self._visual_seq_len_override(), self._census_seq_len(), self._census_spec(),
            self._replacements(),
        )
        if is_auto_models_schema(self.config):
            # A stated depth looked priced and was not, and nothing said so
            # (I2): a search repeats this under its ranking.
            self.ccfg.mtp_unpriced = self._spec_int(spec, "mtp_depth")
            if self.ccfg.mtp_unpriced:
                logger.warning(
                    "The model states %d MTP layer(s), which the AutoModels trainer does not build: "
                    "ND prices none", self.ccfg.mtp_unpriced,
                )
            spec = self._without_mtp(spec)
        self._vision_spec = spec.pop("vision", None)
        if self._vision_spec and not self._builds_vision_tower():
            self._vision_spec = None
        self._tie_word_embeddings = bool(spec.get("tie_word_embeddings"))
        # The dtype the spec states, the checkpoint's own where the section
        # states none: the runtime loads the weights in it (_model_dtype).
        self._spec_dtype = spec.get("torch_dtype")
        self._apply_spec(self.ccfg, spec)
        ops = None if spec.get("ops") is None else ops_from_dict(spec["ops"])
        self.config_op_counts(self.ccfg, spec["arch"], ops)
        self.config_layer_stack(self.ccfg, resolve_layers(
            spec["arch"], layers_from_list(spec["layers"]), ops,
            LinearAttentionDims.from_fields(spec),
        ))
        self._config_census(spec)
        self._model_seq_len = self._spec_int(spec, "max_position_embeddings")

    def _config_census(self, spec: Dict[str, Any]) -> None:
        """Hold the census the spec states, the record of a stack of one kind, and the output layer's.

        The memory model prices a layer whose kind has a record with it,
        rather than with its formulas, and the output layer with its own.
        A stack of several kinds binds each kind's record as the layer
        priced becomes one of it (``arch_hooks``).
        """
        census, output = spec.get("activations"), spec.get("output_activations")
        self.ccfg.census = activations_from_dict(census) if census else None
        self.ccfg.output_census = KindActivations.from_dict(output, "output_activations") if output else None
        kinds = self.ccfg.layer_stack.distinct_kinds()
        single = len(kinds) == 1 and self.ccfg.census
        self.ccfg.kind_activations = self.ccfg.census.get(kinds[0].name) if single else None

    def _census_seq_len(self) -> int:
        """The tokens to run a census of each layer kind at, 0 unless ``context.census`` asks for one.

        A census gives bytes per token, which hardly depend on the length:
        it runs at the dataset's, else at 4096 tokens, never at the model's
        context limit.
        """
        ctx = self._get_cfg_attr(self.config, "context", Config({}))
        if not self._get_cfg_attr(ctx, "census", False):
            return 0
        return self._dataset_seq_len() or 4096

    def _census_spec(self) -> bool:
        """Whether ``context.census_spec`` asks for the census's spec of the model rather than the resolver's."""
        ctx = self._get_cfg_attr(self.config, "context", Config({}))
        return bool(self._get_cfg_attr(ctx, "census_spec", False))

    def _replacements(self) -> Tuple[Any, ...]:
        """The module replacements the run's ``plan_overrides`` install, which a census runs its layers with."""
        entries = self._config_to_flat_dict(self._get_cfg_attr(self.config, "plan_overrides", None))
        return replacement_specs(entries if isinstance(entries, list) else ())

    # The AutoModels classes that build a vision-language checkpoint's
    # vision tower; every other named class builds the language model alone.
    _VISION_TARGETS = ("ImageTextToText", "Vision2Seq", "ConditionalGeneration")

    def _builds_vision_tower(self) -> bool:
        """Whether the run's model class builds the vision tower its checkpoint has.

        HyperParallel's AutoModels trainer builds ``model._target_``:
        ``HyperAutoModelForImageTextToText`` builds the tower, while
        ``HyperAutoModelForCausalLM`` and the recipes' builders build the
        language model alone and load the tower's weights as unexpected.
        A yaml that names no class (the trainer requires one) and the
        legacy schema price what the checkpoint has.
        """
        if not is_auto_models_schema(self.config):
            return True
        model_raw = self._get_cfg_attr(self.config, "model", Config({}))
        target = self._get_cfg_attr(model_raw, "_target_", None)
        return not target or any(name in str(target) for name in self._VISION_TARGETS)

    @staticmethod
    def _without_mtp(spec: Dict[str, Any]) -> Dict[str, Any]:
        """Return *spec* without its MTP layers, as the AutoModels trainer builds the model.

        It builds the Transformers causal LM, which has no MTP layer whatever
        the checkpoint declares: Transformers loads their weights as
        unexpected and never trains them (Qwen3.5's ``mtp.*``, DeepSeek-V3's
        last layer).  The legacy schema's trainer is not this one.
        """
        layers = spec.get("layers")
        if isinstance(layers, list):
            spec["layers"] = [group for group in layers if not (isinstance(group, dict) and group.get("mtp"))]
        spec["mtp_depth"] = 0
        return spec

    def _model_section(self) -> Dict[str, Any]:
        """Return the ``model`` section as a plain mapping, ``_target_`` included.

        ``_config_to_flat_dict`` drops every key beginning with an
        underscore, which is right for the private fields a ``Config``
        carries and wrong for exactly one: ``_target_`` names the factory
        that builds the model.  Stripping it left the resolver reading the
        directory beside it as a checkpoint, and for a model Transformers
        cannot build, falling back to a hand-written description of a config
        that was one call away.
        """
        section = self._get_cfg_attr(self.config, "model", Config({}))
        model_raw = self._config_to_flat_dict(section)
        if not isinstance(model_raw, dict):
            return {}
        target = self._get_cfg_attr(section, "_target_", None)
        if target:
            model_raw["_target_"] = str(target)
        return model_raw

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
        ccfg.s = self._spec_int(spec, "max_position_embeddings") or 4096
        ccfg.n_kv = self._spec_int(spec, "num_key_value_heads") or ccfg.a
        # Transformers exposes head_dim explicitly and it is not always
        # hidden_size / num_attention_heads (Qwen3 has h/a = 64, head_dim 128).
        head_dim = self._spec_int(spec, "head_dim")
        ccfg.dh = head_dim if head_dim else (ccfg.h / ccfg.a if ccfg.a else 0)
        ccfg.v_head_dim = self._spec_int(spec, "v_head_dim") or None
        ccfg.qk_nope_head_dim = self._spec_int(spec, "qk_nope_head_dim") or None
        ccfg.dc_kv = self._spec_int(spec, "kv_lora_rank")
        ccfg.dc_q = self._spec_int(spec, "q_lora_rank")
        ccfg.dhr = self._spec_int(spec, "qk_rope_head_dim")
        # Qwen3.5 fuses the output gate into q_proj, doubling its width.
        ccfg.attn_output_gate = bool(spec.get("attn_output_gate", False))
        # Qwen3 normalizes each head's queries and keys.
        ccfg.qk_norm = bool(spec.get("qk_norm", False))
        # The biases and norms the spec states; unstated, the parameter
        # formulas count their own.
        for name in ("qkv_bias", "o_bias", "mlp_bias", "norm_bias", "shared_expert_gate"):
            setattr(ccfg, name, None if spec.get(name) is None else bool(spec[name]))
        ccfg.layer_norms = self._spec_int(spec, "layer_norms") or None

    def _apply_moe_spec(self, ccfg: Any, spec: Dict[str, Any]) -> None:
        """Map dense defaults and optional MoE fields."""
        ccfg.n_exp = 1
        ccfg.n_chosen_exp = 1
        ccfg.n_shared_exp = 0
        ccfg.hff_exp = ccfg.hff
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

    def _apply_scaling_spec(self, ccfg: Any, spec: Dict[str, Any]) -> None:
        """Map MTP and feed-forward scaling fields."""
        ccfg.n_mtp = self._spec_int(spec, "mtp_depth")
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
        # Each submodule is priced by its own arch.
        self.ccfg.hooks_dict = None
        self.ccfg.n_lay = 0
        self.ccfg.layer_stack = None
        logger.info(
            "Multimodal cost model: vision %s (%d layers, s=%d) + text %s "
            "(%d layers, s=%d)",
            vision_ccfg.model_name, vision_ccfg.n_lay, vision_ccfg.s,
            text_ccfg.model_name, text_ccfg.n_lay, text_ccfg.s,
        )

    def _clone_submodule(self, name: str) -> Any:
        """Return a submodule cost config seeded from the parsed parent.

        The loop below copies references, so every mutable container a
        submodule writes to has to be rebuilt: each submodule is priced by
        its own family and must not see the others' overrides.
        """
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
        cc.overwrite_eval_functions = dict(self.ccfg.overwrite_eval_functions)
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
        # A tower is priced with the vision profile.  Its name carries the
        # language model's type: the family it implies is the one whose
        # activation sharding the tower takes.
        self.config_op_counts(cc)
        cc.inherited_arch, cc.arch = cc.arch, VISION_ARCH
        cc.v = 0  # patch embedding, not a vocabulary table
        cc.n_mtp = 0
        cc.layer_binding = None
        # The language model's census is not the tower's.
        cc.census = cc.kind_activations = cc.output_census = None
        self.config_layer_stack(cc, resolve_layers(
            VISION_ARCH, layers_from_list(vision_spec.get("layers"), "vision.layers"),
        ))
        # The tower runs the language model's strategy with no vocabulary
        # embedding, no MTP layer and no experts, all on the first stage.
        # Its own facts set its derived fields, not those of the language
        # model it was cloned from.
        apply_exec(cc, ExecSpec(
            vocab_emb_dp=False, tie_embeddings=False, mtp_in_offset=False, grouped_gemm=False,
            capacity_factor=1, offset=self._front_loaded_offset(cc.n_lay), loss_parallel=True,
        ), strict=False)
        return cc

    def _even_offset(self):
        """Return a balanced offset of the shape ``is_consistent_pp_config`` wants."""
        if self.ccfg.vp > 1:
            return [[0] * self.ccfg.p for _ in range(self.ccfg.vp)]
        return [0] * self.ccfg.p

    def _front_loaded_offset(self, n_lay: int):
        """Return an offset placing every layer on the first pipeline stage."""
        return front_loaded_offset(n_lay, self.ccfg.p, self.ccfg.vp)

    def _dataset_seq_len(self) -> int:
        """The sequence length the dataset states, the legacy ``data.max_seq_len`` too, else 0.

        The Online path states ``dataset.data_transform.max_seq_len`` and an
        Indexed Dataset ``dataset.data_config.seq_length``; both describe the
        dataset that runs, so they outrank the deprecated ``data.max_seq_len``.
        """
        data_raw = self._get_cfg_attr(self.config, "data", Config({}))
        legacy_seq_len = self._get_cfg_attr(data_raw, "max_seq_len", 0)

        dataset_raw = self._get_cfg_attr(self.config, "dataset", Config({}))
        transform_raw = self._get_cfg_attr(
            dataset_raw, "data_transform", Config({}),
        )
        trainer_seq_len = self._get_cfg_attr(transform_raw, "max_seq_len", 0)
        # Only the Online path carries data_transform.max_seq_len; an Indexed
        # Dataset states its length as data_config.seq_length.
        data_config_raw = self._get_cfg_attr(dataset_raw, "data_config", Config({}))
        indexed_seq_len = self._get_cfg_attr(data_config_raw, "seq_length", 0)
        return int(trainer_seq_len or indexed_seq_len or legacy_seq_len or 0)

    def _resolve_sequence_length(self) -> int:
        """The training sequence length: the dataset's, else the model's limit.

        A model section that states no limit falls back on
        ``config_overrides.seq_length``, then on 4096.
        """
        model_seq_len = self._model_seq_len or int(
            self._get_cfg_attr(self._config_overrides(), "seq_length", 0) or 0
        )
        stated = self._dataset_seq_len()
        # The ranking says so beside its table, where a reader looks (I11).
        self.ccfg.seq_len_stated = bool(stated)
        if not stated:
            # The fallback is the model's context limit, orders of magnitude
            # above any real training length on a long-context model, which
            # makes the quadratic attention term meaningless.
            logger.warning(
                "no training sequence length in the config; costing the model "
                "context limit of %d. Set dataset.data_transform.max_seq_len "
                "(Online) or dataset.data_config.seq_length (Indexed).",
                int(model_seq_len or 4096),
            )
        return int(stated or model_seq_len or 4096)

    def _resolve_device_capacity(self) -> str:
        """The device's memory: ``context.max_device_memory``, else 64 GB."""
        ctx = self._get_cfg_attr(self.config, "context", Config({}))
        device_mem_str = (
            ctx.__dict__.get("max_device_memory", None)
            if isinstance(ctx, (Config, YamlObject))
            else None
        )
        return str(device_mem_str) if device_mem_str else "64GB"

    def _moe_dispatch(self) -> Optional[float]:
        """What a MoE layer's token dispatch costs on this cluster, ``context.moe_dispatch``.

        A run that measured its own states the number, and a model whose
        compute does not grow with EP states 0, which switches the term off;
        unstated (None), the estimate keeps the one measured on A3
        (``estimate.MOE_DISPATCH``).
        """
        ctx = self._get_cfg_attr(self.config, "context", Config({}))
        stated = self._get_cfg_attr(ctx, "moe_dispatch", None)
        if stated is not None and float(stated) < 0:
            raise ValueError(f"context.moe_dispatch takes a cost, not {stated!r}")
        return float(stated) if stated is not None else None

    def _exec_spec(self) -> ExecSpec:
        """State the run the train yaml describes.

        Read after the model, whose expert count, MTP depth and context limit
        set the run's defaults.
        """
        stated = self._parse_parallelism()
        stated.update(self._parse_batch(stated["dp"], stated["pp"]))
        stated.update(self._parse_feature_flags(stated["cp"]))
        stated.update(self._parse_recompute())
        stated.update(self._init_bytes())
        stated.update(self._init_moe_run(stated["etp"]))
        stated.update(self._init_shard())
        stated["offset"] = self._init_offset(stated["pp"], stated["vpp"])
        stated["seq_split"] = 1
        # Match the MF parser: MTP layers the model declares take part in
        # pipeline offset balancing, and without any the offset leaves them
        # out, as the MF parser's num_nextn_predict_layers fallback does.
        stated["mtp_in_offset"] = bool(self.ccfg.n_mtp)
        stated["seq_length"] = self._resolve_sequence_length()
        stated["device_memory"] = self._resolve_device_capacity()
        return ExecSpec(**stated)

    def _config_overrides(self) -> Any:
        """The model section's ``config_overrides``, empty when it has none."""
        model_raw = self._get_cfg_attr(self.config, "model", Config({}))
        return self._get_cfg_attr(model_raw, "config_overrides", Config({}))

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

    def _parse_parallelism(self) -> Dict[str, Any]:
        """The degrees and sharding the AutoModels or legacy Trainer schema states."""
        train_raw = self._get_cfg_attr(self.config, "train", Config({}))
        legacy_accel = self._get_cfg_attr(train_raw, "accelerator", Config({}))
        accel = self._get_cfg_attr(self.config, "accelerator", legacy_accel)
        fsdp = self._get_cfg_attr(self.config, "fsdp_config", Config({}))

        stated, dp_shard = self._parse_parallel_dimensions(accel, fsdp)
        stated.update(self._parse_sequence_parallelism(accel))
        stated.update(self._parse_optimizer_parallelism(accel, dp_shard, stated["dp"]))
        stated.update(self._parse_expert_sharding(fsdp))
        return stated

    def _parse_expert_sharding(self, fsdp) -> Dict[str, Any]:
        """How HyperParallel's FSDP shards a routed expert under expert parallelism.

        Over ``edp_shard_size`` ranks of its expert data-parallel group, 1 by
        default: a run that states none keeps each of its experts whole on
        every rank holding it.  The legacy schema states nothing, and the
        family's rule applies.

        One ``edp_shard_size`` cannot state what a launcher does that sets it,
        strategy by strategy, to the whole expert data-parallel group, since
        that group changes with EP; ``context.expert_shard: group`` states it.
        """
        if not is_auto_models_schema(self.config):
            return {}
        ctx = self._get_cfg_attr(self.config, "context", Config({}))
        rule = self._get_cfg_attr(ctx, "expert_shard", None)
        if rule not in (None, "group"):
            raise ValueError(f"context.expert_shard takes 'group' or nothing, not {rule!r}")
        return {
            "expert_shard": max(1, int(self._get_cfg_attr(fsdp, "edp_shard_size", 1) or 1)),
            "expert_shard_group": rule == "group",
        }

    def _parse_parallel_dimensions(self, accel, fsdp) -> Tuple[Dict[str, Any], int]:
        """Return the mesh's degrees, and the data shard degree."""

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

        degrees = {"tp": max(1, tp), "pp": max(1, pp), "cp": max(1, cp), "ep": max(1, ep)}
        degrees["dp"] = self._resolve_data_parallel(
            dp_replicate, dp_shard, degrees["tp"] * degrees["pp"] * degrees["cp"]
        )
        degrees["etp"] = etp
        degrees["vpp"] = max(1, int(
            self._get_cfg_attr(accel, "pp_interleave_num", 1) or 1
        ))
        return degrees, dp_shard

    def _resolve_data_parallel(self, dp_replicate: int, dp_shard: int, denom: int) -> int:
        """Return the data-parallel degree, preferring an explicit device count.

        ND reads ``d * t * cp * p`` back as the cluster size whenever no
        device count is supplied on the command line. The AutoModels schema
        has no replicate field, since the runtime derives it from the world
        size, so without ``context.device_num`` an HSDP run would understate
        the cluster by exactly its replicate factor.

        Without ``context.device_num``, an AutoModels config takes the device
        count its caller states (``run_nd -d``), as the runtime takes the
        world size its launcher gives it.  Read from ``dp_shard_size`` alone,
        a one-node yaml searched on more devices kept its one node's degree,
        and its micro-batch count with it, in every candidate (M6).

        Args:
            dp_replicate: The replicate degree the config states.
            dp_shard: The shard degree the config states.
            denom: The product ``t * p * cp`` of the other degrees.
        """
        ctx = self._get_cfg_attr(self.config, "context", Config({}))
        device_num = int(self._get_cfg_attr(ctx, "device_num", 0) or 0)
        source = "context.device_num"
        if not device_num and is_auto_models_schema(self.config):
            device_num = int(getattr(self.ccfg, "devices", 0) or 0)
            source = "the device count (-d)"
        if not device_num:
            if is_auto_models_schema(self.config) and dp_replicate == 1:
                logger.warning(
                    "AutoModels config carries no context.device_num; assuming "
                    "dp_replicate=1 (d = dp_shard_size = %d). Pass -d/--devices "
                    "for HSDP runs.", dp_shard,
                )
            return max(1, dp_replicate * dp_shard)
        if denom < 1 or device_num % denom:
            raise ValueError(
                f"{source}={device_num} is not divisible by "
                f"t*p*cp={denom}"
            )
        return max(1, device_num // denom)

    def _parse_sequence_parallelism(self, accel) -> Dict[str, Any]:
        """Sequence parallelism and the pipeline scheduler."""
        use_sp = bool(
            self._get_cfg_attr(accel, "sequence_parallel", False)
            or self._get_cfg_attr(accel, "use_seq_parallel", False)
        )
        return {
            "sequence_parallel": use_sp,
            "pp_schedule": str(self._get_cfg_attr(accel, "pipeline_scheduler", "1f1b")),
        }

    def _parse_optimizer_parallelism(self, accel, dp_shard: int, dp: int) -> Dict[str, Any]:
        """Optimizer and gradient sharding.

        HyperParallel's FSDP holds every gradient sharded as its parameter,
        from the first backward to the optimizer step, whatever the pipeline
        degree.
        """
        is_auto_models = is_auto_models_schema(self.config)
        optimizer_parallel = (
            dp_shard > 1
            if is_auto_models
            else bool(self._get_cfg_attr(
                accel, "enable_parallel_optimizer", True,
            ))
        )
        weight_shard = max(1, int(
            self._get_cfg_attr(accel, "optimizer_weight_shard_size", 0)
        ) or (dp_shard if is_auto_models else dp))
        return {
            "optimizer_parallel": optimizer_parallel,
            "optimizer_shard": weight_shard,
            "grad_shard": bool(self._get_cfg_attr(accel, "gradient_accumulation_shard", False)),
            "grad_shard_as_params": True,
            # Its dp_shard_size spans the DP x CP domain, so CP shards no
            # parameter state beyond it (derive.param_cp).
            "shard_spans_cp": is_auto_models,
            "grad_accumulation": True,
            # It adds each layer's reduce-scatter output to the accumulated
            # gradient only in the root's backward hook.
            "deferred_grad_accumulation": True,
            # It reduce-scatters a layer's gradients while the next layer's
            # backward runs, and the root's in its backward hook.
            "overlapped_grad_reduce": True,
            "reshard_params": self._reshards_params(),
            # It gathers the weights in the compute dtype, and a layer keeps
            # no cast of them, whatever dp_shard.
            "param_casts": False,
        }

    def _reshards_params(self) -> bool:
        """Whether HyperParallel's FSDP frees a layer's gathered parameters once it has run.

        It does after the layer's forward and after its backward, unless the
        run keeps them gathered through either.
        """
        fsdp = self._get_cfg_attr(self.config, "fsdp_config", Config({}))
        return bool(
            self._get_cfg_attr(fsdp, "reshard_after_forward", True)
            and self._get_cfg_attr(fsdp, "reshard_after_backward", True)
        )

    def _parse_batch(self, dp: int, pp: int) -> Dict[str, Any]:
        """Batch settings from ``training`` or legacy ``train``."""
        legacy_train = self._get_cfg_attr(self.config, "train", Config({}))
        train_raw = self._get_cfg_attr(self.config, "training", legacy_train)
        micro = max(1, int(self._get_cfg_attr(train_raw, "micro_batch_size", 1) or 1))
        num = int(self._get_cfg_attr(train_raw, "micro_batch_num", 0) or 0)
        gbs = int(self._get_cfg_attr(train_raw, "global_batch_size", 0) or 0)
        if num <= 0:
            if gbs > 0 and gbs % (micro * dp) == 0:
                num = max(1, gbs // (micro * dp))
            else:
                num = pp
        return {
            "micro_batch_size": micro,
            "micro_batch_num": num,
            "global_batch_size": gbs if gbs > 0 else micro * dp * num,
        }

    def _parse_feature_flags(self, cp: int) -> Dict[str, Any]:
        """Training features: attention kernel, clipping, CP algorithm, optimizer."""
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
        accel = self._get_cfg_attr(
            self.config,
            "accelerator",
            self._get_cfg_attr(legacy_train, "accelerator", Config({})),
        )
        cp_algo = self._get_cfg_attr(accel, "context_parallel_algo", None)
        if not cp_algo:
            cp_algo = "colossalai_cp"
            if cp and cp > 1:
                logger.warning(
                    "context_parallel_algo not set; defaulting to "
                    "'colossalai_cp' (Ring CP). Set "
                    "train.accelerator.context_parallel_algo explicitly "
                    "to 'ulysses_cp' if Ulysses CP is intended."
                )
        # The optimizer, by its target, which decides the states it keeps a
        # parameter; the MindFormers parser names it by its type.
        opt_name, opt_states = self.stated_optimizer(
            self._get_cfg_attr(optimizer, "_target_", None) or self._get_cfg_attr(optimizer, "type", None)
        )
        return {
            "flash_attention": True,
            "vocab_emb_dp": True,
            "tie_embeddings": self._tie_word_embeddings,
            "frozen": False,
            "grad_clip": max_grad_norm > 0,
            # The trainer's lm_head gathers the logits whole on every TP rank
            # unless the loss runs on them sharded.
            "loss_parallel": bool(self._get_cfg_attr(accel, "loss_parallel", False)),
            "cp_algo": cp_algo,
            "optimizer": opt_name,
            **self._optimizer_states(optimizer, opt_states),
        }

    def _optimizer_states(self, optimizer: Any, states: int) -> Dict[str, Any]:
        """What HyperParallel's optimizer keeps per parameter.

        Its AdamW keeps two moments and its Muon one momentum per matrix
        (``stated_optimizer`` counts them), each ``zeros_like`` the gradient,
        which FSDP casts to the stored parameter's dtype.  With
        ``fp32_main_params`` the optimizer keeps an fp32 copy of each
        narrower parameter, and its states in fp32.
        """
        stored = self._stored_param_bytes()
        fp32_main = bool(self._get_cfg_attr(optimizer, "fp32_main_params", False))
        return {
            "optimizer_states": states,
            "optimizer_state_bytes": 4 if fp32_main else stored,
            "main_param_bytes": 4 if fp32_main and stored < 4 else 0,
        }

    def _stored_param_bytes(self) -> int:
        """The width FSDP stores the parameters in: the model's, whatever FSDP gathers them in."""
        return self._bytes_from_dtype(
            self._get_cfg_attr(self.config, "model_init_dtype", None)
            or self._model_dtype("param_init_type")
        )

    def _model_dtype(self, legacy_key: str) -> str:
        """The dtype the runtime loads the model's weights in, by name.

        ``model.torch_dtype`` where the section states one other than
        ``auto``; else, in a legacy config, the legacy *legacy_key*
        (``param_init_type`` for the parameters, ``compute_dtype`` for the
        compute), which the AutoModels trainer does not read (D17); else the
        dtype the spec states, which is the checkpoint's own, as the runtime
        loads the weights under its default ``auto``; else bfloat16
        (``_init_bytes`` warns).  A model stating none was priced in float32
        parameters doing bfloat16 arithmetic, where the runtime loads its
        checkpoint's, mostly two bytes (I12).
        """
        model_raw = self._get_cfg_attr(self.config, "model", Config({}))
        stated = self._get_cfg_attr(model_raw, "torch_dtype", None)
        if stated and stated != "auto":
            return str(stated)
        legacy = None if is_auto_models_schema(self.config) else self._get_cfg_attr(model_raw, legacy_key, None)
        return str(legacy or getattr(self, "_spec_dtype", None) or "bfloat16")

    def _parse_recompute(self) -> Dict[str, Any]:
        """Parse recompute mode.

        Reads ``activation_checkpoint.mode`` from the AutoModels schema, with
        the legacy ``train.gradient_checkpointing`` path as a fallback. When
        ``config_overrides`` supplies ``full_rec`` or ``sel_rec`` (matching
        the MF parser's ``recompute_config.recompute`` /
        ``recompute_config.select_recompute`` fields), those values take
        precedence so that Hyper YAML demo files can express per-stage
        recompute lists for side-by-side comparisons with MindFormers.
        """
        overrides = self._config_overrides()
        full_rec_override = self._get_cfg_attr(overrides, "full_rec", None)
        sel_rec_override = self._get_cfg_attr(overrides, "sel_rec", None)

        train_raw = self._get_cfg_attr(self.config, "train", Config({}))
        gc = self._get_cfg_attr(train_raw, "gradient_checkpointing", Config({}))
        activation_checkpoint = self._get_cfg_attr(
            self.config, "activation_checkpoint", Config({}),
        )
        # Read as the trainer reads it: an unquoted off, which YAML reads as
        # False, is off rather than a fall through to the legacy key, and a
        # name that is no mode is refused rather than priced as off (I13).
        stated, where = self._get_cfg_attr(activation_checkpoint, "mode", None), "activation_checkpoint.mode"
        if stated is None:
            stated = self._get_cfg_attr(gc, "activation_checkpoint", None)
            where = "train.gradient_checkpointing.activation_checkpoint"
        ac_mode = read_activation_checkpoint_mode(stated, where)
        # The mode the run trains with, which a search deriving its own
        # recompute names under its ranking where the two differ (H1).
        overridden = full_rec_override is not None or sel_rec_override is not None
        self.ccfg.stated_ac_mode = None if overridden else ac_mode
        return {
            "full_recompute": full_rec_override if full_rec_override is not None else ac_mode == "full",
            "selective_recompute": sel_rec_override if sel_rec_override is not None else ac_mode == "selective",
            "selective_rule": "hyperparallel",
        }

    def _init_bytes(self) -> Dict[str, Any]:
        """FP byte sizes from AutoModels or legacy dtype fields.

        ``model_init_dtype`` is a top-level AutoModels key applied after the
        weights are loaded, so it outranks ``model.torch_dtype`` for the
        stored parameters but not an explicit FSDP ``param_dtype``, and the
        run computes in the same width: FSDP gathers each parameter in its
        ``param_dtype``, else in the width it is kept in (D17).  Where
        nothing states a dtype, the checkpoint's own is taken
        (:meth:`_model_dtype`), and bfloat16 with a warning where there is
        no checkpoint either.
        """
        model_raw = self._get_cfg_attr(self.config, "model", Config({}))
        fsdp = self._get_cfg_attr(self.config, "fsdp_config", Config({}))
        mix_precision = self._get_cfg_attr(fsdp, "mix_precision", Config({}))
        init_dtype = self._get_cfg_attr(self.config, "model_init_dtype", None)
        param_dtype = self._get_cfg_attr(mix_precision, "param_dtype", None)
        param_bytes = self._bytes_from_dtype(
            param_dtype
            or init_dtype
            or self._model_dtype("param_init_type")
        )
        auto_models = is_auto_models_schema(self.config)
        stated = [self._get_cfg_attr(model_raw, key, None)
                  for key in (("torch_dtype",) if auto_models else ("torch_dtype", "param_init_type"))]
        if (not any(value and value != "auto" for value in stated) and not init_dtype
                and not getattr(self, "_spec_dtype", None)):
            logger.warning(
                "neither the config nor a checkpoint states the model's dtype: its weights are priced in "
                "bfloat16; state model.torch_dtype if they are held in another"
            )
        for key, read_instead in _UNREAD_PRECISION_KEYS.items():
            if auto_models and self._get_cfg_attr(model_raw, key, None):
                logger.warning(
                    "the AutoModels trainer reads no model.%s, and neither does ND: %s", key, read_instead,
                )
        return {
            "param_bytes": param_bytes,
            # FSDP keeps each gradient in its parameter's dtype.
            "grad_bytes": param_bytes,
            "compute_bytes": self._bytes_from_dtype(param_dtype or init_dtype or self._model_dtype("compute_dtype")),
            "softmax_bytes": self._bytes_from_dtype(
                self._get_cfg_attr(model_raw, "softmax_compute_type", "float32")),
        }

    def _init_moe_run(self, etp: int) -> Dict[str, Any]:
        """The run of the expert layers: grouped GEMM, capacity, expert TP.

        A dense model runs no grouped GEMM and a capacity factor of 1.  For
        a MoE model (``n_exp > 1``) both come from ``config_overrides``, and
        ``etp`` defaults to 1 when the YAML states none, matching the MF
        parser's ``expert_model_parallel`` default; a dense model's ``etp=0``
        leaves ``t_exp = t, d_exp = d``.
        """
        if self.ccfg.n_exp <= 1:
            return {"grouped_gemm": False, "capacity_factor": 1}
        overrides = self._config_overrides()
        cap_val = self._get_cfg_attr(
            overrides, "capacity_factor", self._get_cfg_attr(overrides, "cap_fact", None),
        )
        run = {
            "grouped_gemm": bool(self._get_cfg_attr(
                overrides, "use_gmm", self._get_cfg_attr(overrides, "gmm", True),
            )),
            "capacity_factor": 1 if cap_val is None else max(1, float(cap_val)),
        }
        if etp == 0:
            run["etp"] = 1
        return run

    def _init_offset(self, pp: int, vpp: int = 1) -> Any:
        """The pipeline offset.

        The MF parser reads ``model.model_config.offset`` directly from the
        YAML.  When it is a list (e.g. ``[1, 1, ..., -1]``),
        ``CostModelConfig.is_consistent_pp_config`` requires
        ``len(offset) == pp``, so strategies whose pipeline degree differs
        are rejected until ``GlobalConfig.adapt_config`` regenerates a
        matching offset.  A scalar ``0`` is always accepted.

        To match the MF parser's *list*-based filtering behaviour (used by
        DeepSeek-V3 and other models that declare an explicit offset), this
        parser states a list offset of length ``pp`` by default, one that
        places every layer (:meth:`_balanced_offset`).  An explicit offset supplied via
        ``config_overrides.offset`` overrides this: a list is used as-is,
        and a non-zero int is broadcast to ``[int] * pp``.
        """
        model_raw = self._get_cfg_attr(self.config, "model", Config({}))
        explicit = self._get_cfg_attr(self._config_overrides(), "offset", None)
        if explicit is None:
            explicit = self._get_cfg_attr(model_raw, "offset", None)
        if isinstance(explicit, list):
            return list(explicit)
        if isinstance(explicit, int):
            return 0 if explicit == 0 else [explicit] * pp
        return self._balanced_offset(pp, vpp)

    def _balanced_offset(self, pp: int, vpp: int = 1) -> list:
        """An offset of length *pp* that places every layer the pipeline balances.

        Each stage runs the layers per stage, and the first ones one more
        each until the remainder has a stage, as the search's balancing
        places them; all zeros where the pipeline divides the layers, or
        interleaves, which the search balances itself.
        """
        layers = self.ccfg.n_lay + (self.ccfg.n_mtp or 0)
        extra = layers % max(1, pp) if vpp <= 1 else 0
        return [1 if stage < extra else 0 for stage in range(pp)]

    def _init_shard(self) -> Dict[str, Any]:
        """How the embedding and the recompute input are sharded.

        The embedding is sharded over data parallelism, as under the MF
        parser.  ``recompute_slice_activation`` mirrors the MF parser's
        ``recompute_config`` flag: when it is set (DeepSeek-V3), a recomputed
        layer keeps its input sliced over ``ccfg.t``, and when it is not
        (Qwen), whole.  :func:`derive` computes the sharding factors from
        them, and from the activation sharding of the model's family, which
        for Qwen shards them over ``ccfg.t`` in any case.
        """
        train_raw = self._get_cfg_attr(self.config, "train", Config({}))
        gc = self._get_cfg_attr(train_raw, "gradient_checkpointing", Config({}))
        fsdp = self._get_cfg_attr(self.config, "fsdp_config", Config({}))
        ac = self._get_cfg_attr(self.config, "activation_checkpoint", Config({}))
        return {
            "emb_dp_sharded": True,
            "recompute_slice_activation": bool(self._get_cfg_attr(
                ac,
                "recompute_slice_activation",
                self._get_cfg_attr(
                    fsdp,
                    "recompute_slice_activation",
                    self._get_cfg_attr(gc, "recompute_slice_activation", False),
                ),
            )),
            "shard_mtp_param": True,
        }
