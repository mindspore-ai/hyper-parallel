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
    # Op profile the model is priced with, and its op counts per layer kind;
    # both settled by the parser, never matched from model_name.
    arch: str = None
    op_counts: dict = None
    # A vision tower's arch is the vision profile; this is its language
    # model's family, whose activation sharding it takes.
    inherited_arch: str = None
    device_capacity: Memory = Memory.zero()  # float = 0
    mm_ccfgs: any = None
    mm_order: list = None
    # The layer stack the parser settled (an auto_parallel LayerStack), and
    # the fields each of its kinds assigns, bound per candidate by the arch
    # hook from the config it is applied to.
    layer_stack: any = None
    layer_binding: dict = None
    # Fields the family gives every layer of the stack on top of its kind's,
    # such as cm's sharding (derive).
    layer_fields: dict = None
    # What a layer of each kind keeps and holds per token, as the model
    # spec's census states it (an auto_parallel KindActivations per kind
    # name), and the record of the kind of the layer priced, which the
    # parser or the arch hook binds; with none, the formulas price it.
    census: dict = None
    kind_activations: any = None
    # The output layer's record, as the census states it.
    output_census: any = None
    overwrite_eval_functions: dict = None
    parser: any = None
    # The device count a caller states, as run_nd -d does, which a parser
    # takes where its config cannot state one; 0 where none is stated.
    devices: int = 0
    # False where the config states no training sequence length and the
    # parser costs the model's context limit in its place.
    seq_len_stated: bool = True
    # The MTP layers the model states and the AutoModels trainer does not
    # build, so that nothing prices them; 0 where none was dropped.
    mtp_unpriced: int = 0
    # The activation checkpoint mode the run trains with, the trainer's
    # default included, where a HyperParallel parser read one; None elsewhere.
    stated_ac_mode: str = None

    # Strategy
    d: float = 0
    t: float = 0
    p: float = 0
    cp: float = 0
    ep: float = 1
    # Whether activations are split along the sequence over the TP group,
    # as the run's parser or a searched candidate states it; derive sets the
    # factor sp from it.  None where nothing stated it: priced as off, and a
    # searched candidate then runs it at every TP, as the search always did.
    sequence_parallel: bool = None
    # Whether TP shards the activations between layers and the output
    # layer's; None takes the family's (derive).
    shard_activations: bool = None
    # Whether the loss runs on logits sharded over the vocabulary, as stated,
    # None taking its family's; and as derive gives it.
    loss_parallel: bool = None
    shards_logits: bool = True
    sp: float = 0
    vp: float = 0
    os_max_shard: float = 0
    op_weight_shard: float = 0
    # How many ranks of its expert data-parallel group FSDP shards a routed
    # expert over under expert parallelism, as the run states it; None for
    # the optimizer's whole group.
    expert_shard: int = None
    # Whether the run shards each strategy's routed experts over that
    # strategy's whole expert data-parallel group, as a launcher that sets
    # edp_shard_size to the group does: the shard then follows the strategy.
    expert_shard_group: bool = False
    offset: Union[list, int] = None
    full_rec: Union[list, bool] = None
    sel_rec: Union[list, bool] = None
    # MindFormers' select_comm_recompute, which its selective recompute reads.
    sel_comm_rec: Union[list, bool] = None
    # The framework whose selective recompute rec_op follows, "hyperparallel"
    # or "mindformers".
    sel_rec_rule: str = "hyperparallel"
    # Whether a recomputed layer keeps its input sliced over tensor parallelism.
    recompute_slice_activation: bool = False
    # How each layer recomputes, as ranges over the layers in model order
    # (ExecSpec.recompute), when a spec states them; they replace full_rec
    # and sel_rec, and the partition generator places them.
    recompute_ranges: tuple = None
    pp_sched: str = None
    n_s_split: float = 0
    cp_algo: str = "colossalai_cp"
    rec_op: any = None
    # Each layer's own recompute switches in model order, None for a layer
    # that is not selective, where the recompute ranges state several
    # selective settings; None where rec_op holds the one every selective
    # layer runs (derive).
    layer_switches: tuple = None
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
    # An MLA model's value-head width, at which its family prices dh (derive).
    v_head_dim: float = None
    # An MLA model's non-rotary key-head width, its value heads' unless stated.
    qk_nope_head_dim: float = None
    dc_kv: float = 0
    dc_q: float = 0
    dhr: float = 0
    # The biases and norms a model states (its spec's); None where unstated,
    # which the parameter formulas count their own way.
    qkv_bias: bool = None
    o_bias: bool = None
    mlp_bias: bool = None
    norm_bias: bool = None
    layer_norms: int = None
    shared_expert_gate: bool = None
    k_1st_dense: float = 0
    # Attention flavour of a layer group, and the extra parameters a flavour
    # carries that the q/k/v/o formula does not describe (conv, gates).
    attn_kind: str = "full"
    attn_extra_p: float = 0
    lin_n_k: float = 0
    lin_d_k: float = 0
    lin_n_v: float = 0
    lin_d_v: float = 0
    lin_conv: float = 0
    # Recurrent-state update and readout, the linear-attention op. Zero for
    # every other flavour.
    n_linrec: float = 0
    # The QK-norm a layer runs: 1 where the model normalizes each head's
    # queries and keys (derive, from qk_norm), 0 on a linear-attention layer.
    n_qknorm: float = 0
    n_mtp: float = 0
    is_mtp_in_offset: bool = True
    multiple_of: float = 0
    fdm: float = 0

    # MoE
    t_exp: float = 0
    d_exp: float = 0
    # One per MoE layer: the op count of the dispatch the layer runs to put
    # each token on the rank holding the expert it routed to, and to take the
    # result back (``_flavour_tables``).
    n_dispatch: float = 0
    # What that dispatch costs on this cluster, as ``context.moe_dispatch``
    # states it: None, unstated, keeps the measured default
    # (``estimate.MOE_DISPATCH``), and 0 prices no dispatch at all.
    moe_dispatch: float = None
    # The ranks that hold the same experts, as the runtime's expert mesh spans
    # them; see ``_CostModelParser.expert_dp_group``.  ``d_exp`` is the older
    # per-CP, per-expert-TP-rank form of it and cannot carry CP.
    edp_group: float = 0
    hff_exp: float = 0
    n_exp: float = 0
    n_chosen_exp: float = 0
    n_shared_exp: float = 0
    cap_fact: float = 0
    etp: float = 0
    # Global per-expert token count per microbatch (all EP ranks combined,
    # before all-to-all); None = balanced.
    tokens_per_expert: list = None

    # CP modeling
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
    # Whether each gradient is sharded as its parameter is, as FSDP holds it.
    grad_shard_as_params: bool = False
    # Whether the stated FSDP shard spans CP's ranks, as HyperParallel's
    # dp_shard_size does, so that CP shards no parameter state beyond it
    # (derive.param_cp); False keeps the legacy schemas' rule.
    shard_spans_cp: bool = False
    freeze: bool = False
    has_fa: bool = False
    attn_output_gate: bool = False
    # Whether attention normalizes each head's queries and keys (Qwen3).
    qk_norm: bool = False
    # vp_less_mem: bool = False
    has_clip: bool = False
    gmm: bool = False
    vocab_emb_dp: float = 0
    # Whether the embedding table is sharded over data parallelism too.
    emb_dp_sharded: bool = True
    tie_emb_out: bool = False
    emb_out_in_offset: bool = False

    # batch
    b: float = 0
    m: float = 0
    gbs: float = 0

    # shard
    shard_embed: float = 0
    # Over how many of the ranks that shard the embedding table its layer
    # gathers it to compute with it.
    gather_embed: float = 1
    shard_output_activ: float = 0
    shard_recompute_input: float = 0
    is_shard_mtp_param: bool = True

    # bytes
    bytes_p: float = 0
    bytes_compute: float = 0
    bytes_softmax: float = 0
    # The widths the run states, None taking its family's, and whether its
    # gradients take memory without pipeline parallelism; derive gives the
    # estimators bytes_grad, bytes_os, bytes_norm and bytes_dropout from them.
    grad_bytes: float = None
    optimizer_state_bytes: float = None
    optimizer_states: float = None
    main_param_bytes: float = None
    norm_bytes: float = None
    dropout_bytes: float = None
    grad_accumulation: bool = None
    # Whether the run accumulates gradients over micro-batches without
    # pipeline parallelism, as derive gives it from grad_accumulation: a
    # search then gives PP 1 several micro-batches.
    accumulates_grads: bool = False
    # Whether FSDP frees a layer's gathered parameters once it has run, as
    # the run states it, None taking its family's; and as derive gives it.
    reshard_params: bool = None
    reshards: bool = False
    # Whether FSDP holds each layer's reduce-scatter output until the
    # backward ends, as stated, None taking its family's; and as derive
    # gives it.
    deferred_grad_accumulation: bool = None
    defers_grads: bool = False
    # Whether FSDP holds a layer's whole gradients while the next layer's
    # backward runs, and the root's until the backward ends, as stated,
    # None taking its family's; and as derive gives it.
    overlapped_grad_reduce: bool = None
    overlaps_grad_reduce: bool = False
    # Whether a layer keeps the weight casts the formulas price, as stated,
    # None where the optimizer does not shard; and as derive gives it.
    param_casts: bool = None
    keeps_param_casts: bool = None
    bytes_grad: float = 0
    bytes_os: float = 0
    # What the optimizer keeps per parameter: a layer's, and the embedding
    # and output tables', which keep AdamW's two states (derive).
    bytes_optim: float = 0
    bytes_optim_table: float = 0
    bytes_norm: float = 0
    bytes_dropout: float = 0

    def __init__(
        self,
        input_config: Any,
        hook_cls: Any,
        framework: Optional[str],
        source_code: Optional[str],
        devices: int = 0,
    ) -> None:
        """Initialise from a config path and optional hooks/framework/source, and the devices a caller states."""
        super().__init__()
        self.devices = devices or 0
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
        if isinstance(input_config, (str, dict)):
            self.config = Config(input_config)
        elif isinstance(input_config, Config):
            self.config = input_config
        else:
            raise TypeError(
                f"Expecting path string, dict or Config object for {input_config}"
            )
        # An in-memory config names its framework the same way a file does,
        # so a caller that already holds the config need not write it to disk
        # for the naive path sniffer to find the parser.
        if framework:
            logger.debug("Find parser module based on input framework name")
            parser_cls = self.get_framework_parser(framework.lower())
        elif isinstance(input_config, str):
            logger.debug("Naive way to find parser module")
            parser_cls = self.get_framework_parser_naive(input_config)
        else:
            # MindFormers format by default
            parser_cls = self.get_framework_parser_naive("yaml")
        if parser_cls:
            self.parser = parser_cls(self)
            logger.debug("Parser module: %s", self.parser.__class__)
            self.parser.parse()
