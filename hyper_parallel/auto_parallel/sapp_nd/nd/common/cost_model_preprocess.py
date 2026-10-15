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
"""parse config for cost model"""
import inspect
import os
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any, DefaultDict, Iterator, List
import re
from copy import deepcopy
from enum import Enum
from pprint import pformat

from hyper_parallel.auto_parallel.sapp_nd.nd.common.generate_partitions import PartitionGenerator
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger
from hyper_parallel.auto_parallel.sapp_nd.nd.common.apply_exec import apply_exec, strategy_exec


class AttentionType(Enum):
    """Attention type enumeration."""
    MHA = "mha"
    GQA = "gqa"
    MLA = "mla"


def detect_attention_type(ccfg: "CostModelConfig") -> AttentionType:
    """Detect attention type from cost model config.

    Detection rules:
    1. If the keys and values come from a compressed latent (dc_kv > 0,
       the latent width every parser states): MLA
    2. If n_kv < a: GQA
    3. Otherwise: MHA

    Args:
        ccfg: Cost model config.

    Returns:
        AttentionType enum.

    Example:
        >>> ccfg.dc_kv = 512
        >>> ccfg.a = 64
        >>> ccfg.n_kv = 64
        >>> detect_attention_type(ccfg)
        <AttentionType.MLA: 'mla'>
    """
    if ccfg.dc_kv > 0:
        return AttentionType.MLA
    if ccfg.n_kv < ccfg.a:
        return AttentionType.GQA
    return AttentionType.MHA


def compute_kv_dim(ccfg: Any) -> float:
    """Return the width, per TP rank, of each of the K and V that CP exchanges.

    Context parallelism exchanges the attention's own keys and values,
    after the projections, whose heads TP splits.  An MLA layer builds every
    head's key and value from its latent before attention, the key at the
    head width plus the rotary part every head shares, the value at the
    head width, and those are what CP exchanges, not the latent: it acts on
    the attention's inputs.  Their mean width is returned, since K and V
    are counted as two tensors of one width.

    Args:
        ccfg: Cost model config with attributes a, n_kv, dh, dhr, h, t, dc_kv.

    Returns:
        Effective KV dimension per TP rank (float).
    """
    attention_type = detect_attention_type(ccfg)
    t = max(1, ccfg.t)
    if attention_type == AttentionType.MLA:
        return ccfg.n_kv * (2 * ccfg.dh + ccfg.dhr) / 2 / t
    if attention_type == AttentionType.GQA:
        n_kv = min(ccfg.n_kv if ccfg.n_kv > 0 else ccfg.a, ccfg.a)
        return n_kv * ccfg.dh / t
    return ccfg.h / t


# Each read of a config field no parser set, by field and by the function
# that read it. The config hands such a field over as 0, so a model feature
# nobody parsed costs nothing unless its reader has a fallback of its own, and
# the estimate stays plausible; a search says which were read, under its
# ranking (I9). A reader with a fallback, such as _flavour_tables' three
# feed-forward matmuls, is named too: the record cannot tell it apart.
UNSET_READS: DefaultDict[str, Counter] = defaultdict(Counter)
# The fields whose absence means there is nothing to price: a submodule
# copied without its parent's hooks takes the predefined ones.
ABSENT_MEANS_NOTHING = frozenset({"hooks_dict"})
# Whether such a read is refused rather than priced as 0 (run_nd --strict).
_STRICT = {"on": False}


def set_strict(strict: bool) -> None:
    """Refuse, rather than price as 0, a config field no parser set where the estimate reads it.

    Under it a read such as ``ccfg.n_gather`` raises AttributeError naming
    the field and its reader, while ``getattr(ccfg, name, default)``, which
    states what absence means, gets its default.

    Args:
        strict: True to refuse; False to price an unset field as 0, as by default.
    """
    _STRICT["on"] = bool(strict)


def unset_reads_report() -> List[str]:
    """What the config fields no parser set were read by, one line, and forget them; none where none was read."""
    if not UNSET_READS:
        return []
    count = len(UNSET_READS)
    fields = "; ".join(
        f"{attr} by {', '.join(reader for reader, _ in readers.most_common())}"
        for attr, readers in sorted(UNSET_READS.items())
    )
    UNSET_READS.clear()
    return [f"The estimate read {count} config field(s) no parser set, each handed over as 0, which a reader "
            f"without a fallback of its own prices as 0 (run_nd --strict refuses them): {fields}"]


# class CostModelConfig(Config) :
# The strategy fields a config armed by a search takes only through
# set_strategy or apply_exec.
STRATEGY_GUARDED = frozenset(("d", "t", "ep", "p", "vp", "cp", "os_max_shard"))
# Where a config keeps whether its guard is armed; a copy never takes it.
_GUARD = "_strategy_guard"
# What a deep copy shares with its original: the parser and the raw yaml, set
# when the config is parsed and written by no estimate.  A search deep-copies
# its config twice a candidate, and copying these was most of a copy's time.
_SHARED_ON_COPY = frozenset(("parser", "config"))


class CostModelConfig(PartitionGenerator):
    """cost model variables class"""

    def __init__(
        self,
        input_config: Any = None,
        hook_cls: Any = None,
        framework: Any = None,
        source_code: Any = None,
        devices: int = 0,
    ) -> None:
        """Initialise the cost model from a config, hooks and framework name, and the devices a caller states."""
        super().__init__(input_config, hook_cls, framework, source_code, devices)
        logger.debug(
            "parser = %s for %s", str(self.parser), str(self.model_name)
        )

    def __str__(self) -> str:
        """Return the cost model variables as readable text."""
        return "CostModelConfig attributes:\n" + pformat(
            {
                k: v
                for k, v in vars(self).items()
                if isinstance(v, (int, float, str, bool))
            }
        )

    def __getattr__(self, attr):
        call_source = inspect.currentframe().f_back.f_code.co_name
        if attr not in self.__dict__:
            if not attr.startswith("_") and attr not in ABSENT_MEANS_NOTHING:
                frame = inspect.currentframe().f_back
                reader = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_code.co_name}"
                UNSET_READS[attr][reader] += 1
                if _STRICT["on"]:
                    raise AttributeError(
                        f"{reader} reads {attr}, which no parser set: run_nd --strict refuses it rather than "
                        f"pricing it as 0"
                    )
            logger.warning(
                "[%s] Attribute %s does not exist. "
                "Value '0' will be assigned.",
                call_source,
                attr,
            )
            return 0
        return self.__dict__[attr]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in STRATEGY_GUARDED and self.__dict__.get(_GUARD):
            raise AttributeError(f"Cannot directly modify {name}, use set_strategy()")
        object.__setattr__(self, name, value)

    def __copy__(self):
        res = object.__new__(type(self))
        res.__dict__.update(self.__dict__)
        res.__dict__.pop(_GUARD, None)
        return res

    def __deepcopy__(self, memo):
        res = object.__new__(type(self))
        for k, v in self.__dict__.items():
            if k != _GUARD:
                setattr(res, k, v if k in _SHARED_ON_COPY else deepcopy(v, memo))
        return res

    def __getstate__(self) -> dict:
        """Return instance state for multiprocessing serialization."""
        state = self.__dict__.copy()
        state.pop(_GUARD, None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore instance state after multiprocessing deserialization."""
        self.__dict__.update(state)

    def fp_bytes(self, precision: Any) -> int:
        """Return bytes size for datatype"""
        if precision and isinstance(precision, str):
            res = re.match(r"[^0-9]*([0-9]+)[^0-9]*", precision)
            if res:
                return int(res.group(1)) // 8
        logger.warning("No bytes detected from FP Precision: %s", precision)
        return 0

    def print_stages_i(self, stage_id: Any, stage: Any) -> None:
        """for print_stages"""
        stage_layers = []
        for chunk in stage:
            chunk_lay_occ = []
            if chunk:
                layer, count = chunk[0], 1
                for lay_id in range(1, len(chunk)):
                    if chunk[lay_id] == layer:
                        count += 1
                    else:
                        chunk_lay_occ += [f"{count}{layer.name[0]}"]
                        layer, count = chunk[lay_id], 1
                chunk_lay_occ += [f"{count}{layer.name[0]}"]
            stage_layers += [chunk_lay_occ]
        logger.info("stage _%s : %s", stage_id, stage_layers)

    def print_stages(self, stages: Any, spec_stage_id: Any = -1) -> None:
        """Call after generate_partitions"""
        if spec_stage_id == -1:
            for stage_id, stage in enumerate(stages):
                self.print_stages_i(stage_id, stage)
        elif 0 <= spec_stage_id < len(stages):
            self.print_stages_i(spec_stage_id, stages[spec_stage_id])
        else:
            logger.warning("Incorrect spec_stage_id")

    def count_layers(self, stages: Any) -> int:
        """Count non-embedding and non-output layers in generated stages."""
        return sum(sum(len(layer) for layer in chunk) for chunk in stages) - 2

    def print_parallelism(self) -> None:
        """strategy pretty printer"""
        if not self.multimodal:
            logger.info("%s Parallelism used :", self.model_name)
            logger.info(
                "DP %s, TP %s, PP %s, EP %s, CP %s, VPP %s",
                self.d,
                self.t,
                self.p,
                self.ep,
                self.cp,
                self.vp,
            )
            logger.info(
                "d_exp %s, t_exp %s, os_max_shard %s, etp %s",
                self.d_exp,
                self.t_exp,
                self.os_max_shard,
                self.etp,
            )
            logger.info(
                "shard_grad_exp %s, shard_grad_non_exp %s",
                self.shard_grad_exp,
                self.shard_grad_non_exp,
            )
            logger.info(
                "shard_p_os_exp %s, shard_p_os_non_exp %s",
                self.shard_p_os_exp,
                self.shard_p_os_non_exp,
            )
            logger.info(
                "shard_embed %s, shard_output_activ %s, shard_rec_input %s",
                self.shard_embed,
                self.shard_output_activ,
                self.shard_recompute_input,
            )
        else:
            for m in self.mm_ccfgs:
                self.mm_ccfgs[m].print_parallelism()

    def strategy_num_devices(self) -> float:
        """total num devices"""
        return self.d * self.t * self.cp * self.p

    def is_consistent_pp_config(self) -> bool:
        """check if pp/offset/recomputation consistency"""

        def is_valid_cfg(cfg: Any) -> bool:
            """Return whether one per-stage config matches the pipeline shape."""
            if cfg is None or isinstance(cfg, (int, bool)):
                return True
            if not isinstance(cfg, list) or not cfg:
                return False
            if isinstance(cfg[0], int):
                return len(cfg) == self.p
            if isinstance(cfg[0], list):
                return len(cfg) == self.vp and all(
                    isinstance(c, list) and len(c) == self.p for c in cfg
                )
            return False

        return (
            is_valid_cfg(self.offset)
            and is_valid_cfg(self.full_rec)
            and is_valid_cfg(self.sel_rec)
        )

    def __strategy_target(self, model_name):
        """Get the config object targeted by a strategy update."""
        if not self.multimodal:
            return self
        if model_name in self.mm_ccfgs:
            return self.mm_ccfgs[model_name]
        raise TypeError(
            f"{self.model_name}:  model_name is required (multimodal)"
        )

    def set_shared_strategy(self, **kwargs: Any) -> None:
        """Apply a strategy to this config itself, a multimodal parent included.

        :meth:`set_strategy` targets a submodule of a multimodal config,
        never the parent; but the parent holds the strategy its submodules
        share, and the partitions it combines are of that pipeline, so a
        caller that gives every config of one pipeline its own placement
        writes each of them through this.
        """
        apply_exec(self, strategy_exec(self, kwargs))

    def set_strategy(self, **kwargs: Any) -> None:
        """Apply a keyword strategy: the ExecSpec it states, and what follows.

        The keywords are ``strategy_exec``'s; ``model_name`` picks the
        submodule of a multimodal config, and without it every submodule
        takes the strategy.
        """
        model_name = kwargs.get("model_name", None)
        if self.multimodal and model_name is None:
            # Submodules share one pipeline, so a strategy update with no
            # explicit target has to reach every one of them.
            for sub_name in self.mm_order:
                self.mm_ccfgs[sub_name].set_strategy(
                    **{**kwargs, "model_name": None}
                )
            model_name = self.mm_main if self.mm_main else self.mm_order[-1]
        target_ccfg = self.__strategy_target(model_name)
        apply_exec(target_ccfg, strategy_exec(target_ccfg, kwargs))
        logger.debug(
            "in ccfg: DP = %d, TP = %d, EP = %d, CP = %d, "
            "PP = %d, MB = %d, MBS = %d, VPP = %d",
            target_ccfg.d,
            target_ccfg.t,
            target_ccfg.ep,
            target_ccfg.cp,
            target_ccfg.p,
            target_ccfg.m,
            target_ccfg.b,
            target_ccfg.vp,
        )
        if not target_ccfg.is_consistent_pp_config():
            raise AttributeError(
                f"{target_ccfg.model_name}: "
                "Inconsistent pipeline parallel variables "
                f"pp {target_ccfg.p} vpp {target_ccfg.vp} "
                f"offset {target_ccfg.offset} "
                f"full_rec {target_ccfg.full_rec} "
                f"sel_rec {target_ccfg.sel_rec}"
            )

    def get_strategy(self) -> Any:
        """return parallelism/recompute strategies"""

        def strategy(mm: Any) -> Any:
            """Return the strategy dict for one submodule."""
            return {
                "dp": mm.d,
                "tp": mm.t,
                "pp": mm.p,
                "ep": mm.ep,
                "cp": mm.cp,
                "vpp": mm.vp,
                "op": mm.os_max_shard,
                "gbs": mm.b * mm.m * mm.d,
                "sched": mm.pp_sched,
                "offset": mm.offset,
                "full_rec": mm.full_rec,
                "sel_rec": mm.sel_rec,
            }

        # logger.output("get_strat ccfg")
        if self.multimodal:
            return {mm.model_name: strategy(mm) for mm in self.mm_ccfgs.values()}
        return strategy(self)


def arm_strategy_guard(ccfg: Any, armed: bool = True) -> None:
    """Refuse, or allow again, a direct write of a strategy field.

    Armed, the config and its submodules take a new degree only through
    ``set_strategy`` or ``apply_exec``, and a layer kind's own degrees
    through ``apply_layer_strategy``.  A copy of an armed config starts
    unarmed, so an estimator can change its own.

    Args:
        ccfg: The config, multimodal or not.
        armed: Whether to arm the guard or lift it.
    """
    object.__setattr__(ccfg, _GUARD, armed)
    for sub in (getattr(ccfg, "mm_ccfgs", None) or {}).values():
        arm_strategy_guard(sub, armed)


@contextmanager
def strategy_guarded(ccfg: Any) -> Iterator[Any]:
    """Refuse a direct write of a strategy field on *ccfg* inside the block.

    Afterwards the config is armed exactly as it was before.
    """
    before = ccfg.__dict__.get(_GUARD)
    object.__setattr__(ccfg, _GUARD, True)
    try:
        yield ccfg
    finally:
        if before is None:
            ccfg.__dict__.pop(_GUARD, None)
        else:
            object.__setattr__(ccfg, _GUARD, before)
