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
"""Recompute as a dimension of the search: the activation checkpoint modes a candidate may run.

A parallel degree is stated as one value, a list of candidates or ``auto``,
and the recompute dimension is stated the same way: ``full``, ``[off, full]``
or ``auto``. Its values are HyperParallel's activation checkpoint modes. The
search prices every candidate under each mode the dimension allows and ranks
the pairs together, each with its own memory check, so a mode left out of the
list is never proposed: selective, say, which the estimate prices as a policy
that runs well would run, while HyperParallel's own selective path runs
slower than full recompute.

The degrees are integers whose product is the device count, which the search
enumerates together; a mode is no factor of it, so the search runs once per
mode instead (:meth:`ParallelizeLayer.run_generation_to_ordering`).
"""
import copy
from typing import Any, List, Optional, Tuple

from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import hyper_rec_op

RECOMPUTE_MODES = ("off", "selective", "full")
# The parsers whose recompute is HyperParallel's activation checkpoint mode,
# and whose selective switches follow from it alone.
HYPER_FRAMEWORKS = ("hyper_v2", "hyperparallel2")
# Older search configs state no recompute as "none".
_ALIASES = {"none": "off"}


def read_recompute_modes(stated: Any, where: str) -> Optional[Tuple[str, ...]]:
    """Read a recompute dimension stated the way a parallel degree is.

    Args:
        stated: None where nothing states it; ``"auto"`` for every mode; one
            mode; or a list of modes. YAML reads an unquoted ``off`` as False,
            which counts as ``"off"``.
        where: Where it was stated, for the error message.

    Returns:
        The modes in the order stated, each once, or None where nothing
        states the dimension and the search keeps the config's own recompute.

    Raises:
        ValueError: An empty list, or a value that is neither a mode nor auto.
    """
    if stated is None:
        return None
    listed = list(stated) if isinstance(stated, (list, tuple)) else [stated]
    if [str(value).strip().lower() for value in listed] == ["auto"]:
        return RECOMPUTE_MODES
    modes: List[str] = []
    for value in listed:
        mode = "off" if value is False else str(value).strip().lower()
        mode = _ALIASES.get(mode, mode)
        if mode not in RECOMPUTE_MODES:
            raise ValueError(
                f"{where}: {value!r} is not a recompute mode; expected one of "
                f"{', '.join(RECOMPUTE_MODES)}, a list of them, or auto"
            )
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError(f"{where}: an empty list allows no recompute mode")
    return tuple(modes)


def read_activation_checkpoint_mode(stated: Any, where: str) -> str:
    """Read one activation checkpoint mode the way the trainer reads ``activation_checkpoint.mode``.

    Nothing stated is the trainer's default, ``"off"``, as is the older
    ``"none"``. YAML reads an unquoted ``off`` as False, which the trainer's
    config resolver turns back into ``"off"``, as here. Anything else that
    is not one of :data:`RECOMPUTE_MODES` is refused, True included: ND used
    to price a mode it did not know as no recompute, so a typo of ``full``
    was costed at the off memory, and a False fell through to the legacy
    default.

    Args:
        stated: The mode as the yaml states it.
        where: Where it was stated, for the error message.

    Returns:
        One of :data:`RECOMPUTE_MODES`.

    Raises:
        ValueError: A list, auto, True, or a name that is no mode.
    """
    if stated is None:
        return "off"
    if isinstance(stated, (list, tuple)) or str(stated).strip().lower() == "auto":
        raise ValueError(f"{where}: {stated!r} is not one activation checkpoint mode; expected one of "
                         f"{', '.join(RECOMPUTE_MODES)}")
    return read_recompute_modes(stated, where)[0]


def _mode_configs(ccfg: Any) -> List[Any]:
    """The configs a mode is stated on: the model's, and each submodule's of a multimodal one."""
    configs = [ccfg]
    if getattr(ccfg, "multimodal", False):
        configs += [ccfg.mm_ccfgs[name] for name in ccfg.mm_order]
    return configs


def state_recompute_mode(ccfg: Any, mode: str) -> None:
    """State one activation checkpoint mode on a model's cost model config, as the HyperParallel parsers read it.

    Each candidate's full recompute comes from the pipeline balancing, which
    the search states separately (:meth:`GlobalConfig.state_recompute`).

    Args:
        ccfg: The whole model's cost model config.
        mode: One of :data:`RECOMPUTE_MODES`.
    """
    for config in _mode_configs(ccfg):
        config.full_rec = mode == "full"
        config.sel_rec = mode == "selective"
        config.rec_op = Config(hyper_rec_op(config.sel_rec))


def parsed_recompute(ccfg: Any) -> List[Tuple[Any, Any, Any, Any]]:
    """Each config's recompute as its parser stated it, for :func:`restore_recompute`."""
    return [(config, copy.deepcopy(config.full_rec), copy.deepcopy(config.sel_rec),
             copy.deepcopy(getattr(config, "rec_op", None)))
            for config in _mode_configs(ccfg)]


def restore_recompute(parsed: List[Tuple[Any, Any, Any, Any]]) -> None:
    """Give each config back the recompute :func:`parsed_recompute` took from it."""
    for config, full_rec, sel_rec, rec_op in parsed:
        config.full_rec = copy.deepcopy(full_rec)
        config.sel_rec = copy.deepcopy(sel_rec)
        config.rec_op = copy.deepcopy(rec_op)
