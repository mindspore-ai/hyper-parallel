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
"""Expert bias update for auxiliary-loss-free MoE load balancing.

DeepSeek-V3's router scores experts by ``sigmoid(gate) + e_score_correction_bias``
and uses that bias to steer load away from whichever experts are over-used.
Transformers ships the buffer and consumes it, but ships no update rule for it,
and :class:`~hyper_parallel.core.utils.moe_utils.MoEMonitorCallback` targets this
repository's own ``MoE`` class rather than the HuggingFace modules a model family
built from a HF config contains. On that path nothing feeds the routing
statistic back, the bias stays at its initial zero, and the router drifts onto a
few experts over the first few dozen steps.

That drift is not academic: it grows the expert-parallel dispatch buffers, so it
costs both memory and step time -- measured on a 16-node MoE run, device_max
allocated climbed 41.6 -> 48.0 GiB and step time 17.8 -> 27.2 s as the load
concentrated, and both plateaued only once the drift saturated. This callback is
what keeps the routing balanced.

Enabled by environment variable so it can be switched on for an A/B run without
a configuration-schema change::

    HP_EXPERT_BIAS_LR=1e-3    # step size; unset or 0 leaves it disabled
"""
# pylint: disable=wrong-import-position

from __future__ import annotations

import logging
import os
from typing import Any

from hyper_parallel.distributed.expert_parallel.experts import (
    update_expert_bias_from_load,
)
from hyper_parallel.trainer.callbacks.base import Callback

logger = logging.getLogger(__name__)


def _resolve_lr() -> float | None:
    """Read the bias step size from the environment; None when disabled."""
    raw = os.environ.get("HP_EXPERT_BIAS_LR", "").strip()
    if not raw:
        return None
    lr = float(raw)
    return lr if lr > 0 else None


class ExpertBiasUpdateCallback(Callback):
    """Apply one load-balancing step to every routed MoE layer, once per step.

    The MoE layers are looked up lazily rather than captured at construction:
    callbacks are built before the model exists on some paths, and the set this
    needs is the set of layers that actually ran a routed forward (those
    carrying ``_ep_expert_load``).
    """

    def __init__(self, trainer: Any, lr: float | None = None) -> None:
        """Bind to the trainer and resolve the step size.

        Args:
            trainer: Owning trainer.
            lr: Step size; ``None`` reads ``HP_EXPERT_BIAS_LR``.
        """
        super().__init__(trainer)
        self.lr = _resolve_lr() if lr is None else lr
        self._reported = False

    def _routed_modules(self) -> list[Any]:
        """Every MoE layer that accumulated a load this step."""
        model = getattr(self.trainer, "model", None)
        if model is None:
            return []
        return [
            module
            for module in model.modules()
            if getattr(module, "_ep_expert_load", None) is not None
        ]

    def _ep_group(self) -> Any:
        """The process group whose ranks hold distinct tokens.

        The expert mesh carries the ``ep`` axis; under expert parallelism that
        domain is also the set of ranks that saw different data, which is
        exactly the set the load has to be summed over. Resolved per call rather
        than cached in ``__init__`` because callbacks are constructed before the
        mesh is guaranteed to be attached to the trainer.
        """
        moe_mesh = getattr(self.mesh, "fsdp_moe_mesh", None)
        if moe_mesh is None:
            raise RuntimeError(
                "expert bias load balancing needs the MoE mesh to resolve the "
                "EP group, but fsdp_moe_mesh is unset on this trainer"
            )
        return moe_mesh.get_group("ep")

    def on_step_end(
        self,
        state: Any,
        loss: float | None = None,
        loss_dict: dict[str, float] | None = None,
        grad_norm: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Nudge every routed layer's router bias by one load-balancing step.

        No-op while ``self.lr`` is unset, so the callback is safe to register
        unconditionally and costs one attribute read per layer when disabled.
        """
        del state, loss, loss_dict, grad_norm, kwargs
        if not self.lr:
            return
        modules = self._routed_modules()
        if not modules:
            return
        if not self._reported:
            self._reported = True
            logger.info(
                "Expert bias load balancing active on %d routed MoE layer(s), lr=%g",
                len(modules),
                self.lr,
            )
        # Activation checkpoint re-runs each layer's forward, so every token is
        # routed twice per optimizer step. Dividing here keeps the load a
        # per-step statistic rather than a per-forward one.
        recomputations = (
            2 if getattr(self.trainer.config.activation_checkpoint, "mode", None) == "full" else 1
        )
        group = self._ep_group()
        for module in modules:
            update_expert_bias_from_load(
                module,
                lr=self.lr,
                group=group,
                num_recomputations=recomputations,
            )


__all__ = ["ExpertBiasUpdateCallback"]
