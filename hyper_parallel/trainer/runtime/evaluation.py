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
"""Trainer-side evaluation step: batching, loss adapters, token counting.

Everything that knows about text batches, the model's forward contract, or the
configured loss lives here; :mod:`hyper_parallel.evaluation` only sees
additive statistics. The split keeps the leaf package free of Trainer imports
while leaving the expensive, model-specific work on the side that owns it.

The step is a callable over a round index rather than a loop: the evaluator
decides how many rounds run, and this class only answers "what did round N
produce".
"""

from __future__ import annotations

__all__ = ["LossEvalStep"]

from typing import TYPE_CHECKING, Any, Dict, Iterator, Mapping, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.evaluation.types import (
    KIND_LM_NLL,
    KIND_OBJECTIVE,
    EvalStepResult,
    LossStatistics,
)
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.logging import create_logger

if TYPE_CHECKING:
    from hyper_parallel.trainer.base import BaseTrainer

logger = create_logger(__name__)

#: Normalization declarations a loss module may publish.
_NORMALIZATION_ALIASES = {
    "mean": "token_mean",
    "token_mean": "token_mean",
    "local_token_mean": "token_mean",
    "sum": "sum",
    "local_sum": "sum",
    "global_mean": "global_mean",
    "reduced": "global_mean",
}


def _statistic_name(key: str) -> str:
    """Map a loss dict key to its statistic name (``foundation_loss`` -> ``foundation``)."""
    return key.split("_loss", maxsplit=1)[0]


def _declared_normalization(loss_fn: Any, key: str) -> Optional[str]:
    """Read the loss module's declared normalization for one named loss.

    A loss adapter must state whether its value is a local token mean, a local
    sum, or already reduced across ranks: only a local mean can be turned into
    a token-weighted numerator by multiplying with the local valid-token count.
    An undeclared loss is refused rather than guessed at.

    Args:
        loss_fn: Configured loss module.
        key: Loss-dict key the module produced.

    Returns:
        ``"token_mean"``, ``"sum"``, ``"global_mean"``, or ``None`` when the
        module declares nothing for this key.
    """
    declaration = getattr(loss_fn, "loss_normalization", None)
    if declaration is None:
        return None
    if callable(declaration):
        declaration = declaration(key)
    if isinstance(declaration, Mapping):
        declaration = declaration.get(key)
    if declaration is None:
        return None
    normalized = str(declaration).lower()
    try:
        return _NORMALIZATION_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"loss module declared unknown normalization {declaration!r} for {key!r}; "
            "expected one of token_mean, sum, global_mean"
        ) from exc


class LossEvalStep:
    """Produce one additive-statistics step for loss-based evaluation.

    Args:
        trainer: Trainer that owns the model, loss, batch runtime, and mesh.
        split_name: Which DataLoader to iterate (``valid`` or ``test``).
        max_batches: Round limit for this rank, or ``None`` to run the DataLoader
            to exhaustion.
        statistic_kinds: Optional explicit ``{statistic: kind}`` overrides.
    """

    def __init__(
            self,
            trainer: "BaseTrainer",
            *,
            split_name: str = "valid",
            max_batches: Optional[int] = None,
            statistic_kinds: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Bind the trainer and prepare a fresh iterator for the first run."""
        self.trainer = trainer
        self.split_name = split_name
        self.max_batches = max_batches
        self.statistic_kinds = dict(statistic_kinds or {})
        self._iterator: Optional[Iterator[Any]] = None
        self._rounds_executed = 0

    def reset(self) -> None:
        """Start the next pass from the first sample of the evaluation split.

        Dropping the iterator is not enough: the batch sampler keeps a
        ``consumed_samples`` cursor that a finished pass leaves at the end of the
        split, so a second pass over the same DataLoader would yield nothing. A
        standalone task must be repeatable, so the cursor and the dynamic loader's
        candidate buffer are rewound here.
        """
        self._iterator = None
        self._rounds_executed = 0
        sampler = getattr(self.trainer, f"{self.split_name}_batch_sampler", None)
        if sampler is not None and hasattr(sampler, "consumed_samples"):
            sampler.consumed_samples = 0
        dataloader = getattr(self.trainer, f"{self.split_name}_dataloader", None)
        candidate_buffer = getattr(dataloader, "candidate_buffer", None)
        if candidate_buffer is not None:
            candidate_buffer.buffer.clear()
            candidate_buffer.buffer_output_indices.clear()
            candidate_buffer.buffer_token_count = 0

    def _dataloader(self) -> Any:
        """Return the configured evaluation DataLoader."""
        dataloader = getattr(self.trainer, f"{self.split_name}_dataloader", None)
        if dataloader is None:
            raise ValueError(
                f"run_mode='eval' requires a {self.split_name} DataLoader; "
                f"dataset split {self.split_name!r} is empty, or the dataset "
                "returned a single unsplit Dataset (configure evaluation.split='dataset')"
            )
        return dataloader

    def _statistic_kind(self, key: str) -> str:
        """Resolve the declared kind of one loss entry.

        The plain causal-LM objective is an NLL by construction: ``logits`` plus
        ``labels`` produce cross-entropy and nothing else. Auxiliary objectives
        (MoE router aux, MTP, regularizers) are reported under their own keys,
        so their presence does not change what ``foundation`` measures and does
        not disqualify its perplexity. Only a loss module that folds an aux term
        into the foundation value itself is ambiguous, and such a module must
        declare ``statistic_kinds`` explicitly.
        """
        statistic = _statistic_name(key)
        declared = self.statistic_kinds.get(statistic)
        if declared is not None:
            return declared
        if statistic == "foundation":
            return KIND_LM_NLL
        return KIND_OBJECTIVE

    def __call__(self, round_index: int) -> Optional[EvalStepResult]:
        """Run one forward and return additive statistics.

        Args:
            round_index: Zero-based round number, used only for the round cap.

        Returns:
            The step result, or ``None`` when the data source is exhausted or
            the configured round cap is reached.
        """
        if self.max_batches is not None and round_index >= self.max_batches:
            return None
        if self._iterator is None:
            self._iterator = iter(self._dataloader())

        # ``get_batch`` owns iteration: it reads on the data-owning rank and
        # broadcasts within TP, so the step must not call ``next`` itself.
        try:
            model_inputs, loss_inputs = self.trainer.get_batch(self._iterator)
        except StopIteration:
            return None

        prepare_model_inputs = getattr(self.trainer.loss_fn, "prepare_model_inputs", None)
        if callable(prepare_model_inputs):
            model_inputs = prepare_model_inputs(model_inputs, loss_inputs)

        token_counts = count_loss_token(dict(loss_inputs))
        labels = loss_inputs.get("labels")

        with torch.no_grad():
            outputs = self.trainer.model(**model_inputs, use_cache=False)
            local_loss = self.trainer.loss_fn(model_output=outputs, labels=labels)
        # Free vocabulary-sized logits before any collective in aggregation.
        del outputs, model_inputs

        loss_items = dict(local_loss) if isinstance(local_loss, Mapping) else {
            "foundation_loss": local_loss
        }

        statistics: Dict[str, LossStatistics] = {}
        for key, value in loss_items.items():
            if value is None:
                continue
            statistic = _statistic_name(key)
            # ``count_loss_token`` always publishes ``foundation_tokens``; a
            # modality-specific statistic that has no count of its own cannot
            # be weighted, so it is skipped rather than silently given the text
            # count, which would report a loss over the wrong denominator.
            valid_count = token_counts.get(f"{statistic}_tokens")
            if valid_count is None:
                if statistic != "foundation":
                    logger.warning(
                        "Skipping loss %r: no valid-token count for statistic %r",
                        key,
                        statistic,
                    )
                    continue
                valid_count = token_counts["foundation_tokens"]
            pair = self._to_additive(
                value,
                valid_count,
                _declared_normalization(self.trainer.loss_fn, key),
                key,
            )
            if pair is None:
                continue
            numerator, denominator = pair
            statistics[statistic] = LossStatistics(
                numerator=numerator,
                denominator=denominator,
                kind=self._statistic_kind(key),
            )

        return EvalStepResult(
            statistics=statistics,
            sample_count=self._real_sample_count(loss_inputs),
        )

    @staticmethod
    def _to_additive(
            value: torch.Tensor,
            valid_count: torch.Tensor,
            normalization: Optional[str],
            key: str,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Convert one local loss into a numerator/denominator pair.

        Args:
            value: Local loss value reported by the loss module.
            valid_count: Local valid-token count in the loss's own domain.
            normalization: Declared normalization, or ``None`` if undeclared.
            key: Loss key, for the error message.

        Returns:
            The additive numerator and matching denominator, or ``None`` when
            the value was already reduced and cannot be counted per rank.

        Raises:
            ValueError: The loss module declares no normalization.
        """
        if normalization is None:
            raise ValueError(
                f"loss {key!r} has no declared normalization, so evaluation cannot "
                "weight it. Declare `loss_normalization` on the loss module as "
                "token_mean, sum, or global_mean"
            )
        denominator = valid_count.detach().to(dtype=torch.float64)
        detached = value.detach().to(dtype=torch.float64)
        if normalization == "global_mean":
            # Already reduced across contributors: counting it again per rank
            # would multiply the result by the group size.
            return None
        if normalization == "sum":
            return detached, denominator
        return detached * denominator, denominator

    @staticmethod
    def _real_sample_count(loss_inputs: Mapping[str, Any]) -> torch.Tensor:
        """Count real samples in one prepared batch, excluding dummy rows.

        The count stays a device scalar: with background prefetching and many
        batches, one host synchronization per batch costs more than the whole
        reduction it feeds. The evaluator sums these scalars at the end.
        """
        labels = loss_inputs.get("labels")
        if labels is None:
            raise ValueError(
                "evaluation requires 'labels' in the prepared batch to count real "
                "samples; the configured loss recipe must publish them"
            )
        padding_flag = loss_inputs.get("padding_flag")
        if padding_flag is not None and bool(padding_flag):
            return labels.new_zeros((), dtype=torch.float64)
        valid = labels[..., 1:].ne(IGNORE_INDEX)
        loss_mask = loss_inputs.get("loss_mask")
        if loss_mask is not None:
            valid = valid & loss_mask[..., 1:].to(torch.bool)
        # A sample counts once if any of its targets is valid; a fully masked
        # sample contributes a zero numerator and zero denominator alike.
        return valid.any(dim=-1).sum().to(torch.float64)
