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
"""Bounded validation that preserves the surrounding training lifecycle."""

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterator

# Trainer runtime uses native Torch after removal of the platform abstraction.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.rng import capture_rng_state, restore_rng_state


def validate_evaluation_config(trainer: Any) -> None:
    """Validate the explicit validation window and supported parallel domain.

    Args:
        trainer: Shared Trainer owning model, data and topology.

    Raises:
        ValueError: No independent validation loader or positive window exists.
        NotImplementedError: TP/PP or concurrent host prefetch would break isolation.
    """
    if trainer.config.training.eval_iters <= 0:
        raise ValueError("Periodic evaluation requires training.eval_iters > 0")
    loader = trainer.valid_dataloader
    if loader is None or loader is trainer.train_dataloader:
        raise ValueError("Evaluation requires an independent validation DataLoader")
    if trainer.mesh.tp_size != 1 or trainer.mesh.pp_size != 1:
        raise NotImplementedError("Trainer evaluation currently requires TP=PP=1")
    iterator = getattr(trainer, "data_iterator", None)
    data_config = getattr(trainer.config, "dataloader", None)
    background_prefetch = getattr(data_config, "use_background_prefetcher", False)
    if background_prefetch is True or getattr(iterator, "use_background_prefetcher", False) is True:
        raise NotImplementedError("Evaluation RNG isolation requires paused host background prefetch")


@contextmanager
def _evaluation_state(trainer: Any) -> Iterator[None]:
    """Restore modes, validation cursor and RNG even when validation raises."""
    rng_state = capture_rng_state()
    modules = tuple(dict.fromkeys((*trainer.model.modules(), *trainer.loss_fn.modules())))
    modes = tuple(module.training for module in modules)
    loader = trainer.valid_dataloader
    loader_state = deepcopy(loader.state_dict()) if callable(getattr(loader, "state_dict", None)) else None
    try:
        trainer.model.eval()
        trainer.loss_fn.eval()
        with torch.no_grad():
            yield
    finally:
        try:
            # A validation forward has no backward hook to release keep-forward parameters.
            for module in trainer.model.modules():
                if isinstance(module, HSDPModule):
                    module.reshard()
            if loader_state is not None:
                loader.load_state_dict(loader_state)
        finally:
            for module, training in zip(modules, modes):
                module.training = training
            restore_rng_state(rng_state)


def _validation_batch(trainer: Any, iterator: Any) -> tuple[dict, dict]:
    """Compute local mean objectives and their actual supervised-token counts."""
    model_inputs, loss_inputs = trainer.get_batch(iterator)
    model_inputs = trainer.preforward(model_inputs)
    loss_inputs = trainer.preforward(loss_inputs)
    count_inputs = dict(loss_inputs)
    if bool(model_inputs.get("labels_are_shifted", False)):
        count_inputs["labels_are_shifted"] = True
    counts = count_loss_token(count_inputs)
    prepare_inputs = getattr(trainer.loss_fn, "prepare_model_inputs", None)
    if callable(prepare_inputs):
        model_inputs = prepare_inputs(model_inputs, loss_inputs)
    context = trainer.model_fwd_context() if callable(trainer.model_fwd_context) else trainer.model_fwd_context
    with context:
        outputs = trainer.model(**model_inputs, use_cache=False)
    losses = trainer.loss_fn(model_output=outputs, labels=loss_inputs.get("labels"))
    if isinstance(losses, torch.Tensor):
        losses = {"foundation_loss": losses}
    # Chunked objectives report the final count after model-owned modality masking.
    valid_count = getattr(outputs, "valid_token_count", None)
    if valid_count is not None:
        counts["foundation_tokens"] = valid_count
    return losses, counts


def _reduce_validation(trainer: Any, totals: dict, counts: dict) -> dict[str, float]:
    """Reduce sums and counts independently across unique DP/CP token owners."""
    domain = trainer.mesh.dp_cp_mesh
    group = domain.get_group() if domain is not None else None
    metrics = {}
    for name in sorted(totals):
        numerator, denominator = totals[name], counts[name]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(numerator, group=group)
            dist.all_reduce(denominator, group=group)
        if not bool(torch.isfinite(numerator).all()):
            raise ValueError(f"Nonfinite validation objective: {name}")
        metrics[name] = float((numerator / denominator.clamp_min(1)).item())
        metrics[name.removesuffix("_loss") + "_tokens"] = int(denominator.item())
    metrics["loss"] = sum(metrics[name] for name in totals)
    return metrics


def evaluate(trainer: Any) -> dict[str, float]:
    """Run a fixed validation window without optimizer, QB or scheduler updates.

    ``eval_iters`` counts global validation batches, each containing the same
    microbatch count as one training step. Repeated calls revisit the saved
    validation cursor. Training RNG, modes and the training loader are preserved.
    Losses are weighted over the entire window by valid tokens, including CP.

    Args:
        trainer: Shared Trainer with an independently built validation loader.

    Returns:
        Global token-weighted objectives and their supervised-token counts.

    Raises:
        ValueError: Validation configuration or a loss is invalid.
        RuntimeError: The loader ends before the configured window is complete.
    """
    validate_evaluation_config(trainer)
    totals, counts = {}, {}
    batches = trainer.config.training.eval_iters * trainer.num_micro_batches
    with _evaluation_state(trainer):
        iterator = iter(trainer.valid_dataloader)
        for _ in range(batches):
            try:
                losses, token_counts = _validation_batch(trainer, iterator)
            except StopIteration as error:
                raise RuntimeError("Validation DataLoader ended before training.eval_iters was reached") from error
            for name, loss in losses.items():
                count = token_counts[name.removesuffix("_loss") + "_tokens"].to(trainer.device)
                weighted = loss.detach().float() * count
                totals[name] = totals.get(name, 0) + weighted
                counts[name] = counts.get(name, 0) + count
        return _reduce_validation(trainer, totals, counts)
