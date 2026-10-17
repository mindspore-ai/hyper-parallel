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
"""Worker-only observations of real R3 gate and backward execution."""

from functools import partial, wraps
import re
from typing import Any

import torch


_GATE_PATH = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")


class ReplayObserver:
    """Assert real gate outputs against rollout traces independently of the adapter."""

    def __init__(self, actor: Any) -> None:
        """Attach observation after production replay hooks on this Actor only."""
        self.actor = actor
        self.controller = actor.router_replay
        self.handles = []
        self.expected = None
        self.gates = {}
        self.counts = {}
        self.comparisons = []
        self.microbatches = 0
        self.router_gradients = []
        for name, gate in actor.actor_model.named_modules():
            match = _GATE_PATH.search(name)
            if match is not None:
                layer_id = int(match.group(1))
                self.gates[layer_id] = gate
                self.handles.append(gate.register_forward_hook(partial(self._gate_output, layer_id)))
        if tuple(sorted(self.gates)) != self.controller.layer_ids:
            raise RuntimeError("ST could not observe every configured replay layer")
        self.handles.append(actor.actor_model.register_forward_pre_hook(self._model_input, with_kwargs=True))
        original_step = actor.optimizer.step

        @wraps(original_step)
        def observe_optimizer_step(*args: Any, **kwargs: Any) -> Any:
            """Observe reduced gradients before native or chained optimizers consume them."""
            self._before_optimizer()
            return original_step(*args, **kwargs)

        actor.optimizer.step = observe_optimizer_step
        original = actor._experience_log_probs  # pylint: disable=protected-access

        @wraps(original)
        def observe_microbatch(experience: Any, start: int, end: int) -> torch.Tensor:
            """Retain the source rows through checkpoint backward."""
            self.expected = (experience, start, end)
            if torch.is_grad_enabled():
                self.microbatches += 1
            return original(experience, start, end)

        actor._experience_log_probs = observe_microbatch  # pylint: disable=protected-access

    def _model_input(self, module: Any, args: tuple, kwargs: dict) -> None:
        """Check input row order and route padding against the original experience."""
        del module, args
        if self.expected is None or self.controller.current is None:
            raise RuntimeError("R3 Actor forward ran without its microbatch observation context")
        experience, start, end = self.expected
        sequences = kwargs["input_ids"]
        batch_size, length = sequences.shape
        if not torch.equal(sequences, experience.sequences[start:end, :length]):
            raise RuntimeError("R3 input token order differs from the original experience")
        targets = self.controller.current
        expected_ids = torch.zeros(
            (batch_size, length, len(self.controller.layer_ids), self.controller.top_k), dtype=torch.int64,
        )
        expected_mask = torch.zeros((batch_size, length), dtype=torch.bool)
        for row, trace in enumerate(experience.router_traces[start:end]):
            count = trace.expert_ids.shape[0]
            expected_ids[row, :count] = trace.expert_ids.to(torch.int64)
            expected_mask[row, :count] = trace.valid_mask
        actual_ids = targets.expert_ids.detach().cpu().to(torch.int64).reshape_as(expected_ids)
        actual_mask = targets.valid_mask.detach().cpu().reshape_as(expected_mask)
        if not torch.equal(actual_mask, expected_mask):
            raise RuntimeError("R3 active target validity differs from captured trace validity")
        if not torch.equal(actual_ids[expected_mask], expected_ids[expected_mask]):
            raise RuntimeError("R3 active targets changed route positions, rows, layers or expert IDs")

    def _gate_output(self, layer_id: int, gate: Any, inputs: tuple, output: tuple) -> None:
        """Accumulate exact routing comparisons without synchronizing every layer."""
        del inputs
        targets = self.controller.layer_targets(layer_id)
        if targets is None:
            raise RuntimeError("R3 gate ran outside replay scope, including checkpoint backward")
        expected, valid = targets
        probabilities, scores, actual = output
        native_scores, native = probabilities.topk(self.controller.top_k, dim=-1)
        if gate.norm_topk_prob:
            native_scores = native_scores / native_scores.sum(-1, keepdim=True)
        expected_scores = probabilities.gather(-1, expected.long())
        if gate.norm_topk_prob:
            expected_scores = expected_scores / expected_scores.sum(-1, keepdim=True)
        mismatch = ((actual != expected) & valid[:, None]).sum()
        native_difference = ((native != expected) & valid[:, None]).sum()
        invalid_mismatch = ((actual != native) & ~valid[:, None]).sum()
        reference_scores = torch.where(valid[:, None], expected_scores, native_scores).to(scores.dtype)
        weight_mismatch = (scores != reference_scores).sum()
        self.comparisons.append(torch.stack((
            valid.sum() * self.controller.top_k, mismatch, native_difference, invalid_mismatch, weight_mismatch,
        )).detach())
        counts = self.counts.setdefault(layer_id, {"train_grad": 0, "train_no_grad": 0, "diagnostic": 0})
        phase = ("train_grad" if torch.is_grad_enabled() else "train_no_grad") if gate.training else "diagnostic"
        counts[phase] += 1

    def _before_optimizer(self) -> None:
        """Observe reduced router gradients before the optimizer consumes them."""
        if self.controller.current is not None:
            raise RuntimeError("Replay targets leaked into the optimizer step")
        squares = []
        for gate in self.gates.values():
            gradient = gate.weight.grad
            if gradient is not None:
                local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
                squares.append(local.detach().float().square().sum().cpu())
        if not squares:
            raise RuntimeError("No router gradients reached the optimizer")
        norm = float(torch.stack(squares).sum().sqrt())
        if not torch.isfinite(torch.tensor(norm)):
            raise RuntimeError("Non-finite router gradient reached the optimizer")
        self.router_gradients.append(norm)

    def finish_step(self, checkpoint: str) -> dict:
        """Require exact replay and checkpoint execution before returning evidence."""
        if self.controller.current is not None:
            raise RuntimeError("Replay targets leaked beyond the completed update")
        totals = torch.stack(self.comparisons).sum(0).cpu().tolist() if self.comparisons else [0] * 5
        if totals[0] <= 0 or any(totals[index] for index in (1, 3, 4)):
            raise RuntimeError(f"R3 route/weight exactness failed: comparisons={totals}")
        if self.microbatches <= 0:
            raise RuntimeError("R3 acceptance observed no training microbatches")
        for layer_id in self.gates:
            counts = self.counts.get(layer_id, {"train_grad": 0, "train_no_grad": 0, "diagnostic": 0})
            if counts["train_grad"] < self.microbatches:
                raise RuntimeError(f"Missing router backward execution for layer={layer_id}: {counts}")
            if checkpoint == "off" and (counts["train_grad"] != self.microbatches or counts["train_no_grad"]):
                raise RuntimeError(f"Checkpoint off unexpectedly recomputed layer={layer_id}: {counts}")
            if checkpoint == "full" and counts["train_grad"] + counts["train_no_grad"] < 2 * self.microbatches:
                raise RuntimeError(f"Full checkpoint did not replay layer={layer_id}: {counts}")
        result = {
            "compared_expert_ids": totals[0], "replay_mismatches": totals[1],
            "native_choice_differences": totals[2], "invalid_position_mismatches": totals[3],
            "weight_mismatches": totals[4], "layer_calls": self.counts,
            "training_microbatches": self.microbatches, "router_gradient_norms": self.router_gradients,
            "targets_cleared": True,
        }
        self.comparisons = []
        self.counts = {}
        self.microbatches = 0
        self.router_gradients = []
        return result
