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
"""Observe real external-agent calls, episode grouping, publication and updates."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

from rl.dataset.episodes import episode_rows
from rl.trainer import SyncTrainer


def _verify_captured_tokens(row: Any) -> None:
    """Compare both segmented and continuous rows to immutable captured engine calls."""
    record = row.metadata.get("gateway_record")
    if "episode_id" in row.metadata and record is None:
        raise RuntimeError("Segmented agent acceptance requires its actual gateway_record")
    records = [record] if record is not None else row.metadata.get("gateway_records")
    if not records:
        raise RuntimeError("Agent acceptance requires captured model-call evidence")
    tokens, actions, probabilities = [], [], []
    for record in records:
        response = record["response"]
        choice = response["choices"][0]
        prompt = choice.get("input_token_ids", choice.get("prompt_token_ids", response.get("prompt_token_ids")))
        action = choice.get("token_ids", response.get("token_ids"))
        if not isinstance(prompt, list) or not isinstance(action, list) or not action:
            raise RuntimeError("Captured call omitted its actual prompt or action IDs")
        if prompt[:len(tokens)] != tokens:
            raise RuntimeError("Continuous agent evidence rewrote its sampled prefix")
        observed = len(prompt) - len(tokens)
        tokens = prompt + action
        actions.extend([False] * observed + [True] * len(action))
        probabilities.extend([0.0] * observed + [item["logprob"] for item in choice["logprobs"]["content"]])
    if row.token_ids.tolist() != tokens or row.action_mask.tolist() != actions:
        raise RuntimeError("Agent training row differs from captured prompt/actions")
    torch.testing.assert_close(row.rollout_log_probs.cpu(), torch.tensor(probabilities[1:]))


def _padding_counts(rollout: Any) -> tuple[int, int]:
    """Count physical padding directly and prove it carries no actions or reward."""
    padding = [index for index, row in enumerate(rollout.trajectories) if row.metadata.get("dp_padding", False)]
    for index in padding:
        row = rollout.trajectories[index]
        if row.reward != 0 or row.action_mask.any().item():
            raise RuntimeError("Padding trajectory contributes reward or action tokens")
        if rollout.rewards[index].item() != 0 or rollout.action_mask[index].any().item():
            raise RuntimeError("Padding batch row contributes reward or action tokens")
    if len(padding) != rollout.metadata.get("dp_padding_rows", 0):
        raise RuntimeError("Padding metadata differs from physical padding rows")
    return len(rollout.trajectories) - len(padding), len(padding)


def _validate_rank_alignment(ranks: list[dict]) -> bool:
    """Require equal physical batches and consistent TP replicas at each step."""
    if len({row["physical_rows"] for row in ranks}) != 1:
        raise RuntimeError("DP call batches were not aligned to the same physical row count")
    owners = {}
    for row in ranks:
        if (row["real_rows"] != sum(row["calls_per_episode"])
                or row["real_rows"] + row["padding_rows"] != row["physical_rows"]):
            raise RuntimeError("Agent physical, real and padding row counts disagree")
        signature = (row["calls_per_episode"], row["real_rows"], row["padding_rows"], row["rewards"])
        if row["dp_rank"] in owners and owners[row["dp_rank"]] != signature:
            raise RuntimeError("TP replicas disagree on episode or padding evidence")
        owners[row["dp_rank"]] = signature
    real_counts = [signature[1] for signature in owners.values()]
    if ranks[0]["physical_rows"] != max(real_counts):
        raise RuntimeError("DP padding exceeds the maximum real call count")
    return len(set(real_counts)) > 1 and any(row["padding_rows"] > 0 for row in ranks)


class AgentObservedTrainer(SyncTrainer):
    """Run the production trainer while retaining bounded acceptance evidence."""

    def __init__(self, config: dict, output: Path) -> None:
        """Bind evidence output before initializing the unchanged training runtime."""
        self.output = output
        self.records: list[dict] = []
        self.before: dict[str, torch.Tensor] = {}
        super().__init__(config)

    def _samples(self) -> dict[str, torch.Tensor]:
        result = {}
        for name, parameter in self.actor.actor_model.named_parameters():
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            flat = local.detach().reshape(-1)
            if parameter.requires_grad and flat.numel():
                result[name] = flat[::max(1, flat.numel() // 128)][:128].float().cpu().clone()
        if not result or any(not torch.isfinite(value).all() for value in result.values()):
            raise RuntimeError("Missing or invalid trainable parameter samples")
        return result

    def _train_step(self, batch: dict) -> None:
        self.before = self._run_rank_synchronized("agent parameter samples", self._samples)
        super()._train_step(batch)

    def _evidence(self, values: dict) -> dict:
        """Validate sampled calls and collect update evidence for this rank."""
        rollout, step, update = values["rollout"], values["step"], values["actor_update"]
        episodes = episode_rows(rollout.trajectories)
        for rows in episodes:
            for index in rows:
                row = rollout.trajectories[index]
                if row.policy_version != step - 1 or row.worker_policy_version != step - 1:
                    raise RuntimeError("Agent call sampled the wrong policy version")
                _verify_captured_tokens(row)
        real_rows, padding_rows = _padding_counts(rollout)
        after = self._samples()
        delta = max(float((after[name] - value).abs().max()) for name, value in self.before.items())
        if (self.rollout_engine.policy_version != step or not math.isfinite(update.gradient_norm)
                or not math.isfinite(update.total_loss) or update.valid_tokens <= 0):
            raise RuntimeError("Agent update/publication requires finite loss/gradient and valid action tokens")
        return {"rank": dist.get_rank(), "dp_rank": int(self.parallel_dims.dp_rank),
                "tp_rank": int(self.parallel_dims.tp_rank),
                "sampled_version": step - 1, "published_version": step,
                "calls_per_episode": [len(rows) for rows in episodes],
                "model_calls_per_episode": [max(len(rows), int(rollout.trajectories[rows[0]].metadata.get(
                    "deepseek_completion_count", 1))) for rows in episodes],
                "rewards": [rollout.trajectories[rows[0]].reward for rows in episodes],
                "padding_rows": padding_rows, "real_rows": real_rows, "padding_action_tokens": 0,
                "physical_rows": len(rollout.trajectories), "parameter_max_delta": delta,
                "gradient_norm": float(update.gradient_norm), "loss": float(update.total_loss),
                "valid_tokens": update.valid_tokens}

    def _complete_step(self, **values: Any) -> None:
        """Gather rank evidence and record each completed agent update."""
        local = self._run_rank_synchronized("agent call/update evidence", lambda: self._evidence(values))
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        _validate_rank_alignment(gathered)
        self.records.append({"step": values["step"], "ranks": gathered})
        if dist.get_rank() == 0:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "training.json").write_text(
                json.dumps(self.records, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
        super()._complete_step(**values)


def main(argv: list[str] | None = None) -> None:
    """Launch with torchrun and require an observed update after two policy versions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--require-uneven-calls", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = Path(args.output)
    trainer = AgentObservedTrainer(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), output)
    trainer.train()
    rows = [rank for step in trainer.records for rank in step["ranks"]]
    if len(trainer.records) < 2 or not any(row["parameter_max_delta"] > 0 for row in rows):
        raise RuntimeError("Agent acceptance requires two completed steps and a real parameter update")
    if not any(max(row["model_calls_per_episode"], default=0) > 1 for row in rows):
        raise RuntimeError("No real multi-call episode was observed")
    if args.require_uneven_calls and not any(_validate_rank_alignment(step["ranks"]) for step in trainer.records):
        raise RuntimeError("Acceptance requires actual unequal DP call counts and zero-loss padding")
    if int(os.environ.get("RANK", "0")) == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "completed.json").write_text(
            json.dumps({"status": "passed", "steps": len(trainer.records)}) + "\n", encoding="utf-8",
        )


def test_training() -> None:
    """Exercise actual unequal call counts and masked DP padding during training."""
    main([os.environ["RL_ST_CONFIG"], os.environ["RL_ST_RESULT_DIR"], "--require-uneven-calls"])


if __name__ == "__main__":
    main()
