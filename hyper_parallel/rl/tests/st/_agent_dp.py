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
"""Explicit two-rank CPU/Gloo acceptance for episode padding and global token loss."""

import argparse
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from rl.algorithm import build_algorithm
from rl.dataset.batch_builder import ExperiencePreparer, build_experience_batch, pad_agent_call_batch_for_dp
from rl.dataset.contracts import Trajectory, Turn
from rl.roles.rollout.base import GenerationSettings


def _calls(rank: int, version: int) -> tuple[Trajectory, ...]:
    """Rank zero owns 1+2 calls, rank one owns 2+3 calls, with distinct real contexts."""
    rows = []
    for episode, count in enumerate((rank + 1, rank + 2)):
        reward = float(episode == 0)
        for call in range(count):
            base = 10 + rank * 20 + episode * 5 + call
            rows.append(Trajectory(
                trajectory_id=f"rank{rank}-episode{episode}-call{call}", prompt_id=f"prompt{rank}",
                group_id=f"prompt{rank}", policy_version=version, worker_policy_version=version,
                turns=(Turn("user", "actual prompt", 0, 2, False), Turn("assistant", "actual action", 2, 4, True)),
                token_ids=torch.tensor([base, base + 1, base + 2, base + 3]),
                attention_mask=torch.ones(4, dtype=torch.bool), action_mask=torch.tensor([False, False, True, True]),
                rollout_log_probs=torch.tensor([0.0, -0.1, -0.2]), reward=reward,
                reward_components={"success": reward}, done=True, truncated=False, terminal_reason="completed",
                metadata={"episode_id": f"rank{rank}-episode{episode}", "call_index": call, "call_count": count},
            ))
    return tuple(rows)


def _batch(rows: tuple[Trajectory, ...]) -> object:
    """Build the same production rollout tensors that reach the Actor."""
    settings = GenerationSettings(max_new_tokens=2, temperature=1.0, top_p=1.0, top_k=0, do_sample=True,
                                  pad_token_id=0, eos_token_id=2, collect_log_probs=True)
    return build_experience_batch(rows, 0.0, settings, {})


def _reference_loss(model: torch.nn.Module, version: int) -> torch.Tensor:
    """Compute an independent unpadded global-token objective over all real rows."""
    rows = _calls(0, version) + _calls(1, version)
    inputs = torch.stack([row.token_ids[1:] for row in rows]).float() / 100.0
    mask = torch.stack([row.action_mask[1:] for row in rows])
    scale = 0.5 / (torch.tensor([1.0, 0.0]).std() + 1.0e-6)
    targets = torch.tensor([1.0 if row.reward else -1.0 for row in rows]).unsqueeze(-1) * scale
    prediction = model(inputs.unsqueeze(-1)).squeeze(-1)
    return ((prediction - targets).square() * mask).sum() / mask.sum()


def _train(model: DistributedDataParallel, baseline: torch.nn.Module, rank: int) -> list[dict]:
    """Run two synchronized updates and compare gradients with the unpadded objective."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.1)
    preparer = ExperiencePreparer(build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}))
    records = []
    for step in range(2):
        rollout = _batch(_calls(rank, step))
        rollout = pad_agent_call_batch_for_dp(rollout, SimpleNamespace(rank_size=2, group=dist.group.WORLD))
        experience = preparer.prepare(rollout, reference_log_probs=torch.zeros_like(rollout.old_log_probs))
        mask = experience.loss_action_mask
        global_tokens = mask.sum()
        dist.all_reduce(global_tokens)
        if global_tokens.item() != 16 or rollout.sequences.shape[0] != 5:
            raise RuntimeError("DP alignment changed the number of real actions or scheduled rows")
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        prediction = model(experience.sequences[:, 1:].float().unsqueeze(-1) / 100.0).squeeze(-1)
        prediction.retain_grad()
        loss = ((prediction - experience.advantages).square() * mask).sum() * 2 / global_tokens
        loss.backward()
        _reference_loss(baseline, step).backward()
        torch.testing.assert_close(model.module.weight.grad, baseline.weight.grad, atol=1.0e-7, rtol=1.0e-6)
        if prediction.grad[~mask].count_nonzero().item():
            raise RuntimeError("Masked prompt or padding tokens contributed to the gradient")
        gradient = float(model.module.weight.grad.item())
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.module.weight, baseline.weight, atol=1.0e-7, rtol=1.0e-6)
        records.append({"step": step + 1, "rank": rank, "real_rows": len(_calls(rank, step)),
                        "aligned_rows": rollout.sequences.shape[0], "global_action_tokens": global_tokens.item(),
                        "padding_rows": rollout.metadata.get("dp_padding_rows", 0), "gradient": gradient,
                        "weight": float(model.module.weight.item()), "matches_unpadded_baseline": True})
    return records


def main(argv: list[str] | None = None) -> None:
    """Run explicitly through torchrun; this worker never allocates an accelerator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(17)
        model = DistributedDataParallel(torch.nn.Linear(1, 1, bias=False, device="cpu"))
        baseline = deepcopy(model.module)
        rank = dist.get_rank()
        if dist.get_world_size() != 2:
            raise ValueError("This acceptance requires exactly two CPU ranks")
        records = _train(model, baseline, rank)
        gathered = [None, None]
        dist.all_gather_object(gathered, records)
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "result.json").write_text(json.dumps({
                "status": "passed", "backend": "gloo", "world_size": 2, "steps": 2, "ranks": gathered,
            }, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


def test_padding() -> None:
    """Run two real Gloo ranks against the unpadded gradient reference."""
    main(["--output-dir", os.environ["RL_ST_RESULT_DIR"]])


if __name__ == "__main__":
    main()
