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
"""Explicit torchrun worker observing the production MoE training pipeline."""

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
from rl.roles.rollout.router_capture import RouterCaptureSpec, decode_router_trace
from rl.trainer import SyncTrainer

if __package__:
    from ._agent_train import _padding_counts, _validate_rank_alignment, _verify_captured_tokens
    from ._moe_r3_observer import ReplayObserver
else:
    from _agent_train import _padding_counts, _validate_rank_alignment, _verify_captured_tokens
    from _moe_r3_observer import ReplayObserver


class ObservedTrainer(SyncTrainer):
    """Keep training unchanged while recording local expert and policy evidence."""

    def __init__(self, config: dict, result_dir: Path, build_only: bool) -> None:
        """Bind observation output before constructing the production runtime."""
        self.result_dir = result_dir
        self.build_only = build_only
        self.records: list[dict[str, Any]] = []
        self.before: dict[str, torch.Tensor] = {}
        self.router_before: dict[str, torch.Tensor] = {}
        self.replay_observer = None
        super().__init__(config)
        if not build_only and self.actor.router_replay is not None:
            try:
                self.replay_observer = ReplayObserver(self.actor)
            except Exception:
                self._cleanup()
                raise

    def _expert_samples(
        self, suffixes: tuple[str, ...] = (".experts.gate_up_proj", ".experts.down_proj"),
    ) -> dict[str, torch.Tensor]:
        """Sample local expert or router storage without gathering weights."""
        samples = {}
        for name, parameter in self.actor.actor_model.named_parameters():
            if not name.endswith(suffixes):
                continue
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            flat = local.detach().reshape(-1)
            if flat.numel():
                stride = max(1, flat.numel() // 1024)
                samples[name] = flat[::stride][:1024].float().cpu().clone()
        if not samples:
            raise RuntimeError(f"No local parameter storage matched {suffixes}")
        if any(not torch.isfinite(value).all() for value in samples.values()):
            raise RuntimeError("Non-finite expert parameter sample")
        return samples

    def _write_result(self, filename: str, result: dict[str, Any]) -> None:
        if dist.get_rank() == 0:
            self.result_dir.mkdir(parents=True, exist_ok=True)
            (self.result_dir / filename).write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )

    def _build_runtime(self) -> None:
        if self.build_only:
            self._build_models_and_optimizers()
            samples = self._run_rank_synchronized("expert build samples", self._expert_samples)
            self._run_rank_synchronized("build evidence", lambda: self._write_result("build.json", {
                "status": "built", "world_size": dist.get_world_size(),
                "expert_parameter_names": sorted(samples), "config": self.resolved_config,
            }))
        else:
            super()._build_runtime()

    def _train_step(self, batch: dict[str, Any]) -> None:
        self.before = self._run_rank_synchronized("pre-update expert samples", self._expert_samples)
        if self.replay_observer is not None:
            self.router_before = self._run_rank_synchronized(
                "pre-update router samples", lambda: self._expert_samples((".mlp.gate.weight",)),
            )
        super()._train_step(batch)

    def _step_evidence(self, values: dict[str, Any]) -> dict[str, Any]:
        """Validate rollout metrics and capture expert parameter changes."""
        step = values["step"]
        update = values["actor_update"]
        rollout = values["rollout"]
        loss, norm = float(update.total_loss), float(update.gradient_norm)
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise RuntimeError(f"Non-finite training metrics: loss={loss}, gradient_norm={norm}")
        sampled = {trajectory.policy_version for trajectory in rollout.trajectories}
        if sampled != {step - 1} or rollout.worker_policy_version != step - 1:
            raise RuntimeError(f"Unexpected sampled versions: {sampled}, worker={rollout.worker_policy_version}")
        published = self.rollout_engine.policy_version
        if published != step:
            raise RuntimeError(f"Policy publication failed: expected={step}, actual={published}")
        after = self._expert_samples()
        if after.keys() != self.before.keys():
            raise RuntimeError("Expert parameter identity changed during training")
        deltas = {name: float((after[name] - before).abs().max()) for name, before in self.before.items()}
        valid = rollout.old_log_probs[rollout.loss_action_mask.bool()]
        if not valid.numel() or not torch.isfinite(valid).all():
            raise RuntimeError("Rollout action log probabilities are empty or non-finite")
        evidence = {
            "rank": dist.get_rank(), "step": step, "loss": loss, "gradient_norm": norm,
            "sampled_version": step - 1, "published_version": published,
            "action_tokens": valid.numel(), "reward_sum": float(rollout.rewards.sum()),
            "expert_sample_max_deltas": deltas, "optimizer_steps": update.optimizer_steps,
        }
        expected_replay = self.resolved_config["train"].get("router_replay", {}).get("mode") == "R3"
        if (self.replay_observer is not None) != expected_replay:
            raise RuntimeError("Configured R3 mode does not match the observed Actor")
        if self.replay_observer is not None:
            router_after = self._expert_samples((".mlp.gate.weight",))
            checkpoint = self.resolved_config["train"]["accelerator"]["activation_checkpoint"]
            evidence["replay"] = self.replay_observer.finish_step(checkpoint)
            evidence["replay"]["router_sample_max_deltas"] = {
                name: float((router_after[name] - before).abs().max())
                for name, before in self.router_before.items()
            }
        if self.resolved_config["agentic"].get("runner") == "codex":
            evidence["agent"] = self._codex_evidence(rollout)
        return evidence

    def _codex_evidence(self, rollout: Any) -> dict[str, Any]:
        """Verify independent call routes against persisted gateway and real CLI tool evidence."""
        episodes = episode_rows(rollout.trajectories)
        if len(episodes) < 2:
            raise RuntimeError("Codex GRPO acceptance requires at least two sampled episodes per rank")
        completed_tool_episodes = 0
        for indices in episodes:
            rows = [rollout.trajectories[index] for index in indices]
            metadata = rows[0].metadata
            directory = Path(metadata["artifact_dir"])
            if not directory.name.startswith(metadata["codex_session_id"] + "-"):
                raise RuntimeError("Codex artifact directory does not match its session identity")
            events = [json.loads(line) for line in (directory / "gateway-events.jsonl").read_text().splitlines()]
            records = [event["payload"] for event in events if event.get("type") == "completion.recorded"]
            if sorted(record["ordinal"] for record in records) != list(range(len(rows))):
                raise RuntimeError("Persisted gateway calls are incomplete or duplicated")
            records.sort(key=lambda record: record["ordinal"])
            for row, record in zip(rows, records):
                if row.worker_policy_version != row.policy_version:
                    raise RuntimeError("Codex call worker version differs from its trajectory policy")
                token_record = {key: value for key, value in record.items()
                                if key not in ("routed_experts", "router_capture_spec", "metadata")}
                # Recovery events can update diagnostic metadata after the original capture was persisted.
                row_record = {key: value for key, value in row.metadata["gateway_record"].items() if key != "metadata"}
                if (token_record != row_record or record["ordinal"] != row.metadata["call_index"]
                        or record["metadata"]["policy_version"] != row.policy_version
                        or record["metadata"]["session_id"] != metadata["codex_session_id"]):
                    raise RuntimeError("Codex training call differs from its persisted gateway ordinal/record")
                _verify_captured_tokens(row)
                spec = record["router_capture_spec"]
                captured = decode_router_trace(record["routed_experts"], row.token_ids.numel(), RouterCaptureSpec(
                    tuple(spec["layer_ids"]), spec["top_k"], spec["num_experts"],
                ))
                trace = row.router_trace
                if (trace is None or trace.layer_ids != captured.layer_ids
                        or not torch.equal(trace.expert_ids, captured.expert_ids)
                        or not torch.equal(trace.valid_mask, captured.valid_mask)):
                    raise RuntimeError("Codex call trace differs from its persisted captured routes")
            cli = [json.loads(line) for line in (directory / "codex-events.jsonl").read_text().splitlines()]
            completed = any(event.get("type") == "turn.completed" for event in cli)
            commands = [event.get("item", {}) for event in cli if event.get("type") == "item.completed"]
            tool_executed = any(item.get("type") == "command_execution" and item.get("exit_code") == 0
                                for item in commands)
            call_ids = {tool["id"] for record in records
                        for tool in record["response"]["choices"][0].get("message", {}).get("tool_calls", [])
                        if tool.get("function", {}).get("name", "").split(".")[-1] == "exec_command"}
            tool_outputs = any(item.get("type") == "function_call_output" and item.get("output")
                               and item.get("call_id") in call_ids
                               for row in rows[1:] for item in row.metadata["tool_history"])
            completed_tool_episodes += int(len(rows) >= 2 and completed and tool_executed and tool_outputs
                                           and bool(metadata["final_answer"]))
        real, padding = _padding_counts(rollout)
        return {"dp_rank": int(self.parallel_dims.dp_rank), "tp_rank": int(self.parallel_dims.tp_rank),
                "calls_per_episode": [len(indices) for indices in episodes],
                "rewards": [rollout.trajectories[indices[0]].reward for indices in episodes],
                "real_rows": real, "padding_rows": padding, "physical_rows": len(rollout.trajectories),
                "completed_tool_episodes": completed_tool_episodes}

    def _complete_step(self, **values: Any) -> None:
        """Gather and persist evidence for the completed MoE update."""
        evidence = self._run_rank_synchronized("step evidence", lambda: self._step_evidence(values))
        ranks: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, evidence)
        self.records.append({"step": values["step"], "ranks": ranks})

        def validate_and_write() -> None:
            """Persist evidence and require an actual expert update at completion."""
            final = values["step"] == self.state.max_steps
            if self.resolved_config["agentic"].get("runner") == "codex":
                _validate_rank_alignment([rank["agent"] for rank in ranks])
                if final and not any(rank["agent"]["completed_tool_episodes"] > 0
                                     for record in self.records for rank in record["ranks"]):
                    raise RuntimeError("Codex acceptance requires a complete multi-call episode with a real tool")
            changed = any(delta > 0 for record in self.records for rank in record["ranks"]
                          for delta in rank["expert_sample_max_deltas"].values())
            result = {"status": "observed", "expert_sample_changed": changed,
                      "steps": self.records, "config": self.resolved_config}
            self._write_result("training.json", result)
            if final and (len(self.records) < 2 or not changed):
                raise RuntimeError("Acceptance requires two completed steps and an observed expert update")
            if final and self.replay_observer is not None:
                replay_records = [rank["replay"] for record in self.records for rank in record["ranks"]]
                router_changed = any(delta > 0 for record in replay_records
                                     for delta in record["router_sample_max_deltas"].values())
                router_gradient = any(norm > 0 for record in replay_records for norm in record["router_gradient_norms"])
                if not router_changed or not router_gradient:
                    raise RuntimeError("R3 acceptance requires a nonzero router gradient and router update")

        self._run_rank_synchronized("write training evidence", validate_and_write)
        super()._complete_step(**values)


def main(argv: list[str] | None = None) -> None:
    """Run a supplied recipe; RL_ST_RESULT_DIR selects the evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result_dir = Path(os.environ["RL_ST_RESULT_DIR"])
    trainer = ObservedTrainer(config, result_dir, args.build_only)
    if args.build_only:
        trainer._cleanup()  # pylint: disable=protected-access
    else:
        trainer.train()
        if int(os.environ.get("RANK", "0")) == 0:
            (result_dir / "completed.json").write_text(
                json.dumps({"status": "passed", "steps": len(trainer.records)}) + "\n", encoding="utf-8",
            )


def test_training() -> None:
    """Run the configured production recipe with update and publication checks."""
    main([os.environ["RL_ST_CONFIG"]])


if __name__ == "__main__":
    main()
