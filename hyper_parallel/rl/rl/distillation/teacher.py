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
"""Frozen vLLM teachers with strict per-sample routing and token alignment."""

import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.request import Request, urlopen

import torch

from rl.data.contracts import ExperienceBatch
from rl.distillation.config import (
    DistillationConfig, TeacherConfig, distillation_fingerprint, teacher_weight_files, tokenizer_identity,
)

logger = logging.getLogger(__name__)


class TeacherClient:
    """Score original student tokens through vLLM's echoed completions API."""

    def __init__(self, config: TeacherConfig, timeout: float) -> None:
        """Store an external service contract without contacting it."""
        self.config = config
        self.timeout = timeout

    def _request(self, route: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Execute one bounded HTTP request and propagate errors."""
        request = Request(
            self.config.endpoint + "/" + route,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:  # nosec B310 - validated HTTP(S) endpoint
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError("Teacher returned a non-object response")
        return result

    def validate_service(self) -> None:
        """Require the configured served model before training starts."""
        models = self._request("v1/models").get("data", [])
        selected = [model for model in models if model.get("id") == self.config.served_model_name]
        if len(selected) != 1:
            raise ValueError(f"Teacher endpoint does not uniquely serve {self.config.served_model_name!r}")
        if selected[0].get("root") != self.config.model_path:
            raise ValueError("Teacher service checkpoint path differs from configured model_path")

    def score(self, token_ids: list[int]) -> list[float]:
        """Return one raw teacher logprob per next-token prediction position."""
        if len(token_ids) < 2 or len(token_ids) + 1 > self.config.max_model_len:
            raise ValueError("Teacher context must fit the complete student sequence plus one generated token")
        result = self._request(
            "v1/completions",
            {
                "model": self.config.served_model_name,
                "prompt": token_ids,
                "echo": True,
                "logprobs": 0,
                "prompt_logprobs": 0,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "stream": False,
            },
        )
        choices = result.get("choices", [])
        if len(choices) != 1:
            raise ValueError("Teacher must return exactly one scored sequence")
        logprobs = choices[0].get("logprobs") or {}
        tokens = logprobs.get("tokens", [])
        values = logprobs.get("token_logprobs", [])
        expected = [f"token_id:{token}" for token in token_ids]
        if tokens[: len(token_ids)] != expected or len(values) < len(token_ids):
            raise ValueError("Teacher echo does not match the original student token IDs")
        scores = values[1 : len(token_ids)]
        if any(
            isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value > 1.0e-5
            for value in scores
        ):
            raise ValueError("Teacher returned missing, non-finite or positive log probabilities")
        return [float(value) for value in scores]


class TeacherManager:
    """Own routing and clients, without owning teacher model processes."""

    def __init__(self, config: DistillationConfig, client_factory: Callable[..., Any] = TeacherClient) -> None:
        """Create lightweight clients for the configured frozen teachers."""
        self.config = config
        self.clients = {key: client_factory(teacher, config.timeout_seconds) for key, teacher in config.teachers}
        self.fingerprint: Optional[str] = None

    def validate(self, student_tokenizer_path: str) -> None:
        """Validate identical tokenizer semantics and service identities once."""
        self.fingerprint = None
        student = tokenizer_identity(student_tokenizer_path)
        for key, teacher in self.config.teachers:
            if tokenizer_identity(teacher.tokenizer_path) != student:
                raise ValueError(
                    f"Teacher {key!r} tokenizer differs from the student: "
                    f"{teacher.tokenizer_path!r} vs {student_tokenizer_path!r}. "
                    "Sampled-token OPD requires identical token IDs and tokenizer artifacts; "
                    "cross-tokenizer distillation is not supported."
                )
            if not (Path(teacher.model_path) / "config.json").is_file():
                raise ValueError(f"Teacher {key!r} requires local immutable model artifacts")
            if not teacher_weight_files(teacher.model_path):
                raise ValueError(f"Teacher {key!r} requires local checkpoint weights")
            self.clients[key].validate_service()
        logger.info("Capturing teacher identities: reading all checkpoint weight shards once at startup")
        self.fingerprint = distillation_fingerprint(self.config)

    def resolve(self, metadata: dict[str, Any]) -> str:
        """Select exactly one teacher; unknown or missing multi-teacher keys fail."""
        if len(self.clients) == 1:
            return next(iter(self.clients))
        key = metadata.get(self.config.teacher_key)
        if not isinstance(key, str) or key not in self.clients:
            raise ValueError(f"Missing or unknown teacher route {key!r}; expected {sorted(self.clients)}")
        return key

    def score_batch(
        self,
        experience: ExperienceBatch,
        prompt_records: Any,
    ) -> tuple[list[list[float]], tuple[str, ...]]:
        """Score real rows grouped by teacher; restore order and leave padding zero."""
        prompts = {record.prompt_id: record.metadata for record in prompt_records}
        rows, width = experience.sequences.shape
        if len(experience.trajectories) != rows:
            raise ValueError("Teacher scoring requires one original trajectory per batch row")
        scores = [[0.0] * (width - 1) for _ in range(rows)]
        keys = [""] * rows
        grouped: dict[str, list[tuple[int, Any]]] = {}
        for index, trajectory in enumerate(experience.trajectories):
            if trajectory.metadata.get("dp_padding", False):
                continue
            if trajectory.prompt_id not in prompts:
                raise ValueError(f"Teacher route has no original prompt {trajectory.prompt_id!r}")
            key = self.resolve(prompts[trajectory.prompt_id])
            keys[index] = key
            grouped.setdefault(key, []).append((index, trajectory))
        for key, group in grouped.items():
            for index, trajectory in group:
                tokens = trajectory.token_ids.detach().cpu().tolist()
                values = self.clients[key].score(tokens)
                if len(values) != len(tokens) - 1:
                    raise ValueError("Teacher scores must align with original next-token positions")
                if not all(math.isfinite(value) and value <= 1.0e-5 for value in values):
                    raise ValueError("Teacher scores must be finite nonpositive log probabilities")
                scores[index][: len(values)] = values
        return scores, tuple(keys)


def attach_teacher_scores(
    experience: ExperienceBatch, result: tuple[list[list[float]], tuple[str, ...]]
) -> ExperienceBatch:
    """Build an immutable batch after distributed scoring has completed successfully."""
    scores, keys = result
    values = torch.tensor(scores, dtype=torch.float32, device=experience.sequences.device)
    return replace(experience, teacher_log_probs=values.detach(), teacher_keys=keys)
