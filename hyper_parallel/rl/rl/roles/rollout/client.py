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
"""Async HTTP admission and lossless, behavior-versioned response continuation."""
import math
import threading
import time
import uuid
from typing import Any, Mapping

from rl.roles.rollout.vllm import VLLMGenerationEngine, VLLMHTTPClient


class AsyncGenerationClient(VLLMHTTPClient):
    """Block local continuations while the inference service owns the publication barrier."""

    def __init__(self, base_url: str, model_name: str, request_timeout: float, operation_timeout: float) -> None:
        super().__init__(None, base_url, model_name, request_timeout)
        self.operation_timeout = operation_timeout
        self.model_name = model_name
        self.admission = threading.Event()
        self.admission.set()
        self.closed = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any], operation_timeout: float) -> "AsyncGenerationClient":
        """Build identical producer/control transports without sharing mutable client state."""
        vllm = config["rollout"]["vllm"]
        return cls(f"http://{vllm.get('host', '127.0.0.1')}:{vllm['port']}",
                   config["model"]["registry_name"], vllm.get("request_timeout", 600), operation_timeout)

    def generate_segment(self, payload: dict) -> dict:
        """Return original aborted output stamped with the actual server policy."""
        if not self.admission.wait(self.operation_timeout):
            raise TimeoutError("Async rollout admission remained paused")
        if self.closed:
            raise RuntimeError("Async generation client is closed")
        return self._request("POST", "hyper_async/generate", payload)

    def pause(self, *, drain: bool = False) -> None:
        """Stop continuations and await the server's engine/output barrier."""
        self.admission.clear()
        options = {"timeout": self.operation_timeout} if drain else {}
        result = self._request("POST", "hyper_async/pause", {"drain": drain}, **options)
        if result.get("status") != "paused":
            raise RuntimeError("Async generation server did not acknowledge pause")

    def resume_sampling(self, version: int) -> None:
        """Release continuations after the server verifies the committed policy."""
        result = self._request("POST", "hyper_async/resume", {"policy_version": version})
        if result.get("status") != "resumed":
            raise RuntimeError("Async generation server did not acknowledge resume")
        self.admission.set()

    def close(self) -> None:
        """Wake blocked continuations during producer shutdown."""
        self.closed = True
        self.admission.set()
        super().close()


class HTTPGenerationEngine:
    """One candidate, with a separate behavior version for every response segment."""

    name = "vllm"
    is_request_owner = True

    def __init__(self, client: AsyncGenerationClient, version: int) -> None:
        self.client, self.policy_version = client, version
        self.segments: list[dict] = []
        self.interruptions = 0

    @staticmethod
    def _validate_segment(part, probabilities, total, budget) -> None:
        """Reject malformed or oversized original token/probability segments."""
        if (not isinstance(part, list) or len(part) != len(probabilities)
                or total + len(part) > budget
                or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in part)
                or any(not math.isfinite(score) or score > 1.e-5 for score in probabilities)):
            raise ValueError("Invalid partial token/probability payload")

    def generate(self, request: Any) -> Any:
        """Continue exact token prefixes without replacing earlier behavior probabilities."""
        if request.input_ids.shape[0] != 1:
            raise ValueError("Partial generation admits one candidate per request")
        settings = request.settings
        prompt = request.input_ids[0][request.attention_mask[0].bool()].tolist()
        tokens, scores = [], []
        started = time.monotonic()
        reason = None
        while len(tokens) < settings.max_new_tokens:
            if time.monotonic() - started > self.client.operation_timeout:
                raise TimeoutError("Partial response exceeded operation_timeout")
            seed = request.row_seeds[0] if request.row_seeds is not None else settings.seed
            result = self.client.generate_segment({
                "request_id": uuid.uuid4().hex, "prompt_token_ids": prompt + tokens,
                "max_tokens": settings.max_new_tokens - len(tokens), "seed": seed,
                "ignore_eos": settings.ignore_eos, "stop_token_ids": list(settings.eos_token_ids),
            })
            if result.get("paused"):
                continue
            part, probabilities = result["token_ids"], result["log_probs"]
            reason = result["finish_reason"]
            self._validate_segment(part, probabilities, len(tokens), settings.max_new_tokens)
            version = result["policy_version"]
            if part:
                self.segments.append({"start": len(tokens), "end": len(tokens) + len(part),
                                      "policy_version": version, "source_update": version - 1})
                tokens.extend(part)
                scores.extend(probabilities)
            if reason not in ("abort", "aborted"):
                if reason not in ("stop", "length"):
                    raise ValueError(f"Unexpected partial completion finish reason: {reason}")
                break
            self.interruptions += 1
        if len(tokens) == settings.max_new_tokens and reason in ("abort", "aborted"):
            reason = "length"
        return VLLMGenerationEngine.build_generation_result(
            request, [(tokens, scores, reason)], time.monotonic() - started, self.policy_version,
        )

    @staticmethod
    def synchronize_error(error: Exception | None, operation: str) -> None:
        """Propagate candidate errors without joining training collectives."""
        if error is not None:
            raise RuntimeError(f"{operation} failed") from error

    @staticmethod
    def synchronize_agent_payload(payload: Any) -> Any:
        """Return the sole candidate owner's payload."""
        if payload is None:
            raise RuntimeError("Missing candidate payload")
        return payload
