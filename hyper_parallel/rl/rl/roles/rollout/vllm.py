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
"""Lazy vLLM adapter registered without making vLLM a core dependency."""

import time
from typing import Any, Mapping, Optional, Protocol

from rl.roles.model import ModelRegistration
from rl.roles.rollout.base import GenerationRequest, GenerationResult
from rl.roles.rollout.registry import ROLLOUT_ENGINES
from rl.roles.weight_sync import PolicySnapshot
from hyper_parallel import get_platform

platform = get_platform()


class VLLMWeightRefitter(Protocol):
    """Explicit version-sensitive bridge implemented by a concrete deployment."""

    def refit(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Load one policy snapshot into an existing vLLM client."""


class VLLMGenerationEngine:
    """Adapt optional vLLM generation to the shared rollout contract."""

    name = "vllm"

    def __init__(
        self,
        model: ModelRegistration,
        config: Mapping[str, Any],
        client: Optional[Any] = None,
        refitter: Optional[VLLMWeightRefitter] = None,
    ) -> None:
        """Initialize the lazy vLLM client and optional weight refitter."""
        self._model = model
        self._config = dict(config.get("vllm", {}))
        self._client = client
        self._refitter = refitter
        self.policy_version = 0

    @property
    def client_initialized(self) -> bool:
        """Return whether the optional vLLM client has been materialized."""
        return self._client is not None

    def _ensure_client(self) -> Any:
        """Return the existing vLLM client or lazily construct one."""
        if self._client is not None:
            return self._client
        try:
            # vLLM is optional and must stay unloaded for the default Hyper engine.
            from vllm import LLM  # pylint: disable=C0415
        except ImportError as error:
            raise RuntimeError(
                "rollout.engine=vllm requires the optional vLLM package; "
                "the default Hyper engine does not"
            ) from error
        self._client = LLM(
            model=self._model.weights_path,
            tokenizer=self._model.tokenizer_path,
            tensor_parallel_size=int(self._config.get("tensor_parallel_size", 1)),
            dtype=str(self._config.get("dtype", "bfloat16")),
            trust_remote_code=bool(self._config.get("trust_remote_code", True)),
        )
        return self._client

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate variable-length responses and explicit response masks."""
        try:
            # vLLM is optional and imported only after this backend is selected.
            from vllm import SamplingParams  # pylint: disable=C0415
        except ImportError as error:
            raise RuntimeError("The vLLM adapter was selected but vLLM is not installed") from error
        client = self._ensure_client()
        settings = request.settings
        prompt_token_ids = [
            ids[mask.bool()].detach().cpu().tolist()
            for ids, mask in zip(request.input_ids, request.attention_mask)
        ]
        sampling = SamplingParams(
            n=1,
            max_tokens=settings.max_new_tokens,
            temperature=settings.temperature if settings.do_sample else 0.0,
            top_p=settings.top_p,
            top_k=settings.top_k if settings.top_k > 0 else -1,
            logprobs=1 if settings.collect_log_probs else None,
        )
        started = time.perf_counter()
        outputs = client.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling)
        elapsed = time.perf_counter() - started
        response_ids = request.input_ids.new_full(
            (len(outputs), settings.max_new_tokens), settings.pad_token_id
        )
        response_mask = response_ids.new_zeros(
            response_ids.shape,
            dtype=platform.tensor_dtype.bool,
        )
        rollout_log_probs = None
        if settings.collect_log_probs:
            rollout_log_probs = response_ids.new_zeros(
                response_ids.shape,
                dtype=platform.tensor_dtype.float32,
            )
        for row, request_output in enumerate(outputs):
            completion = request_output.outputs[0]
            tokens = list(completion.token_ids)[: settings.max_new_tokens]
            if tokens:
                response_ids[row, : len(tokens)] = response_ids.new_tensor(tokens)
                response_mask[row, : len(tokens)] = True
            if rollout_log_probs is not None and completion.logprobs is not None:
                for column, (token_id, candidates) in enumerate(
                    zip(tokens, completion.logprobs)
                ):
                    rollout_log_probs[row, column] = float(candidates[token_id].logprob)
        sequences = platform.cat((request.input_ids, response_ids), dim=-1)
        return GenerationResult(sequences, rollout_log_probs, elapsed, response_mask)

    def update_weights(self, snapshot: PolicySnapshot) -> None:
        """Refit and acknowledge a strictly newer matching policy snapshot."""
        if snapshot.model_name != self._model.name:
            raise ValueError(
                f"Policy snapshot model mismatch: expected={self._model.name}, "
                f"received={snapshot.model_name}"
            )
        if snapshot.version <= self.policy_version:
            raise ValueError(
                f"Policy snapshot version must increase: current={self.policy_version}, "
                f"received={snapshot.version}"
            )
        if self._refitter is None:
            raise NotImplementedError(
                "vLLM iterative training requires a concrete VLLMWeightRefitter; "
                "the adapter will not acknowledge a new policy version without loading it"
            )
        self._refitter.refit(self._ensure_client(), snapshot)
        self.policy_version = snapshot.version


@ROLLOUT_ENGINES.register("vllm")
def build_vllm_engine(
    config: Mapping[str, Any],
    model: ModelRegistration,
    actor: Any,
) -> VLLMGenerationEngine:
    """Build the registered optional vLLM generation engine."""
    del actor
    return VLLMGenerationEngine(model, config)
