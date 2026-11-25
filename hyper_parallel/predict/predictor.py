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
"""Offline prediction orchestration; generation is owned exclusively by infer."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterator
import uuid

import torch

from hyper_parallel.infer import GenerationOptions, GenerationRuntime, generate, resolve_generation_options
from hyper_parallel.predict.batching import encode_requests, pad_batch, plan_batches
from hyper_parallel.predict.io import PredictionOutput, file_digest, read_requests
from hyper_parallel.predict.types import PredictionContext, PredictionSettings, PredictionSummary


def _package_version(name: str) -> str:
    """Return an installed package version, or a marker when it is absent.

    Running from a source checkout with ``PYTHONPATH`` legitimately leaves the
    distribution metadata uninstalled; provenance must not fail the run for it.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _validate_settings(config: PredictionSettings) -> None:
    """Validate file destinations and static batch limits before binding the model."""
    if not config.input or not config.output_file:
        raise ValueError("prediction.input and prediction.output_file are required")
    if Path(config.input).resolve() == Path(config.output_file).resolve():
        raise ValueError("prediction input and output must be different files")
    if (not isinstance(config.batch_size, int) or isinstance(config.batch_size, bool)) or config.batch_size < 1:
        raise ValueError("prediction.batch_size must be a positive integer")
    if config.max_batch_tokens is not None and (
            not isinstance(config.max_batch_tokens, int)
            or isinstance(config.max_batch_tokens, bool) or config.max_batch_tokens < 1):
        raise ValueError("prediction.max_batch_tokens must be a positive integer or null")


class Predictor:
    """Run a prebuilt model on static text requests using independent DP lanes.

    Args:
        model: Materialized model with weights already restored by the caller.
        tokenizer: Tokenizer used for prompt encoding and completion decoding.
        options: Serializable generation parameters.
        runtime: Code-injected generation state; no YAML runtime objects.
        config: Structural prediction settings, independent of TrainerConfig.
        context: Explicit lane ownership; defaults to one process.
        chat_template: Explicit Jinja template for messages; None permits only prompts.
        chat_template_kwargs: Template parameters, such as enable_thinking.
        metadata: Caller-owned model/checkpoint/seed provenance for the manifest.
    """

    def __init__(
            self, model: Any, tokenizer: Any, options: GenerationOptions,
            runtime: GenerationRuntime, config: PredictionSettings, *,
            context: PredictionContext | None = None, chat_template: str | None = None,
            chat_template_kwargs: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None,
    ) -> None:
        """Bind components without reading files or starting generation."""
        _validate_settings(config)
        if tokenizer is None:
            raise ValueError("prediction requires a tokenizer")
        self.model = model
        self.tokenizer = tokenizer
        self.options = resolve_generation_options(options, model, tokenizer)
        self.runtime = replace(runtime)
        if self.runtime.device is None:
            self.runtime.device = next(model.parameters()).device
        self.runtime.bind(self.options)
        self.config = config
        self.context = context or PredictionContext()
        self.chat_template = chat_template
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.metadata = dict(metadata or {})

    def _metadata(self, input_digest: str) -> dict[str, Any]:
        """Describe the resolved inputs, model, template, parameters and environment."""
        model_config = getattr(self.model, "config", None)
        return {
            "model": type(self.model).__qualname__,
            "model_config": model_config.to_dict() if hasattr(model_config, "to_dict") else None,
            **self.metadata,
            "input": str(Path(self.config.input).absolute()), "input_sha256": input_digest,
            "tokenizer": getattr(self.tokenizer, "name_or_path", type(self.tokenizer).__qualname__),
            "tokenizer_vocab_sha256": hashlib.sha256(json.dumps(
                self.tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False,
            ).encode()).hexdigest(),
            "chat_template": self.chat_template, "chat_template_kwargs": self.chat_template_kwargs,
            "generation": self.options.to_dict(),
            "batch_size": self.config.batch_size, "max_batch_tokens": self.config.max_batch_tokens,
            "bucket_by_length": self.config.bucket_by_length,
            "rng_seed": self.runtime.rng.initial_seed() if self.runtime.rng is not None else torch.initial_seed(),
            "environment": {
                "python": platform.python_version(), "torch": str(torch.__version__),
                "transformers": _package_version("transformers"),
                "hyper_parallel": _package_version("hyper_parallel"),
                "device": str(self.runtime.device),
            },
        }

    def _records(self, batches: list) -> Iterator[dict[str, Any]]:
        """Call the shared generator and decode only each row's completion span."""
        for batch in batches:
            inputs = pad_batch(batch, self.options.pad_token_id, self.runtime.device)
            result = generate(self.model, **inputs, options=self.options, runtime=self.runtime, return_dict=True)
            sequences = result.sequences.cpu()
            prompts, generated = result.prompt_lengths.tolist(), result.generated_lengths.tolist()
            for row, encoded in enumerate(batch):
                start, count = prompts[row], generated[row]
                completion = sequences[row, start:start + count].tolist()
                yield {
                    "id": encoded.request.input_id, "ordinal": encoded.request.ordinal, "sample_idx": 0,
                    "generated_text": self.tokenizer.decode(completion, skip_special_tokens=True),
                    "prompt_tokens": start, "generated_tokens": count, "finish_reason": result.finish_reasons[row],
                }

    def run_file(self) -> PredictionSummary:
        """Validate, generate and commit; each input appears exactly once in input order."""
        started = time.perf_counter()
        context = self.context
        requests = context.phase(lambda: read_requests(self.config.input))
        digest = context.phase(lambda: file_digest(self.config.input))
        if len(set(context.exchange(digest))) != 1:
            raise ValueError("prediction inputs differ across DP lanes; use shared input storage")
        local = requests[context.rank::context.world_size]
        encoded = context.phase(lambda: encode_requests(
            local, self.tokenizer, self.chat_template, self.chat_template_kwargs,
        ))
        batches = context.phase(lambda: list(plan_batches(encoded, self.config)))
        metadata = context.phase(lambda: self._metadata(digest))
        run_id = context.exchange(uuid.uuid4().hex if context.rank == 0 else None)[0]
        output = PredictionOutput(self.config.output_file, run_id, self.config.overwrite)
        try:
            context.phase(lambda: output.prepare() if context.rank == 0 else None)
            context.phase(lambda: output.write_shard(context.rank, self._records(batches)))
            context.phase(lambda: output.commit(requests, context.world_size, metadata) if context.rank == 0 else None)
            summary = PredictionSummary(len(requests), len(requests), str(output.path), time.perf_counter() - started)
            return context.exchange(summary if context.rank == 0 else None)[0]
        finally:
            if context.rank == 0:
                output.release()
