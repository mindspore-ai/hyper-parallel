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
"""Trainer-owned model-integration validation lifecycle."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from hyper_parallel.tools.model_integration.checkpoint_coverage import build_checkpoint_coverage
from hyper_parallel.tools.model_integration.data_contract import (
    modality_gradient_parameters,
    validate_data_contract,
    validate_observed_forward_fields,
)
from hyper_parallel.tools.model_integration.evidence_store import EvidenceStore
from hyper_parallel.tools.model_integration.optimizer_layout import (
    flatten_tensor_layouts,
    snapshot_training_layout,
    validate_optimizer_layout,
)
from hyper_parallel.tools.model_integration.tensor_probes import (
    InputIdentityRecorder,
    ParameterProbeRecorder,
)
from hyper_parallel.tools.model_integration.schemas import IntegrationState
from hyper_parallel.tools.model_integration.schemas import ModelIntegrationReport
from hyper_parallel.tools.model_integration.integration_validator import (
    collect_activation_checkpoint_wrappers,
    inventory_final_model,
    resolve_model_validation_spec,
    validate_final_model,
    validate_state_invariants,
)
from hyper_parallel.trainer.runtime.memory import peak_device_memory, reset_peak_device_memory


_OUTPUT_DIR_ENV = "HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR"
_SMALL_PARAMETER_NUMEL = 8192


def _checkpoint_event_name(event: str, checkpoint_path: str) -> str:
    """Build a filesystem-safe event name that preserves checkpoint identity."""
    checkpoint_name = Path(checkpoint_path).name
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in checkpoint_name
    )
    return f"{event}_{safe_name}" if safe_name else event


class ModelIntegrationSession(Protocol):
    """Trainer-facing lifecycle shared by disabled and enabled sessions."""

    runtime_enabled: bool

    def attach_optimizer(self, optimizer: Any) -> None:
        """Attach the final optimizer."""

    def attach_data_pipeline(self, **pipeline: Any) -> None:
        """Attach resolved data-pipeline objects."""

    def begin_step(self, step: int) -> None:
        """Begin one optimizer step."""

    def record_batch(
        self,
        model_inputs: Mapping[str, Any],
        loss_inputs: Mapping[str, Any],
    ) -> None:
        """Observe one micro-batch."""

    def after_backward_before_clip(self) -> None:
        """Observe gradients before clipping."""

    def after_clip(self, grad_norm: Any) -> None:
        """Observe gradients and their known norm after clipping."""

    def after_optimizer(self) -> None:
        """Observe parameters after the optimizer step."""

    def end_step(self, metrics: Mapping[str, Any]) -> None:
        """Finish one optimizer step."""

    def record_fsdp_trace(self, trace: Mapping[str, Any]) -> None:
        """Record optional FSDP runtime evidence."""

    def capture_checkpoint_payload(
        self,
        event: str,
        checkpoint_path: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Record optional checkpoint layout evidence."""


def _analyze_shared_state_events(
    events: tuple[dict[str, Any], ...],
    producer_keys: tuple[str, ...],
    consumer_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Normalize trace identities and classify shared-state contract failures."""
    state_indices: dict[Any, int] = {}
    normalized_events = []
    for event in events:
        state_index = state_indices.setdefault(event["state_id"], len(state_indices))
        normalized_events.append({**event, "state_id": state_index})

    published = set()
    duplicate_publications = []
    missing_sources = []
    for event in normalized_events:
        identity = (event["state_id"], event["key"], event["source_layer"])
        if event["action"] == "publish":
            if identity in published:
                duplicate_publications.append(identity)
            published.add(identity)
        elif identity not in published:
            missing_sources.append(event)

    produced_keys = {
        event["key"] for event in normalized_events if event["action"] == "publish"
    }
    consumed_keys = {
        event["key"] for event in normalized_events if event["action"] == "consume"
    }
    return {
        "events": normalized_events,
        "missing_producer_keys": sorted(set(producer_keys) - produced_keys),
        "missing_consumer_keys": sorted(set(consumer_keys) - consumed_keys),
        "missing_sources": missing_sources,
        "duplicate_publications": duplicate_publications,
    }


class DisabledModelIntegrationSession:
    """No-op session used to keep validation branches out of training code."""

    runtime_enabled = False

    def attach_optimizer(self, optimizer: Any) -> None:
        """Ignore optimizer construction when validation is disabled."""
        del optimizer

    def attach_data_pipeline(self, **pipeline: Any) -> None:
        """Ignore data-pipeline construction when validation is disabled."""
        del pipeline

    def begin_step(self, step: int) -> None:
        """Ignore the optimizer-step boundary when validation is disabled."""
        del step

    def record_batch(
        self,
        model_inputs: Mapping[str, Any],
        loss_inputs: Mapping[str, Any],
    ) -> None:
        """Ignore model inputs when validation is disabled."""
        del model_inputs, loss_inputs

    def after_backward_before_clip(self) -> None:
        """Ignore the pre-clip boundary when validation is disabled."""

    def after_clip(self, grad_norm: Any) -> None:
        """Ignore the post-clip boundary when validation is disabled."""
        del grad_norm

    def after_optimizer(self) -> None:
        """Ignore the optimizer boundary when validation is disabled."""

    def end_step(self, metrics: Mapping[str, Any]) -> None:
        """Ignore step metrics when validation is disabled."""
        del metrics

    def record_fsdp_trace(self, trace: Mapping[str, Any]) -> None:
        """Ignore FSDP traces when validation is disabled."""
        del trace

    def capture_checkpoint_payload(
        self,
        event: str,
        checkpoint_path: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Ignore checkpoint payloads when validation is disabled."""
        del event, checkpoint_path, payload


class TrainerModelIntegrationSession:
    """Coordinate build checks and runtime evidence without changing execution."""

    def __init__(
        self,
        *,
        model: Any,
        config: Any,
        global_rank: int,
        mesh_context: Any = None,
    ) -> None:
        """Validate the final model and persist integration evidence."""
        self.model = model
        self.config = config
        self.global_rank = global_rank
        self.mode = config.model_integration.mode
        configured_output = os.environ.get(_OUTPUT_DIR_ENV)
        if not configured_output:
            configured_output = str(
                Path(config.checkpoint.checkpoint_dir) / "model_integration"
            )
        self.evidence_store = EvidenceStore(configured_output)
        self.parameter_probes: Optional[ParameterProbeRecorder] = None
        self.optimizer: Any = None
        self.validation_spec = resolve_model_validation_spec(self.model)
        data_spec = self.validation_spec.data if self.validation_spec is not None else None
        self.input_identity = InputIdentityRecorder(
            mesh_context,
            cp_replicated_forward_fields=(
                () if data_spec is None else data_spec.cp_replicated_forward_fields
            ),
        )
        self._current_step = 0
        self._step_start_time = 0.0
        self._grad_norm_post_clip: Optional[float] = None
        self._shared_trace_token: Any = None
        self._observed_forward_fields = False
        self._modality_fields_observed = False
        self._field_ownership: dict[str, str] = {}
        self._validate_final_model()

    @property
    def runtime_enabled(self) -> bool:
        """Whether step-level precision and execution probes are enabled."""
        return self.mode == "runtime"

    def _validate_final_model(self) -> None:
        report, ownership, fsdp_units = validate_final_model(
            self.model,
            context={"trainer_config": self.config.to_dict()},
        )
        self.parameter_ownership = ownership
        if self.global_rank == 0:
            self.evidence_store.write_json("check/findings.json", report.to_dict())
            self.evidence_store.write_json("check/fsdp_units.json", fsdp_units)
            self.evidence_store.write_csv(
                "check/parameter_ownership.csv",
                (entry.to_dict() for entry in ownership),
                fieldnames=(
                    "parameter_fqn",
                    "owner_fqn",
                    "gradient_domain",
                    "source_mesh",
                    "source_placements",
                    "global_shape",
                    "local_shape",
                ),
            )
            self.evidence_store.write_json("trainer_config.resolved.json", self.config.to_dict())
            self.evidence_store.capture_environment()
            self.evidence_store.advance_if_before(
                IntegrationState.DISCOVERED,
                ("environment.json",),
            )
            if not report.errors:
                self.evidence_store.advance_if_before(
                    IntegrationState.STRUCTURE_VALIDATED,
                    (
                        "check/findings.json",
                        "check/parameter_ownership.csv",
                        "check/fsdp_units.json",
                    ),
                )
            self.evidence_store.write_json(
                "inventory/candidate.json",
                inventory_final_model(self.model),
            )
            checkpoint_spec = (
                self.validation_spec.checkpoint
                if self.validation_spec is not None
                else None
            )
            checkpoint_coverage, _ = build_checkpoint_coverage(
                self.model,
                checkpoint_spec,
            )
            self.evidence_store.write_json(
                "checkpoint/coverage.json",
                checkpoint_coverage,
            )
            self.evidence_store.write_json(
                "check/activation_checkpoint.json",
                collect_activation_checkpoint_wrappers(self.model),
            )
        report.raise_for_errors()

    def attach_optimizer(self, optimizer: Any) -> None:
        """Resolve the semantic parameter watchlist after optimizer construction."""
        self.optimizer = optimizer
        findings = validate_optimizer_layout(self.model, optimizer)
        if self.global_rank == 0:
            self.evidence_store.write_json(
                "check/optimizer_layout_initial.json",
                {
                    "status": "FAIL" if findings else "PASS",
                    "findings": [finding.to_dict() for finding in findings],
                    "layout": snapshot_training_layout(self.model, optimizer),
                },
            )
        if findings:
            ModelIntegrationReport(findings=findings).raise_for_errors()
        if not self.runtime_enabled:
            return
        self.parameter_probes = ParameterProbeRecorder(
            self.model,
            optimizer,
            self.evidence_store,
            rank=self.global_rank,
            validation_spec=self.validation_spec,
            include_small_parameters_numel_le=_SMALL_PARAMETER_NUMEL,
            representative_parameter_names=self._representative_parameter_names(),
        )
        if self.global_rank == 0:
            self.evidence_store.write_json(
                "check/parameter_watchlist.json",
                {"parameters": list(self.parameter_probes.watched_parameter_names)},
            )

    def _representative_parameter_names(self) -> tuple[str, ...]:
        """Select at least one parameter from every owner/domain/placement group."""
        representatives = {}
        for entry in self.parameter_ownership:
            key = (
                entry.owner_fqn,
                entry.gradient_domain,
                entry.source_mesh,
                entry.source_placements,
            )
            representatives.setdefault(key, entry.parameter_fqn)
        return tuple(sorted(representatives.values()))

    def attach_data_pipeline(
            self,
            *,
            runtime_adapter: Any,
            get_batch: Any,
            model_assets: tuple[Any, ...],
    ) -> None:
        """Validate and record the resolved data/forward-input ownership."""
        ownership, findings = validate_data_contract(
            self.validation_spec,
            runtime_adapter,
            labels_are_shifted=getattr(get_batch, "labels_are_shifted", None),
            source_type=getattr(get_batch, "source_type", None),
            assets=model_assets,
        )
        self._field_ownership = dict(ownership["fields"])
        if self.global_rank == 0:
            self.evidence_store.write_json("data/field_ownership.json", ownership)
            self.evidence_store.write_json(
                "data/findings.json",
                {
                    "status": "FAIL" if findings else "PASS",
                    "findings": [finding.to_dict() for finding in findings],
                },
            )
        if findings:
            ModelIntegrationReport(findings=findings).raise_for_errors()

    def begin_step(self, step: int) -> None:
        """Begin input and parameter evidence for one optimizer step."""
        if not self.runtime_enabled:
            return
        self._current_step = step
        self._step_start_time = time.perf_counter()
        self._grad_norm_post_clip = None
        reset_peak_device_memory()
        shared_state_spec = (
            self.validation_spec.shared_state
            if self.validation_spec is not None
            else None
        )
        if shared_state_spec is not None and shared_state_spec.begin_trace is not None:
            self._shared_trace_token = shared_state_spec.begin_trace()
        self.input_identity.begin_step()
        if self.parameter_probes is not None:
            self.parameter_probes.begin_step(step)

    def record_batch(
        self,
        model_inputs: Mapping[str, Any],
        loss_inputs: Mapping[str, Any],
    ) -> None:
        """Record exact tensor fields reaching the model/loss boundary.

        Args:
            model_inputs: Keyword inputs passed to the model forward call.
            loss_inputs: Labels and masks consumed by the loss path.
        """
        if not self.runtime_enabled:
            return
        if not self._observed_forward_fields:
            findings = validate_observed_forward_fields(
                self.validation_spec,
                model_inputs,
            )
            data_spec = self.validation_spec.data if self.validation_spec is not None else None
            if data_spec is not None:
                self._modality_fields_observed = bool(
                    set(data_spec.modality_fields).intersection(model_inputs)
                )
            if self.global_rank == 0:
                actual_ownership = {
                    field_name: self._field_ownership.get(
                        field_name,
                        "source_or_get_batch",
                    )
                    for field_name in sorted(model_inputs)
                }
                self.evidence_store.write_json(
                    "data/observed_forward_fields.json",
                    {
                        "status": "FAIL" if findings else "PASS",
                        "model_fields": sorted(model_inputs),
                        "loss_fields": sorted(loss_inputs),
                        "field_ownership": actual_ownership,
                        "findings": [finding.to_dict() for finding in findings],
                    },
                )
            self._observed_forward_fields = True
            if findings:
                ModelIntegrationReport(findings=findings).raise_for_errors()
        self.input_identity.record(model_inputs, loss_inputs)

    def after_backward_before_clip(self) -> None:
        """Capture un-clipped main gradients."""
        if not self.runtime_enabled:
            return
        runtime_findings = validate_state_invariants(
            self.model,
            self.validation_spec,
            {"trainer": self, "step": self._current_step},
            phases=("runtime",),
        )
        if runtime_findings:
            ModelIntegrationReport(findings=runtime_findings).raise_for_errors()
        if self.parameter_probes is not None:
            self.parameter_probes.capture("main_grad_before_clip")
        if self.optimizer is not None:
            findings = validate_optimizer_layout(self.model, self.optimizer)
            if self.global_rank == 0:
                self.evidence_store.write_json(
                    f"cases/runtime/optimizer_layout_step{self._current_step}.json",
                    {
                        "status": "FAIL" if findings else "PASS",
                        "findings": [finding.to_dict() for finding in findings],
                        "layout": snapshot_training_layout(self.model, self.optimizer),
                    },
                )
            if findings:
                ModelIntegrationReport(findings=findings).raise_for_errors()
        if self._modality_fields_observed:
            modality_gradients = modality_gradient_parameters(
                self.validation_spec,
                self.model,
            )
            if self.global_rank == 0:
                self.evidence_store.write_json(
                    f"cases/runtime/modality_gradients_step{self._current_step}.json",
                    modality_gradients,
                )
            if modality_gradients and not all(modality_gradients.values()):
                raise RuntimeError(
                    "[HP-DATA-002] multimodal inputs reached the model but one or more "
                    "declared visual parameters have missing or non-finite gradients"
                )

    def record_fsdp_trace(self, trace: Mapping[str, Any]) -> None:
        """Persist rank-local runtime order before the optimizer-boundary gate."""
        if not self.runtime_enabled:
            return
        self.evidence_store.write_json(
            f"check/fsdp_runtime_rank{self.global_rank}.json",
            dict(trace),
        )

    def after_clip(self, grad_norm: Any) -> None:
        """Capture clipped main gradients."""
        if not self.runtime_enabled:
            return
        if grad_norm is None:
            raise ValueError("runtime model integration requires the post-clip gradient norm")
        if self.parameter_probes is not None:
            self.parameter_probes.capture("main_grad_after_clip")
        self._grad_norm_post_clip = float(grad_norm)

    def after_optimizer(self) -> None:
        """Capture main/model updates and optimizer moment layouts."""
        if not self.runtime_enabled:
            return
        if self.parameter_probes is None:
            return
        self.parameter_probes.capture("main_after_optimizer")
        self.parameter_probes.capture("model_after_copy_back")
        self.parameter_probes.capture("optimizer_state")

    def end_step(self, metrics: Mapping[str, Any]) -> None:
        """Persist structured scalar and global input identity evidence."""
        if not self.runtime_enabled:
            return
        self._finish_shared_state_trace()
        input_identity = self.input_identity.finish_step()
        elapsed = time.perf_counter() - self._step_start_time
        tokens = metrics.get("tokens")
        samples = metrics.get("samples")
        memory = peak_device_memory()
        row = {
            "step": self._current_step,
            "loss": metrics.get("loss"),
            "grad_norm_pre_clip": metrics.get("grad_norm"),
            "grad_norm_post_clip": metrics.get(
                "grad_norm_post_clip",
                self._grad_norm_post_clip,
            ),
            "lr": metrics.get("lr"),
            "global_input_sha256": input_identity["global_sha256"],
            "input_identity": input_identity,
            "step_time_seconds": elapsed,
            "tokens_per_second": None if tokens is None else float(tokens) / elapsed,
            "samples_per_second": None if samples is None else float(samples) / elapsed,
            **memory,
        }
        performance_row = {
            "step": self._current_step,
            "rank": self.global_rank,
            "step_time_seconds": elapsed,
            "tokens_per_second": row["tokens_per_second"],
            "samples_per_second": row["samples_per_second"],
            **memory,
        }
        self.evidence_store.append_jsonl(
            f"cases/runtime/performance_rank{self.global_rank}.jsonl",
            performance_row,
        )
        if self.global_rank == 0:
            self.evidence_store.append_jsonl("cases/runtime/metrics.jsonl", row)
            self.evidence_store.append_jsonl(
                "cases/runtime/performance.jsonl",
                performance_row,
            )

    def _finish_shared_state_trace(self) -> None:
        """Validate model-declared shared-state publication and consumption."""
        if self._shared_trace_token is None or self.validation_spec is None:
            return
        shared_state_spec = self.validation_spec.shared_state
        if shared_state_spec is None or shared_state_spec.finish_trace is None:
            return
        events = shared_state_spec.finish_trace(self._shared_trace_token)
        self._shared_trace_token = None
        analysis = _analyze_shared_state_events(
            events,
            shared_state_spec.producer_keys,
            shared_state_spec.consumer_keys,
        )
        if self.global_rank == 0:
            self.evidence_store.write_json(
                f"cases/runtime/shared_state_step{self._current_step}.json",
                analysis,
            )
        if (
            analysis["missing_sources"]
            or analysis["missing_producer_keys"]
            or analysis["missing_consumer_keys"]
        ):
            raise RuntimeError(
                "[HP-STATE-001] shared-state producer/consumer contract failed: "
                f"missing_producers={analysis['missing_producer_keys']}, "
                f"missing_consumers={analysis['missing_consumer_keys']}, "
                f"missing_sources={analysis['missing_sources']}"
            )
        if analysis["duplicate_publications"]:
            raise RuntimeError(
                "[HP-STATE-002] shared-state producer replayed inside one forward: "
                f"{analysis['duplicate_publications']}"
            )

    def capture_checkpoint_payload(
            self,
            event: str,
            checkpoint_path: str,
            payload: Mapping[str, Any],
    ) -> None:
        """Record versioned DCP layouts without rebuilding checkpoint chunks."""
        if not self.runtime_enabled:
            return
        event_name = _checkpoint_event_name(event, checkpoint_path)
        self.evidence_store.write_json(
            f"checkpoint/{event_name}_rank{self.global_rank}.json",
            {
                "checkpoint_path": checkpoint_path,
                "tensor_layouts": flatten_tensor_layouts(payload),
            },
        )


def build_model_integration_session(
    model: Any,
    config: Any,
    global_rank: int,
    mesh_context: Any = None,
) -> TrainerModelIntegrationSession | DisabledModelIntegrationSession:
    """Build the configured validation session or a branch-free no-op session."""
    if getattr(getattr(config, "model_integration", None), "mode", "off") == "off":
        return DisabledModelIntegrationSession()
    return TrainerModelIntegrationSession(
        model=model,
        config=config,
        global_rank=global_rank,
        mesh_context=mesh_context,
    )


__all__ = [
    "DisabledModelIntegrationSession",
    "ModelIntegrationSession",
    "TrainerModelIntegrationSession",
    "build_model_integration_session",
]
