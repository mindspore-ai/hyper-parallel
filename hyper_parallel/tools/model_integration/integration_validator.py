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
"""Build/runtime model integration validators backed by final model and FSDP state facts."""

from __future__ import annotations

import fnmatch
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from hyper_parallel.tools.model_integration.structure_inventory import build_model_inventory
from hyper_parallel.tools.model_integration.checkpoint_coverage import build_checkpoint_coverage
from hyper_parallel.tools.model_integration.schemas import (
    ModelIntegrationReport,
    FindingSeverity,
    ModelIntegrationFinding,
)
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.models.validation_spec import ModelValidationSpec


@dataclass(frozen=True)
class ParameterOwnership:
    """One final parameter's inferred FSDP owner and gradient domain."""

    parameter_fqn: str
    owner_fqn: str
    gradient_domain: str
    source_mesh: tuple[str, ...]
    source_placements: tuple[str, ...]
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    parameter_identity: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a flat representation suitable for CSV or JSON."""
        return {
            "parameter_fqn": self.parameter_fqn,
            "owner_fqn": self.owner_fqn,
            "gradient_domain": self.gradient_domain,
            "source_mesh": ",".join(self.source_mesh),
            "source_placements": ",".join(self.source_placements),
            "global_shape": repr(self.global_shape),
            "local_shape": repr(self.local_shape),
        }


def resolve_model_validation_spec(model: Any) -> Optional[ModelValidationSpec]:
    """Resolve and type-check the final model family's validation provider."""
    config = getattr(model, "config", None)
    identities = [getattr(config, "model_type", None)]
    identities.extend(getattr(config, "architectures", None) or ())
    adapter_spec = None
    for identity in identities:
        if not identity:
            continue
        adapter_spec = get_model_adapter(identity)
        if adapter_spec is not None:
            break
    provider = getattr(adapter_spec, "validation", None)
    if provider is None:
        return None
    if not callable(provider):
        raise TypeError("ModelAdapterSpec.validation must be callable")
    validation_spec = provider()  # pylint: disable=not-callable
    if not isinstance(validation_spec, ModelValidationSpec):
        raise TypeError(
            "ModelAdapterSpec.validation must return ModelValidationSpec, "
            f"got {type(validation_spec).__name__}"
        )
    return validation_spec


def _tensor_local_shape(tensor: Any) -> tuple[int, ...]:
    to_local = getattr(tensor, "to_local", None)
    local = to_local() if callable(to_local) else tensor
    return tuple(int(size) for size in getattr(local, "shape", ()))


def _source_layout(hsdp_param: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    source_info = getattr(hsdp_param, "source_shard_info", None)
    source_mesh = getattr(source_info, "mesh", None)
    names = tuple(getattr(source_mesh, "mesh_dim_names", None) or ())
    placements = tuple(repr(value) for value in getattr(source_info, "placements", ()) or ())
    return names, placements


def _gradient_domain(hsdp_param: Any) -> str:
    names, placements = _source_layout(hsdp_param)
    expert_axis = any(name in ("ep", "expert") or name.startswith("ep_") for name in names)
    expert_shard = any("Shard" in placement for placement in placements)
    if expert_axis and expert_shard:
        return "EXPERT"
    return "DENSE"


def collect_parameter_ownership(model: Any) -> tuple[list[ParameterOwnership], dict[str, Any]]:
    """Collect parameter-to-FSDP-unit ownership from final runtime state."""
    ownership = []
    units = {}
    seen_parameters = set()
    for module_fqn, module in model.named_modules():
        scheduler = getattr(module, "hsdp_scheduler", None)
        state = getattr(scheduler, "hsdp_state", None)
        hsdp_params = tuple(getattr(state, "hsdp_params", ()) or ())
        if scheduler is None:
            continue
        owner_fqn = module_fqn or "<root>"
        unit_domains = defaultdict(list)
        for parameter_index, hsdp_param in enumerate(hsdp_params):
            parameter = getattr(hsdp_param, "sharded_param", None)
            parameter_fqn = getattr(hsdp_param, "_param_fqn", None) or (
                f"{owner_fqn}.param[{parameter_index}]"
            )
            names, placements = _source_layout(hsdp_param)
            domain = _gradient_domain(hsdp_param)
            unit_domains[domain].append(parameter_fqn)
            global_shape = tuple(
                int(size) for size in getattr(hsdp_param, "_orig_size", getattr(parameter, "shape", ()))
            )
            ownership.append(
                ParameterOwnership(
                    parameter_fqn=parameter_fqn,
                    owner_fqn=owner_fqn,
                    gradient_domain=domain,
                    source_mesh=names,
                    source_placements=placements,
                    global_shape=global_shape,
                    local_shape=_tensor_local_shape(parameter),
                    parameter_identity=id(parameter) if parameter is not None else 0,
                )
            )
            if parameter is not None:
                seen_parameters.add(id(parameter))
        units[owner_fqn] = {
            "domains": {name: sorted(fqns) for name, fqns in sorted(unit_domains.items())},
            "parameter_count": len(hsdp_params),
            "forward_prefetch_targets": [
                _module_fqn(model, target)
                for target in getattr(scheduler, "forward_prefetch_cells", ())
            ],
            "backward_prefetch_targets": [
                _module_fqn(model, target)
                for target in getattr(scheduler, "backward_prefetch_cells", ())
            ],
        }
    for parameter_fqn, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in seen_parameters:
            continue
        ownership.append(
            ParameterOwnership(
                parameter_fqn=parameter_fqn,
                owner_fqn="<unmanaged>",
                gradient_domain="LOCAL" if not parameter.requires_grad else "DENSE",
                source_mesh=(),
                source_placements=(),
                global_shape=tuple(int(size) for size in parameter.shape),
                local_shape=_tensor_local_shape(parameter),
                parameter_identity=id(parameter),
            )
        )
    return ownership, units


def _module_fqn(model: Any, target: Any) -> str:
    for module_fqn, module in model.named_modules():
        if module is target:
            return module_fqn or "<root>"
    return f"<untracked:{type(target).__name__}>"


def _validate_gradient_domains(
    ownership: Iterable[ParameterOwnership],
) -> list[ModelIntegrationFinding]:
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in ownership:
        if entry.owner_fqn == "<unmanaged>" or entry.gradient_domain == "LOCAL":
            continue
        grouped[entry.owner_fqn][entry.gradient_domain].append(entry)
    findings = []
    aliases = defaultdict(list)
    for entry in ownership:
        if entry.parameter_identity:
            aliases[entry.parameter_identity].append(entry)
    for entries in aliases.values():
        contracts = {
            (
                entry.owner_fqn,
                entry.gradient_domain,
                entry.source_mesh,
                entry.source_placements,
            )
            for entry in entries
        }
        if len(contracts) <= 1:
            continue
        findings.append(
            ModelIntegrationFinding(
                code="HP-FSDP-006",
                phase="D3",
                owner_fqn=entries[0].parameter_fqn,
                severity=FindingSeverity.ERROR,
                message="aliased parameter has conflicting ownership or gradient domains",
                facts={
                    "aliases": [entry.parameter_fqn for entry in entries],
                    "contracts": [repr(contract) for contract in sorted(contracts, key=repr)],
                },
                why_unsafe="one storage cannot be reduced or sharded under two incompatible contracts",
                remediation="assign every alias of the tied parameter to one owner and source layout",
                related_config=("plan_overrides", "model adapter fsdp_wrap_modules"),
            )
        )
    for owner_fqn, domain_entries in sorted(grouped.items()):
        domains = set(domain_entries)
        if len(domains) > 1:
            facts = {
                domain: [entry.parameter_fqn for entry in entries]
                for domain, entries in sorted(domain_entries.items())
            }
            findings.append(
                ModelIntegrationFinding(
                    code="HP-FSDP-004",
                    phase="D3",
                    owner_fqn=owner_fqn,
                    severity=FindingSeverity.ERROR,
                    message="FSDP unit has incompatible gradient domains",
                    facts=facts,
                    why_unsafe=(
                        "one FSDP unit selects one collective mesh, so dense and expert "
                        "parameters cannot both receive their required gradient reduction"
                    ),
                    remediation=(
                        "declare homogeneous child FSDP units and keep dense gates/norms "
                        "in a dense owner"
                    ),
                    related_config=("model adapter fsdp_wrap_modules", "plan_overrides"),
                )
            )
        if owner_fqn == "<root>" and "EXPERT" in domains:
            findings.append(
                ModelIntegrationFinding(
                    code="HP-FSDP-005",
                    phase="D3",
                    owner_fqn=owner_fqn,
                    severity=FindingSeverity.ERROR,
                    message="expert-domain parameter remains in the dense root FSDP unit",
                    facts={
                        "expert_parameters": [
                            entry.parameter_fqn for entry in domain_entries["EXPERT"]
                        ]
                    },
                    why_unsafe="the dense root does not own the expert reduction domain",
                    remediation="declare a homogeneous nested expert FSDP unit",
                    related_config=("model adapter fsdp_wrap_modules",),
                )
            )
    return findings


def _validate_materialization(model: Any) -> list[ModelIntegrationFinding]:
    meta_state = []
    for name, value in (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ):
        if bool(getattr(value, "is_meta", False)):
            meta_state.append(name)
    findings = []
    if meta_state:
        findings.append(
            ModelIntegrationFinding(
                code="HP-MAT-001",
                phase="D4",
                owner_fqn="<root>",
                severity=FindingSeverity.ERROR,
                message="materialized model still contains meta tensors",
                facts={"state_fqns": sorted(meta_state)},
                why_unsafe="forward or checkpoint access would read storage that does not exist",
                remediation="materialize storage and run the adapter rebuild lifecycle before validation",
                related_config=("training.init_device", "model adapter materialization"),
            )
        )
    invalid_rebuildable_buffers = []
    for module_fqn, module in model.named_modules(remove_duplicate=False):
        specs = module.__dict__.get("_hp_rebuildable_buffer_specs", {})
        for buffer_name, spec in specs.items():
            buffer = module._buffers.get(buffer_name)  # pylint: disable=protected-access
            reason = None
            if buffer is None:
                reason = "buffer is missing"
            elif buffer_name not in module._non_persistent_buffers_set:  # pylint: disable=protected-access
                reason = "buffer became persistent"
            elif bool(getattr(buffer, "is_meta", False)):
                reason = "buffer still has meta storage"
            elif getattr(spec, "source", None) is None and getattr(spec, "factory", None) is None:
                reason = "rebuild recipe has neither source nor factory"
            if reason is not None:
                invalid_rebuildable_buffers.append(
                    {
                        "fqn": f"{module_fqn}.{buffer_name}" if module_fqn else buffer_name,
                        "reason": reason,
                    }
                )
    if invalid_rebuildable_buffers:
        findings.append(
            ModelIntegrationFinding(
                code="HP-MAT-002",
                phase="D4",
                owner_fqn="<root>",
                severity=FindingSeverity.ERROR,
                message="rebuildable derived-buffer lifecycle is incomplete",
                facts={"buffers": invalid_rebuildable_buffers},
                why_unsafe="to_empty() may leave model-derived non-persistent state undefined",
                remediation="register and execute a deterministic rebuild recipe after materialization",
                related_config=("training.init_device", "model adapter materialization"),
            )
        )
    return findings


def _validate_parameter_probes(
    model: Any,
    validation_spec: Optional[ModelValidationSpec],
) -> list[ModelIntegrationFinding]:
    if validation_spec is None:
        return []
    parameter_names = tuple(name for name, _ in model.named_parameters(remove_duplicate=False))
    findings = []
    for probe in validation_spec.parameter_probes:
        matches = [name for name in parameter_names if fnmatch.fnmatchcase(name, probe.match)]
        if probe.required and not matches:
            findings.append(
                ModelIntegrationFinding(
                    code="HP-PREC-001",
                    phase="V0",
                    owner_fqn=probe.match,
                    severity=FindingSeverity.ERROR,
                    message="required parameter probe matched no final parameter",
                    facts={"pattern": probe.match},
                    why_unsafe="a renamed or missing critical parameter would be silently unobserved",
                    remediation="update the model validation provider to match final parameter FQNs",
                    related_config=("ModelAdapterSpec.validation.parameter_probes",),
                )
            )
    return findings


def validate_state_invariants(
    model: Any,
    validation_spec: Optional[ModelValidationSpec],
    context: Any,
    phases: tuple[str, ...] = ("structure", "materialization", "checkpoint"),
) -> list[ModelIntegrationFinding]:
    """Evaluate model-owned invariants at their declared lifecycle phase."""
    findings = []
    if validation_spec is None:
        return findings
    phase_by_name = {
        "structure": "D2",
        "materialization": "D4",
        "checkpoint": "D5",
        "runtime": "D7",
    }
    for invariant in validation_spec.state_invariants:
        if invariant.phase not in phase_by_name:
            findings.append(
                ModelIntegrationFinding(
                    code="HP-SPEC-001",
                    phase="D1",
                    owner_fqn=invariant.name,
                    severity=FindingSeverity.ERROR,
                    message="model state invariant declares an unsupported lifecycle phase",
                    facts={"phase": invariant.phase, "supported": tuple(phase_by_name)},
                    why_unsafe="an invariant with an unknown phase would never be evaluated",
                    remediation="use structure, materialization, checkpoint, or runtime",
                    related_config=("ModelAdapterSpec.validation.state_invariants",),
                )
            )
            continue
        if invariant.phase not in phases:
            continue
        result = invariant.checker(model, context)
        if result is True or result is None:
            continue
        facts = result if isinstance(result, Mapping) else {"result": result}
        findings.append(
            ModelIntegrationFinding(
                code=invariant.error_code,
                phase=phase_by_name[invariant.phase],
                owner_fqn=invariant.name,
                severity=FindingSeverity.ERROR,
                message=f"model state invariant failed: {invariant.name}",
                facts=dict(facts),
                why_unsafe="the final model state does not satisfy its authoritative lifecycle contract",
                remediation="fix model initialization/checkpoint mapping or the adapter invariant declaration",
                related_config=("model adapter validation",),
            )
        )
    return findings


def validate_final_model(
    model: Any,
    *,
    context: Any = None,
) -> tuple[ModelIntegrationReport, list[ParameterOwnership], dict[str, Any]]:
    """Run model/FSDP-state model integration on the final built model.

    This function is observational. It does not change module structure,
    parameters, communication state, or optimizer state.
    """
    validation_spec = resolve_model_validation_spec(model)
    ownership, fsdp_units = collect_parameter_ownership(model)
    report = ModelIntegrationReport(
        metadata={
            "model_type": getattr(getattr(model, "config", None), "model_type", None),
            "inventory_counts": {
                "modules": len(list(model.modules())),
                "parameters": len(list(model.parameters())),
                "buffers": len(list(model.buffers())),
            },
        }
    )
    report.extend(_validate_materialization(model))
    report.extend(_validate_gradient_domains(ownership))
    checkpoint_spec = validation_spec.checkpoint if validation_spec is not None else None
    checkpoint_coverage, checkpoint_findings = build_checkpoint_coverage(model, checkpoint_spec)
    report.metadata["checkpoint_coverage"] = checkpoint_coverage
    report.extend(checkpoint_findings)
    report.extend(_validate_parameter_probes(model, validation_spec))
    report.extend(validate_state_invariants(model, validation_spec, context))
    return report, ownership, fsdp_units


def inventory_final_model(model: Any, source: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Build a serializable final-tree inventory for integration evidence."""
    return build_model_inventory(model, source=source).to_dict()


def collect_activation_checkpoint_wrappers(model: Any) -> dict[str, Any]:
    """List final activation-checkpoint wrappers without backend imports."""
    wrappers = []
    for module_fqn, module in model.named_modules(remove_duplicate=False):
        if type(module).__name__ != "CheckpointWrapper":
            continue
        wrapped = getattr(module, "_wrapped_module", None)
        wrappers.append(
            {
                "fqn": module_fqn or "<root>",
                "wrapped_type": (
                    None
                    if wrapped is None
                    else f"{type(wrapped).__module__}.{type(wrapped).__qualname__}"
                ),
            }
        )
    return {"count": len(wrappers), "wrappers": wrappers}


__all__ = [
    "ParameterOwnership",
    "collect_activation_checkpoint_wrappers",
    "collect_parameter_ownership",
    "inventory_final_model",
    "resolve_model_validation_spec",
    "validate_final_model",
    "validate_state_invariants",
]
