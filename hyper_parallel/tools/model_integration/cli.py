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
"""Auditable CLI for model inspection, model integration, parity, and precision validation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from hyper_parallel.components.checkpoint import STEP_PREFIX
from hyper_parallel.tools.model_integration.evidence_store import (
    EvidenceStore,
    find_repository_root,
)
from hyper_parallel.tools.model_integration.structure_inventory import (
    ModelInventory,
    build_model_inventory,
    diff_inventories,
    inspect_local_model_assets,
)
from hyper_parallel.tools.model_integration.manifest import (
    ManifestError,
    ValidationManifest,
    load_manifest,
)
from hyper_parallel.tools.model_integration.module_parity import (
    ParityContext,
    run_module_parity,
)
from hyper_parallel.tools.model_integration.case_compare import (
    compare_checkpoint_layouts,
    compare_parameter_probes,
    compare_preflight_configs,
    compare_scalar_curves,
    load_jsonl,
    summarize_performance,
)
from hyper_parallel.tools.model_integration.case_matrix import (
    generate_validation_cases,
)
from hyper_parallel.tools.model_integration.schemas import (
    ModelIntegrationReport,
    FindingSeverity,
    IntegrationState,
    ModelIntegrationFinding,
    SCHEMA_VERSION,
)
from hyper_parallel.tools.model_integration.integration_validator import (
    validate_final_model,
)
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.models.adapter_spec import RecomputePolicy
from hyper_parallel.models.validation_spec import ModelValidationSpec


_REPOSITORY_ROOT = find_repository_root()
_SHARED_INITIAL_CHECKPOINT_STEP = 1


def _manifest_from_args(args: argparse.Namespace) -> ValidationManifest:
    manifest = load_manifest(args.manifest, output_dir=args.output_dir)
    manifest.validate_for(args.command)
    return manifest


def _store(manifest: ValidationManifest) -> EvidenceStore:
    store = EvidenceStore(manifest.output_dir)
    store.write_yaml("manifest.resolved.yaml", manifest.to_dict())
    store.capture_environment()
    return store


def _ensure_discovered(store: EvidenceStore) -> None:
    """Initialize the first validation gate for an existing adapter."""
    if store.current_state() is None:
        store.update_state(IntegrationState.DISCOVERED, ("manifest.resolved.yaml",))


def _require_state(store: EvidenceStore, expected: IntegrationState, command: str) -> None:
    """Require the previous evidence-backed gate before a dependent command."""
    actual = store.current_state()
    if actual is not expected:
        actual_name = "none" if actual is None else actual.value
        raise ManifestError(
            f"{command} requires integration state {expected.value}, got {actual_name}; "
            "run the preceding model-integration gates in the same output directory"
        )


def _require_precision_handoff(
    store: EvidenceStore,
    manifest: ValidationManifest,
) -> None:
    """Accept same-run parity state or import an explicit passing handoff."""
    current_state = store.current_state()
    if current_state is IntegrationState.MODULE_PARITY_PASSED:
        return
    if current_state in (IntegrationState.FAILED, IntegrationState.BLOCKED):
        raise ManifestError(
            f"validate cannot import a handoff into terminal integration state "
            f"{current_state.value}; use a new output directory"
        )
    handoff_path = manifest.integration_handoff_path
    if handoff_path is None:
        _require_state(store, IntegrationState.MODULE_PARITY_PASSED, "validate")
        return
    handoff = yaml.safe_load(handoff_path.read_text(encoding="utf-8"))
    store.write_yaml("integration_handoff.yaml", handoff)
    imported_evidence = []
    for handoff_key, default_path, destination in (
        ("structure_findings", "check/findings.json", "check/findings.json"),
        ("module_parity", "module_parity/comparison.json", "module_parity/comparison.json"),
        ("checkpoint_coverage", "checkpoint/coverage.json", "checkpoint/coverage.json"),
    ):
        source = Path(handoff.get(handoff_key, default_path))
        if not source.is_absolute():
            source = handoff_path.parent / source
        if not source.is_file():
            raise ManifestError(
                f"integration_handoff {handoff_key} evidence does not exist: {source}"
            )
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError(
                f"could not read integration_handoff {handoff_key} evidence: {error}"
            ) from error
        store.write_json(destination, payload)
        imported_evidence.append(destination)
    for state in (
        IntegrationState.DISCOVERED,
        IntegrationState.STRUCTURE_VALIDATED,
        IntegrationState.MODULE_PARITY_PASSED,
    ):
        store.advance_if_before(
            state,
            ("integration_handoff.yaml", *imported_evidence),
            reason="imported passing integration handoff",
        )


def _build_declared_model(manifest: ValidationManifest, section_name: str) -> Any:
    section = getattr(manifest, section_name)
    builder = section.get("builder")
    if not isinstance(builder, Mapping):
        raise ManifestError(
            f"{section_name}.builder is required and must contain _target_ for this command"
        )
    target = builder.get("_target_")
    callable_value = manifest.import_callable(target)
    kwargs = {name: value for name, value in builder.items() if name != "_target_"}
    result = callable_value(**kwargs)
    if not hasattr(result, "named_modules"):
        raise ManifestError(
            f"{section_name}.builder returned {type(result).__name__}, expected a module"
        )
    return result


def _static_inventory(inspection: Mapping[str, Any], source: Mapping[str, Any]) -> ModelInventory:
    return ModelInventory(
        source=dict(source),
        config=dict(inspection.get("config", {})),
        checkpoint=dict(inspection.get("checkpoint", {})),
    )


def _inspect_side(
    manifest: ValidationManifest,
    section_name: str,
) -> ModelInventory:
    section = getattr(manifest, section_name)
    if isinstance(section.get("builder"), Mapping):
        return build_model_inventory(
            _build_declared_model(manifest, section_name),
            source={
                "section": section_name,
                "revision": section.get("revision"),
            },
        )
    path = manifest.model_path if section_name == "model" else manifest.reference_path
    if path is None:
        raise ManifestError(
            f"{section_name} requires either a local path or an explicit builder"
        )
    return _static_inventory(
        inspect_local_model_assets(path),
        {"section": section_name, "path": str(path), "revision": section.get("revision")},
    )


def command_inspect(args: argparse.Namespace) -> int:
    """Create reference/candidate inventories and a classified diff."""
    manifest = _manifest_from_args(args)
    store = _store(manifest)
    candidate = _inspect_side(manifest, "model")
    store.write_json("inventory/candidate.json", candidate.to_dict())
    differences = []
    if manifest.reference:
        reference = _inspect_side(manifest, "reference")
        store.write_json("inventory/reference.json", reference.to_dict())
        differences = [
            difference.to_dict()
            for difference in diff_inventories(reference, candidate)
        ]
    store.write_json(
        "inventory/structure_diff.json",
        {"status": "REVIEW_REQUIRED" if differences else "PASS", "differences": differences},
    )
    store.write_json("source_fingerprints.json", manifest.fingerprint_paths())
    store.advance_if_before(
        IntegrationState.DISCOVERED,
        (
            "environment.json",
            "manifest.resolved.yaml",
            "inventory/candidate.json",
            "inventory/structure_diff.json",
        ),
    )
    print(store.root)
    return 0


_LICENSE = """# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""


def command_scaffold(args: argparse.Namespace) -> int:
    """Generate a minimal adapter/validation skeleton in a new directory."""
    manifest = _manifest_from_args(args)
    destination = Path(args.scaffold_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ManifestError(f"scaffold destination must be new or empty: {destination}")
    adapter_dir = destination / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    family = manifest.family
    files = {
        destination / "__init__.py": _LICENSE + f'"""{family} model integration package."""\n',
        adapter_dir / "__init__.py": _LICENSE + f'"""{family} adapter providers."""\n',
        adapter_dir / "registration.py": _LICENSE
        + '"""Register model-owned integration providers."""\n\n'
        + "from hyper_parallel.models.adapter_spec import ModelAdapterSpec\n"
        + "from hyper_parallel.models.registry import register_model_adapter\n\n\n"
        + "def _load_validation():\n"
        + '    """Load validation-only declarations lazily."""\n'
        + "    from .validation import get_validation_spec  # pylint: disable=C0415\n\n"
        + "    return get_validation_spec()\n\n\n"
        + "MODEL_ADAPTER_SPEC = ModelAdapterSpec(\n"
        + f"    architecture={manifest.model.get('architecture', family)!r},\n"
        + f"    model_type={manifest.model.get('model_type', family)!r},\n"
        + "    validation=_load_validation,\n"
        + ")\n"
        + "register_model_adapter(MODEL_ADAPTER_SPEC)\n",
        adapter_dir / "validation.py": _LICENSE
        + '"""Declare authoritative module and self-consistency validation."""\n\n'
        + "from hyper_parallel.models.validation_spec import ModelValidationSpec\n\n\n"
        + "def get_validation_spec() -> ModelValidationSpec:\n"
        + '    """Return model-owned validation declarations."""\n'
        + "    return ModelValidationSpec()\n",
        destination / "validation.yaml": (
            "schema_version: 1\n"
            f"model:\n  adapter: {family}\n"
            "launcher:\n"
            "  module: your.training.module\n"
            "  config: path/to/trainer.yaml\n"
            "matrix:\n"
            "  devices: 1\n"
            "  steps: 10\n"
            '  baseline: {tp: 1, cp: 1, ep: 1, fsdp: 1, recompute: "off"}\n'
            "acceptance:\n  same_topology: {loss_max_abs: 0.0, norm_max_abs: 0.0}\n"
        ),
        destination / "test_registration.py": _LICENSE
        + '"""Registration smoke test for the generated model adapter."""\n\n'
        + "import unittest\n\n"
        + "from hyper_parallel.models.registry import get_model_adapter\n\n\n"
        + "from .adapter import registration  # noqa: F401\n\n\n"
        + "class TestRegistration(unittest.TestCase):\n"
        + '    """Verify model-family lookup after registration import."""\n\n'
        + "    def test_registration(self) -> None:\n"
        + '        """Require the generated adapter to be discoverable."""\n'
        + f"        self.assertIsNotNone(get_model_adapter({family!r}))\n",
    }
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
    store = _store(manifest)
    store.write_json(
        "scaffold.json",
        {"files": tuple(str(path.relative_to(destination)) for path in files)},
    )
    print(destination)
    return 0


def command_check(args: argparse.Namespace) -> int:
    """Run strict final-tree/materialization/FSDP ownership diagnostics."""
    manifest = _manifest_from_args(args)
    store = _store(manifest)
    try:
        model = _build_declared_model(manifest, "model")
    except ManifestError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        match = re.search(r"\[(HP-[A-Z]+-[0-9]{3})\]", str(error))
        code = match.group(1) if match is not None else "HP-BUILD-001"
        phase = "D1" if code.startswith("HP-REPLACE") else "D2"
        finding = ModelIntegrationFinding(
            code=code,
            phase=phase,
            owner_fqn="<model_builder>",
            severity=FindingSeverity.ERROR,
            message="final model construction failed before model integration could inspect the tree",
            facts={"error_type": type(error).__name__, "error": str(error)},
            why_unsafe="a partially built or partially planned model cannot enter training",
            remediation="follow the coded replacement/planner error and rebuild the final model",
            related_config=("model.builder", "plan_overrides"),
        )
        report = ModelIntegrationReport(findings=[finding])
        store.write_json("check/findings.json", report.to_dict())
        _ensure_discovered(store)
        store.update_state(
            IntegrationState.FAILED,
            ("check/findings.json",),
            reason="model construction failed",
        )
        print(finding.format_human(), file=sys.stderr)
        return 1
    report, ownership, fsdp_units = validate_final_model(
        model,
        context={"manifest": manifest.to_dict()},
    )
    store.write_json("check/findings.json", report.to_dict())
    store.write_json("check/fsdp_units.json", fsdp_units)
    store.write_json(
        "checkpoint/coverage.json",
        report.metadata.get("checkpoint_coverage", {}),
    )
    store.write_csv(
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
    if report.errors:
        store.update_state(
            IntegrationState.FAILED,
            ("check/findings.json", "check/parameter_ownership.csv"),
            reason="structure model integration reported ERROR findings",
        )
        print("\n\n".join(finding.format_human() for finding in report.errors), file=sys.stderr)
        return 1
    _ensure_discovered(store)
    store.update_state(
        IntegrationState.STRUCTURE_VALIDATED,
        ("check/findings.json", "check/parameter_ownership.csv", "check/fsdp_units.json"),
    )
    print(store.root)
    return 0


def _validation_spec(manifest: ValidationManifest) -> ModelValidationSpec:
    adapter = get_model_adapter(manifest.family)
    if adapter is None:
        raise ManifestError(f"no model adapter registered for {manifest.family!r}")
    provider = adapter.validation
    if provider is None or not callable(provider):
        raise ManifestError(
            f"model adapter {manifest.family!r} does not declare a validation provider"
        )
    validation_spec = provider()
    if not isinstance(validation_spec, ModelValidationSpec):
        raise ManifestError("adapter validation provider returned the wrong type")
    return validation_spec


def _recompute_policy(manifest: ValidationManifest) -> Optional[RecomputePolicy]:
    """Resolve normal-training recompute policy for precision case generation."""
    adapter = get_model_adapter(manifest.family)
    provider = None if adapter is None else adapter.recompute
    if provider is None:
        return None
    recompute_policy = provider()
    if not isinstance(recompute_policy, RecomputePolicy):
        raise ManifestError("adapter recompute provider returned the wrong type")
    return recompute_policy


def _parameter_tolerance(
        acceptance: Mapping[str, Any],
        profile: str,
) -> dict[str, Any]:
    """Resolve strict base parameter limits plus an optional named override."""
    parameters = acceptance.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ManifestError("acceptance.parameters must be a mapping")
    profile_names = ("module_parity", "same_topology", "cross_topology")
    tolerance = {
        name: value
        for name, value in parameters.items()
        if name not in profile_names
    }
    override = parameters.get(profile)
    if override is None:
        return tolerance
    if not isinstance(override, Mapping):
        raise ManifestError(f"acceptance.parameters.{profile} must be a mapping")
    override = dict(override)
    if "summary" in override:
        base_summary = tolerance.get("summary", {})
        override_summary = override["summary"]
        if not isinstance(base_summary, Mapping) or not isinstance(override_summary, Mapping):
            raise ManifestError(
                f"acceptance.parameters.{profile}.summary must be a mapping"
            )
        override["summary"] = {**base_summary, **override_summary}
    tolerance.update(override)
    return tolerance


def command_parity(args: argparse.Namespace) -> int:
    """Run final-module authoritative parity cases from the model provider."""
    manifest = _manifest_from_args(args)
    store = _store(manifest)
    _require_state(store, IntegrationState.STRUCTURE_VALIDATED, "parity")
    validation_spec = _validation_spec(manifest)
    tolerance = _parameter_tolerance(manifest.acceptance, "module_parity")
    context = ParityContext(
        manifest=manifest,
        dtype=args.dtype,
        device=args.device,
        evidence_store=store,
        options={"tolerance": tolerance},
    )
    report = run_module_parity(validation_spec, context)
    if report["status"] == "PASS":
        store.write_yaml(
            "integration_handoff.yaml",
            {
                "schema_version": SCHEMA_VERSION,
                "family": manifest.family,
                "manifest": "manifest.resolved.yaml",
                "structure_findings": "check/findings.json",
                "module_parity": "module_parity/comparison.json",
                "checkpoint_coverage": "checkpoint/coverage.json",
                "status": "PASS",
            },
        )
        store.update_state(
            IntegrationState.MODULE_PARITY_PASSED,
            ("module_parity/comparison.json", "integration_handoff.yaml"),
        )
        return 0
    terminal = IntegrationState.FAILED if report["status"] == "FAIL" else IntegrationState.BLOCKED
    store.update_state(terminal, ("module_parity/comparison.json",), reason=report["status"])
    return 1 if report["status"] == "FAIL" else 2


def _trainer_launch_argv(
    manifest: ValidationManifest,
    case: Any,
    phase: str,
    evidence_dir: Path,
    resume_dir: Path,
    topology: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Build the standard Trainer command from one concise launcher declaration."""
    launcher = manifest.launcher
    module = launcher["module"]
    config = manifest.launcher_config_path
    if config is None:
        raise ManifestError("launcher.config is required for the standard Trainer launcher")
    topology = case.topology if topology is None else topology
    steps = int(manifest.matrix["steps"])
    split_step = int(manifest.matrix.get("resume_split_step", steps))
    if phase == "initialize":
        train_iters = _SHARED_INITIAL_CHECKPOINT_STEP
    elif phase == "prepare":
        train_iters = split_step
    else:
        train_iters = steps
    tokens = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={manifest.matrix['devices']}",
        "--module",
        module,
        str(config),
    ]
    controlled_overrides = {
        "training.train_iters": train_iters,
        # Split-run preparation stops early but must retain the full-run LR
        # curve so its checkpoint is numerically comparable with the baseline.
        "training.lr_scheduler_iters": steps,
        "accelerator.tp_size": topology.get("tp", 1),
        "accelerator.cp_size": topology.get("cp", 1),
        "accelerator.ep_size": topology.get("ep", 1),
        "accelerator.sequence_parallel": topology.get("sequence_parallel", False),
        "fsdp_config.dp_shard_size": topology.get("fsdp", 1),
        "activation_checkpoint.mode": "selective",
        "activation_checkpoint.selection": {
            "source": "model_adapter_safe_regions",
            **topology.get("recompute", {"layer_count": 0}),
        },
        "model_integration.mode": "runtime",
    }
    if "validate_placement" in topology:
        controlled_overrides["model.validate_placement"] = topology["validate_placement"]
    shared_initial_checkpoint = bool(manifest.matrix.get("shared_initial_checkpoint"))
    checkpoint_enabled = case.name == "baseline" and bool(
        manifest.matrix.get("same_topology_resume")
        or manifest.matrix.get("cross_topology_resume")
    )
    initial_checkpoint_dir = (
        manifest.output_dir / "cases" / "baseline" / "initial_checkpoint"
    )
    if shared_initial_checkpoint and phase not in ("initialize", "restore"):
        controlled_overrides["checkpoint.restore_from"] = str(
            initial_checkpoint_dir
            / f"{STEP_PREFIX}{_SHARED_INITIAL_CHECKPOINT_STEP}"
        )
        # The warm-start step exists to align model/optimizer/RNG state. Every
        # measured case must replay from the same configured data start; a
        # stateful Online cursor may also be impossible to restore after TP/CP
        # changes the DP world size. True resume phases keep the default and
        # restore the cursor saved at resume_split_step.
        controlled_overrides["checkpoint.restore_dataloader_state"] = False
    if phase == "initialize":
        controlled_overrides.update(
            {
                "checkpoint.save_ckpt": True,
                "checkpoint.save_steps": 1,
                "checkpoint.save_epochs": 0,
                "checkpoint.checkpoint_dir": str(initial_checkpoint_dir),
            }
        )
    elif phase == "prepare":
        controlled_overrides.update(
            {
                "checkpoint.save_ckpt": True,
                "checkpoint.save_steps": split_step,
                "checkpoint.save_epochs": 0,
                "checkpoint.checkpoint_dir": str(resume_dir),
            }
        )
    elif phase == "restore":
        controlled_overrides.update(
            {
                "checkpoint.save_ckpt": False,
                "checkpoint.restore_from": str(resume_dir / f"{STEP_PREFIX}{split_step}"),
                "checkpoint.checkpoint_dir": str(resume_dir),
            }
        )
    else:
        controlled_overrides["checkpoint.save_ckpt"] = checkpoint_enabled
        if checkpoint_enabled:
            controlled_overrides.update(
                {
                    "checkpoint.save_steps": split_step,
                    "checkpoint.save_epochs": 0,
                    "checkpoint.checkpoint_dir": str(evidence_dir / "dcp"),
                }
            )
    tokens.extend(
        f"--{name}={value}"
        for name, value in controlled_overrides.items()
    )
    return tokens


def _resolve_case_launches(
    manifest: ValidationManifest,
    case: Any,
    case_dir: Path,
) -> list[tuple[str, list[str], Path]]:
    """Resolve train or prepare/restore commands without executing them."""
    resume_dir = case_dir / "resume_checkpoint"
    if case.kind != "resume":
        launches = []
        if case.name == "baseline" and manifest.matrix.get("shared_initial_checkpoint"):
            initialize_dir = case_dir / "initialize"
            launches.append(
                (
                    "initialize",
                    _trainer_launch_argv(
                        manifest,
                        case,
                        "initialize",
                        initialize_dir,
                        resume_dir,
                    ),
                    initialize_dir,
                )
            )
        launches.append(
            (
                "train",
                _trainer_launch_argv(
                    manifest, case, "train", case_dir, resume_dir
                ),
                case_dir,
            )
        )
        return launches
    prepare_dir = case_dir / "prepare"
    prepare_topology = case.metadata.get("prepare_topology", case.topology)
    return [
        (
            "prepare",
            _trainer_launch_argv(
                manifest,
                case,
                "prepare",
                prepare_dir,
                resume_dir,
                topology=prepare_topology,
            ),
            prepare_dir,
        ),
        (
            "restore",
            _trainer_launch_argv(
                manifest, case, "restore", case_dir, resume_dir
            ),
            case_dir,
        ),
    ]


def _case_environment(
    case: Any,
    phase: str,
    evidence_dir: Path,
    resume_dir: Path,
    *,
    port_slot: int = 0,
    port_stride: int = 1,
) -> dict[str, str]:
    """Add case identity and isolate sequential distributed launch ports."""
    environment = os.environ.copy()
    environment.update(
        {
            "HP_MODEL_INTEGRATION_CASE": case.name,
            "HP_MODEL_INTEGRATION_KIND": case.kind,
            "HP_MODEL_INTEGRATION_PHASE": phase,
            "HP_MODEL_INTEGRATION_COMPARE_TO": case.compare_to,
            "HP_MODEL_INTEGRATION_EVIDENCE_DIR": str(evidence_dir),
            "HP_MODEL_INTEGRATION_RESUME_DIR": str(resume_dir),
            "HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR": str(evidence_dir),
        }
    )
    hccl_base_port = environment.get("HCCL_IF_BASE_PORT")
    if hccl_base_port is not None:
        try:
            isolated_port = int(hccl_base_port) + port_slot * port_stride
        except ValueError as error:
            raise ManifestError(
                "HCCL_IF_BASE_PORT must be an integer when model-integration "
                f"validation isolates case ports, got {hccl_base_port!r}"
            ) from error
        if not 0 < isolated_port <= 65535:
            raise ManifestError(
                "model-integration HCCL port isolation exceeds the valid port "
                f"range: base={hccl_base_port}, slot={port_slot}, "
                f"stride={port_stride}"
            )
        environment["HCCL_IF_BASE_PORT"] = str(isolated_port)
    return environment


def _execute_case_launch(
    manifest: ValidationManifest,
    case: Any,
    launch: Mapping[str, Any],
    resume_dir: Path,
) -> tuple[Optional[int], Optional[str]]:
    """Execute one resolved process and persist its stdout and stderr."""
    phase = launch["phase"]
    evidence_dir = Path(launch["evidence_dir"])
    evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            launch["argv"],
            check=False,
            capture_output=True,
            text=True,
            env=_case_environment(
                case,
                phase,
                evidence_dir,
                resume_dir,
                port_slot=int(launch.get("port_slot", 0)),
                port_stride=max(32, int(manifest.matrix.get("devices", 1))),
            ),
            cwd=_REPOSITORY_ROOT,
            timeout=manifest.launcher.get("timeout_seconds"),
        )
    except OSError as error:
        return None, f"{phase} launcher could not start: {error}"
    except subprocess.TimeoutExpired as error:
        evidence_dir.joinpath("stdout.log").write_text(error.stdout or "", encoding="utf-8")
        evidence_dir.joinpath("stderr.log").write_text(error.stderr or "", encoding="utf-8")
        timeout = manifest.launcher.get("timeout_seconds")
        return None, f"{phase} launcher exceeded timeout_seconds={timeout}"
    evidence_dir.joinpath("stdout.log").write_text(result.stdout, encoding="utf-8")
    evidence_dir.joinpath("stderr.log").write_text(result.stderr, encoding="utf-8")
    return result.returncode, None


def _run_validation_case(
    manifest: ValidationManifest,
    store: EvidenceStore,
    case: Any,
    case_index: int,
) -> dict[str, Any]:
    case_dir = store.path(f"cases/{case.name}")
    case_dir.mkdir(parents=True, exist_ok=True)
    resume_dir = case_dir / "resume_checkpoint"
    launches = _resolve_case_launches(manifest, case, case_dir)
    commands = [
        {
            "phase": phase,
            "argv": phase_argv,
            "evidence_dir": str(phase_evidence_dir),
            "port_slot": case_index * 2 + launch_index,
        }
        for launch_index, (phase, phase_argv, phase_evidence_dir) in enumerate(launches)
    ]
    store.write_json(
        f"cases/{case.name}/preflight.json",
        {"case": case.to_dict(), "launches": commands},
    )
    returncodes = []
    for launch in commands:
        returncode, blocked_reason = _execute_case_launch(
            manifest,
            case,
            launch,
            resume_dir,
        )
        if blocked_reason is not None:
            return {"case": case.name, "status": "BLOCKED", "reason": blocked_reason}
        returncodes.append(returncode)
        if returncode:
            break
    return {
        "case": case.name,
        "status": "PASS" if not any(returncodes) else "FAIL",
        "returncodes": returncodes,
        "metrics": f"cases/{case.name}/metrics.jsonl",
    }


def _case_artifact(case_dir: Path, relative_path: str) -> Path:
    """Resolve a launcher-owned artifact or the standard Trainer model integration location."""
    direct = case_dir / relative_path
    if direct.exists():
        return direct
    runtime = case_dir / "cases" / "runtime" / relative_path
    return runtime


def _performance_summary(case_dir: Path, warmup_steps: int) -> dict[str, Any]:
    """Summarize rank-zero timing and maximum allocator usage over all ranks."""
    runtime_dir = case_dir / "cases" / "runtime"
    rank_files = tuple(sorted(runtime_dir.glob("performance_rank*.jsonl")))
    memory_rows = [row for path in rank_files for row in load_jsonl(path)]
    return summarize_performance(
        load_jsonl(_case_artifact(case_dir, "performance.jsonl")),
        warmup_steps=warmup_steps,
        memory_rows=memory_rows,
    )


def _aggregate_statuses(statuses: Iterable[str]) -> str:
    """Reduce phase statuses using FAIL > BLOCKED > PASS precedence."""
    statuses = tuple(statuses)
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    return "PASS"


def _compare_validation_case(
    manifest: ValidationManifest,
    case: Any,
    baseline_dir: Path,
    candidate_dir: Path,
    launch_statuses: tuple[str, str],
) -> dict[str, Any]:
    """Compare one candidate against the baseline using declared contracts."""
    try:
        tolerance_name = case.metadata["acceptance"]
    except (AttributeError, KeyError) as error:
        raise ManifestError(
            f"validation case {case.name!r} has no acceptance classification; "
            "regenerate the case matrix with generate_validation_cases()"
        ) from error
    preflight = compare_preflight_configs(
        baseline_dir,
        candidate_dir,
    )
    scalar = compare_scalar_curves(
        load_jsonl(_case_artifact(baseline_dir, "metrics.jsonl")),
        load_jsonl(_case_artifact(candidate_dir, "metrics.jsonl")),
        manifest.acceptance.get(tolerance_name, {}),
        allow_candidate_subset=case.kind == "resume",
    )
    parameter = compare_parameter_probes(
        baseline_dir,
        candidate_dir,
        _parameter_tolerance(manifest.acceptance, tolerance_name),
        allow_candidate_subset=case.kind == "resume",
        compare_optimizer_states=tolerance_name == "same_topology",
    )
    checkpoint_layout = None
    statuses = (*launch_statuses, preflight["status"], scalar["status"], parameter["status"])
    if case.kind == "resume":
        checkpoint_layout = compare_checkpoint_layouts(
            baseline_dir,
            candidate_dir,
            same_topology=case.metadata.get("resume") == "same_topology",
        )
        statuses = (*statuses, checkpoint_layout["status"])
    return {
        "status": _aggregate_statuses(statuses),
        "preflight": preflight,
        "scalar": scalar,
        "parameter_probes": parameter,
        "checkpoint_layout": checkpoint_layout,
    }


def _finalize_validation(
    store: EvidenceStore,
    results: list[dict[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    performance: Mapping[str, Mapping[str, Any]],
) -> int:
    """Persist the matrix result and advance the evidence state machine."""
    status = _aggregate_statuses(
        [result["status"] for result in results]
        + [comparison["status"] for comparison in comparisons.values()]
    )
    store.write_json(
        "comparison.json",
        {
            "status": status,
            "cases": results,
            "comparisons": comparisons,
            "performance": performance,
        },
    )
    if status == "PASS":
        store.advance_if_before(IntegrationState.MATRIX_PASSED, ("comparison.json",))
        return 0
    terminal = IntegrationState.FAILED if status == "FAIL" else IntegrationState.BLOCKED
    store.update_state(terminal, ("comparison.json",), reason=status)
    return 1 if status == "FAIL" else 2


def command_validate(args: argparse.Namespace) -> int:
    """Generate and optionally execute the minimum precision matrix."""
    manifest = _manifest_from_args(args)
    store = _store(manifest)
    _require_precision_handoff(store, manifest)
    validation_spec = _validation_spec(manifest)
    try:
        cases = generate_validation_cases(
            manifest.matrix,
            _recompute_policy(manifest),
            validation_spec.topology_constraints,
        )
    except (TypeError, ValueError) as error:
        raise ManifestError(f"invalid validation matrix: {error}") from error
    store.write_json("cases/resolved_cases.json", [case.to_dict() for case in cases])
    if args.generate_only:
        print(store.path("cases/resolved_cases.json"))
        return 0
    results = [
        _run_validation_case(manifest, store, case, case_index)
        for case_index, case in enumerate(cases)
    ]
    results_by_name = {result["case"]: result for result in results}
    baseline_dir = store.path("cases/baseline")
    warmup_steps = int(manifest.matrix.get("performance_warmup_steps", 1))
    comparisons = {}
    performance = {"baseline": _performance_summary(baseline_dir, warmup_steps)}
    for case in cases:
        if case.name == "baseline":
            continue
        candidate_dir = store.path(f"cases/{case.name}")
        launch_statuses = (
            results_by_name["baseline"]["status"],
            results_by_name[case.name]["status"],
        )
        comparisons[case.name] = _compare_validation_case(
            manifest,
            case,
            baseline_dir,
            candidate_dir,
            launch_statuses,
        )
        performance[case.name] = _performance_summary(candidate_dir, warmup_steps)
    return _finalize_validation(store, results, comparisons, performance)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    """Read an optional JSON evidence file."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _report_entries(
    findings: Optional[Mapping[str, Any]],
    parity: Optional[Mapping[str, Any]],
    comparison: Optional[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Render compact structure, parity, precision, and performance entries."""
    structure_entries = ["not executed"]
    if findings is not None:
        structure_entries = [
            f"{finding['code']} {finding['severity']}: {finding['message']} "
            f"(fix: {finding.get('remediation') or 'n/a'})"
            for finding in findings.get("findings", ())
        ] or ["PASS: no ERROR findings"]
    parity_entries = ["not executed"]
    if parity is not None:
        parity_entries = [
            f"{case.get('case', '<unnamed>')}: {case.get('status')}"
            + (f" ({case.get('reason')})" if case.get("reason") else "")
            for case in parity.get("cases", ())
        ] or [f"{parity.get('status')}: no executed cases"]
    precision_entries = ["not executed"]
    performance_entries = ["not executed"]
    if comparison is not None:
        precision_entries = [
            f"{name}: {value.get('status')}"
            for name, value in comparison.get("comparisons", {}).items()
        ] or [f"{comparison.get('status')}: baseline only"]
        performance_entries = [
            f"{name}: step_p50={value.get('step_time_p50_seconds')}, "
            f"tokens/s={value.get('tokens_per_second_p50')}, "
            f"peak_allocated={value.get('peak_memory_allocated_bytes')} bytes"
            for name, value in comparison.get("performance", {}).items()
        ] or ["not measured"]
    return structure_entries, parity_entries, precision_entries, performance_entries


def _resolved_report_cases(store: EvidenceStore) -> list[dict[str, Any]]:
    """Load resolved matrix cases for report rendering."""
    cases_path = store.path("cases/resolved_cases.json")
    if not cases_path.is_file():
        return []
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _report_manifest(store: EvidenceStore) -> dict[str, Any]:
    """Load the resolved manifest without requiring the original input path."""
    manifest_path = store.path("manifest.resolved.yaml")
    if not manifest_path.is_file():
        return {}
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _case_category(case: Mapping[str, Any]) -> str:
    """Classify one case without inflating strategy coverage with mechanics."""
    name = str(case.get("name", ""))
    if name == "baseline":
        return "baseline"
    if name.startswith(("axis-", "combined-")):
        return "strategy"
    if name.startswith("recompute_"):
        return "recompute"
    if case.get("kind") == "resume":
        return "resume"
    if case.get("kind") == "production_validate":
        return "production_validate"
    return "other"


def _coverage_counts(resolved_cases: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count independent evidence categories."""
    counts = {
        "baseline": 0,
        "strategy": 0,
        "recompute": 0,
        "resume": 0,
        "production_validate": 0,
        "other": 0,
    }
    for case in resolved_cases:
        counts[_case_category(case)] += 1
    return counts


def _coverage_notes(manifest: Mapping[str, Any], language: str) -> list[str]:
    """Return localized exclusions, falling back to the canonical notes."""
    matrix = manifest.get("matrix", {})
    if not isinstance(matrix, Mapping):
        return []
    key = "coverage_notes_zh" if language == "zh" else "coverage_notes"
    notes = matrix.get(key, matrix.get("coverage_notes", ()))
    if not isinstance(notes, (list, tuple)):
        return []
    return [note for note in notes if isinstance(note, str)]


def _precision_coverage_entries(store: EvidenceStore, language: str = "en") -> list[str]:
    """Summarize real matrix coverage separately from resume mechanics."""
    resolved_cases = _resolved_report_cases(store)
    counts = _coverage_counts(resolved_cases)
    if language == "zh":
        entries = [
            f"基线={counts['baseline']}，策略泛化={counts['strategy']}，"
            f"重计算={counts['recompute']}，断点续训={counts['resume']}，"
            f"生产/Validate 对照={counts['production_validate']}"
        ]
    else:
        entries = [
            f"baseline={counts['baseline']}, strategy_generalization={counts['strategy']}, "
            f"recompute={counts['recompute']}, resume={counts['resume']}, "
            f"production_validate={counts['production_validate']}"
        ]

    manifest = _report_manifest(store)
    exclusion_prefix = "排除项" if language == "zh" else "excluded"
    entries.extend(
        f"{exclusion_prefix}: {note}"
        for note in _coverage_notes(manifest, language)
    )
    return entries


def _format_report_number(value: Any) -> str:
    """Format optional numerical evidence compactly and deterministically."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_topology(topology: Mapping[str, Any]) -> str:
    """Render the stable topology dimensions in one line."""
    keys = ("tp", "cp", "ep", "fsdp", "sequence_parallel", "recompute")
    return ", ".join(f"{key}={topology.get(key)}" for key in keys)


def _scalar_case_summary(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse per-step scalar evidence while preserving identity gates."""
    steps = comparison.get("scalar", {}).get("steps", ())

    def maximum(name: str) -> Optional[float]:
        """Return the largest observed value without converting absent evidence."""
        values = [float(row[name]) for row in steps if row.get(name) is not None]
        return max(values, default=None)

    return {
        "steps": len(steps),
        "max_loss": maximum("loss_max_abs"),
        "max_loss_rel": maximum("loss_max_rel"),
        "max_norm": maximum("norm_max_abs"),
        "max_norm_rel": maximum("norm_max_rel"),
        "max_post_clip_norm": maximum("post_clip_norm_max_abs"),
        "max_post_clip_norm_rel": maximum("post_clip_norm_max_rel"),
        "input_identity": all(bool(row.get("input_identity")) for row in steps) if steps else False,
        "learning_rate_identity": (
            all(bool(row.get("learning_rate_equal")) for row in steps) if steps else False
        ),
        "finite": all(bool(row.get("finite")) for row in steps) if steps else False,
    }


def _parameter_acceptance_entry(
        tolerance: Mapping[str, Any],
        label: str,
        language: str,
) -> str:
    """Render exact-value and summary-only parameter limits without conflating them."""
    summary = tolerance.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    if language == "zh":
        return (
            f"{label}：完整值 max_abs <= {_format_report_number(tolerance.get('max_abs'))}，"
            f"relative_l2 <= {_format_report_number(tolerance.get('relative_l2'))}，"
            f"组合规则={tolerance.get('combination', 'all')}；仅摘要不声明逐元素误差，"
            "L2 范数相对误差 <= "
            f"{_format_report_number(summary.get('l2_norm_relative'))}"
        )
    return (
        f"{label}: exact-value max_abs <= {_format_report_number(tolerance.get('max_abs'))}, "
        f"relative_l2 <= {_format_report_number(tolerance.get('relative_l2'))}, "
        f"combination={tolerance.get('combination', 'all')}; summary-only probes make no "
        "elementwise claim, L2-norm relative error <= "
        f"{_format_report_number(summary.get('l2_norm_relative'))}"
    )


def _acceptance_entries(manifest: Mapping[str, Any], language: str) -> list[str]:
    """Render every scalar and parameter threshold used by the matrix."""
    acceptance = manifest.get("acceptance", {})
    if not isinstance(acceptance, Mapping):
        acceptance = {}
    same = acceptance.get("same_topology", {})
    cross = acceptance.get("cross_topology", {})
    parity_parameters = _parameter_tolerance(acceptance, "module_parity")
    same_parameters = _parameter_tolerance(acceptance, "same_topology")
    cross_parameters = _parameter_tolerance(acceptance, "cross_topology")
    if language == "zh":
        return [
            "同拓扑："
            f"loss 最大绝对误差 <= {_format_report_number(same.get('loss_max_abs'))}，"
            f"loss 最大相对误差 <= {_format_report_number(same.get('loss_max_rel'))}，"
            f"梯度范数最大绝对误差 <= {_format_report_number(same.get('norm_max_abs'))}，"
            f"梯度范数最大相对误差 <= {_format_report_number(same.get('norm_max_rel'))}，"
            f"组合规则={same.get('combination', 'all')}",
            "跨拓扑："
            f"loss 最大绝对误差 <= {_format_report_number(cross.get('loss_max_abs'))}，"
            f"loss 最大相对误差 <= {_format_report_number(cross.get('loss_max_rel'))}，"
            f"梯度范数最大绝对误差 <= {_format_report_number(cross.get('norm_max_abs'))}，"
            f"梯度范数最大相对误差 <= {_format_report_number(cross.get('norm_max_rel'))}，"
            f"组合规则={cross.get('combination', 'all')}",
            _parameter_acceptance_entry(parity_parameters, "模块对拍参数", "zh"),
            _parameter_acceptance_entry(same_parameters, "同拓扑参数", "zh"),
            _parameter_acceptance_entry(cross_parameters, "跨拓扑参数", "zh"),
        ]
    return [
        "same topology: "
        f"loss max abs <= {_format_report_number(same.get('loss_max_abs'))}, "
        f"loss max rel <= {_format_report_number(same.get('loss_max_rel'))}, "
        f"gradient-norm max abs <= {_format_report_number(same.get('norm_max_abs'))}, "
        f"gradient-norm max rel <= {_format_report_number(same.get('norm_max_rel'))}, "
        f"combination={same.get('combination', 'all')}",
        "cross topology: "
        f"loss max abs <= {_format_report_number(cross.get('loss_max_abs'))}, "
        f"loss max rel <= {_format_report_number(cross.get('loss_max_rel'))}, "
        f"gradient-norm max abs <= {_format_report_number(cross.get('norm_max_abs'))}, "
        f"gradient-norm max rel <= {_format_report_number(cross.get('norm_max_rel'))}, "
        f"combination={cross.get('combination', 'all')}",
        _parameter_acceptance_entry(parity_parameters, "module parity parameters", "en"),
        _parameter_acceptance_entry(same_parameters, "same-topology parameters", "en"),
        _parameter_acceptance_entry(cross_parameters, "cross-topology parameters", "en"),
    ]


def _precision_detail_entries(
    resolved_cases: Iterable[Mapping[str, Any]],
    comparison: Optional[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    language: str,
) -> list[str]:
    """Render topology, thresholds, scalar gates, probes, and DCP per case."""
    comparisons = {} if comparison is None else comparison.get("comparisons", {})
    acceptance = manifest.get("acceptance", {})
    matrix = manifest.get("matrix", {})
    default_steps = matrix.get("steps", "n/a") if isinstance(matrix, Mapping) else "n/a"
    entries = []
    category_names = {
        "en": {
            "baseline": "baseline",
            "strategy": "strategy generalization",
            "recompute": "recompute",
            "resume": "resume",
            "production_validate": "Production/Validate",
            "other": "other",
        },
        "zh": {
            "baseline": "基线",
            "strategy": "策略泛化",
            "recompute": "重计算",
            "resume": "断点续训",
            "production_validate": "生产/Validate 对照",
            "other": "其他",
        },
    }
    for case in resolved_cases:
        name = str(case.get("name", "<unnamed>"))
        category = category_names[language][_case_category(case)]
        topology = _format_topology(case.get("topology", {}))
        if name == "baseline":
            if language == "zh":
                entries.append(
                    f"[{category}] `{name}`：{topology}；正式步数={default_steps}；作为参考曲线。"
                )
            else:
                entries.append(
                    f"[{category}] `{name}`: {topology}; measured_steps={default_steps}; reference curve."
                )
            continue

        case_comparison = comparisons.get(name, {})
        scalar = _scalar_case_summary(case_comparison)
        metadata = case.get("metadata", {})
        profile = metadata.get("acceptance", "unknown")
        threshold = acceptance.get(profile, {}) if isinstance(acceptance, Mapping) else {}
        parameter = case_comparison.get("parameter_probes", {})
        checkpoint = case_comparison.get("checkpoint_layout")
        if isinstance(checkpoint, Mapping):
            checkpoint_status = checkpoint.get("status", "n/a")
        elif case.get("kind") != "resume":
            checkpoint_status = (
                "not applicable (non-resume case)"
                if language == "en"
                else "不适用（非断点续训用例）"
            )
        else:
            checkpoint_status = "not recorded" if language == "en" else "未记录"
        identity_status = "PASS" if scalar["input_identity"] else "FAIL"
        lr_status = "PASS" if scalar["learning_rate_identity"] else "FAIL"
        finite_status = "PASS" if scalar["finite"] else "FAIL"
        if language == "zh":
            entries.append(
                f"[{category}] `{name}`：{topology}；阈值档={profile} "
                f"(loss<={_format_report_number(threshold.get('loss_max_abs'))}, "
                f"loss_rel<={_format_report_number(threshold.get('loss_max_rel'))}, "
                f"norm<={_format_report_number(threshold.get('norm_max_abs'))}, "
                f"norm_rel<={_format_report_number(threshold.get('norm_max_rel'))}, "
                f"组合={threshold.get('combination', 'all')})；"
                f"对比步数={scalar['steps']}；max_loss_abs={_format_report_number(scalar['max_loss'])}；"
                f"max_loss_rel={_format_report_number(scalar['max_loss_rel'])}；"
                f"max_norm_abs={_format_report_number(scalar['max_norm'])}；"
                f"max_norm_rel={_format_report_number(scalar['max_norm_rel'])}；"
                f"max_post_clip_norm_abs={_format_report_number(scalar['max_post_clip_norm'])}；"
                f"max_post_clip_norm_rel={_format_report_number(scalar['max_post_clip_norm_rel'])}；"
                f"输入身份={identity_status}；LR 身份={lr_status}；有限性={finite_status}；"
                f"参数探针={parameter.get('status', 'n/a')} "
                f"(optimizer 数值={parameter.get('optimizer_state_numeric_comparison', 'n/a')}, "
                f"跳过状态数={parameter.get('skipped_optimizer_states', 0)})；"
                f"checkpoint layout={checkpoint_status}；总状态={case_comparison.get('status', 'n/a')}。"
            )
        else:
            entries.append(
                f"[{category}] `{name}`: {topology}; tolerance_profile={profile} "
                f"(loss<={_format_report_number(threshold.get('loss_max_abs'))}, "
                f"loss_rel<={_format_report_number(threshold.get('loss_max_rel'))}, "
                f"norm<={_format_report_number(threshold.get('norm_max_abs'))}, "
                f"norm_rel<={_format_report_number(threshold.get('norm_max_rel'))}, "
                f"combination={threshold.get('combination', 'all')}); "
                f"compared_steps={scalar['steps']}; max_loss_abs={_format_report_number(scalar['max_loss'])}; "
                f"max_loss_rel={_format_report_number(scalar['max_loss_rel'])}; "
                f"max_norm_abs={_format_report_number(scalar['max_norm'])}; "
                f"max_norm_rel={_format_report_number(scalar['max_norm_rel'])}; "
                f"max_post_clip_norm_abs={_format_report_number(scalar['max_post_clip_norm'])}; "
                f"max_post_clip_norm_rel={_format_report_number(scalar['max_post_clip_norm_rel'])}; "
                f"input_identity={identity_status}; LR_identity={lr_status}; finite={finite_status}; "
                f"parameter_probes={parameter.get('status', 'n/a')} "
                f"(optimizer_numeric={parameter.get('optimizer_state_numeric_comparison', 'n/a')}, "
                f"skipped_states={parameter.get('skipped_optimizer_states', 0)}); "
                f"checkpoint_layout={checkpoint_status}; overall={case_comparison.get('status', 'n/a')}."
            )
    return entries or (["未执行"] if language == "zh" else ["not executed"])


def _worst_case_entries(
    comparison: Optional[Mapping[str, Any]],
    language: str,
) -> list[str]:
    """Identify the worst observed scalar deltas with case and step."""
    worst_loss = (-1.0, "n/a", "n/a")
    worst_norm = (-1.0, "n/a", "n/a")
    for name, case in ({} if comparison is None else comparison.get("comparisons", {})).items():
        for row in case.get("scalar", {}).get("steps", ()):
            loss = float(row.get("loss_max_abs", 0.0))
            norm = float(row.get("norm_max_abs", 0.0))
            if loss > worst_loss[0]:
                worst_loss = (loss, name, row.get("step"))
            if norm > worst_norm[0]:
                worst_norm = (norm, name, row.get("step"))
    if worst_loss[0] < 0:
        return ["未测量"] if language == "zh" else ["not measured"]
    if language == "zh":
        return [
            f"全矩阵最大 loss 绝对误差={_format_report_number(worst_loss[0])}，"
            f"用例=`{worst_loss[1]}`，step={worst_loss[2]}",
            f"全矩阵最大梯度范数绝对误差={_format_report_number(worst_norm[0])}，"
            f"用例=`{worst_norm[1]}`，step={worst_norm[2]}",
        ]
    return [
        f"matrix maximum loss absolute error={_format_report_number(worst_loss[0])}, "
        f"case=`{worst_loss[1]}`, step={worst_loss[2]}",
        f"matrix maximum gradient-norm absolute error={_format_report_number(worst_norm[0])}, "
        f"case=`{worst_norm[1]}`, step={worst_norm[2]}",
    ]


def _detailed_parity_entries(
    parity: Optional[Mapping[str, Any]],
    language: str,
) -> list[str]:
    """Render each native module parity observation and compatibility note."""
    if parity is None:
        return ["未执行"] if language == "zh" else ["not executed"]
    entries = []
    for case in parity.get("cases", ()):
        case_name = case.get("case", "<unnamed>")
        entries.append(
            f"`{case_name}`：{case.get('status')}。" if language == "zh"
            else f"`{case_name}`: {case.get('status')}."
        )
        metric_rows = case.get("metrics", ())
        for metric in metric_rows:
            maximum = metric.get("max_abs")
            relative_l2 = metric.get("relative_l2")
            if language == "zh":
                entries.append(
                    f"模块 `{metric.get('name', '<unnamed>')}`：{metric.get('status')}；"
                    f"最大绝对误差={_format_report_number(maximum)}；"
                    f"相对 L2 误差={_format_report_number(relative_l2)}。"
                )
            else:
                entries.append(
                    f"module `{metric.get('name', '<unnamed>')}`: {metric.get('status')}; "
                    f"maximum absolute error={_format_report_number(maximum)}; "
                    f"relative L2 error={_format_report_number(relative_l2)}."
                )
        for observation in case.get("observations", ()):
            metrics = observation.get("metrics")
            errors = observation.get("max_abs_errors", {})
            maximum = (
                metrics.get("max_abs")
                if isinstance(metrics, Mapping)
                else max((float(value) for value in errors.values()), default=None)
            )
            relative_l2 = metrics.get("relative_l2") if isinstance(metrics, Mapping) else None
            if language == "zh":
                detail = (
                    f"最大绝对误差={_format_report_number(maximum)}；"
                    f"相对 L2 误差={_format_report_number(relative_l2)}"
                    if maximum is not None
                    else f"精确一致性={observation.get('status')}"
                )
                entries.append(
                    f"模块 `{observation.get('name', '<unnamed>')}`：{observation.get('status')}；"
                    f"{detail}。"
                )
            else:
                detail = (
                    f"maximum absolute error={_format_report_number(maximum)}; "
                    f"relative L2 error={_format_report_number(relative_l2)}"
                    if maximum is not None
                    else f"exact equality={observation.get('status')}"
                )
                entries.append(
                    f"module `{observation.get('name', '<unnamed>')}`: {observation.get('status')}; "
                    f"{detail}."
                )
        for note in case.get("compatibility", ()):
            prefix = "兼容性说明（原文）" if language == "zh" else "compatibility note"
            entries.append(f"{prefix}: {note}")
    return entries or (["无已执行用例"] if language == "zh" else ["no executed cases"])


def _detailed_performance_entries(
    comparison: Optional[Mapping[str, Any]],
    language: str,
) -> list[str]:
    """Render warm performance and memory without making them precision gates."""
    performance = {} if comparison is None else comparison.get("performance", {})
    entries = []
    for name, value in performance.items():
        allocated = value.get("peak_memory_allocated_bytes")
        reserved = value.get("peak_memory_reserved_bytes")
        allocated_mib = None if allocated is None else float(allocated) / (1024 ** 2)
        reserved_mib = None if reserved is None else float(reserved) / (1024 ** 2)
        if language == "zh":
            entries.append(
                f"`{name}`：稳态步数={value.get('steady_steps', 'n/a')}；"
                f"step p50={_format_report_number(value.get('step_time_p50_seconds'))}s；"
                f"p90={_format_report_number(value.get('step_time_p90_seconds'))}s；"
                f"tokens/s p50={_format_report_number(value.get('tokens_per_second_p50'))}；"
                f"samples/s p50={_format_report_number(value.get('samples_per_second_p50'))}；"
                f"峰值 allocated={_format_report_number(allocated_mib)} MiB；"
                f"reserved={_format_report_number(reserved_mib)} MiB。"
            )
        else:
            entries.append(
                f"`{name}`: steady_steps={value.get('steady_steps', 'n/a')}; "
                f"step p50={_format_report_number(value.get('step_time_p50_seconds'))}s; "
                f"p90={_format_report_number(value.get('step_time_p90_seconds'))}s; "
                f"tokens/s p50={_format_report_number(value.get('tokens_per_second_p50'))}; "
                f"samples/s p50={_format_report_number(value.get('samples_per_second_p50'))}; "
                f"peak allocated={_format_report_number(allocated_mib)} MiB; "
                f"reserved={_format_report_number(reserved_mib)} MiB."
            )
    return entries or (["未测量"] if language == "zh" else ["not measured"])


def _reproducibility_entries(
    store: EvidenceStore,
    environment: Optional[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    language: str,
) -> list[str]:
    """Render revisions and import paths needed to replay the evidence."""
    environment = environment or {}
    torch_info = environment.get("torch") or {}
    transformers_info = environment.get("transformers") or {}
    framework_info = environment.get("hyper_parallel") or {}
    reference = manifest.get("reference", {})
    labels = {
        "en": ("resolved manifest", "environment", "repository revision", "framework import", "reference source"),
        "zh": ("已解析清单", "运行环境", "代码仓 revision", "框架导入路径", "参考源码"),
    }[language]
    return [
        f"{labels[0]}: `{store.path('manifest.resolved.yaml').name}`",
        f"{labels[1]}: `{store.path('environment.json').name}`",
        f"{labels[2]}: `{environment.get('git_revision', 'n/a')}`",
        f"{labels[3]}: `{framework_info.get('import_path', 'n/a')}`",
        f"PyTorch: `{torch_info.get('version', 'n/a')}`; Transformers: "
        f"`{transformers_info.get('version', 'n/a')}`",
        f"{labels[4]}: `{reference.get('source_path', 'n/a')}`",
    ]


def _structure_entries(
    findings: Optional[Mapping[str, Any]],
    language: str,
) -> list[str]:
    """Render structure status, inventory size, and actionable findings."""
    if findings is None:
        return ["未执行"] if language == "zh" else ["not executed"]
    metadata = findings.get("metadata", {})
    counts = metadata.get("inventory_counts", {})
    if language == "zh":
        entries = [
            f"状态={findings.get('status')}；model_type={metadata.get('model_type', 'n/a')}；"
            f"模块={counts.get('modules', 'n/a')}；参数={counts.get('parameters', 'n/a')}；"
            f"buffer={counts.get('buffers', 'n/a')}。"
        ]
    else:
        entries = [
            f"status={findings.get('status')}; model_type={metadata.get('model_type', 'n/a')}; "
            f"modules={counts.get('modules', 'n/a')}; parameters={counts.get('parameters', 'n/a')}; "
            f"buffers={counts.get('buffers', 'n/a')}."
        ]
    for finding in findings.get("findings", ()):
        entries.append(
            f"{finding.get('code')} {finding.get('severity')}: {finding.get('message')} "
            f"(fix: {finding.get('remediation') or 'n/a'})"
        )
    return entries


def _localized_value(value: Mapping[str, Any], key: str, language: str) -> Any:
    """Read a localized evidence field with an English fallback."""
    localized_key = f"{key}_zh" if language == "zh" else key
    return value.get(localized_key, value.get(key, "n/a"))


def _localized_sentence(value: Mapping[str, Any], key: str, language: str) -> str:
    """Return localized prose without duplicate terminal punctuation."""
    return str(_localized_value(value, key, language)).rstrip(".。")


def _verdict_entries(
    store: EvidenceStore,
    findings: Optional[Mapping[str, Any]],
    parity: Optional[Mapping[str, Any]],
    comparison: Optional[Mapping[str, Any]],
    checkpoint: Optional[Mapping[str, Any]],
    language: str,
) -> list[str]:
    """State exactly what the aggregate PASS covers and does not cover."""
    operator_audit = _read_json(store.path("analysis/operator_support.json")) or {}
    statuses = (
        (findings or {}).get("status", "NOT_EXECUTED"),
        (parity or {}).get("status", "NOT_EXECUTED"),
        (comparison or {}).get("status", "NOT_EXECUTED"),
    )
    strict_validate = operator_audit.get("strict_validate_status", "NOT_RECORDED")
    checkpoint_status = (checkpoint or {}).get("status", "NOT_EXECUTED")
    if language == "zh":
        return [
            f"已执行门禁：结构={statuses[0]}；原生模块对齐={statuses[1]}；"
            f"已声明精度矩阵={statuses[2]}。",
            "首行 PASS 只适用于 resolved cases 中实际执行的范围；未执行或明确排除的能力不会因该 PASS "
            f"自动视为已覆盖。严格 Validate={strict_validate}；预训练 checkpoint={checkpoint_status}。",
        ]
    return [
        f"executed gates: structure={statuses[0]}; native module parity={statuses[1]}; "
        f"declared precision matrix={statuses[2]}.",
        "The first-line PASS applies only to the executed resolved cases; omitted or explicitly excluded "
        f"capabilities are not covered by that PASS. strict Validate={strict_validate}; "
        f"pretrained checkpoint={checkpoint_status}.",
    ]


def _operator_support_entries(store: EvidenceStore, language: str) -> list[str]:
    """Render the exact-callable distributed-operator audit when recorded."""
    audit = _read_json(store.path("analysis/operator_support.json"))
    if audit is None:
        return ["未记录"] if language == "zh" else ["not recorded"]
    if language == "zh":
        entries = [
            f"审计状态={audit.get('status', 'n/a')}；"
            f"严格 Validate 状态={audit.get('strict_validate_status', 'n/a')}。"
        ]
        for operator in audit.get("operators", ()):
            consumer = _localized_sentence(operator, "consumer", language)
            reason = _localized_sentence(operator, "reason", language)
            entries.append(
                f"`{operator.get('callable', 'n/a')}`：状态={operator.get('status', 'n/a')}；"
                f"registry key=`{operator.get('registry_key', 'n/a')}`；"
                f"实现={operator.get('implementation', 'n/a')}；"
                f"注册={operator.get('registered', False)}；"
                f"使用方={consumer}；结论={reason}。"
            )
        return entries
    entries = [
        f"audit status={audit.get('status', 'n/a')}; "
        f"strict Validate status={audit.get('strict_validate_status', 'n/a')}."
    ]
    for operator in audit.get("operators", ()):
        consumer = _localized_sentence(operator, "consumer", language)
        reason = _localized_sentence(operator, "reason", language)
        entries.append(
            f"`{operator.get('callable', 'n/a')}`: status={operator.get('status', 'n/a')}; "
            f"registry_key=`{operator.get('registry_key', 'n/a')}`; "
            f"implementation={operator.get('implementation', 'n/a')}; "
            f"registered={operator.get('registered', False)}; "
            f"consumer={consumer}; finding={reason}."
        )
    return entries


def _parallel_ownership_entries(store: EvidenceStore, language: str) -> list[str]:
    """Render actual parameter placements instead of inferring them from launch size."""
    audit = _read_json(store.path("analysis/parallel_ownership.json"))
    if audit is None:
        return ["未记录"] if language == "zh" else ["not recorded"]
    if language == "zh":
        entries = [
            f"审计状态={audit.get('status', 'n/a')}；证据拓扑="
            f"{_format_topology(audit.get('topology', {}))}。"
        ]
        for subtree in audit.get("subtrees", ()):
            examples = "；".join(
                subtree.get("placement_evidence_zh", subtree.get("placement_evidence", ()))
            ) or "n/a"
            conclusion = _localized_sentence(subtree, "conclusion", language)
            entries.append(
                f"{_localized_value(subtree, 'name', language)}："
                f"TP 状态={subtree.get('tp_status', 'n/a')}；证据={examples}；"
                f"结论={conclusion}。"
            )
        conclusion = _localized_sentence(audit, "conclusion", language)
        entries.append(f"总体结论：{conclusion}。")
        return entries
    entries = [
        f"audit status={audit.get('status', 'n/a')}; evidence topology="
        f"{_format_topology(audit.get('topology', {}))}."
    ]
    for subtree in audit.get("subtrees", ()):
        examples = "; ".join(subtree.get("placement_evidence", ())) or "n/a"
        conclusion = _localized_sentence(subtree, "conclusion", language)
        entries.append(
            f"{_localized_value(subtree, 'name', language)}: "
            f"TP status={subtree.get('tp_status', 'n/a')}; evidence={examples}; "
            f"conclusion={conclusion}."
        )
    conclusion = _localized_sentence(audit, "conclusion", language)
    entries.append(f"overall conclusion: {conclusion}.")
    return entries


def _issue_ledger_entries(store: EvidenceStore, language: str) -> list[str]:
    """Render resolved and open integration issues with stable identifiers."""
    ledger = _read_json(store.path("analysis/issues.json"))
    if ledger is None:
        return ["未记录"] if language == "zh" else ["not recorded"]
    entries = []
    for issue in ledger.get("issues", ()):
        title = _localized_value(issue, "title", language)
        detail = _localized_value(issue, "detail", language)
        if language == "zh":
            entries.append(
                f"`{issue.get('id', 'n/a')}` [{issue.get('status', 'n/a')}/"
                f"{issue.get('severity', 'n/a')}] {title}：{detail}"
            )
        else:
            entries.append(
                f"`{issue.get('id', 'n/a')}` [{issue.get('status', 'n/a')}/"
                f"{issue.get('severity', 'n/a')}] {title}: {detail}"
            )
    return entries or (["无记录"] if language == "zh" else ["no entries"])


def _checkpoint_entries(
    checkpoint: Optional[Mapping[str, Any]],
    language: str,
) -> list[str]:
    """Explain whether checkpoint tensors were loaded or scratch-initialized."""
    if checkpoint is None:
        return ["未执行"] if language == "zh" else ["not executed"]
    if language == "zh":
        reason = checkpoint.get("reason", "n/a")
        if reason == "model was initialized from scratch or no load report was retained":
            reason = "模型从随机状态初始化，或没有保留权重加载报告"
        return [
            f"状态={checkpoint.get('status')}；loaded={checkpoint.get('loaded')}；"
            f"missing={checkpoint.get('missing')}；unexpected={checkpoint.get('unexpected')}；"
            f"原因={reason}。"
        ]
    return [
        f"status={checkpoint.get('status')}; loaded={checkpoint.get('loaded')}; "
        f"missing={checkpoint.get('missing')}; unexpected={checkpoint.get('unexpected')}; "
        f"reason={checkpoint.get('reason', 'n/a')}."
    ]


def _report_sections(
    store: EvidenceStore,
    findings: Optional[Mapping[str, Any]],
    parity: Optional[Mapping[str, Any]],
    comparison: Optional[Mapping[str, Any]],
    checkpoint: Optional[Mapping[str, Any]],
    environment: Optional[Mapping[str, Any]],
    language: str,
) -> list[tuple[str, list[str]]]:
    """Build the detailed English or Chinese report sections."""
    manifest = _report_manifest(store)
    cases = _resolved_report_cases(store)
    if language == "zh":
        headings = (
            "可复现性",
            "结论与适用范围",
            "结构验证",
            "Checkpoint 覆盖",
            "分布式算子审计",
            "真实并行归属",
            "问题台账",
            "原生模块精度对齐",
            "精度覆盖范围",
            "验收阈值",
            "精度矩阵详情",
            "全局最差误差",
            "性能与内存（不作为精度门禁）",
            "证据索引",
        )
        evidence_entries = [
            "完整机器可读比较：`comparison.json`",
            "展开后的用例：`cases/resolved_cases.json`",
            "分布式算子审计：`analysis/operator_support.json`",
            "并行归属证据：`analysis/parallel_ownership.json`",
            "问题台账：`analysis/issues.json`",
            "每个用例的日志、metrics、参数探针和 checkpoint 证据：`cases/<case>/`",
        ]
    else:
        headings = (
            "Reproducibility",
            "Verdict and scope",
            "Structure validation",
            "Checkpoint coverage",
            "Distributed operator audit",
            "Actual parallel ownership",
            "Issue ledger",
            "Native module parity",
            "Precision coverage",
            "Acceptance thresholds",
            "Precision matrix details",
            "Worst observed errors",
            "Performance and memory (non-gating)",
            "Evidence index",
        )
        evidence_entries = [
            "complete machine-readable comparison: `comparison.json`",
            "resolved cases: `cases/resolved_cases.json`",
            "distributed operator audit: `analysis/operator_support.json`",
            "parallel ownership evidence: `analysis/parallel_ownership.json`",
            "issue ledger: `analysis/issues.json`",
            "per-case logs, metrics, parameter probes, and checkpoint evidence: `cases/<case>/`",
        ]
    return [
        (headings[0], _reproducibility_entries(store, environment, manifest, language)),
        (
            headings[1],
            _verdict_entries(store, findings, parity, comparison, checkpoint, language),
        ),
        (headings[2], _structure_entries(findings, language)),
        (headings[3], _checkpoint_entries(checkpoint, language)),
        (headings[4], _operator_support_entries(store, language)),
        (headings[5], _parallel_ownership_entries(store, language)),
        (headings[6], _issue_ledger_entries(store, language)),
        (headings[7], _detailed_parity_entries(parity, language)),
        (headings[8], _precision_coverage_entries(store, language)),
        (headings[9], _acceptance_entries(manifest, language)),
        (headings[10], _precision_detail_entries(cases, comparison, manifest, language)),
        (headings[11], _worst_case_entries(comparison, language)),
        (headings[12], _detailed_performance_entries(comparison, language)),
        (headings[13], evidence_entries),
    ]


def command_report(args: argparse.Namespace) -> int:
    """Render detailed English and Chinese PASS/FAIL/BLOCKED summaries."""
    store = EvidenceStore(args.output_dir)
    findings = _read_json(store.path("check/findings.json"))
    parity = _read_json(store.path("module_parity/comparison.json"))
    comparison = _read_json(store.path("comparison.json"))
    checkpoint = _read_json(store.path("checkpoint/coverage.json"))
    environment = _read_json(store.path("environment.json"))
    required_reports = (findings, parity, comparison)
    status = _aggregate_statuses(
        [value.get("status") for value in required_reports if value is not None]
        + (["BLOCKED"] if any(value is None for value in required_reports) else [])
    )
    english_summary = store.render_summary(
        status,
        "Model integration validation",
        _report_sections(
            store,
            findings,
            parity,
            comparison,
            checkpoint,
            environment,
            "en",
        ),
    )
    chinese_summary = store.render_summary(
        status,
        "模型接入验证报告",
        _report_sections(
            store,
            findings,
            parity,
            comparison,
            checkpoint,
            environment,
            "zh",
        ),
        relative_path="summary.zh-CN.md",
    )
    print(f"{status}: {english_summary}; {chinese_summary}")
    return 0 if status == "PASS" else 1 if status == "FAIL" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hyper_parallel.tools.model_integration",
        description=__doc__,
        epilog="exit codes: 0=PASS, 1=FAIL or execution error, 2=BLOCKED or manifest error",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("inspect", command_inspect, "inspect source and final model structure"),
        ("scaffold", command_scaffold, "generate a minimal model adapter scaffold"),
        ("check", command_check, "validate the final built model"),
        ("parity", command_parity, "run model-owned module parity cases"),
        ("validate", command_validate, "run the precision validation matrix"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--manifest", required=True)
        subparser.add_argument("--output-dir")
        subparser.set_defaults(handler=handler)
        if name == "scaffold":
            subparser.add_argument("--scaffold-dir", required=True)
        elif name == "parity":
            subparser.add_argument("--device", default="cpu")
            subparser.add_argument("--dtype", default="float32")
        elif name == "validate":
            subparser.add_argument("--generate-only", action="store_true")
    report_parser = subparsers.add_parser("report", help="render an evidence summary")
    report_parser.add_argument("--output-dir", required=True)
    report_parser.set_defaults(handler=command_report)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Execute the selected model-integration command."""
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except ManifestError as error:
        print(f"manifest error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
