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
"""Unit tests for model-integration contracts, model integration, parity, and evidence."""

from __future__ import annotations

import gc
import json
import os
import tempfile
import unittest
import weakref
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedAttentionState,
)
from hyper_parallel.models.deepseek_v41.adapter.validation import (
    shared_state_trace as shared_state_trace_module,
)
from hyper_parallel.models.deepseek_v41.adapter.validation.shared_state_trace import (
    begin_shared_state_trace,
    finish_shared_state_trace,
)
from hyper_parallel.tools.model_integration.checkpoint_coverage import build_checkpoint_coverage
from hyper_parallel.tools.model_integration.data_contract import (
    modality_gradient_parameters,
    validate_data_contract,
    validate_observed_forward_fields,
)
from hyper_parallel.tools.model_integration.evidence_store import (
    EvidenceStore,
    find_repository_root,
)
from hyper_parallel.tools.model_integration.structure_inventory import (
    build_model_inventory,
    diff_inventories,
    inspect_local_model_assets,
)
from hyper_parallel.tools.model_integration.optimizer_layout import validate_optimizer_layout
from hyper_parallel.tools.model_integration.manifest import ManifestError, load_manifest
from hyper_parallel.tools.model_integration.module_parity import ParityContext, run_module_parity
from hyper_parallel.tools.model_integration import case_compare
from hyper_parallel.tools.model_integration import tensor_probes
from hyper_parallel.tools.model_integration.case_compare import (
    compare_checkpoint_layouts,
    compare_parameter_probes,
    compare_scalar_curves,
    summarize_performance,
)
from hyper_parallel.tools.model_integration.case_matrix import generate_validation_cases
from hyper_parallel.tools.model_integration.tensor_probes import InputIdentityRecorder
from hyper_parallel.tools.model_integration.schemas import (
    FindingSeverity,
    IntegrationState,
    ModelIntegrationFinding,
)
from hyper_parallel.tools.model_integration.integration_validator import (
    validate_final_model,
    validate_state_invariants,
)
from hyper_parallel.models.adapter_spec import RecomputePolicy
from hyper_parallel.models.deepseek_v41.adapter.validation.model_validation_spec import (
    _run_native_module_parity,
)
from hyper_parallel.models.validation_spec import (
    CheckpointValidationSpec,
    DataValidationSpec,
    ModelValidationSpec,
    ModuleParityCase,
    ObservationSpec,
    StateInvariantSpec,
)
from hyper_parallel.tools.model_integration import cli as model_integration_cli
from hyper_parallel.tools.model_integration.cli import main as model_integration_main
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config.parallelism import ActivationCheckpointSelection
from hyper_parallel.trainer.config.training import ModelIntegrationConfig
from tests.common.mark_utils import arg_mark


class _ReferenceLinear(nn.Module):
    """Independent reference used by the parity test."""

    def __init__(self) -> None:
        """Create an uninitialized reference parameter."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(3, 4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the reference formula."""
        return inputs @ self.weight.t()


class _CandidateLinear(nn.Module):
    """Independent candidate used by the parity test."""

    def __init__(self) -> None:
        """Create the deterministic candidate parameter."""
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).view(3, 4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the candidate formula."""
        return torch.einsum("bi,oi->bo", inputs, self.weight)


class _CandidateTree(nn.Module):
    """Final-tree holder proving selector-based candidate lookup."""

    def __init__(self) -> None:
        """Build the final candidate module tree."""
        super().__init__()
        self.block = _CandidateLinear()


class _IncorrectCandidateLinear(_CandidateLinear):
    """Candidate with a deliberate forward mismatch."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add an observable error to the otherwise equivalent output."""
        return super().forward(inputs) + 1.0


def _copy_parity_weight(reference: nn.Module, candidate: nn.Module, _context: object):
    """Copy one authoritative tensor and return complete gradient mapping."""
    reference.weight.data.copy_(candidate.weight.data)
    return {"weight": "weight"}


class _ExpertMesh:
    """Duck-typed source mesh exposing an expert dimension."""

    mesh_dim_names = ("ep",)


class _FakeUnit(nn.Module):
    """Duck-typed HSDP owner for gradient-domain validation."""

    def __init__(self, domains: tuple[str, ...]) -> None:
        """Create fake sharded parameters with the requested domains."""
        super().__init__()
        self.parameters_by_domain = nn.ParameterList(
            [nn.Parameter(torch.ones(2)) for _ in domains]
        )
        hsdp_params = []
        for index, (domain, parameter) in enumerate(zip(domains, self.parameters_by_domain)):
            source_info = SimpleNamespace(
                mesh=_ExpertMesh() if domain == "EXPERT" else SimpleNamespace(mesh_dim_names=("dp",)),
                placements=("Shard(dim=0)",) if domain == "EXPERT" else ("Replicate()",),
            )
            hsdp_params.append(
                SimpleNamespace(
                    sharded_param=parameter,
                    source_shard_info=source_info,
                    _orig_size=parameter.shape,
                    _param_fqn=f"parameters_by_domain.{index}",
                )
            )
        self.hsdp_scheduler = SimpleNamespace(
            hsdp_state=SimpleNamespace(hsdp_params=hsdp_params),
            forward_prefetch_cells=(),
            backward_prefetch_cells=(),
        )


class _UnevenShardTensor:
    """DTensor-shaped test double whose full logical tensor cannot be rebuilt."""

    shape = torch.Size((16,))
    dtype = torch.float32

    def __init__(self) -> None:
        """Represent rank 15's empty local shard of a 16-element parameter."""
        placement = SimpleNamespace(
            dim=0,
            is_shard=lambda: True,
            is_ragged_shard=lambda: False,
        )
        mesh = SimpleNamespace(
            mesh=torch.arange(16),
            mesh_dim_names=("dp",),
            get_coordinate=lambda: (15,),
        )
        self.layout = SimpleNamespace(mesh=mesh, placements=(placement,))

    def to_local(self) -> torch.Tensor:
        """Return the valid empty local shard."""
        return torch.empty(0)

    @staticmethod
    def full_tensor() -> torch.Tensor:
        """Prove that diagnostics never request unsupported redistribution."""
        raise AssertionError("full_tensor must not be called for uneven shards")


class TestModelIntegrationContracts(unittest.TestCase):
    """Exercise hardware-independent framework contracts."""

    def test_trainer_config_exposes_only_validation_scope(self) -> None:
        """Keep probe selection and output policy out of the user YAML schema."""
        self.assertEqual(
            [config_field.name for config_field in fields(ModelIntegrationConfig)],
            ["mode"],
        )
        with self.assertRaisesRegex(ValueError, "off, build, runtime"):
            ModelIntegrationConfig(mode="strict")

    def test_structured_finding_requires_actionable_error(self) -> None:
        """Reject malformed codes and non-actionable ERROR findings."""
        with self.assertRaisesRegex(ValueError, "invalid model integration error code"):
            ModelIntegrationFinding(
                code="BAD",
                phase="D1",
                owner_fqn="root",
                severity=FindingSeverity.WARNING,
                message="bad code",
            )
        with self.assertRaisesRegex(ValueError, "require why_unsafe"):
            ModelIntegrationFinding(
                code="HP-TEST-001",
                phase="D1",
                owner_fqn="root",
                severity=FindingSeverity.ERROR,
                message="missing remediation",
            )

    def test_evidence_state_rejects_skip_and_regression(self) -> None:
        """Persist only adjacent validation gates in forward order."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = EvidenceStore(temporary_directory)
            store.write_json("environment.json", {})
            store.update_state(IntegrationState.DISCOVERED, ("environment.json",))
            with self.assertRaisesRegex(ValueError, "cannot skip"):
                store.update_state(IntegrationState.MODULE_PARITY_PASSED)
            store.update_state(IntegrationState.STRUCTURE_VALIDATED)
            with self.assertRaisesRegex(ValueError, "cannot regress"):
                store.update_state(IntegrationState.DISCOVERED)
            store.advance_if_before(IntegrationState.DISCOVERED)
            self.assertEqual(
                store.current_state(),
                IntegrationState.STRUCTURE_VALIDATED,
            )

    def test_validation_specs_reject_ambiguous_contracts(self) -> None:
        """Reject ignored tolerances and incomplete in-process parity cases."""
        with self.assertRaisesRegex(ValueError, "must not declare"):
            ObservationSpec(name="tokens", comparison="exact", atol=0.0)
        with self.assertRaisesRegex(TypeError, "candidate_builder"):
            ModuleParityCase(name="missing", candidate_selector="<root>")
        isolated = ModuleParityCase(
            name="isolated",
            candidate_selector="<root>",
            execution="isolated_process",
            isolated_runner=lambda _context: {"status": "PASS"},
        )
        self.assertIsNone(isolated.candidate_builder)

    def test_activation_checkpoint_selection_requires_one_exact_selector(self) -> None:
        """Require exactly one valid count-based or index-based selector."""
        with self.assertRaisesRegex(ValueError, "only valid"):
            ActivationCheckpointSelection(source="default", layer_count=1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ActivationCheckpointSelection(source="model_adapter_safe_regions")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ActivationCheckpointSelection(
                source="model_adapter_safe_regions",
                layer_count=1,
                layer_indices=[0],
            )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            ActivationCheckpointSelection(
                source="model_adapter_safe_regions",
                layer_count=-1,
            )
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            ActivationCheckpointSelection(
                source="model_adapter_safe_regions",
                layer_indices=[0, 0],
            )
        self.assertEqual(
            ActivationCheckpointSelection(
                source="model_adapter_safe_regions",
                layer_count=0,
            ).layer_count,
            0,
        )
        self.assertEqual(
            ActivationCheckpointSelection(
                source="model_adapter_safe_regions",
                layer_indices=[0, 2],
            ).layer_indices,
            [0, 2],
        )

    def test_manifest_is_local_only_and_command_validated(self) -> None:
        """Reject implicit remote model sources before command execution."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                "schema_version: 1\n"
                "model:\n  adapter: example\n  id_or_path: https://example.invalid/model\n",
                encoding="utf-8",
            )
            manifest = load_manifest(manifest_path)
            with self.assertRaisesRegex(ManifestError, "must be local"):
                manifest.validate_for("inspect")

    def test_manifest_fingerprints_model_and_reference_sources(self) -> None:
        """Hash bounded source metadata without user-authored digest fields."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_dir = root / "model"
            native_dir = root / "native" / "inference"
            model_dir.mkdir()
            native_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text('{"model_type": "example"}', encoding="utf-8")
            (native_dir / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                "schema_version: 1\n"
                f"model:\n  adapter: example\n  id_or_path: {model_dir}\n"
                f"reference:\n  source_path: {root / 'native'}\n",
                encoding="utf-8",
            )

            fingerprints = load_manifest(manifest_path).fingerprint_paths()

            self.assertIn("config.json", fingerprints["model.id_or_path"]["metadata"])
            self.assertIn(
                "inference/model.py",
                fingerprints["reference.source_path"]["metadata"],
            )
            self.assertIn(
                "sha256",
                fingerprints["model.id_or_path"]["metadata"]["config.json"],
            )

    def test_scaffold_emits_only_user_validation_choices(self) -> None:
        """Keep generated hashes, environment, and model assets out of validation YAML."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                "schema_version: 1\n"
                "model:\n  adapter: example\n"
                f"output_dir: {root / 'evidence'}\n",
                encoding="utf-8",
            )
            scaffold_dir = root / "generated"

            status = model_integration_main(
                [
                    "scaffold",
                    "--manifest",
                    str(manifest_path),
                    "--scaffold-dir",
                    str(scaffold_dir),
                ]
            )
            generated = load_manifest(scaffold_dir / "validation.yaml")

        self.assertEqual(status, 0)
        self.assertEqual(
            set(generated.raw),
            {"schema_version", "model", "launcher", "matrix", "acceptance"},
        )
        self.assertFalse(generated.reference)

    def test_inventory_retains_aliases_and_classifies_state_diff(self) -> None:
        """Record tied parameter aliases and classify changed state."""
        model = nn.Module()
        model.first = nn.Linear(2, 2, bias=False)
        model.second = nn.Linear(2, 2, bias=False)
        model.second.weight = model.first.weight
        reference = build_model_inventory(model)
        candidate_model = nn.Sequential(nn.Linear(2, 3, bias=False))
        candidate = build_model_inventory(candidate_model)

        self.assertTrue(reference.aliases["parameters"])
        differences = diff_inventories(reference, candidate)
        self.assertTrue(any(item.path.startswith("parameters.") for item in differences))

    def test_inventory_records_source_only_git_lfs_pointers(self) -> None:
        """Treat source-only checkpoint placeholders as unavailable payloads."""
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:0123456789abcdef\n"
            "size 1560000000000\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                pointer,
                encoding="utf-8",
            )
            (root / "model-00001-of-00001.safetensors").write_text(
                pointer,
                encoding="utf-8",
            )
            assets = inspect_local_model_assets(root)

        checkpoint = assets["checkpoint"]
        self.assertFalse(checkpoint["weights_available"])
        self.assertEqual(checkpoint["weight_map"], {})
        self.assertEqual(len(checkpoint["lfs_pointers"]), 2)
        self.assertEqual(
            checkpoint["lfs_pointers"]["model.safetensors.index.json"]["payload_size"],
            1560000000000,
        )

    def test_mixed_gradient_domain_fails_before_collectives(self) -> None:
        """Reject an HSDP unit that owns expert and dense parameters."""
        report, _, _ = validate_final_model(
            _FakeUnit(("DENSE", "EXPERT")),
        )

        self.assertIn("HP-FSDP-004", {finding.code for finding in report.errors})

    def test_homogeneous_gradient_domain_passes(self) -> None:
        """Accept homogeneous dense ownership."""
        report, _, _ = validate_final_model(_FakeUnit(("DENSE", "DENSE")))

        self.assertFalse(report.errors)

    def test_parity_compares_output_input_grad_and_parameter_grad(self) -> None:
        """Run final selector, full weight coverage, and backward comparison."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = ParityContext(
                manifest=SimpleNamespace(),
                dtype="float32",
                device="cpu",
                evidence_store=EvidenceStore(temporary_directory),
                options={"tolerance": {"atol": 0.0, "rtol": 0.0}},
            )
            validation_spec = ModelValidationSpec(
                module_cases=(
                    ModuleParityCase(
                        name="linear",
                        candidate_selector="block",
                        candidate_builder=lambda _context: _CandidateTree(),
                        reference_builder=lambda _context: _ReferenceLinear(),
                        input_builder=lambda _context: torch.arange(
                            8, dtype=torch.float32
                        ).view(2, 4).requires_grad_(),
                        weight_adapter=_copy_parity_weight,
                    ),
                )
            )

            report = run_module_parity(validation_spec, context)

        self.assertEqual(report["status"], "PASS")
        names = {metric["name"] for metric in report["cases"][0]["metrics"]}
        self.assertIn("output.0", names)
        self.assertIn("input_grad.0", names)
        self.assertIn("parameter_grad.weight", names)

    def test_parity_reports_numerical_failure(self) -> None:
        """A differing final candidate must fail instead of emitting a partial PASS."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = ParityContext(
                manifest=SimpleNamespace(),
                dtype="float32",
                device="cpu",
                evidence_store=EvidenceStore(temporary_directory),
                options={"tolerance": {"atol": 0.0, "rtol": 0.0}},
            )
            candidate_tree = nn.Module()
            candidate_tree.block = _IncorrectCandidateLinear()
            spec = ModelValidationSpec(
                module_cases=(
                    ModuleParityCase(
                        name="incorrect_linear",
                        candidate_selector="block",
                        candidate_builder=lambda _context: candidate_tree,
                        reference_builder=lambda _context: _ReferenceLinear(),
                        input_builder=lambda _context: torch.ones(2, 4, requires_grad=True),
                        weight_adapter=_copy_parity_weight,
                    ),
                )
            )

            report = run_module_parity(spec, context)

        self.assertEqual(report["status"], "FAIL")

    def test_parity_accepts_parameter_probe_tolerance_schema(self) -> None:
        """Apply max-abs and relative-L2 limits from acceptance.parameters."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = ParityContext(
                manifest=SimpleNamespace(),
                dtype="float32",
                device="cpu",
                evidence_store=EvidenceStore(temporary_directory),
                options={"tolerance": {"max_abs": 1.0, "relative_l2": 1.0}},
            )
            candidate_tree = nn.Module()
            candidate_tree.block = _IncorrectCandidateLinear()
            spec = ModelValidationSpec(
                module_cases=(
                    ModuleParityCase(
                        name="tolerated_linear",
                        candidate_selector="block",
                        candidate_builder=lambda _context: candidate_tree,
                        reference_builder=lambda _context: _ReferenceLinear(),
                        input_builder=lambda _context: torch.ones(2, 4, requires_grad=True),
                        weight_adapter=_copy_parity_weight,
                    ),
                )
            )

            report = run_module_parity(spec, context)

        self.assertEqual(report["status"], "PASS")

    def test_parity_treats_null_tolerance_as_unspecified(self) -> None:
        """Allow an explicit YAML null without passing None to float()."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = ParityContext(
                manifest=SimpleNamespace(),
                dtype="float32",
                device="cpu",
                evidence_store=EvidenceStore(temporary_directory),
                options={"tolerance": {"max_abs": None, "relative_l2": None}},
            )
            spec = ModelValidationSpec(
                module_cases=(
                    ModuleParityCase(
                        name="linear",
                        candidate_selector="block",
                        candidate_builder=lambda _context: _CandidateTree(),
                        reference_builder=lambda _context: _ReferenceLinear(),
                        input_builder=lambda _context: torch.ones(2, 4, requires_grad=True),
                        weight_adapter=_copy_parity_weight,
                    ),
                )
            )

            report = run_module_parity(spec, context)

        self.assertEqual(report["status"], "PASS")

    def test_native_parity_requires_zero_subprocess_returncode(self) -> None:
        """Reject a native harness that writes a report and then exits nonzero."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            context = SimpleNamespace(
                manifest=SimpleNamespace(reference_path=root, reference={}),
                evidence_store=EvidenceStore(root / "evidence"),
                device="cpu",
                dtype="float32",
            )

            def _failed_process(command, **_kwargs):
                output_path = Path(command[command.index("--output") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps({"summary": {"pass": 3, "fail": 0, "error": 0}}),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=7, stdout="", stderr="crashed")

            with mock.patch(
                "hyper_parallel.models.deepseek_v41.adapter.validation.model_validation_spec.subprocess.run",
                side_effect=_failed_process,
            ):
                report = _run_native_module_parity(context)

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["returncode"], 7)

    def test_case_generator_forces_ep1_and_single_axis_cases(self) -> None:
        """An EP production baseline always gains an EP1 control."""
        cases = generate_validation_cases(
            {
                "baseline": {"tp": 1, "cp": 1, "ep": 16, "fsdp": 16},
                "axes": {"tp": [1, 2], "cp": [1, 2]},
                "combined": [{"tp": 2, "cp": 2}],
            }
        )

        self.assertTrue(any(case.topology["ep"] == 1 for case in cases))
        self.assertTrue(any(case.topology["tp"] == 2 for case in cases))
        self.assertTrue(any(case.topology["cp"] == 2 for case in cases))

    def test_case_generator_rejects_unlaunched_topology_fields(self) -> None:
        """Reject matrix axes that cannot alter the standard Trainer command."""
        with self.assertRaisesRegex(ValueError, "unsupported field 'mbs'"):
            generate_validation_cases(
                {
                    "baseline": {"tp": 1, "cp": 1, "ep": 1, "fsdp": 1},
                    "axes": {"mbs": [1, 2]},
                }
            )

    def test_validate_wraps_invalid_matrix_as_manifest_error(self) -> None:
        """Keep user configuration errors on the CLI's traceback-free path."""
        manifest = SimpleNamespace(
            matrix={
                "baseline": {"tp": 1, "cp": 1, "ep": 1, "fsdp": 1},
                "axes": {"mbs": [1, 2]},
            }
        )
        with (
            mock.patch.object(model_integration_cli, "_manifest_from_args", return_value=manifest),
            mock.patch.object(model_integration_cli, "_store", return_value=mock.Mock()),
            mock.patch.object(model_integration_cli, "_require_precision_handoff"),
            mock.patch.object(
                model_integration_cli,
                "_validation_spec",
                return_value=ModelValidationSpec(),
            ),
            mock.patch.object(model_integration_cli, "_recompute_policy", return_value=None),
        ):
            with self.assertRaisesRegex(ManifestError, "invalid validation matrix"):
                model_integration_cli.command_validate(SimpleNamespace())

    def test_case_generator_classifies_real_topology_changes(self) -> None:
        """Select tolerances and checkpoint source topology from launched fields."""
        baseline = {"tp": 1, "cp": 1, "ep": 1, "fsdp": 16}
        cases = generate_validation_cases(
            {
                "baseline": baseline,
                "recompute_selections": [{"layer_count": 2}],
                "cross_topology_resume": {"tp": 2, "ep": 16, "fsdp": 8},
            },
            RecomputePolicy(
                safe_module_patterns=("model.layers.*.mlp",),
            ),
        )
        by_name = {case.name: case for case in cases}

        recompute_case = next(
            case
            for case in cases
            if case.topology.get("recompute") == {"layer_count": 2}
        )
        self.assertEqual(recompute_case.metadata["acceptance"], "same_topology")
        cross_resume = by_name["cross_topology_resume"]
        self.assertEqual(cross_resume.metadata["acceptance"], "cross_topology")
        self.assertEqual(cross_resume.metadata["prepare_topology"], baseline)

    def test_case_generator_rejects_recompute_only_cross_resume(self) -> None:
        """Explain that cross-topology resume cannot vary recompute alone."""
        with self.assertRaisesRegex(ValueError, "recompute-only resume matrix"):
            generate_validation_cases(
                {
                    "baseline": {
                        "tp": 1,
                        "cp": 1,
                        "ep": 1,
                        "fsdp": 16,
                        "recompute": {"layer_count": 0},
                    },
                    "cross_topology_resume": {
                        "recompute": {"layer_indices": [0, 1]},
                    },
                }
            )

    def test_case_generator_omits_unconsumed_execution_mode(self) -> None:
        """Represent the Production/Validate pair only with launched fields."""
        cases = generate_validation_cases(
            {
                "baseline": {"tp": 1, "cp": 1, "ep": 1, "fsdp": 1},
                "production_validate_pair": {"tp": 2, "fsdp": 2},
            }
        )
        pair = [case for case in cases if case.kind == "production_validate"]

        self.assertEqual(len(pair), 2)
        self.assertEqual(
            {case.topology["validate_placement"] for case in pair},
            {False, True},
        )
        self.assertTrue(
            all("execution_mode" not in case.topology for case in pair)
        )

    def test_deepseek_recompute_examples_are_explicitly_cross_topology(self) -> None:
        """Keep the public EP16 recompute example classified against EP1."""
        manifest = load_manifest(
            find_repository_root()
            / "examples/training_demo/deepseek_v41/deepseek_v41_validation.yaml"
        )
        cases = generate_validation_cases(
            manifest.matrix,
            RecomputePolicy(
                safe_module_patterns=("model.layers.*.mlp",),
            ),
        )
        recompute_cases = [case for case in cases if case.name.startswith("recompute_")]

        self.assertTrue(recompute_cases)
        self.assertTrue(
            all(case.metadata["acceptance"] == "cross_topology" for case in recompute_cases)
        )

    def test_compare_validation_case_requires_acceptance_metadata(self) -> None:
        """Never weaken comparison limits when generated metadata is incomplete."""
        case = SimpleNamespace(name="incomplete", metadata={})
        with self.assertRaisesRegex(ManifestError, "no acceptance classification"):
            model_integration_cli._compare_validation_case(  # pylint: disable=protected-access
                SimpleNamespace(),
                case,
                Path("baseline"),
                Path("candidate"),
                ("PASS", "PASS"),
            )

    def test_case_generator_uses_production_recompute_policy(self) -> None:
        """Generate recompute cases without importing validation declarations."""
        cases = generate_validation_cases(
            {
                "baseline": {"tp": 1, "cp": 1, "ep": 1, "fsdp": 1},
                "recompute_selections": [
                    {"layer_count": 0},
                    {"layer_count": 3},
                    {"layer_indices": [0, 2]},
                ],
            },
            RecomputePolicy(
                safe_module_patterns=("model.layers.*.mlp",),
            ),
        )

        self.assertEqual(
            [case.topology.get("recompute") for case in cases],
            [
                None,
                {"layer_count": 0},
                {"layer_count": 3},
                {"layer_indices": [0, 2]},
            ],
        )

    def test_input_identity_is_invariant_to_cp_shards_and_tp_replicas(self) -> None:
        """Canonical input identity reconstructs CP and ignores TP copies."""
        full = torch.tensor([[1, 2, 3, 4]])
        entry = {
            "micro_step": 0,
            "owner": "model",
            "field": "input_ids",
            "value_index": 0,
        }
        baseline = [
            {
                "coordinates": {"dp": 0, "cp": 0, "tp": 0, "pp": 0},
                "entries": [{**entry, "tensor": full}],
            }
        ]
        sharded = [
            {
                "coordinates": {"dp": 0, "cp": 0, "tp": 0, "pp": 0},
                "entries": [{**entry, "tensor": full[:, :2]}],
            },
            {
                "coordinates": {"dp": 0, "cp": 1, "tp": 0, "pp": 0},
                "entries": [{**entry, "tensor": full[:, 2:]}],
            },
            {
                "coordinates": {"dp": 0, "cp": 0, "tp": 1, "pp": 0},
                "entries": [{**entry, "tensor": full[:, :2]}],
            },
        ]

        self.assertEqual(
            InputIdentityRecorder._canonical_global_hash(baseline),
            InputIdentityRecorder._canonical_global_hash(sharded),
        )

    def test_optimizer_layout_accepts_plain_parameter_state(self) -> None:
        """A local Adam state follows its owning local parameter layout."""
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        self.assertEqual(validate_optimizer_layout(model, optimizer), [])

    def test_parameter_summary_does_not_rebuild_uneven_dtensor(self) -> None:
        """Inspect an empty FSDP shard without requesting unsupported redistribution."""
        summary = tensor_probes._tensor_summary(  # pylint: disable=protected-access
            "small_parameter",
            _UnevenShardTensor(),
        )

        self.assertEqual(summary["global_shape"], [16])
        self.assertEqual(summary["layout"]["local_shape"], [0])
        self.assertTrue(summary["finite"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_data_contract_reports_declared_collision_and_shift(self) -> None:
        """Report ownership and causal-shift mismatches.

        Feature: Model-integration data contract.
        Description: Declare a runtime collision and an incompatible label shift.
        Expectation: Validation returns both ``HP-DATA-001`` and ``HP-DATA-002``.
        """
        runtime_adapter = SimpleNamespace(
            runtime_input_fields=lambda: ("input_ids", "packed_seq_params")
        )
        validation_spec = ModelValidationSpec(
            data=DataValidationSpec(
                runtime_fields=("packed_seq_params",),
                labels_are_shifted=True,
            )
        )

        _, findings = validate_data_contract(
            validation_spec,
            runtime_adapter,
            labels_are_shifted=False,
            source_type="online",
        )

        self.assertEqual({finding.code for finding in findings}, {"HP-DATA-001", "HP-DATA-002"})

    def test_runtime_data_observations_cover_fields_and_modality_gradients(self) -> None:
        """Validate first-forward fields and finite gradients for modality groups."""
        model = nn.Module()
        model.vision = nn.Linear(2, 2, bias=False)
        model.vision(torch.ones(1, 2)).sum().backward()
        spec = ModelValidationSpec(
            data=DataValidationSpec(
                required_forward_fields=("input_ids", "pixel_values"),
                modality_parameter_patterns=("vision.*",),
            )
        )

        findings = validate_observed_forward_fields(spec, {"input_ids": torch.ones(1)})
        gradients = modality_gradient_parameters(spec, model)

        self.assertEqual({finding.code for finding in findings}, {"HP-DATA-002"})
        self.assertEqual(gradients, {"vision.*": True})

    def test_invalid_invariant_phase_is_structured(self) -> None:
        """Turn an adapter phase typo into an actionable finding."""
        spec = ModelValidationSpec(
            state_invariants=(
                StateInvariantSpec(
                    name="bad_phase",
                    checker=lambda _model, _context: None,
                    phase="typo",  # type: ignore[arg-type]
                ),
            )
        )

        findings = validate_state_invariants(nn.Linear(1, 1), spec, {})

        self.assertEqual({finding.code for finding in findings}, {"HP-SPEC-001"})

    def test_summary_first_line_is_machine_readable(self) -> None:
        """Generate a deterministic PASS/FAIL/BLOCKED report header."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = EvidenceStore(temporary_directory)
            summary = store.render_summary("PASS", "Example", (("Gate", ("ok",)),))

            self.assertEqual(summary.read_text(encoding="utf-8").splitlines()[0], "PASS")
            self.assertEqual(json.loads(json.dumps({"state": "PASS"}))["state"], "PASS")

    def test_precision_coverage_separates_strategy_from_resume(self) -> None:
        """Make summary coverage counts resistant to resume-case inflation."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = EvidenceStore(temporary_directory)
            store.write_json(
                "cases/resolved_cases.json",
                [
                    {"name": "baseline", "kind": "topology"},
                    {"name": "axis-ep_example", "kind": "topology"},
                    {"name": "combined-0_example", "kind": "topology"},
                    {"name": "recompute_example", "kind": "topology"},
                    {"name": "same_topology_resume", "kind": "resume"},
                ],
            )
            store.write_yaml(
                "manifest.resolved.yaml",
                {"matrix": {"coverage_notes": ["CP is unsupported for this crop."]}},
            )

            entries = model_integration_cli._precision_coverage_entries(store)  # pylint: disable=protected-access

        self.assertIn("strategy_generalization=2", entries[0])
        self.assertIn("recompute=1", entries[0])
        self.assertIn("resume=1", entries[0])
        self.assertIn("excluded: CP is unsupported for this crop.", entries)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_report_writes_detailed_english_and_chinese_summaries(self) -> None:
        """Render detailed bilingual precision evidence.

        Feature: Bilingual model-integration report rendering.
        Description: Render floating and discrete native parity observations.
        Expectation: Numeric errors and exact-equality results replace false ``n/a`` values.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = EvidenceStore(temporary_directory)
            store.write_json(
                "check/findings.json",
                {
                    "status": "PASS",
                    "findings": [],
                    "metadata": {
                        "model_type": "example",
                        "inventory_counts": {"modules": 3, "parameters": 2, "buffers": 1},
                    },
                },
            )
            store.write_json(
                "module_parity/comparison.json",
                {
                    "status": "PASS",
                    "cases": [{
                        "case": "native",
                        "status": "PASS",
                        "observations": [{
                            "name": "attention",
                            "status": "pass",
                            "metrics": {
                                "max_abs": 1.0e-5,
                                "relative_l2": 2.0e-6,
                            },
                        }, {
                            "name": "attention.topk_indices",
                            "status": "pass",
                            "dtype": "torch.int64",
                        }],
                    }],
                },
            )
            comparison = {
                "status": "PASS",
                "cases": [{"case": "baseline", "status": "PASS"}],
                "comparisons": {
                    "axis-tp_example": {
                        "status": "PASS",
                        "scalar": {
                            "status": "PASS",
                            "steps": [{
                                "step": 1,
                                "status": "PASS",
                                "loss_max_abs": 1.0e-5,
                                "norm_max_abs": 2.0e-4,
                                "post_clip_norm_max_abs": 0.0,
                                "input_identity": True,
                                "learning_rate_equal": True,
                                "finite": True,
                            }],
                        },
                        "parameter_probes": {
                            "status": "PASS",
                            "optimizer_state_numeric_comparison": "skipped_cross_topology",
                            "skipped_optimizer_states": 2,
                        },
                        "checkpoint_layout": None,
                    }
                },
                "performance": {
                    "baseline": {
                        "steady_steps": 1,
                        "step_time_p50_seconds": 1.0,
                        "step_time_p90_seconds": 1.1,
                        "tokens_per_second_p50": 2.0,
                        "samples_per_second_p50": 1.0,
                        "peak_memory_allocated_bytes": 1024,
                        "peak_memory_reserved_bytes": 2048,
                    }
                },
            }
            store.write_json("comparison.json", comparison)
            store.write_json(
                "checkpoint/coverage.json",
                {"status": "NOT_LOADED", "reason": "scratch model"},
            )
            store.write_json(
                "cases/resolved_cases.json",
                [
                    {
                        "name": "baseline",
                        "kind": "topology",
                        "topology": {
                            "tp": 1,
                            "cp": 1,
                            "ep": 1,
                            "fsdp": 2,
                            "sequence_parallel": False,
                            "recompute": {"layer_count": 0},
                        },
                        "metadata": {"acceptance": "same_topology"},
                    },
                    {
                        "name": "axis-tp_example",
                        "kind": "topology",
                        "topology": {
                            "tp": 2,
                            "cp": 1,
                            "ep": 1,
                            "fsdp": 1,
                            "sequence_parallel": False,
                            "recompute": {"layer_count": 0},
                        },
                        "metadata": {"acceptance": "cross_topology"},
                    },
                ],
            )
            store.write_yaml(
                "manifest.resolved.yaml",
                {
                    "reference": {"source_path": "/source"},
                    "matrix": {
                        "steps": 10,
                        "coverage_notes": ["CP excluded."],
                        "coverage_notes_zh": ["未覆盖 CP。"],
                    },
                    "acceptance": {
                        "same_topology": {"loss_max_abs": 1.0e-4, "norm_max_abs": 1.0e-3},
                        "cross_topology": {"loss_max_abs": 5.0e-3, "norm_max_abs": 1.25e-1},
                        "parameters": {
                            "max_abs": 1.0e-3,
                            "relative_l2": 6.0e-2,
                            "combination": "any",
                        },
                    },
                },
            )
            store.write_json(
                "environment.json",
                {
                    "git_revision": "revision",
                    "hyper_parallel": {"import_path": "/checkout/hyper_parallel/__init__.py"},
                    "torch": {"version": "2.9"},
                    "transformers": {"version": "4.57"},
                },
            )

            report_status = model_integration_main(
                ["report", "--output-dir", temporary_directory]
            )
            english = store.path("summary.md").read_text(encoding="utf-8")
            chinese = store.path("summary.zh-CN.md").read_text(encoding="utf-8")

        self.assertEqual(report_status, 0)
        self.assertEqual(english.splitlines()[0], "PASS")
        self.assertEqual(chinese.splitlines()[0], "PASS")
        self.assertIn("strategy_generalization=1", english)
        self.assertIn("max_loss_abs=1e-05", english)
        self.assertIn("checkpoint_layout=not applicable (non-resume case)", english)
        self.assertIn("策略泛化=1", chinese)
        self.assertIn("输入身份=PASS", chinese)
        self.assertIn("最大绝对误差=1e-05", chinese)
        self.assertIn("精确一致性=pass", chinese)
        self.assertIn("checkpoint layout=不适用（非断点续训用例）", chinese)
        self.assertNotIn("记录的最大绝对误差=n/a", chinese)

    def test_scalar_comparison_blocks_empty_evidence_and_checks_lr(self) -> None:
        """Never turn absent Trainer evidence into a successful comparison."""
        self.assertEqual(compare_scalar_curves([], [], {})["status"], "BLOCKED")
        baseline = [{
            "step": 1,
            "loss": 1.0,
            "grad_norm_pre_clip": 2.0,
            "grad_norm_post_clip": 1.0,
            "global_input_sha256": "same",
            "lr": 0.1,
        }]
        candidate = [{**baseline[0], "lr": 0.2}]
        self.assertEqual(compare_scalar_curves(baseline, candidate, {})["status"], "FAIL")

    def test_checkpoint_layout_allows_cross_topology_resharding(self) -> None:
        """Require global shape always and placements only for same-topology resume."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline" / "checkpoint"
            candidate = root / "candidate" / "checkpoint"
            baseline.mkdir(parents=True)
            candidate.mkdir(parents=True)
            baseline_layout = {
                "tensor_layouts": {
                    "model.weight": {
                        "global_shape": [8],
                        "mesh_dim_names": ["dp"],
                        "mesh_shape": [8],
                        "placements": ["Shard(0)"],
                    }
                }
            }
            candidate_layout = {
                "tensor_layouts": {
                    "model.weight": {
                        "global_shape": [8],
                        "mesh_dim_names": ["dp"],
                        "mesh_shape": [4],
                        "placements": ["Shard(0)"],
                    }
                }
            }
            (baseline / "before_save_rank0.json").write_text(
                json.dumps(baseline_layout), encoding="utf-8"
            )
            (candidate / "after_load_rank0.json").write_text(
                json.dumps(candidate_layout), encoding="utf-8"
            )

            self.assertEqual(
                compare_checkpoint_layouts(root / "baseline", root / "candidate", same_topology=False)[
                    "status"
                ],
                "PASS",
            )
            self.assertEqual(
                compare_checkpoint_layouts(root / "baseline", root / "candidate", same_topology=True)[
                    "status"
                ],
                "FAIL",
            )

    def test_checkpoint_layout_matches_the_restored_step(self) -> None:
        """Compare the saved layout from the restored step, not a later save."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline" / "checkpoint"
            candidate = root / "candidate" / "checkpoint"
            baseline.mkdir(parents=True)
            candidate.mkdir(parents=True)
            step_two = {"tensor_layouts": {"model.weight": {"global_shape": [8]}}}
            step_four = {"tensor_layouts": {"model.weight": {"global_shape": [16]}}}
            (baseline / "before_save_global_step_2_rank0.json").write_text(
                json.dumps(step_two), encoding="utf-8"
            )
            (baseline / "before_save_global_step_4_rank0.json").write_text(
                json.dumps(step_four), encoding="utf-8"
            )
            (candidate / "after_load_global_step_2_rank0.json").write_text(
                json.dumps(step_two), encoding="utf-8"
            )

            comparison = compare_checkpoint_layouts(
                root / "baseline", root / "candidate", same_topology=False
            )

            self.assertEqual(comparison["status"], "PASS")
            self.assertEqual(comparison["baseline_event"], "before_save_global_step_2")
            self.assertEqual(comparison["candidate_event"], "after_load_global_step_2")

    def test_checkpoint_layout_allows_load_primed_optimizer_entries(self) -> None:
        """Accept optimizer tensors initialized only to provide a load skeleton."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline" / "checkpoint"
            candidate = root / "candidate" / "checkpoint"
            baseline.mkdir(parents=True)
            candidate.mkdir(parents=True)
            baseline_layout = {"tensor_layouts": {"model.weight": {"global_shape": [8]}}}
            candidate_layout = {
                "tensor_layouts": {
                    **baseline_layout["tensor_layouts"],
                    "optimizer.state.model.weight.exp_avg": {"global_shape": [8]},
                }
            }
            (baseline / "before_save_global_step_2_rank0.json").write_text(
                json.dumps(baseline_layout), encoding="utf-8"
            )
            (candidate / "after_load_global_step_2_rank0.json").write_text(
                json.dumps(candidate_layout), encoding="utf-8"
            )

            comparison = compare_checkpoint_layouts(
                root / "baseline", root / "candidate", same_topology=True
            )

            self.assertEqual(comparison["status"], "PASS")
            self.assertEqual(
                comparison["initialized_optimizer_entries"],
                ["optimizer.state.model.weight.exp_avg"],
            )

    def test_parameter_probes_allow_absent_lazy_state_to_match_zero_state(self) -> None:
        """Accept a zero optimizer state materialized only for checkpoint load."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline" / "parameter_probes"
            candidate = root / "candidate" / "parameter_probes"
            baseline.mkdir(parents=True)
            candidate.mkdir(parents=True)
            common = {
                "step": 3,
                "stage": "main_after_optimizer",
                "name": "model.weight",
                "rank": 0,
                "global_shape": [1],
                "layout": {"global_shape": [1]},
                "finite": True,
                "sum": 1.0,
                "l2": 1.0,
                "min": 1.0,
                "max": 1.0,
                "values": [1.0],
            }
            zero_summary = {
                "global_shape": [1],
                "layout": {"global_shape": [1]},
                "finite": True,
                "sum": 0.0,
                "l2": 0.0,
                "min": 0.0,
                "max": 0.0,
                "values": [0.0],
            }
            (baseline / "rank0.jsonl").write_text(
                json.dumps(common) + "\n", encoding="utf-8"
            )
            (candidate / "rank0.jsonl").write_text(
                "\n".join((
                    json.dumps(common),
                    json.dumps({
                        "step": 3,
                        "stage": "optimizer_state",
                        "name": "model.bias",
                        "rank": 0,
                        "optimizer_state": {"exp_avg": zero_summary},
                    }),
                )) + "\n",
                encoding="utf-8",
            )

            comparison = compare_parameter_probes(
                root / "baseline",
                root / "candidate",
                {"max_abs": 0.0, "relative_l2": 0.0},
            )

            self.assertEqual(comparison["status"], "PASS")
            self.assertFalse(comparison["missing"])
            self.assertEqual(len(comparison["zero_initialized_optimizer_states"]), 1)

    def test_cross_topology_parameter_probes_skip_optimizer_state_values(self) -> None:
        """Keep topology-local optimizer internals out of numerical parity."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline" / "parameter_probes"
            candidate = root / "candidate" / "parameter_probes"
            baseline.mkdir(parents=True)
            candidate.mkdir(parents=True)

            def _summary(value: float) -> dict[str, object]:
                return {
                    "global_shape": [1],
                    "layout": {"global_shape": [1]},
                    "finite": True,
                    "sum": value,
                    "l2": abs(value),
                    "min": value,
                    "max": value,
                    "values": [value],
                }

            common = {
                "step": 1,
                "stage": "optimizer_state",
                "name": "model.weight",
                "rank": 0,
            }
            (baseline / "rank0.jsonl").write_text(
                json.dumps({**common, "optimizer_state": {"momentum": _summary(1.0)}}) + "\n",
                encoding="utf-8",
            )
            (candidate / "rank0.jsonl").write_text(
                json.dumps({**common, "optimizer_state": {"momentum": _summary(2.0)}}) + "\n",
                encoding="utf-8",
            )

            strict = compare_parameter_probes(
                root / "baseline",
                root / "candidate",
                {"max_abs": 0.0, "relative_l2": 0.0},
            )
            cross_topology = compare_parameter_probes(
                root / "baseline",
                root / "candidate",
                {"max_abs": 0.0, "relative_l2": 0.0},
                compare_optimizer_states=False,
            )

        self.assertEqual(strict["status"], "FAIL")
        self.assertEqual(cross_topology["status"], "PASS")
        self.assertEqual(cross_topology["skipped_optimizer_states"], 1)

    def test_parameter_tolerance_combination_is_explicit(self) -> None:
        """Allow either absolute or relative error only when requested."""
        baseline = {
            "layout": {"global_shape": [1]},
            "finite": True,
            "values": [0.0],
        }
        candidate = {
            "layout": {"global_shape": [1]},
            "finite": True,
            "values": [5.0e-4],
        }

        strict = case_compare._numeric_summary_delta(  # pylint: disable=protected-access
            baseline,
            candidate,
            {"max_abs": 1.0e-3, "relative_l2": 1.0e-6},
        )
        either = case_compare._numeric_summary_delta(  # pylint: disable=protected-access
            baseline,
            candidate,
            {
                "max_abs": 1.0e-3,
                "relative_l2": 1.0e-6,
                "combination": "any",
            },
        )

        self.assertEqual(strict["status"], "FAIL")
        self.assertEqual(strict["tolerance_combination"], "all")
        self.assertEqual(either["status"], "PASS")
        self.assertEqual(either["tolerance_combination"], "any")

        with self.assertRaisesRegex(ValueError, "must be 'all' or 'any'"):
            case_compare._numeric_summary_delta(  # pylint: disable=protected-access
                baseline,
                candidate,
                {
                    "max_abs": 1.0e-3,
                    "relative_l2": 1.0e-6,
                    "combination": "invalid",
                },
            )

    def test_summary_parameter_probe_does_not_report_elementwise_error(self) -> None:
        """Do not label aggregate-statistic differences as tensor max_abs."""
        baseline = {
            "layout": {"global_shape": [10000]},
            "finite": True,
            "sum": 100.0,
            "l2": 20.0,
            "min": -1.0,
            "max": 1.0,
        }
        candidate = {
            "layout": {"global_shape": [10000]},
            "finite": True,
            "sum": 100.5,
            "l2": 20.1,
            "min": -1.1,
            "max": 1.2,
        }

        comparison = case_compare._numeric_summary_delta(  # pylint: disable=protected-access
            baseline,
            candidate,
            {
                "max_abs": 1.0e-6,
                "relative_l2": 1.0e-6,
                "summary": {"l2_norm_relative": 1.0e-2},
            },
        )

        self.assertEqual(comparison["status"], "PASS")
        self.assertEqual(comparison["comparison_mode"], "aggregate_summary")
        self.assertIsNone(comparison["max_abs"])
        self.assertIsNone(comparison["relative_l2"])
        self.assertAlmostEqual(comparison["l2_norm_relative_error"], 5.0e-3)
        self.assertEqual(comparison["aggregate_deltas"]["sum"], 0.5)

    def test_parameter_tolerance_is_profiled_by_acceptance_class(self) -> None:
        """Keep cross-topology BF16 limits from weakening strict comparisons."""
        acceptance = {
            "parameters": {
                "max_abs": 1.0e-6,
                "relative_l2": 1.0e-6,
                "combination": "all",
                "summary": {
                    "l2_norm_relative": 1.0e-6,
                    "l2_norm_max_abs": 1.0e-5,
                },
                "cross_topology": {
                    "max_abs": 1.0e-3,
                    "relative_l2": 2.0e-2,
                    "combination": "any",
                    "summary": {"l2_norm_relative": 2.0e-2},
                },
            },
        }

        strict = model_integration_cli._parameter_tolerance(  # pylint: disable=protected-access
            acceptance,
            "same_topology",
        )
        cross = model_integration_cli._parameter_tolerance(  # pylint: disable=protected-access
            acceptance,
            "cross_topology",
        )

        self.assertEqual(strict["max_abs"], 1.0e-6)
        self.assertEqual(strict["combination"], "all")
        self.assertEqual(cross["max_abs"], 1.0e-3)
        self.assertEqual(cross["combination"], "any")
        self.assertEqual(cross["summary"]["l2_norm_relative"], 2.0e-2)
        self.assertEqual(cross["summary"]["l2_norm_max_abs"], 1.0e-5)

    def test_scalar_relative_tolerance_is_explicit_and_scale_aware(self) -> None:
        """Allow a large absolute delta only through a declared relative bound."""
        common = {
            "step": 7,
            "global_input_sha256": "same-input",
            "lr": 1.0e-5,
            "grad_norm_post_clip": 1.0,
        }
        baseline = [{**common, "loss": 11.0, "grad_norm_pre_clip": 3600.0}]
        candidate = [{**common, "loss": 11.044, "grad_norm_pre_clip": 3613.0}]

        strict = compare_scalar_curves(
            baseline,
            candidate,
            {
                "loss_max_abs": 5.0e-3,
                "loss_max_rel": 5.0e-3,
                "norm_max_abs": 1.25e-1,
                "norm_max_rel": 5.0e-3,
            },
        )
        scale_aware = compare_scalar_curves(
            baseline,
            candidate,
            {
                "loss_max_abs": 5.0e-3,
                "loss_max_rel": 5.0e-3,
                "norm_max_abs": 1.25e-1,
                "norm_max_rel": 5.0e-3,
                "combination": "any",
            },
        )

        self.assertEqual(strict["status"], "FAIL")
        self.assertEqual(scale_aware["status"], "PASS")
        self.assertAlmostEqual(scale_aware["steps"][0]["loss_max_rel"], 0.004)
        self.assertAlmostEqual(
            scale_aware["steps"][0]["norm_max_rel"],
            13.0 / 3600.0,
        )

    def test_scalar_comparison_is_blocked_for_different_inputs(self) -> None:
        """Do not convert a data-replay mismatch into a numerical failure."""
        baseline = [{
            "step": 2,
            "loss": 1.0,
            "grad_norm_pre_clip": 2.0,
            "grad_norm_post_clip": 1.0,
            "global_input_sha256": "baseline-input",
            "lr": 1.0e-5,
        }]
        candidate = [{
            **baseline[0],
            "loss": 1.5,
            "global_input_sha256": "candidate-input",
        }]

        comparison = compare_scalar_curves(
            baseline,
            candidate,
            {"loss_max_abs": 0.0, "norm_max_abs": 0.0},
        )

        self.assertEqual(comparison["status"], "BLOCKED")
        self.assertEqual(comparison["steps"][0]["status"], "BLOCKED")
        self.assertFalse(comparison["steps"][0]["input_identity"])

    def test_checkpoint_coverage_requires_explicit_omission_policy(self) -> None:
        """Classify an intentionally initialized target instead of accepting strict=False."""
        model = nn.Linear(2, 2)
        model._hp_checkpoint_load_report = SimpleNamespace(  # pylint: disable=protected-access
            loaded_keys=("weight",),
            missing_keys=("bias",),
            unexpected_keys=(),
        )
        _, findings = build_checkpoint_coverage(model)
        self.assertEqual({finding.code for finding in findings}, {"HP-CKPT-001"})

        coverage, findings = build_checkpoint_coverage(
            model,
            CheckpointValidationSpec(training_only_target_patterns=("bias",)),
        )
        self.assertFalse(findings)
        self.assertEqual(coverage["intentional_training_only"], 1)

    def test_shared_state_trace_records_producer_consumer_identity(self) -> None:
        """Expose CSA2 cross-layer dependencies without retaining tensor values."""
        original_publish = SharedCompressedAttentionState.publish_compressed_kv
        token = begin_shared_state_trace()
        self.assertIsNot(
            SharedCompressedAttentionState.publish_compressed_kv,
            original_publish,
        )
        state = SharedCompressedAttentionState()
        value = torch.ones(1)
        state.publish_compressed_kv(1, value)
        self.assertIs(state.require_compressed_kv(1, 2), value)
        events = finish_shared_state_trace(token)

        self.assertEqual([event["action"] for event in events], ["publish", "consume"])
        self.assertEqual(len({event["state_id"] for event in events}), 1)
        self.assertIs(
            SharedCompressedAttentionState.publish_compressed_kv,
            original_publish,
        )

    def test_shared_state_trace_uses_unique_ids_without_retaining_states(self) -> None:
        """Separate micro-batches without retaining their activation state."""
        token = begin_shared_state_trace()
        state = SharedCompressedAttentionState()
        state_reference = weakref.ref(state)
        state.publish_compressed_kv(1, torch.ones(1))

        del state
        gc.collect()
        self.assertIsNone(state_reference())

        next_state = SharedCompressedAttentionState()
        next_state.publish_compressed_kv(1, torch.ones(1))
        events = finish_shared_state_trace(token)
        self.assertEqual(len({event["state_id"] for event in events}), 2)

    def test_shared_state_trace_has_a_diagnostic_memory_bound(self) -> None:
        """Fail explicitly instead of retaining an unbounded diagnostic trace."""
        token = begin_shared_state_trace()
        try:
            state = SharedCompressedAttentionState()
            with mock.patch.object(
                shared_state_trace_module,
                "_MAX_SHARED_STATE_TRACE_EVENTS",
                1,
            ):
                state.publish_compressed_kv(1, torch.ones(1))
                with self.assertRaisesRegex(RuntimeError, "HP-STATE-003"):
                    state.require_compressed_kv(1, 2)
        finally:
            finish_shared_state_trace(token)

    def test_trainer_reuses_clip_result_for_runtime_diagnostics(self) -> None:
        """Derive the post-clip norm without a second full-model reduction."""
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.model = nn.Linear(2, 2)
        trainer.config = SimpleNamespace(training=SimpleNamespace(max_grad_norm=1.0))
        trainer.fsdp_runtime_diagnostics = None
        trainer.model_integration = mock.Mock()
        trainer.model_integration.runtime_enabled = True
        with mock.patch(
            "hyper_parallel.trainer.base.clip_grad_norm_",
            return_value=torch.tensor(2.5),
        ) as clip:
            grad_norm = trainer.prepare_optimizer_step()

        self.assertEqual(float(grad_norm), 2.5)
        clip.assert_called_once_with(trainer.model, 1.0)
        recorded_norm = trainer.model_integration.after_clip.call_args.args[0]
        self.assertAlmostEqual(recorded_norm, 1.0, places=5)

    def test_trainer_measures_norm_once_when_clipping_is_disabled(self) -> None:
        """Compute the diagnostic norm once and reuse it as the step metric."""
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.model = nn.Linear(2, 2)
        trainer.config = SimpleNamespace(training=SimpleNamespace(max_grad_norm=0.0))
        trainer.fsdp_runtime_diagnostics = None
        trainer.model_integration = mock.Mock()
        trainer.model_integration.runtime_enabled = True
        with mock.patch(
            "hyper_parallel.trainer.base.clip_grad_norm_",
            return_value=torch.tensor(2.5),
        ) as clip:
            grad_norm = trainer.prepare_optimizer_step()

        self.assertEqual(float(grad_norm), 2.5)
        clip.assert_called_once_with(trainer.model, float("inf"))
        trainer.model_integration.after_clip.assert_called_once_with(grad_norm)

    def test_performance_summary_keeps_memory_out_of_precision_status(self) -> None:
        """Report allocator maxima independently from scalar precision gates."""
        summary = summarize_performance(
            [
                {"step_time_seconds": 2.0, "tokens_per_second": 4.0},
                {"step_time_seconds": 1.0, "tokens_per_second": 8.0},
            ],
            warmup_steps=1,
            memory_rows=[
                {"peak_memory_allocated_bytes": 10, "peak_memory_reserved_bytes": 20},
                {"peak_memory_allocated_bytes": 30, "peak_memory_reserved_bytes": 40},
            ],
        )

        self.assertEqual(summary["status"], "MEASURED")
        self.assertEqual(summary["peak_memory_allocated_bytes"], 30)

    def test_cli_inspect_then_check_uses_evidence_state_machine(self) -> None:
        """Run local-only inspection and final-tree model integration through the public CLI."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "evidence"
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                "schema_version: 1\n"
                "model:\n"
                "  adapter: example\n"
                "  builder:\n"
                "    _target_: torch.nn.Linear\n"
                "    in_features: 2\n"
                "    out_features: 2\n"
                f"output_dir: {output_dir}\n",
                encoding="utf-8",
            )

            inspect_status = model_integration_main(
                ["inspect", "--manifest", str(manifest_path)]
            )
            check_status = model_integration_main(
                ["check", "--manifest", str(manifest_path)]
            )
            report_status = model_integration_main(
                ["report", "--output-dir", str(output_dir)]
            )

            invalid_manifest = root / "invalid_manifest.yaml"
            invalid_manifest.write_text(
                "schema_version: 1\nmodel:\n  adapter: example\n  builder: {}\n"
                f"output_dir: {root / 'invalid_evidence'}\n",
                encoding="utf-8",
            )
            invalid_status = model_integration_main(
                ["check", "--manifest", str(invalid_manifest)]
            )

            self.assertEqual(inspect_status, 0)
            self.assertEqual(check_status, 0)
            self.assertEqual(report_status, 2)
            self.assertEqual(invalid_status, 2)
            state = json.loads(
                (output_dir / "integration_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["state"], "STRUCTURE_VALIDATED")
            self.assertEqual(
                (output_dir / "summary.md").read_text(encoding="utf-8").splitlines()[0],
                "BLOCKED",
            )

    def test_case_launcher_runs_from_repository_root(self) -> None:
        """Keep checked-in example module and YAML paths resolvable."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                "schema_version: 1\nmodel:\n  adapter: example\nlauncher: {}\n",
                encoding="utf-8",
            )
            manifest = load_manifest(manifest_path)
            case = SimpleNamespace(
                name="baseline",
                kind="topology",
                compare_to="baseline",
                topology={},
                metadata={},
            )
            launch = {
                "phase": "train",
                "argv": ["python", "-c", "pass"],
                "evidence_dir": str(root / "case"),
            }
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            with mock.patch(
                "hyper_parallel.tools.model_integration.cli.subprocess.run",
                return_value=completed,
            ) as run:
                returncode, error = model_integration_cli._execute_case_launch(  # pylint: disable=protected-access
                    manifest,
                    case,
                    launch,
                    root / "resume",
                )

        self.assertEqual((returncode, error), (0, None))
        self.assertEqual(
            run.call_args.kwargs["cwd"],
            find_repository_root(),
        )

    def test_case_environment_isolates_inherited_hccl_port(self) -> None:
        """A failed NPU launch must not poison the next matrix case port."""
        case = SimpleNamespace(
            name="combined-1",
            kind="topology",
            compare_to="baseline",
        )
        with mock.patch.dict(os.environ, {"HCCL_IF_BASE_PORT": "62500"}):
            first = model_integration_cli._case_environment(  # pylint: disable=protected-access
                case,
                "train",
                Path("case-1"),
                Path("resume-1"),
                port_slot=3,
                port_stride=32,
            )
            second = model_integration_cli._case_environment(  # pylint: disable=protected-access
                case,
                "restore",
                Path("case-2"),
                Path("resume-2"),
                port_slot=4,
                port_stride=32,
            )

        self.assertEqual(first["HCCL_IF_BASE_PORT"], "62596")
        self.assertEqual(second["HCCL_IF_BASE_PORT"], "62628")

    def test_parameter_probes_persist_only_canonical_rank(self) -> None:
        """Global probe summaries are not duplicated into every rank file."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = EvidenceStore(temporary_directory)
            model = nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            rank_one = tensor_probes.ParameterProbeRecorder(
                model,
                optimizer,
                store,
                rank=1,
            )
            rank_one.begin_step(1)
            self.assertFalse(store.path("parameter_probes/rank1.jsonl").exists())

            rank_zero = tensor_probes.ParameterProbeRecorder(
                model,
                optimizer,
                store,
                rank=0,
            )
            rank_zero.begin_step(1)
            self.assertTrue(store.path("parameter_probes/rank0.jsonl").is_file())

    def test_tensor_summary_allows_rank_local_lazy_state_absence(self) -> None:
        """Keep probe collectives aligned when one rank has no optimizer state."""
        def _gather(rank_summaries, local_summary):
            rank_summaries[:] = [local_summary, None]

        with (
            mock.patch.object(tensor_probes.dist, "is_initialized", return_value=True),
            mock.patch.object(tensor_probes.dist, "get_rank", return_value=0),
            mock.patch.object(tensor_probes.dist, "get_world_size", return_value=2),
            mock.patch.object(
                tensor_probes.dist,
                "all_gather_object",
                side_effect=_gather,
            ),
        ):
            summary = tensor_probes._tensor_summary(  # pylint: disable=protected-access
                "exp_avg",
                torch.ones(2),
            )

        self.assertIsNotNone(summary)
        self.assertEqual(summary["missing_ranks"], [1])
        self.assertTrue(summary["finite"])

    def test_optimizer_state_names_use_global_union(self) -> None:
        """Run identical state-summary collectives despite lazy local keys."""
        def _gather(rank_names, local_names):
            rank_names[:] = [local_names, ("exp_avg_sq",)]

        with (
            mock.patch.object(tensor_probes.dist, "is_initialized", return_value=True),
            mock.patch.object(tensor_probes.dist, "get_world_size", return_value=2),
            mock.patch.object(
                tensor_probes.dist,
                "all_gather_object",
                side_effect=_gather,
            ),
        ):
            names = tensor_probes._global_tensor_state_names(  # pylint: disable=protected-access
                ("exp_avg",),
            )

        self.assertEqual(names, ("exp_avg", "exp_avg_sq"))

    def test_validate_manifest_builds_standard_trainer_launches(self) -> None:
        """Keep user validation input limited to recipe and experiment choices."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            manifest_path = root / "validation.yaml"
            (root / "trainer.yaml").write_text("training: {}\n", encoding="utf-8")
            manifest_path.write_text(
                "schema_version: 1\n"
                "model:\n  adapter: example\n"
                "launcher:\n"
                "  module: examples.training_demo.train_text\n"
                "  config: trainer.yaml\n"
                "matrix:\n"
                "  devices: 16\n"
                "  steps: 10\n"
                "  resume_split_step: 5\n"
                "  shared_initial_checkpoint: true\n"
                "  baseline: {tp: 2, cp: 1, ep: 16, fsdp: 8, "
                "recompute: {layer_indices: [0, 1, 2, 3]}}\n"
                "  same_topology_resume: true\n",
                encoding="utf-8",
            )
            manifest = load_manifest(manifest_path)
            manifest.validate_for("validate")
            topology = {
                "tp": 2,
                "cp": 1,
                "ep": 16,
                "fsdp": 8,
                "sequence_parallel": True,
                "recompute": {"layer_indices": [0, 1, 2, 3]},
            }
            ordinary_case = SimpleNamespace(
                name="baseline",
                kind="topology",
                compare_to="baseline",
                topology=topology,
                metadata={},
            )
            candidate_case = SimpleNamespace(
                name="axis-fsdp_example",
                kind="topology",
                compare_to="baseline",
                topology={**topology, "fsdp": 4},
                metadata={},
            )
            resume_case = SimpleNamespace(
                name="same_topology_resume",
                kind="resume",
                compare_to="baseline",
                topology=topology,
                metadata={"resume": "same_topology"},
            )
            cross_resume_case = SimpleNamespace(
                name="cross_topology_resume",
                kind="resume",
                compare_to="baseline",
                topology={**topology, "tp": 1, "fsdp": 16},
                metadata={
                    "resume": "cross_topology",
                    "prepare_topology": topology,
                },
            )
            validate_case = SimpleNamespace(
                name="validate",
                kind="production_validate",
                compare_to="baseline",
                topology={**topology, "validate_placement": True},
                metadata={},
            )

            ordinary = model_integration_cli._resolve_case_launches(  # pylint: disable=protected-access
                manifest,
                ordinary_case,
                root / "ordinary",
            )
            candidate = model_integration_cli._resolve_case_launches(  # pylint: disable=protected-access
                manifest,
                candidate_case,
                root / "candidate",
            )
            resume = model_integration_cli._resolve_case_launches(  # pylint: disable=protected-access
                manifest,
                resume_case,
                root / "resume",
            )
            cross_resume = model_integration_cli._resolve_case_launches(  # pylint: disable=protected-access
                manifest,
                cross_resume_case,
                root / "cross-resume",
            )
            validate = model_integration_cli._resolve_case_launches(  # pylint: disable=protected-access
                manifest,
                validate_case,
                root / "validate",
            )

        self.assertEqual([phase for phase, _, _ in ordinary], ["initialize", "train"])
        self.assertEqual([phase for phase, _, _ in resume], ["prepare", "restore"])
        initialize_argv = ordinary[0][1]
        ordinary_argv = ordinary[1][1]
        self.assertIn(str(root / "trainer.yaml"), ordinary_argv)
        self.assertIn("--nproc_per_node=16", ordinary_argv)
        self.assertIn("--accelerator.tp_size=2", ordinary_argv)
        self.assertIn("--accelerator.ep_size=16", ordinary_argv)
        self.assertIn("--fsdp_config.dp_shard_size=8", ordinary_argv)
        self.assertIn(
            "--activation_checkpoint.selection={'source': "
            "'model_adapter_safe_regions', 'layer_indices': [0, 1, 2, 3]}",
            ordinary_argv,
        )
        self.assertIn("--training.train_iters=10", ordinary_argv)
        self.assertIn("--training.lr_scheduler_iters=10", ordinary_argv)
        self.assertIn("--training.train_iters=1", initialize_argv)
        initial_checkpoint = manifest.output_dir / "cases/baseline/initial_checkpoint/global_step_1"
        self.assertIn(f"--checkpoint.restore_from={initial_checkpoint}", ordinary_argv)
        self.assertIn("--checkpoint.restore_dataloader_state=False", ordinary_argv)
        self.assertIn(
            f"--checkpoint.restore_from={initial_checkpoint}",
            candidate[0][1],
        )
        self.assertIn("--checkpoint.restore_dataloader_state=False", candidate[0][1])
        self.assertIn(f"--checkpoint.restore_from={initial_checkpoint}", resume[0][1])
        self.assertIn("--checkpoint.restore_dataloader_state=False", resume[0][1])
        self.assertIn("--training.train_iters=5", resume[0][1])
        self.assertIn("--training.lr_scheduler_iters=10", resume[0][1])
        self.assertIn(
            f"--checkpoint.restore_from={root / 'resume/resume_checkpoint/global_step_5'}",
            resume[1][1],
        )
        self.assertNotIn("--checkpoint.restore_dataloader_state=False", resume[1][1])
        self.assertIn("--accelerator.tp_size=2", cross_resume[0][1])
        self.assertIn("--fsdp_config.dp_shard_size=8", cross_resume[0][1])
        self.assertIn("--accelerator.tp_size=1", cross_resume[1][1])
        self.assertIn("--fsdp_config.dp_shard_size=16", cross_resume[1][1])
        self.assertIn("--model.validate_placement=True", validate[0][1])

    def test_terminal_state_rejects_handoff_with_structured_error(self) -> None:
        """Tell users to select a new evidence directory after a terminal run."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            handoff_path = root / "handoff.yaml"
            handoff_path.write_text("status: PASS\n", encoding="utf-8")
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                "schema_version: 1\nmodel:\n  adapter: example\n"
                f"integration_handoff: {handoff_path}\n",
                encoding="utf-8",
            )
            manifest = load_manifest(manifest_path)
            store = EvidenceStore(root / "evidence")
            store.update_state(IntegrationState.FAILED)

            with self.assertRaisesRegex(ManifestError, "new output directory"):
                model_integration_cli._require_precision_handoff(  # pylint: disable=protected-access
                    store,
                    manifest,
                )

    def test_precision_handoff_imports_report_evidence(self) -> None:
        """Keep split gate and matrix runs reportable as one evidence bundle."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gate_dir = root / "gate"
            matrix_dir = root / "matrix"
            for relative_path, payload in (
                ("check/findings.json", {"status": "PASS", "findings": []}),
                ("module_parity/comparison.json", {"status": "PASS", "cases": []}),
                ("checkpoint/coverage.json", {"status": "PASS", "entries": []}),
            ):
                path = gate_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
            handoff_path = gate_dir / "integration_handoff.yaml"
            handoff_path.write_text(
                "schema_version: 1\n"
                "family: example\n"
                "status: PASS\n"
                "structure_findings: check/findings.json\n"
                "module_parity: module_parity/comparison.json\n"
                "checkpoint_coverage: checkpoint/coverage.json\n",
                encoding="utf-8",
            )
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                "schema_version: 1\n"
                "model:\n  adapter: example\n"
                f"integration_handoff: {handoff_path}\n",
                encoding="utf-8",
            )
            manifest = load_manifest(manifest_path, output_dir=matrix_dir)
            store = EvidenceStore(matrix_dir)

            model_integration_cli._require_precision_handoff(  # pylint: disable=protected-access
                store,
                manifest,
            )

            self.assertEqual(
                json.loads((matrix_dir / "check/findings.json").read_text(encoding="utf-8"))["status"],
                "PASS",
            )
            self.assertEqual(
                json.loads(
                    (matrix_dir / "module_parity/comparison.json").read_text(encoding="utf-8")
                )["status"],
                "PASS",
            )
            self.assertEqual(
                json.loads(
                    (matrix_dir / "checkpoint/coverage.json").read_text(encoding="utf-8")
                )["status"],
                "PASS",
            )


if __name__ == "__main__":
    unittest.main()
