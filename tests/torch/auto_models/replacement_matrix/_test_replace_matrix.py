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
"""Small, asset-free NPU training cases for public replacement actions."""

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2MoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.replacement import (
    apply_module_replacements,
    compile_module_replacements,
)
from hyper_parallel.models._transformers.model_builder import (
    apply_model_init_dtype,
    instantiate_infrastructure,
)
from hyper_parallel.trainer.config import (
    TrainerConfig,
    entries_to_module_replacements,
    entries_to_plan_overrides,
    normalize_distributed_setup_overrides,
)
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.runtime.distributed import (
    create_distributed_setup_from_config,
    destroy_process_group,
    initialize_distributed,
)
from hyper_parallel.trainer.runtime.device import get_device_type
from tests.torch.auto_models.optimizer import _test_muon_checkpoint_resume as qwen2_case


_YAML = Path(__file__).with_name("test_yamls") / "qwen3_moe_rms_dp2.yaml"
_LLAMA_YAML = Path(__file__).with_name("test_yamls") / "llama_rms_dp2.yaml"
_QWEN2_COMBINED_YAML = Path(__file__).with_name("test_yamls") / "qwen2_moe_module_forward_dp4_cp2_ep2.yaml"
_STEPS = 10
_TOLERANCE = {"atol": 1e-3, "rtol": 1e-3}
_BF16_TOLERANCE = {"atol": 1e-2, "rtol": 1e-2}


def _qwen3_config() -> Qwen3MoeConfig:
    """Build a two-layer Qwen3-MoE with learned routing and no external assets."""
    return Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _llama_config() -> LlamaConfig:
    """Build a two-layer dense Llama with no external assets."""
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _global_tokens(step: int) -> torch.Tensor:
    """Replay the same global batch for native and replacement candidates."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(27182 + step)
    return torch.randint(0, 128, (4, 16), generator=generator)


def _train_qwen3_step(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global mean loss and pre-clip gradient norm after one update."""
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=tokens, labels=tokens).loss
    if loss.dtype != torch.float32:
        raise AssertionError(f"Qwen3-MoE loss must be FP32, got {loss.dtype}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    optimizer.step()
    global_loss = loss.detach().clone()
    dist.all_reduce(global_loss)
    global_loss /= dist.get_world_size()
    return global_loss, grad_norm


def _check_rms_parity(
    native_norm: torch.nn.Module, replaced_norm: torch.nn.Module,
    tolerance: dict[str, float],
) -> None:
    """Compare one native/replacement RMSNorm output and input/weight gradients."""
    native_input = torch.randn(
        2, 4, 64, device=native_norm.weight.device,
        dtype=native_norm.weight.dtype, requires_grad=True,
    )
    replaced_input = native_input.detach().clone().requires_grad_(True)
    native_output = native_norm(native_input)
    replaced_output = replaced_norm(replaced_input)
    torch.testing.assert_close(replaced_output, native_output, **tolerance)
    native_gradients = torch.autograd.grad(native_output.sum(), (native_input, native_norm.weight))
    replaced_gradients = torch.autograd.grad(replaced_output.sum(), (replaced_input, replaced_norm.weight))
    for native_gradient, replaced_gradient in zip(native_gradients, replaced_gradients):
        torch.testing.assert_close(replaced_gradient, native_gradient, **tolerance)


def _build_qwen2_with_module_replacements(
    config: TrainerConfig, setup: DistributedSetup, device: torch.device,
) -> torch.nn.Module:
    """Apply module replacements before planning and assert their state contract."""
    torch.manual_seed(31415)
    model = Qwen2MoeForCausalLM(qwen2_case._build_model_config()).to(device)  # pylint: disable=protected-access
    apply_model_init_dtype(model, config.model_init_dtype)
    model.train()
    native_norm = model.model.layers[0].input_layernorm
    original_parameters = dict(model.named_parameters())
    rules = entries_to_module_replacements(config.plan_overrides)
    replacement_plan = compile_module_replacements(model, rules)
    if len(replacement_plan.targets) != 9:
        raise AssertionError(f"expected nine RMSNorm targets, got {len(replacement_plan.targets)}")
    apply_module_replacements(model, replacement_plan)
    _check_rms_parity(native_norm, model.model.layers[0].input_layernorm, _BF16_TOLERANCE)
    replaced_parameters = dict(model.named_parameters())
    if original_parameters.keys() != replaced_parameters.keys():
        raise AssertionError("Qwen2 RMSNorm replacement changed parameter FQNs")
    if any(replaced_parameters[name] is not parameter for name, parameter in original_parameters.items()):
        raise AssertionError("Qwen2 RMSNorm replacement changed parameter identity")
    planner, fsdp_manager = instantiate_infrastructure(distributed_setup=setup, device=device)
    mesh = setup.mesh_context
    if fsdp_manager is None or mesh.device_mesh is None:
        raise RuntimeError("Qwen2 replacement matrix requires FSDP and a device mesh")
    plan = planner.plan(
        model, mesh.device_mesh, tp_size=mesh.tp_size, cp_size=mesh.cp_size,
        ep_size=mesh.ep_size, sequence_parallel=mesh.sequence_parallel,
        loss_parallel=mesh.loss_parallel,
    )
    model, source_shard_info = apply_sharding_plan(model, plan, mesh)
    fsdp_manager.parallelize(model, source_shard_info)
    return model


def _run_paired_rms_training(
    *, yaml_path: Path, model_factory: Callable[[], torch.nn.Module], model_name: str,
) -> None:
    """Compare native and replaced modules for ten aligned DP2 updates."""
    config = parse_training_args([str(yaml_path)])
    initialize_distributed(backend=config.training.backend)
    try:
        if dist.get_world_size() != 2:
            raise AssertionError(f"{model_name} case requires two NPU ranks")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(get_device_type(), local_rank)
        torch.manual_seed(31415)
        native = model_factory().to(device)
        torch.manual_seed(31415)
        replaced = model_factory().to(device)
        source_parameters = dict(replaced.named_parameters())
        replaced.load_state_dict(native.state_dict())
        rules = entries_to_module_replacements(config.plan_overrides)
        plan = compile_module_replacements(replaced, rules)
        if len(plan.targets) != 5:
            raise AssertionError(f"expected five {model_name} RMSNorm targets, got {len(plan.targets)}")
        apply_module_replacements(replaced, plan)
        actual_parameters = dict(replaced.named_parameters())
        if actual_parameters.keys() != source_parameters.keys():
            raise AssertionError("RMSNorm replacement changed parameter FQNs")
        if any(actual_parameters[name] is not parameter for name, parameter in source_parameters.items()):
            raise AssertionError("RMSNorm replacement changed parameter identity")
        _check_rms_parity(
            native.model.layers[0].input_layernorm,
            replaced.model.layers[0].input_layernorm,
            _TOLERANCE,
        )
        native.train()
        replaced.train()
        native_ddp = DistributedDataParallel(native, device_ids=[local_rank])
        replaced_ddp = DistributedDataParallel(replaced, device_ids=[local_rank])
        native_optimizer = torch.optim.AdamW(native_ddp.parameters(), lr=1e-3)
        replaced_optimizer = torch.optim.AdamW(replaced_ddp.parameters(), lr=1e-3)
        for step in range(_STEPS):
            tokens = _global_tokens(step).chunk(2, dim=0)[local_rank].to(device)
            native_loss, native_norm = _train_qwen3_step(native_ddp, native_optimizer, tokens)
            replaced_loss, replaced_norm = _train_qwen3_step(replaced_ddp, replaced_optimizer, tokens)
            if not all(math.isfinite(float(value)) for value in (
                native_loss, native_norm, replaced_loss, replaced_norm
            )):
                raise AssertionError(f"non-finite {model_name} metrics at step {step + 1}")
            torch.testing.assert_close(replaced_loss, native_loss, **_TOLERANCE)
            torch.testing.assert_close(replaced_norm, native_norm, **_TOLERANCE)
            if dist.get_rank() == 0:
                print(
                    f"{model_name} step={step + 1} loss={float(replaced_loss):.8f} "
                    f"norm={float(replaced_norm):.8f}",
                    flush=True,
                )
        if dist.get_rank() == 0:
            print(f"PASS: {model_name} DP2 native/replaced 10-step loss and norm", flush=True)
    finally:
        destroy_process_group()


def test_qwen3_moe_rms_module_replacement() -> None:
    """Compare ten paired DP2 Qwen3-MoE updates with five RMSNorm replacements."""
    _run_paired_rms_training(
        yaml_path=_YAML,
        model_factory=lambda: Qwen3MoeForCausalLM(_qwen3_config()),
        model_name="Qwen3-MoE",
    )


def test_llama_rms_module_replacement() -> None:
    """Compare ten paired DP2 Llama updates with five RMSNorm replacements."""
    _run_paired_rms_training(
        yaml_path=_LLAMA_YAML,
        model_factory=lambda: LlamaForCausalLM(_llama_config()),
        model_name="Llama",
    )


def _run_qwen2_training(
    *, cp_size: int, ep_size: int, dp_shard_size: int, expected_world: int,
    config_path: Optional[Path] = None,
) -> None:
    """Run a ten-step Qwen2-MoE topology from one conditional-rule recipe."""
    config = parse_training_args([
        str(config_path or qwen2_case._TEST_YAML),  # pylint: disable=protected-access
        f"--accelerator.cp_size={cp_size}",
        f"--accelerator.ep_size={ep_size}",
        f"--fsdp_config.dp_shard_size={dp_shard_size}",
        f"--fsdp_config.edp_shard_size={2 if expected_world == 8 else 1}",
    ])
    active = entries_to_plan_overrides(
        config.plan_overrides, cp_size=cp_size, ep_size=ep_size,
    )
    if ("*.self_attn" in active) != (cp_size > 1):
        raise AssertionError("CP forward replacement activation does not match cp_size")
    if ("*.mlp" in active) != (ep_size > 1):
        raise AssertionError("EP forward replacement activation does not match ep_size")
    qwen2_case.enable_full_determinism(31415)
    initialize_distributed(backend=config.training.backend)
    try:
        if dist.get_world_size() != expected_world:
            raise AssertionError(f"expected {expected_world} ranks, got {dist.get_world_size()}")
        device = torch.device(get_device_type(), int(os.environ["LOCAL_RANK"]))
        setup = create_distributed_setup_from_config(config)
        normalize_distributed_setup_overrides(setup, config)
        if setup.mesh_context.cp_size != cp_size or setup.mesh_context.ep_size != ep_size:
            raise AssertionError("runtime mesh differs from the requested CP/EP topology")
        if config_path is None:
            model = qwen2_case._build_model(setup, device, config.model_init_dtype)  # pylint: disable=protected-access
        else:
            model = _build_qwen2_with_module_replacements(config, setup, device)
        optimizer = qwen2_case._build_optimizer(config, model)  # pylint: disable=protected-access
        for step in range(_STEPS):
            loss, norm = qwen2_case._train_step(model, optimizer, setup, step, device)  # pylint: disable=protected-access
            if not math.isfinite(loss) or not math.isfinite(norm) or norm <= 0:
                raise AssertionError(f"invalid Qwen2-MoE metrics at step {step + 1}: {loss}, {norm}")
            if dist.get_rank() == 0:
                print(f"qwen2 step={step + 1} loss={loss:.8f} norm={norm:.8f}", flush=True)
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"PASS: Qwen2-MoE CP{cp_size}/EP{ep_size} 10-step training", flush=True)
    finally:
        destroy_process_group()


def test_qwen2_moe_cp_only_forward_replacement() -> None:
    """Validate the attention action without expert dispatch."""
    _run_qwen2_training(cp_size=2, ep_size=1, dp_shard_size=1, expected_world=2)


def test_qwen2_moe_module_only_replacement() -> None:
    """Validate nine RMSNorm modules while both conditional actions are inactive."""
    _run_qwen2_training(
        cp_size=1, ep_size=1, dp_shard_size=2, expected_world=2,
        config_path=_QWEN2_COMBINED_YAML,
    )


def test_qwen2_moe_module_and_cp_forward_replacements() -> None:
    """Validate RMSNorm and CP attention actions while EP is inactive."""
    _run_qwen2_training(
        cp_size=2, ep_size=1, dp_shard_size=1, expected_world=2,
        config_path=_QWEN2_COMBINED_YAML,
    )


def test_qwen2_moe_module_and_ep_forward_replacements() -> None:
    """Validate RMSNorm and EP expert actions while CP is inactive."""
    _run_qwen2_training(
        cp_size=1, ep_size=2, dp_shard_size=2, expected_world=2,
        config_path=_QWEN2_COMBINED_YAML,
    )


def test_qwen2_moe_ep_only_forward_replacement() -> None:
    """Validate the expert action without context communication."""
    _run_qwen2_training(cp_size=1, ep_size=2, dp_shard_size=2, expected_world=2)


def test_qwen2_moe_cp_ep_four_ranks() -> None:
    """Validate concurrent attention and expert actions on four ranks."""
    _run_qwen2_training(cp_size=2, ep_size=2, dp_shard_size=2, expected_world=4)


def test_qwen2_moe_cp_ep_forward_replacements() -> None:
    """Validate concurrent attention and expert actions on eight ranks."""
    _run_qwen2_training(cp_size=2, ep_size=2, dp_shard_size=2, expected_world=8)


def test_qwen2_moe_module_and_forward_replacements() -> None:
    """Validate module plus both forward actions in one eight-rank training run."""
    _run_qwen2_training(
        cp_size=2, ep_size=2, dp_shard_size=2, expected_world=8,
        config_path=_QWEN2_COMBINED_YAML,
    )
