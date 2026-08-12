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
"""Two-rank loss and gradient byte regression for Qwen3.5 training TP policy."""

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.models.qwen3_5.model import Qwen3_5Config, Qwen3_5ForCausalLM
from hyper_parallel.models.qwen3_5.parallelize import parallelize_qwen3_5_tp
from tests.torch.utils import init_dist

_INIT_SEED = 1234
_DATA_SEED = 2026
_NUM_STEPS = 3
_EXPECTED_GRAD_RESULTS = {
    0: (
        ("ffeb0dbff3dd5e46b5e14d50847db09420e69d6f41cc7a19a65156334839dadd", "1ec0db42"),
        ("9575a9fada33e510c5ff12580426d628e641e86a165bf6fba9c9eccc31f3693f", "94f32643"),
        ("0310a2e9a4c76b5b818beca4da34685af083077811df62ffdb421f6caafdea82", "b3ac0243"),
    ),
    1: (
        ("d1eeb1bce31b64ec1342e846df6bcb0b43f4ca0ace3da6c16269209aa29619ad", "1ec0db42"),
        ("4bf9807ab523aebebcacd38776d11ce824f7a0f536c94c35e8f5512fbe88a48f", "94f32643"),
        ("31f12cfab14dfa0e5cadde7f3d1717ebddb32ae4a6fde84a972ba2a5fb26793d", "b3ac0243"),
    ),
}
_EXPECTED_LOSS_BYTES = ("969ce442", "6b91ed42", "27a2e542")


def _build_model(device: torch.device) -> Qwen3_5ForCausalLM:
    """Build the deterministic tiny hybrid model used for TP policy regression."""
    torch.manual_seed(_INIT_SEED)
    torch.npu.manual_seed(_INIT_SEED)
    config = Qwen3_5Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        partial_rotary_factor=0.375,
        mrope_section=[2, 2, 2],
        linear_num_value_heads=4,
        linear_num_key_heads=4,
        linear_value_head_dim=16,
        linear_key_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    return Qwen3_5ForCausalLM(config).to(device=device, dtype=torch.float32)


def _build_batch(step: int, device: torch.device) -> torch.Tensor:
    """Return one deterministic replicated token batch."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_DATA_SEED + step)
    tokens = torch.randint(0, 128, (2, 8), generator=generator, dtype=torch.long)
    return tokens.to(device=device)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the rank-local value of a Tensor or DTensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return deterministic raw bytes for one local tensor."""
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def _gradient_digest(model: Qwen3_5ForCausalLM) -> tuple[str, torch.Tensor]:
    """Hash every local gradient and return its local float32 square sum."""
    digest = hashlib.sha256()
    square_sum = torch.zeros((), dtype=torch.float32, device="npu")
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"Expected gradient for parameter {name}")
        local_gradient = _local_tensor(parameter.grad)
        digest.update(name.encode("utf-8"))
        digest.update(str(local_gradient.dtype).encode("ascii"))
        digest.update(str(tuple(local_gradient.shape)).encode("ascii"))
        digest.update(_tensor_bytes(local_gradient))
        square_sum += local_gradient.detach().float().square().sum()
    return digest.hexdigest(), square_sum


def _write_report(report: dict, rank: int) -> None:
    """Write one rank's deterministic baseline when requested by the caller."""
    report_path = os.environ.get("QWEN35_TP_REPORT")
    if not report_path:
        return
    path = Path(f"{report_path}.rank{rank}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_qwen3_5_training_tp_policy_bit_exact() -> None:
    """Assert deterministic TP+SP loss and gradient bytes on two NPU ranks."""
    os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    os.environ.setdefault("HCCL_DETERMINISTIC", "true")
    os.environ.setdefault("PYTHONHASHSEED", str(_INIT_SEED))
    rank, device_id = init_dist()
    torch.use_deterministic_algorithms(True)
    if dist.get_world_size() != 2:
        raise ValueError("Qwen3.5 TP policy regression requires exactly two ranks")
    device = torch.device(f"npu:{device_id}")
    mesh = init_device_mesh("npu", (2,), mesh_dim_names=("tp",))
    model = _build_model(device)
    parallelize_qwen3_5_tp(model, mesh)

    steps = []
    for step in range(_NUM_STEPS):
        model.zero_grad(set_to_none=True)
        tokens = _build_batch(step, device)
        loss = model(input_ids=tokens, labels=tokens)["loss"]
        loss.backward()
        grad_digest, square_sum = _gradient_digest(model)
        dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
        grad_norm = square_sum.sqrt()
        steps.append(
            {
                "grad_digest": grad_digest,
                "grad_norm": float(grad_norm.cpu()),
                "grad_norm_bytes": _tensor_bytes(grad_norm).hex(),
                "loss": float(loss.detach().cpu()),
                "loss_bytes": _tensor_bytes(loss).hex(),
            }
        )

    for step, (actual, expected_grad) in enumerate(zip(steps, _EXPECTED_GRAD_RESULTS[rank])):
        assert (actual["grad_digest"], actual["grad_norm_bytes"]) == expected_grad
        assert actual["loss_bytes"] == _EXPECTED_LOSS_BYTES[step]

    _write_report(
        {
            "rank": rank,
            "steps": steps,
            "world_size": dist.get_world_size(),
        },
        rank,
    )
