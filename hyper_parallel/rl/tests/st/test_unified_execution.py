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
"""Explicit single-NPU launch for native unified attention numerical acceptance."""

import os
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_unified_attention(tmp_path: Path) -> None:
    """Run native operator backward and tiny HP/HF optimizer steps on an explicitly selected NPU."""
    device = os.environ.get("RL_UNIFIED_ATTENTION_ST_DEVICE")
    if device is None:
        pytest.skip("Set RL_UNIFIED_ATTENTION_ST_DEVICE to an available logical NPU index")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_unified_attention.py")), "--device", device,
        "--output", str(tmp_path / "attention.json")], check=True,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_unified_attention_backend(tmp_path: Path) -> None:
    """Verify real paged-cache writes and updated-metadata graph replay against FP64."""
    device = os.environ.get("RL_UNIFIED_ATTENTION_ST_DEVICE")
    model = os.environ.get("RL_UNIFIED_ST_MODEL")
    if device is None or not model:
        pytest.skip("Set RL_UNIFIED_ATTENTION_ST_DEVICE and RL_UNIFIED_ST_MODEL for the NPU backend test")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_unified_attention_backend.py")),
         "--device", device, "--model", model, "--output", str(tmp_path / "backend.json")], check=True,
    )


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_pik_gloo() -> None:
    """Run the distributed CPU forward and gradient checks outside the unit suite."""
    torchrun_case(str(Path(__file__).with_name("_pik_gloo.py")), "test_pik_gloo", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pik_npu() -> None:
    """Launch only when devices are explicitly reserved by the caller."""
    if os.environ.get("RL_PIK_ST") != "1":
        pytest.skip("set RL_PIK_ST=1 after reserving NPU devices")
    world = int(os.environ.get("RL_PIK_WORLD_SIZE", "2"))
    torchrun_case(str(Path(__file__).with_name("_pik.py")), "test_pik_npu", num_proc=world)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_qk_norm_rope() -> None:
    """Compare the bound rollout Q/K to HP before and after a captured weight update."""
    if not os.environ.get("RL_QK_ST_DEVICE"):
        pytest.skip("Set RL_QK_ST_DEVICE to an allocated Ascend device")

    # Keep accelerator and model dependencies behind the explicit device opt-in.
    # pylint: disable=import-outside-toplevel
    import torch
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm

    from hyper_parallel.components.modules.attention.grouped_query_attention.grouped_query_attention import (
        GroupedQueryAttentionModule,
    )
    from rl.unified_execution.vllm.attention.grouped_query_attention import project_qkv_with_fused_rope
    from rl.unified_execution.vllm.qwen3 import optimize_rollout_projections
    # pylint: enable=import-outside-toplevel

    class InheritedNorm(Qwen3RMSNorm):
        """A new model family may inherit the same normalization forward."""

    def expose_qk(module, query, key, value, mask, **kwargs):
        del module, value, mask, kwargs
        return query.transpose(1, 2).contiguous(), key

    torch.npu.set_device(int(os.environ["RL_QK_ST_DEVICE"]))
    torch.npu.set_deterministic_level(3)
    config = Qwen3Config(hidden_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=128)
    attention = GroupedQueryAttentionModule(module=Qwen3Attention(config, 0))
    attention.attention_interface = expose_qk
    attention.q_norm = InheritedNorm(128, eps=attention.q_norm.variance_epsilon)
    attention.o_proj = torch.nn.Identity()
    attention.to("npu", torch.bfloat16).eval()
    reference = deepcopy(attention)
    optimize_rollout_projections(attention)
    assert attention.forward.__func__ is GroupedQueryAttentionModule.forward
    assert attention.project_qkv.__func__ is project_qkv_with_fused_rope

    # Multiple KV groups and a partial token tile exercise the interleaved QKV layout.
    inputs = torch.randn(1, 19, 512, device="npu", dtype=torch.bfloat16)
    phase = torch.randn(1, 19, 64, device="npu").repeat(1, 1, 2)
    positions = (phase.cos().bfloat16(), phase.sin().bfloat16())
    with torch.inference_mode():
        attention(inputs, positions)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = attention(inputs, positions)
        inputs.normal_()
        attention.q_norm.weight.normal_()
        attention.k_norm.weight.normal_()
        graph.replay()
        reference.load_state_dict(attention.state_dict())
        expected = reference(inputs, positions)
        for result in (attention(inputs, positions), captured):
            for actual, wanted in zip(result, expected, strict=True):
                assert torch.equal(actual.contiguous().view(torch.int16), wanted.contiguous().view(torch.int16))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_unified_expert_parallel() -> None:
    """Exercise production dispatch, grouped experts, native graph replay and gradients."""
    devices = os.environ.get("RL_EP_ST_DEVICES", "").split(",")
    if devices == [""]:
        pytest.skip("Set RL_EP_ST_DEVICES to explicitly allocated idle NPU IDs")
    if not all(device.isdigit() for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("RL_EP_ST_DEVICES must contain distinct NPU IDs")
    ep_size = int(os.environ.get("RL_EP_ST_SIZE", len(devices)))
    if ep_size not in (2, 4, 8) or len(devices) % ep_size:
        raise ValueError("EP ST requires complete EP2, EP4 or EP8 groups")
    model = os.environ.get("RL_EP_ST_MODEL")
    if not model:
        pytest.skip("Set RL_EP_ST_MODEL to a local Qwen3-MoE model configuration")
    subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--local-addr=127.0.0.1",
        f"--nproc_per_node={len(devices)}", str(Path(__file__).with_name("_unified_expert_parallel.py")),
        "--ep", str(ep_size), "--model", model,
    ], env=dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=",".join(devices)), check=True, timeout=1800)
