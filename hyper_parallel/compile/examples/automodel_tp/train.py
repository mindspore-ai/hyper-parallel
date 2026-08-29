#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""TP + FSDP graph-mode demo.

Combines automodel's TP sharding with the graph-mode FSDP pass:

  1. automodel ``MeshContext.build_meshs`` builds a ``(dp, cp, tp)`` mesh.
     The TP group is created here and the boundary forwards will hold the
     group object directly.
  2. ``ShardingPlanner.plan`` derives a TP ``ShardingPlan``.
  3. ``apply_sharding_plan(production)`` turns parameters into plain TP
     shards (``to_local``) and wraps module forwards with boundary
     redistribution — TP collectives are inserted on **activations**
     (colwise all-reduce on exit, rowwise input gather, etc.).
  4. FSDP2Manager is NOT used (no ``fully_shard`` hooks). Instead the
     TP-sharded model is handed to ``GraphTrainer``:
       - the automodel mesh is reused — only the dp sub-mesh is registered
         under ``"fsdp"`` (the TP group is already wired into the boundary
         forwards);
       - ``trace_model_graph`` traces a joint fwd+bwd FX graph; parameters
         are static inputs whose shape is the TP shard;
       - ``FSDPPass`` shards each already-TP-sharded parameter along dim 0
         (FSDP axis, orthogonal to the TP axis), inserts all_gather on the
         parameter placeholders (recovers the TP shard) and reduce_scatter
         on the gradient outputs;
       - ``AutoOverlapPass`` moves wait_tensor for comm/compute overlap.

The two communication domains are orthogonal: FSDP talks on parameters/grads
(the "fsdp" group), TP talks on activations (the "tp" group baked into the
boundary forwards). FSDPPass only sees the FX graph — it is TP-agnostic.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from hyper_parallel.compile import (  # pylint: disable=C0413
    GraphTrainer,
    ParallelConfig,
    ShardingPlan,
)
from hyper_parallel.auto_models.components.distributed.infrastructure import (  # pylint: disable=C0413
    MeshContext,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import (  # pylint: disable=C0413
    ShardingPlanner,
)
from hyper_parallel.auto_models.components.distributed.sharding_applier import (  # pylint: disable=C0413
    apply_sharding_plan,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TP + FSDP graph-mode demo (automodel TP-shard + FSDPPass)"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_distributed() -> str:
    """Init the process group; return the device type to build meshes on."""
    if not dist.is_initialized():
        if hasattr(torch, "npu") and torch.npu.is_available():
            dist.init_process_group(backend="hccl")
            torch.npu.set_device(dist.get_rank() % torch.npu.device_count())
            return "npu"
        dist.init_process_group(backend="gloo")
        return "cpu"
    return "cpu"


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    """Build a tiny Llama on ``device`` (real tensors — distribute_tensor
    needs non-meta parameters when apply_sharding_plan runs)."""
    # pylint: disable=C0415
    from transformers import LlamaConfig, LlamaForCausalLM

    model_cfg = LlamaConfig(
        vocab_size=cfg["model"]["vocab_size"],
        hidden_size=cfg["model"]["hidden_size"],
        intermediate_size=cfg["model"]["intermediate_size"],
        num_hidden_layers=cfg["model"]["num_hidden_layers"],
        num_attention_heads=cfg["model"]["num_attention_heads"],
        num_key_value_heads=cfg["model"]["num_key_value_heads"],
        max_position_embeddings=cfg["model"]["max_position_embeddings"],
        torch_dtype=cfg["model"].get("torch_dtype", "float32"),
    )
    model = LlamaForCausalLM(model_cfg).to(device)
    return model


def build_mesh_context(parallel_cfg: dict, device_type: str) -> MeshContext:
    """Build the automodel (dp, cp, tp) mesh + the dense FSDP sub-mesh."""
    tp = parallel_cfg["tp_size"]
    dp = parallel_cfg["dp_size"]
    world_size = dist.get_world_size()
    if tp * dp != world_size:
        raise ValueError(
            f"tp_size({tp}) * dp_size({dp}) != world_size({world_size}) — this "
            "demo assumes tp*dp == world_size (cp=pp=ep=1)."
        )
    ctx = MeshContext(
        tp_size=tp,
        dp_size=dp,
        cp_size=1,
        pp_size=1,
        ep_size=1,
        dp_replicate_size=1,
        dp_shard_size=dp,
        edp_shard_size=1,
        sequence_parallel=parallel_cfg.get("sequence_parallel", False),
        loss_parallel=parallel_cfg.get("loss_parallel", False),
    )
    ctx.build_meshs(device_type, world_size)
    return ctx


def train_fn(model, input_ids, labels):
    """Standard CE loss on the boundary-wrapped model forward."""
    logits = model(input_ids).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def inspect_graph(trainer: GraphTrainer) -> None:
    """Print FSDP and TP comm nodes in the compiled joint graph.

    FSDPPass tags its nodes with ``meta['comm_type']`` starting with ``fsdp_``.
    TP collectives live inside the boundary forwards and are plain
    ``call_function`` nodes whose op name mentions all_gather/all_reduce/
    reduce_scatter.
    """
    gm = trainer._joint_graph.graph_module  # pylint: disable=W0212
    fsdp, tp_nodes = [], []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        comm = node.meta.get("comm_type")
        target = getattr(node.target, "_qualified_op_name", None) or str(node.target)
        if comm and "fsdp" in str(comm):
            fsdp.append((node.name, comm))
        elif any(k in target for k in ("all_gather", "reduce_scatter", "all_reduce")):
            tp_nodes.append((node.name, target))
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("=" * 70)
        print(f"[graph] FSDP comm nodes: {len(fsdp)}")
        for n, c in fsdp[:6]:
            print(f"    {n}: {c}")
        print(f"[graph] TP(boundary) comm nodes: {len(tp_nodes)}")
        for n, t in tp_nodes[:6]:
            print(f"    {n}: {t}")
        print("=" * 70)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device_type = setup_distributed()
    device = (
        torch.device(device_type)
        if device_type == "cpu"
        else torch.device(device_type, dist.get_rank() % torch.npu.device_count())
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("=" * 70)
        print("TP + FSDP graph-mode demo")
        print(f"  world_size={world_size}, device={device_type}")
        print("=" * 70)

    # 1. Model on a real device (distribute_tensor needs real params).
    model = build_model(cfg, device)

    # 2. automodel mesh: (dp, cp, tp) + dense FSDP sub-mesh. TP group ready.
    mesh_ctx = build_mesh_context(cfg["parallel"], device_type)

    # 3. TP sharding plan + production apply. Parameters become plain TP
    #    shards; module forwards are wrapped with boundary TP collectives on
    #    activations. source_shard_info is for FSDP2 — we skip FSDP2 entirely.
    planner = ShardingPlanner()
    plan = planner.plan(
        model,
        mesh_ctx.device_mesh,
        tp_size=cfg["parallel"]["tp_size"],
        sequence_parallel=cfg["parallel"].get("sequence_parallel", False),
        loss_parallel=cfg["parallel"].get("loss_parallel", False),
    )
    model, _src = apply_sharding_plan(model, plan, mesh_ctx, validate_mode=False)
    if rank == 0:
        print(
            "[automodel] TP sharding applied — params are now plain TP shards, "
            "boundary forwards wrap TP collectives on activations"
        )

    # 4. FSDP wrap plan for the graph-mode FSDPPass (wrap everything).
    fsdp_plan = ShardingPlan()
    fsdp_plan.fsdp_wrap_pattern("*")

    # 5. GraphTrainer reuses the automodel mesh (tp group already created);
    #    it registers the dp sub-mesh as "fsdp" and back-fills fsdp_degree.
    trainer = GraphTrainer(
        model=model,
        train_fn=train_fn,
        parallel_config=ParallelConfig(
            enable_overlap=cfg["parallel"].get("enable_overlap", True),
            fsdp_degree=cfg["parallel"]["dp_size"],
            tp_size=cfg["parallel"]["tp_size"],
        ),
        sharding_plan=fsdp_plan,
        mesh_context=mesh_ctx,
        device=device,
    )

    # 6. Dummy data + explicit compile so we can inspect the graph.
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    sample_input = torch.randint(
        0, mcfg["vocab_size"], (tcfg["batch_size"], tcfg["seq_len"]), device=device
    )
    sample_label = torch.randint(
        0, mcfg["vocab_size"], (tcfg["batch_size"], tcfg["seq_len"]), device=device
    )
    trainer.compile(sample_input, sample_label)
    inspect_graph(trainer)

    def data_iter():
        for _ in range(tcfg["max_steps"]):
            yield (
                torch.randint(
                    0,
                    mcfg["vocab_size"],
                    (tcfg["batch_size"], tcfg["seq_len"]),
                    device=device,
                ),
                torch.randint(
                    0,
                    mcfg["vocab_size"],
                    (tcfg["batch_size"], tcfg["seq_len"]),
                    device=device,
                ),
            )

    print("\nStarting training...")
    trainer.train(
        data_iter(),
        max_steps=tcfg["max_steps"],
        log_interval=cfg["logging"]["log_interval"],
    )
    if rank == 0:
        print("=" * 70)
        print("Training completed!")
        print("=" * 70)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
